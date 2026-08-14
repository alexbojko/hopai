"""
benchmarks/bench_vectors.py

Measures the exact-cosine vector search at a few scales, so the cost
model quoted in hopai/vectors.py and the README stays a measurement
rather than a claim. Self-contained: generates its own random vectors,
loads them through set_vectors(), times unfiltered and filtered
searches plus a similarity-seeded traversal, and prints per-element
cost -- the number that transfers across dimensions.

Usage:
    python bench_vectors.py --dsn postgresql+psycopg2://user:pass@host/db \
        [--rows 20000] [--dims 384] [--out bench_vector_results.json]

Numbers recorded during development (Postgres 16, one core, in-repo
container): ~0.25us per vector element -- 20k x 384-dim unfiltered
~2.1s, the same search filtered to 25% of rows ~0.6s.
"""

from __future__ import annotations

import argparse
import json
import random
import time

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from hopai import Graph, Hop, Near, Start, Vector

SCHEMA = "hopai_vec_bench"


def load(graph: Graph, rows: int, dims: int, batch: int = 2000) -> float:
    random.seed(42)
    started = time.perf_counter()
    for low in range(0, rows, batch):
        ids = range(low, min(low + batch, rows))
        graph.add_nodes([{"id": i, "type": "doc" if i % 4 else "person"} for i in ids])
        graph.set_vectors(nodes=[
            {"id": i, "emb": [random.uniform(-1.0, 1.0) for _ in range(dims)]} for i in ids
        ])
    return time.perf_counter() - started


def timed(fn, repeats: int = 5) -> float:
    fn()  # warm the cache so the numbers compare like bench_hopai's warm runs
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - started) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--dims", type=int, default=384)
    parser.add_argument("--out", default="bench_vector_results.json")
    args = parser.parse_args()

    engine = create_engine(args.dsn, poolclass=NullPool,
                           connect_args={"options": f"-c search_path={SCHEMA}"})
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))

    graph = Graph(engine)
    graph.create_schema()
    graph.define_vectors(nodes=[Vector("emb", args.dims)])
    graph.migrate_vectors()
    load_seconds = load(graph, args.rows, args.dims)

    random.seed(7)
    query = [random.uniform(-1.0, 1.0) for _ in range(args.dims)]
    results = {"rows": args.rows, "dims": args.dims, "load_seconds": round(load_seconds, 2)}

    unfiltered = timed(lambda: graph.vector_search(Near("emb", query), k=10))
    filtered = timed(lambda: graph.vector_search(Near("emb", query), k=10,
                                                 where={"type": "person"}))
    seeded = timed(lambda: graph.traverse(Start(near=Near("emb", query), k=25),
                                          Hop(optional=True)))
    results["search_unfiltered_ms"] = round(unfiltered * 1000, 1)
    results["search_filtered_25pct_ms"] = round(filtered * 1000, 1)
    results["traverse_seeded_ms"] = round(seeded * 1000, 1)
    # The transferable number: per-element cost of the exact scan.
    results["us_per_element"] = round(unfiltered / (args.rows * args.dims) * 1e6, 3)

    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA {SCHEMA} CASCADE"))

    print(json.dumps(results, indent=2))
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
