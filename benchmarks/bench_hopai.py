"""
benchmarks/bench_hopai.py

Loads a generated graph (see generate_graph.py) into PostgreSQL and runs
a suite of queries through hopai, covering direction, multi-hop
bounds, compound chains, AND/OR/NOT, range comparisons, and OPTIONAL --
timing each one cold and warm.

Usage:
    python generate_graph.py --nodes 1000000 --out-dir ./data
    python bench_hopai.py --data-dir ./data --dsn postgresql+psycopg2://user:pass@host/db
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sqlalchemy import create_engine, text


def load_data(engine, data_dir: Path, schema: str):
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {schema}"))
        conn.execute(text(f"""
            CREATE TABLE {schema}.nodes (id BIGINT PRIMARY KEY, properties JSONB NOT NULL DEFAULT '{{}}')
        """))
        conn.execute(text(f"""
            CREATE TABLE {schema}.edges (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                start_id BIGINT NOT NULL, end_id BIGINT NOT NULL,
                properties JSONB NOT NULL DEFAULT '{{}}'
            )
        """))

    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        with open(data_dir / "nodes.csv") as f:
            cur.copy_expert(f"COPY {schema}.nodes(id, properties) FROM STDIN WITH (FORMAT csv)", f)
        cur2 = raw.cursor()
        cur2.execute(f"CREATE TEMP TABLE stage_edges(start_id bigint, end_id bigint, tag text)")
        with open(data_dir / "edges.csv") as f:
            cur2.copy_expert("COPY stage_edges FROM STDIN WITH (FORMAT csv)", f)
        cur2.execute(f"""
            INSERT INTO {schema}.edges (start_id, end_id, properties)
            SELECT start_id, end_id, jsonb_build_object('tag', tag) FROM stage_edges
        """)
        raw.commit()
    finally:
        raw.close()

    with engine.begin() as conn:
        conn.execute(text(f"CREATE INDEX ON {schema}.edges (start_id)"))
        conn.execute(text(f"CREATE INDEX ON {schema}.edges (end_id)"))
        conn.execute(text(f"CREATE INDEX ON {schema}.nodes USING GIN (properties)"))
        conn.execute(text(f"CREATE INDEX ON {schema}.edges USING GIN (properties)"))
        conn.execute(text(f"ANALYZE {schema}.nodes"))
        conn.execute(text(f"ANALYZE {schema}.edges"))


def run_suite(graph, hub_id: int):
    from hopai import AND, BETWEEN, GT, NOT, Hop, Start

    suite = [
        ("forward_1hop", [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=1)]),
        ("backward_1hop", [Start(where={"type": "hub"}), Hop(hops=1, direction="backward")]),
        ("forward_bounded_4hop", [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=(1, 4))]),
        ("backward_bounded_3hop", [Start(where={"type": "hub"}), Hop(hops=(1, 3), direction="backward")]),
        ("compound_2segment", [
            Start(where={"type": "leaf"}),
            Hop(where={"flag": 1}, hops=(1, 4)),
            Hop(where={"type": "hub"}, hops=(1, 3)),
        ]),
        ("edge_or_tag", [
            Start(where={"type": "leaf"}),
            Hop(where={"flag": 1}, via={"tag": ["p1", "p2"]}, hops=(1, 4)),
        ]),
        ("not_filter", [
            Start(where={"type": "hub"}),
            Hop(where=NOT({"type": "leaf"}), hops=(1, 3), direction="backward"),
        ]),
        ("range_gt", [Start(where=AND({"type": "leaf"}, GT("priority", 5)))]),
        ("optional_last_hop", [
            Start(where={"type": "hub"}),
            Hop(where=NOT({"type": "leaf"}), hops=1, direction="backward"),
            Hop(where={"type": "leaf"}, hops=1, optional=True),
        ]),
    ]

    results = []
    for name, hops in suite:
        start_hop, *rest = hops
        times = []
        r = None
        for i in range(2):
            t0 = time.perf_counter()
            r = graph.traverse(start_hop, *rest)
            times.append((time.perf_counter() - t0) * 1000)
        results.append({
            "query": name,
            "cold_ms": round(times[0], 1),
            "warm_ms": round(times[1], 1),
            "nodes": len(r.nodes),
            "edges": len(r.edges),
        })
        print(f"{name:28s} cold={times[0]:9.1f}ms warm={times[1]:9.1f}ms nodes={len(r.nodes):8d} edges={len(r.edges):8d}")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from hopai import Graph

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--dsn", type=str, required=True)
    ap.add_argument("--schema", type=str, default="hopai_bench")
    ap.add_argument("--hub-id", type=int, default=None,
                     help="defaults to nodes/2, matching generate_graph.py's default")
    ap.add_argument("--skip-load", action="store_true", help="reuse already-loaded data")
    args = ap.parse_args()

    engine = create_engine(args.dsn)
    if not args.skip_load:
        print("Loading data...")
        load_data(engine, Path(args.data_dir), args.schema)

    engine_scoped = create_engine(args.dsn, connect_args={"options": f"-c search_path={args.schema}"})
    graph = Graph(engine_scoped)

    print(f"\n{'Query':28s} {'Cold':>13s} {'Warm':>13s} {'Nodes':>10s} {'Edges':>10s}")
    print("-" * 90)
    results = run_suite(graph, args.hub_id or 0)

    with open("bench_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to bench_results.json")
