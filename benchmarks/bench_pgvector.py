"""
benchmarks/bench_pgvector.py

Measures the optional pgvector backend against the default exact one --
latency, RECALL, and the index's own build cost -- so the claims in
hopai/pgvector.py's module docstring stay measurements rather than
assertions. Self-contained apart from the extension itself: generates
its own vectors, loads ONE dataset, and stores it TWICE in the same
rows (a real[] `vec_exact` column for the exact backend, a pgvector
`vector(d)` + HNSW `vec_approx` column for the other), so every number
below compares two backends over identical data rather than two runs
over two datasets.

RECALL IS THE POINT OF THIS FILE. The exact backend answers exactly;
the HNSW index answers approximately, and a speedup quoted without the
recall it bought is the misleading number CLAUDE.md's "refuse, don't
approximate" ethos exists to prevent. So the exact backend's top-k is
taken as ground truth and recall@k is reported beside every latency.
Also checked, as a pass/fail rather than a timing: a filter matching
FEWER than k rows must return every matching row under both backends
-- what pgvector 0.8's `hnsw.iterative_scan = strict_order` buys and
the reason hopai sets it (see hopai/pgvector.py).

Only single-`Near` searches are measured, because they are the only
ones this backend serves: multivector and `boost=` are refused under
it by name.

Usage:
    python bench_pgvector.py --dsn postgresql+psycopg2://user:pass@host/db \
        [--rows 20000,100000] [--dims 384] [--k 10] [--queries 20] \
        [--repeats 5] [--clusters 0] [--ef-search 0] \
        [--out bench_pgvector_results.json]

Needs pgvector >= 0.8 installed in the target database and a role that
may CREATE SCHEMA. `benchmarks/README.md` records what a run produced,
on what machine; absolute numbers do not transfer across boxes, the
exact-vs-pgvector RATIO at a given size roughly does.
"""

from __future__ import annotations

import argparse
import json
import random
import time

from sqlalchemy import MetaData, create_engine, event, text
from sqlalchemy.dialects import postgresql

from hopai import Graph, Near, Vector
from hopai import pgvector as pg
from hopai.models import Edge, GraphRegistry, Node

SCHEMA = "hopai_pgvector_bench"

#: One dataset, two storage columns over the same rows. Named fields
#: rather than one field per Graph handle: the vec_* column IS the
#: field name, and the two backends cannot share a column (one is
#: real[], the other vector(d)), so the only way to hold the data
#: constant is to store it twice and search each side with the handle
#: that understands it.
EXACT_FIELD = "exact"
PGVECTOR_FIELD = "approx"

#: The selective filter, ~25% of rows -- the same shape and share
#: bench_vectors.py filters on, so the two files' filtered numbers can
#: be read against each other.
FILTER = {"type": "person"}

#: The filter that matches FEWER rows than k. Its whole job is the
#: correctness property, not a timing: pre-0.8 pgvector returns fewer
#: rows than match here and reports success.
RARE_FILTER = {"type": "rare"}
RARE_ROWS = 2


def parse_sizes(raw: str) -> list:
    return [int(part) for part in str(raw).split(",") if part.strip()]


def make_engine(dsn: str, ef_search: int):
    """The engine both handles share.

    The default pool, NOT bench_vectors.py's NullPool, and the
    difference is load-bearing here rather than stylistic: an indexed
    search can finish in a couple of milliseconds, the same order as
    opening a fresh connection, so NullPool would price connection
    setup into the number this file exists to report. The exact
    backend, at hundreds of milliseconds, never noticed.

    `--ef-search` is set on the CONNECTION, by a listener, because
    hopai has no API for it: the library sets `hnsw.iterative_scan`
    (which is a correctness setting, see hopai/pgvector.py) and leaves
    `hnsw.ef_search` at the server's default of 40. It is measured here
    anyway because it is the whole recall/latency dial -- a recall
    number quoted without saying which side of that dial it sits on is
    not a measurement of anything -- and because this listener is
    exactly what a caller wanting more recall has to write today.
    """
    engine = create_engine(dsn)
    if ef_search:
        @event.listens_for(engine, "connect")
        def _set_ef_search(dbapi_connection, _record):
            # Accepted before the extension's module is loaded into the
            # backend (Postgres keeps a placeholder for a dotted GUC it
            # does not recognize yet), so this does not need the
            # connection to have touched a vector first.
            cursor = dbapi_connection.cursor()
            cursor.execute(f"SET hnsw.ef_search = {int(ef_search)}")
            cursor.close()
    return engine


def make_vector(rng: random.Random, dims: int, centroids: list) -> list:
    """One stored vector. With `--clusters 0` (the default) these are
    uniform random, matching bench_vectors.py's dataset -- and that is
    the WORST case for an approximate index: uniform vectors in high
    dimension are near-orthogonal, so a query's true neighbors are
    barely closer than its non-neighbors and there is no structure for
    HNSW's graph to exploit. Real embeddings are clustered, which is
    what `--clusters N` generates, and why the recall this file
    measures at `--clusters 0` should be read as a floor rather than as
    what a corpus of real embeddings would give."""
    if not centroids:
        return [rng.uniform(-1.0, 1.0) for _ in range(dims)]
    center = centroids[rng.randrange(len(centroids))]
    return [value + rng.gauss(0.0, 0.25) for value in center]


def make_centroids(rng: random.Random, count: int, dims: int) -> list:
    return [[rng.uniform(-1.0, 1.0) for _ in range(dims)] for _ in range(count)]


def timed(fn, repeats: int = 5) -> float:
    fn()  # warm the cache so the numbers compare like bench_vectors' runs
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - started) / repeats


def graphs(engine, tables: dict) -> tuple:
    """One handle per backend over the SAME tables. Both are ordinary
    Graphs; the only difference is `vector_backend=`, which is the
    thing under measurement."""
    def build(**kwargs):
        return Graph(engine, node_table=tables["nodes"], edge_table=tables["edges"],
                     graph_table=tables["graphs"], **kwargs)
    return build(), build(vector_backend="pgvector")


def load(exact: Graph, approx: Graph, rows: int, dims: int, clusters: int,
         batch: int = 2000) -> dict:
    """Insert the rows once, then write the same vectors into both
    columns, timing each write separately.

    The two set_vectors() timings are not a curiosity: under the
    pgvector backend every write also maintains the HNSW index, which
    is a real cost of the backend and belongs in the same accounting as
    the search it makes fast."""
    rng = random.Random(42)
    centroids = make_centroids(rng, clusters, dims) if clusters else []
    insert_seconds = exact_seconds = approx_seconds = 0.0
    for low in range(0, rows, batch):
        ids = range(low, min(low + batch, rows))
        vectors = {i: make_vector(rng, dims, centroids) for i in ids}
        started = time.perf_counter()
        exact.add_nodes([{"id": i, "type": row_type(i, rows)} for i in ids])
        insert_seconds += time.perf_counter() - started

        started = time.perf_counter()
        exact.set_vectors(nodes=[{"id": i, EXACT_FIELD: v} for i, v in vectors.items()])
        exact_seconds += time.perf_counter() - started

        started = time.perf_counter()
        approx.set_vectors(nodes=[{"id": i, PGVECTOR_FIELD: v} for i, v in vectors.items()])
        approx_seconds += time.perf_counter() - started
    return {"insert_seconds": round(insert_seconds, 2),
            "set_vectors_exact_seconds": round(exact_seconds, 2),
            "set_vectors_pgvector_seconds": round(approx_seconds, 2)}


def row_type(i: int, rows: int) -> str:
    """~25% "person", exactly RARE_ROWS "rare", the rest "doc"."""
    if i >= rows - RARE_ROWS:
        return "rare"
    return "person" if i % 4 == 0 else "doc"


def rebuild_index(engine, table, column: str) -> dict:
    """Drop and rebuild the HNSW index over the loaded data, timing it.

    migrate_vectors() creates the index on an EMPTY column, which costs
    nothing and measures nothing. The honest number is what the index
    costs over the data it will serve -- so this drops the library's
    own index and re-runs the library's own DDL (hopai.pgvector's
    index_ddl(), not a hand-written CREATE INDEX, so a change to the
    index hopai builds shows up here).

    The DROP is spelled out here rather than taken from
    pg.drop_index_ddl(), which emits the index name UNQUALIFIED and so
    resolves it through search_path -- correct for the ordinary caller
    whose tables are on the path, a silent no-op for this benchmark's
    throwaway schema, which is not. It cost a whole first run: the drop
    hit nothing, CREATE INDEX IF NOT EXISTS found the index already
    there, and 'index_build_seconds' came back 0.0 at every size.
    """
    index = pg.index_name(table.name, column)
    qualified = f'"{table.schema}"."{table.name}"'
    with engine.begin() as conn:
        conn.execute(text(f'DROP INDEX IF EXISTS "{table.schema}"."{index}"'))
        # Belt and braces: a build timed against an index that is still
        # there measures nothing at all, and reports success.
        assert conn.execute(text(
            "SELECT count(*) FROM pg_indexes WHERE schemaname = :schema AND indexname = :name"
        ), {"schema": table.schema, "name": index}).scalar() == 0
    started = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(text(pg.index_ddl(qualified, table.name, column)))
    seconds = time.perf_counter() - started
    with engine.begin() as conn:
        conn.execute(text(f"ANALYZE {qualified}"))
        size = conn.execute(text("SELECT pg_relation_size(:name)"),
                            {"name": f"{table.schema}.{index}"}).scalar()
    return {"index_build_seconds": round(seconds, 2),
            "index_mb": round((size or 0) / 1024 / 1024, 1)}


def plan_of(engine, graph: Graph, near: Near, k: int, where=None) -> str:
    """The plan for a search, as the planner actually chose it.

    CLAUDE.md: "Inspect the emitted SQL for anything touching the query
    path; 'probably fine' is not a measurement." A pgvector search that
    silently fell back to a sequential scan would still return the right
    rows at the exact backend's cost, and no latency number reads as
    obviously wrong on a small table -- so the plan is recorded rather
    than inferred from the timings.

    Compiled with bound parameters and handed to the driver, not with
    literal_binds: the guard against a directionless stored vector binds
    a float NaN, which renders as a bare `nan` identifier when inlined.
    """
    compiled = graph.build_vector_search_query(near, k=k, where=where).compile(
        dialect=postgresql.dialect())
    with engine.connect() as conn:
        pg.apply_scan_settings(conn)
        rows = conn.exec_driver_sql("EXPLAIN " + str(compiled), compiled.params).fetchall()
    # The scan node only: the rest of the plan is the same subquery
    # wrapper under either answer, and the full text carries an inlined
    # query vector that would swamp the JSON.
    scans = [row[0].strip().lstrip("-> ") for row in rows if "Scan" in row[0]]
    return scans[-1] if scans else "\n".join(row[0] for row in rows)


def recall(exact: Graph, approx: Graph, queries: list, k: int, where=None) -> dict:
    """recall@k against the exact backend's answer as ground truth.

    Both backends rank the SAME vectors (float4 either way), so a
    disagreement is the index's approximation and nothing else. Reported
    as the mean, the worst single query, and the share of queries the
    index got perfectly right -- a mean of 0.97 hides whether one query
    in twenty lost three neighbors or every query lost one, and those
    are different products.

    `similarity_ratio` is reported beside it and is not a softer
    restatement of the same thing: recall counts IDS, and two neighbors
    at cosine 0.404 and 0.403 are a recall miss and a difference of
    nothing. The ratio is the mean similarity of what pgvector returned
    over the mean similarity of the true top-k, so a low recall with a
    ratio near 1.0 says "different rows, equally close" while a ratio
    that also drops says the index genuinely returned worse neighbors.
    Both numbers are needed to read a synthetic dataset, where near-ties
    are the norm.
    """
    scores, ratios, top1 = [], [], []
    for query in queries:
        truth = exact.vector_search(Near(EXACT_FIELD, query), k=k, where=where)
        got = approx.vector_search(Near(PGVECTOR_FIELD, query), k=k, where=where)
        if not truth:
            continue
        truth_ids = [hit["id"] for hit in truth]
        got_ids = [hit["id"] for hit in got]
        scores.append(len(set(truth_ids) & set(got_ids)) / len(truth_ids))
        top1.append(1.0 if truth_ids[0] in got_ids else 0.0)
        best = sum(hit["similarity"] for hit in truth) / len(truth)
        if got and best:
            ratios.append((sum(hit["similarity"] for hit in got) / len(got)) / best)
    if not scores:
        return {"mean": None, "min": None, "perfect_share": None, "queries": 0}
    return {"mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "perfect_share": round(sum(1 for s in scores if s == 1.0) / len(scores), 3),
            "top1_share": round(sum(top1) / len(top1), 3),
            "similarity_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "queries": len(scores)}


def fewer_than_k(exact: Graph, approx: Graph, query: list, k: int) -> dict:
    """The correctness property, as pass/fail: a filter matching fewer
    than k rows returns ALL of them under both backends.

    This is what pgvector >= 0.8's iterative scan buys and what hopai
    refuses an older server over -- below it the HNSW scan applies
    `where=` to a fixed candidate window and returns fewer rows than
    match, reporting success. A timing here would say nothing; the
    count is the whole result."""
    got_exact = exact.vector_search(Near(EXACT_FIELD, query), k=k, where=RARE_FILTER)
    got_approx = approx.vector_search(Near(PGVECTOR_FIELD, query), k=k, where=RARE_FILTER)
    return {"expected": RARE_ROWS, "exact": len(got_exact), "pgvector": len(got_approx),
            "pass": len(got_exact) == RARE_ROWS and len(got_approx) == RARE_ROWS,
            "same_ids": sorted(h["id"] for h in got_exact) == sorted(
                h["id"] for h in got_approx)}


def measure(engine, rows: int, dims: int, args) -> dict:
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
    # Schema-qualified tables rather than a search_path override: the
    # `vector` type lives in whichever schema CREATE EXTENSION put it
    # in (usually public), so a search_path narrowed to the throwaway
    # schema would leave `vector(d)` unresolvable at migrate time.
    metadata = MetaData()
    tables = {name: table.to_metadata(metadata, schema=SCHEMA)
              for name, table in (("nodes", Node), ("edges", Edge), ("graphs", GraphRegistry))}
    exact, approx = graphs(engine, tables)
    exact.create_schema()
    exact.define_vectors(nodes=[Vector(EXACT_FIELD, dims)], migrate=True)
    approx.define_vectors(nodes=[Vector(PGVECTOR_FIELD, dims)], migrate=True)

    result = {"rows": rows, "dims": dims, "k": args.k, "clusters": args.clusters,
              "ef_search": args.ef_search or "server default"}
    result.update(load(exact, approx, rows, dims, args.clusters))
    result.update(rebuild_index(engine, tables["nodes"], f"vec_{PGVECTOR_FIELD}"))

    rng = random.Random(7)
    query = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
    queries = [[rng.uniform(-1.0, 1.0) for _ in range(dims)] for _ in range(args.queries)]

    exact_unfiltered = timed(
        lambda: exact.vector_search(Near(EXACT_FIELD, query), k=args.k), args.repeats)
    approx_unfiltered = timed(
        lambda: approx.vector_search(Near(PGVECTOR_FIELD, query), k=args.k), args.repeats)
    exact_filtered = timed(
        lambda: exact.vector_search(Near(EXACT_FIELD, query), k=args.k, where=FILTER),
        args.repeats)
    approx_filtered = timed(
        lambda: approx.vector_search(Near(PGVECTOR_FIELD, query), k=args.k, where=FILTER),
        args.repeats)

    result["exact_unfiltered_ms"] = round(exact_unfiltered * 1000, 1)
    result["pgvector_unfiltered_ms"] = round(approx_unfiltered * 1000, 1)
    result["unfiltered_speedup"] = round(exact_unfiltered / approx_unfiltered, 2)
    result["exact_filtered_25pct_ms"] = round(exact_filtered * 1000, 1)
    result["pgvector_filtered_25pct_ms"] = round(approx_filtered * 1000, 1)
    result["filtered_speedup"] = round(exact_filtered / approx_filtered, 2)
    # bench_vectors.py's transferable number, recomputed here so the
    # exact side of this table can be checked against that file's --
    # the exact scan's cost is linear in rows x dims, which is the
    # premise the pgvector backend exists to escape. The pgvector side
    # has NO such number by design: an index search is sublinear in
    # rows, so a per-element figure for it would fall as the table grows
    # and mean nothing.
    result["exact_us_per_element"] = round(exact_unfiltered / (rows * dims) * 1e6, 3)

    result["recall_unfiltered"] = recall(exact, approx, queries, args.k)
    result["recall_filtered_25pct"] = recall(exact, approx, queries, args.k, where=FILTER)
    result["fewer_than_k_filter"] = fewer_than_k(exact, approx, query, args.k)
    # Not a timing, and the reason it is here: a pgvector search that
    # fell back to a sequential scan returns identical rows at the exact
    # backend's cost.
    result["pgvector_scan_unfiltered"] = plan_of(engine, approx, Near(PGVECTOR_FIELD, query),
                                                 args.k)
    result["pgvector_scan_filtered"] = plan_of(engine, approx, Near(PGVECTOR_FIELD, query),
                                               args.k, where=FILTER)
    result["pgvector_uses_index"] = all(
        pg.index_name(tables["nodes"].name, f"vec_{PGVECTOR_FIELD}") in result[key]
        for key in ("pgvector_scan_unfiltered", "pgvector_scan_filtered"))

    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA {SCHEMA} CASCADE"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--rows", default="20000,100000",
                        help="comma-separated row counts to sweep")
    parser.add_argument("--dims", default="384", help="comma-separated dimensionalities")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=20,
                        help="query vectors used for the recall measurement")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=0,
                        help="0 (default) generates uniform random vectors, matching "
                             "bench_vectors.py; N > 0 draws them around N centroids, which is "
                             "closer to how real embeddings sit and raises recall")
    parser.add_argument("--ef-search", type=int, default=0,
                        help="0 (default) leaves hnsw.ef_search at the server's default of "
                             "40, which is what hopai itself runs with; N sets it per "
                             "connection to trade latency for recall")
    parser.add_argument("--out", default="bench_pgvector_results.json")
    args = parser.parse_args()

    engine = make_engine(args.dsn, args.ef_search)
    results = {"k": args.k, "queries": args.queries, "repeats": args.repeats,
               "clusters": args.clusters, "ef_search": args.ef_search or "server default",
               "configs": []}
    for rows in parse_sizes(args.rows):
        for dims in parse_sizes(args.dims):
            results["configs"].append(measure(engine, rows, dims, args))
            print(json.dumps(results["configs"][-1], indent=2), flush=True)

    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
