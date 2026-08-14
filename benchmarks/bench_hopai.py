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
        cur2.execute("CREATE TEMP TABLE stage_edges(start_id bigint, end_id bigint, tag text)")
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


def build_suite():
    from hopai import AND, GT, NOT, Hop, Start

    return [
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


def time_raw_sql(graph, start_hop, rest) -> float:
    """Execute the SQL hopai just built, straight through the driver.

    This is the floor the README has always talked about and never
    measured: the SAME query, with none of hopai's Python around it --
    no SQLAlchemy result mapping, no property hydration round trips, no
    Subgraph. The gap between this and traverse() is what the library
    layer costs, and it is the honest way to quote that, because both
    numbers come from one statement rather than from two hand-written
    queries somebody has to keep in step.
    """
    statement = graph.build_query(start_hop, list(rest))
    compiled = statement.compile(dialect=graph.engine.dialect,
                                 compile_kwargs={"literal_binds": True})
    raw = graph.engine.raw_connection()
    try:
        cursor = raw.cursor()
        t0 = time.perf_counter()
        cursor.execute(str(compiled))
        cursor.fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        raw.close()
    return elapsed


def run_suite(graph, hub_id: int, baseline: bool = True):
    suite = build_suite()
    results = []
    for name, hops in suite:
        start_hop, *rest = hops
        times = []
        r = None
        for _ in range(2):  # first pass cold, second warm
            t0 = time.perf_counter()
            r = graph.traverse(start_hop, *rest)
            times.append((time.perf_counter() - t0) * 1000)
        row = {
            "query": name,
            "cold_ms": round(times[0], 1),
            "warm_ms": round(times[1], 1),
            "nodes": len(r.nodes),
            "edges": len(r.edges),
        }
        if baseline:
            time_raw_sql(graph, start_hop, rest)          # warm the same way
            row["raw_sql_ms"] = round(time_raw_sql(graph, start_hop, rest), 1)
        results.append(row)
        extra = f" raw={row['raw_sql_ms']:9.1f}ms" if baseline else ""
        print(f"{name:28s} cold={times[0]:9.1f}ms warm={times[1]:9.1f}ms{extra} "
              f"nodes={len(r.nodes):8d} edges={len(r.edges):8d}")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))
    from hopai import Graph

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--dsn", type=str, required=True)
    ap.add_argument("--schema", type=str, default="hopai_bench")
    ap.add_argument("--hub-id", type=int, default=None,
                     help="defaults to nodes/2, matching generate_graph.py's default")
    ap.add_argument("--skip-load", action="store_true", help="reuse already-loaded data")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the raw-SQL floor (hopai's own SQL run through the driver)")
    args = ap.parse_args()

    engine = create_engine(args.dsn)
    if not args.skip_load:
        print("Loading data...")
        load_data(engine, Path(args.data_dir), args.schema)

    engine_scoped = create_engine(args.dsn, connect_args={"options": f"-c search_path={args.schema}"})
    graph = Graph(engine_scoped)

    print(f"\n{'Query':28s} {'Cold':>13s} {'Warm':>13s} {'Nodes':>10s} {'Edges':>10s}")
    print("-" * 90)
    results = run_suite(graph, args.hub_id or 0, baseline=not args.no_baseline)

    from report import machine_profile, render

    # The profile is read from the server that just answered the
    # queries, not from a config file, so the report describes the run
    # that actually happened.
    with engine_scoped.connect() as conn:
        profile = machine_profile(conn)
        counts = {
            "nodes": conn.execute(text(f"SELECT count(*) FROM {args.schema}.nodes")).scalar(),
            "edges": conn.execute(text(f"SELECT count(*) FROM {args.schema}.edges")).scalar(),
            "schema": args.schema,
        }

    results_path = Path(__file__).parent / "bench_results.json"
    report_path = Path(__file__).parent / "RESULTS.md"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    report_path.write_text(render(results, profile, counts))

    print(f"\nSaved {results_path.name} and {report_path.name} (both overwritten)")
