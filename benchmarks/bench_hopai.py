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
import statistics
import time
from pathlib import Path

from sqlalchemy import create_engine, text


def load_data(engine, data_dir: Path, schema: str):
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {schema}"))
        # Mirrors create_schema(), with one deliberate omission: the
        # composite foreign keys that keep an edge inside its own graph.
        # They cost a great deal on a COPY of millions of rows and
        # nothing at all on a SELECT, so leaving them out speeds up
        # loading without touching what is being measured.
        conn.execute(text(f"""
            CREATE TABLE {schema}.nodes (
                id BIGINT PRIMARY KEY,
                graph_id TEXT NOT NULL DEFAULT 'default',
                properties JSONB NOT NULL DEFAULT '{{}}'
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE {schema}.edges (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                graph_id TEXT NOT NULL DEFAULT 'default',
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
        # graph_id LEADS, exactly as create_schema() builds them -- a
        # benchmark against differently-indexed tables measures the
        # indexes, not the library.
        conn.execute(text(f"CREATE INDEX ON {schema}.edges (graph_id, start_id)"))
        conn.execute(text(f"CREATE INDEX ON {schema}.edges (graph_id, end_id)"))
        conn.execute(text(f"CREATE INDEX ON {schema}.nodes (graph_id)"))
        conn.execute(text(f"CREATE INDEX ON {schema}.nodes USING GIN (properties)"))
        conn.execute(text(f"CREATE INDEX ON {schema}.edges USING GIN (properties)"))
        conn.execute(text(f"ANALYZE {schema}.nodes"))
        conn.execute(text(f"ANALYZE {schema}.edges"))


def build_suite():
    """Fourteen queries, one per capability the library claims.

    The point of the breadth is that a two-query benchmark proves
    nothing about a traversal engine: cost is driven by which side of a
    pattern anchors the walk, how deep it goes, and whether the filter
    is on nodes or edges. Each entry carries the feature it exercises so
    the report can say what was covered rather than just how fast it was.
    """
    from hopai import AND, BETWEEN, GT, NOT, OR, Hop, Start

    return [
        ("Q1", "Forward 1-hop, equality", "direction",
         [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=1)]),
        ("Q2", "Backward 1-hop, equality", "direction",
         [Start(where={"type": "hub"}), Hop(hops=1, direction="backward")]),
        ("Q3", "Forward bounded <=4 hops", "multi-hop",
         [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=(1, 4))]),
        ("Q4", "Backward bounded <=3 hops", "multi-hop",
         [Start(where={"type": "hub"}), Hop(hops=(1, 3), direction="backward")]),
        ("Q5", "2-segment compound chain", "compound",
         [Start(where={"type": "leaf"}),
          Hop(where={"flag": 1}, hops=(1, 4)),
          Hop(where={"type": "hub"}, hops=(1, 3))]),
        ("Q6", "Mixed backward -> forward", "direction",
         [Start(where={"type": "hub"}),
          Hop(hops=1, direction="backward"),
          Hop(hops=1, direction="forward")]),
        ("Q7", "Edge OR (tag p1|p2)", "OR: edge",
         [Start(where={"type": "leaf"}),
          Hop(where={"flag": 1}, via={"tag": ["p1", "p2"]}, hops=(1, 4))]),
        ("Q8", "Node OR, explicit", "OR: node",
         [Start(where=OR({"type": "leaf"}, {"type": "hub"})), Hop(hops=1)]),
        ("Q9", "NOT (missing-key semantics)", "NOT",
         [Start(where={"type": "hub"}),
          Hop(where=NOT({"type": "leaf"}), hops=(1, 3), direction="backward")]),
        ("Q10", "GT range comparison", "range",
         [Start(where=AND({"type": "leaf"}, GT("priority", 5)))]),
        ("Q11", "BETWEEN range comparison", "range",
         [Start(where=AND({"type": "leaf"}, BETWEEN("priority", 5, 15)))]),
        ("Q12", "AND(OR(...), {...})", "composition",
         [Start(where=AND(OR({"type": "leaf"}, {"type": "hub"}), {"type": "leaf"}))]),
        ("Q13", "OPTIONAL last hop", "OPTIONAL",
         [Start(where={"type": "hub"}),
          Hop(where=NOT({"type": "leaf"}), hops=1, direction="backward"),
          Hop(where={"type": "leaf"}, hops=1, optional=True)]),
        ("Q14", "Deep bounded backward <=12", "deep multi-hop",
         [Start(where={"type": "hub"}), Hop(hops=(1, 12), direction="backward")]),
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


def run_suite(graph, hub_id: int, baseline: bool = True, repeat: int = 5,
              budget_s: float = 150.0):
    """Time every query, marking any that blow the time budget as DNF.

    A query that never returns is not a slow query, it is a different
    outcome -- reporting it as a large number would let it average in
    with the rest. It is recorded as `dnf` with the budget it exceeded.
    """
    results = []
    for qid, label, feature, hops in build_suite():
        start_hop, *rest = hops
        row = {"id": qid, "query": label, "feature": feature}

        try:
            t0 = time.perf_counter()
            r = _with_timeout(graph, budget_s, start_hop, rest)
            cold = (time.perf_counter() - t0) * 1000
        except TimeoutError:
            row.update({"dnf": True, "budget_s": budget_s, "cold_ms": budget_s * 1000,
                        "warm_ms": budget_s * 1000, "nodes": None, "edges": None})
            results.append(row)
            print(f"{qid:4s} {label:30s} DID NOT FINISH within {budget_s:.0f}s")
            continue

        # First call is cold. Everything after is warm, and the MEDIAN of
        # those is reported: a single warm sample on a shared machine
        # moves 10-20% between runs, wider than most regressions worth
        # catching, so one shot cannot tell a real change from noise.
        warm = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            r = graph.traverse(start_hop, *rest)
            warm.append((time.perf_counter() - t0) * 1000)

        row.update({
            "cold_ms": round(cold, 1),
            "warm_ms": round(statistics.median(warm), 1),
            "warm_min_ms": round(min(warm), 1),
            "warm_max_ms": round(max(warm), 1),
            "samples": repeat,
            "nodes": len(r.nodes),
            "edges": len(r.edges),
        })
        if baseline:
            time_raw_sql(graph, start_hop, rest)          # warm it the same way
            row["raw_sql_ms"] = round(
                statistics.median(time_raw_sql(graph, start_hop, rest)
                                  for _ in range(repeat)), 1)
        results.append(row)
        extra = f" raw={row['raw_sql_ms']:8.1f}ms" if baseline else ""
        print(f"{qid:4s} {label:30s} cold={cold:9.1f}ms warm={row['warm_ms']:9.1f}ms"
              f" [{row['warm_min_ms']:.1f}-{row['warm_max_ms']:.1f}]{extra}"
              f" rows={len(r.nodes):8d}")
    return results


def _with_timeout(graph, budget_s: float, start_hop, rest):
    """Run one traversal, letting the SERVER cancel it if it overruns.

    The budget is set as `statement_timeout` on the engine's connections
    (see __main__), not with SET LOCAL here -- traverse() checks out its
    own connection from the pool, so a session setting applied to some
    other connection would simply not apply to the query being timed. A
    client-side give-up would also leave the query running and hold a
    backend for the rest of the suite; the server cancelling it is what
    actually stops the work.
    """
    from sqlalchemy.exc import OperationalError

    try:
        return graph.traverse(start_hop, *rest)
    except OperationalError as exc:
        if "statement timeout" in str(exc).lower() or "canceling statement" in str(exc).lower():
            raise TimeoutError from exc
        raise


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
    ap.add_argument("--budget", type=float, default=150.0,
                    help="per-query time budget in seconds; overruns are reported DNF")
    ap.add_argument("--repeat", type=int, default=5,
                    help="warm runs per query; the median is reported (default 5)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the raw-SQL floor (hopai's own SQL run through the driver)")
    args = ap.parse_args()

    engine = create_engine(args.dsn)
    if not args.skip_load:
        print("Loading data...")
        load_data(engine, Path(args.data_dir), args.schema)

    # The time budget lives on the connection, so every query the suite
    # runs is under it -- including the ones that would otherwise hold a
    # backend for minutes.
    engine_scoped = create_engine(args.dsn, connect_args={
        "options": f"-c search_path={args.schema} "
                   f"-c statement_timeout={int(args.budget * 1000)}"})
    graph = Graph(engine_scoped)

    print(f"\n{'ID':4s} {'Query':30s} {'Cold':>14s} {'Warm':>14s}")
    print("-" * 110)
    results = run_suite(graph, args.hub_id or 0, baseline=not args.no_baseline, repeat=args.repeat, budget_s=args.budget)

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
