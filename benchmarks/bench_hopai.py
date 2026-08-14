"""
benchmarks/bench_hopai.py

Loads a generated graph (see generate_graph.py) into PostgreSQL and runs
a suite of queries through hopai, covering direction, multi-hop
bounds, compound chains, AND/OR/NOT, range comparisons, OPTIONAL, and
aggregation -- timing each one cold and warm.

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
        # graph_id mirrors the real schema (see models.py): Graph()
        # builds against the default model tables, which carry the
        # column, so the physical tables must too or every query fails at
        # runtime. The composite foreign keys are deliberately omitted --
        # they cost a great deal on a COPY of millions of rows and
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
        # graph_id LEADS, matching create_schema(): every hop filters on
        # it, and a trailing position would stop paying for itself the
        # moment a second graph exists. Benchmarking differently-indexed
        # tables would measure the indexes, not the library.
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
        ("Q1", "Forward 1-hop, equality", "direction", "simple",
         [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=1)]),
        ("Q2", "Backward 1-hop, equality", "direction", "simple",
         [Start(where={"type": "hub"}), Hop(hops=1, direction="backward")]),
        ("Q3", "Forward bounded <=4 hops", "multi-hop", "complex",
         [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=(1, 4))]),
        ("Q4", "Backward bounded <=3 hops", "multi-hop", "complex",
         [Start(where={"type": "hub"}), Hop(hops=(1, 3), direction="backward")]),
        ("Q5", "2-segment compound chain", "compound", "complex",
         [Start(where={"type": "leaf"}),
          Hop(where={"flag": 1}, hops=(1, 4)),
          Hop(where={"type": "hub"}, hops=(1, 3))]),
        ("Q6", "Mixed backward -> forward", "direction", "complex",
         [Start(where={"type": "hub"}),
          Hop(hops=1, direction="backward"),
          Hop(hops=1, direction="forward")]),
        ("Q7", "Edge OR (tag p1|p2)", "OR: edge", "complex",
         [Start(where={"type": "leaf"}),
          Hop(where={"flag": 1}, via={"tag": ["p1", "p2"]}, hops=(1, 4))]),
        ("Q8", "Node OR, explicit", "OR: node", "complex",
         [Start(where=OR({"type": "leaf"}, {"type": "hub"})), Hop(hops=1)]),
        ("Q9", "NOT (missing-key semantics)", "NOT", "complex",
         [Start(where={"type": "hub"}),
          Hop(where=NOT({"type": "leaf"}), hops=(1, 3), direction="backward")]),
        ("Q10", "GT range comparison", "range", "simple",
         [Start(where=AND({"type": "leaf"}, GT("priority", 5)))]),
        ("Q11", "BETWEEN range comparison", "range", "simple",
         [Start(where=AND({"type": "leaf"}, BETWEEN("priority", 5, 15)))]),
        ("Q12", "AND(OR(...), {...})", "composition", "simple",
         [Start(where=AND(OR({"type": "leaf"}, {"type": "hub"}), {"type": "leaf"}))]),
        ("Q13", "OPTIONAL last hop", "OPTIONAL", "complex",
         [Start(where={"type": "hub"}),
          Hop(where=NOT({"type": "leaf"}), hops=1, direction="backward"),
          Hop(where={"type": "leaf"}, hops=1, optional=True)]),
        ("Q14", "Deep bounded backward <=12", "deep multi-hop", "very complex",
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
    for qid, label, feature, tier, hops in build_suite():
        start_hop, *rest = hops
        row = {"id": qid, "query": label, "feature": feature,
               "tier": tier}

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
            "empty": len(r.nodes) == 0,
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

    results.extend(run_aggregate_suite(graph, repeat=repeat, baseline=baseline))
    return results


def build_aggregate_suite():
    """Aggregations at three tiers of difficulty.

    Aggregation is not one operation whose cost you can quote once. What
    it costs depends on how many nodes the traversal had to match, how
    many aggregates run over them, and whether DISTINCT forces a sort.
    A suite that only counts rows after a one-hop walk answers none of
    that, so these are graded deliberately:

      simple        no traversal -- the aggregate is the whole query
      complex       a real walk underneath, one or a few aggregates
      very complex  deep or compound walks, many aggregates at once,
                    DISTINCT over large match sets

    Q15 and Q20 reuse Q3's and Q14's chains exactly, so each can be read
    against its traversal twin: the difference is what skipping edge
    reconstruction and node hydration is worth on identical work.
    """
    from hopai import AND, GT, NOT, OR, Avg, Count, Hop, Max, Min, Start, Sum

    return [
        # -- simple: no traversal, the aggregate IS the query -----------
        ("Q15", "count, equality filter", "agg: count", "simple",
         [Start(where={"type": "leaf"})], {"n": Count()}),
        ("Q16", "avg over a range filter", "agg: avg", "simple",
         [Start(where=AND({"type": "leaf"}, GT("priority", 5)))],
         {"mean": Avg("priority")}),
        ("Q17", "min/max, equality filter", "agg: min/max", "simple",
         [Start(where={"type": "leaf"})],
         {"lo": Min("priority"), "hi": Max("priority")}),

        # -- complex: a real traversal underneath -----------------------
        ("Q18", "count after a 1-hop walk", "agg: count", "complex",
         [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=1)],
         {"n": Count()}),
        ("Q19#Q3", "count over Q3's bounded chain", "agg: count", "complex",
         [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=(1, 4))],
         {"n": Count()}),
        ("Q20", "count of nodes HAVING a property", "agg: count(prop)", "complex",
         [Start(where={"type": "hub"}), Hop(hops=(1, 3), direction="backward")],
         {"with_priority": Count("priority")}),
        ("Q21", "five statistics, backward <=4", "agg: multi-stat", "complex",
         [Start(where={"flag": 1}), Hop(hops=(1, 4), direction="backward")],
         {"n": Count(), "total": Sum("priority"), "mean": Avg("priority"),
          "lo": Min("priority"), "hi": Max("priority")}),
        ("Q22", "count over a NOT filter", "agg: count", "complex",
         [Start(where={"type": "hub"}),
          Hop(where=NOT({"type": "leaf"}), hops=(1, 3), direction="backward")],
         {"n": Count()}),

        # -- very complex: deep, compound, DISTINCT, many at once -------
        ("Q23", "distinct values over a bounded walk", "agg: distinct", "very complex",
         [Start(where={"type": "hub"}), Hop(hops=(1, 4), direction="backward")],
         {"distinct_priorities": Count("priority", distinct=True)}),
        ("Q24", "distinct sum over a bounded walk", "agg: distinct", "very complex",
         [Start(where={"type": "hub"}), Hop(hops=(1, 4), direction="backward")],
         {"total": Sum("priority", distinct=True), "mean": Avg("priority", distinct=True)}),
        ("Q25#Q5", "count over a 2-segment compound chain", "agg: compound", "very complex",
         [Start(where={"type": "leaf"}),
          Hop(where={"flag": 1}, hops=(1, 4)),
          Hop(where={"type": "hub"}, hops=(1, 3))],
         {"n": Count()}),
        ("Q26#Q7", "count over an edge-filtered walk", "agg: edge filter", "very complex",
         [Start(where={"type": "leaf"}),
          Hop(where={"flag": 1}, via={"tag": ["p1", "p2"]}, hops=(1, 4))],
         {"n": Count()}),
        ("Q27#Q14", "count over Q14's deep 12-hop walk", "agg: deep", "very complex",
         [Start(where={"type": "hub"}), Hop(hops=(1, 12), direction="backward")],
         {"n": Count()}),
        ("Q28", "seven aggregates at once, deep walk", "agg: wide", "very complex",
         [Start(where={"type": "hub"}), Hop(hops=(1, 12), direction="backward")],
         {"n": Count(), "have": Count("priority"),
          "distinct": Count("priority", distinct=True),
          "total": Sum("priority"), "mean": Avg("priority"),
          "lo": Min("priority"), "hi": Max("priority")}),
        ("Q29", "count over AND(OR(...)) composition", "agg: composition", "very complex",
         [Start(where=AND(OR({"type": "leaf"}, {"type": "hub"}), {"type": "leaf"})),
          Hop(hops=(1, 2))],
         {"n": Count()}),
    ]


def time_raw_aggregate_sql(graph, start_hop, rest, aggregates) -> float:
    """The same floor as time_raw_sql, for an aggregate query."""
    statement = graph.build_aggregate_query(start_hop, list(rest), aggregates)
    compiled = statement.compile(dialect=graph.engine.dialect,
                                 compile_kwargs={"literal_binds": True})
    raw = graph.engine.raw_connection()
    try:
        cursor = raw.cursor()
        t0 = time.perf_counter()
        cursor.execute(str(compiled))
        cursor.fetchall()
        return (time.perf_counter() - t0) * 1000
    finally:
        raw.close()


def run_aggregate_suite(graph, repeat: int = 5, baseline: bool = True):
    results = []
    for entry, label, feature, tier, hops, aggregates in build_aggregate_suite():
        qid, _, twin = entry.partition("#")
        start_hop, *rest = hops
        t0 = time.perf_counter()
        values = graph.aggregate(start_hop, *rest, aggregates=aggregates)
        cold = (time.perf_counter() - t0) * 1000

        warm = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            values = graph.aggregate(start_hop, *rest, aggregates=aggregates)
            warm.append((time.perf_counter() - t0) * 1000)

        row = {
            "id": qid, "query": label, "feature": feature, "tier": tier,
            "twin_of": twin or None,
            "cold_ms": round(cold, 1),
            "warm_ms": round(statistics.median(warm), 1),
            "warm_min_ms": round(min(warm), 1),
            "warm_max_ms": round(max(warm), 1),
            "samples": repeat,
            # an aggregate returns numbers, not a subgraph. `nodes` holds
            # the count it computed where there is one, so the report's
            # row column stays meaningful instead of showing zero.
            "nodes": values.get("n"),
            "edges": None,
            # An aggregate without a Count() has no row count, which is
            # not the same as having none. Emptiness is decided from the
            # values themselves so a query that really did compute
            # something is never reported as having measured nothing.
            "empty": all(v in (None, 0) for v in values.values()),
            "aggregates": len(aggregates),
            "values": {k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in values.items()},
        }
        if baseline:
            time_raw_aggregate_sql(graph, start_hop, rest, aggregates)
            row["raw_sql_ms"] = round(
                statistics.median(time_raw_aggregate_sql(graph, start_hop, rest, aggregates)
                                  for _ in range(repeat)), 1)
        results.append(row)
        extra = f" raw={row['raw_sql_ms']:8.1f}ms" if baseline else ""
        print(f"{qid:4s} {label:34s} cold={cold:9.1f}ms warm={row['warm_ms']:9.1f}ms"
              f"{extra}  {values}")
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
