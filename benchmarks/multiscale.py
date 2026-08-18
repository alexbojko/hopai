"""
Multi-scale, multi-engine benchmark driver.

Generates a graph at a target edge count, loads it into hopai/Postgres,
Apache AGE and Neo4j (raw_cte.py queries the same Postgres tables hopai
uses, so no separate load), runs the 9 traversal + 3 aggregation queries
against all four "engines", and writes one JSON file per tier with
warm-median timings, result counts, and per-query timeout/error status.

Run standalone, not through the notebook -- this can take a long time at
the larger tiers and is meant to be run once in the background, with the
notebook itself loading the resulting JSON rather than re-running the
full sweep on every read.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

import psycopg2
from sqlalchemy import create_engine

from hopai import AND, Avg, Count, Graph, GT, Hop, Max, Min, NOT, Start, Sum
from generate_graph import generate
from raw_cte import build_aggregate_sql, build_chain_sql, run_tagged_hydrated

PG_DSN_ADMIN = "postgresql+psycopg2://postgres:testpass@localhost:5432/hopai_bench"
PG_CONN_KW = {"host": "localhost", "port": 5432, "user": "postgres", "password": "testpass", "dbname": "hopai_bench"}
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "testpass123")

STATEMENT_TIMEOUT_MS = 90_000  # AGE / raw-CTE ceiling; matches the earlier "60-150s" finding
NEO4J_TIMEOUT_S = 90

TIERS = {
    "50k": 62_500,
    "500k": 625_000,
    "5M": 6_250_000,
}


SLOW_SKIP_REPEAT_MS = 3_000  # a query this slow cold gets ONE measurement, not four


def timed(fn, repeats=3, timeout_error=None):
    """Cold + `repeats` warm reps, median of the warm reps. Returns
    (cold_ms, warm_ms, result) or (None, None, "timeout") / (None, None, "error: ...").
    Skips the warm reps (reports cold_ms as warm_ms too) once the cold run
    alone is already slow -- averaging four multi-second/multi-minute runs
    buys nothing a single one didn't already show, and the wait compounds
    badly on Apache AGE's slower query shapes."""
    try:
        t0 = time.perf_counter()
        result = fn()
        cold_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001 -- benchmark driver, records the failure and moves on
        if timeout_error and timeout_error(exc):
            return None, None, "timeout"
        return None, None, f"error: {exc}"

    if cold_ms >= SLOW_SKIP_REPEAT_MS:
        return cold_ms, cold_ms, result

    warm_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        warm_times.append((time.perf_counter() - t0) * 1000)
    return cold_ms, statistics.median(warm_times), result


# ---------------------------------------------------------------------
# hopai / raw_cte (share one Postgres load)
# ---------------------------------------------------------------------

def load_postgres(data_dir: Path, schema: str):
    import bench_hopai
    engine = create_engine(PG_DSN_ADMIN)
    bench_hopai.load_data(engine, data_dir, schema)
    engine_scoped = create_engine(PG_DSN_ADMIN, connect_args={"options": f"-c search_path={schema}"})
    graph = Graph(engine_scoped)
    pg_conn = psycopg2.connect(**PG_CONN_KW)
    pg_conn.autocommit = True
    pg_cur = pg_conn.cursor()
    pg_cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
    return graph, pg_cur


TRAVERSAL_SUITE = [
    ("forward_1hop", [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=1)],
     {"type": "leaf"}, [{"where": {"flag": 1}, "min_hops": 1, "max_hops": 1}]),
    ("backward_1hop", [Start(where={"type": "hub"}), Hop(hops=1, direction="backward")],
     {"type": "hub"}, [{"direction": "backward", "min_hops": 1, "max_hops": 1}]),
    ("forward_bounded_4hop", [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=(1, 4))],
     {"type": "leaf"}, [{"where": {"flag": 1}, "min_hops": 1, "max_hops": 4}]),
    ("backward_bounded_3hop", [Start(where={"type": "hub"}), Hop(hops=(1, 3), direction="backward")],
     {"type": "hub"}, [{"direction": "backward", "min_hops": 1, "max_hops": 3}]),
    ("compound_2segment",
     [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=(1, 4)), Hop(where={"type": "hub"}, hops=(1, 3))],
     {"type": "leaf"}, [{"where": {"flag": 1}, "min_hops": 1, "max_hops": 4},
                         {"where": {"type": "hub"}, "min_hops": 1, "max_hops": 3}]),
    ("edge_or_tag", [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, via={"tag": ["p1", "p2"]}, hops=(1, 4))],
     {"type": "leaf"}, [{"where": {"flag": 1}, "via": {"tag": ["p1", "p2"]}, "min_hops": 1, "max_hops": 4}]),
    ("not_filter", [Start(where={"type": "hub"}), Hop(where=NOT({"type": "leaf"}), hops=(1, 3), direction="backward")],
     {"type": "hub"}, [{"where": {"$not": {"type": "leaf"}}, "direction": "backward", "min_hops": 1, "max_hops": 3}]),
    ("range_gt", [Start(where=AND({"type": "leaf"}, GT("priority", 5)))],
     {"$and": [{"type": "leaf"}, {"$gt": ["priority", 5]}]}, []),
    ("optional_last_hop",
     [Start(where={"type": "hub"}), Hop(where=NOT({"type": "leaf"}), hops=1, direction="backward"),
      Hop(where={"type": "leaf"}, hops=1, optional=True)],
     {"type": "hub"}, [{"where": {"$not": {"type": "leaf"}}, "direction": "backward", "min_hops": 1, "max_hops": 1},
                        {"where": {"type": "leaf"}, "min_hops": 1, "max_hops": 1, "optional": True}]),
]

AGG_SUITE = [
    ("agg_count_4hop", [Start(where={"type": "leaf"}), Hop(where={"flag": 1}, hops=(1, 4))], {"n": Count()},
     {"type": "leaf"}, [{"where": {"flag": 1}, "min_hops": 1, "max_hops": 4}], {"n": ("count", None)}),
    ("agg_stats_backward_4hop", [Start(where={"flag": 1}), Hop(hops=(1, 4), direction="backward")],
     {"n": Count(), "total": Sum("priority"), "mean": Avg("priority"), "lo": Min("priority"), "hi": Max("priority")},
     {"flag": 1}, [{"direction": "backward", "min_hops": 1, "max_hops": 4}],
     {"n": ("count", None), "total": ("sum", "priority"), "mean": ("avg", "priority"),
      "lo": ("min", "priority"), "hi": ("max", "priority")}),
    ("agg_range_count", [Start(where=AND({"type": "leaf"}, GT("priority", 5)))], {"n": Count(), "mean": Avg("priority")},
     {"$and": [{"type": "leaf"}, {"$gt": ["priority", 5]}]}, [], {"n": ("count", None), "mean": ("avg", "priority")}),
]


def run_hopai_traversals(graph):
    out = {}
    for name, hopai_hops, _raw_start, _raw_hops in TRAVERSAL_SUITE:
        start, *rest = hopai_hops
        cold, warm, res = timed(lambda start=start, rest=rest: graph.traverse(start, *rest))
        out[name] = {"cold_ms": cold, "warm_ms": warm,
                      "nodes": len(res.nodes) if warm else None, "edges": len(res.edges) if warm else None,
                      "status": res if isinstance(res, str) else "ok"}
    return out


def run_hopai_aggregates(graph):
    out = {}
    for name, hopai_hops, hopai_aggs, *_raw in AGG_SUITE:
        start, *rest = hopai_hops
        cold, warm, res = timed(lambda start=start, rest=rest, aggs=hopai_aggs: graph.aggregate(start, *rest, aggregates=aggs))
        out[name] = {"cold_ms": cold, "warm_ms": warm,
                      "result": dict(res) if warm else None, "status": res if isinstance(res, str) else "ok"}
    return out


def _is_timeout(exc):
    return "statement timeout" in str(exc).lower() or "query_canceled" in str(exc).lower()


def run_raw_cte_traversals(pg_cur, schema):
    out = {}
    for name, _hopai_hops, raw_start, raw_hops in TRAVERSAL_SUITE:
        sql, params = build_chain_sql(raw_start, raw_hops, schema)

        def go(sql=sql, params=params):
            return run_tagged_hydrated(pg_cur, sql, params, schema)

        cold, warm, res = timed(go, timeout_error=_is_timeout)
        out[name] = {"cold_ms": cold, "warm_ms": warm,
                      "nodes": len(res[0]) if warm else None, "edges": len(res[1]) if warm else None,
                      "status": res if isinstance(res, str) else "ok"}
    return out


def run_raw_cte_aggregates(pg_cur, schema):
    out = {}
    for name, _hopai_hops, _hopai_aggs, raw_start, raw_hops, raw_aggs in AGG_SUITE:
        sql, params = build_aggregate_sql(raw_start, raw_hops, schema, raw_aggs)

        def go(sql=sql, params=params):
            pg_cur.execute(sql, params)
            cols = [c.name for c in pg_cur.description]
            return dict(zip(cols, pg_cur.fetchone(), strict=True))

        cold, warm, res = timed(go, timeout_error=_is_timeout)
        out[name] = {"cold_ms": cold, "warm_ms": warm,
                      "result": res if warm else None, "status": res if isinstance(res, str) else "ok"}
    return out


# ---------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------

NEO4J_TRAVERSAL = {
    "forward_1hop": "MATCH (a:Node {type:'leaf'})-[:EDGE*1..1]->(m:Node {flag:1}) RETURN count(DISTINCT a) AS n",
    "backward_1hop": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..1]-(d:Node) RETURN count(DISTINCT d) AS n",
    "forward_bounded_4hop": "MATCH (a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) RETURN count(DISTINCT a) AS n",
    "backward_bounded_3hop": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..3]-(a:Node) RETURN count(DISTINCT a) AS n",
    "compound_2segment": ("MATCH (a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) "
                           "MATCH (m)-[:EDGE*1..3]->(h:Node {type:'hub'}) RETURN count(DISTINCT a) AS n"),
    "edge_or_tag": ("MATCH p=(a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) "
                     "WHERE all(r IN relationships(p) WHERE r.tag IN ['p1','p2']) RETURN count(DISTINCT a) AS n"),
    "not_filter": ("MATCH (h:Node {type:'hub'})<-[:EDGE*1..3]-(a:Node) "
                   "WHERE a.type IS NULL OR a.type <> 'leaf' RETURN count(DISTINCT a) AS n"),
    "range_gt": "MATCH (a:Node {type:'leaf'}) WHERE a.priority > 5 RETURN count(a) AS n",
    "optional_last_hop": ("MATCH (h:Node {type:'hub'})<-[:EDGE*1..1]-(a:Node) "
                           "WHERE a.type IS NULL OR a.type <> 'leaf' "
                           "OPTIONAL MATCH (a)-[:EDGE*1..1]->(dep:Node {type:'leaf'}) "
                           "RETURN count(DISTINCT a) AS n"),
}

NEO4J_AGG = {
    "agg_count_4hop": "MATCH (a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) RETURN count(DISTINCT m) AS n",
    "agg_stats_backward_4hop": ("MATCH (h:Node {flag:1})<-[:EDGE*1..4]-(a:Node) WITH DISTINCT a "
                                 "RETURN count(a) AS n, sum(a.priority) AS total, avg(a.priority) AS mean, "
                                 "min(a.priority) AS lo, max(a.priority) AS hi"),
    "agg_range_count": "MATCH (a:Node {type:'leaf'}) WHERE a.priority > 5 RETURN count(a) AS n, avg(a.priority) AS mean",
}


def load_neo4j(data_dir: Path):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, notifications_min_severity="OFF")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
        for index in ("type", "flag", "priority"):
            session.run(f"CREATE INDEX node_{index} IF NOT EXISTS FOR (n:Node) ON (n.{index})")
        session.run("CREATE INDEX edge_tag IF NOT EXISTS FOR ()-[r:EDGE]-() ON (r.tag)")

        batch = []
        with open(data_dir / "nodes.csv") as f:
            for row in csv.reader(f):
                props = json.loads(row[1]) if row[1].strip() else {}
                batch.append({"id": int(row[0]), **props})
                if len(batch) >= 20_000:
                    session.run("UNWIND $rows AS row CREATE (n:Node) SET n = row", rows=batch)
                    batch = []
            if batch:
                session.run("UNWIND $rows AS row CREATE (n:Node) SET n = row", rows=batch)

        batch = []
        with open(data_dir / "edges.csv") as f:
            for row in csv.reader(f):
                batch.append({"a": int(row[0]), "b": int(row[1]), "tag": row[2]})
                if len(batch) >= 20_000:
                    session.run("UNWIND $rows AS row MATCH (a:Node {id: row.a}), (b:Node {id: row.b}) "
                                "CREATE (a)-[:EDGE {tag: row.tag}]->(b)", rows=batch)
                    batch = []
            if batch:
                session.run("UNWIND $rows AS row MATCH (a:Node {id: row.a}), (b:Node {id: row.b}) "
                            "CREATE (a)-[:EDGE {tag: row.tag}]->(b)", rows=batch)
    return driver


def run_neo4j_query(driver, query):
    with driver.session() as session:
        tx = session.begin_transaction(timeout=NEO4J_TIMEOUT_S)
        try:
            rec = tx.run(query).single()
            tx.commit()
            return dict(rec)
        finally:
            tx.close()


def run_neo4j_traversals(driver):
    out = {}
    for name, q in NEO4J_TRAVERSAL.items():
        cold, warm, res = timed(lambda q=q: run_neo4j_query(driver, q))
        out[name] = {"cold_ms": cold, "warm_ms": warm,
                      "count": res.get("n") if warm else None, "status": res if isinstance(res, str) else "ok"}
    return out


def run_neo4j_aggregates(driver):
    out = {}
    for name, q in NEO4J_AGG.items():
        cold, warm, res = timed(lambda q=q: run_neo4j_query(driver, q))
        out[name] = {"cold_ms": cold, "warm_ms": warm,
                      "result": res if warm else None, "status": res if isinstance(res, str) else "ok"}
    return out


# ---------------------------------------------------------------------
# Apache AGE -- bulk SQL load (not row-by-row Cypher CREATE; that path
# is not what a real bulk import would use either), GIN index on
# properties to match hopai's own indexing, per-query statement_timeout.
# ---------------------------------------------------------------------

AGE_GRAPH = "hopai_bench_age"

AGE_TRAVERSAL = dict(NEO4J_TRAVERSAL)
AGE_TRAVERSAL["edge_or_tag"] = """
MATCH (a:Node {type:'leaf'})-[r1:EDGE]->(m:Node {flag:1}) WHERE r1.tag IN ['p1','p2'] RETURN a
UNION
MATCH (a:Node {type:'leaf'})-[r1:EDGE]->(x1:Node)-[r2:EDGE]->(m:Node {flag:1})
  WHERE r1.tag IN ['p1','p2'] AND r2.tag IN ['p1','p2'] RETURN a
UNION
MATCH (a:Node {type:'leaf'})-[r1:EDGE]->(x1:Node)-[r2:EDGE]->(x2:Node)-[r3:EDGE]->(m:Node {flag:1})
  WHERE r1.tag IN ['p1','p2'] AND r2.tag IN ['p1','p2'] AND r3.tag IN ['p1','p2'] RETURN a
UNION
MATCH (a:Node {type:'leaf'})-[r1:EDGE]->(x1:Node)-[r2:EDGE]->(x2:Node)-[r3:EDGE]->(x3:Node)-[r4:EDGE]->(m:Node {flag:1})
  WHERE r1.tag IN ['p1','p2'] AND r2.tag IN ['p1','p2'] AND r3.tag IN ['p1','p2'] AND r4.tag IN ['p1','p2'] RETURN a
""".strip()

AGE_AGG = dict(NEO4J_AGG)


def load_age(data_dir: Path):
    conn = psycopg2.connect(**PG_CONN_KW)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS age")
    cur.execute("LOAD 'age'")
    cur.execute('SET search_path = ag_catalog, "$user", public')
    cur.execute("SELECT count(*) FROM ag_graph WHERE name = %s", (AGE_GRAPH,))
    if cur.fetchone()[0]:
        cur.execute("SELECT drop_graph(%s, true)", (AGE_GRAPH,))
    conn.commit()
    cur.execute("SELECT create_graph(%s)", (AGE_GRAPH,))
    cur.execute("SELECT create_vlabel(%s, 'Node')", (AGE_GRAPH,))
    cur.execute("SELECT create_elabel(%s, 'EDGE')", (AGE_GRAPH,))

    cur.execute("CREATE TEMP TABLE stage_nodes (id bigint, props jsonb)")
    with open(data_dir / "nodes.csv") as f:
        cur.copy_expert(
            "COPY stage_nodes (id, props) FROM STDIN WITH (FORMAT csv, NULL '')", f)
    cur.execute(f"""
        INSERT INTO {AGE_GRAPH}."Node" (id, properties)
        SELECT _graphid(_label_id(%s,'Node')::int, id), coalesce(props, '{{}}'::jsonb)::text::agtype
        FROM stage_nodes
    """, (AGE_GRAPH,))

    cur.execute("CREATE TEMP TABLE stage_edges (start_id bigint, end_id bigint, tag text)")
    with open(data_dir / "edges.csv") as f:
        cur.copy_expert(
            "COPY stage_edges (start_id, end_id, tag) FROM STDIN WITH (FORMAT csv)", f)
    cur.execute(f"""
        INSERT INTO {AGE_GRAPH}."EDGE" (id, start_id, end_id, properties)
        SELECT _graphid(_label_id(%s,'EDGE')::int, row_number() over ()),
               _graphid(_label_id(%s,'Node')::int, start_id),
               _graphid(_label_id(%s,'Node')::int, end_id),
               json_build_object('tag', tag)::text::agtype
        FROM stage_edges
    """, (AGE_GRAPH, AGE_GRAPH, AGE_GRAPH))

    cur.execute(f'CREATE INDEX ON {AGE_GRAPH}."Node" USING gin (properties)')
    cur.execute(f'CREATE INDEX ON {AGE_GRAPH}."EDGE" USING gin (properties)')
    cur.execute(f'ANALYZE {AGE_GRAPH}."Node"')
    cur.execute(f'ANALYZE {AGE_GRAPH}."EDGE"')
    conn.commit()
    cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
    return conn, cur


def run_age_query(cur, query, columns):
    """columns: list of output names. For edge_or_tag's UNION-of-nodes
    form, pass columns=None to count rows via plain SQL COUNT(*) instead
    (AGE 1.5.0 has no `all()`/list-comprehension, so that query is
    rewritten as a UNION of per-hop-length MATCHes each RETURNing the
    node -- see AGE_TRAVERSAL["edge_or_tag"])."""
    if columns is None:
        cur.execute(f"SELECT count(*) FROM cypher('{AGE_GRAPH}', $$ {query} $$) as (a agtype)")
        return {"n": cur.fetchone()[0]}
    decl = ", ".join(f"c{i} agtype" for i in range(len(columns)))
    cur.execute(f"SELECT * FROM cypher('{AGE_GRAPH}', $$ {query} $$) as ({decl})")
    row = cur.fetchone()
    return {name: json.loads(val) for name, val in zip(columns, row, strict=True)} if row else {}


AGE_TRAVERSAL_COLUMNS = {name: None if name == "edge_or_tag" else ["n"] for name in AGE_TRAVERSAL}
AGE_AGG_COLUMNS = {
    "agg_count_4hop": ["n"],
    "agg_stats_backward_4hop": ["n", "total", "mean", "lo", "hi"],
    "agg_range_count": ["n", "mean"],
}


def run_age_traversals(conn, cur):
    out = {}
    for name, q in AGE_TRAVERSAL.items():
        cols = AGE_TRAVERSAL_COLUMNS[name]

        def go(q=q, cols=cols):
            conn.rollback()
            return run_age_query(cur, q, cols)

        cold, warm, res = timed(go, timeout_error=_is_timeout)
        out[name] = {"cold_ms": cold, "warm_ms": warm,
                      "count": res.get("n") if warm else None, "status": res if isinstance(res, str) else "ok"}
    return out


def run_age_aggregates(conn, cur):
    out = {}
    for name, q in AGE_AGG.items():
        cols = AGE_AGG_COLUMNS[name]

        def go(q=q, cols=cols):
            conn.rollback()
            return run_age_query(cur, q, cols)

        cold, warm, res = timed(go, timeout_error=_is_timeout)
        out[name] = {"cold_ms": cold, "warm_ms": warm,
                      "result": res if warm else None, "status": res if isinstance(res, str) else "ok"}
    return out


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def run_tier(tier_name: str, n_nodes: int, out_path: Path):
    print(f"=== tier {tier_name} (n_nodes={n_nodes}) ===", flush=True)
    data_dir = Path(f"/tmp/bench_data_{tier_name}")
    schema = f"hopai_bench_{tier_name}"
    t0 = time.time()
    summary = generate(n_nodes=n_nodes, seed=42, out_dir=data_dir)
    print(f"generated: {summary} ({time.time()-t0:.1f}s)", flush=True)

    results = {"tier": tier_name, "summary": summary}

    def stage(label, fn):
        """Run one engine's block; a crash (a dropped service, e.g.) is
        recorded rather than losing every other engine's results for
        this tier -- each tier already costs real time to regenerate."""
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- driver script, keep going
            print(f"{label} FAILED: {exc}", flush=True)
            results[f"{label}_error"] = str(exc)
        finally:
            out_path.write_text(json.dumps(results, indent=2, default=str))

    def do_postgres():
        t0 = time.time()
        graph, pg_cur = load_postgres(data_dir, schema)
        print(f"postgres/hopai loaded ({time.time()-t0:.1f}s)", flush=True)
        results["hopai_traversal"] = run_hopai_traversals(graph)
        print("hopai traversals done", flush=True)
        results["hopai_aggregate"] = run_hopai_aggregates(graph)
        print("hopai aggregates done", flush=True)
        results["raw_cte_traversal"] = run_raw_cte_traversals(pg_cur, schema)
        print("raw_cte traversals done", flush=True)
        results["raw_cte_aggregate"] = run_raw_cte_aggregates(pg_cur, schema)
        print("raw_cte aggregates done", flush=True)

    def do_age():
        t0 = time.time()
        conn_age, cur_age = load_age(data_dir)
        print(f"age loaded ({time.time()-t0:.1f}s)", flush=True)
        results["age_traversal"] = run_age_traversals(conn_age, cur_age)
        print("age traversals done", flush=True)
        results["age_aggregate"] = run_age_aggregates(conn_age, cur_age)
        print("age aggregates done", flush=True)

    def do_neo4j():
        t0 = time.time()
        neo_driver = load_neo4j(data_dir)
        print(f"neo4j loaded ({time.time()-t0:.1f}s)", flush=True)
        results["neo4j_traversal"] = run_neo4j_traversals(neo_driver)
        print("neo4j traversals done", flush=True)
        results["neo4j_aggregate"] = run_neo4j_aggregates(neo_driver)
        print("neo4j aggregates done", flush=True)
        neo_driver.close()

    stage("postgres", do_postgres)
    stage("age", do_age)
    stage("neo4j", do_neo4j)

    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"=== tier {tier_name} done, written to {out_path} ===", flush=True)


if __name__ == "__main__":
    tiers = sys.argv[1:] or list(TIERS.keys())
    for t in tiers:
        out_dir = Path(__file__).parent / "multiscale_results"
        out_dir.mkdir(exist_ok=True)
        run_tier(t, TIERS[t], out_dir / f"{t}.json")
