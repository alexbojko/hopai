"""Hand-written recursive-CTE equivalents of benchmarks/bench_hopai.py's
suite, executed with plain psycopg2 -- no SQLAlchemy, no hopai. This is
the "honest floor": the same walk shape hopai's own build_query() emits,
timed without the query-building and result-hydration layer around it.
"""
from __future__ import annotations
import json


def compile_filter(col, f):
    """dict / {'$not': f} / {'$and': [f, ...]} / {'$gt': [key, val]} ->
    (sql_fragment, params), mirroring filters.resolve()'s semantics for
    exactly the shapes this benchmark suite uses."""
    if isinstance(f, dict) and "$not" in f:
        inner_sql, inner_params = compile_filter(col, f["$not"])
        return f"NOT ({inner_sql})", inner_params
    if isinstance(f, dict) and "$and" in f:
        parts, params = [], []
        for sub in f["$and"]:
            s, sp = compile_filter(col, sub)
            parts.append(s)
            params.extend(sp)
        return "(" + " AND ".join(parts) + ")", params
    if isinstance(f, dict) and "$gt" in f:
        key, val = f["$gt"]
        return f"(({col}->>%s)::numeric > %s)", [key, val]
    # Plain dict: AND across keys; a list value is OR-of-containment
    # across its items (IN-like), matching filters.resolve()'s per-key
    # compilation -- not one single containment check of the whole dict,
    # which cannot express the list/IN case at all.
    parts, params = [], []
    for key, value in f.items():
        if isinstance(value, list):
            sub_parts, sub_params = [], []
            for item in value:
                sub_parts.append(f"({col} @> %s::jsonb)")
                sub_params.append(json.dumps({key: item}))
            parts.append("(" + " OR ".join(sub_parts) + ")")
            params += sub_params
        else:
            parts.append(f"({col} @> %s::jsonb)")
            params.append(json.dumps({key: value}))
    return "(" + " AND ".join(parts) + ")", params


def build_chain_sql(start_where, hops, schema):
    """hops: list of dicts with where=, via=, min_hops=, max_hops=,
    direction=, optional=. Returns (sql, params) for the tagged
    (kind, id) union hopai's build_query() also produces."""
    params = []
    seed_sql, p = compile_filter("nodes.properties", start_where)
    params += p
    ctes = [f"seed AS (SELECT nodes.id AS node_id FROM {schema}.nodes WHERE {seed_sql})"]

    if not hops:
        sql = "WITH " + ctes[0] + " SELECT 'node' AS kind, CAST(seed.node_id AS VARCHAR) AS id FROM seed"
        return sql, params

    prev, prev_col = "seed", "node_id"
    anchor_before_last = None  # for optional=True on the last hop

    for i, hop in enumerate(hops):
        direction = hop.get("direction", "forward")
        join_col, move_col = ("start_id", "end_id") if direction == "forward" else ("end_id", "start_id")
        min_hops = hop.get("min_hops", 1)
        max_hops = hop.get("max_hops", min_hops)
        via_sql, via_p = ("TRUE", [])
        if hop.get("via"):
            via_sql, via_p = compile_filter("edges.properties", hop["via"])
        via_sql_rec = via_sql.replace("edges.properties", "e.properties")

        ctes.append(
            f"walk_{i}(from_id, to_id, depth, local_path, local_edges) AS ("
            f"SELECT {prev}.{prev_col} AS from_id, edges.{move_col} AS to_id, 1 AS depth, "
            f"ARRAY[{prev}.{prev_col}, edges.{move_col}] AS local_path, ARRAY[edges.id] AS local_edges "
            f"FROM {schema}.edges JOIN {prev} ON edges.{join_col} = {prev}.{prev_col} "
            f"WHERE {via_sql} "
            f"UNION ALL "
            f"SELECT w.from_id, e.{move_col}, w.depth + 1, w.local_path || e.{move_col}, w.local_edges || e.id "
            f"FROM walk_{i} w JOIN {schema}.edges e ON e.{join_col} = w.to_id "
            f"WHERE w.depth < %s AND {via_sql_rec} "
            f"AND NOT (e.{move_col} = ANY(w.local_path))"
            f")"
        )
        params += via_p
        params.append(max_hops)
        params += via_p

        where_sql, where_p = ("TRUE", [])
        if hop.get("where"):
            where_sql, where_p = compile_filter("nodes.properties", hop["where"])
        ctes.append(
            f"match_{i} AS (SELECT DISTINCT walk_{i}.to_id AS node_id FROM walk_{i} "
            f"JOIN {schema}.nodes ON nodes.id = walk_{i}.to_id AND {where_sql} "
            f"WHERE walk_{i}.depth >= %s)"
        )
        params += where_p
        params.append(min_hops)

        ctes.append(
            f"hop_edges_{i} AS (SELECT DISTINCT unnest(walk_{i}.local_edges) AS edge_id FROM walk_{i} "
            f"WHERE walk_{i}.depth >= %s AND walk_{i}.to_id IN (SELECT node_id FROM match_{i}))"
        )
        params.append(min_hops)

        if i == len(hops) - 1 and hop.get("optional"):
            anchor_before_last = (prev, prev_col)

        prev, prev_col = f"match_{i}", "node_id"

    ctes.append(
        "all_edges AS (" +
        " UNION ".join(f"SELECT edge_id FROM hop_edges_{i}" for i in range(len(hops))) +
        ")"
    )
    ctes.append(
        f"edge_rows AS (SELECT edges.id AS eid, edges.start_id AS a, edges.end_id AS b "
        f"FROM {schema}.edges JOIN all_edges ON edges.id = all_edges.edge_id)"
    )

    branches = [
        "SELECT 'node' AS kind, CAST(edge_rows.a AS VARCHAR) AS id FROM edge_rows",
        "SELECT 'node', CAST(edge_rows.b AS VARCHAR) FROM edge_rows",
    ]
    if anchor_before_last is not None:
        a_tbl, a_col = anchor_before_last
        branches.append(f"SELECT 'node', CAST({a_tbl}.{a_col} AS VARCHAR) FROM {a_tbl}")
    branches.append("SELECT 'edge', CAST(edge_rows.eid AS VARCHAR) FROM edge_rows")

    sql = "WITH RECURSIVE " + ", ".join(ctes) + " " + " UNION ".join(branches)
    return sql, params


def build_aggregate_sql(start_where, hops, schema, aggregates):
    """aggregates: dict of {alias: ('count'|'sum'|'avg'|'min'|'max', key_or_None)}.
    Aggregates over the DISTINCT nodes the last hop matched (or the seed
    set with no hops), mirroring Graph.aggregate() -- no edge
    reconstruction needed, so no hop_edges/edge_rows CTEs at all."""
    params = []
    seed_sql, p = compile_filter("nodes.properties", start_where)
    params += p
    ctes = [f"seed AS (SELECT nodes.id AS node_id FROM {schema}.nodes WHERE {seed_sql})"]

    prev, prev_col = "seed", "node_id"
    for i, hop in enumerate(hops):
        direction = hop.get("direction", "forward")
        join_col, move_col = ("start_id", "end_id") if direction == "forward" else ("end_id", "start_id")
        min_hops = hop.get("min_hops", 1)
        max_hops = hop.get("max_hops", min_hops)
        via_sql, via_p = ("TRUE", [])
        if hop.get("via"):
            via_sql, via_p = compile_filter("edges.properties", hop["via"])
        via_sql_rec = via_sql.replace("edges.properties", "e.properties")

        ctes.append(
            f"walk_{i}(from_id, to_id, depth, local_path) AS ("
            f"SELECT {prev}.{prev_col} AS from_id, edges.{move_col} AS to_id, 1 AS depth, "
            f"ARRAY[{prev}.{prev_col}, edges.{move_col}] AS local_path "
            f"FROM {schema}.edges JOIN {prev} ON edges.{join_col} = {prev}.{prev_col} "
            f"WHERE {via_sql} "
            f"UNION ALL "
            f"SELECT w.from_id, e.{move_col}, w.depth + 1, w.local_path || e.{move_col} "
            f"FROM walk_{i} w JOIN {schema}.edges e ON e.{join_col} = w.to_id "
            f"WHERE w.depth < %s AND {via_sql_rec} "
            f"AND NOT (e.{move_col} = ANY(w.local_path))"
            f")"
        )
        params += via_p
        params.append(max_hops)
        params += via_p

        where_sql, where_p = ("TRUE", [])
        if hop.get("where"):
            where_sql, where_p = compile_filter("nodes.properties", hop["where"])
        ctes.append(
            f"match_{i} AS (SELECT DISTINCT walk_{i}.to_id AS node_id FROM walk_{i} "
            f"JOIN {schema}.nodes ON nodes.id = walk_{i}.to_id AND {where_sql} "
            f"WHERE walk_{i}.depth >= %s)"
        )
        params += where_p
        params.append(min_hops)
        prev, prev_col = f"match_{i}", "node_id"

    agg_exprs = []
    for alias, (fn, key) in aggregates.items():
        if fn == "count" and key is None:
            agg_exprs.append(f"count(*) AS {alias}")
        else:
            field = f"(nodes.properties->>'{key}')::numeric"
            agg_exprs.append(f"{fn}({field}) AS {alias}")
    sql = ("WITH RECURSIVE " + ", ".join(ctes) +
           f" SELECT {', '.join(agg_exprs)} FROM {prev} "
           f"JOIN {schema}.nodes ON nodes.id = {prev}.{prev_col}")
    return sql, params


def run_tagged(cur, sql, params):
    cur.execute(sql, params)
    nodes, edges = set(), set()
    for kind, id_ in cur.fetchall():
        (nodes if kind == "node" else edges).add(id_)
    return nodes, edges


def run_tagged_hydrated(cur, sql, params, schema):
    """The full round trip hopai's traverse() makes: the tagged-union
    walk, then two hydration SELECTs for node/edge properties -- so
    "raw CTE" times the same total work, not just the walk fragment."""
    nodes, edges = run_tagged(cur, sql, params)
    if nodes:
        cur.execute(f"SELECT id, properties FROM {schema}.nodes WHERE id = ANY(%s)",
                    ([int(n) for n in nodes],))
        cur.fetchall()
    if edges:
        cur.execute(f"SELECT id, start_id, end_id, properties FROM {schema}.edges WHERE id = ANY(%s)",
                    ([int(e) for e in edges],))
        cur.fetchall()
    return nodes, edges
