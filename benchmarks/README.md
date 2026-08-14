# Benchmarks

## hopai vs. raw Postgres, on the same data

```bash
python generate_graph.py --nodes 1000000 --seed 42 --out-dir ./data
python bench_hopai.py --data-dir ./data --dsn "postgresql+psycopg2://user:pass@host/db"
```

`generate_graph.py` produces a graph with a known, verifiable shape: a
sparse random background DAG plus a deliberately structured "hub"
subgraph (a widely-shared node with real fan-in across several depth
levels — 15/75/225/675/1350/2700/5400 nodes at depths 1 through 7 with
the default settings). That structure, not a purely random graph, is
what actually stresses a graph engine — random graphs rarely have the
convergent fan-in that real dependency graphs do.

`bench_hopai.py` loads it and times twelve queries — nine traversals
covering direction, multi-hop bounds, compound chains, `OR`, `NOT`,
range comparisons, and `OPTIONAL`, plus three aggregations
(`Count`/`Sum`/`Avg`/`Min`/`Max`) — cold and warm, writing results to
`bench_results.json`.

`agg_count_4hop` deliberately runs the same chain as
`forward_bounded_4hop`: the pair shows what `graph.aggregate()` saves by
skipping edge reconstruction and node hydration on identical traversal
work. In the run recorded during development (default 1M-node graph,
local Postgres 16) the aggregate answered in 75ms warm against the
traversal's 653ms — the difference is the `hop_edges`/`edge_rows` CTEs
and the hydration of ~10k nodes and edges that a count does not need.

## Comparing against Neo4j and Apache AGE

These require separate running instances this repo doesn't set up for
you. The Cypher equivalents for the same nine queries (substitute your
own label/property names):

```cypher
// forward_1hop
MATCH (a:Node {type:'leaf'})-[:EDGE*1..1]->(m:Node {flag:1}) RETURN count(DISTINCT a)

// backward_1hop
MATCH (h:Node {type:'hub'})<-[:EDGE*1..1]-(d:Node) RETURN count(DISTINCT d)

// forward_bounded_4hop
MATCH (a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) RETURN count(DISTINCT a)

// backward_bounded_3hop
MATCH (h:Node {type:'hub'})<-[:EDGE*1..3]-(a:Node) RETURN count(DISTINCT a)

// compound_2segment
MATCH (a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1})
MATCH (m)-[:EDGE*1..3]->(h:Node {type:'hub'})
RETURN count(DISTINCT a)

// edge_or_tag -- Neo4j supports all(); AGE (as of 1.5.0) does not and
// needs the UNWIND+CASE workaround documented in the main investigation
MATCH p=(a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1})
WHERE all(r IN relationships(p) WHERE r.tag IN ['p1','p2'])
RETURN count(DISTINCT a)

// not_filter -- use the NULL-safe form; naive `<> 'leaf'` silently
// excludes nodes with no `type` property at all on both Neo4j and AGE
MATCH (h:Node {type:'hub'})<-[:EDGE*1..3]-(a:Node)
WHERE a.type IS NULL OR a.type <> 'leaf'
RETURN count(DISTINCT a)

// range_gt
MATCH (a:Node {type:'leaf'}) WHERE a.priority > 5 RETURN count(a)

// optional_last_hop
MATCH (h:Node {type:'hub'})<-[:EDGE*1..1]-(a:Node)
WHERE a.type IS NULL OR a.type <> 'leaf'
OPTIONAL MATCH (a)-[:EDGE*1..1]->(dep:Node {type:'leaf'})
RETURN count(DISTINCT a), count(DISTINCT dep)
```

A note on those `RETURN count(DISTINCT a)` tails now that hopai
translates aggregation: they aggregate the *start* variable, which hopai
still refuses (only the last node of a chain can be aggregated — see
`hopai/cypher.py`). They are written that way because Neo4j and AGE
count them fine, and on those systems the tail is just "how many rows".
The three `agg_*` queries in `bench_hopai.py` are the shapes hopai's own
Cypher front end runs directly, e.g.
`MATCH (a {type: 'leaf'})-[*1..4]->(m {flag: 1}) RETURN count(DISTINCT m)`.

**A finding worth knowing before you run these on AGE:** in the full
investigation this library came out of, two of these nine query shapes
(a bounded backward traversal, and `OPTIONAL` layered on one) did not
complete on Apache AGE 1.5.0 within a 60-150 second budget, on the exact
same data that answered in under a second on Neo4j and raw Postgres. Set
a `statement_timeout` before running these against AGE, or a single
query can tie up your session indefinitely.

## bench_postgres_cte -- the honest floor

`bench_hopai.py`'s numbers include hopai's own overhead (SQLAlchemy
query construction, cycle-protection path tracking). If you want the
absolute floor -- hand-written recursive CTEs against the same data with
none of that -- the query shapes hopai generates are visible by
calling `graph.build_query(...)` and inspecting the compiled statement.
In the original investigation, raw CTEs were faster than hopai on
most queries by a factor of 2-5x, and hopai was faster than raw CTEs
on none -- that gap is the honest price of the API's convenience and
correctness guarantees (path tracking, dead-end pruning, automatic
node/edge derivation), not a hidden cost.
