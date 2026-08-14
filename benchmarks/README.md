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

`bench_hopai.py` loads it and times nine queries covering direction,
multi-hop bounds, compound chains, `OR`, `NOT`, range comparisons, and
`OPTIONAL` — cold and warm.

Each run writes two files, **both overwritten every time**:

- `RESULTS.md` — the report: ASCII charts of cold and warm latency with
  the numbers beside them, a table of every measurement, and the machine
  it ran on (CPU, cores, RAM, OS, Python, PostgreSQL version, and the
  server settings that actually move these numbers — `shared_buffers`,
  `work_mem`, `effective_cache_size`, parallel workers, JIT).
- `bench_results.json` — the same measurements, raw.

Both are git-ignored, because a benchmark number belongs to the machine
that produced it. Commit one deliberately (`git add -f`) when you want to
publish a specific run.

**A query that returns zero rows is called out, in the chart and at the
top of the report.** Its timing is real and meaningless — finding nothing
is always fast — so the report refuses to let it read as the winner.

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
