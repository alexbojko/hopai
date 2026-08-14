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

`bench_hopai.py` loads it and times **29 queries**:

- **Q1–Q14, traversals** — forward, backward and mixed direction, bounded and
  deep 12-hop, compound chains, `OR` on nodes and on edges, `NOT`, range
  comparisons, composition and `OPTIONAL`.
- **Q15–Q29, aggregations** — graded, because aggregation is not one operation
  whose cost you can quote once. What it costs depends on how much the walk
  underneath had to match, how many aggregates run over it, and whether
  `DISTINCT` forces a sort:
  - *simple* — no traversal at all; the aggregate is the whole query
    (`Count`, `Avg` over a range, `Min`/`Max`).
  - *complex* — a real walk underneath (`Count` after one hop, over a bounded
    chain, over a `NOT` filter, `Count(property)` counting nodes that *have* it,
    five statistics at once).
  - *very complex* — deep and compound walks, `DISTINCT` over large match sets,
    edge-filtered chains, and seven aggregates in a single query.

Four aggregates deliberately reuse a traversal's exact chain — `Q19`/`Q3`,
`Q25`/`Q5`, `Q26`/`Q7`, `Q27`/`Q14`. Each pair is the only honest way to quote
what `graph.aggregate()` saves: same walk, same match set, one materialising a
subgraph and one returning a number. The report computes the ratio for you.

The report also groups every query by tier, so "how fast are aggregations" is a
question with an answer instead of a single misleading number.

Each query runs once cold, then `--repeat` times warm (default 5), and the
**median** is reported with its range. A single warm sample on a shared machine
moves further than most regressions worth catching, so a one-shot number cannot
tell a real change from the machine breathing — comparing two commits means
comparing medians and checking whether the ranges overlap.

A query that overruns `--budget` seconds (default 150) is cancelled by the
server and reported as **DNF** — its own outcome, never a large number that
would average in with real measurements.

Each run writes two files, **both overwritten every time**:

- `RESULTS.md` — the report: headline figures, a log-scale ASCII chart of every
  query, the full results table, derived findings, and the machine it ran on
  (CPU, cores, RAM, OS, Python, PostgreSQL version, and the server settings that
  actually move these numbers — `shared_buffers`, `work_mem`,
  `effective_cache_size`, parallel workers, JIT).
- `bench_results.json` — the same measurements, raw.

Both are git-ignored, because a benchmark number belongs to the machine that
produced it. Commit one deliberately (`git add -f`) to publish a specific run.

**A query that returns zero rows is called out** in the findings. Its timing is
real and meaningless — finding nothing is always fast — so the report refuses to
let it read as the winner.

## Comparing against Neo4j and Apache AGE

```bash
docker compose --profile compare up -d          # neo4j + apache/age
python bench_hopai.py --data-dir ./data --dsn "postgresql+psycopg2://..." \
    --neo4j-url http://localhost:7474 \
    --age-dsn "postgresql://postgres:testpass@localhost:5433/agebench"
```

Both load the **same CSVs** and answer the **same question**, and the report
verifies the answers agree before putting the timings side by side — a speed
comparison between systems returning different results is not a comparison.
Timings, answers and any disagreement all land in `RESULTS.md`.

**The comparable metric is `count(DISTINCT endpoint)`, not hopai's node count.**
`traverse()` returns every node on a matching chain — seeds and intermediates
included — which is a different question from what Cypher is asked. The runner
uses `graph.aggregate(..., Count())`, which counts exactly what the last hop
matched.

**One real disagreement, recorded rather than smoothed over.** On `Q6`
(`(h)<-[]-(m)-[]->(x)`) hopai answers 1 and both Neo4j and AGE answer 0. Cypher
enforces *relationship uniqueness* within a MATCH — the same edge may not be
walked twice in one pattern — so `x = h` is excluded. hopai's hops are
independent steps and have no such rule. Neither is wrong; they are different
questions. It also means `traverse_cypher()` of that pattern does not answer
what Neo4j would, which is worth knowing before treating the Cypher front end
as a drop-in.

No new dependencies: Neo4j is driven through its HTTP query API with `urllib`,
and AGE is a PostgreSQL extension that `psycopg2` already reaches.

### The Cypher run by each system

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

## The honest floor

`bench_hopai.py` measures it on every run (`--no-baseline` to skip). For each
query it takes the statement hopai just built, executes it **straight through
the driver** — no SQLAlchemy result mapping, no property hydration, no
`Subgraph` — and reports both numbers with the ratio between them.

That is the floor for the same query. It is deliberately not two hand-written
queries: one statement, measured twice, cannot drift out of step with the
library the way a parallel hand-maintained SQL file always does.

On a 50k-node graph the gap ran 1.3x-10.8x, clustering around 3.5-4.8x on the
queries that return real volume. Read the small-result rows carefully: hopai's
per-call cost is roughly fixed (two extra round trips to hydrate properties),
so it dominates a query returning 16 nodes and nearly disappears on one
returning 10,000. Quote the ratio next to the row size, never on its own.

The gap is what the API buys — result mapping, dead-end pruning, automatic
node/edge derivation, cycle protection. It is a price, stated, not a hidden
cost.
