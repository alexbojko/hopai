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

## Vector search

```bash
python bench_vectors.py --dsn "postgresql+psycopg2://user:pass@host/db" --rows 20000 --dims 384
```

Self-contained (generates its own vectors), and exists so the cost
model in `hopai/vectors.py` stays a measurement. The transferable
number is `us_per_element`: the exact-cosine scan pays a fixed
executor cost per vector element, so a candidate row costs about
`dimensions x that`. Recorded during development (Postgres 16, one
core): **0.13 µs/element** — an unfiltered 20k × 384-dim search in
~1.0 s, the same search `where=`-filtered to 25 % of rows in ~0.25 s,
a similarity-seeded traversal within a few ms of its seed search.

That number was **0.28 µs/element until the similarity moved into a
LATERAL**. As a correlated scalar subquery inside a plain sub-SELECT
it was pulled up by the planner and re-evaluated at every site the
outer query named it — the filter, the score, the `ORDER BY` — so the
`unnest` ran twice per candidate (three times with `min_similarity`)
for identical results. `EXPLAIN ANALYZE` showed the duplicate SubPlans
at `loops=<candidates>` each. Worth knowing before "simplifying" the
LATERAL back into a scalar subquery: it reads tidier and costs 2×.

Per-element casting variants (array-level float8 cast, float4
accumulation) were measured within ±15 % of each other — the cost is
the executor's per-tuple work, not the arithmetic — which is why the
SQL keeps the formulation whose float8 accumulation is correct.

### Boost normalization (`scale="normalized"`, the default)

`Boost`'s default min-max window-function rescaling adds a per-query cost
on top of the search above, measured the same way — 20k rows, 384 dims,
`where=`-filtered to ~25% of rows (~5000 candidates), 5 repeats, warm:

| | ms |
| --- | ---: |
| no boost | 13.5 |
| `scale="raw"` | 13.2 |
| `scale="normalized"` (default) | 13.2 |

Indistinguishable from noise at this scale — the two window functions
(`min`/`max`) are cheap next to the LATERAL similarity scan they ride
alongside. One caveat worth recording rather than glossing over: the
normalized form's SQL evaluates `min(coalesced) OVER ()` **twice**
(once directly, once again inside `max - min`) where two distinct
window functions would suffice — at a much larger candidate count
(~200k, well past where a single traversal is comfortable regardless)
that redundant third pass measured roughly 8% slower than a form that
computes `lo`/`hi` once and reuses them. Left as the simpler
implementation rather than restructured, since the win is invisible at
the candidate-set sizes this library targets (see "What this costs" in
the main README) and the restructuring itself would touch every boost
call site — a real optimization if `Boost` is ever profiled at that
scale, not a correctness concern either way.

## Building reranker documents

```bash
python bench_documents.py           # no database, no network, no arguments
```

Nothing here measured `Rerank.build_documents()`, which is why a 121ms
event-loop block lived in the reranking path with a green suite in front
of it. This measures µs/row against **row size** — the axis that was
invisible — and reports both the CPU cost and how long the event loop
went without its thread while the build ran.

The cost was never the filter's. `document_from` is evaluated through
`program.input_value(candidate)`, and `input_value` marshals the
**whole candidate** into jq's value representation before the filter
reads a field of it. So the price was proportional to the *candidate*,
not to the *projection*: `.properties.title` (60 characters out) cost
the same as `.properties.field_0` on the same 100KB row, because both
handed jq 100KB. `fields=` narrows what a filter may **read**; it never
narrowed what jq was **handed**.

Recorded during development (jq 1.12, one core, 50 candidates,
`.properties.title`), before pruning against after:

| KB per row | before | after | build held the loop, before → after |
| ---: | ---: | ---: | --- |
| 1 | 40 µs/row | 10 µs/row | 2.2 ms → 0.7 ms |
| 10 | 327 µs/row | 10 µs/row | 15.5 ms → 0.7 ms |
| 100 | 2879 µs/row | 13 µs/row | 146 ms → 0.7 ms |
| 500 | 15685 µs/row | 21 µs/row | 719 ms → 1.4 ms |

The transferable number is **`us_per_kb`**: the slope against payload,
**31.4 before and 0.02 after**. A run where it climbs back toward 25–30
is the regression this file exists to catch — `build_documents()` now
hands jq only the paths `jqsafe.paths_read()` says the filter reads
(`hopai/rerankers.py`'s `_projection_paths`/`_pruned`), and anything
that widens what is handed over puts the slope back.

`longest_gap_ms` in the same run is the least ambiguous of the three
loop numbers: how long, in one stretch, a 1ms ticker did not get the
thread. At 500KB it *equalled the build* — 719 ms, 0 ticks/s against an
idle 970 — total starvation, the signature of a `time.sleep` rather than
of a slow function. That is precisely what `ticks > 0` in
`tests/test_async.py` could not distinguish from scheduling noise, and
why those tests now compare against an idle baseline ratio instead.

One knob to know about before running this at scale: `build_documents()`
refuses above **`MAX_DOCUMENTS` (5000)**. `candidates=` bounds the
*nodes* a step offers, but `per_path=True` bills one document per
(node, route) and the route count is the graph's answer, not an option's
— `candidates=500` over nodes reached 200 ways measured **4.64 s** of
document building before a single provider call.

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
