# Benchmarks

## hopai vs. Apache AGE, a hand-written recursive CTE, and Neo4j

**[`benchmarks.ipynb`](https://hopai.readthedocs.io/en/latest/benchmarks/benchmarks/)**
is the main comparison: the same graph, the same nine traversal shapes and
three aggregations `bench_hopai.py` has run since this library's first
commit, timed against Apache AGE, a hand-written recursive CTE (the honest
floor — same walk, no SQLAlchemy, no Python-side hydration layer) and a
real Neo4j instance loaded through its Python driver — each swept across
**three graph sizes** (50k, 500k and 5M edges), so the notebook shows how
each gap moves with scale rather than what it is at one arbitrary size.
Charts, tables, and the methodology notes (timeouts, repeats, the one
asymmetry worth reading the AGE/Neo4j numbers through) are all in the
notebook rather than repeated here.

```bash
python generate_graph.py --nodes 1000000 --seed 42 --out-dir ./data
python bench_hopai.py --data-dir ./data --dsn "postgresql+psycopg2://user:pass@host/db"
```

`generate_graph.py` (used by the notebook too) produces a graph with a
known, verifiable shape: a sparse random background DAG plus a deliberately
structured "hub" subgraph (a widely-shared node with real fan-in across
several depth levels — 15/75/225/675/1350/2700/5400 nodes at depths 1
through 7 with the default settings). That structure, not a purely random
graph, is what actually stresses a graph engine — random graphs rarely
have the convergent fan-in that real dependency graphs do.

`bench_hopai.py` is the standalone, single-size CLI version of the same
hopai-only measurement: loads the graph and times the same twelve
queries — nine traversals covering direction, multi-hop bounds, compound
chains, `OR`, `NOT`, range comparisons, and `OPTIONAL`, plus three
aggregations (`Count`/`Sum`/`Avg`/`Min`/`Max`) — cold and warm, writing
results to `bench_results.json`. `raw_cte.py` is the hand-written-CTE
query builder both `bench_hopai.py` and the notebook's driver import; run
standalone it's a library, not a script. `multiscale.py` is that
driver — it's what actually produced the notebook's numbers, generating
each tier's graph, loading it into hopai/Postgres, Apache AGE and Neo4j,
and writing one JSON file per tier to `multiscale_results/`, which the
notebook reads rather than re-running the full sweep (see the notebook's
own "Reproducing this" section for exact commands and setup).

`agg_count_4hop` deliberately runs the same chain as
`forward_bounded_4hop`: the pair shows what `graph.aggregate()` saves by
skipping edge reconstruction and node hydration on identical traversal
work — see the notebook for the current measured gap.

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

## Comparing against Apache AGE

Covered live in `benchmarks.ipynb` now, alongside Neo4j and the raw CTE,
across all three size tiers — not just whether AGE keeps up, but whether
the gap widens or narrows as the graph grows. Two findings worth knowing
before you read the notebook's numbers:

- **AGE 1.5.0 has neither `all()` nor list comprehension**, so `edge_or_tag`'s
  Neo4j form (`WHERE all(r IN relationships(p) WHERE r.tag IN [...])`)
  doesn't translate directly. `multiscale.py` rewrites it as a `UNION` of
  one `MATCH` per hop length, each naming its own relationship variables
  so every edge's `tag` can be filtered directly — correct, verified
  node-for-node against hopai's own count, but a real limitation of the
  query language, not a benchmark artifact.
- **Several bounded and backward-direction shapes don't finish within a
  90-second statement timeout**, starting at the smallest (50k-edge) tier
  already — the same behavior the original investigation behind this
  library found at a different scale. The notebook reports those as
  `TIMEOUT` rather than omitting the row.

The query shapes hopai itself generates are visible without any of this
by calling `graph.build_query(...)` and inspecting the compiled
statement, per the main README.
