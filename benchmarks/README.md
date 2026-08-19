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

## The pgvector backend: what the index buys, and what it costs

```bash
python bench_pgvector.py --dsn "postgresql+psycopg2://user:pass@host/db" \
    --rows 20000,100000 --dims 384
```

`Graph(dsn, vector_backend="pgvector")` (opt-in, needs the extension —
see `hopai/pgvector.py`) stores vectors in `vector(d)` columns behind an
HNSW cosine index and compiles a single-`Near` search to
`ORDER BY vec_x <=> :q LIMIT k`, which that index can serve.
`bench_pgvector.py` measures it against the default exact backend and
reports **recall beside every latency**. That pairing is the point of
the file: the index answers approximately, and a speedup quoted without
the recall it bought is exactly the number "refuse, don't approximate"
exists to keep out of this repository.

One dataset per configuration, stored twice — the same generated
vectors go into a `real[]` column for the exact backend and a
`vector(d)` column for the pgvector one, **on the same rows** — so both
sides rank identical data and the exact backend's top-k is usable as
ground truth. Only single-`Near` searches are measured because they are
the only ones this backend serves; multivector and `boost=` are refused
under it by name.

### What each number means

- **`exact_*_ms` / `pgvector_*_ms`** — warm mean of `--repeats` whole
  `vector_search()` calls, hydration included, the same way
  `bench_vectors.py` times.
- **`exact_us_per_element`** — `bench_vectors.py`'s transferable number,
  recomputed here so the exact column can be checked against that file.
  There is deliberately no pgvector twin: an index search is sublinear
  in rows, so a per-element figure for it would fall as the table grows
  and mean nothing.
- **`recall@k`** — overlap of returned ids with the exact backend's
  top-k, meaned over `--queries` query vectors; `min`, `perfect_share`
  and `top1_share` come with it, because a mean hides whether one query
  lost everything or every query lost one.
- **`similarity_ratio`** — mean cosine of what pgvector returned over
  mean cosine of the true top-k. Not a softer restatement of recall:
  recall counts **ids**, and two neighbors at 0.404 and 0.403 are a
  recall miss and a difference of nothing.
- **`index_build_seconds` / `index_mb`** — the HNSW index rebuilt over
  loaded data, not the free build over an empty column that
  `migrate_vectors()` performs.
- **`set_vectors_*_seconds`** — the same vectors written under each
  backend; the pgvector column maintains its index on every write.
- **`fewer_than_k_filter`** — pass/fail, never a timing.
- **`pgvector_uses_index`** — the scan node `EXPLAIN` actually chose.

### Measured

Postgres 16.14, pgvector 0.8.0, **a shared 4-vCPU container** —
`shared_buffers` 128MB, `work_mem` 4MB, `maintenance_work_mem` 64MB,
HNSW at its defaults (`m=16`, `ef_construction=64`, `ef_search=40`),
k=10, 20 query vectors, 5 warm repeats. Absolute milliseconds on a box
like this are noisy and do not transfer; the exact-vs-pgvector ratio at
a given size roughly does. Uniform random vectors, the same dataset
shape `bench_vectors.py` generates.

| rows × dims | exact | pgvector | | recall@10 | sim ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 000 × 384 | 127.3 ms | 4.7 ms | **27×** | 0.74 | 0.97 |
| 20 000 × 384 | 1057.0 ms | 5.8 ms | **183×** | 0.21 | 0.88 |
| 100 000 × 384 | 5010.6 ms | 6.5 ms | **770×** | 0.05 | 0.81 |
| 20 000 × 768 | 1841.4 ms | 6.9 ms | **268×** | 0.12 | 0.84 |

The same searches `where=`-filtered to ~25% of rows:

| rows × dims | exact | pgvector | | recall@10 | sim ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 000 × 384 | 37.6 ms | 4.6 ms | **8×** | 0.70 | 0.95 |
| 20 000 × 384 | 301.0 ms | 4.6 ms | **65×** | 0.22 | 0.83 |
| 100 000 × 384 | 1444.7 ms | 12.5 ms | **115×** | 0.10 | 0.77 |
| 20 000 × 768 | 481.2 ms | 33.3 ms | **14×** | 1.00 | 1.00 |

That last row is not a typo and is worth more than the three above it:
at 768 dims the planner **did not use the HNSW index** for the filtered
search (`EXPLAIN` chose a bitmap scan on the properties GIN index and
sorted), so the answer came back exact — recall 1.00 — and still 14×
faster than the exact backend, because pgvector's `<=>` over a
`vector(d)` column is far cheaper than `unnest`+`sum` over a `real[]`
one even with no index involved. Which of the two you get is the
planner's decision, not the library's, which is why
`pgvector_uses_index` is recorded per configuration rather than assumed.

The write and build costs, which belong in the same accounting:

| rows × dims | index build | index size | `set_vectors` exact | `set_vectors` pgvector |
| --- | ---: | ---: | ---: | ---: |
| 2 000 × 384 | 0.7 s | 3.9 MB | 2.5 s | 5.2 s |
| 20 000 × 384 | 5.0 s | 39.1 MB | 29.7 s | 84.3 s |
| 100 000 × 384 | 158.1 s | 195.3 MB | 129.5 s | 612.7 s |

Writes cost **2–5× more** under this backend — every `set_vectors()`
maintains the graph — and the 100k index build is an upper bound rather
than a fair figure: 100k × 384 float4 is ~150MB of vectors against
`maintenance_work_mem` of 64MB, so it built on disk. Raising that
setting is the first thing to try before quoting this number as the
cost of the index.

**`fewer_than_k_filter` passed in every configuration above**: a filter
matching 2 rows with k=10 returned both rows, the same two ids, under
both backends. That is not the index being complete — an HNSW scan
applies `where=` to the candidates its walk reaches and can come back
short of the rows that match, which `hnsw.iterative_scan` reduces and
does not close. It is hopai **completing** a short filtered result with
a second exact query (`vectors._pgvector_needs_completion()`), so the
line below is the price of that guarantee, next to the same search when
the filter is wide enough to fill k:

### Recall < 1.0 is the whole trade, and 0.21 is what the defaults gave

The speedups above are real and the recall beside them is real. On this
dataset, at the settings hopai actually runs, **a 20k-row 384-dim search
returned 2 of the 10 true nearest neighbors** and the exact backend's
own top hit was in the result only 15% of the time. Three things move
that number, all measured at 20 000 × 384:

| dataset / setting | pgvector | recall@10 | sim ratio | top-1 found |
| --- | ---: | ---: | ---: | ---: |
| uniform, `ef_search=40` (the default) | 5.8 ms | 0.21 | 0.88 | 15% |
| clustered (`--clusters 64`), `ef_search=40` | 4.1 ms | 0.39 | 0.89 | 40% |
| uniform, `ef_search=200` | 8.9 ms | 0.61 | 0.97 | 85% |
| clustered, `ef_search=200` | 4.1 ms | 0.67 | 0.97 | 70% |

- **The dataset is a floor, not a forecast.** Uniform random vectors in
  high dimension are near-orthogonal: every candidate sits at almost the
  same distance, so there is little structure for HNSW to navigate and
  the true top-10 is a near-tie. Real embeddings are clustered, which
  `--clusters N` imitates and which nearly doubled recall here. Read the
  uniform rows as a worst case and measure your own corpus before
  trusting either.
- **`similarity_ratio` says how much of that miss matters.** At
  `ef_search=200` recall is 0.61 while the ratio is 0.97 — mostly
  *different* neighbors that are *nearly as close*, which for retrieval
  feeding an LLM is a different situation from the 0.81 ratio the 100k
  row shows, where the index genuinely returned worse rows.
- **`ef_search` is the dial, and hopai does not expose it.** Tripling it
  bought 3× the recall for 1.5× the latency and still ran 117× faster
  than the exact backend. hopai sets `hnsw.iterative_scan` (correctness)
  and leaves `ef_search` at the server's 40, so a caller who needs more
  recall has to set it themselves on the connection — which is what
  `--ef-search` in this script does, and is worth knowing before
  adopting the backend.

### When it is worth it

There is **no latency crossover to wait for**. The exact scan costs
~0.13 µs per element (`rows × dims`), the indexed search costs a roughly
flat 4–13 ms, so they meet at about 40 000 elements — around a hundred
rows at 384 dims. Above any size worth indexing, the index wins on
latency, and by 100k rows it is not close.

So the decision is never "is it faster". It is:

- **Can you afford an approximate answer at all?** If a missing
  neighbor is a wrong answer rather than a slightly worse one, stay on
  the exact backend — it is not merely correct, it is the only one of
  the two that can say so.
- **Are you past the exact backend's comfortable range?** A 5-second
  unfiltered search at 100k × 384 is not a live query path, and
  `where=` only helps if your filters are selective (25% still cost
  1.4 s there). That is the situation this backend exists for.
- **Do you need multivector or `boost=`?** They are refused here, and
  the exact backend answers them correctly with no extension.
- **Are you willing to tune?** The out-of-the-box recall measured above
  is poor. Budget for measuring your own corpus and raising
  `ef_search` — the backend's speed is free, its quality is not.

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

## Regression check: the traversal path across a batch of features

`bench_hopai.py`'s twelve queries exist to answer one question when a
batch of work lands: did any of it slow the walk down. Run once against
the commit a batch started from and once against the commit it ended at,
same graph, same machine, and the queries that never touch the new
feature should not move.

Recorded for the batch adding `ids=` selection, grouped aggregation,
declared edge types, the graphs registry, and the rerank-probe
projection (`338402b` → `cfea68b`). 210k nodes / 170k edges, five runs
per version, medians of the warm timings:

| query | before (ms) | after (ms) | |
| --- | ---: | ---: | ---: |
| `forward_1hop` | 56.1 | 57.7 | +2.9% |
| `backward_1hop` | 56.4 | 55.0 | -2.5% |
| `forward_bounded_4hop` | 528.2 | 525.8 | -0.5% |
| `backward_bounded_3hop` | 60.5 | 61.1 | +1.0% |
| `compound_2segment` | 553.5 | 545.9 | -1.4% |
| `edge_or_tag` | 434.1 | 425.1 | -2.1% |
| `not_filter` | 61.0 | 59.2 | -3.0% |
| `range_gt` | 80.2 | 76.2 | -5.0% |
| `optional_last_hop` | 56.2 | 53.1 | -5.5% |
| `agg_count_4hop` | 197.3 | 190.6 | -3.4% |
| `agg_stats_backward_4hop` | 64.1 | 64.7 | +0.9% |
| `agg_range_count` | 27.6 | 25.0 | -9.4% |
| **total** | **2175** | **2139** | **-1.6%** |

Every query returned identical rows on both versions — the timings are
only worth reading because the answers did not change.

The spread is noise in both directions and the batch is not responsible
for either end of it: **the compiled SQL for these shapes is
byte-identical across the two commits**. That is the check worth
running, and the one to run first — timings on a shared machine move by
a few percent on their own, so a table like the one above can only ever
fail to show a regression, while a diff of the emitted statement either
is empty or is not. It needs no database:

```python
from sqlalchemy.dialects import postgresql
from hopai import Graph, Start, Hop
g = Graph("postgresql+psycopg2://u:p@localhost/db")   # never connects
print(g.build_query(Start(where={"type": "node"}),
                    [Hop(hops=(1, 4))]).compile(dialect=postgresql.dialect()))
```

Compiled that way on both commits across eight shapes — 1-hop each
direction, bounded, compound, `OR`, `NOT`, range, and `optional` — the
output differs nowhere. `ids=` adds a predicate only when a caller
passes it, `group_by=` only reaches `build_aggregate_query`, and a
declared edge type changes which index Postgres can use, not the
statement. This is the "no performance regression" rule in CLAUDE.md
being a measurement rather than an assurance.

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
