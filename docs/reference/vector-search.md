# Vector search

Exact cosine similarity over nodes and edges, on plain `real[]` columns —
no pgvector, no extension, no approximate index.
[`09_vector_search`](../notebooks/09_vector_search.ipynb) is the full,
worked tour: declaring fields (`define_vectors`/`migrate_vectors`),
storing (`set_vectors`), searching nodes and edges with `where=` filtering
applied before ranking, multivector queries and hybrid `Boost` ranking
(including the "a boost with a limit decides membership, not just
ranking" gotcha), seeding a traversal from similarity (`Start(near=,
keep=)`, `Hop(near=, keep=)`, `Hop(via_near=, via_keep=)`), composing with
aggregation, batching several queries in one round trip
(`vector_search_many`), the per-graph dimension CHECK, giving a field text
instead of floats (`embed=`, `source=`, `embed_stale()`), the two vector
refusals (a model may send `"text"`, never `"vector"`; a `Near` is not a
filter), changing the embedding model end to end, and the exact SQL a
search compiles to.

See [hopai.vectors](../api/vectors.md) for `Vector`/`Near`/`Boost`'s exact
signatures, and [hopai.embeddings](../api/embeddings.md) for `Embedder`.

## What this costs

Exact means no ANN index, so every candidate row is scored, and the cost is
a measured constant, not a guess: **0.13 µs per vector element** per
candidate row (Postgres 16, one core — `benchmarks/README.md` has the
methodology). Worked through at that rate, a candidate costs about
`dimensions × 0.13 µs`:

| rows | 384-dim | 1536-dim |
| ---: | ---: | ---: |
| 10k | ~0.5 s | ~2 s |
| 100k | ~5 s | ~20 s |
| 1M | ~50 s | ~200 s |

That's fine for the shape this is built for — a knowledge graph, filtered by
`where=` down to a manageable candidate set before ranking — but it should
never be a surprise found in production, which is why it's a number and not
a vibe. Two knobs look like they both narrow the search; they do opposite
things to that bill:

- **`where=` reduces cost.** It removes rows *before* they reach the
  ranking, the same index-backed filter every other read here uses.
  Measured: a 20k × 384-dim search filtered to 25% of rows dropped from
  ~1.0s to ~0.25s.
- **`min_similarity=` reduces results, not cost.** Every candidate is
  scanned and scored regardless; the bound only drops rows from the output
  *after* scoring — unlike an ANN index's search radius, it never skips a
  candidate. See `Near.min_similarity` in `hopai/vectors.py`.

Outgrowing this is a planned move, not a rewrite: `pgvector_exit_ddl()`
prints the migration onto pgvector whenever the numbers above say it's time.
