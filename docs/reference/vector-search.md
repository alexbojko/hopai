# Vector search

Exact cosine similarity over nodes and edges, on plain `real[]` columns —
no pgvector, no extension, no approximate index. That is the default and
what the rest of this page describes; [an optional pgvector
backend](#the-pgvector-backend) trades it for an index once the numbers
below say it is time.
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

Outgrowing this is a planned move, not a rewrite — see below.

## The pgvector backend

`Graph(dsn, vector_backend="pgvector")` stores vectors in pgvector's
`vector(d)` type behind an HNSW cosine index, and compiles a search to
`ORDER BY vec_x <=> :query LIMIT k` so the index answers it. Everything
else — `where=`, traversal seeding, `set_vectors()`, results — keeps its
shape. The default is `vector_backend="exact"`, and a Graph that never
asks for pgvector emits byte-identical SQL to the pre-pgvector engine.

```python
graph = Graph(dsn, vector_backend="pgvector")
graph.define_vectors(nodes=[Vector("summary", 1536)])
graph.migrate_vectors()      # vector(1536) + HNSW, and the extension
```

It is opt-in because it is a real dependency: the `vector` extension
must be installed in the server. No Python package is added — the type
is rendered as SQL text and the operator as `<=>`.

**What you give up.** An HNSW index answers *approximately*: it can miss
a true nearest neighbor, and no setting makes that absolute. That is the
whole trade, and it is the reason this is not the default.

**What you keep: `where=` still means `where=`.** That one is not free,
and it is the most important thing on this page.

A filtered HNSW scan can return *fewer rows than match the filter* —
not ranked differently, but rows you asked for, missing, with nothing to
indicate it. Measured:

| rows | rows matching `where=` | `k` | setting | returned |
| ---: | ---: | ---: | --- | ---: |
| 20,000 | 2 | 10 | `iterative_scan=off` (pgvector < 0.8) | **1** |
| 20,000 | 2 | 10 | `strict_order` (≥ 0.8) | 2 |
| 120,000 | 3 | 10 | `strict_order`, `max_scan_tuples` raised past the table | **0** |

So pgvector 0.8's `hnsw.iterative_scan` *reduces* the problem and does
not remove it — the graph traversal runs out of reachable candidates
long before a selective filter is satisfied, and no SQL-level setting
changes that. hopai therefore does two things: it requires **pgvector ≥
0.8** (refusing an older server by name) and sets `strict_order`, *and*
it **completes a short filtered result exactly** — re-asking with an
ordering the index cannot serve, which is the exact backend's cost and
the exact backend's answer.

That second query runs only when a filtered search comes back short of
`k`, so the ordinary path keeps the index's speed and the awkward path
keeps the right answer. What stays approximate is the *ranking* of an
unfiltered search — that is what an ANN index buys, and it cannot be
given back.

**One field per search.** Multivector (`Near` on several fields) and
`Boost` are refused under this backend rather than served. An HNSW index
accelerates exactly one ordering — the one it was built for — and a
weighted sum across two independent distance spaces is not it, so
Postgres would scan every row and sort the sum: the exact backend's cost,
having silently become approximate. Rank one field here, or use the exact
backend, which answers both correctly and needs no extension. A negative
`Near` weight is refused for the same reason (it asks for the *least*
similar rows, and HNSW indexes one direction).

**Per Graph, not per field.** `CREATE EXTENSION vector` is a property of
the database, and `vector(d)` fixes the dimensions in the column's type —
which every graph in those tables shares. So a field cannot be 1536-dim
in one graph and 768-dim in another under this backend, the way a
per-graph CHECK allows under the exact one; `migrate_vectors()` refuses
that by name.

`pgvector_exit_ddl()` remains, and is now the *other* door: it prints the
same migration and leaves the querying to you, for the cases this backend
refuses.
