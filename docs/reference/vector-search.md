# Vector search

Exact cosine similarity over nodes and edges — computed by Postgres
itself, on plain `real[]` columns. No pgvector, no extension, no
approximate index, and (because it is exact) metadata filtering costs
nothing extra:

```python
from hopai import Vector, Near

graph.define_vectors(nodes=[Vector("summary", 1536), Vector("title", 384)],
                     edges=[Vector("relation", 384)],
                     migrate=True)   # ALTER TABLE, idempotent; vector_ddl() previews it
                                     # (the dimension CHECK it adds is real SA metadata too)

graph.set_vectors(nodes=[{"id": 1, "summary": embedding}])

graph.vector_search(Near("summary", query_embedding), k=10,
                    where={"type": "person"})
# [{"id": "1", "similarity": 0.93, "properties": {...}}, ...]
```

`migrate=True` runs `migrate_vectors()` for you and hands back what it
returned, so the DDL stays visible instead of happening invisibly. Skip it
(`migrate=False`, the default) and call `migrate_vectors()` yourself when
migrations run separately from application code — a deploy step with
schema-changing credentials, kept apart from the read-only process that
declares fields and calls `vector_search()`.

Declare as many fields as you need — each is one migration-managed
column, dimension-checked by the server **per graph**, so two graphs
sharing the tables can give the same field different dimensionality.

Several fields can rank one query together (multivector search), each
with a weight, an optional `min_similarity` floor, and a say over rows
missing its vector:

```python
graph.vector_search(
    Near("summary", q_summary, weight=0.7),
    Near("title",   q_title,   weight=0.3, missing="zero"),  # title optional
    k=10,
)
```

And similarity composes with traversal — seed a walk with the most
similar nodes, keep only the most similar of what a hop reaches, or
follow only the most similar **edges** out of each node:

```python
graph.traverse(
    Start(near=Near("summary", query_embedding), keep=25),  # 25 nearest seeds
    Hop(via={"kind": "cites"}, hops=(1, 3)),
)
graph.traverse(                                     # keep the nearest reached
    Start(where={"type": "paper"}),
    Hop(via={"kind": "cites"}, near=Near("summary", q), keep=10),
)
graph.traverse(                                     # a beam over edges
    Start(where={"type": "paper"}),
    Hop(via_near=Near("relation", q), via_keep=3),  # 3 nearest edges per node
)
graph.aggregate(Start(near=Near("summary", q), keep=100),
                aggregates={"avg_score": Avg("score")})
```

**Several queries at once.** A question expanded into sub-queries is
one statement, not one per query:

```python
graph.vector_search_many([Near("summary", q1), Near("summary", q2)], k=5)
# -> [[...5 hits for q1...], [...5 hits for q2...]]
```

This buys **round trips, not arithmetic** — every query still scores
every candidate, so against a local database it measured 1.08×. The
saving is `N-1` network round trips, which is the real cost when
Postgres isn't on localhost.

**Hybrid ranking.** A numeric property can contribute to the score
alongside similarity. A boost cannot push a row past a
`min_similarity` floor — but it does reorder, so with `k` it changes
which rows come back, and each result's `similarity` is then the
combined score, no longer a cosine in `[-1, 1]`:

```python
graph.vector_search(Near("summary", q), boost=Boost("importance", 0.2), k=10)
```

By default the property is rescaled into `[0, 1]` — similarity's own
scale, not its sign — with a min-max window function over the candidate rows — the ones
still in play after `where=` — before `weight` is applied, so
`Boost("importance", 0.2)` means "20% weight" whatever raw range
`importance` holds: a raw view count in the thousands would otherwise
overwhelm a cosine that never exceeds 1, and the "boost" would replace
the ranking instead of nudging it. `Boost("importance", 0.2,
scale="raw")` opts back into the unscaled coefficient — for a property
you already normalized, or when you want the per-query window function
off the query path.

Two things worth knowing about the traversal forms. A traversal returns
a **subgraph, not a ranking** — the scores and their order don't survive
into the result, so use `vector_search()` when you need them. And with
no hops, `Start(near=…, keep=N)` selects exactly what `vector_search()
` would, minus the score: `near=` on `Start` earns its place because a
traversal cannot be seeded from a list of ids.

## What this costs

Exact means no ANN index, so every candidate row is scored, and the cost is
a measured constant, not a guess: **0.13 µs per vector element** per
candidate row (Postgres 16, one core — `benchmarks/README.md`
has the methodology, including why the similarity is a LATERAL and not a
scalar subquery: the naive form re-evaluates it at every site the query
names it and was measured at 2× the cost for identical results). Worked
through at that rate, a candidate costs about `dimensions × 0.13 µs`:

| rows | 384-dim | 1536-dim |
| ---: | ---: | ---: |
| 10k | ~0.5 s | ~2 s |
| 100k | ~5 s | ~20 s |
| 1M | ~50 s | ~200 s |

That's fine for the shape this is built for — a knowledge graph, filtered
by `where=` down to a manageable candidate set before ranking — but it
should never be a surprise found in production, which is why it's a number
and not a vibe. Two knobs look like they both narrow the search; they do
opposite things to that bill:

- **`where=` reduces cost.** It removes rows *before* they reach the
  LATERAL, the same index-backed filter every other read here uses.
  Measured: a 20k × 384-dim search filtered to 25% of rows dropped from
  ~1.0s to ~0.25s.
- **`min_similarity=` reduces results, not cost.** Every candidate is
  scanned and scored regardless; the bound only drops rows from the
  output *after* scoring — unlike an ANN index's search radius, it never
  skips a candidate. See `Near.min_similarity` in `hopai/vectors.py`.

Outgrowing this is a planned move, not a rewrite: `pgvector_exit_ddl()`
(below) prints the migration onto pgvector whenever the numbers above say
it's time.

## Text in, vectors out

You do not have to produce the floats. Give a field the embedding
client you already have and hopai calls it for you — on the way in, on
the way out, and for the backfill in between:

```python
import openai
from hopai import Vector, Near

graph.define_vectors(nodes=[
    Vector("summary", 1536, source="abstract",
           embed=openai.OpenAI()),          # or cohere, voyage, google...
], migrate=True)

graph.set_vectors(nodes=[{"id": 1, "summary": "a paper about Raft"}])
graph.vector_search(Near("summary", "how do nodes agree?"), k=10)

graph.embed_stale()      # embed every row that has no vector yet
# -> {"nodes": {"summary": {"embedded": ["2", "3"], "skipped": []}}, "edges": {}}
```

`Near`'s second argument takes **either** a vector or the text to embed
into one — a string and a sequence of numbers can never be confused for
one another, so there is no keyword to remember:

```python
Near("summary", "how do nodes agree?")   # embedded by the field's client
Near("summary", [0.12, 0.44, ...])       # you already have the floats
Near("summary", text="[0.1, 0.2]")       # explicit, for a string that
                                         # looks like a serialized vector
```

A string that looks like a serialized vector (`"[0.1, 0.2]"`) is
**refused** rather than embedded, since embedding those characters
ranks against whatever the phrase means and attaches a confident score
to it. `text=` is how you say you meant it.

`source=` names the **property** holding the text and defaults to the
field's own name, so `Vector("title", 768, embed=…)` embeds each row's
`title`. `embed_stale()` reads that property for every row
`stale_vectors()` reports, embeds them, and writes them; rows whose
property is missing or blank come back under `skipped` rather than
raising, because a paper with no abstract legitimately has no abstract
vector.

It is a **backfill, not a one-shot**: one call walks the whole field in
pages of `batch` (default 1000), each its own embed call and its own
transaction, so a million rows cost bounded memory and a run that dies
partway resumes instead of restarting. That paging is a keyset cursor
rather than a `LIMIT` window on purpose — rows that can never be filled
in stay stale forever, and a window would hand back those same rows on
every pass and never reach the work behind them.

**No new dependency.** hopai imports no provider package — not even to
recognize one; clients are matched by module name and duck typing. The
extras are a convenience: `pip install "hopai[openai]"`, `[cohere]`,
`[voyageai]`, `[google]`, `[sentence-transformers]`, or `[embeddings]`
for all of them. Anything with `embed_documents`/`embed_query`
(LangChain), `get_text_embedding_batch`/`get_query_embedding`
(LlamaIndex), a `SentenceTransformer`, or a plain
`callable(texts) -> vectors` works with no extra at all. `Embedder`
wraps whatever you pass and handles the parts that are easy to get
wrong: per-provider batch caps, and the document/query asymmetry that
several providers score differently and that silently costs recall.

This is the one place hopai makes a **network call** — always to the
client you constructed and configured, and always outside the write
transaction, so a provider failure never leaves a half-written batch.

**Transient failures are retried, terminal ones are not**, and the
difference is the point. A 429 or a 503 is the provider saying "later"
and is retried with exponential backoff plus full jitter; a 401 or a
400 fails identically forever, so retrying it only burns your rate
limit to reach the same error more slowly. Which is which is decided by
the HTTP status the exception carries, or its class name when it has
none — hopai imports no provider package, so it cannot name
`RateLimitError`, but it can read a `429`. A `Retry-After` header wins
over the computed backoff, being the only number involved that isn't a
guess.

```python
Embedder(openai.OpenAI(), model="text-embedding-3-small",
         retries=2, backoff=0.5)      # the defaults: 3 attempts, 0.5s doubling
```

Your client almost certainly retries too, and **the two policies
multiply** — three attempts inside three is nine calls. Pick a side:
`Embedder(retries=0)` leaves it to the client, `openai.OpenAI(max_retries=0)`
leaves it to hopai.

When the retries are spent, `EmbeddingError` still carries the
provider's own exception as `__cause__`, for classifying more precisely
than the heuristic can:

```python
try:
    graph.embed_stale()
except EmbeddingError as failed:
    if isinstance(failed.__cause__, openai.RateLimitError):
        ...          # back off and re-run -- embed_stale() resumes
```

Every provider call is logged to the `hopai.embeddings` logger: the
size at `DEBUG`, each retry and every final failure at `WARNING` — a
retry that succeeded is not an error and does not claim to be.

**Re-embedding and the exit door.** `stale_vectors()` lists the rows
with no vector or a vector the current declaration no longer fits (the
window a dimension change opens) — the report behind `embed_stale()`,
and what you loop over yourself for fields you fill in by hand. And if
the exact scan is outgrown, `pgvector_exit_ddl()` prints the migration
onto pgvector — generated without importing or requiring the extension:

```python
for node_id in graph.stale_vectors()["nodes"]["summary"]["missing"]:
    graph.set_vectors(nodes=[{"id": node_id, "summary": embed(text_for(node_id))}])

print("\n".join(graph.pgvector_exit_ddl()))   # one-way; read vectors.py first
```

One honest limit, documented in depth in `hopai/vectors.py`: the search
is an exact scan, linear in the candidates left after filtering — see
[What this costs](#what-this-costs) above for the numbers, and why
`where=` and `min_similarity=` do opposite things to the bill.

And one rule, in one key: a model may send `"text"`, never `"vector"`.
Text is embedded by the field itself, with your client, so the query
embedding comes from the model that wrote the stored ones. A `"vector"`
asked of a model is invented, and an invented embedding finds
confidently wrong neighbors — so it is the single thing the tool
schemas never advertise, and the JSON front ends refuse it unless you
pass `allow_vectors=True` from your own code.

