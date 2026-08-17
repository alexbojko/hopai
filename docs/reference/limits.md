# What this doesn't do (yet)

- No disjoint multi-pattern matching (`MATCH (a)-[]->(b), (c)-[]->(d)`
  joined on shared variables) — one linear chain of hops only.
- `OPTIONAL` only on the last hop, not mid-chain.
- Aggregation covers `count`/`sum`/`avg`/`min`/`max` over the last
  step's matched nodes, numeric properties only. No grouping
  (`RETURN b.city, count(b)`), no edge-property aggregates, no
  `stddev`/percentiles, no lexicographic `min`/`max` on strings — each
  refuses with a message rather than approximating.
- Deletes and updates select rows by their properties, not by where a
  traversal arrived: `MATCH (a)-[:knows]->(b) DELETE b` refuses rather
  than guessing which of the two readings you meant. There is no way to
  target a row by its `id` column either — `where=` filters the JSONB
  properties, so give the rows you care about a property you can name
  (and a `Unique` on it).
- `enforce_schema(endpoints=True)` polices endpoint types with a trigger
  on the *edges* table, so retyping a node with `update_nodes` (or
  `merge_nodes(replace=True)`, which could always do it) can leave a
  declared edge connecting types the schema forbids without raising.
  Re-run `enforce_schema()` after retyping nodes.
- Vector search is exact and unindexed by design — no ANN (HNSW/IVF)
  and no late-interaction/ColBERT multivectors; `hopai/vectors.py`
  spells out the cost model and why each refusal is a refusal. Cosine
  is the only metric — on the unit-normalized vectors every current
  embedding API ships, dot and euclidean rank identically anyway.
- Embedding is a thin seam, not a framework: transient failures are
  retried with backoff and jitter, but there is no caching and no rate
  limiting. A cache belongs to the application and a rate limiter
  belongs to the client, which already has one configured the way you
  wanted it.
- A cycle-protection path array is carried on every recursive row. Cheap
  at moderate depth, measurably not-cheap on single-segment traversals
  past roughly 10 hops — see `benchmarks/` for the actual numbers rather
  than a guess.

