# What this doesn't do (yet)

- No disjoint multi-pattern matching (`MATCH (a)-[]->(b), (c)-[]->(d)`
  joined on shared variables) — one linear chain of hops only.
- `OPTIONAL` only on the last hop, not mid-chain.
- Aggregation covers `count`/`sum`/`avg`/`min`/`max` over the last
  step's matched nodes, numeric properties only. No grouping
  (`RETURN b.city, count(b)`), no edge-property aggregates, no
  `stddev`/percentiles, no lexicographic `min`/`max` on strings — each
  refuses with a message rather than approximating.
- Deletes and updates select rows by their properties (`where=`) or by
  where a traversal arrived is still refused: `MATCH (a)-[:knows]->(b)
  DELETE b` refuses rather than guessing which of the two readings you
  meant. Targeting a row by its `id` column is `ids=` (`delete_nodes`,
  `delete_edges`, `update_nodes`, `update_edges`, and `Start(ids=...)`
  for a traversal's seed) — `where=` still only reaches JSONB
  properties, so `where={"id": 7}` compiles a containment test that
  matches nothing; see [Filters](filters.md#addressing-a-row-by-id).
  `merge_nodes`/`merge_edges`' `on=` reaches `id` too, via
  `Col("id")` (`on=["id"]`, a bare string, refuses the same way any
  other real-column name does — see `hopai/ingest.py`); the JSON
  document form (`ingest()`'s `merge_nodes_on=`/`merge_edges_on=`, and
  the `ingest_graph` MCP tool) accepts the plain string `"id"` directly,
  since JSON has no way to spell `Col(...)`.
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
- Reranking doesn't fuse rank lists, cache, or carry its own `top_n` —
  `candidates=` bounds the input and `k`/`keep` bounds the output, on
  purpose kept from overlapping (see [Reranking](reranking.md)).
  `document_from=` is restricted to a **total** jq subset: no recursion
  (`def`), no generator (`range`), no loop (`while`/`until`/`repeat`/
  `recurse`/`..`), no fold (`reduce`/`foreach`), no re-entry
  (`label`/`break`), no module loading, and no `env`/`$ENV`/`input` —
  deliberate and structural, not a temporary gap, since libjq holds the
  GIL uninterruptibly for the length of one evaluation and `abort()`s on
  memory exhaustion rather than raising. `build_documents()` refuses
  above `MAX_DOCUMENTS` (5000) rather than growing unbounded, and
  `per_path=True` can reach that bound fast — it bills one document per
  *(node, route)*, and the route count is the graph's answer, not an
  option's. A reranked traversal's provider calls are serial by nature
  across steps — hop N+1's candidates are whatever hop N left, so there
  is nothing to parallelize until the previous step's rerank returns.
- A cycle-protection path array is carried on every recursive row. Cheap
  at moderate depth, measurably not-cheap on single-segment traversals
  past roughly 10 hops — see `benchmarks/` for the actual numbers rather
  than a guess.

