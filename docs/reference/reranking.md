# Reranking

Retrieval that works is three stages, not two: retrieve wide and cheap (the
cosine `vector_search()` already does), **rerank** a bounded top-N by
actually reading each candidate against the query, then keep `k`.
`Rerank(client, document_from='<jq>', candidates=N)` adds that stage to a
flat search *and* to a traversal step (`Start`/`Hop`), where a candidate is
a node plus how it was reached — `document_from` may read `.paths` at a hop.
[`10_reranking`](../notebooks/10_reranking.ipynb) is the full, worked tour:
the reranker contract, why a dense-only ranking gets a query wrong,
`document_from=` as a jq rule evaluated once per candidate, `candidates` vs
`k`/`keep`, step-wise reranking inside a walk (probe → rerank → the
ordinary traversal with survivors pinned), the refusals below in context,
and a model-written `document_from` validated against a safe jq subset.

One method is the whole contract — `score(query: str, documents: list[str])
-> list[float]` — matched by duck typing (a Cohere/Voyage client, a
`sentence-transformers` `CrossEncoder`, or a plain callable); hopai imports
no provider package here either. `pip install "hopai[rerankers]"` brings
`jq`, not a provider, because `document_from=` needs libjq's bindings to
evaluate, not a client to call.

Seven refusals, each because reranking cannot mean something else here:

- **A raw-vector `Near` with `rerank=`.** A reranker reads a query and a
  document; a list of floats isn't readable. Use `Near(field, text=...)`.
- **`rerank=` with no `near=`.** A reranker reorders a ranked list; it
  doesn't produce one.
- **`rerank=` with no `keep=`** on a `Start`/`Hop`. With no truncation the
  reranked order is discarded and `candidates` quietly becomes the bound
  instead — a different subgraph than without `rerank=`.
- **`candidates` not above `keep`** at a step (or below `k` on a flat
  search). Reranking a pool no larger than what survives it can't change
  which rows survive.
- **`per_path=True` at a `Start`.** A seed has no route; the mode belongs
  to a `Hop`.
- **`.paths` read at a `Start`.** A seed has no provenance either.
- **A spent provider call fails loudly** as `RerankError` — never a silent
  fall back to the pre-rerank order, which is a different answer with no
  signal. The provider's own exception rides along as `__cause__`.

Over JSON the **client never travels** — a spec names only the filter and
the budget (`{"rerank": {"document_from": ..., "candidates": ...}}`); the
reranker itself comes from your side of the call as a `RerankPolicy`, whose
`fields=` allowlists what `document_from` may read and `max_candidates=`
caps what it may spend. `tool_schemas()` leaves `rerank` out of the
traversal/aggregation tool definitions unless you pass
`tool_schemas(rerank=True)`. Over MCP it's the same shape: the operator
configures `serve(rerank=..., rerank_fields=[...], max_candidates=...)`
in Python, and a model chooses only `document_from` (within the published
fields) and `candidates` (within the ceiling).

**A model may write `document_from` itself.** Field selection is a
retrieval decision a model is well placed to make, unlike an embedding it
would have to invent — so every `document_from` is validated against a
**total** jq subset (`hopai.jqsafe`) rather than restricted to trusted
callers: sound because jq has no dynamic dispatch to builtins by name
(`env`, `$ENV`, `input`, `import` don't parse at all), and terminating
because no unbounded-iteration construct parses either. There is no
`trusted=` escape on `Rerank` itself — every query is validated, whoever
wrote the filter; `trusted=True` exists only on
`Rerank.build_documents()`, the preview call an operator runs in their own
process to check a filter without spending a provider call.

See [hopai.rerankers](../api/rerankers.md) for `Rerank`/`RerankError`'s
exact signature, and [hopai.json_api](../api/json_api.md) for
`RerankPolicy`/`parse_rerank`.
