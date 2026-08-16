# Cypher as input syntax

For callers who already think in Cypher — and for a model, which has seen
far more Cypher than it has seen any Python API. `graph.cypher()` returns a
`Subgraph` for a read, an `IngestResult` for a write, a `MutationResult` for
a delete/update, and a plain `dict` for an aggregating `RETURN`;
`cypher_to_traversal`/`cypher_to_mutations`/`graph.cypher_operations` show
the translation without running anything.

[`04_json_and_cypher`](../notebooks/04_json_and_cypher.ipynb) is the full,
worked tour — reads, label-to-property mapping, `CREATE`, `MERGE` (its
unique-index requirement and its idempotency), the aggregation-translation
rules with live `CypherError` refusals, `strict_schema=True`, `SET`/`DELETE`
mutations, and `cypher_to_mutations()` as a dry run before anything is
written.

**The exhaustive rule catalog — every refusal, and why — is
[hopai.cypher](../api/cypher.md), generated straight from the module's own
docstring so it can't drift from what the parser actually accepts.** That
one page is the answer whenever a specific query is refused and the reason
isn't obvious from the notebook.
