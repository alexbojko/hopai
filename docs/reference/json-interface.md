# The JSON interface

For callers that shouldn't or can't write Python — an LLM tool call, an
HTTP handler, config-driven traversal. Same keys, same meaning, same
engine as the Python API: `traverse_json`, `aggregate_json`, and
`graph.tool_schemas()` (the ready-made LLM tool definitions, narrowed to
*your* graph's node types, edge kinds and vector fields once a
[graph schema](graph-schema.md) is defined) are demonstrated in
[`04_json_and_cypher`](../notebooks/04_json_and_cypher.ipynb); selecting by
**meaning** through the same JSON — `{"near": {"field": ..., "text": ...}}`
— is in [`09_vector_search`](../notebooks/09_vector_search.ipynb).

Filters accept the same grammar, spelled as JSON operators:
`{"and": [...]}`, `{"or": [...]}`, `{"not": ...}`, `{"gt": [key, value]}`,
`{"gte": [...]}`, `{"lt": [...]}`, `{"lte": [...]}`, `{"between": [key, lo, hi]}`.

`hopai.TRAVERSE_TOOL_SCHEMA` is a ready-to-use JSON Schema for wiring this
into an LLM function-calling definition directly, alongside
`AGGREGATE_TOOL_SCHEMA`, `INGEST_TOOL_SCHEMA`, `MUTATE_TOOL_SCHEMA` and
`VECTOR_SEARCH_TOOL_SCHEMA` — that is every front end, `mutate_graph`
included; hand over the subset you actually want the model to have. See
[hopai.json_api](../api/json_api.md) for every function's exact signature.
