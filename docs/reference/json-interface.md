# The JSON interface

For callers that shouldn't or can't write Python — an LLM tool call, an
HTTP handler, config-driven traversal:

```python
from hopai import traverse_json

# The same question as the Quick start, in JSON: "which companies do
# Alice's friends — up to four hops out, active only — work for?"
traverse_json(graph, {
    "start": {"where": {"name": "Alice"}},
    "hops": [
        {"via": {"kind": "friend"}, "hops": [1, 4], "where": {"active": True}},
        {"via": {"kind": "works_at"}, "where": {"type": "company"}},
    ],
})
```

Same keys, same meaning, same engine — JSON in, JSON out.

Filters accept the same grammar, spelled as JSON operators:
`{"and": [...]}`, `{"or": [...]}`, `{"not": ...}`, `{"gt": [key, value]}`,
`{"gte": [...]}`, `{"lt": [...]}`, `{"lte": [...]}`, `{"between": [key, lo, hi]}`.

A traversal can also select by **meaning**, with the same `near` a
Python caller writes — a model sends the words, and the field embeds
them with the client your application declared:

```python
traverse_json(graph, {
    "start": {"near": {"field": "summary", "text": "distributed consensus"},
              "keep": 25},
    "hops": [{"via": {"kind": "cites"}, "hops": [1, 3]}],
})
```

`hopai.TRAVERSE_TOOL_SCHEMA` is a ready-to-use JSON Schema for wiring
this into an LLM function-calling definition directly, alongside
`AGGREGATE_TOOL_SCHEMA`, `INGEST_TOOL_SCHEMA`, `MUTATE_TOOL_SCHEMA` and
`VECTOR_SEARCH_TOOL_SCHEMA` — and with a
[graph schema](graph-schema.md) defined, `graph.tool_schemas()` returns
the four traversal/write definitions with *your* node types, edge kinds
and properties summarized into the descriptions, so the model stops
hallucinating labels — plus `VECTOR_SEARCH_TOOL_SCHEMA`, its `field`
narrowed to an enum of *your* declared vector fields, whenever
`define_vectors()` has declared any:

```python
tools = graph.tool_schemas()   # traverse / aggregate / ingest / mutate,
                               # plus vector search if this graph
                               # declares any vector fields
```

That is every front end, `mutate_graph` included — hand over the subset
you actually want the model to have.

