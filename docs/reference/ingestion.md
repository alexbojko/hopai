# Getting data in

`add_nodes()`/`add_edges()` are demonstrated in
[`01_quickstart`](../notebooks/01_quickstart.ipynb) — this page is the rules
behind that call, not a repeat of it.

A row is written one of two ways, and the rule is one line: **a row with
a `properties` key is nested; any other row is flat, and every key that
isn't an identity key is a property.**

```python
{"id": 1, "type": "person"}                    # flat — what you write by hand
{"id": 1, "properties": {"type": "person"}}    # nested — what a traversal returns
```

The nested form is exactly `result.nodes`, so a subgraph loads into
another graph without reshaping.

Supply `id` yourself when it means something to you, or leave it out and
let Postgres assign one — but **not both in the same call**. A batch where
some rows have an id and others don't is refused, because it would insert
`NULL` for the rest instead of generating them; split it into two calls.

Edges take endpoints as `start_id`/`end_id`, or as `start`/`end`
property dicts matching one existing node each — because whatever just
wrote the nodes usually doesn't know their generated ids. References are
resolved in one batched lookup; matching nothing or several raises.

```python
graph.merge_nodes([{"email": "a@x.com", "name": "Alice"}], on=["email"])
```

`INSERT ... ON CONFLICT DO UPDATE`, needing a `Unique` on the `on` keys.
A match merges the new properties over the old ones and leaves the rest
alone (Cypher's `ON MATCH SET`); `replace=True` overwrites the bag.
Merging is idempotent, which is what makes it the right call for an
agent that might retry.

For agents and HTTP handlers, one document, and one schema to hand a
model:

```python
from hopai import INGEST_TOOL_SCHEMA

graph.ingest({
    "nodes": [{"id": 1, "type": "person"}],
    "edges": [{"start_id": 1, "end_id": 2, "kind": "knows"}],
})
```

Nodes are written before edges, so a single document can create a node
and an edge that references it. `graph.add_networkx(g)` loads a networkx
graph — the inverse of `result.to_networkx()`.

