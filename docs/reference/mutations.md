# Changing and deleting

The other half of a graph an agent maintains rather than only fills.
`where=` is the same filter language a traversal uses, and it selects a
**set** of rows — every row it matches is changed, exactly as Cypher's
`SET` and `DELETE` do.

```python
# "Everyone over 65 is retired."
graph.update_nodes(where=GT("age", 65), set={"retired": True})

# `set` merges over what's there; `remove` drops keys; `replace=True`
# makes `set` the whole property bag.
graph.update_nodes(where={"email": "a@x.com"}, remove=["nickname"])

# "Alice doesn't know Bob any more." Endpoint filters, so you never have
# to look an id up first.
graph.delete_edges(where={"kind": "knows"},
                   start={"email": "a@x.com"}, end={"email": "b@x.com"})

# "Forget Alice." detach=True deletes her edges with her.
graph.delete_nodes(where={"email": "a@x.com"}, detach=True)

graph.clear()          # this graph, and no other, in one transaction
```

Every call returns a `MutationResult` — `deleted_nodes`,
`deleted_edges`, `updated_nodes`, `updated_edges`, `elapsed_ms` — four
counters because one delete touches both tables.

`start`/`end` are filters, not references: any number of nodes may match
one, and every edge touching any of them goes. That is the opposite of
`add_edges()`, where `start`/`end` must identify exactly one node and
ambiguity raises.

**A call with no filter raises rather than matching everything.**
`where=None` and `where={}` are what an empty variable looks like, and
the cost of being wrong here is the data. Say it on purpose with
`all=True`, or call `clear()`. `all`, `detach` and `replace` must be
real booleans — `all="false"` raises rather than being read as truthy,
because JSON booleans arriving as strings is an ordinary tool-call
failure and this one would empty the graph.

**Deleting a node that still has edges fails**, and the error names
`detach=True` — the composite foreign key is doing its job, and an edge
pointing at a node that no longer exists is exactly the corruption it
was added to prevent.

## Naming one specific row

`where=` filters properties, and **an id is not a property** — it is a
column. `where={"id": 7}` is a containment test against the JSONB bag,
so it matches nothing and says nothing about why. `ids=` is how a caller
already holding a row names it, on every mutating call — `delete_nodes`,
`delete_edges`, `update_nodes`, `update_edges`:

```python
graph.delete_nodes(ids=[12, 13], detach=True)
graph.delete_edges(ids=[7])
graph.update_nodes(ids=[12], set={"reviewed": True})
```

`Start(ids=...)` is the read-side counterpart, seeding a traversal from
specific ids instead of (or alongside) `where=` — see
[Traversal](traversal.md).

An empty list refuses exactly as an empty filter does — it is what an
empty selection looks like, and `all=True` stays the only opt-in. Given
both, `ids=` and `where=` narrow **together**: a union would delete rows
the caller named neither way.

Moving an edge is not an update — `set=` writes properties, and an
edge's endpoints are real columns:

```python
graph.repoint_edge(7, end_id=3)      # one endpoint; the other stays
```

One `UPDATE`, so the edge keeps its id and its properties and there is no
window where it does not exist. Nothing checks that the new endpoint
belongs to this graph, because the composite foreign key already does:
Postgres refuses the write rather than this code remembering to look.

The same operations translate from Cypher (`SET`/`REMOVE`/`DELETE`/`DETACH
DELETE`) and compile from one JSON document, `{"operations": [...]}`
(`MUTATE_TOOL_SCHEMA` is the ready-made tool definition), run in order as one
transaction — demonstrated in
["Changing and deleting, in the same three notations"](../notebooks/04_json_and_cypher.ipynb).

