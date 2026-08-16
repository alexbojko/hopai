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

The same thing in Cypher, and as one JSON document for a tool-calling
model (`MUTATE_TOOL_SCHEMA`):

```python
graph.cypher("MATCH (a:person) WHERE a.age > 65 SET a.retired = true")
graph.cypher("MATCH (a {email: 'a@x.com'})-[r:knows]->() DELETE r")
graph.cypher("MATCH (a {email: 'a@x.com'}) DETACH DELETE a")

graph.mutate({"operations": [                    # in order, one transaction
    {"op": "update_nodes", "where": {"type": "draft"}, "set": {"status": "archived"}},
    {"op": "delete_nodes", "where": {"type": "spam"}, "detach": True},
]})
```

