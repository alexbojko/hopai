# Filters

A quick-reference cheat sheet for the one filter grammar every `where=`/`via=`
accepts — see [`02_traversal`](../notebooks/02_traversal.ipynb) for it used
in context, and [hopai.filters](../api/filters.md) for exact signatures.

Anywhere a `where=` or `via=` is accepted:

```python
{"type": "person"}                    # people
{"type": "person", "active": True}    # people who are ALSO active      (AND of keys)
{"type": ["person", "company"]}       # people OR companies             (IN-like)

OR({"type": "person"}, {"type": "company"})            # the same, spelled out
AND(OR({"type": "person"}, {"type": "company"}),
    {"active": True})                                  # ...and active

NOT({"type": "person"})               # everything that is not a person — INCLUDING
                                      # rows with no `type` key at all

GT("age", 18)                         # age > 18        (GTE, LT, LTE likewise)
BETWEEN("age", 18, 65)                # 18 <= age <= 65

lambda col: col.op("->>")("name").op("~")("^A")   # escape hatch: any SQLAlchemy
                                                  # expression — here, names
                                                  # starting with "A"
```

A bare list at the top level (`[{"a": 1}, {"b": 2}]`) raises `TypeError`
rather than being guessed at — it reads ambiguously as "both of these"
to a human, when it would have meant OR. Use `OR(...)` explicitly.

`NOT` is built on JSONB containment specifically because it handles a
missing property correctly (excluded from the positive filter → included
under `NOT`), unlike naive equality-based negation, which treats a
missing property as SQL `NULL` and silently drops it under `NOT` too.
Verified during development to be a real trap, not a hypothetical one —
see `tests/test_hopai.py::test_not_includes_missing_key`.

## The STORED_IN shorthand for via

`via=` additionally accepts a bare, non-empty string — shorthand for "this
edge's declared type (its `kind` property) equals this string":

```python
via="friend"                          # exactly via={"kind": "friend"}
```

Unlike every other filter above, this does **not** compile to JSONB
containment (`properties @> '{"kind": "friend"}'`) — it compiles to a text
equality (`properties ->> 'kind' = 'friend'`), which is the shape a
**declared edge type**'s functional btree index can serve. A twelve-way
`kind` vocabulary otherwise shares one general-purpose GIN index with
every other property in the table, re-tested on every hop of a recursive
walk; `Graph.define_edge_type()` gives it a narrow index of its own. See
[Schema](schema.md#declared-edge-type) for what that call does and when
it is worth it.

The shorthand compiles to the fast shape whether or not
`define_edge_type()` has run — an index only changes how fast a
(correct either way) query runs. `where=` has no equivalent: a node has
no single property this could universally name the way `kind` names an
edge's relationship type.

## Addressing a row by id

`where=`/`via=` compile against the JSONB `properties` bag — `id` is a
real column, not a property, so `where={"id": 7}` is a containment test
that matches nothing and says nothing while doing it. Naming a row you
already hold (a UI with a node selected, a traversal result fed back
in, an agent's own previous write) is a separate, deliberate parameter
instead:

```python
graph.delete_nodes(ids=[7])                       # not where={"id": 7}
graph.update_nodes(ids=[7], set={"reviewed": True})
graph.traverse(Start(ids=[7]))                     # seed a traversal by id
```

`ids=` combines with `where=`/endpoint filters as **AND** — both are
constraints on the same row, never an OR across two ways of naming one.
It works the same way on `delete_edges`/`update_edges`, against the
edge's own `id`. On the mutating side an empty `ids=[]` counts toward
the "no filter" refusal the same way an empty `where={}` does — see
[Mutations](mutations.md#naming-one-specific-row). On `Start(ids=...)`,
a read with no such danger, an empty list is instead an explicit
selection that matches nothing, exactly like an empty
`where={"key": []}` already does for a property.

`merge_nodes`/`merge_edges`' `on=` reaches `id` too, spelled
`Col("id")` (`hopai.constraints.Col`) rather than the bare string
`"id"` — a bare string always names a property, so it is refused as a
column-name collision the same way `Unique("id")` would be. The JSON
document form (`graph.ingest(doc, merge_nodes_on=[...])`, and the
`ingest_graph` MCP tool) accepts the plain string `"id"` directly and
translates it for you, since JSON has no way to spell `Col(...)`.

