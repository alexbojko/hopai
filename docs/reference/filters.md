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

