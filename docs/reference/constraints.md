# Constraints

Neo4j puts uniqueness, composite and existence constraints behind an
enterprise licence. Postgres has always had them, and a JSONB property
is as constrainable as a column once the expression is indexed:

```python
from hopai import Unique, Required, Check, Index, PropertyType, Col, GT

graph.define_constraints(
    nodes=[
        Required("type"),                            # the key must be present
        Unique("email"),                             # no two nodes share one
        Unique("tenant", "slug"),                    # composite
        Unique("email", where={"type": "person"}),   # only among people
        PropertyType("age", "number"),               # not the string "42"
        Check(GT("age", 0), name="age_positive"),    # any filter, as a CHECK
        Index("type"),                               # plain lookup index
    ],
    edges=[
        Unique(Col("start_id"), Col("end_id"), "kind"),   # one edge of a kind per pair
    ],
)
```

Idempotent, so it belongs next to `create_schema()`. A violation raises
`ConstraintViolation` naming the constraint and the offending row rather
than a driver error. `graph.constraint_ddl(...)` returns the exact SQL
without running it; `graph.drop_constraints(...)` is the inverse.

Every constraint here is a real `sqlalchemy.Index`/`CheckConstraint`
attached to `graph.nodes_tbl`/`edges_tbl`, not DDL kept off to the side
-- so if your project points its own Alembic `target_metadata` at those
tables (or a custom `node_table=`/`edge_table=` you pass to `Graph()`),
`alembic revision --autogenerate` sees hopai's constraints as declared
schema instead of drift to propose dropping. Call `create_schema()` and
`define_constraints(...)` (or their `_ddl` previews, which attach
without running anything) from `env.py` before autogenerate runs, the
same way you would import your own models.

`PropertyType` is worth the line when a model writes your data: an LLM
emitting `"42"` where you expected `42` breaks every numeric comparison
downstream, silently and much later.

`where=` is the one with no Neo4j equivalent at any price — "email is
unique among people" is a partial index, and a partial index is just an
index.

Two SQL semantics to know, both of which are what you want once stated:

- A unique index doesn't constrain rows where the property is **missing**
  (`->>'email'` is NULL, and NULLs repeat). `Unique("email")` means "no
  two share an email", not "everyone has one" — pair it with
  `Required("email")` for both. Neo4j's uniqueness constraint behaves
  the same way.
- Postgres evaluates `CHECK` **before** resolving `ON CONFLICT`, so a
  merge row must satisfy every check on its own even when it is destined
  to update a row that already does.

