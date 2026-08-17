# Constraints

Postgres has always had uniqueness, composite and existence constraints —
enterprise-only in Neo4j — and a JSONB property is as constrainable as a
column once the expression is indexed: `Unique`, `Required`, `Check`,
`Index`, `PropertyType`, and `Col(...)` for a real column instead of a
property. [`05_constraints`](../notebooks/05_constraints.ipynb) is the
worked demo: every declaration, the two SQL semantics that surprise people
(a unique index doesn't constrain a *missing* property; `CHECK` runs
**before** `ON CONFLICT` resolves), and the partial-index trick with no
Neo4j equivalent (`Unique("email", where={"type": "person"})`).

One thing that page doesn't cover: every constraint here is a real
`sqlalchemy.Index`/`CheckConstraint` attached to `graph.nodes_tbl`/
`edges_tbl`, not DDL kept off to the side — so if your project points its
own Alembic `target_metadata` at those tables (or a custom
`node_table=`/`edge_table=` you pass to `Graph()`), `alembic revision
--autogenerate` sees hopai's constraints as declared schema instead of
drift to propose dropping. Call `create_schema()` and
`define_constraints(...)` (or their `_ddl` previews, which attach without
running anything) from `env.py` before autogenerate runs, the same way you
would import your own models.

`graph.constraint_ddl(...)` returns the exact SQL without running it;
`graph.drop_constraints(...)` is the inverse of `define_constraints(...)`.
See [hopai.constraints](../api/constraints.md) for every declaration's
exact signature.
