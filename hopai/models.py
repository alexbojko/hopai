"""
hopai.models

The schema hopai expects: two tables, a typed identity column plus a
flexible JSONB properties bag on each. This split is deliberate -- real
constraints (foreign keys, NOT NULL, uniqueness) belong on real typed
columns; the properties you filter and traverse on in this library live
in JSONB, which is what lets `where=`/`via=` filter on arbitrary,
evolving attributes without a schema migration per new property.

    CREATE TABLE nodes (
        id BIGINT PRIMARY KEY,
        properties JSONB NOT NULL DEFAULT '{}'
    );
    CREATE TABLE edges (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        start_id BIGINT NOT NULL REFERENCES nodes(id),
        end_id   BIGINT NOT NULL REFERENCES nodes(id),
        properties JSONB NOT NULL DEFAULT '{}'
    );
    CREATE INDEX ON edges (start_id);
    CREATE INDEX ON edges (end_id);
    CREATE INDEX ON nodes USING GIN (properties);
    CREATE INDEX ON edges USING GIN (properties);

Vector fields add optional `vec_<name> real[]` columns BESIDE the
properties bag (via Graph.migrate_vectors() -- see vectors.py):
deliberately real columns, not JSONB keys, so the GIN index and every
traversal result stay exactly as above.

If your table/column names differ, pass them to Graph() -- see core.py.
Nothing in this library requires the Table objects below specifically;
they're the default, not a hard dependency of the query-building logic.

EXTENDING THE MODEL: a project that needs a field no JSONB property can
give it -- a foreign key to a `users` table, say -- adds an ordinary
`Column()` to its own nodes/edges Table and passes it as
`node_table=`/`edge_table=` to Graph(). Nothing further to declare:
Graph() diffs the table's columns against the ones it already knows
about (the identity column, `start_id`/`end_id`, `graph_id`,
`properties`, and any `vec_*` vector field) and treats every column
left over as an EXTRA COLUMN, exposed as `graph.node_extra_cols` /
`graph.edge_extra_cols`.

    from sqlalchemy import BigInteger, Column, ForeignKey, Identity, MetaData, Table
    from sqlalchemy.dialects.postgresql import JSONB
    md = MetaData()
    nodes = Table(
        "nodes", md,
        Column("id", BigInteger, Identity(always=False), primary_key=True),
        Column("graph_id", Text, nullable=False, server_default="default"),
        Column("properties", JSONB, nullable=False, server_default="{}"),
        Column("user_id", BigInteger, ForeignKey("users.id"), nullable=False),
    )
    graph = Graph(engine, node_table=nodes)
    graph.create_schema()   # emits user_id and its foreign key too

From there an extra column behaves like `id` or `start_id`: a flat
ingestion row addresses it by name --
`add_nodes([{"id": 1, "user_id": 7, "type": "person"}])` -- and it is
written to the real column, never folded into `properties`; a
traversal result and `to_networkx()` hand it back the same way.
`Col("user_id")` (constraints.py) is how a Unique/Index/merge `on=`
names it, exactly as it already names `start_id`/`end_id`. What this
library will NOT do for an extra column: validate it (a real NOT NULL
or FOREIGN KEY does that, same as the two built-in identity columns),
or change it once written -- `update_nodes()`/`update_edges()` remain
`properties`-only, since `set=`/`remove=` are a JSONB merge with no
equivalent for a plain column; write a new value with `UPDATE ...`
through the engine directly, or re-`merge_nodes()`/`merge_edges()` it,
which DOES refresh an extra column's value on conflict, in step with
`properties`. See ingest.py's ingestion docstring and core.py's
`traverse()` for the write and read halves.

THE ONE MISTAKE THIS GUARDS AGAINST BY NAME: declaring a `Property`
(or dataclass field, or bare-string constraint key, or merge `on=`
entry) called `user_id` without realizing the table already has a real
column by that name -- easy to do, since an extra column is a
project's own addition, not a universal convention like `id`. Every
entry point that would otherwise silently compile a JSONB rule on a
key ingestion never populates refuses instead, naming the collision:
`Graph.define_schema()`/`load_schema()` (schema.py's
`check_no_column_collisions()`), `Graph.define_constraints()`
(`Unique`/`Index`/`Required`/`PropertyType`, via
`constraints._reject_column_collision()`), and `merge_nodes()`/
`merge_edges()`'s `on=` (the same helper, since `key_sql()` is what
both an index and a merge conflict target render through). `Col(...)`
is always the escape hatch when the real column genuinely is what is
meant.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger, Column, ForeignKeyConstraint, Identity, MetaData, Table, Text,
    UniqueConstraint, text,
)
from sqlalchemy.dialects.postgresql import JSONB

#: The graph a row belongs to. One pair of tables holds every graph, and
#: every query carries `graph_id = ...` -- so a new graph costs a string,
#: not a CREATE SCHEMA, and thousands of them are ordinary rows.
DEFAULT_GRAPH = "default"

#: Node row keys that identify a row rather than name a property --
#: what ingest.py's split_row() pulls out of a flat row before whatever
#: is left becomes `properties`. Graph() also diffs a node table's
#: columns against this set: any column not in here (and not the
#: configured id/graph columns, and not a vec_* vector field) is an
#: EXTRA COLUMN -- see the "EXTENDING THE MODEL" note above.
NODE_IDENTITY_KEYS = frozenset({"id", "properties"})

#: The edge twin of NODE_IDENTITY_KEYS. "start"/"end" are the
#: node-by-property references add_edges()/merge_edges() resolve before
#: writing, which is why they can never also be an extra column's name.
EDGE_IDENTITY_KEYS = frozenset({"id", "start_id", "end_id", "start", "end", "properties"})

# GENERATED BY DEFAULT, not ALWAYS: ingestion accepts rows that carry
# their own id (re-importing a graph, or ids that mean something to the
# caller) and rows that do not (let Postgres pick). ALWAYS would reject
# the first kind. See ingest.py for the sequence resync that has to
# follow a batch of explicit ids.
_IDENTITY = Identity(always=False)

#: Owns Node/Edge (and nothing else -- callers passing their own
#: node_table/edge_table to Graph() bring their own MetaData) so
#: create_schema()/drop_schema() on the defaults never collide with a
#: caller's own declarative or Core metadata.
metadata = MetaData()

Node = Table(
    "nodes", metadata,
    Column("id", BigInteger, _IDENTITY, primary_key=True),
    Column("graph_id", Text, nullable=False, server_default=text(f"'{DEFAULT_GRAPH}'")),
    Column("properties", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    # UNIQUE(id, graph_id) is redundant on its own -- id is already the
    # primary key -- and exists solely so edges can carry a COMPOSITE
    # foreign key. That is what makes an edge between two different
    # graphs impossible rather than merely discouraged.
    UniqueConstraint("id", "graph_id", name="uq_nodes_id_graph"),
)

Edge = Table(
    "edges", metadata,
    Column("id", BigInteger, _IDENTITY, primary_key=True),
    Column("graph_id", Text, nullable=False, server_default=text(f"'{DEFAULT_GRAPH}'")),
    Column("start_id", BigInteger, nullable=False),
    Column("end_id", BigInteger, nullable=False),
    Column("properties", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    # Both endpoints are tied to the edge's OWN graph_id, so an edge can
    # only ever join two nodes of its own graph. Postgres refuses the
    # write; nothing has to remember to check.
    ForeignKeyConstraint(["start_id", "graph_id"], ["nodes.id", "nodes.graph_id"],
                         name="fk_edges_start_same_graph"),
    ForeignKeyConstraint(["end_id", "graph_id"], ["nodes.id", "nodes.graph_id"],
                         name="fk_edges_end_same_graph"),
    # UNIQUE(id, graph_id) mirrors the nodes table's own -- id alone is
    # already the primary key, so this exists solely so merge_edges(on=
    # [Col("id")]) has a matching unique index to infer an ON CONFLICT
    # target from (constraints.py's _Target.scope_index() always leads
    # with graph_id). Without it, an id-keyed re-import of edges (the
    # same natural-key workflow merge_edges() already supports for
    # properties) has no way to conflict on the row's own id.
    UniqueConstraint("id", "graph_id", name="uq_edges_id_graph"),
)
