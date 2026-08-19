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

THE GRAPHS REGISTRY: `graph_id` on nodes/edges is a bare string with no
row of its own recording what it is called or what it is for (issue
#85) -- `graphs(id, name, description)` below is the optional, purely
DESCRIPTIVE fix. "Purely descriptive" is the load-bearing word: `id`
holds the same value already used as `graph_id`, but there is
deliberately NO foreign key from nodes.graph_id/edges.graph_id into
it. A hard FK would need every graph_id already written -- on a
database that has been accumulating them for years -- to gain a row
here before the constraint could even be added, which is exactly the
migration/backfill this feature is not allowed to require; it would
also turn `Graph(engine, graph="anything")` into a write that can fail
for a caller who never opted into the registry at all. So a graph
with rows and no registry entry is not an error anywhere in this
library -- it is simply unnamed, the same as before this table
existed.

Sits ALONGSIDE `hopai_schema` (schema.py's save_schema()/load_schema()
table), not merged into it: that one answers "what is the declared
node/edge type contract for this graph", stored as one JSON document;
this one answers "what is this graph called and what is it for",
three flat columns meant to be listed and skimmed. Genuinely different
questions, so two tables rather than one growing a second, unrelated
job.

Unlike `hopai_schema` -- which is created lazily, by save_schema()
itself, the first time a caller saves -- this table follows the
node_table=/edge_table= PATTERN instead: a real Table object
(`GraphRegistry` below), swappable via `Graph(graph_table=...)` and
extended past id/name/description exactly the way NODE_IDENTITY_KEYS/
EDGE_IDENTITY_KEYS work for nodes/edges (`graph_extra_cols`, computed
in core.py's `Graph.__init__` the same way `node_extra_cols`/
`edge_extra_cols` are). Row creation, though, follows `hopai_schema`'s
lazy-on-first-use timing rather than nodes/edges': `create_schema()`
does not touch this table, so an existing caller who never calls
`Graph.create_graph()` sees no schema change at all. `create_graph()`
issues `CREATE TABLE IF NOT EXISTS` (via this Table's own `.create()`,
which is what makes a customized `graph_table=` bring its own extra
columns along for free) and then upserts THIS handle's graph id --
calling it again to update the name/description is the normal way to
use it, not a conflict.

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

#: The registry's own identity keys -- see "THE GRAPHS REGISTRY" above.
#: `Graph.__init__` requires all three on any `graph_table=` and treats
#: every other column as an EXTRA COLUMN, exactly as NODE_IDENTITY_KEYS/
#: EDGE_IDENTITY_KEYS do for nodes/edges.
GRAPH_IDENTITY_KEYS = frozenset({"id", "name", "description"})

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
)

#: The graph registry -- see "THE GRAPHS REGISTRY" above. `id` is text,
#: matching graph_id's type, and holds the same value a Graph(graph=...)
#: already uses; there is no foreign key from nodes/edges into it, on
#: purpose. No server_default on name/description: an unpopulated row
#: reads as NULL rather than an empty string standing in for "unnamed".
GraphRegistry = Table(
    "graphs", metadata,
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=True),
    Column("description", Text, nullable=True),
)
