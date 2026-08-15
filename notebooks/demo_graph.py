"""
Shared setup for the hopai notebooks.

Every notebook owns one PostgreSQL schema outright and rebuilds it in its
first cell, so "Run All" means the same thing the second time and no
notebook depends on what another one left behind. `graph.clear()` would
empty a graph in place, but dropping the schema also resets the identity
sequences and any constraints a notebook declared, which is what makes a
re-run identical rather than merely similar.

    from demo_graph import arrows, connect, names, node_ids, seed

    graph = connect("nb_quickstart")   # empty nodes/edges tables + indexes
    seed(graph)                        # the seven-node demo graph below

The demo graph, shared by every notebook so the same names mean the same
thing throughout:

    Alice ──friend──▶ Bob ──friend──▶ Dave ──friend──▶ Erin
      └───friend──▶ Carol ──friend──▶ ┘                 │
                                                        │
      ▲──────────────────── friend ─────────────────────┘   (a real cycle)

    Bob ──works_at 2019──▶ Acme ◀──works_at 2021── Carol
    Dave ──works_at 2015──▶ Globex

Alice has no `works_at` edge (a dead end for that hop), Erin carries no
`city` key (the NOT-with-a-missing-key case), Dave is inactive, and two
parents -- Bob and Carol -- feed the same intermediate node, Dave, which
is the fan-in the traversal engine is built to preserve.

Point HOPAI_DSN at your own PostgreSQL to run the notebooks against it.
The default matches docker-compose.yml, so `docker compose up -d` at the
repository root is the whole setup.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, text

from hopai import Graph, Start

DSN = os.environ.get(
    "HOPAI_DSN",
    "postgresql+psycopg2://postgres:testpass@localhost:5432/hopai",
)

#: Flat rows: every key that is not an identity key is a property. Ids are
#: left out on purpose -- Postgres assigns them, and edges below reference
#: their endpoints by property rather than by an id nobody knows yet.
NODES = [
    {"type": "person", "name": "Alice", "age": 34, "active": True,
     "city": "Berlin", "email": "alice@example.com"},
    {"type": "person", "name": "Bob", "age": 41, "active": True,
     "city": "Berlin", "email": "bob@example.com"},
    {"type": "person", "name": "Carol", "age": 29, "active": True,
     "city": "Lisbon", "email": "carol@example.com"},
    {"type": "person", "name": "Dave", "age": 52, "active": False,
     "city": "Lisbon", "email": "dave@example.com"},
    # No `city` key at all -- not an empty string, absent. Notebook 02
    # turns on that difference.
    {"type": "person", "name": "Erin", "age": 23, "active": True,
     "email": "erin@example.com"},
    {"type": "company", "name": "Acme", "founded": 1999},
    {"type": "company", "name": "Globex", "founded": 2012},
]

EDGES = [
    {"start": {"name": "Alice"}, "end": {"name": "Bob"}, "kind": "friend"},
    {"start": {"name": "Alice"}, "end": {"name": "Carol"}, "kind": "friend"},
    {"start": {"name": "Bob"}, "end": {"name": "Dave"}, "kind": "friend"},
    {"start": {"name": "Carol"}, "end": {"name": "Dave"}, "kind": "friend"},
    {"start": {"name": "Dave"}, "end": {"name": "Erin"}, "kind": "friend"},
    {"start": {"name": "Erin"}, "end": {"name": "Alice"}, "kind": "friend"},
    {"start": {"name": "Bob"}, "end": {"name": "Acme"}, "kind": "works_at", "since": 2019},
    {"start": {"name": "Carol"}, "end": {"name": "Acme"}, "kind": "works_at", "since": 2021},
    {"start": {"name": "Dave"}, "end": {"name": "Globex"}, "kind": "works_at", "since": 2015},
]


def connect(schema: str, graph: str = "default") -> Graph:
    """A Graph on freshly created `nodes` / `edges` tables in `schema`.

    The schema is dropped and recreated, which is what makes a notebook
    re-runnable. Nothing outside `schema` is touched, so two notebooks
    never collide -- and neither does the test suite, which owns
    `hopai_test` and `hopai_write`."""
    engine = create_engine(DSN, connect_args={"options": f"-c search_path={schema}"})
    with engine.begin() as connection:
        connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    handle = Graph(engine, graph=graph)
    handle.create_schema()
    return handle


def seed(graph: Graph) -> dict:
    """Write the demo graph and return {name: id}."""
    graph.add_nodes(NODES)
    graph.add_edges(EDGES)
    return node_ids(graph)


def node_ids(graph: Graph) -> dict:
    """{name: id} for every node in the graph, since Postgres assigned them."""
    return {n["properties"]["name"]: n["id"] for n in graph.traverse(Start()).nodes}


def names(result) -> list:
    """The `name` of every node in a Subgraph, sorted -- the readable form
    of a result when the ids themselves are not the point."""
    return sorted(n["properties"].get("name", n["id"]) for n in result.nodes)


def arrows(result) -> list:
    """Every edge in a Subgraph as `Alice -friend-> Bob`, sorted.

    A traversal returns the edges it really walked, and reading them as
    arrows is how you check that -- counting nodes hides a dropped edge."""
    by_id = {n["id"]: n["properties"].get("name", n["id"]) for n in result.nodes}
    return sorted(
        f"{by_id.get(e['start_id'], e['start_id'])} "
        f"-{e['properties'].get('kind', '?')}-> "
        f"{by_id.get(e['end_id'], e['end_id'])}"
        for e in result.edges
    )
