"""
Fixtures for the hopai test suite.

Requires a reachable PostgreSQL instance. By default connects to
postgresql+psycopg2://postgres:testpass@localhost:5432/ageexp and uses
the `hopai_test` schema; override with the HOPAI_TEST_DSN env var
to point at a different database. Every test starts from a freshly
truncated, deterministically-seeded graph -- tests do not depend on
execution order or leftover state from a previous run.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

DSN = os.environ.get(
    "HOPAI_TEST_DSN",
    # Matches docker-compose.yml, so `docker compose up -d && pytest` works
    # with no configuration at all.
    "postgresql+psycopg2://postgres:testpass@localhost:5432/hopai",
)
SCHEMA = "hopai_test"

# AsyncGraph (hopai/asyncio.py) needs an async DBAPI -- psycopg2 has none.
# Same server, same credentials, just the driver swapped: psycopg3 (the
# `asyncio` extra) speaks both sync and async, so one DSN's worth of
# connection info serves both engines.
ASYNC_DSN = DSN.replace("+psycopg2", "+psycopg")

SETUP_SQL = f"""
DROP SCHEMA IF EXISTS {SCHEMA} CASCADE;
CREATE SCHEMA {SCHEMA};
CREATE TABLE {SCHEMA}.nodes (
    id BIGINT PRIMARY KEY,
    graph_id TEXT NOT NULL DEFAULT 'default',
    properties JSONB NOT NULL DEFAULT '{{}}',
    UNIQUE (id, graph_id)
);
CREATE TABLE {SCHEMA}.edges (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    graph_id TEXT NOT NULL DEFAULT 'default',
    start_id BIGINT NOT NULL,
    end_id BIGINT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{{}}',
    FOREIGN KEY (start_id, graph_id) REFERENCES {SCHEMA}.nodes(id, graph_id),
    FOREIGN KEY (end_id, graph_id) REFERENCES {SCHEMA}.nodes(id, graph_id)
);
CREATE INDEX ON {SCHEMA}.edges (graph_id, start_id);
CREATE INDEX ON {SCHEMA}.edges (graph_id, end_id);
CREATE INDEX ON {SCHEMA}.nodes USING GIN (properties);
CREATE INDEX ON {SCHEMA}.edges USING GIN (properties);
"""

# A small, hand-verifiable graph exercising every case this suite checks:
#   - a "dead end" node (n4) whose only edge is the wrong kind
#   - two parents (n1, n2) feeding the SAME intermediate (m1) -- fan-in
#   - a real cycle (h1 -> m1) for cycle-protection testing
#   - numeric `priority` values for range-comparison tests
NODES_SQL = f"""
INSERT INTO {SCHEMA}.nodes (id, properties) VALUES
  (1, '{{"type": "leaf", "name": "n1", "priority": 3}}'),
  (2, '{{"type": "leaf", "name": "n2", "priority": 7}}'),
  (3, '{{"type": "leaf", "name": "n3", "priority": 15}}'),
  (4, '{{"type": "leaf", "name": "n4-deadend"}}'),
  (5, '{{"flag": 1, "name": "m1"}}'),
  (6, '{{"flag": 1, "name": "m2"}}'),
  (7, '{{"type": "hub", "name": "h1"}}');
"""

EDGES_SQL = f"""
INSERT INTO {SCHEMA}.edges (start_id, end_id, properties) VALUES
  (1, 5, '{{"kind": "knows"}}'),
  (2, 5, '{{"kind": "knows"}}'),
  (3, 6, '{{"kind": "knows"}}'),
  (4, 6, '{{"kind": "wrong_kind"}}'),
  (5, 7, '{{"kind": "refers"}}'),
  (6, 7, '{{"kind": "refers"}}'),
  (7, 5, '{{"kind": "refers"}}');
"""


def _engine(schema: str):
    """A test engine that holds no connection between uses.

    NullPool, not the default QueuePool, because mutmut runs each mutant
    in a FORKED child. A pooled libpq connection inherited across fork is
    shared by two processes and segfaults the moment both touch it --
    which showed up as three mutants "segfault" rather than as any honest
    verdict. Nothing is pooled here, so a child always opens its own.

    gssencmode=disable for the same reason, and this one is macOS-only:
    libpq probes the Kerberos credential cache while connecting, that
    probe goes through XPC, and XPC is not fork-safe on Darwin -- the
    child dies in `pg_GSS_have_cred_cache` before it ever reaches
    Postgres. Linux has no XPC and never saw it. Nothing here
    authenticates with GSSAPI, so turning the probe off costs nothing."""
    return create_engine(DSN, poolclass=NullPool,
                         connect_args={"options": f"-c search_path={schema}",
                                       "gssencmode": "disable"})


def _async_engine(schema: str):
    """The async counterpart of _engine() -- same NullPool/gssencmode
    reasoning, psycopg3 instead of psycopg2."""
    from sqlalchemy.ext.asyncio import create_async_engine
    return create_async_engine(ASYNC_DSN, poolclass=NullPool,
                               connect_args={"options": f"-c search_path={schema}",
                                             "gssencmode": "disable"})


@pytest.fixture(scope="session")
def engine():
    eng = _engine(SCHEMA)
    try:
        eng.connect().close()
    except OperationalError as exc:
        # A large part of this suite -- the whole translation and
        # query-shape layer -- needs no database, and a contributor
        # without one should still get a useful run rather than a wall of
        # connection errors. Set HOPAI_REQUIRE_DB=1 in CI so a missing
        # database fails loudly instead of quietly skipping.
        if os.environ.get("HOPAI_REQUIRE_DB"):
            raise
        eng.dispose()
        pytest.skip(
            f"no PostgreSQL at {DSN} ({exc.orig.__class__.__name__}) -- set HOPAI_TEST_DSN, "
            f"or HOPAI_REQUIRE_DB=1 to make this an error",
            allow_module_level=True,
        )
    with eng.begin() as conn:
        for stmt in SETUP_SQL.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
        for stmt in NODES_SQL.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
        for stmt in EDGES_SQL.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    yield eng
    eng.dispose()


@pytest.fixture()
def graph(engine):
    from hopai import Graph
    return Graph(engine)


def _require_async_driver():
    """AsyncGraph needs psycopg3 (the `asyncio` extra); a contributor
    running the plain `dev` extra install from before it was added
    should get a skip naming the fix, not an ImportError with no
    context -- same courtesy the `engine` fixture extends for a missing
    database, and same HOPAI_REQUIRE_DB escape hatch so CI still fails
    loudly if the driver is ever missing there."""
    try:
        import psycopg  # noqa: F401
    except ImportError:
        if os.environ.get("HOPAI_REQUIRE_DB"):
            raise
        pytest.skip("no psycopg3 driver installed -- pip install hopai[asyncio] "
                    "(or HOPAI_REQUIRE_DB=1 to make this an error)")


@pytest.fixture()
def async_graph(engine):
    """An AsyncGraph over the SAME seeded, read-only schema `graph`
    reads -- for traverse/aggregate/vector_search tests that need no
    write isolation of their own."""
    _require_async_driver()
    from hopai.asyncio import AsyncGraph
    return AsyncGraph(_async_engine(SCHEMA))


WRITE_SCHEMA = "hopai_write"
ASYNC_WRITE_SCHEMA = "hopai_async_write"


@pytest.fixture(scope="session")
def write_engine(engine):
    """A second engine pointed at a schema the write tests own outright."""
    eng = _engine(WRITE_SCHEMA)
    yield eng
    eng.dispose()


@pytest.fixture()
def fresh_graph(write_engine):
    """An empty graph with hopai's own schema, rebuilt for every test.

    Constraints are schema-level and outlive a TRUNCATE, so tests that
    declare them would leak into the next test. Dropping the schema is
    the only isolation that actually holds."""
    from hopai import Graph

    with write_engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {WRITE_SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {WRITE_SCHEMA}"))
    graph = Graph(write_engine)
    graph.create_schema()
    return graph


@pytest.fixture()
def async_fresh_graph():
    """An empty AsyncGraph, schema owned outright, rebuilt for every
    test -- the async counterpart of fresh_graph().

    Schema setup stays on a plain sync Graph even here: AsyncGraph
    deliberately does not implement create_schema() (see the "WHAT THIS
    DOES NOT COVER" section of hopai/asyncio.py's module docstring) --
    it is a one-time admin call, not a traversal/mutation the design is
    for. The returned handle is what the test actually exercises."""
    _require_async_driver()
    from hopai import Graph
    from hopai.asyncio import AsyncGraph

    setup_engine = _engine(ASYNC_WRITE_SCHEMA)
    with setup_engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {ASYNC_WRITE_SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {ASYNC_WRITE_SCHEMA}"))
    Graph(setup_engine).create_schema()
    setup_engine.dispose()
    return AsyncGraph(_async_engine(ASYNC_WRITE_SCHEMA))


@pytest.fixture()
def async_admin_graph(async_fresh_graph):
    """A plain sync Graph on the SAME schema async_fresh_graph owns --
    the documented escape hatch for the admin/schema-DDL calls
    AsyncGraph deliberately does not implement (define_constraints(),
    create_schema(), ...). Depends on async_fresh_graph purely for
    ordering (so the schema already exists); it adds nothing schema-wise
    of its own."""
    from hopai import Graph
    return Graph(_engine(ASYNC_WRITE_SCHEMA))


POOL1_SCHEMA = "hopai_async_pool1"


@pytest.fixture()
def async_fresh_graph_pool1():
    """Same shape as async_fresh_graph, but the async engine's pool is
    capped at exactly ONE connection (pool_size=1, max_overflow=0, a
    short pool_timeout) -- NullPool, which every other async fixture
    uses, has no capacity limit at all and cannot tell "opened a second
    connection" from "reused the first".

    This is the general test for "does an AsyncGraph write method
    actually pass connection=c through to the sync Mutator/Ingestor/
    vectors function it wraps". Wired correctly, a call only ever needs
    the ONE connection AsyncGraph already checked out via
    engine.begin()/.connect(). If connection= is silently dropped (as
    several mutation-testing survivors on hopai/asyncio.py turned out
    to be), the sync function opens its own transaction, which means
    checking out a SECOND connection while the first is still held --
    on a one-slot pool, that blocks until pool_timeout and raises,
    rather than quietly costing an extra round trip the way it would on
    an unbounded pool.

    Also declares Unique(email) on nodes and Unique(tag) on edges
    up front, so tests that exercise merge_nodes()/merge_edges() need
    no per-test admin ceremony of their own."""
    _require_async_driver()
    from hopai import Graph, Unique
    from hopai.asyncio import AsyncGraph
    from sqlalchemy.ext.asyncio import create_async_engine

    setup_engine = _engine(POOL1_SCHEMA)
    with setup_engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {POOL1_SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {POOL1_SCHEMA}"))
    admin = Graph(setup_engine)
    admin.create_schema()
    admin.define_constraints(nodes=[Unique("email")], edges=[Unique("tag")])
    setup_engine.dispose()

    async_engine = create_async_engine(
        ASYNC_DSN, pool_size=1, max_overflow=0, pool_timeout=2,
        connect_args={"options": f"-c search_path={POOL1_SCHEMA}", "gssencmode": "disable"})
    return AsyncGraph(async_engine)


@pytest.fixture(scope="session")
def offline_graph():
    """A Graph bound to a DSN nothing listens on.

    create_engine() does not connect, and query building never executes,
    so everything up to and including the compiled SQL can be tested with
    no database at all -- which is how the suite covers query shape,
    filter compilation and the Cypher translator on any machine."""
    from hopai import Graph
    return Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline")
