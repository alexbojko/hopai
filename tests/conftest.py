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
import time

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
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


def _retry_ddl_race(setup):
    """Run a schema-(re)creation callable, tolerant of a concurrent
    creator racing the same schema/table name.

    mutmut runs many mutant subprocesses concurrently against the ONE
    Postgres the suite uses, and every one of them independently runs
    this same DROP SCHEMA / CREATE SCHEMA / CREATE TABLE sequence
    against the same schema name. Two of them can both reach CREATE
    TABLE for the same (typname, namespace) before either commits, and
    Postgres's own pg_type_typname_nsp_index -- a system catalog
    constraint, not a hopai one -- raises UniqueViolation. Under
    mutmut's `-x` stats-collection run that one transient DDL collision
    is fatal: it aborts the whole run before a single mutant is checked,
    rather than surfacing as the lock-wait timeout docs/testing.md
    already documents for the same contention on the write schemas. The
    DDL is idempotent (DROP ... IF EXISTS first), so retrying once after
    a short pause is enough for the other creator to get out of the way."""
    try:
        setup()
    except IntegrityError as exc:
        if "pg_type_typname_nsp_index" not in str(exc.orig):
            raise
        time.sleep(0.5)
        setup()


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
    def _seed():
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

    _retry_ddl_race(_seed)
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


def _vector_columns():
    """The vec_* columns currently attached to the shared Node/Edge
    tables, as {table: {name: column}}."""
    from hopai.models import Edge, Node
    from hopai.vectors import VECTOR_COLUMN_PREFIX
    return {table: {name: column for name, column in table.c.items()
                    if name.startswith(VECTOR_COLUMN_PREFIX)}
            for table in (Node, Edge)}


def _restore_vector_columns(before) -> None:
    """Put the shared Node/Edge metadata back to the vec_* attachment
    set `before` recorded -- see fresh_graph() for why that matters.

    `_columns.remove()` is SQLAlchemy-private because removing a column
    from a live Table is not a thing an application does -- but a test
    process that reuses the metadata is not an application, and there
    is no public inverse of append_column()."""
    from hopai.vectors import VECTOR_COLUMN_PREFIX
    for table, kept in before.items():
        for name, column in list(table.c.items()):
            if name.startswith(VECTOR_COLUMN_PREFIX) and name not in kept:
                table._columns.remove(column)


@pytest.fixture()
def fresh_graph(write_engine):
    """An empty graph with hopai's own schema, rebuilt for every test.

    Constraints are schema-level and outlive a TRUNCATE, so tests that
    declare them would leak into the next test. Dropping the schema is
    the only isolation that actually holds.

    THE SAME IS TRUE OF vec_* COLUMNS, one level up: define_vectors()
    attaches them to the shared SQLAlchemy Table metadata, which is
    process-global, and create_schema() above emits every column ever
    attached. Dropping the SCHEMA does not undo that -- so a field one
    test declared is silently CREATED for every later test, and a test
    that needs a declared-but-never-migrated column cannot reach that
    state at all.

    Within one pytest run this is invisible, because such tests declare
    a name nothing else uses. It stops being invisible the moment the
    suite runs TWICE IN ONE PROCESS, which is exactly what mutmut does:
    the second pass inherits the first pass's columns, four tests in
    TestSearchRefusalsNameTheRightFixLive stop raising, and mutmut's
    baseline fails -- so it aborts before checking a single mutant and
    every PR reports mutation as though nothing needed triage. That was
    live on main; see the pr_report fix that made it visible.

    So the fixture restores the attachment set it was given. It removes
    only what the test ADDED, never what it inherited, so a module that
    legitimately declares vectors at import time is untouched."""
    from hopai import Graph

    before = _vector_columns()
    graph = Graph(write_engine)

    def _rebuild():
        with write_engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {WRITE_SCHEMA} CASCADE"))
            conn.execute(text(f"CREATE SCHEMA {WRITE_SCHEMA}"))
        graph.create_schema()

    _retry_ddl_race(_rebuild)
    yield graph
    _restore_vector_columns(before)


PGVECTOR_SCHEMA = "hopai_pgvector"

#: The two tables, in the shape create_schema() would build them.
#: Raw DDL rather than Graph.create_schema() for one reason: the
#: pgvector engine's search_path has to carry the schema the `vector`
#: TYPE lives in as well as the test schema, and create_schema()'s
#: checkfirst=True finds any `nodes` reachable through that path and
#: then quietly creates nothing -- after which every unqualified
#: reference in the test would read somebody else's table.
PGVECTOR_SETUP_SQL = SETUP_SQL.replace(SCHEMA, PGVECTOR_SCHEMA)


def _pgvector_extension(eng):
    """(schema the `vector` type lives in, its version), creating the
    extension when the server has the files and this role may.

    None when pgvector is not usable here at all -- which is a skip and
    not a failure outside CI, exactly as an unreachable Postgres is."""
    with eng.connect() as conn:
        where = text("SELECT n.nspname, e.extversion FROM pg_extension e "
                     "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'vector'")
        found = conn.execute(where).first()
        if found is None:
            available = conn.execute(text(
                "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")).scalar()
            if available is None:
                return None
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            except Exception:                          # noqa: BLE001 -- reported as a skip
                conn.rollback()
                return None
            found = conn.execute(where).first()
    return None if found is None else (found[0], found[1])


@pytest.fixture(scope="session")
def pgvector_engine(engine):
    """An engine for the schema the pgvector-backend tests own.

    Skips the whole lot when the extension is missing or older than the
    backend's floor -- same courtesy (and same HOPAI_REQUIRE_DB escape
    hatch) the `engine` fixture extends for a missing database, because
    `vector_backend="pgvector"` is opt-in and a contributor on a plain
    postgres:16 should still get a useful run. CI sets HOPAI_REQUIRE_DB=1
    and runs pgvector/pgvector:pg16, so a skip there is an error.

    The search_path carries the extension's schema after the test one:
    hopai emits `vector(d)` and `<=>` unqualified (it renders SQL text
    rather than importing pgvector's Python package), so the type and
    the operator have to be reachable by name."""
    from hopai.pgvector import MINIMUM_PGVECTOR, parse_version

    probe = create_engine(DSN, poolclass=NullPool, connect_args={"gssencmode": "disable"})
    try:
        found = _pgvector_extension(probe)
    finally:
        probe.dispose()
    reason = None
    if found is None:
        reason = f"no usable pgvector extension at {DSN}"
    elif parse_version(found[1]) < MINIMUM_PGVECTOR:
        reason = (f"pgvector {found[1]} at {DSN}, and the backend needs "
                  f">= {'.'.join(str(p) for p in MINIMUM_PGVECTOR)}")
    if reason is not None:
        if os.environ.get("HOPAI_REQUIRE_DB"):
            raise RuntimeError(f"{reason} -- run the pgvector/pgvector image, or unset "
                               f"HOPAI_REQUIRE_DB to skip these tests")
        pytest.skip(f"{reason} -- run the pgvector/pgvector image, or set HOPAI_REQUIRE_DB=1 "
                    f"to make this an error")
    eng = create_engine(
        DSN, poolclass=NullPool,
        connect_args={"options": f"-c search_path={PGVECTOR_SCHEMA},{found[0]}",
                      "gssencmode": "disable"})
    yield eng
    eng.dispose()


@pytest.fixture()
def pgvector_graph(pgvector_engine):
    """An empty `vector_backend="pgvector"` graph, rebuilt per test.

    Its own schema, for fresh_graph()'s reason (migrations are DDL and
    outlive a TRUNCATE) plus one this backend adds: a vec_* column here
    is a `vector(d)`, and every graph on a table shares its columns --
    so an exact-backend test finding one would refuse rather than run.

    It restores the shared Node/Edge metadata for the same reason
    fresh_graph() does, and here it is load-bearing rather than
    hygienic: `_attach()` REFUSES a vec_* column whose declared type
    disagrees with this handle's backend, so a `vector`-typed column
    left attached would fail every later exact-backend test that
    declares that field name."""
    from hopai import Graph

    before = _vector_columns()
    graph = Graph(pgvector_engine, vector_backend="pgvector")

    def _rebuild():
        with pgvector_engine.begin() as conn:
            for stmt in PGVECTOR_SETUP_SQL.strip().split(";"):
                if stmt.strip():
                    conn.execute(text(stmt))

    _retry_ddl_race(_rebuild)
    yield graph
    _restore_vector_columns(before)


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

    def _rebuild():
        with setup_engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {ASYNC_WRITE_SCHEMA} CASCADE"))
            conn.execute(text(f"CREATE SCHEMA {ASYNC_WRITE_SCHEMA}"))
        Graph(setup_engine).create_schema()

    _retry_ddl_race(_rebuild)
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
    admin = Graph(setup_engine)

    def _rebuild():
        with setup_engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {POOL1_SCHEMA} CASCADE"))
            conn.execute(text(f"CREATE SCHEMA {POOL1_SCHEMA}"))
        admin.create_schema()
        admin.define_constraints(nodes=[Unique("email")], edges=[Unique("tag")])

    _retry_ddl_race(_rebuild)
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
