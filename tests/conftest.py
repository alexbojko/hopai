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

DSN = os.environ.get(
    "HOPAI_TEST_DSN",
    # Matches docker-compose.yml, so `docker compose up -d && pytest` works
    # with no configuration at all.
    "postgresql+psycopg2://postgres:testpass@localhost:5432/hopai",
)
SCHEMA = "hopai_test"

SETUP_SQL = f"""
DROP SCHEMA IF EXISTS {SCHEMA} CASCADE;
CREATE SCHEMA {SCHEMA};
CREATE TABLE {SCHEMA}.nodes (id BIGINT PRIMARY KEY, properties JSONB NOT NULL DEFAULT '{{}}');
CREATE TABLE {SCHEMA}.edges (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    start_id BIGINT NOT NULL REFERENCES {SCHEMA}.nodes(id),
    end_id BIGINT NOT NULL REFERENCES {SCHEMA}.nodes(id),
    properties JSONB NOT NULL DEFAULT '{{}}'
);
CREATE INDEX ON {SCHEMA}.edges (start_id);
CREATE INDEX ON {SCHEMA}.edges (end_id);
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


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(DSN, connect_args={"options": f"-c search_path={SCHEMA}"})
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


@pytest.fixture(scope="session")
def offline_graph():
    """A Graph bound to a DSN nothing listens on.

    create_engine() does not connect, and query building never executes,
    so everything up to and including the compiled SQL can be tested with
    no database at all -- which is how the suite covers query shape,
    filter compilation and the Cypher translator on any machine."""
    from hopai import Graph
    return Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline")
