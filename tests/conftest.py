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

DSN = os.environ.get(
    "HOPAI_TEST_DSN",
    "postgresql+psycopg2://postgres:testpass@localhost:5432/ageexp",
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
