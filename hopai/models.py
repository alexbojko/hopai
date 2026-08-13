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

If your table/column names differ, pass them to Graph() -- see core.py.
Nothing in this library requires the SQLModel classes below specifically;
they're the default, not a hard dependency of the query-building logic.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class Node(SQLModel, table=True):
    __tablename__ = "nodes"
    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    properties: dict = Field(default_factory=dict, sa_column=Column(JSONB))


class Edge(SQLModel, table=True):
    __tablename__ = "edges"
    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    start_id: int = Field(foreign_key="nodes.id")
    end_id: int = Field(foreign_key="nodes.id")
    properties: dict = Field(default_factory=dict, sa_column=Column(JSONB))
