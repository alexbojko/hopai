"""
Test suite for extending the model: custom nodes/edges tables carrying
extra typed columns beyond `id`/`start_id`/`end_id`/`graph_id`/
`properties` -- a foreign key to a `users` table is the motivating case
(see models.py's "EXTENDING THE MODEL" note). Graph() discovers such
columns from the table itself; this file exercises what that discovery
then lets add_nodes()/merge_nodes()/add_edges()/merge_edges()/
traverse()/to_networkx()/add_networkx() do with them.
"""

from __future__ import annotations

import pytest
from sqlalchemy import (
    BigInteger, Column, ForeignKey, ForeignKeyConstraint, Identity, MetaData, Table, Text,
    UniqueConstraint, text,
)
from sqlalchemy.dialects.postgresql import JSONB

from hopai import ConstraintViolation, Graph, Hop, Start, Unique, Vector
from hopai.constraints import Col

WRITE_SCHEMA = "hopai_write"


@pytest.fixture()
def extra_graph(write_engine):
    """A Graph over custom nodes/edges tables, each carrying one EXTRA
    COLUMN beyond hopai's own shape: `user_id` on nodes, a required real
    foreign key into a `users` table -- the exact case this feature
    exists for -- and `note`, an optional plain column on edges, so both
    a required and an optional extra column are covered."""
    with write_engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {WRITE_SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {WRITE_SCHEMA}"))
        conn.execute(text("CREATE TABLE users (id BIGINT PRIMARY KEY, name TEXT)"))
        conn.execute(text("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')"))

    md = MetaData()
    # A Table object purely so ForeignKey("users.id") below can resolve
    # at DDL-compile time -- the real `users` table was already created
    # by raw SQL above; create_schema() only ever emits nodes/edges.
    Table("users", md, Column("id", BigInteger, primary_key=True))
    nodes = Table(
        "nodes", md,
        Column("id", BigInteger, Identity(always=False), primary_key=True),
        Column("graph_id", Text, nullable=False, server_default=text("'default'")),
        Column("user_id", BigInteger, ForeignKey("users.id"), nullable=False),
        Column("properties", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        UniqueConstraint("id", "graph_id"),
    )
    edges = Table(
        "edges", md,
        Column("id", BigInteger, Identity(always=False), primary_key=True),
        Column("graph_id", Text, nullable=False, server_default=text("'default'")),
        Column("start_id", BigInteger, nullable=False),
        Column("end_id", BigInteger, nullable=False),
        Column("note", Text),
        Column("properties", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        ForeignKeyConstraint(["start_id", "graph_id"], ["nodes.id", "nodes.graph_id"]),
        ForeignKeyConstraint(["end_id", "graph_id"], ["nodes.id", "nodes.graph_id"]),
    )
    graph = Graph(write_engine, node_table=nodes, edge_table=edges)
    graph.create_schema()
    return graph


class TestDiscovery:
    def test_extra_columns_are_found_automatically(self, extra_graph):
        assert extra_graph.node_extra_cols == ("user_id",)
        assert extra_graph.edge_extra_cols == ("note",)

    def test_default_tables_have_none(self, fresh_graph):
        assert fresh_graph.node_extra_cols == ()
        assert fresh_graph.edge_extra_cols == ()

    def test_vector_columns_never_become_extra_columns(self, extra_graph):
        """define_vectors()/migrate_vectors() attach vec_* columns to
        the SAME shared Table object -- a second handle built after that
        must still not see vec_summary as an extra column, even though
        it is by then an ordinary column sitting on the table."""
        extra_graph.define_vectors(nodes=[Vector("summary", 4)])
        extra_graph.migrate_vectors()
        second = Graph(extra_graph.engine, node_table=extra_graph.nodes_tbl,
                       edge_table=extra_graph.edges_tbl)
        assert second.node_extra_cols == ("user_id",)
        assert "vec_summary" not in second.node_extra_cols


class TestWrite:
    def test_add_nodes_writes_the_real_column_not_properties(self, extra_graph):
        extra_graph.add_nodes([{"id": 1, "user_id": 1, "type": "task"}])
        with extra_graph.engine.connect() as conn:
            row = conn.execute(text("SELECT user_id, properties FROM nodes WHERE id = 1")).one()
        assert row.user_id == 1
        assert row.properties == {"type": "task"}

    def test_nested_form_accepts_the_extra_column_too(self, extra_graph):
        """Before this feature, an extra-column key beside `properties`
        in nested form would raise "mixes both spellings" -- it is now
        an identity key, exactly like `id`."""
        extra_graph.add_nodes([{"id": 1, "user_id": 2, "properties": {"type": "task"}}])
        node = extra_graph.traverse(Start()).nodes[0]
        assert node["user_id"] == 2

    def test_missing_required_extra_column_is_refused_by_the_database(self, extra_graph):
        with pytest.raises(ConstraintViolation):
            extra_graph.add_nodes([{"id": 1, "type": "task"}])

    def test_the_real_foreign_key_is_enforced(self, extra_graph):
        with pytest.raises(ConstraintViolation):
            extra_graph.add_nodes([{"id": 1, "user_id": 999, "type": "task"}])

    def test_a_batch_may_not_half_specify_an_extra_column(self, extra_graph):
        """Same executemany trap _require_uniform guards for `id`,
        generalized: a mixed batch would bind NULL for the rows that
        omitted user_id rather than leaving them to fail loudly."""
        with pytest.raises(ValueError, match="'user_id'"):
            extra_graph.add_nodes([{"id": 1, "user_id": 1}, {"id": 2, "user_id": 2}] +
                                  [{"id": 3}])

    def test_edges_route_their_extra_column_too(self, extra_graph):
        extra_graph.add_nodes([{"id": 1, "user_id": 1}, {"id": 2, "user_id": 1}])
        extra_graph.add_edges([{"start_id": 1, "end_id": 2, "note": "hi", "kind": "knows"}])
        edge = extra_graph.traverse(Start(), Hop()).edges[0]
        assert edge["note"] == "hi"
        assert edge["properties"] == {"kind": "knows"}

    def test_an_optional_extra_column_may_be_omitted_entirely(self, extra_graph):
        extra_graph.add_nodes([{"id": 1, "user_id": 1}, {"id": 2, "user_id": 1}])
        extra_graph.add_edges([{"start_id": 1, "end_id": 2}])
        edge = extra_graph.traverse(Start(), Hop()).edges[0]
        assert edge["note"] is None


class TestRead:
    def test_traverse_returns_the_extra_column(self, extra_graph):
        extra_graph.add_nodes([{"id": 1, "user_id": 1, "type": "task"}])
        node = extra_graph.traverse(Start()).nodes[0]
        assert node == {"id": "1", "properties": {"type": "task"}, "user_id": 1}

    def test_to_networkx_carries_extra_columns_as_attributes(self, extra_graph):
        extra_graph.add_nodes([{"id": 1, "user_id": 1, "type": "task"},
                               {"id": 2, "user_id": 1, "type": "task"}])
        extra_graph.add_edges([{"start_id": 1, "end_id": 2, "note": "hi"}])
        nxg = extra_graph.traverse(Start(), Hop()).to_networkx()
        assert nxg.nodes["1"]["user_id"] == 1
        assert nxg.nodes["1"]["type"] == "task"
        assert nxg.edges["1", "2"]["note"] == "hi"

    def test_add_networkx_round_trips_extra_columns(self, extra_graph):
        import networkx as nx

        nxg = nx.DiGraph()
        nxg.add_node(1, user_id=1, type="task")
        nxg.add_node(2, user_id=1, type="task")
        nxg.add_edge(1, 2, note="hi")
        extra_graph.add_networkx(nxg)

        result = extra_graph.traverse(Start(), Hop())
        nodes_by_id = {n["id"]: n for n in result.nodes}
        assert nodes_by_id["1"]["user_id"] == 1
        assert nodes_by_id["1"]["properties"] == {"type": "task"}
        assert result.edges[0]["note"] == "hi"


class TestMerge:
    def test_merge_refreshes_the_extra_column_on_conflict(self, extra_graph):
        extra_graph.define_constraints(nodes=[Unique("email")])
        extra_graph.merge_nodes([{"email": "a@x.com", "user_id": 1}], on=["email"])
        extra_graph.merge_nodes([{"email": "a@x.com", "user_id": 2}], on=["email"])
        node = extra_graph.traverse(Start()).nodes[0]
        assert node["user_id"] == 2

    def test_merge_on_may_target_the_extra_column_itself(self, extra_graph):
        """Col() already names a real column for Unique()/merge(on=...)
        -- start_id/end_id proved that before this feature existed. An
        extra column is no different."""
        extra_graph.define_constraints(nodes=[Unique(Col("user_id"))])
        extra_graph.merge_nodes([{"user_id": 1, "name": "first"}], on=[Col("user_id")])
        extra_graph.merge_nodes([{"user_id": 1, "name": "second"}], on=[Col("user_id")])
        nodes = extra_graph.traverse(Start()).nodes
        assert len(nodes) == 1
        assert nodes[0]["properties"]["name"] == "second"
