"""
Several graphs in one pair of tables.

`Graph(engine, graph="marketing")` scopes every read and every write to
one `graph_id`. The tests here are about the failure this design can
produce and SQL cannot catch for you: a query that forgets the
discriminator does not error, it quietly returns or writes another
graph's rows. So each one checks the *other* graph is untouched, not
merely that this one worked.

Cross-graph edges are the exception — those Postgres does catch, via a
composite foreign key.
"""

from __future__ import annotations

import pytest

from hopai import Col, ConstraintViolation, Hop, Required, Start, Unique


@pytest.fixture()
def two(fresh_graph):
    """The same tables, two graphs, each with one node and one edge."""
    marketing = fresh_graph.in_graph("marketing")
    support = fresh_graph.in_graph("support")
    for graph, name in ((marketing, "m"), (support, "s")):
        graph.ingest({
            "nodes": [{"type": "person", "name": f"{name}1"},
                      {"type": "person", "name": f"{name}2"}],
            "edges": [{"start": {"name": f"{name}1"}, "end": {"name": f"{name}2"},
                       "kind": "knows"}],
        })
    return marketing, support


def names(graph, *hops) -> set:
    return {n["properties"].get("name") for n in graph.traverse(Start(), *hops).nodes}


class TestIsolation:
    def test_each_graph_sees_only_its_own_nodes(self, two):
        marketing, support = two
        assert names(marketing) == {"m1", "m2"}          # seed only
        assert names(marketing, Hop()) == {"m1", "m2"}   # and after a hop
        assert names(support, Hop()) == {"s1", "s2"}

    def test_a_filter_matching_both_graphs_still_returns_one(self, two):
        """The filter is identical in both graphs; only the scope differs."""
        marketing, support = two
        assert names(marketing.in_graph("marketing"), Hop()) == {"m1", "m2"}
        assert len(marketing.traverse(Start(where={"type": "person"}), Hop()).nodes) == 2

    def test_traversal_does_not_cross_into_another_graph(self, two):
        marketing, _ = two
        result = marketing.traverse(Start(where={"name": "m1"}), Hop(hops=(1, 5)))
        assert {n["properties"]["name"] for n in result.nodes} == {"m1", "m2"}

    def test_edges_are_scoped_too(self, two):
        marketing, support = two
        assert len(marketing.traverse(Start(), Hop()).edges) == 1
        assert len(support.traverse(Start(), Hop()).edges) == 1

    def test_a_third_graph_starts_empty(self, two):
        marketing, _ = two
        assert names(marketing.in_graph("brand-new"), Hop()) == set()

    def test_writes_land_only_in_their_own_graph(self, two):
        marketing, support = two
        marketing.add_nodes([{"type": "person", "name": "m3"}])
        assert "m3" not in names(support, Hop())

    def test_aggregation_counts_only_its_own_graph(self, two):
        """The aggregation path is a separate query builder, so it needs
        its own proof that _scoped() reaches it -- a count that quietly
        summed both graphs would be the silently-wrong answer this whole
        design exists to prevent."""
        from hopai import Count

        marketing, support = two
        support.add_nodes([{"type": "person", "name": "s3"}])
        assert marketing.aggregate(Start(where={"type": "person"}),
                                   aggregates={"n": Count()}) == {"n": 2}
        assert support.aggregate(Start(where={"type": "person"}),
                                 aggregates={"n": Count()}) == {"n": 3}
        assert marketing.aggregate(Start(), Hop(via={"kind": "knows"}),
                                   aggregates={"reached": Count()}) == {"reached": 1}

    def test_fifty_graphs_cost_fifty_rows(self, fresh_graph):
        """Graph names are strings, not schemas: a new graph is a row, not
        a CREATE SCHEMA, which is the whole reason for this design. Each
        one still sees only itself."""
        for i in range(50):
            fresh_graph.in_graph(f"tenant-{i}").add_nodes([{"n": i}])
        for i in (0, 7, 49):
            got = fresh_graph.in_graph(f"tenant-{i}").traverse(Start()).nodes
            assert {n["properties"]["n"] for n in got} == {i}


class TestCrossGraphEdges:
    def test_an_edge_between_graphs_is_rejected_by_the_database(self, two):
        """The composite foreign key makes this impossible rather than
        merely discouraged -- nothing in Python has to remember to check."""
        marketing, support = two
        their_node = support.traverse(Start(), Hop()).nodes[0]["id"]
        mine = marketing.traverse(Start(), Hop()).nodes[0]["id"]
        with pytest.raises(ConstraintViolation):
            marketing.add_edges([{"start_id": int(mine), "end_id": int(their_node)}])

    def test_a_property_reference_never_resolves_across_graphs(self, two):
        """Endpoint lookup is scoped, so referencing another graph's node
        reads as 'no node matches' rather than silently linking them."""
        marketing, _ = two
        with pytest.raises(ValueError, match="no node matches"):
            marketing.add_edges([{"start": {"name": "m1"}, "end": {"name": "s1"}}])


class TestScopedConstraints:
    def test_uniqueness_is_per_graph(self, fresh_graph):
        """Two graphs may each have a node with the same email -- they are
        separate graphs. One index serves both, with graph_id leading."""
        fresh_graph.define_constraints(nodes=[Unique("email")])
        fresh_graph.in_graph("a").add_nodes([{"email": "x@x.com"}])
        fresh_graph.in_graph("b").add_nodes([{"email": "x@x.com"}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.in_graph("a").add_nodes([{"email": "x@x.com"}])

    def test_a_check_declared_in_one_graph_does_not_bind_another(self, fresh_graph):
        """A CHECK covers the whole table, so an unscoped one would make
        this graph's rules law everywhere. The guard limits it."""
        strict = fresh_graph.in_graph("strict")
        loose = fresh_graph.in_graph("loose")
        strict.define_constraints(nodes=[Required("type")])
        with pytest.raises(ConstraintViolation):
            strict.add_nodes([{"name": "no type"}])
        assert loose.add_nodes([{"name": "no type"}]) == 1

    def test_merge_conflicts_only_within_the_graph(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("email")])
        a, b = fresh_graph.in_graph("a"), fresh_graph.in_graph("b")
        a.merge_nodes([{"email": "x@x.com", "v": 1}], on=["email"])
        b.merge_nodes([{"email": "x@x.com", "v": 2}], on=["email"])
        a.merge_nodes([{"email": "x@x.com", "v": 3}], on=["email"])
        assert len(a.traverse(Start()).nodes) == 1
        assert a.traverse(Start()).nodes[0]["properties"]["v"] == 3
        assert b.traverse(Start()).nodes[0]["properties"]["v"] == 2

    def test_edge_uniqueness_is_per_graph(self, fresh_graph):
        fresh_graph.define_constraints(edges=[Unique(Col("start_id"), Col("end_id"), "kind")])
        for name in ("a", "b"):
            graph = fresh_graph.in_graph(name)
            graph.ingest({"nodes": [{"n": 1}, {"n": 2}],
                          "edges": [{"start": {"n": 1}, "end": {"n": 2}, "kind": "knows"}]})
        with pytest.raises(ConstraintViolation):
            graph = fresh_graph.in_graph("a")
            ids = [int(n["id"]) for n in graph.traverse(Start(), Hop()).nodes]
            graph.add_edges([{"start_id": min(ids), "end_id": max(ids), "kind": "knows"}])


class TestCypherAndDefaults:
    def test_cypher_is_scoped_like_everything_else(self, fresh_graph):
        a, b = fresh_graph.in_graph("a"), fresh_graph.in_graph("b")
        a.cypher("CREATE (x:person {email: 'a@x.com'})-[:knows]->(y:person {email: 'b@x.com'})")
        assert len(a.cypher("MATCH (x:person)-[:knows]->(y) RETURN y").nodes) == 2
        assert len(b.cypher("MATCH (x:person)-[:knows]->(y) RETURN y").nodes) == 0
        assert len(b.traverse(Start(), Hop()).nodes) == 0

    def test_the_default_graph_is_a_graph_like_any_other(self, fresh_graph):
        fresh_graph.add_nodes([{"n": 1}])
        assert fresh_graph.graph == "default"
        assert len(fresh_graph.in_graph("other").traverse(Start()).nodes) == 0

    def test_in_graph_keeps_the_engine_and_tables(self, fresh_graph):
        other = fresh_graph.in_graph("other")
        assert other.engine is fresh_graph.engine
        assert other.nodes_tbl is fresh_graph.nodes_tbl

    @pytest.mark.parametrize("bad", ["", None, 3])
    def test_a_graph_name_must_be_a_non_empty_string(self, fresh_graph, bad):
        with pytest.raises(ValueError, match="non-empty string"):
            fresh_graph.in_graph(bad)

    def test_repr_says_which_graph(self, fresh_graph):
        assert "graph='default'" in repr(fresh_graph)
