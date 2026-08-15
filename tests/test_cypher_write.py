"""
Test suite for Cypher CREATE and MERGE.

Split the same way as the read tests: translation to an ingestion plan
needs no database, execution does. The plan is plain dicts naming the
same operations the Python API exposes, so these tests also pin the
promise that Cypher is a front end and not a second write path.

The refusals matter more here than on the read side. A write that means
something slightly different from what Cypher means does not return a
wrong answer you might notice -- it puts wrong data in your database.
"""

from __future__ import annotations

import pytest

from hopai import (
    Col, ConstraintViolation, CypherError, Hop, Start, Unique,
    cypher_to_operations,
)


def ops(query: str, **options) -> list:
    return cypher_to_operations(query, **options)


def properties_of(graph, **where) -> list:
    return [n["properties"] for n in graph.traverse(Start(where=where or None)).nodes]


def edges_of(graph) -> set:
    result = graph.traverse(Start(), Hop())
    return {(e["start_id"], e["end_id"], e["properties"].get("kind")) for e in result.edges}


@pytest.fixture()
def keyed_graph(fresh_graph):
    """A graph with the unique indexes MERGE needs, which is the setup
    any real use of MERGE requires anyway."""
    fresh_graph.define_constraints(
        nodes=[Unique("type", "email"), Unique("email")],
        edges=[Unique(Col("start_id"), Col("end_id"), "kind")],
    )
    return fresh_graph


# ---------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------

class TestPlanTranslation:
    def test_create_a_node(self):
        assert ops("CREATE (a:person {email: 'a@x.com'})") == [
            {"op": "create_nodes", "rows": [{"type": "person", "email": "a@x.com"}],
             "vars": ["a"]},
        ]

    def test_create_a_relationship_binds_both_ends(self):
        """Edges refer to variables, not ids -- the ids do not exist until
        the insert runs."""
        assert ops("CREATE (a {n: 1})-[:knows {since: 2019}]->(b {n: 2})") == [
            {"op": "create_nodes", "rows": [{"n": 1}, {"n": 2}], "vars": ["a", "b"]},
            {"op": "create_edges",
             "rows": [{"start_var": "a", "end_var": "b",
                       "properties": {"kind": "knows", "since": 2019}}]},
        ]

    def test_backward_arrow_reverses_the_edge(self):
        plan = ops("CREATE (a {n: 1})<-[:knows]-(b {n: 2})")
        assert plan[1]["rows"][0]["start_var"] == "b"
        assert plan[1]["rows"][0]["end_var"] == "a"

    def test_match_before_create_becomes_a_lookup(self):
        assert ops("MATCH (a {email: 'a'}), (b {email: 'b'}) CREATE (a)-[:knows]->(b)") == [
            {"op": "match", "var": "a", "where": {"email": "a"}},
            {"op": "match", "var": "b", "where": {"email": "b"}},
            {"op": "create_edges",
             "rows": [{"start_var": "a", "end_var": "b", "properties": {"kind": "knows"}}]},
        ]

    def test_merge_matches_on_every_pattern_property(self):
        """Cypher's MERGE matches the whole property map, so those are the
        conflict keys -- which is also why anything that should not take
        part in matching belongs in ON CREATE SET."""
        assert ops("MERGE (a:person {email: 'a@x.com'})") == [
            {"op": "merge_nodes", "rows": [{"type": "person", "email": "a@x.com"}],
             "on": ["email", "type"], "on_create": {}, "on_match": {}, "vars": ["a"]},
        ]

    def test_on_create_and_on_match_are_carried_separately(self):
        plan = ops("MERGE (a {email: 'a'}) ON CREATE SET a.n = 1 ON MATCH SET a.seen = 2")
        assert plan[0]["on_create"] == {"n": 1}
        assert plan[0]["on_match"] == {"seen": 2}
        assert plan[0]["on"] == ["email"]

    def test_several_clauses_keep_their_order(self):
        plan = ops("CREATE (a {n: 1}) MERGE (b {n: 2}) CREATE (c {n: 3})")
        assert [step["op"] for step in plan] == ["create_nodes", "merge_nodes", "create_nodes"]

    def test_multiple_patterns_in_one_create(self):
        plan = ops("CREATE (a {n: 1}), (b {n: 2})")
        assert plan == [{"op": "create_nodes", "rows": [{"n": 1}, {"n": 2}],
                         "vars": ["a", "b"]}]

    def test_label_and_type_keys_are_configurable(self):
        plan = ops("CREATE (a:Person)-[:KNOWS]->(b:Person)",
                   node_label_key="label", edge_type_key="rel")
        assert plan[0]["rows"] == [{"label": "Person"}, {"label": "Person"}]
        assert plan[1]["rows"][0]["properties"] == {"rel": "KNOWS"}

    def test_a_read_query_is_refused(self):
        with pytest.raises(CypherError, match="only reads"):
            ops("MATCH (a) RETURN a")


class TestWriteRefusals:
    def test_full_path_merge_is_refused(self):
        """Cypher's `MERGE (a {..})-[:x]->(b {..})` matches the WHOLE path
        and creates all of it when it does not match -- duplicating nodes
        that already exist. Reproducing that quietly would be worse than
        refusing."""
        with pytest.raises(CypherError, match="both endpoints already bound"):
            ops("MERGE (a {n: 1})-[:knows]->(b {n: 2})")

    def test_variable_length_cannot_be_written(self):
        with pytest.raises(CypherError, match="variable-length"):
            ops("CREATE (a {n: 1})-[:knows*1..3]->(b {n: 2})")

    def test_anonymous_endpoints_are_refused(self):
        with pytest.raises(CypherError, match="has to be named"):
            ops("CREATE ({n: 1})-[:knows]->({n: 2})")

    def test_traversal_before_a_write_is_refused(self):
        with pytest.raises(CypherError, match="may only bind single nodes"):
            ops("MATCH (a)-[:knows]->(b) CREATE (a)-[:x]->(b)")

    def test_where_before_a_write_is_refused(self):
        with pytest.raises(CypherError, match="WHERE is not supported before a write"):
            ops("MATCH (a) WHERE a.n = 1 CREATE (a)-[:x]->(b {n: 2})")

    def test_match_must_name_and_identify_what_it_binds(self):
        with pytest.raises(CypherError, match="nothing to find the node by"):
            ops("MATCH (a) CREATE (a)-[:x]->(b {n: 2})")

    def test_redefining_a_bound_variable_is_refused(self):
        with pytest.raises(CypherError, match="already bound"):
            ops("MATCH (a {n: 1}) CREATE (a {n: 2})-[:x]->(b {n: 3})")

    def test_multiple_relationship_types_cannot_be_written(self):
        with pytest.raises(CypherError, match="one type"):
            ops("CREATE (a {n: 1})-[:knows|likes]->(b {n: 2})")

    def test_merge_needs_properties(self):
        """Start-anchored: MERGE is spelled as the keyword the caller
        must type, and a lowercased mutant (_merge_node__mutmut_9) kept
        the loosely-matched middle intact."""
        with pytest.raises(CypherError) as exc:
            ops("MERGE (a)")
        assert str(exc.value).startswith("MERGE needs properties to match on")

    def test_relationship_merge_needs_a_type(self):
        with pytest.raises(CypherError, match="something to match it by"):
            ops("MATCH (a {n: 1}), (b {n: 2}) MERGE (a)-[]->(b)")

    def test_on_set_referring_to_an_unbound_variable(self):
        with pytest.raises(CypherError, match="does not bind"):
            ops("MERGE (a {n: 1}) ON CREATE SET b.x = 1")

    def test_a_mutating_query_is_refused_by_the_write_translator(self):
        """CREATE/MERGE and DELETE/SET compile to different plans run by
        different executors, so the ingestion translator hands these
        over by name instead of returning half of what was asked for."""
        for query in ("MATCH (a {n: 1}) SET a.x = 2", "MATCH (a {n: 1}) DELETE a"):
            with pytest.raises(CypherError, match="graph.mutate_cypher"):
                ops(query)

    # Refusal texts pinned by their actionable phrase -- see the same
    # note in test_cypher.py: mutation testing proved these messages
    # could be rewritten freely, and the message is the behavior.

    def test_alternation_in_a_created_relationship_names_the_pattern(self):
        with pytest.raises(CypherError) as exc:
            ops("CREATE (a {i: 1})-[:x|y]->(b {i: 2})")
        assert str(exc.value).startswith("a relationship being written has one type")
        assert "`[:x|y]` is a pattern to match, not to create" in str(exc.value)

    def test_traversing_match_before_a_write_refused(self):
        with pytest.raises(CypherError) as exc:
            ops("MATCH (a {x: 1})-[:k]->(b) CREATE (b)-[:z]->(c {n: 1})")
        assert str(exc.value).startswith(
            "a MATCH that precedes CREATE or MERGE may only bind single nodes")

    def test_where_before_a_write_refused(self):
        with pytest.raises(CypherError) as exc:
            ops("MATCH (a {x: 1}) WHERE a.y = 2 CREATE (a)-[:z]->(c {n: 1})")
        assert str(exc.value).startswith("WHERE is not supported before a write")

    def test_anonymous_edge_endpoints_refused(self):
        with pytest.raises(CypherError) as exc:
            ops("CREATE ({x: 1})-[:knows]->({y: 2})")
        assert str(exc.value).startswith("a node an edge connects has to be named")

    def test_on_match_set_refusal_names_the_clause(self):
        """ON CREATE SET and ON MATCH SET share one loop distinguished
        only by the interpolated label -- lowercase either label (mutants
        _merge_node 12/14) and the error blames the wrong clause."""
        with pytest.raises(CypherError) as exc:
            ops("MERGE (a {n: 1}) ON MATCH SET b.x = 1")
        assert "ON MATCH SET refers to ['b']" in str(exc.value)
        with pytest.raises(CypherError) as exc:
            ops("MERGE (a {n: 1}) ON CREATE SET b.x = 1")
        assert "ON CREATE SET refers to ['b']" in str(exc.value)

    def test_multi_assignment_set_keeps_every_assignment(self):
        """`SET a.x = 1, a.y = 2` -- the comma loop in
        _parse_set_assignments had no test, so breaking the separator
        check (mutant _13) was free."""
        plan = ops("MERGE (a {n: 1}) ON CREATE SET a.x = 1, a.y = 2")
        assert plan[0]["on_create"] == {"x": 1, "y": 2}

    def test_a_three_node_chain_wires_each_edge_to_its_own_ends(self):
        """(a)-[:x]->(b)-[:y]->(c): edge i connects nodes i and i+1. The
        off-by-one mutant (_create__mutmut_10) is invisible on two-node
        chains, where nodes[index-1] happens to equal the other end."""
        plan = ops("CREATE (a {i: 1})-[:x]->(b {i: 2})-[:y]->(c {i: 3})")
        assert [(r["start_var"], r["end_var"]) for r in plan[1]["rows"]] == [
            ("a", "b"), ("b", "c")]

    def test_matching_label_and_type_property_agree(self):
        """`(a:person {type: 'person'})` is redundant but consistent, so
        it must pass -- the consistency check indexes labels[0], and the
        off-by-one mutant (_node_properties__mutmut_9) crashed on any
        single-label node that spelled its type out."""
        plan = ops("CREATE (a:person {type: 'person', x: 1})")
        assert plan[0]["rows"] == [{"type": "person", "x": 1}]

    def test_typeless_arrow_creates_an_edge_with_no_kind(self):
        """`(a)-->(b)` has empty rel props, and the write path must
        treat that {} as {} -- the props-to-None mutant
        (_parse_rel__mutmut_8) crashed exactly here."""
        plan = ops("CREATE (a {x: 1})-->(b {y: 2})")
        assert plan[0]["op"] == "create_nodes"
        assert plan[1]["op"] == "create_edges"
        assert plan[1]["rows"][0]["properties"] == {}

    def test_null_property_values_are_literals(self):
        plan = ops("CREATE (a {gone: NULL, missing: null})")
        assert plan[0]["rows"] == [{"gone": None, "missing": None}]


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------

class TestCreate:
    def test_a_node(self, fresh_graph):
        result = fresh_graph.write_cypher("CREATE (a:person {email: 'a@x.com'})")
        assert (result.nodes, result.edges) == (1, 0)
        assert properties_of(fresh_graph) == [{"type": "person", "email": "a@x.com"}]

    def test_a_relationship_wires_the_generated_ids(self, fresh_graph):
        """The ids come back from the INSERT, so the edge connects exactly
        the two rows just written -- not whatever a property lookup would
        have matched, which duplicates make ambiguous."""
        fresh_graph.write_cypher(
            "CREATE (a:person {n: 1})-[:knows {since: 2019}]->(b:person {n: 2})")
        nodes = {tuple(sorted(p.items())) for p in properties_of(fresh_graph)}
        assert len(nodes) == 2
        edge = fresh_graph.traverse(Start(), Hop()).edges[0]
        assert edge["properties"] == {"kind": "knows", "since": 2019}

    def test_creating_a_duplicate_node_is_unambiguous(self, fresh_graph):
        """Two nodes with identical properties, each with its own edge.
        Resolving endpoints by property would have raised 'ambiguous'
        here; returned ids make it exact."""
        fresh_graph.write_cypher("CREATE (a {n: 1})-[:knows]->(b {n: 2})")
        fresh_graph.write_cypher("CREATE (c {n: 1})-[:knows]->(d {n: 2})")
        result = fresh_graph.traverse(Start(), Hop())
        assert len(result.edges) == 2
        assert len({e["start_id"] for e in result.edges}) == 2

    def test_a_chain_in_one_statement(self, fresh_graph):
        fresh_graph.write_cypher(
            "CREATE (a {n: 1})-[:k]->(b {n: 2})-[:k]->(c {n: 3})")
        assert len(fresh_graph.traverse(Start(), Hop()).edges) == 2

    def test_backward_arrow(self, fresh_graph):
        fresh_graph.write_cypher("CREATE (a {n: 1})<-[:k]-(b {n: 2})")
        by_n = {p["n"]: i for i, p in
                zip([n["id"] for n in fresh_graph.traverse(Start()).nodes],
                    properties_of(fresh_graph), strict=True)}
        assert edges_of(fresh_graph) == {(by_n[2], by_n[1], "k")}

    def test_match_then_create_an_edge(self, fresh_graph):
        fresh_graph.write_cypher("CREATE (a {email: 'a'}) CREATE (b {email: 'b'})")
        result = fresh_graph.write_cypher(
            "MATCH (a {email: 'a'}), (b {email: 'b'}) CREATE (a)-[:knows]->(b)")
        assert (result.nodes, result.edges) == (0, 1)

    def test_matching_nothing_is_refused(self, fresh_graph):
        with pytest.raises(ValueError, match="no node matches"):
            fresh_graph.write_cypher("MATCH (a {email: 'ghost'}) CREATE (a)-[:k]->(b {n: 1})")

    def test_constraints_still_apply(self, keyed_graph):
        keyed_graph.write_cypher("CREATE (a:person {email: 'a@x.com'})")
        with pytest.raises(ConstraintViolation):
            keyed_graph.write_cypher("CREATE (b:person {email: 'a@x.com'})")


class TestMerge:
    def test_creates_then_matches(self, keyed_graph):
        for _ in range(3):
            keyed_graph.write_cypher("MERGE (a:person {email: 'a@x.com'})")
        assert len(properties_of(keyed_graph)) == 1

    def test_on_create_set_applies_only_on_creation(self, keyed_graph):
        keyed_graph.write_cypher(
            "MERGE (a:person {email: 'a@x.com'}) ON CREATE SET a.name = 'Alice'")
        keyed_graph.write_cypher(
            "MERGE (a:person {email: 'a@x.com'}) ON CREATE SET a.name = 'Overwritten'")
        assert properties_of(keyed_graph)[0]["name"] == "Alice"

    def test_on_match_set_applies_only_on_a_match(self, keyed_graph):
        query = "MERGE (a:person {email: 'a@x.com'}) ON MATCH SET a.seen = 1"
        keyed_graph.write_cypher(query)
        assert "seen" not in properties_of(keyed_graph)[0]
        keyed_graph.write_cypher(query)
        assert properties_of(keyed_graph)[0]["seen"] == 1

    def test_both_clauses_together(self, keyed_graph):
        query = ("MERGE (a:person {email: 'a@x.com'}) "
                 "ON CREATE SET a.created = 1 ON MATCH SET a.seen = 1")
        keyed_graph.write_cypher(query)
        keyed_graph.write_cypher(query)
        stored = properties_of(keyed_graph)[0]
        assert stored["created"] == 1 and stored["seen"] == 1

    def test_without_the_unique_index_it_says_what_to_declare(self, fresh_graph):
        with pytest.raises(ConstraintViolation, match="define_constraints"):
            fresh_graph.write_cypher("MERGE (a:person {email: 'a@x.com'})")

    def test_merging_a_relationship_between_bound_nodes(self, keyed_graph):
        keyed_graph.write_cypher("MERGE (a {email: 'a'}) MERGE (b {email: 'b'})")
        for _ in range(3):
            keyed_graph.write_cypher(
                "MATCH (a {email: 'a'}), (b {email: 'b'}) MERGE (a)-[:knows]->(b)")
        assert len(keyed_graph.traverse(Start(), Hop()).edges) == 1

    def test_merge_then_create_an_edge_in_one_statement(self, keyed_graph):
        result = keyed_graph.write_cypher(
            "MERGE (a {email: 'a'}) MERGE (b {email: 'b'}) CREATE (a)-[:knows]->(b)")
        assert (result.nodes, result.edges) == (2, 1)


class TestAtomicityAndDispatch:
    def test_a_failing_statement_writes_nothing(self, keyed_graph):
        """One query, one transaction: the second node violating a
        constraint must take the first one with it."""
        with pytest.raises(ConstraintViolation):
            keyed_graph.write_cypher(
                "CREATE (a:person {email: 'x@x.com'}) CREATE (b:person {email: 'x@x.com'})")
        assert properties_of(keyed_graph) == []

    def test_a_failing_edge_rolls_back_its_nodes(self, fresh_graph):
        with pytest.raises(ValueError):
            fresh_graph.write_cypher(
                "CREATE (a {n: 1}) MATCH (b {email: 'ghost'}) CREATE (a)-[:k]->(b)")
        assert properties_of(fresh_graph) == []

    def test_cypher_dispatches_on_what_the_query_does(self, keyed_graph):
        written = keyed_graph.cypher("CREATE (a:person {email: 'a@x.com'})")
        assert written.nodes == 1                       # IngestResult
        read = keyed_graph.cypher("MATCH (a:person) RETURN a")
        assert len(read.nodes) == 1 and hasattr(read, "to_networkx")   # Subgraph

    def test_cypher_operations_does_not_execute(self, fresh_graph):
        fresh_graph.cypher_operations("CREATE (a {n: 1})")
        assert properties_of(fresh_graph) == []

    def test_written_data_is_traversable(self, keyed_graph):
        """End to end in one notation: write a graph in Cypher, then walk
        it in Cypher."""
        keyed_graph.write_cypher(
            "CREATE (a:person {email: 'a'})-[:knows]->(b:person {email: 'b'})")
        result = keyed_graph.cypher("MATCH (x:person {email: 'a'})-[:knows]->(y) RETURN y")
        assert {n["properties"]["email"] for n in result.nodes} == {"a", "b"}
