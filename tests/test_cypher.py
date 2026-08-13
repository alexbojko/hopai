"""
Test suite for hopai.cypher.

Two halves, and the split matters:

  - Translation tests, which need no database. They assert on the
    Start/Hop objects the translator emits, so they run anywhere and
    cover the whole grammar including every refusal.
  - End-to-end tests against the fixture graph, which assert that a
    Cypher query returns the SAME ROWS as the equivalent Python API call
    from test_hopai.py. Structural equivalence is not enough for the
    cases where Cypher and hopai disagree about semantics -- the
    NULL-safe negation idiom especially, which is only worth translating
    if it really does select what NOT({...}) selects.

Filters are compared by repr(): the filter classes are deliberately
plain holders with no __eq__ (defining one would make them unhashable),
and their reprs are exact and readable.
"""

from __future__ import annotations

import pytest

from hopai import (
    NOT, Hop, Start, CypherError, cypher_to_traversal, traverse_cypher,
)


def tr(query: str, **options):
    return cypher_to_traversal(query, **options)


def bounds(hop: Hop) -> tuple:
    return (hop.min_hops, hop.max_hops)


# ---------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------

class TestPatternTranslation:
    def test_label_type_and_props(self):
        start, hops = tr("MATCH (a:person)-[:friend]->(b {active: true}) RETURN b")
        assert start.where == {"type": "person"}
        assert start.label == "a"
        assert len(hops) == 1
        assert hops[0].where == {"active": True}
        assert hops[0].via == {"kind": "friend"}
        assert bounds(hops[0]) == (1, 1)

    def test_variables_become_labels(self):
        """Hop.label exists for the caller's own reference, and a Cypher
        variable is exactly that -- so it survives translation."""
        start, hops = tr("MATCH (a)-[]->(b)-[]->(c) RETURN c")
        assert start.label == "a"
        assert [h.label for h in hops] == ["b", "c"]

    def test_anonymous_nodes_and_rels(self):
        start, hops = tr("MATCH ()-[]->({flag: 1}) RETURN 1")
        assert start.where is None and start.label is None
        assert hops[0].where == {"flag": 1} and hops[0].via is None

    @pytest.mark.parametrize("pattern,expected", [
        ("-[]->", "forward"),
        ("<-[]-", "backward"),
        ("-->", "forward"),
        ("<--", "backward"),
    ])
    def test_direction(self, pattern, expected):
        _, hops = tr(f"MATCH (a){pattern}(b) RETURN b")
        assert hops[0].direction == expected

    def test_mixed_direction_chain(self):
        """Direction is per-hop in hopai, so a chain may alternate."""
        _, hops = tr("MATCH (a)-[:x]->(b)<-[:y]-(c) RETURN c")
        assert [h.direction for h in hops] == ["forward", "backward"]
        assert [h.via for h in hops] == [{"kind": "x"}, {"kind": "y"}]

    @pytest.mark.parametrize("spec,expected", [
        ("", (1, 1)),
        ("*3", (3, 3)),
        ("*1..4", (1, 4)),
        ("*..4", (1, 4)),
    ])
    def test_variable_length(self, spec, expected):
        _, hops = tr(f"MATCH (a)-[{spec}]->(b) RETURN b")
        assert bounds(hops[0]) == expected

    def test_unbounded_allowed_with_explicit_cap(self):
        _, hops = tr("MATCH (a)-[*]->(b) RETURN b", max_var_length=5)
        assert bounds(hops[0]) == (1, 5)
        _, hops = tr("MATCH (a)-[*2..]->(b) RETURN b", max_var_length=6)
        assert bounds(hops[0]) == (2, 6)

    def test_multiple_rel_types_become_or_shorthand(self):
        _, hops = tr("MATCH (a)-[:knows|refers]->(b) RETURN b")
        assert hops[0].via == {"kind": ["knows", "refers"]}

    def test_rel_property_map(self):
        _, hops = tr("MATCH (a)-[{kind: 'knows'}]->(b) RETURN b")
        assert hops[0].via == {"kind": "knows"}

    def test_chained_match_clauses_extend_one_chain(self):
        """Two MATCH clauses joined on a shared variable are the same
        linear chain written in two pieces -- the shape the Cypher in
        benchmarks/README.md uses for its compound query."""
        start, hops = tr("""
            MATCH (a {type: 'leaf'})-[*1..4]->(m {flag: 1})
            MATCH (m)-[*1..3]->(h {type: 'hub'})
            RETURN h
        """)
        assert start.where == {"type": "leaf"}
        assert [bounds(h) for h in hops] == [(1, 4), (1, 3)]
        assert [h.where for h in hops] == [{"flag": 1}, {"type": "hub"}]

    def test_rebinding_adds_filters_to_the_join_node(self):
        _, hops = tr("MATCH (a)-[]->(m {flag: 1}) MATCH (m {x: 2})-[]->(z) RETURN z")
        assert hops[0].where == {"flag": 1, "x": 2}

    def test_optional_match_is_the_last_hop(self):
        _, hops = tr("""
            MATCH (h {type: 'hub'})<-[]-(a)
            OPTIONAL MATCH (a)-[]->(d {type: 'leaf'})
            RETURN d
        """)
        assert [h.optional for h in hops] == [False, True]

    def test_start_only_pattern(self):
        start, hops = tr("MATCH (a {type: 'leaf'}) RETURN a")
        assert hops == []
        assert start.where == {"type": "leaf"}

    def test_comments_and_case_insensitive_keywords(self):
        start, hops = tr("""
            // find people
            match (a:person)-[:friend]->(b) Where b.age > 18
            return b
        """)
        assert start.where == {"type": "person"}
        assert repr(hops[0].where) == "GT('age', 18)"


class TestLabelMapping:
    def test_custom_keys(self):
        start, hops = tr(
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN b",
            node_label_key="label", edge_type_key="rel",
        )
        assert start.where == {"label": "Person"}
        assert hops[0].via == {"rel": "KNOWS"}

    def test_labels_ignored_when_key_is_none(self):
        """`(a:Node {type: 'leaf'})` -- the benchmark README's spelling,
        where the label carries no information and the property does."""
        start, _ = tr("MATCH (a:Node {type: 'leaf'})-[]->(b) RETURN b", node_label_key=None)
        assert start.where == {"type": "leaf"}

    def test_rel_types_ignored_when_key_is_none(self):
        _, hops = tr("MATCH (a)-[:EDGE]->(b) RETURN b", edge_type_key=None)
        assert hops[0].via is None


# ---------------------------------------------------------------------
# WHERE
# ---------------------------------------------------------------------

class TestWhereTranslation:
    def test_conjuncts_split_across_variables(self):
        """One WHERE, two variables: each conjunct has to land on the hop
        that binds its variable, since a filter binds one node."""
        start, hops = tr("MATCH (a)-[]->(b) WHERE a.x = 1 AND b.y = 2 RETURN b")
        assert start.where == {"x": 1}
        assert hops[0].where == {"y": 2}

    def test_same_variable_or_is_expressible(self):
        start, _ = tr("MATCH (a) WHERE a.x = 1 OR a.y = 2 RETURN a")
        assert repr(start.where) == "OR({'x': 1}, {'y': 2})"

    @pytest.mark.parametrize("op,expected", [
        (">", "GT('age', 18)"),
        (">=", "GTE('age', 18)"),
        ("<", "LT('age', 18)"),
        ("<=", "LTE('age', 18)"),
    ])
    def test_range_comparisons(self, op, expected):
        start, _ = tr(f"MATCH (a) WHERE a.age {op} 18 RETURN a")
        assert repr(start.where) == expected

    def test_in_becomes_value_list_shorthand(self):
        start, _ = tr("MATCH (a) WHERE a.type IN ['person', 'company'] RETURN a")
        assert start.where == {"type": ["person", "company"]}

    def test_inline_props_and_where_merge_into_one_containment(self):
        """Two dicts for the same node fold into a single `@>` test rather
        than an AND of two -- same result, one less operation."""
        start, _ = tr("MATCH (a:person {active: true}) WHERE a.city = 'NYC' RETURN a")
        assert start.where == {"type": "person", "active": True, "city": "NYC"}

    def test_null_safe_negation_idiom(self):
        """The one Cypher spelling that maps exactly onto hopai's NOT.
        Recognized as a two-term unit -- neither half translates alone."""
        start, _ = tr("MATCH (a) WHERE a.type IS NULL OR a.type <> 'leaf' RETURN a")
        assert repr(start.where) == repr(NOT({"type": "leaf"}))

    def test_null_safe_negation_either_order(self):
        start, _ = tr("MATCH (a) WHERE a.type <> 'leaf' OR a.type IS NULL RETURN a")
        assert repr(start.where) == repr(NOT({"type": "leaf"}))

    def test_is_null_uses_the_callable_escape_hatch(self):
        """Containment tests for a value, so 'this key is absent' has no
        containment form -- it goes out through the escape hatch."""
        start, _ = tr("MATCH (a) WHERE a.name IS NULL RETURN a")
        assert callable(start.where)

    def test_predicate_on_relationship_variable_becomes_via(self):
        _, hops = tr("MATCH (a)-[r]->(b) WHERE r.kind = 'knows' RETURN b")
        assert hops[0].via == {"kind": "knows"}

    def test_all_relationships_becomes_via_on_every_hop_of_the_path(self):
        """`relationships(p)` is every edge on the path, and Hop.via is
        every edge a hop traverses -- so the predicate lands on each hop
        the path variable spans."""
        _, hops = tr("""
            MATCH p=(a)-[*1..4]->(m)-[*1..2]->(z)
            WHERE all(r IN relationships(p) WHERE r.tag IN ['p1', 'p2'])
            RETURN z
        """)
        assert [h.via for h in hops] == [{"tag": ["p1", "p2"]}, {"tag": ["p1", "p2"]}]

    def test_where_on_each_match_clause(self):
        start, hops = tr("""
            MATCH (a)-[]->(b) WHERE a.x = 1
            MATCH (b)-[]->(c) WHERE c.y = 2
            RETURN c
        """)
        assert start.where == {"x": 1}
        assert hops[1].where == {"y": 2}


# ---------------------------------------------------------------------
# Refusals. Each of these is a construct that either has no hopai
# equivalent, or -- worse and the reason this module refuses rather than
# approximates -- one that would translate into different semantics.
# ---------------------------------------------------------------------

class TestSemanticRefusals:
    def test_bare_not_equal_refused(self):
        """`a.type <> 'leaf'` drops rows missing `type` in Cypher and
        keeps them under hopai's NOT. Same spelling, different answer, so
        it must not translate silently."""
        with pytest.raises(CypherError, match="NULL-safe"):
            tr("MATCH (a) WHERE a.type <> 'leaf' RETURN a")

    def test_not_on_equality_refused(self):
        with pytest.raises(CypherError, match="NULL-safe"):
            tr("MATCH (a) WHERE NOT a.type = 'leaf' RETURN a")

    def test_cross_variable_or_refused(self):
        with pytest.raises(CypherError, match="several variables"):
            tr("MATCH (a)-[]->(b) WHERE a.x = 1 OR b.y = 2 RETURN b")

    def test_label_colliding_with_property_refused(self):
        """`(a:Node {type: 'leaf'})` asks for type='Node' AND type='leaf'
        once the label is mapped onto a property -- unsatisfiable, and
        silently returning nothing would be the worst outcome."""
        with pytest.raises(CypherError, match="nothing can match"):
            tr("MATCH (a:Node {type: 'leaf'})-[]->(b) RETURN b")

    def test_multiple_labels_refused(self):
        with pytest.raises(CypherError, match="multiple labels"):
            tr("MATCH (a:A:B) RETURN a")

    def test_unbounded_star_refused_without_cap(self):
        with pytest.raises(CypherError, match="unbounded"):
            tr("MATCH (a)-[*]->(b) RETURN b")

    def test_zero_length_refused(self):
        with pytest.raises(CypherError, match="zero-length"):
            tr("MATCH (a)-[*0..3]->(b) RETURN b")

    def test_undirected_refused(self):
        with pytest.raises(CypherError, match="[Uu]ndirected"):
            tr("MATCH (a)-[]-(b) RETURN b")

    def test_aggregation_refused(self):
        """Every Cypher example in benchmarks/README.md ends in count().
        Ignoring it would answer a different question than was asked."""
        with pytest.raises(CypherError, match="aggregation"):
            tr("MATCH (a)-[]->(b) RETURN count(DISTINCT a)")

    @pytest.mark.parametrize("query", [
        "MATCH (a)-[]->(b) RETURN b ORDER BY b.x",
        "MATCH (a)-[]->(b) RETURN b LIMIT 10",
        "MATCH (a)-[]->(b) WITH b MATCH (b)-[]->(c) RETURN c",
        "MATCH (a)-[]->(b) RETURN b UNION MATCH (c)-[]->(d) RETURN d",
        "MATCH (a) SET a.x = 1",
        "MATCH (a) DETACH DELETE a",
    ])
    def test_unsupported_clauses_refused(self, query):
        with pytest.raises(CypherError, match="not supported"):
            tr(query)

    def test_a_write_query_is_refused_by_the_read_translator(self):
        """Reading and writing return different things, so the wrong
        entry point says so rather than half-working."""
        with pytest.raises(CypherError, match="this query writes"):
            tr("CREATE (a:person {email: 'a@x.com'})")

    def test_disjoint_comma_patterns_refused(self):
        with pytest.raises(CypherError, match="disjoint"):
            tr("MATCH (a)-[]->(b), (c)-[]->(d) RETURN b")

    def test_second_match_must_extend_the_chain_end(self):
        with pytest.raises(CypherError, match="linear chain"):
            tr("MATCH (a)-[]->(b)-[]->(c) MATCH (b)-[]->(z) RETURN z")

    def test_second_match_from_unbound_variable_refused(self):
        with pytest.raises(CypherError, match="bound variable"):
            tr("MATCH (a)-[]->(b) MATCH (x)-[]->(y) RETURN y")

    def test_revisiting_a_variable_refused(self):
        """`(a)-[]->(b)-[]->(a)` constrains the walk to come back to a --
        a cycle constraint the chain model cannot express."""
        with pytest.raises(CypherError, match="re-used"):
            tr("MATCH (a)-[]->(b)-[]->(a) RETURN b")

    def test_optional_must_be_last(self):
        with pytest.raises(CypherError, match="last clause"):
            tr("""
                MATCH (a)-[]->(b)
                OPTIONAL MATCH (b)-[]->(c)
                OPTIONAL MATCH (c)-[]->(d)
                RETURN d
            """)

    def test_optional_must_add_exactly_one_hop(self):
        with pytest.raises(CypherError, match="exactly one relationship"):
            tr("MATCH (a)-[]->(b) OPTIONAL MATCH (b)-[]->(c)-[]->(d) RETURN d")

    def test_optional_where_on_earlier_variable_refused(self):
        """In Cypher this filters the optional extension, not the row --
        a distinction hopai's last-hop flag cannot draw."""
        with pytest.raises(CypherError, match="OPTIONAL MATCH's WHERE"):
            tr("MATCH (a)-[]->(b) OPTIONAL MATCH (b)-[]->(c) WHERE a.x = 1 RETURN c")

    def test_unknown_variable_refused(self):
        with pytest.raises(CypherError, match="unknown variable"):
            tr("MATCH (a)-[]->(b) WHERE zz.x = 1 RETURN b")

    def test_non_numeric_range_comparison_refused(self):
        with pytest.raises(CypherError, match="needs a number"):
            tr("MATCH (a) WHERE a.name > 'x' RETURN a")

    def test_existential_relationship_predicates_refused(self):
        with pytest.raises(CypherError, match="not supported"):
            tr("MATCH p=(a)-[*1..2]->(b) WHERE any(r IN relationships(p) WHERE r.k = 1) RETURN b")


class TestSyntaxErrors:
    @pytest.mark.parametrize("query,message", [
        ("MATCH (a)-[]->(b RETURN b", "expected"),
        ("MATCH (a) WHERE a.x = RETURN a", "literal"),
        ("MATCH (a) WHERE a = 1 RETURN a", "property"),
        ("MATCH (a) WHERE a.x RETURN a", "comparison"),
        ("RETURN 1", "no MATCH"),
        ("MATCH (a) WHERE a.name = 'unterminated RETURN a", "unterminated string"),
        ("MATCH (a) WHERE a.name = `x` RETURN a", "unexpected character"),
    ])
    def test_syntax_errors_are_cypher_errors(self, query, message):
        with pytest.raises(CypherError, match=message):
            tr(query)

    def test_cypher_error_is_a_value_error(self):
        """Callers already catching ValueError from Hop/Start validation
        keep working."""
        assert issubclass(CypherError, ValueError)


# ---------------------------------------------------------------------
# End to end: same rows as the Python API, against the fixture graph.
# ---------------------------------------------------------------------

class TestAgainstFixtureGraph:
    def test_dead_end_excluded_by_rel_type(self, graph):
        """The Cypher spelling of test_hopai.py's
        test_dead_end_excluded_when_edge_kind_filtered -- n4's only edge
        is the wrong kind."""
        result = traverse_cypher(
            graph, "MATCH (a {type: 'leaf'})-[:knows]->(m {flag: 1}) RETURN m"
        )
        ids = {n["id"] for n in result.nodes}
        assert "4" not in ids
        assert {"1", "2", "3", "5", "6"} <= ids

    def test_null_safe_negation_selects_the_same_rows_as_NOT(self, graph):
        """The payoff test. test_hopai.py asserts NOT({'type': 'leaf'})
        selects m1, m2 and h1 -- including the two nodes with no `type`
        key at all. The translated idiom has to select exactly those, or
        the translation is only structurally right and factually wrong."""
        result = traverse_cypher(
            graph, "MATCH (a) WHERE a.type IS NULL OR a.type <> 'leaf' RETURN a"
        )
        assert {n["id"] for n in result.nodes} == {"5", "6", "7"}

    def test_is_null_selects_nodes_missing_the_key(self, graph):
        """n4 is the only leaf with no `priority`."""
        result = traverse_cypher(
            graph, "MATCH (a {type: 'leaf'}) WHERE a.priority IS NULL RETURN a"
        )
        assert {n["id"] for n in result.nodes} == {"4"}

    def test_range_comparison_matches_python_api(self, graph):
        result = traverse_cypher(
            graph, "MATCH (a {type: 'leaf'}) WHERE a.priority > 5 RETURN a"
        )
        assert {n["id"] for n in result.nodes} == {"2", "3"}

    def test_backward_traversal(self, graph):
        result = traverse_cypher(graph, "MATCH (h {type: 'hub'})<-[]-(a) RETURN a")
        assert {"5", "6", "7"} <= {n["id"] for n in result.nodes}

    def test_all_relationships_filter_matches_via(self, graph):
        result = traverse_cypher(graph, """
            MATCH p=(a {type: 'leaf'})-[*1..1]->(m {flag: 1})
            WHERE all(r IN relationships(p) WHERE r.kind = 'knows')
            RETURN m
        """)
        assert "4" not in {n["id"] for n in result.nodes}

    def test_multi_hop_reconstruction(self, graph):
        """Mirrors test_multi_hop_edge_reconstruction: a hop spanning
        several real edges must report each of them."""
        result = traverse_cypher(graph, """
            MATCH (a {type: 'leaf'})-[:knows*1..3]->(m)
            MATCH (m)-[*1..2]->(h {type: 'hub'})
            RETURN h
        """)
        pairs = {(e["start_id"], e["end_id"]) for e in result.edges}
        assert ("1", "5") in pairs and ("5", "7") in pairs
        assert ("1", "7") not in pairs  # no such real edge exists

    def test_optional_keeps_matches_with_no_extension(self, graph):
        """The Cypher spelling of test_optional_keeps_matches_with_no_extension:
        the optional hop finds nothing, and m1/m2/h1 survive anyway."""
        result = traverse_cypher(graph, """
            MATCH (h {type: 'hub'})<-[]-(a) WHERE a.type IS NULL OR a.type <> 'leaf'
            OPTIONAL MATCH (a)-[]->(d {type: 'leaf'})
            WHERE d.priority >= 100 AND d.priority <= 200
            RETURN a, d
        """)
        assert {"5", "6", "7"} <= {n["id"] for n in result.nodes}

    def test_cypher_matches_python_api_exactly(self, graph):
        """One compound traversal, written both ways -- the translator's
        whole contract in a single assertion."""
        from hopai import AND, GT

        via_cypher = traverse_cypher(graph, """
            MATCH (a {type: 'leaf'})-[:knows*1..3]->(m {flag: 1})
            WHERE a.priority > 5
            RETURN m
        """)
        via_python = graph.traverse(
            Start(where=AND({"type": "leaf"}, GT("priority", 5))),
            Hop(where={"flag": 1}, via={"kind": "knows"}, hops=(1, 3), label="m"),
        )
        assert ({n["id"] for n in via_cypher.nodes}
                == {n["id"] for n in via_python.nodes})
        assert ({(e["start_id"], e["end_id"]) for e in via_cypher.edges}
                == {(e["start_id"], e["end_id"]) for e in via_python.edges})

    def test_traverse_cypher_returns_a_subgraph(self, graph):
        """Unlike traverse_json's dict -- this one is called from Python,
        so it hands back the object with .nodes / .to_networkx()."""
        result = traverse_cypher(graph, "MATCH (a {type: 'hub'}) RETURN a")
        assert hasattr(result, "to_networkx")
        assert result.to_dict()["nodes"]
