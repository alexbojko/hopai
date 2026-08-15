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
    NOT, Hop, Start, CypherError, aggregate_cypher, cypher_to_aggregation,
    cypher_to_traversal, traverse_cypher,
)


def tr(query: str, **options):
    return cypher_to_traversal(query, **options)


def agg(query: str, **options) -> dict:
    """The {name: aggregate-repr} dict a query translates to -- reprs for
    the same reason filters are compared by repr()."""
    _, _, aggregates = cypher_to_aggregation(query, **options)
    return {name: repr(a) for name, a in aggregates.items()}


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

    def test_backward_relationship_is_a_single_hop(self):
        """A plain `<-[:x]-` is one hop exactly, same as its forward
        twin -- a mutation-run survivor showed the backward arm's bounds
        were pinned nowhere, and (1, 2) there would quietly walk one hop
        further than the query says."""
        _, hops = tr("MATCH (a:person)<-[:works_at]-(b) RETURN b")
        assert hops[0].direction == "backward"
        assert bounds(hops[0]) == (1, 1)

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

    def test_a_trailing_comment_with_no_newline_terminates(self):
        """The tokenizer's comment branch has an at-end-of-input arm that
        only a comment ending the query exercises; mutant
        x__tokenize__mutmut_26 turned it into an infinite loop and
        nothing noticed."""
        start, _ = tr("MATCH (a:person) // that's everyone")
        assert start.where == {"type": "person"}

    def test_decimal_literals_stay_floats(self):
        """`1.5` must come through as the float 1.5 -- mutants
        x__tokenize__mutmut_97/98 broke the float branch and no test had
        a single decimal point in it."""
        start, _ = tr("MATCH (a {score: 1.5, count: 2})")
        assert start.where == {"score": 1.5, "count": 2}
        assert isinstance(start.where["score"], float)

    def test_capital_x_is_not_whitespace(self):
        """Mutant x__tokenize__mutmut_7 wrapped the whitespace set in XX,
        which silently made the letter X a token separator -- any name
        containing X would tokenize shredded."""
        start, hops = tr("MATCH (aXa:Xperson)-[:foo]->(b) WHERE aXa.Xfactor > 1 RETURN b")
        assert repr(start.where) in (
            "AND({'type': 'Xperson'}, GT('Xfactor', 1))",
            "AND(GT('Xfactor', 1), {'type': 'Xperson'})",
        )

    def test_type_alternation_with_repeated_colon(self):
        """`[:a|:b]` is the second legal spelling of alternation; the
        `|:` branch was never exercised."""
        _, hops = tr("MATCH (a)-[:knows|:likes]->(b) RETURN b")
        assert hops[0].via == {"kind": ["knows", "likes"]}

    def test_spaced_arrow_is_a_single_hop(self):
        """`(a)- ->(b)` reaches the no-bracket arm of _parse_rel, the one
        place the default hop bounds apply -- mutant _parse_rel__mutmut_44
        changed the default to 2 and no test went through that arm."""
        _, hops = tr("MATCH (a)- ->(b) RETURN b")
        assert bounds(hops[0]) == (1, 1)

    def test_null_literal_in_a_property_map(self):
        """NULL (any case) is a literal, mapping to JSON null -- mutant
        _parse_literal__mutmut_29 broke the keyword comparison and NULL
        fell through to the not-a-literal refusal."""
        start, _ = tr("MATCH (a {gone: NULL, missing: null})")
        assert start.where == {"gone": None, "missing": None}

    def test_negative_number_literals(self):
        """`-5` goes through its own minus-then-number branch; nothing
        observed it, so the branch's peek (mutant _parse_literal_13) was
        freely mutable."""
        start, _ = tr("MATCH (a {depth: -5, delta: -1.5})")
        assert start.where == {"depth": -5, "delta": -1.5}

    def test_string_escapes_produce_the_escaped_value(self):
        r"""A successfully-parsed escape was never OBSERVED -- the escape
        branch ran only inside unterminated-string error paths, whose
        buffer is discarded, so the entire _ESCAPES append cluster
        (x__tokenize 42-56) was freely mutable. \' must yield a quote,
        \n a newline, \\ one backslash, and an unknown escape its bare
        character."""
        start, _ = tr(r"MATCH (a {s: 'it\'s', t: 'a\nb', u: 'c\\d', v: 'x\qy'})")
        assert start.where == {"s": "it's", "t": "a\nb", "u": "c\\d", "v": "xqy"}

    def test_boolean_literals_in_a_property_map(self):
        """true AND false: only true had coverage, so the FALSE keyword
        comparison was freely mutable."""
        start, _ = tr("MATCH (a {flag: false, ok: true})")
        assert start.where == {"flag": False, "ok": True}

    def test_a_comment_ending_directly_before_a_keyword(self):
        """The comment skip must resume at exactly newline+1: the +2
        mutant ate the next character, which every existing comment test
        hid behind indentation whitespace."""
        start, _ = tr("MATCH (a:person) // c\nRETURN a")
        assert start.where == {"type": "person"}


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

    def test_a_whereless_clause_does_not_stop_later_wheres(self):
        """_attach_where skips clauses with no WHERE via `continue`;
        mutant _attach_where__mutmut_5 turned it into `break` and every
        WHERE after the first bare MATCH silently vanished -- a filter
        dropped without an error, the worst failure mode there is."""
        start, hops = tr("""
            MATCH (a)-[]->(b)
            MATCH (b)-[]->(c) WHERE c.y = 2
            RETURN c
        """)
        assert start.where is None
        assert hops[1].where == {"y": 2}

    def test_parenthesized_groups_translate(self):
        """`(x OR y) AND z` exercises the grouping branch of
        _parse_predicate, which no test entered -- without it the paren
        would be a syntax error and precedence inexpressible."""
        start, _ = tr("MATCH (a) WHERE (a.x = 1 OR a.y = 2) AND a.z = 3")
        assert repr(start.where) == "AND({'z': 3}, OR({'x': 1}, {'y': 2}))"

    @staticmethod
    def compiled(filt) -> str:
        from sqlalchemy import column as sa_column
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.dialects.postgresql import JSONB
        from hopai.filters import resolve
        return str(resolve(sa_column("properties", type_=JSONB), filt).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))

    def test_is_not_null_compiles_to_a_key_presence_test(self):
        """IS NOT NULL means `properties ? 'key'` -- and IS NULL its
        negation. The NOT-detection inside the IS parse was never
        asserted, so flipping `negated` (mutants _parse_predicate 29-40)
        swapped present for missing and every test still passed."""
        start, _ = tr("MATCH (a) WHERE a.email IS NOT NULL")
        sql = self.compiled(start.where)
        assert "properties ? 'email'" in sql and "NOT" not in sql
        start, _ = tr("MATCH (a) WHERE a.email IS NULL")
        assert "NOT (properties ? 'email')" in self.compiled(start.where)

    def test_an_or_that_is_not_the_null_safe_idiom_stays_an_or(self):
        """Only `x.k IS NULL OR x.k <> v` collapses to NOT({k: v});
        loosening the detection (mutant _null_safe_negation__mutmut_8)
        made any OR with an IS NULL half collapse -- silently answering
        a different question."""
        start, _ = tr("MATCH (a) WHERE a.k IS NULL OR a.n > 5")
        sql = self.compiled(start.where)
        assert "properties ? 'k'" in sql and "NOT" in sql   # the IS NULL half, intact
        assert "> 5" in sql                                  # the comparison half, intact


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

    def test_an_aggregating_query_is_refused_by_the_traversal_translator(self):
        """A subgraph and a number are different things, so the wrong
        entry point says so rather than dropping the count on the floor
        -- the mirror of the write-query refusal below."""
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a)-[]->(b) RETURN count(DISTINCT b)")
        # startswith, not substring: XX-padded string mutants keep the
        # inner text and survive a `match=`
        assert str(exc.value).startswith("this query aggregates")

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


# ---------------------------------------------------------------------
# Aggregating RETURN -- accepted spellings
# ---------------------------------------------------------------------

class TestAggregationTranslation:
    """Each accepted spelling is one whose Cypher meaning is EXACTLY what
    Graph.aggregate() computes -- the acceptance matrix from cypher.py's
    AGGREGATION docstring section, one test per rule."""

    def test_count_distinct_node(self):
        """count(DISTINCT b) is the distinct-node count, which is the one
        bare-variable aggregate that stays exact with paths involved."""
        assert agg("MATCH (a)-[]->(b) RETURN count(DISTINCT b)") == {"count": "Count()"}

    def test_distinct_property_aggregates(self):
        """DISTINCT collapses to distinct VALUES on both sides, so these
        translate exactly however many paths reach each node."""
        assert agg("MATCH (a)-[]->(b) RETURN sum(DISTINCT b.age), avg(DISTINCT b.age), "
                   "count(DISTINCT b.age)") == {
            "sum_age": "Sum('age', distinct=True)",
            "avg_age": "Avg('age', distinct=True)",
            "count_age": "Count('age', distinct=True)",
        }

    def test_min_max_are_exact_bare(self):
        """An extremum is immune to path multiplicity, so min/max need no
        DISTINCT -- refusing them would refuse a correct translation."""
        assert agg("MATCH (a)-[:x*1..3]->(b) RETURN min(b.age), max(b.age)") == {
            "min_age": "Min('age')", "max_age": "Max('age')",
        }

    def test_min_max_accept_redundant_distinct(self):
        assert agg("MATCH (a)-[]->(b) RETURN min(DISTINCT b.age)") == {"min_age": "Min('age')"}

    def test_zero_hops_allows_bare_aggregates(self):
        """With no hops every node is its own only row -- no multiplicity
        exists, so `RETURN count(a)` on a single-node pattern is exact.
        This is the shape of benchmarks/README.md's range_gt query."""
        assert agg("MATCH (a:leaf) WHERE a.priority > 5 RETURN count(a), avg(a.priority)") == {
            "count": "Count()", "avg_priority": "Avg('priority')",
        }

    def test_with_distinct_prefix_means_per_node(self):
        """`WITH DISTINCT b RETURN avg(b.age)` is Cypher's spelling of
        hopai's native per-matched-node aggregation -- recognized as a
        unit, like the null-safe negation idiom."""
        assert agg("MATCH (a)-[]->(b) WITH DISTINCT b RETURN "
                   "count(b), sum(b.age), avg(b.age)") == {
            "count": "Count()", "sum_age": "Sum('age')", "avg_age": "Avg('age')",
        }

    def test_with_distinct_count_star(self):
        assert agg("MATCH (a)-[]->(b) WITH DISTINCT b RETURN count(*)") == {"count": "Count()"}

    def test_with_distinct_duplicate_default_keys_refused(self):
        """count(b) and count(*) both land on the key 'count'; silently
        overwriting one would return fewer numbers than were asked for."""
        with pytest.raises(CypherError, match="alias"):
            agg("MATCH (a)-[]->(b) WITH DISTINCT b RETURN count(b), count(*)")

    def test_with_distinct_inner_distinct_still_means_values(self):
        assert agg("MATCH (a)-[]->(b) WITH DISTINCT b RETURN sum(DISTINCT b.age)") == {
            "sum_age": "Sum('age', distinct=True)",
        }

    def test_aliases_name_the_results(self):
        assert agg("MATCH (a)-[]->(b) RETURN count(DISTINCT b) AS n, "
                   "min(b.age) AS youngest") == {"n": "Count()", "youngest": "Min('age')"}

    def test_translation_returns_the_traversal_too(self):
        start, hops, aggregates = cypher_to_aggregation(
            "MATCH (a:person)-[:friend*1..4]->(b {active: true}) RETURN count(DISTINCT b)"
        )
        assert start.where == {"type": "person"}
        assert (hops[0].min_hops, hops[0].max_hops) == (1, 4)
        assert repr(hops[0].where) == repr({"active": True})
        # via included: a mutant that quietly defaulted edge_type_key
        # survived while only the option-override direction was tested
        assert repr(hops[0].via) == repr({"kind": "friend"})
        assert list(aggregates) == ["count"]


# ---------------------------------------------------------------------
# Aggregating RETURN -- refusals, each naming its rewrite
# ---------------------------------------------------------------------

class TestAggregationRefusals:
    def test_bare_count_with_hops_names_both_rewrites(self):
        """Bare count(b) counts one row per PATH -- inexpressible here.
        The error must hand back both exact spellings, or the caller is
        left knowing only what they cannot do."""
        with pytest.raises(CypherError, match=r"per PATH(.|\n)*DISTINCT(.|\n)*WITH DISTINCT b"):
            agg("MATCH (a)-[]->(b) RETURN count(b)")

    @pytest.mark.parametrize("item", ["count(*)", "sum(b.age)", "avg(b.age)", "count(b.age)"])
    def test_per_path_spellings_refused(self, item):
        with pytest.raises(CypherError, match="path"):
            agg(f"MATCH (a)-[]->(b) RETURN {item}")

    def test_bare_count_star_refusal_names_itself(self):
        """The refusal quotes the spelling it refuses -- `count(*)`,
        star included. A mutant that broke the `*` fallback printed
        `count(None)` and the loose "path" match above let it through."""
        with pytest.raises(CypherError, match=r"bare count\(\*\) aggregates one row per PATH"):
            agg("MATCH (a)-[]->(b) RETURN count(*)")

    def test_bare_aggregate_on_anonymous_last_node_names_the_rewrite(self):
        """With an anonymous last node there is no variable to suggest,
        so the rewrite hint falls back to the literal `<var>`
        placeholder -- pinned here because mutants that mangled it
        survived every named-variable test."""
        with pytest.raises(CypherError, match=r"WITH DISTINCT <var> RETURN"):
            agg("MATCH (a)-[]->() RETURN count(*)")

    def test_non_last_variable_refused(self):
        """The start-side count every benchmarks/README.md example uses:
        mid-chain match sets include nodes with no continuation to the
        chain's end, which Cypher would not count -- so this names the
        reversal instead of silently counting them."""
        with pytest.raises(CypherError, match="LAST node(.|\n)*[Rr]everse"):
            agg("MATCH (a)-[]->(b) RETURN count(DISTINCT a)")

    def test_relationship_variable_refused(self):
        with pytest.raises(CypherError, match="relationships"):
            agg("MATCH (a)-[r:knows]->(b) RETURN count(DISTINCT r)")

    def test_path_variable_refused(self):
        with pytest.raises(CypherError, match="relationships"):
            agg("MATCH p=(a)-[*1..2]->(b) RETURN count(DISTINCT p)")

    def test_unknown_variable_refused(self):
        with pytest.raises(CypherError, match="unknown variable"):
            agg("MATCH (a)-[]->(b) RETURN count(DISTINCT zz)")

    def test_mixing_aggregates_with_projection_is_grouping(self):
        """`RETURN b.city, count(DISTINCT b)` means GROUP BY, a feature
        hopai does not have -- silently computing the global count would
        answer a different question."""
        with pytest.raises(CypherError, match="GROUP BY"):
            agg("MATCH (a)-[]->(b) RETURN b, count(DISTINCT b)")

    def test_unsupported_aggregate_functions_name_the_supported_ones(self):
        with pytest.raises(CypherError, match="avg, count, max, min, sum"):
            agg("MATCH (a)-[]->(b) RETURN collect(b)")

    def test_whole_node_sum_refused(self):
        with pytest.raises(CypherError, match="whole node"):
            agg("MATCH (a)-[]->(b) RETURN sum(b)")

    def test_sum_star_refused(self):
        with pytest.raises(CypherError, match="nothing in particular"):
            agg("MATCH (a)-[]->(b) RETURN sum(*)")

    def test_optional_match_cannot_feed_an_aggregation(self):
        """count(DISTINCT c) over an OPTIONAL MATCH equals the count over
        the required MATCH -- accepting the flag would let callers
        believe it changed the number. The head phrase is pinned AT THE
        START: a substring match also matched it mid-message, so the
        XX-padding mutant (emit_aggregates__mutmut_6) survived."""
        with pytest.raises(CypherError) as exc:
            agg("MATCH (a)-[]->(b) OPTIONAL MATCH (b)-[]->(c) RETURN count(DISTINCT c)")
        assert str(exc.value).startswith("an OPTIONAL MATCH cannot feed an aggregation")

    def test_with_distinct_must_name_the_last_node(self):
        """The error names the node that WOULD be right -- `(b)` --
        which is the actionable half; a mutant that replaced it with the
        anonymous-node hint survived a match on the first half alone."""
        with pytest.raises(CypherError,
                           match=r"WITH DISTINCT a must name the last node of the chain \(b\)"):
            agg("MATCH (a)-[]->(b) WITH DISTINCT a RETURN count(a)")

    def test_with_distinct_wrong_var_and_anonymous_last_node(self):
        """When the last node has no variable the error cannot name it,
        so it says to add one -- the one WITH DISTINCT path no other
        test walks."""
        with pytest.raises(CypherError, match="give it a variable"):
            agg("MATCH (a)-[]->() WITH DISTINCT a RETURN count(a)")

    def test_general_with_still_refused(self):
        """Only the `WITH DISTINCT <var> RETURN <aggregates>` unit is
        recognized; every other WITH keeps its original refusal, now
        naming that one exception."""
        with pytest.raises(CypherError, match="WITH is not supported"):
            agg("MATCH (a)-[]->(b) WITH b RETURN count(b)")

    @pytest.mark.parametrize("query", [
        "MATCH (a)-[]->(b) WITH DISTINCT RETURN count(b)",                       # no variable
        "MATCH (a)-[]->(b) WITH DISTINCT b MATCH (b)-[]->(c) RETURN count(c)",   # no RETURN next
        "MATCH (a)-[]->(b) WITH b b RETURN count(b)",                            # no DISTINCT
        "MATCH (a)-[]->(b) WITH DISTINCT 5 RETURN count(b)",                     # not a name
    ])
    def test_near_miss_with_forms_get_the_canonical_refusal(self, query):
        """Each of these is one boolean-operator slip away from the gate
        accepting garbage -- consuming RETURN as the variable, or
        silently reading `WITH b b` as WITH DISTINCT b. Mutation testing
        produced exactly those slips and every one survived, because no
        test fed the gate a near-miss. All must fall through to the one
        honest WITH refusal."""
        with pytest.raises(CypherError, match="WITH is not supported"):
            agg(query)

    def test_with_distinct_before_a_non_aggregating_return_expression(self):
        """`RETURN 5` after the unit prefix: the gate accepts (a RETURN
        does follow), and the missing aggregate is what refuses -- a
        mutant that peeked one token past the gate re-routed this to the
        generic WITH error instead."""
        with pytest.raises(CypherError, match="aggregating RETURN"):
            agg("MATCH (a)-[]->(b) WITH DISTINCT b RETURN 5")

    def test_with_distinct_without_aggregates_refused(self):
        with pytest.raises(CypherError, match="aggregating RETURN"):
            agg("MATCH (a)-[]->(b) WITH DISTINCT b RETURN b")

    def test_order_by_after_aggregates_still_refused(self):
        with pytest.raises(CypherError, match="ORDER BY is not supported"):
            agg("MATCH (a)-[]->(b) RETURN count(DISTINCT b) ORDER BY b.x")

    def test_aggregate_after_a_write_refused(self):
        """A write produces an IngestResult; quietly dropping the count
        would be worse than either running it or refusing."""
        from hopai import cypher_to_operations
        with pytest.raises(CypherError, match="after a write"):
            cypher_to_operations("CREATE (a:person {email: 'a@x.com'}) RETURN count(a)")

    def test_a_plain_traversal_is_refused_by_the_aggregation_translator(self):
        """The mirror of cypher_to_traversal's refusal: each entry point
        names the other, so the caller is never stranded."""
        with pytest.raises(CypherError, match="no aggregating RETURN"):
            cypher_to_aggregation("MATCH (a)-[]->(b) RETURN b")

    # The refusal texts below are pinned by their ACTIONABLE phrase, not
    # just their exception type: mutation testing showed most of these
    # messages could be arbitrarily rewritten (XX-padded, case-flipped,
    # variables dropped) with every test still green -- and for an API
    # whose contract is "errors that name the fix", the message IS the
    # behavior.

    @pytest.mark.parametrize("word", ["any", "none", "single"])
    def test_existential_list_predicates_refused_by_name(self, word):
        with pytest.raises(CypherError) as exc:
            tr(f"MATCH (a) WHERE {word}(x IN a.tags WHERE x.v = 1) RETURN a")
        assert f"{word}(...) is not supported" in str(exc.value)
        assert "Hop.via filters every edge" in str(exc.value)

    def test_exists_refused_with_the_is_not_null_rewrite(self):
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a) WHERE exists(a.email) RETURN a")
        assert str(exc.value).startswith("exists(...) is not supported")
        assert "IS NOT NULL" in str(exc.value)

    def test_comma_separated_patterns_refused(self):
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a {x: 1}), (b {y: 2}) RETURN a")
        assert str(exc.value).startswith("comma-separated patterns describe disjoint")

    def test_optional_match_must_be_last(self):
        with pytest.raises(CypherError) as exc:
            tr("OPTIONAL MATCH (a)-[:x]->(b) MATCH (b)-[:y]->(c) RETURN c")
        assert str(exc.value).startswith("OPTIONAL MATCH must be the last clause")

    def test_later_match_must_continue_from_a_bound_variable(self):
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a)-[:x]->(b) MATCH (c)-[:y]->(d) RETURN d")
        assert str(exc.value).startswith("every MATCH after the first must continue the chain")

    def test_cross_variable_refusal_names_the_variables(self):
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a)-[]->(b) WHERE a.x = 1 OR b.y = 2 RETURN b")
        assert "several variables (a, b)" in str(exc.value)

    def test_undirected_refusal_names_hop_direction(self):
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a)-[]-(b) RETURN b")
        assert "Hop.direction is 'forward' or 'backward'" in str(exc.value)
        # the bare `--` spelling reaches the refusal through a different
        # tokenizer arm than `-[]-`; only the second had coverage
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a)--(b) RETURN b")
        assert "Hop.direction is 'forward' or 'backward'" in str(exc.value)

    def test_conflicting_relationship_filters_name_the_hop(self):
        """`[r:x {kind: 'y'}]` asks one edge property for two values; the
        refusal's label ('relationship at hop 0') was interpolated by
        code no test reached."""
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a)-[r:x {kind: 'y'}]->(b) RETURN b")
        assert str(exc.value).startswith("relationship at hop 0 requires")

    def test_rel_variable_reusing_a_node_variable_refused(self):
        """The already-bound check is an OR of two indexes; mutant
        _add_rel__mutmut_5 made it an AND, so a relationship reusing a
        NODE's variable slipped straight through."""
        with pytest.raises(CypherError, match="'a' is already bound"):
            tr("MATCH (a)-[a:knows]->(b) RETURN b")

    def test_anonymous_node_is_named_as_such_in_refusals(self):
        """Every message interpolating `var or '(anonymous)'` must
        actually say (anonymous) when the node has no variable -- the
        or-to-and mutants printed None instead."""
        with pytest.raises(CypherError) as exc:
            tr("MATCH (:A:B)")
        assert "node (anonymous) has multiple labels (A:B)" in str(exc.value)
        with pytest.raises(CypherError) as exc:
            tr("MATCH (:person {type: 'robot'})")
        assert "node (anonymous) requires" in str(exc.value)
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a)-[:x]->(:person {type: 'robot'})")
        assert "node (anonymous) requires" in str(exc.value)

    def test_unbounded_star_refusal_names_hop_and_rewrite(self):
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a)-[*]->(b) RETURN b")
        assert str(exc.value).startswith("hop 0: an unbounded")
        assert "*1..N" in str(exc.value)

    def test_in_is_reserved_in_relationship_variable_position(self):
        """`-[IN:knows]->` is refused (IN is a keyword there); mutant
        _parse_rel_detail__mutmut_8 lowercased the guard's argument,
        which silently un-reserved it."""
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a)-[IN:knows]->(b) RETURN b")
        assert "got 'IN'" in str(exc.value)

    def test_with_distinct_on_an_anonymous_last_node_says_to_name_it(self):
        with pytest.raises(CypherError) as exc:
            cypher_to_aggregation("MATCH (a)-[:x]->() WITH DISTINCT a RETURN count(a)")
        # parenthesized exactly: the XX-padding mutant kept the phrase
        # but broke the interpolation's delimiters
        assert "(which is anonymous -- give it a variable)" in str(exc.value)

    def test_entry_point_refusals_lead_with_the_diagnosis(self):
        """Both wrong-entry-point messages must OPEN with what the query
        does -- an XX-padded or reshuffled message still contained the
        matched fragment, so the loose pins let it drift."""
        with pytest.raises(CypherError) as exc:
            tr("CREATE (a:person {x: 1})")
        assert str(exc.value).startswith("this query writes")
        with pytest.raises(CypherError) as exc:
            cypher_to_aggregation("MATCH (a)-[]->(b) RETURN b")
        assert str(exc.value).startswith("this query has no aggregating RETURN")


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

    @pytest.mark.parametrize("query,phrase", [
        # Every token kind carries its source position, and _describe()
        # renders the offender -- the ONLY consumers are these error
        # messages, so unless one error per token kind is pinned
        # verbatim, position-dropping and description mutants all
        # survive (x__tokenize 54/74/91/110/122, _describe 1-6).
        ("MATCH (a", "expected ')' at position 8, got end of query"),
        ("MATCH (5)", "expected ')' at position 7, got 5"),
        ("MATCH ('x')", "expected ')' at position 7, got 'x'"),
        ("MATCH (a b)", "expected ')' at position 9, got 'b'"),
        ("MATCH (a) )", "unexpected ')' at position 10"),
        # path-variable detection peeks two tokens ahead; these two pin
        # which token each check reads (mutants _parse_match 3/5)
        ("MATCH foo (a)", "expected '(' at position 6, got 'foo'"),
        ("MATCH p = q", "got 'q'"),
        # the escape lookahead's boundary: a backslash-quote right at end
        # of input is still an unterminated string, not a closed one
        # (mutant x__tokenize__mutmut_43)
        ("MATCH (a {s: 'oops", "unterminated string literal at position 13"),
        ("MATCH (a {s: 'ab\\'", "unterminated string literal at position 13"),
        # a lone backslash as the LAST character: the escape lookahead
        # must treat it as a plain character and report the unterminated
        # string -- the loosened-bound mutant (x__tokenize__mutmut_42)
        # indexed one past the end and died with IndexError instead
        ("MATCH (a {s: 'ab\\", "unterminated string literal at position 13"),
    ])
    def test_error_positions_and_token_descriptions(self, query, phrase):
        with pytest.raises(CypherError) as exc:
            tr(query)
        assert phrase in str(exc.value)

    def test_no_match_clause_message_is_the_whole_message(self):
        with pytest.raises(CypherError) as exc:
            tr("RETURN 1")
        assert str(exc.value) == "query has no MATCH clause"


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

    def test_aggregation_matches_python_api_exactly(self, graph):
        """The aggregating counterpart of the contract test above -- and
        the fan-in check that matters most: two parents feed m1, and a
        per-path count would say 3 where the answer is 2."""
        from hopai import Avg, Count, Max, Min, Sum

        via_cypher = aggregate_cypher(graph, """
            MATCH (a {type: 'leaf'})-[:knows]->(m)
            WITH DISTINCT m RETURN count(m) AS n
        """)
        via_python = graph.aggregate(
            Start(where={"type": "leaf"}),
            Hop(via={"kind": "knows"}, label="m"),
            aggregates={"n": Count()},
        )
        assert via_cypher == via_python == {"n": 2}

        stats = aggregate_cypher(graph, """
            MATCH (a:leaf) RETURN count(a), avg(a.priority) AS mean,
                min(a.priority), max(a.priority), sum(DISTINCT a.priority) AS total
        """)
        assert stats == graph.aggregate(
            Start(where={"type": "leaf"}),
            aggregates={"count": Count(), "mean": Avg("priority"),
                        "min_priority": Min("priority"), "max_priority": Max("priority"),
                        "total": Sum("priority", distinct=True)},
        )
        assert stats["count"] == 4 and stats["min_priority"] == 3 and stats["max_priority"] == 15

    def test_graph_cypher_dispatches_aggregation_to_a_dict(self, graph):
        """graph.cypher() returns a Subgraph, an IngestResult, or -- new
        -- a plain dict of numbers, decided by what the query says."""
        result = graph.cypher("MATCH (a:leaf) RETURN count(a)")
        assert result == {"count": 4}


class TestOptionForwarding:
    """Every dispatching entry point takes the same keyword options as
    cypher_to_traversal(), and each must actually pass them through --
    mutation testing caught graph.cypher() variants that dropped
    **options in one branch and nothing failed, because no test passed
    an option whose loss changes the answer. node_label_key=None makes
    `(a:leaf)` match all 7 fixture nodes instead of the 4 leaves, which
    is exactly such an option."""

    def test_graph_cypher_forwards_options_to_a_traversal(self, graph):
        assert len(graph.cypher("MATCH (a:leaf) RETURN a").nodes) == 4
        assert len(graph.cypher("MATCH (a:leaf) RETURN a", node_label_key=None).nodes) == 7

    def test_graph_cypher_forwards_options_to_an_aggregation(self, graph):
        assert graph.cypher("MATCH (a:leaf) RETURN count(a)") == {"count": 4}
        assert graph.cypher("MATCH (a:leaf) RETURN count(a)",
                            node_label_key=None) == {"count": 7}

    def test_graph_cypher_forwards_options_to_a_write(self, fresh_graph):
        fresh_graph.cypher("CREATE (a:person {x: 1})", node_label_key=None)
        result = fresh_graph.traverse(Start(where={"x": 1}))
        assert result.nodes[0]["properties"] == {"x": 1}   # label ignored, no "type"

    def test_traverse_cypher_forwards_options(self, graph):
        result = traverse_cypher(graph, "MATCH (a:leaf) RETURN a", node_label_key=None)
        assert len(result.nodes) == 7

    def test_aggregate_cypher_forwards_options(self, graph):
        assert aggregate_cypher(graph, "MATCH (a:leaf) RETURN count(a)",
                                node_label_key=None) == {"count": 7}

    def test_cypher_to_aggregation_forwards_every_option(self):
        """One query whose translation visibly consumes all three
        options: without node_label_key=None the label filters, without
        edge_type_key=None the type filters, and without max_var_length
        the unbounded `*` raises."""
        start, hops, aggregates = cypher_to_aggregation(
            "MATCH (a:leaf)-[r:knows*]->(b) RETURN count(DISTINCT b)",
            node_label_key=None, edge_type_key=None, max_var_length=3,
        )
        assert start.where is None
        assert hops[0].via is None
        assert (hops[0].min_hops, hops[0].max_hops) == (1, 3)
        assert list(aggregates) == ["count"]

    def test_cypher_operations_forwards_options(self, offline_graph):
        """Same **options gap as graph.cypher() had: cypher_operations()
        dropped them (mutant xǁGraphǁcypher_operationsǁ__mutmut_3) and no
        test passed an option whose loss changes the plan."""
        with_label = offline_graph.cypher_operations("CREATE (a:person {x: 1})")
        assert with_label[0]["rows"] == [{"type": "person", "x": 1}]
        without = offline_graph.cypher_operations("CREATE (a:person {x: 1})",
                                                  node_label_key=None)
        assert without[0]["rows"] == [{"x": 1}]


class TestStrictSchema:
    """Opt-in vocabulary validation against a declared schema: with it,
    a hallucinated label is an immediate CypherError naming the defined
    vocabulary instead of a silently empty result. Validation runs over
    the TRANSLATION OUTPUT, so the front ends stay front ends."""

    @staticmethod
    def schema():
        from hopai import EdgeType, GraphSchema, NodeType, Property
        return GraphSchema(
            node_types=[NodeType("person", properties=[Property("email", "string"),
                                                       Property("age", "number")]),
                        NodeType("company", properties=[Property("name", "string")])],
            edge_types=[EdgeType("works_at", source="person", target="company",
                                 properties=[Property("since", "number")])],
        )

    def test_unknown_label_kind_and_property_are_refused_by_name(self):
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a:persn) RETURN a", schema=self.schema())
        assert "unknown label 'persn'" in str(exc.value)
        assert "company, person" in str(exc.value)
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a:person)-[:worksat]->(b:company) RETURN b", schema=self.schema())
        assert "unknown relationship kind 'worksat'" in str(exc.value)
        assert "works_at" in str(exc.value)
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a:person) WHERE a.emial = 'x' RETURN a", schema=self.schema())
        assert "unknown property ['emial'] for person" in str(exc.value)
        assert "age, email" in str(exc.value)

    def test_valid_queries_translate_identically(self):
        query = ("MATCH (a:person {email: 'a@x.com'})-[:works_at]->(b:company) "
                 "WHERE b.name = 'acme' RETURN b")
        strict = tr(query, schema=self.schema())
        assert repr(strict) == repr(tr(query))

    def test_label_less_patterns_stay_outside_the_schema(self):
        """An untyped pattern's properties are legitimately unknown to
        the schema; refusing them would forbid valid untyped queries --
        the documented limit."""
        start, _ = tr("MATCH (a {anything: 1}) RETURN a", schema=self.schema())
        assert start.where == {"anything": 1}

    def test_write_plans_are_validated_too(self):
        from hopai import cypher_to_operations
        with pytest.raises(CypherError) as exc:
            cypher_to_operations("CREATE (a:persn {email: 'x'})", schema=self.schema())
        assert "unknown label 'persn'" in str(exc.value)
        with pytest.raises(CypherError) as exc:
            cypher_to_operations(
                "MATCH (a {email: 'a'}), (b {name: 'acme'}) CREATE (a)-[:worksat]->(b)",
                schema=self.schema())
        assert "unknown relationship kind 'worksat'" in str(exc.value)

    def test_hop_position_labels_are_checked_too(self):
        """Validation walks EVERY pattern position: a hallucinated label
        on the far end of a chain is exactly as wrong as one at the
        start, and a start-only check would let it match nothing
        silently -- the failure mode strict mode exists to prevent."""
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a:person)-[:works_at]->(b:companyy) RETURN b",
               schema=self.schema())
        assert "unknown label 'companyy'" in str(exc.value)

    def test_merge_plans_are_validated_like_create(self):
        """MERGE reaches the database through the same vocabulary, node
        and relationship spellings both -- an upsert against a
        hallucinated label would silently build a parallel graph."""
        from hopai import cypher_to_operations
        with pytest.raises(CypherError) as exc:
            cypher_to_operations("MERGE (a:persn {email: 'x'})", schema=self.schema())
        assert "unknown label 'persn'" in str(exc.value)
        with pytest.raises(CypherError) as exc:
            cypher_to_operations(
                "MATCH (a {email: 'a'}), (b {name: 'acme'}) MERGE (a)-[:worksat]->(b)",
                schema=self.schema())
        assert "unknown relationship kind 'worksat'" in str(exc.value)

    def test_match_lookups_in_write_plans_are_validated(self):
        """The match ops a MATCH ... CREATE plan starts with carry label
        vocabulary of their own; leaving them unchecked would let the
        lookup half of a write hallucinate freely."""
        from hopai import cypher_to_operations
        with pytest.raises(CypherError) as exc:
            cypher_to_operations(
                "MATCH (a:persn {email: 'a'}), (b:company {name: 'acme'}) "
                "CREATE (a)-[:works_at]->(b)",
                schema=self.schema())
        assert "unknown label 'persn'" in str(exc.value)

    def test_validators_apply_the_conventional_keys_by_default(self):
        """validate_traversal/validate_operations default to the same
        type/kind discriminators the rest of the library speaks, so a
        caller validating a translation by hand needs no extra
        arguments. Mutating either default must fail here."""
        from hopai import cypher_to_operations
        from hopai.schema import validate_operations, validate_traversal
        start, hops = tr("MATCH (a:persn) RETURN a")
        with pytest.raises(CypherError, match="unknown label 'persn'"):
            validate_traversal(self.schema(), start, hops)
        start, hops = tr("MATCH (a:person)-[:worksat]->(b:company) RETURN b")
        with pytest.raises(CypherError, match="unknown relationship kind 'worksat'"):
            validate_traversal(self.schema(), start, hops)
        node_plan = cypher_to_operations("CREATE (a:persn {email: 'x'})")
        with pytest.raises(CypherError, match="unknown label 'persn'"):
            validate_operations(self.schema(), node_plan)
        edge_plan = cypher_to_operations(
            "MATCH (a {email: 'a'}), (b {name: 'acme'}) CREATE (a)-[:worksat]->(b)")
        with pytest.raises(CypherError, match="unknown relationship kind 'worksat'"):
            validate_operations(self.schema(), edge_plan)

    def test_edge_rows_without_a_property_bag_pass(self):
        """A hand-built plan may spell a kindless edge as a row with no
        properties key at all; validation treats that as an empty bag,
        never as an error about the row's shape."""
        from hopai.schema import validate_operations
        validate_operations(self.schema(), [
            {"op": "create_edges", "rows": [{"start_id": 1, "end_id": 2}]}])

    def test_several_unknown_properties_and_labels_are_listed_together(self):
        """The refusal names EVERY unknown property (plural spelling
        included) and every label whose vocabulary was searched -- an
        agent fixing one name at a time across round trips is the slow
        version of this one message."""
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a:person) WHERE a.emial = 'x' AND a.aeg = 2 RETURN a",
               schema=self.schema())
        assert "unknown properties ['aeg', 'emial'] for person" in str(exc.value)
        with pytest.raises(CypherError) as exc:
            tr("MATCH (a) WHERE a.type IN ['person', 'company'] AND a.emial = 'x' "
               "RETURN a", schema=self.schema())
        assert "unknown property ['emial'] for company/person" in str(exc.value)
        assert "age, email, name" in str(exc.value)

    def test_custom_discriminator_keys_reach_validation(self):
        """cypher_to_traversal forwards node_label_key/edge_type_key to
        the validator: with edge_type_key='rel' the kind lives under
        'rel', and validation must look there, not at the default."""
        with pytest.raises(CypherError, match="unknown relationship kind 'worksat'"):
            tr("MATCH (a:person)-[:worksat]->(b:company) RETURN b",
               schema=self.schema(), edge_type_key="rel")

    def test_graph_level_strict_flag_reaches_writes_and_aggregates(self):
        """strict_schema=True is one flag on every Graph-level front
        door -- cypher_operations and aggregate_cypher resolve it
        against THIS graph's schema exactly as traverse_cypher does."""
        from hopai import Graph
        graph = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline")
        graph.define_schema(schema=self.schema())
        with pytest.raises(CypherError, match="unknown label 'persn'"):
            graph.cypher_operations("CREATE (a:persn {email: 'x'})", strict_schema=True)
        with pytest.raises(CypherError, match="unknown label 'persn'"):
            aggregate_cypher(graph, "MATCH (a:persn) RETURN count(a)",
                             strict_schema=True)

    def test_strict_flag_without_a_schema_names_the_fix(self, offline_graph):
        with pytest.raises(CypherError) as exc:
            traverse_cypher(offline_graph, "MATCH (a:person) RETURN a", strict_schema=True)
        # verbatim: the padded-string mutants keep every inner word, so
        # a substring match cannot tell them from the real message
        assert str(exc.value) == (
            "strict_schema=True needs a schema and none is defined for this Graph "
            "-- call define_schema(...) first"
        )

    def test_aggregating_queries_validate_their_whole_chain(self):
        """The aggregation translator forwards start, hops AND both
        discriminator keys to validation. Each was droppable on its own:
        a hop-position label, a relationship kind, and a kind under a
        configured key all have to refuse here exactly as they do on the
        traversal path."""
        with pytest.raises(CypherError, match="unknown label 'companyy'"):
            cypher_to_aggregation(
                "MATCH (a:person)-[:works_at]->(b:companyy) RETURN count(DISTINCT b)",
                schema=self.schema())
        with pytest.raises(CypherError, match="unknown relationship kind 'worksat'"):
            cypher_to_aggregation(
                "MATCH (a:person)-[:worksat]->(b:company) RETURN count(DISTINCT b)",
                schema=self.schema())
        with pytest.raises(CypherError, match="unknown relationship kind 'worksat'"):
            cypher_to_aggregation(
                "MATCH (a:person)-[:worksat]->(b:company) RETURN count(DISTINCT b)",
                schema=self.schema(), edge_type_key="rel")

    def test_default_stays_permissive(self):
        """Without the flag, the exact queries strict mode refuses keep
        translating -- silently matching nothing, as before. Changing
        THAT default would break every schema-less caller."""
        start, _ = tr("MATCH (a:persn) RETURN a")
        assert start.where == {"type": "persn"}
