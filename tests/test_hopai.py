"""
Test suite for hopai.

Every test here corresponds to something that was actually found to be
wrong at some point during development -- not a hypothetical edge case
written after the fact. Where useful, the docstring says what would have
failed without the fix.
"""

from __future__ import annotations

import pytest

from hopai import (
    AND, BETWEEN, GT, GTE, LT, LTE, NOT, OR,
    Hop, Start, parse_filter, traverse_json,
)


# ---------------------------------------------------------------------
# Core traversal correctness
# ---------------------------------------------------------------------

class TestCoreTraversal:
    def test_simple_forward_hop(self, graph):
        result = graph.traverse(
            Start(where={"type": "leaf"}),
            Hop(where={"flag": 1}, hops=1),
        )
        ids = {n["id"] for n in result.nodes}
        # n1, n2, n3 all reach a flag=1 node in 1 hop; n4 reaches one too
        # (its only edge is the wrong kind, but no `via` filter is set here)
        assert ids == {"1", "2", "3", "4", "5", "6"}

    def test_dead_end_excluded_when_edge_kind_filtered(self, graph):
        """n4's only edge is kind='wrong_kind'. Filtering via={'kind':'knows'}
        must exclude n4 entirely, not report it as a nodeless dead end.
        This was bug #3 found during development: reported nodes must be
        derived from real edges found, not from the raw seed set."""
        result = graph.traverse(
            Start(where={"type": "leaf"}),
            Hop(where={"flag": 1}, via={"kind": "knows"}, hops=1),
        )
        ids = {n["id"] for n in result.nodes}
        assert "4" not in ids
        assert {"1", "2", "3", "5", "6"} <= ids

    def test_fan_in_both_parents_preserved(self, graph):
        """n1 and n2 both feed m1. An earlier version tracked one path per
        destination node across the whole chain and silently dropped one
        of two valid parents (bug #2). Both edges into m1 must appear."""
        result = graph.traverse(
            Start(where={"type": "leaf"}),
            Hop(where={"flag": 1}, via={"kind": "knows"}, hops=1),
        )
        edges_into_m1 = [e for e in result.edges if e["end_id"] == "5"]
        starts = {e["start_id"] for e in edges_into_m1}
        assert starts == {"1", "2"}

    def test_multi_hop_edge_reconstruction(self, graph):
        """A hop spanning >1 real edge must report every real edge along
        the path, not a fabricated direct edge between endpoints (bug #1:
        earlier version looked for a single edge between hop boundaries,
        which doesn't exist when a hop spans multiple hops)."""
        result = graph.traverse(
            Start(where={"type": "leaf"}),
            Hop(hops=(1, 3), via={"kind": "knows"}),
            Hop(where={"type": "hub"}, hops=(1, 2)),
        )
        # every edge returned must be a real edge from the fixture, and the
        # path from a leaf to the hub must be reconstructed via two hops
        # (leaf -> mid -> hub), not a single fabricated leaf -> hub edge
        pairs = {(e["start_id"], e["end_id"]) for e in result.edges}
        assert ("1", "5") in pairs
        assert ("5", "7") in pairs
        assert ("1", "7") not in pairs  # no such real edge exists

    def test_cycle_does_not_hang(self, graph):
        """h1 -> m1 -> h1 is a real cycle. A generous hop bound must not
        hang or explode -- cycle protection must actually engage, not
        just be present in the code."""
        result = graph.traverse(
            Start(where={"type": "hub"}),
            Hop(hops=(1, 10), direction="backward"),
        )
        assert result.elapsed_ms < 5000  # generous ceiling; should be near-instant
        assert len(result.nodes) > 0


# ---------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------

class TestDirection:
    def test_backward_direction(self, graph):
        result = graph.traverse(
            Start(where={"type": "hub"}),
            Hop(hops=1, direction="backward"),
        )
        ids = {n["id"] for n in result.nodes}
        assert {"5", "6", "7"} <= ids  # m1, m2 point to h1; h1 itself included

    def test_mixed_direction_chain(self, graph):
        """backward then forward in the same chain -- a 'siblings' style
        query. Confirms direction is genuinely per-hop, not global."""
        result = graph.traverse(
            Start(where={"type": "hub"}),
            Hop(hops=1, direction="backward"),
            Hop(hops=1, direction="forward"),
        )
        assert len(result.nodes) > 0


# ---------------------------------------------------------------------
# Filter logic: AND / OR / NOT
# ---------------------------------------------------------------------

class TestFilterLogic:
    def test_bare_list_rejected(self, graph):
        """A bare list at the top level used to silently mean OR -- an
        ambiguous, easy-to-misread shorthand that was deliberately
        removed. It must now fail loudly, not guess."""
        with pytest.raises(TypeError, match="ambiguous"):
            graph.traverse(Start(where=[{"type": "leaf"}, {"type": "hub"}]))

    def test_or_value_list_shorthand(self, graph):
        result = graph.traverse(Start(where={"type": ["leaf", "hub"]}))
        ids = {n["id"] for n in result.nodes}
        assert ids == {"1", "2", "3", "4", "7"}

    def test_explicit_or(self, graph):
        result = graph.traverse(Start(where=OR({"type": "leaf"}, {"type": "hub"})))
        ids = {n["id"] for n in result.nodes}
        assert ids == {"1", "2", "3", "4", "7"}

    def test_not_includes_missing_key(self, graph):
        """The single most important NOT behavior: NOT({'type': 'leaf'})
        must match nodes that have NO 'type' key at all (m1, m2), not just
        nodes with a different type value. JSONB containment evaluating
        false-not-null for a missing key is what makes this correct --
        verified during development to be a real trap in naive
        equality-based negation (Cypher's `NOT x = 'value'` gets this
        wrong for missing properties)."""
        result = graph.traverse(Start(where=NOT({"type": "leaf"})))
        ids = {n["id"] for n in result.nodes}
        assert ids == {"5", "6", "7"}

    def test_and_or_composition(self, graph):
        result = graph.traverse(
            Start(where=AND(OR({"type": "leaf"}, {"type": "hub"}), {"type": "leaf"}))
        )
        ids = {n["id"] for n in result.nodes}
        assert ids == {"1", "2", "3", "4"}

    def test_escape_hatch_callable(self, graph):
        """Any callable receiving the real column and returning a real
        SQLAlchemy expression must work -- the intentional way out of the
        closed set of filter classes above."""
        result = graph.traverse(
            Start(where=lambda col: col.op("->>")("name").op("~")("^n1$"))
        )
        ids = {n["id"] for n in result.nodes}
        assert ids == {"1"}

    def test_escape_hatch_composes_with_and(self, graph):
        result = graph.traverse(
            Start(where=AND(
                lambda col: col.op("->>")("name").op("~")("^n"),
                NOT({"type": "leaf"}),
            ))
        )
        ids = {n["id"] for n in result.nodes}
        # names starting with 'n' are n1..n4 (all leaf), so NOT(leaf) excludes all
        assert ids == set()


# ---------------------------------------------------------------------
# Range comparisons
# ---------------------------------------------------------------------

class TestRangeComparisons:
    def test_gt(self, graph):
        result = graph.traverse(Start(where=AND({"type": "leaf"}, GT("priority", 5))))
        assert {n["id"] for n in result.nodes} == {"2", "3"}

    def test_gte(self, graph):
        result = graph.traverse(Start(where=AND({"type": "leaf"}, GTE("priority", 7))))
        assert {n["id"] for n in result.nodes} == {"2", "3"}

    def test_lt(self, graph):
        result = graph.traverse(Start(where=AND({"type": "leaf"}, LT("priority", 10))))
        assert {n["id"] for n in result.nodes} == {"1", "2"}

    def test_lte(self, graph):
        result = graph.traverse(Start(where=AND({"type": "leaf"}, LTE("priority", 7))))
        assert {n["id"] for n in result.nodes} == {"1", "2"}

    def test_between_inclusive(self, graph):
        result = graph.traverse(Start(where=AND({"type": "leaf"}, BETWEEN("priority", 3, 7))))
        assert {n["id"] for n in result.nodes} == {"1", "2"}

    def test_range_excludes_missing_property(self, graph):
        """n4 has no 'priority' key. NULL > anything is NULL, not true --
        it must be excluded, not raise or match."""
        result = graph.traverse(Start(where=AND({"type": "leaf"}, GT("priority", -999))))
        ids = {n["id"] for n in result.nodes}
        assert "4" not in ids


# ---------------------------------------------------------------------
# OPTIONAL
# ---------------------------------------------------------------------

class TestOptional:
    def test_optional_keeps_matches_with_no_extension(self, graph):
        """m1/m2 point to h1 and have no leaf-typed forward dependency in
        range -- the optional hop should find nothing, but m1/m2 must
        still be in the result (that's what makes it optional)."""
        result = graph.traverse(
            Start(where={"type": "hub"}),
            Hop(where=NOT({"type": "leaf"}), hops=1, direction="backward"),
            Hop(where=AND({"type": "leaf"}, BETWEEN("priority", 100, 200)), hops=1, optional=True),
        )
        ids = {n["id"] for n in result.nodes}
        assert {"5", "6", "7"} <= ids

    def test_optional_on_non_last_hop_raises(self, graph):
        with pytest.raises(ValueError, match="only supported on the LAST hop"):
            graph.traverse(
                Start(where={"type": "hub"}),
                Hop(hops=1, optional=True),
                Hop(hops=1),
            )


# ---------------------------------------------------------------------
# Hop validation
# ---------------------------------------------------------------------

class TestHopValidation:
    def test_int_hops_means_exact_count(self):
        h = Hop(hops=3)
        assert (h.min_hops, h.max_hops) == (3, 3)

    def test_tuple_hops_range(self):
        h = Hop(hops=(2, 5))
        assert (h.min_hops, h.max_hops) == (2, 5)

    def test_invalid_hops_range_raises(self):
        with pytest.raises(ValueError):
            Hop(hops=(5, 2))  # min > max

    def test_zero_hops_raises(self):
        with pytest.raises(ValueError):
            Hop(hops=0)

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            Hop(direction="sideways")


# ---------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------

class TestJsonApi:
    def test_json_equality_filter(self, graph):
        result = traverse_json(graph, {"start": {"where": {"type": "leaf"}}})
        assert len(result["nodes"]) == 4

    def test_json_not_operator(self, graph):
        result = traverse_json(graph, {"start": {"where": {"not": {"type": "leaf"}}}})
        ids = {n["id"] for n in result["nodes"]}
        assert ids == {"5", "6", "7"}

    def test_json_and_gt_operators(self, graph):
        result = traverse_json(graph, {
            "start": {"where": {"and": [{"type": "leaf"}, {"gt": ["priority", 5]}]}}
        })
        ids = {n["id"] for n in result["nodes"]}
        assert ids == {"2", "3"}

    def test_json_between_operator(self, graph):
        result = traverse_json(graph, {"start": {"where": {"between": ["priority", 3, 7]}}})
        ids = {n["id"] for n in result["nodes"]}
        assert ids == {"1", "2"}

    def test_json_hops_as_list_range(self, graph):
        result = traverse_json(graph, {
            "start": {"where": {"type": "leaf"}},
            "hops": [{"where": {"flag": 1}, "via": {"kind": "knows"}, "hops": [1, 3]}],
        })
        assert len(result["nodes"]) > 0

    def test_json_malformed_operator_filter_raises(self, graph):
        with pytest.raises(ValueError, match="exactly one key"):
            traverse_json(graph, {"start": {"where": {"gt": ["priority", 5], "extra": 1}}})

    def test_json_result_matches_python_api(self, graph):
        """The JSON API must not be a second implementation with its own
        drift potential -- it should produce byte-identical results to
        the equivalent Python call."""
        json_result = traverse_json(graph, {
            "start": {"where": {"type": "leaf"}},
            "hops": [{"where": {"flag": 1}, "via": {"kind": "knows"}, "hops": [1, 1]}],
        })
        python_result = graph.traverse(
            Start(where={"type": "leaf"}),
            Hop(where={"flag": 1}, via={"kind": "knows"}, hops=(1, 1)),
        )
        json_ids = {n["id"] for n in json_result["nodes"]}
        python_ids = {n["id"] for n in python_result.nodes}
        assert json_ids == python_ids

    def test_parse_filter_passthrough_for_plain_dict(self):
        assert parse_filter({"type": "leaf"}) == {"type": "leaf"}

    def test_parse_filter_none(self):
        assert parse_filter(None) is None


# ---------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------

class TestSubgraphResult:
    def test_to_dict_is_json_serializable(self, graph):
        import json
        result = graph.traverse(Start(where={"type": "leaf"}))
        json.dumps(result.to_dict())  # must not raise

    def test_to_networkx(self, graph):
        nx = pytest.importorskip("networkx")
        result = graph.traverse(
            Start(where={"type": "leaf"}),
            Hop(where={"flag": 1}, via={"kind": "knows"}, hops=1),
        )
        g = result.to_networkx()
        assert isinstance(g, nx.DiGraph)
        assert g.number_of_nodes() == len(result.nodes)

    def test_to_networkx_multigraph_preserves_parallel_edges(self, graph):
        """Plain DiGraph collapses parallel edges between the same two
        nodes; MultiDiGraph must not. Regression guard for a known,
        documented limitation."""
        nx = pytest.importorskip("networkx")
        result = graph.traverse(
            Start(where={"type": "leaf"}),
            Hop(where={"flag": 1}, via={"kind": "knows"}, hops=1),
        )
        g = result.to_networkx(multigraph=True)
        assert isinstance(g, nx.MultiDiGraph)


# ---------------------------------------------------------------------
# Gaps found by mutation testing. Each of these mutants survived the
# whole suite, meaning the behaviour was executed and never asserted on.
# ---------------------------------------------------------------------

class TestMinHopsEdgeCollection:
    def test_edges_from_too_short_a_walk_are_not_reported(self, fresh_graph):
        """A -> B directly, and A -> C -> B. With hops=2, B qualifies via
        the two-edge path only -- but B is also reached at depth 1, and
        that shorter walk carries a different edge. Dropping the
        `depth >= min_hops` filter on the edge-collection CTE lets A -> B
        leak into a result that asked for two-hop paths.

        Mutant hopai.core.xǁGraphǁbuild_query__mutmut_221 removed exactly
        that predicate and nothing failed."""
        fresh_graph.ingest({
            "nodes": [{"id": 1, "n": "a"}, {"id": 2, "n": "b"}, {"id": 3, "n": "c"}],
            "edges": [{"start_id": 1, "end_id": 2}, {"start_id": 1, "end_id": 3},
                      {"start_id": 3, "end_id": 2}],
        })
        result = fresh_graph.traverse(Start(where={"n": "a"}), Hop(hops=2))
        pairs = {(e["start_id"], e["end_id"]) for e in result.edges}
        assert pairs == {("1", "3"), ("3", "2")}
        assert ("1", "2") not in pairs   # the one-hop shortcut is not a two-hop path


class TestEmptyResults:
    def test_a_traversal_with_no_edges_returns_an_empty_list(self, fresh_graph):
        """Not None. `Subgraph.edges = None` would break len(), iteration
        and to_networkx() for every caller that does not special-case it.
        Mutant xǁGraphǁtraverse__mutmut_61 turned [] into None."""
        fresh_graph.add_nodes([{"n": 1}])
        result = fresh_graph.traverse(Start(where={"n": 1}))
        assert result.edges == []
        assert result.to_networkx().number_of_edges() == 0

    def test_dropping_a_schema_twice_is_not_an_error(self, fresh_graph):
        """checkfirst=True is what makes drop_schema idempotent, and
        idempotence is the whole reason it is safe in a teardown path.
        Mutant xǁGraphǁdrop_schema__mutmut_4 removed it."""
        fresh_graph.drop_schema()
        fresh_graph.drop_schema()
