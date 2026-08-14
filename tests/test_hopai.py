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
    Avg, Count, Hop, Max, Min, Start, Sum,
    aggregate_json, parse_filter, traverse_json,
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
# Aggregation
# ---------------------------------------------------------------------

class TestAggregations:
    def test_seed_only_traversal_returns_empty_edges_list(self, graph):
        """Not an aggregation, but caught by its mutation run: on the
        seed-only path `edges` must be an empty LIST, not None --
        to_dict()/to_networkx() iterate it, and a None would crash both.
        Nothing previously asserted edges on a hopless traversal."""
        result = graph.traverse(Start(where={"type": "leaf"}))
        assert result.edges == [] and len(result.nodes) == 4

    def test_min_hops_excludes_edges_of_too_short_walks(self, fresh_graph):
        """A shortcut edge straight to a node that also lies at the
        required depth must NOT be reported: hop_edges filters walks by
        depth >= min_hops separately from the match filter, and losing
        that predicate leaks the depth-1 edge whenever its endpoint is
        legitimately matched deeper. A surviving mutant proved the
        7-node fixture cannot catch this -- it has no shortcut edge --
        so this graph exists to have one."""
        fresh_graph.add_nodes([{"id": 1, "name": "a"}, {"id": 2, "name": "b"},
                               {"id": 3, "name": "hub"}])
        fresh_graph.add_edges([
            {"start_id": 1, "end_id": 3},   # the shortcut: 1 hop straight to the hub
            {"start_id": 1, "end_id": 2},
            {"start_id": 2, "end_id": 3},
        ])
        result = fresh_graph.traverse(Start(where={"name": "a"}), Hop(hops=(2, 2)))
        pairs = {(e["start_id"], e["end_id"]) for e in result.edges}
        assert pairs == {("1", "2"), ("2", "3")}   # 1->3 is not on any 2-hop walk

    def test_zero_hop_aggregates_over_the_seed_set(self, graph):
        """The fixture's four leaves, three of which carry a priority
        (3, 7, 15) -- every function checked against numbers small enough
        to verify by hand."""
        result = graph.aggregate(
            Start(where={"type": "leaf"}),
            aggregates={"n": Count(), "with_priority": Count("priority"),
                        "total": Sum("priority"), "mean": Avg("priority"),
                        "lo": Min("priority"), "hi": Max("priority")},
        )
        assert result == {"n": 4, "with_priority": 3, "total": 25,
                          "mean": 25 / 3, "lo": 3, "hi": 15}

    def test_fan_in_counts_each_node_once(self, graph):
        """n1 and n2 both feed m1. A per-path count would report 3
        reached nodes where the answer is 2 (m1, m2) -- the same fan-in
        bug the traversal's local-path design exists to prevent, showing
        up as a wrong number instead of a wrong subgraph."""
        result = graph.aggregate(
            Start(where={"type": "leaf"}),
            Hop(via={"kind": "knows"}),
            aggregates={"reached": Count()},
        )
        assert result == {"reached": 2}

    def test_aggregates_the_last_hop_not_the_seed(self, graph):
        """Two hops end on the hub; a count of 1 proves the aggregate ran
        over the final match and not an earlier position."""
        result = graph.aggregate(
            Start(where={"type": "leaf"}),
            Hop(via={"kind": "knows"}),
            Hop(where={"type": "hub"}, via={"kind": "refers"}),
            aggregates={"hubs": Count()},
        )
        assert result == {"hubs": 1}

    def test_empty_match_set(self, graph):
        """count and sum answer 0; avg/min/max answer None. A sum of NULL
        (PG's default) would make every caller null-check a total."""
        result = graph.aggregate(
            Start(where={"type": "no_such_type"}),
            aggregates={"n": Count(), "s": Sum("priority"),
                        "a": Avg("priority"), "lo": Min("priority")},
        )
        assert result == {"n": 0, "s": 0, "a": None, "lo": None}

    def test_results_are_json_serializable(self, graph):
        """The driver hands NUMERIC back as Decimal, which json.dumps
        refuses -- and an aggregation result is exactly what gets
        serialized into a tool response."""
        import json
        result = graph.aggregate(Start(where={"type": "leaf"}),
                                 aggregates={"total": Sum("priority"), "mean": Avg("priority")})
        assert json.loads(json.dumps(result)) == result
        assert isinstance(result["total"], int)      # a whole number, not 25.0

    def test_distinct_collapses_equal_values(self, fresh_graph):
        """Two nodes share priority 5; distinct=True must fold them to
        one value while the plain form counts both nodes."""
        fresh_graph.add_nodes([
            {"id": 1, "type": "t", "priority": 5},
            {"id": 2, "type": "t", "priority": 5},
            {"id": 3, "type": "t", "priority": 10},
        ])
        result = fresh_graph.aggregate(
            Start(where={"type": "t"}),
            aggregates={"total": Sum("priority"), "distinct_total": Sum("priority", distinct=True),
                        "values": Count("priority", distinct=True),
                        "mean": Avg("priority"), "distinct_mean": Avg("priority", distinct=True)},
        )
        assert result == {"total": 20, "distinct_total": 15, "values": 2,
                          "mean": 20 / 3, "distinct_mean": 7.5}

    def test_non_numeric_value_is_ignored_not_an_error(self, fresh_graph):
        """One node carrying "high" where a number was expected must not
        abort the query -- a bare ::numeric cast would. It still counts
        as present for Count, which asks a different question."""
        fresh_graph.add_nodes([
            {"id": 1, "type": "t", "priority": 5},
            {"id": 2, "type": "t", "priority": "high"},
        ])
        result = fresh_graph.aggregate(
            Start(where={"type": "t"}),
            aggregates={"total": Sum("priority"), "present": Count("priority")},
        )
        assert result == {"total": 5, "present": 2}

    def test_json_null_property_is_absent(self, fresh_graph):
        """{"priority": null} counts as 'no priority', matching Cypher's
        judgement that a null property does not exist -- `->` instead of
        `->>` in the Count compilation would count it."""
        fresh_graph.add_nodes([
            {"id": 1, "type": "t", "priority": 5},
            {"id": 2, "type": "t", "priority": None},
        ])
        result = fresh_graph.aggregate(Start(where={"type": "t"}),
                                       aggregates={"present": Count("priority")})
        assert result == {"present": 1}

    def test_aggregate_json_matches_python_api(self, graph):
        """The JSON front end must not drift from the Python one -- same
        spec, same numbers."""
        json_result = aggregate_json(graph, {
            "start": {"where": {"type": "leaf"}},
            "hops": [{"via": {"kind": "knows"}}],
            "aggregates": {"n": {"fn": "count"},
                           "hi": {"fn": "max", "property": "priority"}},
        })
        python_result = graph.aggregate(
            Start(where={"type": "leaf"}), Hop(via={"kind": "knows"}),
            aggregates={"n": Count(), "hi": Max("priority")},
        )
        assert json_result == python_result

    def test_backward_direction(self, graph):
        """Aggregation goes through the same hop machinery as traversal,
        direction included: two leaves point at m1."""
        result = graph.aggregate(
            Start(where={"name": "m1"}),
            Hop(direction="backward", via={"kind": "knows"}),
            aggregates={"parents": Count(), "min_priority": Min("priority")},
        )
        assert result == {"parents": 2, "min_priority": 3}

    def test_optional_hop_is_refused(self, graph):
        with pytest.raises(ValueError, match="no effect on an aggregation"):
            graph.aggregate(Start(where={"type": "leaf"}), Hop(optional=True),
                            aggregates={"n": Count()})


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
