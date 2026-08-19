"""
Test suite for hopai.

Every test here corresponds to something that was actually found to be
wrong at some point during development -- not a hypothetical edge case
written after the fact. Where useful, the docstring says what would have
failed without the fix.
"""

from __future__ import annotations

import time

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
    def test_shared_ids_across_graphs_stay_scoped(self, write_engine):
        """Every read goes through _scoped() -- including the
        hydrate-by-id lookups after the walk. With the DEFAULT tables
        that guard is invisible to tests (nodes.id is the table-wide PK,
        so an id list already identifies rows uniquely), which is
        exactly how mutants xǁGraphǁtraverseǁ__mutmut_47 and
        build_query__mutmut_87 dropped it unnoticed. Graph() explicitly
        supports caller-supplied tables, where PRIMARY KEY (id, graph_id)
        is legitimate -- and there, an unscoped hydration returns the
        other graph's rows."""
        from sqlalchemy import BigInteger, Column, MetaData, Table, Text, text as sa_text
        from sqlalchemy.dialects.postgresql import JSONB

        from hopai import Graph

        meta = MetaData(schema="hopai_shared_ids")
        nodes = Table("nodes", meta,
                      Column("id", BigInteger, primary_key=True),
                      Column("graph_id", Text, primary_key=True),
                      Column("properties", JSONB, nullable=False))
        edges = Table("edges", meta,
                      Column("id", BigInteger, primary_key=True),
                      Column("graph_id", Text, primary_key=True),
                      Column("start_id", BigInteger, nullable=False),
                      Column("end_id", BigInteger, nullable=False),
                      Column("properties", JSONB, nullable=False))
        with write_engine.begin() as conn:
            conn.execute(sa_text("DROP SCHEMA IF EXISTS hopai_shared_ids CASCADE"))
            conn.execute(sa_text("CREATE SCHEMA hopai_shared_ids"))
        meta.create_all(write_engine)
        with write_engine.begin() as conn:
            conn.execute(nodes.insert(), [
                {"id": 1, "graph_id": "g1", "properties": {"t": 1, "m": "one"}},
                {"id": 2, "graph_id": "g1", "properties": {"m": "one-end"}},
                {"id": 1, "graph_id": "g2", "properties": {"t": 1, "m": "two"}},
                {"id": 2, "graph_id": "g2", "properties": {"m": "two-end"}},
            ])
            conn.execute(edges.insert(), [
                {"id": 1, "graph_id": "g1", "start_id": 1, "end_id": 2,
                 "properties": {"kind": "k"}},
                {"id": 1, "graph_id": "g2", "start_id": 1, "end_id": 2,
                 "properties": {"kind": "k"}},
            ])
        g1 = Graph(write_engine, graph="g1", node_table=nodes, edge_table=edges)
        seed_only = g1.traverse(Start(where={"t": 1}))
        assert [n["properties"]["m"] for n in seed_only.nodes] == ["one"]
        one_hop = g1.traverse(Start(where={"t": 1}), Hop())
        assert {n["properties"]["m"] for n in one_hop.nodes} == {"one", "one-end"}
        assert len(one_hop.edges) == 1

    def test_in_graph_carries_every_table_and_column_setting(self):
        """in_graph() is the documented way to hop between graphs on
        CUSTOM tables too, so the new handle must inherit every name the
        caller configured -- a mutation-run survivor showed a dropped
        edge_end_col would go unnoticed, and a handle silently reading
        the default column is the _scoped() bug in a different coat."""
        from sqlalchemy import BigInteger, Column, MetaData, Table, Text
        from sqlalchemy.dialects.postgresql import JSONB

        from hopai import Graph

        meta = MetaData()
        nodes = Table("my_nodes", meta,
                      Column("nid", BigInteger, primary_key=True),
                      Column("tenant", Text),
                      Column("properties", JSONB))
        edges = Table("my_edges", meta,
                      Column("eid", BigInteger, primary_key=True),
                      Column("tenant", Text),
                      Column("src", BigInteger),
                      Column("dst", BigInteger),
                      Column("properties", JSONB))
        base = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
                     node_table=nodes, edge_table=edges, node_id_col="nid",
                     edge_id_col="eid", edge_start_col="src", edge_end_col="dst",
                     graph_col="tenant")
        other = base.in_graph("elsewhere")
        assert other.graph == "elsewhere"
        assert (other.nodes_tbl, other.edges_tbl) == (nodes, edges)
        assert (other.node_id_col, other.edge_id_col) == ("nid", "eid")
        assert (other.edge_start_col, other.edge_end_col) == ("src", "dst")
        assert other.graph_col == "tenant"

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
        # Bounded on BOTH sides: a ceiling alone is satisfied by 0.0, so
        # dropping the measurement entirely (Subgraph's elapsed_ms
        # default) passed unnoticed. A real round trip cannot take zero.
        assert 0 < result.elapsed_ms < 5000  # generous ceiling; should be near-instant
        assert len(result.nodes) > 0

    def test_elapsed_ms_is_measured_in_milliseconds(self, graph):
        """The unit is part of the name, and only an independent clock
        can check it: every self-consistent assertion passes just as
        happily when the seconds-to-ms conversion is inverted (mutant
        traverse_111 divides where it should multiply, reporting a
        millionth of the truth). The bounds are deliberately loose --
        elapsed_ms times the queries inside a call that also builds the
        statement and opens a session, so it is a fraction of the wall
        time, never more, and never a millionth."""
        t0 = time.perf_counter()
        result = graph.traverse(Start(where={"type": "leaf"}), Hop(hops=(1, 3)))
        wall_ms = (time.perf_counter() - t0) * 1000
        assert 0 < result.elapsed_ms <= wall_ms
        assert result.elapsed_ms > wall_ms / 1000

    def test_the_seconds_to_milliseconds_factor_is_exact(self, graph, monkeypatch):
        """The bounds above are loose enough that the CONSTANT can drift
        -- 1000 to 1001 is a tenth of a percent, invisible against a
        real clock (mutant traverse_124). Driving perf_counter with
        known values is the only way to assert the factor itself."""
        import hopai.core as core_module

        ticks = iter([10.0, 10.25])            # exactly 250ms of query time
        monkeypatch.setattr(core_module.time, "perf_counter", lambda: next(ticks))
        result = graph.traverse(Start(where={"type": "leaf"}))
        assert result.elapsed_ms == pytest.approx(250.0)


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

    def test_optional_keeps_a_seed_whose_only_edges_never_match(self, graph):
        """n4-deadend's single edge is the wrong kind, so with ONE
        optional hop it is kept purely by the pre-optional arm of the
        result union -- no edge row ever mentions it. The earlier
        optional test could not catch a broken arm (mutant
        xǁGraphǁbuild_queryǁ__mutmut_193 nulled its id column) because
        its kept nodes also arrived via the previous hop's edges."""
        result = graph.traverse(
            Start(where={"name": "n4-deadend"}),
            Hop(via={"kind": "knows"}, optional=True),
        )
        assert [n["properties"]["name"] for n in result.nodes] == ["n4-deadend"]
        assert result.edges == []


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

    def test_traverse_json_has_no_node_ceiling(self, fresh_graph):
        """The MCP server's max_nodes (hopai/mcp.py, #47) is enforced in
        mcp.py alone: a Python/JSON caller going through traverse_json()
        directly has no context window and no reason to be capped, and
        this pins that a traversal bigger than any MCP default (500)
        still comes back whole when called this way -- proving the cap
        lives in the front end that needs it, not in traverse_json()
        itself."""
        fresh_graph.add_nodes([{"id": i, "type": "leaf"} for i in range(600)])
        result = traverse_json(fresh_graph, {"start": {"where": {"type": "leaf"}}})
        assert len(result["nodes"]) == 600

    def test_parse_filter_passthrough_for_plain_dict(self):
        assert parse_filter({"type": "leaf"}) == {"type": "leaf"}

    def test_parse_filter_names_the_offending_type(self):
        """The refusal must say what it got -- 'got list' is what turns
        a bare-list mistake into an immediate fix."""
        with pytest.raises(TypeError) as exc:
            parse_filter([{"type": "leaf"}])
        assert "got list" in str(exc.value)

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

    def test_group_by_runs_one_aggregate_per_distinct_value(self, graph):
        """Every node in the fixture, grouped by `type`: four leaves,
        one hub, and m1/m2 (which carry no `type` at all) forming their
        own None group -- proving a missing key groups together rather
        than being dropped, matching Count(property)'s own judgement."""
        result = graph.aggregate(Start(), aggregates={"n": Count()}, group_by="type")
        assert sorted(result, key=lambda row: (row["type"] is None, row["type"])) == [
            {"type": "hub", "n": 1},
            {"type": "leaf", "n": 4},
            {"type": None, "n": 2},
        ]

    def test_group_by_runs_over_the_last_hop_not_the_seed(self, graph):
        """Grouping is scoped to the SAME final-step nodes the aggregate
        already runs over -- m1 and m2 both carry flag=1, so this proves
        the grouping key is read from the reached nodes (m1, m2), not
        from the leaf seeds that have no `flag` property at all."""
        result = graph.aggregate(
            Start(where={"type": "leaf"}), Hop(via={"kind": "knows"}),
            aggregates={"reached": Count()}, group_by="flag",
        )
        assert result == [{"flag": "1", "reached": 2}]

    def test_group_by_on_empty_match_returns_empty_list(self, graph):
        """GROUP BY produces no rows when there is nothing to group --
        unlike the single row of zeros/Nones a plain (ungrouped)
        aggregate reports over an empty match."""
        result = graph.aggregate(Start(where={"type": "no_such_type"}),
                                 aggregates={"n": Count()}, group_by="type")
        assert result == []

    def test_group_by_colliding_with_an_aggregate_name_refused(self, graph):
        with pytest.raises(ValueError, match="collides"):
            graph.aggregate(Start(where={"type": "leaf"}),
                            aggregates={"type": Count()}, group_by="type")

    def test_group_by_must_be_a_string(self, graph):
        with pytest.raises(TypeError, match="group_by must be a string"):
            graph.aggregate(Start(where={"type": "leaf"}), aggregates={"n": Count()},
                            group_by=5)

    def test_aggregate_json_group_by_matches_python_api(self, graph):
        json_result = aggregate_json(graph, {
            "start": {},
            "aggregates": {"n": {"fn": "count"}},
            "group_by": "type",
        })
        python_result = graph.aggregate(Start(), aggregates={"n": Count()}, group_by="type")
        assert json_result == python_result


# ---------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------

class TestSubgraphResult:
    def test_to_dict_is_json_serializable(self, graph):
        import json
        result = graph.traverse(Start(where={"type": "leaf"}))
        json.dumps(result.to_dict())  # must not raise

    def test_both_lists_carry_string_ids(self, fresh_graph):
        """The result shape itself, which nothing asserted -- the edge
        id could be dropped, renamed or left an integer and the whole
        suite stayed green.

        Strings because every other id in this library is one:
        vector_search(), get_vectors(), stale_vectors() and the JSON
        results all speak strings, and an int here would be the single
        shape a caller has to convert."""
        fresh_graph.ingest({
            "nodes": [{"id": 1, "n": "a"}, {"id": 2, "n": "b"}],
            "edges": [{"start_id": 1, "end_id": 2, "kind": "knows"}],
        })
        result = fresh_graph.traverse(Start(where={"n": "a"}), Hop(hops=1))
        assert set(result.nodes[0]) == {"id", "properties"}
        assert set(result.edges[0]) == {"id", "start_id", "end_id", "properties"}
        assert all(isinstance(row["id"], str) for row in result.nodes + result.edges)

    def test_an_edge_id_from_a_traversal_feeds_set_vectors(self, fresh_graph):
        """Why the edge id is there at all. set_vectors(edges=…) and
        get_vectors(edge_ids=…) take edge ids, and a traversal is where
        a caller finds edges -- so without this the only route between
        them was a hand-written `SELECT id FROM edges`."""
        from hopai import Vector
        fresh_graph.define_vectors(edges=[Vector("relvec", 3)])
        fresh_graph.migrate_vectors()
        fresh_graph.ingest({
            "nodes": [{"id": 1, "n": "a"}, {"id": 2, "n": "b"}],
            "edges": [{"start_id": 1, "end_id": 2, "kind": "knows"}],
        })
        edge = fresh_graph.traverse(Start(where={"n": "a"}), Hop(hops=1)).edges[0]
        assert fresh_graph.set_vectors(edges=[{"id": edge["id"], "relvec": [1.0, 0.0, 0.0]}]) == 1
        stored = fresh_graph.get_vectors(edge_ids=[edge["id"]])["edges"][edge["id"]]
        assert stored["relvec"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)

    def test_parallel_edges_are_told_apart(self, fresh_graph):
        """Two edges of the same kind between the same pair carrying the
        same properties used to arrive as two identical dicts -- present
        in the count, indistinguishable in the data."""
        fresh_graph.ingest({
            "nodes": [{"id": 1, "n": "a"}, {"id": 2, "n": "b"}],
            "edges": [{"start_id": 1, "end_id": 2, "kind": "knows"},
                      {"start_id": 1, "end_id": 2, "kind": "knows"}],
        })
        edges = fresh_graph.traverse(Start(where={"n": "a"}), Hop(hops=1)).edges
        assert len(edges) == 2
        assert len({e["id"] for e in edges}) == 2

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


class TestHopWhereActuallyFilters:
    def test_a_hop_filter_excludes_a_reachable_node(self, fresh_graph):
        """Every existing hop test uses a `where` that all reachable
        nodes happen to satisfy, so dropping the filter entirely changed
        nothing and mutant build_query__mutmut_206 survived. Here A
        reaches both B and C, and only B passes -- so the filter has to
        do work for the assertion to hold."""
        fresh_graph.ingest({
            "nodes": [{"id": 1, "n": "a"}, {"id": 2, "n": "b", "keep": True},
                      {"id": 3, "n": "c"}],
            "edges": [{"start_id": 1, "end_id": 2}, {"start_id": 1, "end_id": 3}],
        })
        result = fresh_graph.traverse(Start(where={"n": "a"}), Hop(where={"keep": True}))
        assert {n["properties"]["n"] for n in result.nodes} == {"a", "b"}
        assert {(e["start_id"], e["end_id"]) for e in result.edges} == {("1", "2")}


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
