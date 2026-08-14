"""
Tests that need no database.

`create_engine()` does not connect and query building never executes, so
everything up to the compiled SQL is testable anywhere. That covers more
than it sounds like: the recursive CTE's structure, the cycle guard, hop
bounds, custom table/column names, every filter's compiled form, the
equivalence of the Python and JSON filter grammars, and whether the docs
tell the truth.

Assertions are made against normalized SQL text. That is deliberately a
white-box test -- it will notice a rewrite of build_query that changes
the emitted query, which is exactly the review signal wanted here, since
the alternative (only checking rows come back) cannot tell a fast query
from a slow one that returns the same answer.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from hopai import (
    AGGREGATE_TOOL_SCHEMA, AND, BETWEEN, GT, GTE, LT, LTE, NOT, OR,
    Avg, Count, Graph, Hop, Max, Min, Start, Subgraph, Sum, TRAVERSE_TOOL_SCHEMA,
    parse_aggregate, parse_filter, spec_to_aggregation, spec_to_traversal,
)
from hopai.filters import resolve
from hopai.models import Node


def norm(statement, literal_binds: bool = False) -> str:
    """Compiled SQL as one whitespace-normalized line."""
    kwargs = {"compile_kwargs": {"literal_binds": True}} if literal_binds else {}
    return " ".join(str(statement.compile(dialect=postgresql.dialect(), **kwargs)).split())


def filter_sql(filt, literal_binds: bool = False) -> str:
    nt = Node.__table__
    return norm(select(nt.c.id).where(resolve(nt.c.properties, filt)), literal_binds)


# ---------------------------------------------------------------------
# The recursive CTE's shape
# ---------------------------------------------------------------------

class TestQueryStructure:
    def test_seed_only_query_is_not_recursive(self, offline_graph):
        """No hops means no walk at all -- emitting RECURSIVE here would
        be dead weight on the simplest possible query."""
        sql = norm(offline_graph.build_query(Start(where={"type": "person"}), []))
        assert "WITH seed AS" in sql
        assert "RECURSIVE" not in sql
        assert "walk_0" not in sql

    def test_seed_only_query_is_graph_scoped(self, offline_graph):
        """The no-hops branch hydrates by joining `nodes` back to `seed`,
        and that join needs the discriminator like every other table
        access -- the aggregate path has the same proof. It is only
        *currently* redundant because `id` is the table's primary key
        and therefore globally unique; a caller's own tables keyed
        (id, graph_id) would return another graph's row instead. A
        mutant that dropped this predicate survived the whole suite."""
        sql = norm(offline_graph.build_query(Start(where={"type": "person"}), []),
                   literal_binds=True)
        assert sql.count("graph_id = 'default'") == 2   # seed CTE, and the join back

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
    def test_one_cte_per_hop(self, offline_graph, n):
        """Chains longer than two hops used to raise AttributeError:
        the per-hop edge CTEs were unioned by folding .union() over the
        result, and Select.union() returns a CompoundSelect, which has no
        .union(). Nothing in the suite exercised a third hop, so every
        chain longer than two was unrunnable."""
        sql = norm(offline_graph.build_query(Start(), [Hop(hops=i + 1) for i in range(n)]))
        for i in range(n):
            assert f"walk_{i}" in sql
            assert f"match_{i}" in sql
            assert f"hop_edges_{i}" in sql
        assert f"walk_{n}" not in sql
        # "all_edges AS", not just "all_edges": a mutant renamed the CTE
        # to XXall_edgesXX and the bare substring check still matched.
        assert "all_edges AS" in sql and "edge_rows AS" in sql
        assert sql.count("UNION ALL") == n           # one recursive term per hop
        # (n-1) to fold the per-hop edge CTEs together, then 2 to union
        # the node and edge result rows
        assert sql.count(" UNION ") - sql.count(" UNION ALL") == n + 1

    def test_single_statement_one_round_trip(self, offline_graph):
        """The whole traversal is one statement -- the claim the README
        makes. A stray ';' would mean it stopped being one."""
        sql = norm(offline_graph.build_query(Start(), [Hop(hops=(1, 4))]))
        assert ";" not in sql
        assert sql.count("WITH RECURSIVE") == 1

    def test_forward_hop_joins_start_to_end(self, offline_graph):
        sql = norm(offline_graph.build_query(Start(), [Hop(direction="forward")]))
        assert "edges.end_id AS to_id" in sql
        assert "edges.start_id = seed.node_id" in sql

    def test_backward_hop_reverses_the_join(self, offline_graph):
        sql = norm(offline_graph.build_query(Start(), [Hop(direction="backward")]))
        assert "edges.start_id AS to_id" in sql
        assert "edges.end_id = seed.node_id" in sql

    def test_cycle_guard_is_present(self, offline_graph):
        """The local path array exists to stop a cycle from looping
        forever; losing this predicate turns a cyclic graph into a hang."""
        sql = norm(offline_graph.build_query(Start(), [Hop(hops=(1, 5))]))
        assert "local_path" in sql
        assert "ANY" in sql and "NOT" in sql

    def test_hop_bounds_appear_as_depth_predicates(self, offline_graph):
        sql = norm(offline_graph.build_query(Start(), [Hop(hops=(2, 7))]), literal_binds=True)
        assert "depth < 7" in sql      # recursion stops at max
        assert "depth >= 2" in sql     # results start at min

    def test_edges_are_reconstructed_from_the_path_array(self, offline_graph):
        """A hop spanning several edges reports each of them, which is
        what unnest(local_edges) is for."""
        sql = norm(offline_graph.build_query(Start(), [Hop(hops=(1, 3))]))
        assert "unnest" in sql and "local_edges" in sql

    def test_ids_are_cast_to_text(self, offline_graph):
        """Node and edge ids share one union'd column, so both are cast.
        The string-id contract in the results follows from this -- see
        CLAUDE.md."""
        sql = norm(offline_graph.build_query(Start(), [Hop()]))
        assert "CAST(edge_rows.a AS VARCHAR)" in sql
        assert "CAST(edge_rows.eid AS VARCHAR)" in sql

    def test_optional_last_hop_keeps_the_prior_match(self, offline_graph):
        with_optional = norm(offline_graph.build_query(Start(), [Hop(), Hop(optional=True)]))
        without = norm(offline_graph.build_query(Start(), [Hop(), Hop()]))
        assert with_optional != without
        assert "match_0.node_id" in with_optional

    def test_optional_on_non_last_hop_raises_before_any_sql(self, offline_graph):
        """Validation lives in build_query, so it fails without needing a
        connection -- and names which hop is at fault."""
        with pytest.raises(ValueError, match="only supported on the LAST hop"):
            offline_graph.build_query(Start(), [Hop(optional=True, label="bad"), Hop()])

    def test_via_filter_applies_to_both_walk_terms(self, offline_graph):
        """A `via` filter has to constrain the recursive step too, or
        edges past the first are unfiltered."""
        sql = norm(offline_graph.build_query(Start(), [Hop(via={"kind": "knows"}, hops=(1, 3))]))
        assert sql.count("properties @> CAST") >= 2


class TestCustomSchema:
    def test_custom_table_and_column_names_are_used(self):
        """Nothing may hardcode `nodes` / `start_id` -- Graph() takes
        overrides, and this is the test that proves the query respects
        them."""
        from sqlalchemy import BigInteger, Column, MetaData, Table
        from sqlalchemy.dialects.postgresql import JSONB

        md = MetaData()
        v = Table("vertex", md, Column("vid", BigInteger, primary_key=True),
                  Column("properties", JSONB))
        e = Table("link", md, Column("lid", BigInteger, primary_key=True),
                  Column("src", BigInteger), Column("dst", BigInteger),
                  Column("properties", JSONB))
        g = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/x",
                  node_table=v, edge_table=e, node_id_col="vid", edge_id_col="lid",
                  edge_start_col="src", edge_end_col="dst", graph_col=None)

        sql = norm(g.build_query(Start(where={"a": 1}), [Hop(hops=(1, 2))]))
        assert "graph_id" not in sql   # graph_col=None means no discriminator at all
        for name in ("vertex", "link", "vid", "lid", "src", "dst"):
            assert name in sql
        for default in ("nodes.id", "edges.start_id", "edges.end_id"):
            assert default not in sql

    def test_repr_does_not_leak_the_password(self, offline_graph):
        """Graph.__repr__ turns up in logs, tracebacks and agent
        transcripts. SQLAlchemy's URL repr masks the password; this test
        exists so a future __repr__ that formats the DSN by hand cannot
        quietly undo that."""
        assert "***" in repr(offline_graph)
        assert "offline:offline" not in repr(offline_graph)


# ---------------------------------------------------------------------
# The aggregate query's shape
# ---------------------------------------------------------------------

class TestAggregateQueryShape:
    @staticmethod
    def agg_sql(graph, start, hops, aggregates, literal_binds=False) -> str:
        return norm(graph.build_aggregate_query(start, hops, aggregates), literal_binds)

    def test_no_edge_reconstruction_ctes(self, offline_graph):
        """The whole point of aggregating in the database: no edges are
        reported, so none of the edge CTEs may be emitted. If hop_edges
        reappears here, the aggregation is paying for a subgraph it
        throws away."""
        sql = self.agg_sql(offline_graph, Start(), [Hop(hops=(1, 4)), Hop()], {"n": Count()})
        for cte in ("hop_edges", "all_edges", "edge_rows"):
            assert cte not in sql
        for cte in ("walk_0", "match_0", "walk_1", "match_1"):
            assert cte in sql

    def test_single_statement_one_round_trip(self, offline_graph):
        sql = self.agg_sql(offline_graph, Start(), [Hop()], {"n": Count(), "s": Sum("x")})
        assert ";" not in sql
        assert sql.count("WITH RECURSIVE") == 1

    def test_zero_hops_aggregates_the_seed_without_recursion(self, offline_graph):
        sql = self.agg_sql(offline_graph, Start(where={"type": "leaf"}), [], {"n": Count()})
        assert "seed" in sql and "RECURSIVE" not in sql and "walk_0" not in sql

    def test_aggregates_run_over_the_last_match(self, offline_graph):
        """Aggregating match_0 in a two-hop chain would count nodes with
        no continuation to the chain's end -- the semantic the whole
        design refuses."""
        sql = self.agg_sql(offline_graph, Start(), [Hop(), Hop()], {"n": Count()})
        assert "JOIN match_1 ON" in sql or "FROM match_1" in sql
        assert "FROM match_0 JOIN nodes" not in sql

    def test_numeric_aggregates_guard_with_jsonb_typeof(self, offline_graph):
        """A bare ::numeric cast would abort the whole query on one node
        carrying a string; the CASE guard turns that row into an ignored
        NULL instead, which is what both Cypher and PG do with nulls."""
        sql = self.agg_sql(offline_graph, Start(), [], {"a": Avg("age")}, literal_binds=True)
        assert "jsonb_typeof" in sql and "CASE WHEN" in sql
        assert "AS NUMERIC" in sql

    def test_sum_coalesces_the_empty_set_to_zero(self, offline_graph):
        """PG says NULL, Cypher and Python's sum([]) say 0; 0 is the
        answer nobody has to null-check."""
        sql = self.agg_sql(offline_graph, Start(), [], {"s": Sum("x")})
        assert "coalesce" in sql
        assert "coalesce" not in self.agg_sql(offline_graph, Start(), [], {"a": Avg("x")})

    def test_distinct_lands_inside_the_aggregate(self, offline_graph):
        plain = self.agg_sql(offline_graph, Start(), [], {"s": Sum("x")})
        distinct = self.agg_sql(offline_graph, Start(), [], {"s": Sum("x", distinct=True)})
        assert "sum(DISTINCT" in distinct and "sum(DISTINCT" not in plain

    def test_count_of_property_uses_astext(self, offline_graph):
        """->> is SQL NULL for a missing key AND for an explicit JSON
        null, so both read as 'property absent' -- the judgement Cypher
        makes. `->` would count an explicit null."""
        sql = self.agg_sql(offline_graph, Start(), [], {"n": Count("age")})
        assert "->>" in sql
        assert "->>" not in self.agg_sql(offline_graph, Start(), [], {"n": Count()})

    def test_result_names_become_column_labels(self, offline_graph):
        sql = self.agg_sql(offline_graph, Start(), [], {"friends": Count(), "avg_age": Avg("age")})
        assert "AS friends" in sql and "AS avg_age" in sql

    def test_aggregate_query_is_graph_scoped(self, offline_graph):
        """'Every read and write goes through Graph._scoped()' -- the
        aggregation path is a new query builder, so it needs its own
        proof: the discriminator must appear at every table access
        (seed, walk base, recursive term, match join, final properties
        join), or an aggregate quietly counts other graphs' rows."""
        sql = self.agg_sql(offline_graph, Start(), [Hop()], {"n": Count()}, literal_binds=True)
        assert sql.count("graph_id = 'default'") == 5

    @pytest.mark.parametrize("bad", [{}, None, [Count()], "count"])
    def test_aggregates_must_be_a_non_empty_dict(self, offline_graph, bad):
        with pytest.raises(ValueError, match="non-empty dict"):
            offline_graph.build_aggregate_query(Start(), [], bad)

    def test_non_aggregate_values_are_rejected(self, offline_graph):
        """`-- got str` included: the message names the offending type,
        and a mutant that hardcoded a different type there survived a
        match on the first half alone."""
        with pytest.raises(TypeError, match="must be Count, Sum, Avg, Min or Max -- got str"):
            offline_graph.build_aggregate_query(Start(), [], {"n": "count"})

    @pytest.mark.parametrize("hops", [
        [Hop(optional=True)],
        [Hop(), Hop(optional=True)],
        [Hop(optional=True), Hop()],   # invalid for traverse too -- still the agg message
    ])
    def test_optional_hops_are_rejected(self, offline_graph, hops):
        """optional cannot change an aggregate (the aggregate runs over
        what the hop matched), so accepting it would let callers believe
        it did."""
        with pytest.raises(ValueError, match="no effect on an aggregation"):
            offline_graph.build_aggregate_query(Start(), hops, {"n": Count()})

    def test_optional_rejection_names_the_offending_hop(self, offline_graph):
        """Same standard as the traverse-side optional error: the
        message says which hop, by index and label (or 'unlabeled'), so
        the caller of a long chain is not left counting Hops by hand."""
        with pytest.raises(ValueError, match=r"hop 1 \(unlabeled\): optional=True"):
            offline_graph.build_aggregate_query(
                Start(), [Hop(), Hop(optional=True)], {"n": Count()})
        with pytest.raises(ValueError, match=r"hop 0 \(fanout\): optional=True"):
            offline_graph.build_aggregate_query(
                Start(), [Hop(optional=True, label="fanout")], {"n": Count()})

    def test_custom_table_and_column_names_are_used(self):
        from sqlalchemy import BigInteger, Column, MetaData, Table
        from sqlalchemy.dialects.postgresql import JSONB

        md = MetaData()
        v = Table("vertex", md, Column("vid", BigInteger, primary_key=True),
                  Column("properties", JSONB))
        e = Table("link", md, Column("lid", BigInteger, primary_key=True),
                  Column("src", BigInteger), Column("dst", BigInteger),
                  Column("properties", JSONB))
        g = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/x",
                  node_table=v, edge_table=e, node_id_col="vid", edge_id_col="lid",
                  edge_start_col="src", edge_end_col="dst", graph_col=None)
        sql = norm(g.build_aggregate_query(Start(where={"a": 1}), [Hop()], {"n": Count()}))
        for name in ("vertex", "link", "vid", "src", "dst"):
            assert name in sql
        for default in ("nodes.id", "edges.start_id", "edges.end_id"):
            assert default not in sql
        assert "graph_id" not in sql   # graph_col=None means no discriminator at all


# ---------------------------------------------------------------------
# The JSON aggregate grammar must mean what the Python grammar means
# ---------------------------------------------------------------------

class TestAggregateJsonPythonEquivalence:
    """parse_aggregate() feeds the same resolve_aggregate() the Python
    classes do; reprs are exact, so comparing them proves the mapping."""

    @pytest.mark.parametrize("json_form,python_form", [
        ({"fn": "count"}, Count()),
        ({"fn": "count", "property": "age"}, Count("age")),
        ({"fn": "count", "property": "age", "distinct": True}, Count("age", distinct=True)),
        ({"fn": "sum", "property": "age"}, Sum("age")),
        ({"fn": "sum", "property": "age", "distinct": True}, Sum("age", distinct=True)),
        ({"fn": "avg", "property": "age"}, Avg("age")),
        # avg WITH distinct is the one input that can tell "avg is in the
        # sum/avg branch" from "avg fell through to min/max" -- there it
        # raises "does not apply" instead of returning Avg(distinct=True).
        # A mutant that broke exactly that survived without this case.
        ({"fn": "avg", "property": "age", "distinct": True}, Avg("age", distinct=True)),
        ({"fn": "min", "property": "age"}, Min("age")),
        ({"fn": "max", "property": "age"}, Max("age")),
    ])
    def test_same_aggregate(self, json_form, python_form):
        assert repr(parse_aggregate(json_form)) == repr(python_form)

    @pytest.mark.parametrize("agg,expected", [
        (Count(), "Count()"),
        (Count("age"), "Count('age')"),
        (Count("age", distinct=True), "Count('age', distinct=True)"),
        (Sum("age"), "Sum('age')"),
        (Sum("age", distinct=True), "Sum('age', distinct=True)"),
        (Avg("age"), "Avg('age')"),
        (Avg("age", distinct=True), "Avg('age', distinct=True)"),
        (Min("age"), "Min('age')"),
        (Max("age"), "Max('age')"),
    ])
    def test_reprs_are_exact(self, agg, expected):
        """The equivalence tests above (and the Cypher suite) compare
        aggregates BY repr, so a drifted repr matches its own garbage on
        both sides and hides real translation bugs -- a mutant that
        mangled Count.__repr__ survived the whole suite to prove it.
        One literal assertion per form pins them."""
        assert repr(agg) == expected

    @pytest.mark.parametrize("bad,message", [
        ({"fn": "median", "property": "age"}, "must be one of"),
        ({}, "must be one of"),
        ({"fn": "sum"}, "aggregates a property"),
        ({"fn": "min", "property": "age", "distinct": True}, "does not apply"),
        ({"fn": "count", "distinct": True}, "matched nodes are already distinct"),
        ({"fn": "count", "distinct": "yes", "property": "age"}, "true or false"),
        ({"fn": "count", "prop": "age"}, "unknown aggregate keys"),
        # "got str", not just "must be an object": the message names the
        # actual offending type, and a mutant that hardcoded NoneType
        # there survived a looser match.
        ("count", "must be an object -- got str"),
    ])
    def test_malformed_aggregates_are_rejected(self, bad, message):
        """Errors name the fix -- the same standard the filter grammar is
        held to. Several matches are deliberately long phrases: mutation
        testing showed the short ones still matched after the message was
        mangled."""
        with pytest.raises((ValueError, TypeError), match=message):
            parse_aggregate(bad)

    @pytest.mark.parametrize("cls", [Count, Sum, Avg, Min, Max])
    def test_non_string_property_rejected(self, cls):
        """Passing a filter, a number, or a whole node where a key
        belongs should fail at construction with the type named, not
        surface later as a broken SQL expression."""
        with pytest.raises(TypeError, match="string key"):
            cls(42)

    def test_count_distinct_without_property_rejected(self):
        with pytest.raises(ValueError, match="matched nodes are already distinct"):
            Count(distinct=True)

    def test_spec_to_aggregation_returns_the_full_triple(self):
        start, hops, aggregates = spec_to_aggregation({
            "start": {"where": {"type": "person"}},
            "hops": [{"via": {"kind": "friend"}, "hops": [1, 4]}],
            "aggregates": {"n": {"fn": "count"}, "mean": {"fn": "avg", "property": "age"}},
        })
        assert start.where == {"type": "person"}
        assert (hops[0].min_hops, hops[0].max_hops) == (1, 4)
        assert repr(aggregates["n"]) == "Count()" and repr(aggregates["mean"]) == "Avg('age')"

    @pytest.mark.parametrize("spec", [
        {"start": {"where": {"a": 1}}},
        {"start": {"where": {"a": 1}}, "aggregates": {}},
    ])
    def test_missing_or_empty_aggregates_rejected(self, spec):
        with pytest.raises(ValueError, match='non-empty "aggregates" object'):
            spec_to_aggregation(spec)


class TestAggregateToolSchema:
    """Same contract as TestToolSchema: the schema is what an agent reads
    instead of documentation."""

    def test_is_json_serializable(self):
        assert json.loads(json.dumps(AGGREGATE_TOOL_SCHEMA)) == AGGREGATE_TOOL_SCHEMA

    def test_traversal_half_matches_traverse_schema_minus_optional(self):
        """The two schemas describe the same traversal spec, except that
        aggregate() refuses `optional` -- so the hop schemas must agree
        on everything else, or the specs drift apart key by key."""
        t_hops = TRAVERSE_TOOL_SCHEMA["parameters"]["properties"]["hops"]["items"]["properties"]
        a_hops = AGGREGATE_TOOL_SCHEMA["parameters"]["properties"]["hops"]["items"]["properties"]
        assert set(t_hops) - set(a_hops) == {"optional"}
        for key in a_hops:
            assert a_hops[key] == t_hops[key]

    def test_fn_enum_matches_what_parse_aggregate_accepts(self):
        agg_schema = AGGREGATE_TOOL_SCHEMA["parameters"]["properties"]["aggregates"]
        for fn in agg_schema["additionalProperties"]["properties"]["fn"]["enum"]:
            parse_aggregate({"fn": fn, "property": "age"})  # must not raise

    def test_an_example_from_the_schema_translates(self):
        start, hops, aggregates = spec_to_aggregation({
            "start": {"where": {"type": "person"}},
            "hops": [{"via": {"kind": "friend"}, "hops": [1, 4], "direction": "forward"}],
            "aggregates": {"friends": {"fn": "count"}},
        })
        assert start.where and len(hops) == 1 and list(aggregates) == ["friends"]


# ---------------------------------------------------------------------
# Filter compilation
# ---------------------------------------------------------------------

class TestFilterCompilation:
    def test_equality_uses_jsonb_containment(self):
        """Containment, not `->> = value`: it is indexable by the GIN
        index and it treats a missing key as false rather than null,
        which is what makes NOT correct."""
        assert "properties @> CAST" in filter_sql({"type": "person"})

    def test_and_of_keys_in_one_dict(self):
        sql = filter_sql({"type": "person", "active": True}, literal_binds=True)
        assert " AND " in sql
        assert '{"type": "person"}' in sql and '{"active": true}' in sql

    def test_value_list_is_an_or(self):
        sql = filter_sql({"type": ["person", "company"]}, literal_binds=True)
        assert " OR " in sql
        assert '{"type": "person"}' in sql and '{"type": "company"}' in sql

    def test_explicit_or_and_and_nest(self):
        sql = filter_sql(AND(OR({"a": 1}, {"b": 2}), {"c": 3}))
        assert " OR " in sql and " AND " in sql

    def test_not_negates_the_containment(self):
        sql = filter_sql(NOT({"type": "person"}))
        assert sql.count("NOT") == 1
        assert "@>" in sql

    @pytest.mark.parametrize("filt,operator", [
        (GT("age", 18), ">"),
        (GTE("age", 18), ">="),
        (LT("age", 18), "<"),
        (LTE("age", 18), "<="),
    ])
    def test_range_filters_cast_to_numeric(self, filt, operator):
        """Comparing as text would order 9 after 10; the NUMERIC cast is
        what makes it arithmetic. A missing or non-numeric value casts to
        NULL and drops out, rather than raising."""
        sql = filter_sql(filt, literal_binds=True)
        assert "CAST((nodes.properties ->> 'age') AS NUMERIC)" in sql
        assert f"{operator} 18" in sql

    def test_between_is_inclusive_on_both_ends(self):
        sql = filter_sql(BETWEEN("age", 18, 65), literal_binds=True)
        assert ">= 18" in sql and "<= 65" in sql

    def test_none_is_a_true_literal(self):
        assert "true" in filter_sql(None, literal_binds=True).lower()

    def test_empty_dict_is_a_true_literal(self):
        assert "true" in filter_sql({}, literal_binds=True).lower()

    def test_callable_escape_hatch_is_passed_the_column(self):
        sql = filter_sql(lambda col: col.op("->>")("name").op("~")("^A"))
        assert "~" in sql and "->>" in sql

    def test_deeply_nested_composition(self):
        filt = AND(OR(NOT({"a": 1}), AND({"b": 2}, GT("c", 3))), OR({"d": [4, 5]}, LTE("e", 6)))
        sql = filter_sql(filt)
        assert sql.count("@>") == 4  # a, b, d=4, d=5
        assert "NOT" in sql

    @pytest.mark.parametrize("bad", [
        [{"a": 1}, {"b": 2}],
        [],
    ])
    def test_bare_list_is_rejected(self, bad):
        with pytest.raises(TypeError, match="ambiguous"):
            filter_sql(bad)

    @pytest.mark.parametrize("bad", ["a string", 42, object()])
    def test_unsupported_filter_types_are_rejected(self, bad):
        with pytest.raises(TypeError, match="filter must be"):
            filter_sql(bad)


class TestFilterSafety:
    @pytest.mark.parametrize("payload", [
        "'; DROP TABLE nodes; --",
        "person' OR '1'='1",
        "%s",
        "100%",
    ])
    def test_values_are_bound_parameters_not_inlined(self, payload):
        """Properties are caller data, and for an agent-driven graph they
        are model output. They must never reach the SQL text."""
        sql = filter_sql({"type": payload})
        assert payload not in sql
        assert "%(param" in sql

    def test_property_keys_are_bound_too(self):
        sql = filter_sql({"'; DROP TABLE nodes; --": "x"})
        assert "DROP TABLE" not in sql

    def test_unicode_survives_json_encoding(self):
        """json.dumps escapes non-ASCII to \\uXXXX, and PostgreSQL string
        literals then double the backslash. Both layers have to line up
        or a non-ASCII property silently never matches."""
        payload = json.dumps({"名前": "日本語"})
        sql = filter_sql({"名前": "日本語"}, literal_binds=True)
        assert payload.replace("\\", "\\\\") in sql


# ---------------------------------------------------------------------
# The JSON grammar must mean exactly what the Python grammar means
# ---------------------------------------------------------------------

class TestJsonPythonEquivalence:
    """Both front ends compile through one resolve(); these pairs prove
    it by comparing the SQL, values included, rather than trusting it."""

    @pytest.mark.parametrize("json_form,python_form", [
        ({"type": "person"}, {"type": "person"}),
        ({"type": ["a", "b"]}, {"type": ["a", "b"]}),
        ({"and": [{"a": 1}, {"b": 2}]}, AND({"a": 1}, {"b": 2})),
        ({"or": [{"a": 1}, {"b": 2}]}, OR({"a": 1}, {"b": 2})),
        ({"not": {"a": 1}}, NOT({"a": 1})),
        ({"gt": ["age", 18]}, GT("age", 18)),
        ({"gte": ["age", 18]}, GTE("age", 18)),
        ({"lt": ["age", 65]}, LT("age", 65)),
        ({"lte": ["age", 65]}, LTE("age", 65)),
        ({"between": ["age", 18, 65]}, BETWEEN("age", 18, 65)),
        ({"and": [{"or": [{"a": 1}, {"b": 2}]}, {"not": {"c": 3}}]},
         AND(OR({"a": 1}, {"b": 2}), NOT({"c": 3}))),
    ])
    def test_same_sql(self, json_form, python_form):
        assert (filter_sql(parse_filter(json_form), literal_binds=True)
                == filter_sql(python_form, literal_binds=True))

    def test_none_round_trips(self):
        assert parse_filter(None) is None

    def test_plain_dict_passes_through_unchanged(self):
        spec = {"type": "person", "active": True}
        assert parse_filter(spec) == spec

    @pytest.mark.parametrize("bad,message", [
        ({"and": [], "or": []}, "exactly one key"),
        ({"gt": ["age", 1], "extra": 2}, "exactly one key"),
        ("not a dict", "must be an object"),
        (42, "must be an object"),
    ])
    def test_malformed_operator_filters_are_rejected(self, bad, message):
        with pytest.raises((ValueError, TypeError), match=message):
            parse_filter(bad)


# ---------------------------------------------------------------------
# Hop / Start validation
# ---------------------------------------------------------------------

class TestHopValidationEdges:
    @pytest.mark.parametrize("value,expected", [
        (1, (1, 1)),
        (7, (7, 7)),
        ((1, 1), (1, 1)),
        ((2, 9), (2, 9)),
    ])
    def test_accepted_hop_counts(self, value, expected):
        h = Hop(hops=value)
        assert (h.min_hops, h.max_hops) == expected

    @pytest.mark.parametrize("bad,message", [
        (0, "hops must be >= 1"),
        (-1, "hops must be >= 1"),
        ((0, 3), "1 <= min <= max"),
        ((5, 2), "1 <= min <= max"),
        ((-1, 1), "1 <= min <= max"),
    ])
    def test_rejected_hop_counts(self, bad, message):
        """Asserting the message, not just the type: an error that does
        not say which rule was broken sends the caller back to the source
        to find out. Surviving mutants replaced each of these strings
        with None and nothing failed."""
        with pytest.raises(ValueError, match=message):
            Hop(hops=bad)

    @pytest.mark.parametrize("bad", [(1, 2, 3), (1,), "3", 1.5, None, [1, 2]])
    def test_rejected_hop_types(self, bad):
        with pytest.raises(TypeError, match="int or a .min, max. tuple"):
            Hop(hops=bad)

    @pytest.mark.parametrize("bad", ["Forward", "up", "", None, "backwards"])
    def test_rejected_directions(self, bad):
        with pytest.raises(ValueError, match="direction must be"):
            Hop(direction=bad)

    def test_defaults(self):
        h = Hop()
        assert (h.where, h.via, h.direction, h.optional, h.label) == (None, None, "forward", False, None)
        assert (h.min_hops, h.max_hops) == (1, 1)

    def test_start_has_no_traversal_concepts(self):
        """Start deliberately offers no via/hops/direction -- see hop.py.
        If this ever passes, the type split has been undone."""
        with pytest.raises(TypeError):
            Start(via={"kind": "x"})


# ---------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------

class TestJsonSpecTranslation:
    def test_full_spec(self):
        start, hops = spec_to_traversal({
            "start": {"where": {"type": "person"}, "label": "p"},
            "hops": [
                {"where": {"active": True}, "via": {"kind": "friend"}, "hops": [1, 4]},
                {"where": {"type": "company"}, "hops": 3, "direction": "backward",
                 "optional": True, "label": "c"},
            ],
        })
        assert start.where == {"type": "person"} and start.label == "p"
        assert (hops[0].min_hops, hops[0].max_hops) == (1, 4)
        assert (hops[1].min_hops, hops[1].max_hops) == (3, 3)
        assert hops[1].direction == "backward" and hops[1].optional is True

    def test_missing_start_is_rejected(self):
        with pytest.raises(ValueError, match="must have a 'start' key"):
            spec_to_traversal({"hops": []})

    def test_hops_key_is_optional(self):
        start, hops = spec_to_traversal({"start": {"where": {"a": 1}}})
        assert hops == []

    def test_json_operator_filters_reach_the_hops(self):
        _, hops = spec_to_traversal({
            "start": {"where": {"not": {"type": "person"}}},
            "hops": [{"where": {"between": ["age", 18, 65]}}],
        })
        assert repr(hops[0].where) == repr(BETWEEN("age", 18, 65))


class TestToolSchema:
    """The schema is what an agent reads instead of documentation, so it
    has to stay true to what spec_to_traversal actually accepts."""

    def test_is_json_serializable(self):
        assert json.loads(json.dumps(TRAVERSE_TOOL_SCHEMA)) == TRAVERSE_TOOL_SCHEMA

    def test_advertises_the_keys_the_parser_reads(self):
        hop_props = TRAVERSE_TOOL_SCHEMA["parameters"]["properties"]["hops"]["items"]["properties"]
        assert set(hop_props) == {"where", "via", "hops", "direction", "optional"}

    def test_direction_enum_matches_what_hop_accepts(self):
        for direction in TRAVERSE_TOOL_SCHEMA["parameters"]["properties"]["hops"]["items"] \
                ["properties"]["direction"]["enum"]:
            Hop(direction=direction)  # must not raise

    def test_an_example_from_the_schema_translates(self):
        spec = {"start": {"where": {"type": "person"}},
                "hops": [{"where": {"active": True}, "hops": [1, 4], "direction": "forward"}]}
        start, hops = spec_to_traversal(spec)
        assert start.where and len(hops) == 1


# ---------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------

class TestSubgraph:
    @staticmethod
    def sample(**kw) -> Subgraph:
        return Subgraph(
            nodes=[{"id": "1", "properties": {"name": "a"}},
                   {"id": "2", "properties": {"name": "b"}}],
            edges=[{"start_id": "1", "end_id": "2", "properties": {"kind": "knows"}}],
            **kw,
        )

    def test_to_dict_is_json_serializable(self):
        d = self.sample(elapsed_ms=1.5).to_dict()
        assert json.loads(json.dumps(d))["elapsed_ms"] == 1.5
        assert set(d) == {"nodes", "edges", "elapsed_ms"}

    def test_to_networkx(self):
        g = self.sample().to_networkx()
        assert g.number_of_nodes() == 2 and g.number_of_edges() == 1
        assert g.nodes["1"]["name"] == "a"
        assert g.edges["1", "2"]["kind"] == "knows"

    def test_multigraph_preserves_parallel_edges(self):
        s = self.sample()
        s.edges.append({"start_id": "1", "end_id": "2", "properties": {"kind": "refers"}})
        assert s.to_networkx().number_of_edges() == 1           # DiGraph collapses
        assert s.to_networkx(multigraph=True).number_of_edges() == 2

    def test_empty_result(self):
        empty = Subgraph()
        assert empty.to_dict() == {"nodes": [], "edges": [], "elapsed_ms": 0.0}
        assert empty.to_networkx().number_of_nodes() == 0

    def test_repr_summarizes(self):
        assert "nodes=2" in repr(self.sample()) and "edges=1" in repr(self.sample())

    def test_default_lists_are_not_shared_between_instances(self):
        """A mutable default would make every Subgraph share one list --
        the classic dataclass trap."""
        a, b = Subgraph(), Subgraph()
        a.nodes.append({"id": "1"})
        assert b.nodes == []
