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
    nt = Node
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
        connection -- and names which hop is at fault, falling back to
        the 'unlabeled' spelling when the hop has no label to name."""
        with pytest.raises(ValueError, match="only supported on the LAST hop"):
            offline_graph.build_query(Start(), [Hop(optional=True, label="bad"), Hop()])
        with pytest.raises(ValueError) as exc:
            offline_graph.build_query(Start(), [Hop(optional=True), Hop()])
        assert str(exc.value).startswith("hop 0 (unlabeled): optional=True")

    def test_via_filter_applies_to_both_walk_terms(self, offline_graph):
        """A `via` filter has to constrain the recursive step too, or
        edges past the first are unfiltered."""
        sql = norm(offline_graph.build_query(Start(), [Hop(via={"kind": "knows"}, hops=(1, 3))]))
        assert sql.count("properties @> CAST") >= 2

    def test_no_pins_emits_byte_identical_sql(self, offline_graph):
        """Step-wise reranking threads an optional `pins` through the
        whole walk. A traversal with no rerank= passes None, and None
        must add NOTHING -- the same promise
        test_defining_vectors_changes_no_near_less_query makes for
        declared vector fields. Without this, every existing query plan
        would be up for renegotiation the moment reranking landed."""
        start, hops = Start(where={"type": "person"}), [Hop(hops=(1, 3)), Hop(optional=True)]
        assert (norm(offline_graph.build_query(start, hops))
                == norm(offline_graph.build_query(start, hops, pins=None)))
        # The aggregate path shares _walk_matches and so shares the
        # promise; optional= is dropped only because aggregation refuses
        # it for reasons of its own.
        plain = [Hop(hops=(1, 3)), Hop()]
        assert (norm(offline_graph.build_aggregate_query(start, plain, {"n": Count()}))
                == norm(offline_graph.build_aggregate_query(start, plain, {"n": Count()},
                                                            pins=None)))

    def test_pinning_narrows_the_step_it_names_and_nothing_else(self, offline_graph):
        """Pinning is an `id IN (...)` on the SEED and on `match_i`, and
        each step gets only its own list. A pin leaking onto the wrong
        step would silently traverse from nodes the reranker dropped."""
        sql = norm(offline_graph.build_query(
            Start(where={"type": "person"}),
            [Hop(via={"kind": "knows"}), Hop()], pins={-1: [1, 2], 1: [9]}), literal_binds=True)
        assert "nodes.id IN (1, 2)" in sql
        assert "walk_1.to_id IN (9)" in sql
        # hop 0 was not pinned, so its match keeps the shape it had: the
        # only `walk_0.to_id IN` left is hop_edges_0's own subquery
        # against match_0, never a literal id list.
        assert "walk_0.to_id IN (SELECT match_0.node_id" in sql
        assert "walk_0.to_id IN (9)" not in sql
        assert "walk_0.to_id IN (1, 2)" not in sql

    def test_a_pinned_step_still_reports_its_edges_from_the_walk(self, offline_graph):
        """The whole safety argument for re-running an ORDINARY
        traversal with survivors pinned: `hop_edges_i` still derives
        from `full_walk` joined against `match_i`, so multi-hop edge
        reconstruction and dead-end pruning are untouched. If pinning
        ever started reporting edges from the pinned id list itself,
        this is what would notice."""
        sql = norm(offline_graph.build_query(Start(), [Hop(hops=(1, 3))], pins={0: [4, 5]}))
        assert "hop_edges_0 AS" in sql
        assert "unnest(walk_0.local_edges)" in sql
        assert "walk_0.to_id IN (SELECT match_0.node_id" in sql


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

    def test_default_tables_have_no_extra_columns(self, offline_graph):
        assert offline_graph.node_extra_cols == ()
        assert offline_graph.edge_extra_cols == ()

    def test_extra_columns_are_discovered_from_the_table(self):
        """A custom table's columns beyond the ones this library already
        has a use for -- a foreign key to a `users` table, say -- are
        found with no separate declaration; see models.py's "EXTENDING
        THE MODEL" note."""
        from sqlalchemy import BigInteger, Column, MetaData, Table, Text
        from sqlalchemy.dialects.postgresql import JSONB

        md = MetaData()
        v = Table("vertex", md, Column("vid", BigInteger, primary_key=True),
                  Column("user_id", BigInteger), Column("properties", JSONB))
        e = Table("link", md, Column("lid", BigInteger, primary_key=True),
                  Column("src", BigInteger), Column("dst", BigInteger),
                  Column("note", Text), Column("properties", JSONB))
        g = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/x",
                  node_table=v, edge_table=e, node_id_col="vid", edge_id_col="lid",
                  edge_start_col="src", edge_end_col="dst", graph_col=None)
        assert g.node_extra_cols == ("user_id",)
        assert g.edge_extra_cols == ("note",)

    def test_reserved_edge_keys_are_never_extra_columns(self):
        """"start"/"end" address a node-by-property reference in
        add_edges()/merge_edges(); a column that happens to be named one
        of them -- even under custom start_id/end_id column names --
        must never be treated as a plain extra column, or ingestion
        would try to write the reference dict into it."""
        from sqlalchemy import BigInteger, Column, MetaData, Table
        from sqlalchemy.dialects.postgresql import JSONB

        md = MetaData()
        e = Table("link", md, Column("lid", BigInteger, primary_key=True),
                  Column("src", BigInteger), Column("dst", BigInteger),
                  Column("start", BigInteger), Column("properties", JSONB))
        g = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/x",
                  node_table=Node, edge_table=e, edge_id_col="lid",
                  edge_start_col="src", edge_end_col="dst", graph_col=None)
        assert g.edge_extra_cols == ()

    def test_vector_columns_are_never_extra_columns(self):
        """define_vectors()/migrate_vectors() attach vec_* columns to
        the SAME shared Table object a later Graph() may reuse; those
        follow their own write path (set_vectors()) and must stay
        invisible to extra-column discovery even though they are, by
        then, ordinary columns on the table."""
        from sqlalchemy import BigInteger, Column, MetaData, Table
        from sqlalchemy.dialects.postgresql import ARRAY, JSONB, REAL

        md = MetaData()
        v = Table("vertex", md, Column("vid", BigInteger, primary_key=True),
                  Column("vec_summary", ARRAY(REAL)), Column("properties", JSONB))
        g = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/x",
                  node_table=v, node_id_col="vid", graph_col=None)
        assert g.node_extra_cols == ()

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
    def test_an_unsupported_filter_type_is_named(self):
        """resolve()'s catch-all, asserted VERBATIM for the same reason
        as the bare-list message: it enumerates the accepted forms and
        names what it got, and any unpinned fragment is a surviving
        mutant waiting to flap into a CI report (x_resolve__mutmut_81
        printed NoneType for every wrong-type filter)."""
        from sqlalchemy import column as sa_column

        from hopai.filters import resolve
        with pytest.raises(TypeError) as exc:
            resolve(sa_column("properties"), 42)
        assert str(exc.value) == (
            "filter must be None, a dict, AND/OR/NOT/GT/GTE/LT/LTE/BETWEEN, or a callable "
            "-- got int"
        )

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
        """Asserted VERBATIM, the same rule hop.py's messages earned:
        this message is a paste-able rewrite, and successive CI runs
        surfaced its string mutants one flap at a time (x_resolve 9, 10,
        ...) as long as any fragment went unpinned. If you reword it,
        update this test."""
        with pytest.raises(TypeError) as exc:
            filter_sql(bad)
        assert str(exc.value) == (
            "a bare list is ambiguous -- use OR(...) to mean 'any of these filters', "
            "e.g. OR({'type': 'person'}, {'type': 'company'}) instead of "
            "[{'type': 'person'}, {'type': 'company'}]"
        )

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

def rerank_policy(**options):
    """A RerankPolicy over a client that touches no network: the
    operator's Rerank, the properties it publishes, and its ceiling.

    `score(query, documents) -> [float]` is the whole reranker contract,
    so a plain function IS a client here -- which is also what keeps
    these tests off a provider."""
    from hopai import Rerank
    from hopai.json_api import RerankPolicy
    return RerankPolicy(
        Rerank(options.pop("client", None) or (lambda query, documents: [1.0] * len(documents)),
               document_from=options.pop("document_from", ".properties.title"),
               candidates=options.pop("candidates", 50),
               retries=options.pop("retries", 0)),
        fields=options.pop("fields", ["properties.title", "properties.body"]),
        max_candidates=options.pop("max_candidates", 100),
    )


class TestSpecSuppliedReranking:
    """`rerank` is the one spec key whose value is CODE -- a jq filter
    that a model may write and that hopai then runs over rows and posts
    to a third party. parse_rerank() is the single site that gates it,
    the way refuse_vectors() is for `vector`, so everything here is a
    property of that one function."""

    def test_a_rerank_spec_refuses_when_no_reranker_was_configured(self):
        """The client cannot travel in JSON -- it holds an API key and a
        socket -- so a spec asking to rerank with nothing configured has
        to be told what to pass rather than quietly not reranking, which
        would be a different answer with no signal."""
        with pytest.raises(ValueError) as exc:
            spec_to_traversal({"start": {"where": {"a": 1}, "near": {"field": "s", "text": "x"},
                                         "keep": 3, "rerank": {"document_from": ".id"}}})
        assert "RerankPolicy" in str(exc.value)

    def test_a_spec_overrides_the_filter_and_the_budget_and_nothing_else(self):
        """The whole trust boundary in one assertion: what a model sends
        decides WHAT is ranked and HOW MUCH is spent, and every field
        that holds a credential or picks a model stays the operator's.
        Without this a spec could swap the client."""
        from hopai import Rerank
        client = lambda query, documents: [1.0] * len(documents)      # noqa: E731
        policy = rerank_policy(client=client, document_from=".properties.title",
                               candidates=50)
        start, _ = spec_to_traversal(
            {"start": {"near": {"field": "s", "text": "x"}, "keep": 3,
                       "rerank": {"document_from": ".properties.body", "candidates": 20}}},
            policy)
        assert isinstance(start.rerank, Rerank)
        assert start.rerank.document_from == ".properties.body"
        assert start.rerank.candidates == 20
        assert start.rerank.client is client
        assert (start.rerank.retries, start.rerank.model) == (0, None)

    def test_a_spec_that_names_neither_key_inherits_the_operators_own(self):
        start, _ = spec_to_traversal(
            {"start": {"near": {"field": "s", "text": "x"}, "keep": 3, "rerank": {}}},
            rerank_policy(document_from=".properties.title", candidates=50))
        assert (start.rerank.document_from, start.rerank.candidates) == (".properties.title", 50)

    def test_a_filter_outside_the_safe_subset_refuses(self):
        """`env` returns the process environment and the filter's OUTPUT
        IS THE DOCUMENT, which is POSTed to a reranking vendor -- so
        this exact filter is one-line exfiltration of every credential
        the process holds. It does not parse in hopai.jqsafe's subset,
        and parse_rerank() is where that subset is applied."""
        from hopai.jqsafe import UnsafeFilter
        with pytest.raises(UnsafeFilter) as exc:
            spec_to_traversal(
                {"start": {"near": {"field": "s", "text": "x"}, "keep": 3,
                           "rerank": {"document_from": "env.DATABASE_URL"}}},
                rerank_policy())
        assert "`env` is not available" in str(exc.value)

    def test_a_filter_reading_an_unpublished_property_refuses_naming_the_published_ones(self):
        """`.properties.ssn` parses perfectly in the safe subset, so the
        grammar alone is not a defence: what stops it is the operator's
        published list, and the refusal has to name that list or a model
        cannot write a filter that would be accepted."""
        from hopai.jqsafe import UnsafeFilter
        with pytest.raises(UnsafeFilter) as exc:
            spec_to_traversal(
                {"start": {"near": {"field": "s", "text": "x"}, "keep": 3,
                           "rerank": {"document_from": ".properties.ssn"}}},
                rerank_policy(fields=["properties.title"]))
        assert "properties.ssn" in str(exc.value) and "properties.title" in str(exc.value)

    def test_candidates_over_the_ceiling_refuses_naming_it(self):
        """Named, never clamped -- the same judgement `candidates < keep`
        gets. A reranker is billed per document, so silently serving 100
        where 500 was asked for hides the disagreement exactly where it
        costs money."""
        with pytest.raises(ValueError) as exc:
            spec_to_traversal(
                {"start": {"near": {"field": "s", "text": "x"}, "keep": 3,
                           "rerank": {"candidates": 500}}},
                rerank_policy(max_candidates=100))
        assert "500" in str(exc.value) and "100" in str(exc.value)

    def test_an_unknown_rerank_key_refuses_rather_than_being_ignored(self):
        """`top_n` is what LanceDB calls this and what a model reaches
        for; ignoring it would silently rerank a different number of
        candidates than the call asked for."""
        with pytest.raises(ValueError) as exc:
            spec_to_traversal(
                {"start": {"near": {"field": "s", "text": "x"}, "keep": 3,
                           "rerank": {"top_n": 5}}},
                rerank_policy())
        assert "top_n" in str(exc.value) and "candidates" in str(exc.value)

    def test_a_hop_rerank_is_named_by_its_position(self):
        """Three hops each carrying a filter, and a refusal that said
        only "rerank" would leave a model guessing which one to fix."""
        from hopai.jqsafe import UnsafeFilter
        with pytest.raises(UnsafeFilter) as exc:
            spec_to_traversal(
                {"start": {"where": {"a": 1}},
                 "hops": [{}, {"near": {"field": "s", "text": "x"}, "keep": 2,
                               "rerank": {"document_from": ".properties.ssn"}}]},
                rerank_policy())
        assert str(exc.value).startswith("hops[1].rerank.document_from")

    def test_the_operators_own_filter_is_held_to_the_list_they_published(self):
        """Refused where both were written, not on the first query that
        inherits the template: a policy whose own filter reads outside
        its own allowlist would make every spec naming no
        `document_from` refuse for a reason the caller did not cause."""
        from hopai.jqsafe import UnsafeFilter
        with pytest.raises(UnsafeFilter):
            rerank_policy(document_from=".properties.secret",
                          fields=["properties.title"])

    def test_a_policy_publishing_nothing_refuses(self):
        """There is no "everything" spelling on purpose. Defaulting to
        the whole row would put `.properties.ssn` one accepted filter
        away from a vendor, and an operator who never thought about it
        would never find out."""
        from hopai import Rerank
        from hopai.json_api import RerankPolicy
        template = Rerank(lambda query, documents: [1.0], document_from=".properties.title")
        with pytest.raises(ValueError) as exc:
            RerankPolicy(template)
        assert "properties.title" in str(exc.value)

    @pytest.mark.parametrize("spec, expected", [
        ("just rerank it", TypeError),                  # a string, not an object
        ({"document_from": 42}, ValueError),            # not a filter
        ({"document_from": "   "}, ValueError),         # nothing to evaluate
        ({"candidates": 0}, ValueError),                # nothing to rerank
        ({"candidates": True}, ValueError),             # a bool IS an int in Python
        ({"candidates": "20"}, ValueError),             # a JSON number that arrived as text
    ])
    def test_a_malformed_rerank_spec_refuses_by_shape(self, spec, expected):
        """Checked, never coerced -- the rule CLAUDE.md states for
        all=/detach=/replace=, applied where the same tool-call failures
        arrive: `candidates: true` is 1 to Python (one candidate, which
        can reorder nothing) and `"20"` is a JSON type error, not a
        number to read."""
        with pytest.raises(expected):
            spec_to_traversal(
                {"start": {"near": {"field": "s", "text": "x"}, "keep": 3,
                           "rerank": spec}},
                rerank_policy())

    @pytest.mark.parametrize("fields, expected", [
        ("properties.title", TypeError),                # one path, not a list of them
        ([], ValueError),                               # publishes nothing
        ([42], TypeError),                              # not a path
        (["  "], TypeError),                            # nor is whitespace
    ])
    def test_a_malformed_field_list_refuses_where_it_was_written(self, fields, expected):
        """A bare string is the likely mistake and the dangerous one:
        it iterates into characters, so `fields="properties.title"` would
        publish paths named `p`, `r`, `o`... and refuse every real
        filter with a list of letters."""
        with pytest.raises(expected):
            rerank_policy(fields=fields)

    def test_a_template_that_is_not_a_rerank_refuses(self):
        from hopai.json_api import RerankPolicy
        with pytest.raises(TypeError, match="must be a Rerank"):
            RerankPolicy(lambda query, documents: [1.0], fields=["properties.title"])

    @pytest.mark.parametrize("ceiling", [0, True, "100"])
    def test_a_malformed_ceiling_refuses(self, ceiling):
        with pytest.raises(ValueError, match="max_candidates"):
            rerank_policy(max_candidates=ceiling)

    def test_one_candidate_is_still_a_list_of_candidates(self):
        """The base's own call-shape refusal, which must survive the
        guard: a dict iterates its KEYS, so building documents from one
        candidate would run the filter against the string 'id' and
        blame the filter for the call."""
        from hopai.json_api import parse_rerank
        rerank = parse_rerank({}, rerank_policy(), "start.rerank")
        with pytest.raises(TypeError, match="Pass \\[candidate\\]"):
            rerank.build_documents({"id": "7", "properties": {"title": "Raft"}})

    def test_the_async_path_hides_the_provider_too(self):
        """hopai/asyncio.py awaits ascore(), so a leak fixed only in
        score() would come straight back on the async read path -- which
        is the one an MCP server over HTTP would eventually reach."""
        import asyncio

        from hopai import RerankError
        from hopai.json_api import parse_rerank

        async def explode(query, documents):
            raise RuntimeError("401 for key sk-live-DEADBEEF")

        rerank = parse_rerank({}, rerank_policy(client=explode), "start.rerank")
        with pytest.raises(RerankError) as exc:
            asyncio.run(rerank.ascore("q", ["a document"]))
        assert "DEADBEEF" not in str(exc.value) and "server-side failure" in str(exc.value)

    def test_the_async_path_still_scores_when_the_provider_works(self):
        """The guard changes failures only -- a wrapper that swallowed
        the answer would pass every leak test above."""
        import asyncio

        from hopai.json_api import parse_rerank

        async def score(query, documents):
            return [float(len(d)) for d in documents]

        rerank = parse_rerank({}, rerank_policy(client=score), "start.rerank")
        assert asyncio.run(rerank.ascore("q", ["abc"])) == [3.0]

    def test_a_flat_search_reranks_at_the_top_level(self):
        """`rerank` sits beside `near` on a search spec rather than
        inside a step, because a flat search has ONE ranked list and one
        place to reorder it -- and the policy has to reach
        Graph.vector_search(), or the tool would advertise reranking and
        quietly return similarity order."""
        from hopai.json_api import vector_search_json
        graph = TestToolSchema.offgraph()
        seen = {}
        graph.vector_search = lambda *near, **options: seen.update(options) or []
        vector_search_json(graph, {"near": {"field": "s", "text": "x"}, "k": 3,
                                   "rerank": {"document_from": ".properties.body"}},
                           rerank=rerank_policy())
        assert seen["rerank"].document_from == ".properties.body"

    def test_a_search_spec_refuses_reranking_with_nothing_configured(self):
        """Refused before the query is built, not after: a spec asking
        for a stage that cannot run must not first spend a round trip."""
        from hopai.json_api import vector_search_json
        with pytest.raises(ValueError, match="RerankPolicy"):
            vector_search_json(TestToolSchema.offgraph(),
                               {"near": {"field": "s", "text": "x"}, "rerank": {}})

    def test_a_template_over_its_own_ceiling_refuses(self):
        """A spec naming no `candidates` inherits the template's, so
        this configuration would make every default call refuse for a
        reason that is the operator's rather than the caller's."""
        with pytest.raises(ValueError) as exc:
            rerank_policy(candidates=200, max_candidates=100)
        assert "200" in str(exc.value) and "100" in str(exc.value)


class TestASpecSuppliedFilterNeverQuotesTheRow:
    """MEASURED: 3 of 5 jq runtime type errors quote the offending value
    verbatim (`string ("SSN-123-45-6789") cannot be parsed as a
    number`), and two of _evaluate()'s own repr the value the filter
    produced. `fields=` decides what a filter may READ, so this is not a
    privilege escalation -- it is a row value taking a route out of the
    process that nobody chose, into a tool result and whatever logs the
    far side keeps. A Rerank written in Python keeps jq's own message;
    one parse_rerank() built does not."""

    SECRET = "SSN-123-45-6789"

    def candidate(self, **extra) -> dict:
        return {"id": "7", "properties": {"title": self.SECRET, **extra},
                "similarity": 0.5, "similarities": {}, "boosts": {}}

    def built(self, document_from: str, **options):
        from hopai.json_api import parse_rerank
        return parse_rerank({"document_from": document_from},
                            rerank_policy(**options), "start.rerank")

    def test_a_jq_runtime_error_names_the_filter_and_the_candidate_and_nothing_else(self):
        """`tonumber` on text is the measured case. Without the guard
        the tool result carries the row's own value; with it, a model
        still learns which filter failed on which candidate and how to
        fix it."""
        rerank = self.built(".properties.title | tonumber")
        with pytest.raises(ValueError) as exc:
            rerank.build_documents([self.candidate()])
        message = str(exc.value)
        assert self.SECRET not in message
        assert ".properties.title | tonumber" in message and "id='7'" in message
        assert "tonumber" in message and "//" in message          # names the fix

    def test_a_non_string_document_is_refused_without_repring_it(self):
        """_evaluate()'s own message reprs the value -- here a number
        that is the row's, not the filter's."""
        rerank = self.built(".properties.pin",
                            fields=["properties.title", "properties.pin"])
        with pytest.raises(ValueError) as exc:
            rerank.build_documents([self.candidate(pin=90210)])
        assert "90210" not in str(exc.value)
        assert "not a string" in str(exc.value) and "id='7'" in str(exc.value)

    def test_a_filter_selecting_nothing_still_names_the_fix(self):
        """The common case, and the one with no row data in it either
        way -- so the guard must not flatten every failure into one
        message: "give it a fallback" is the fix here and nonsense for
        a type error."""
        rerank = self.built('.properties.title | select(. == "no")')
        with pytest.raises(ValueError) as exc:
            rerank.build_documents([self.candidate()])
        assert "no document at all" in str(exc.value)
        assert "untitled" in str(exc.value)

    def test_an_unsafe_filter_still_reports_itself(self):
        """The guard catches ValueError, and UnsafeFilter IS one --
        flattening it would replace "`env` is not available, read a
        property instead" with a message about a candidate, which
        blames the row for the filter."""
        from hopai.jqsafe import UnsafeFilter
        rerank = self.built(".properties.title")
        with pytest.raises(UnsafeFilter) as exc:
            rerank.build_documents([self.candidate()], fields=["properties.other"])
        assert "properties.other" in str(exc.value)

    def test_a_spent_provider_failure_never_quotes_the_provider(self):
        """An SDK exception's repr can carry the configuration it was
        constructed with, API key included, and RerankError quotes it
        verbatim so a Python caller can debug. A Python caller is inside
        the trust boundary; the far side of a tool call is not."""
        from hopai import RerankError

        def explode(query, documents):
            raise RuntimeError("401 Unauthorized for key sk-live-DEADBEEF")

        rerank = self.built(".properties.title", client=explode)
        with pytest.raises(RerankError) as exc:
            rerank.score("a query", ["a document"])
        assert "DEADBEEF" not in str(exc.value) and "401" not in str(exc.value)
        assert "server-side failure" in str(exc.value)

    def test_a_working_reranker_is_untouched_by_any_of_it(self):
        """The guard must change failures only. If it changed the
        documents or the scores, every one of the tests above would
        still pass and the feature would be broken."""
        rerank = self.built(".properties.title", client=lambda q, d: [len(x) for x in d])
        assert rerank.build_documents([self.candidate()]) == [self.SECRET]
        assert rerank.score("q", ["abc"]) == [3.0]


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

    def test_an_empty_hop_object_gets_every_documented_default(self):
        """{} is a legal hop, and each default is part of the JSON
        contract the tool schema documents -- mutants replacing any of
        them (hops=2, optional=None, direction mangled) survived because
        no test spelled a hop with everything omitted."""
        _, hops = spec_to_traversal({"start": {"where": {"a": 1}}, "hops": [{}]})
        (hop,) = hops
        assert (hop.min_hops, hop.max_hops) == (1, 1)
        assert hop.direction == "forward"
        assert hop.optional is False
        assert hop.where is None and hop.via is None and hop.label is None

    def test_hop_range_and_label_forwarding(self):
        _, hops = spec_to_traversal({
            "start": {"where": {"a": 1}},
            "hops": [{"hops": [2, 3], "label": "L"}],
        })
        assert (hops[0].min_hops, hops[0].max_hops) == (2, 3)
        assert hops[0].label == "L"

    def test_spec_errors_lead_with_the_missing_key(self):
        """XX-padding mutants kept the matched fragment mid-string, so
        the pins must anchor at the start of the message."""
        with pytest.raises(ValueError) as exc:
            spec_to_traversal({"hops": []})
        assert str(exc.value).startswith("spec must have a 'start' key")
        with pytest.raises(ValueError) as exc:
            spec_to_aggregation({"start": {"where": {"a": 1}}})
        assert str(exc.value).startswith('spec must have a non-empty "aggregates"')

    def test_aggregate_query_needs_a_non_empty_dict(self):
        """The same start-anchored pin for build_aggregate_query's own
        refusal (mutant xǁGraphǁbuild_aggregate_query__mutmut_5)."""
        from hopai import Graph, Start
        offline = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline")
        with pytest.raises(ValueError) as exc:
            offline.build_aggregate_query(Start(), [], {})
        assert str(exc.value).startswith("aggregates must be a non-empty dict")


class TestToolSchema:
    """The schema is what an agent reads instead of documentation, so it
    has to stay true to what spec_to_traversal actually accepts."""

    def test_is_json_serializable(self):
        assert json.loads(json.dumps(TRAVERSE_TOOL_SCHEMA)) == TRAVERSE_TOOL_SCHEMA

    def test_advertises_the_keys_the_parser_reads(self):
        """Derived from the parser's own key sets rather than listed
        here, so widening one without the other fails in both
        directions -- an unadvertised key is documentation a model
        never sees, and an advertised one the parser rejects is a tool
        call that errors.

        `label` is the single omission: it names result groups for the
        caller's own bookkeeping and builds no SQL, so there is nothing
        for a model to decide. The near-level omission -- "vector" --
        is pinned separately in tests/test_vectors.py."""
        from hopai.json_api import _HOP_KEYS, _START_KEYS
        properties = TRAVERSE_TOOL_SCHEMA["parameters"]["properties"]
        assert set(properties["hops"]["items"]["properties"]) == _HOP_KEYS - {"label"}
        assert set(properties["start"]["properties"]) == _START_KEYS - {"label"}

    def test_the_search_schema_advertises_the_keys_its_parser_reads(self):
        """The same derivation test_advertises_the_keys_the_parser_reads
        makes for a traversal, for the flat search -- which had none, so
        `rerank` could have reached _SEARCH_KEYS with nothing advertising
        it. No omissions here at all: `label` groups nothing in a flat
        result and "vector" lives one level down inside the near spec,
        which tests/test_vectors.py pins."""
        from hopai import VECTOR_SEARCH_TOOL_SCHEMA
        from hopai.json_api import _SEARCH_KEYS
        assert set(VECTOR_SEARCH_TOOL_SCHEMA["parameters"]["properties"]) == _SEARCH_KEYS

    def test_the_rerank_spec_shows_a_whole_candidate_and_a_worked_filter(self):
        """`document_from` is the one parameter whose value is CODE, and
        a model writes markedly better jq shown a concrete candidate
        than told about one in prose. So the description has to carry
        the whole shape a filter runs against AND one filter that works
        on it -- a description that only said "a jq filter over the
        candidate" is what left a model guessing whether `.title` or
        `.properties.title` is the path.

        Pinned as a parsable object rather than as a sentence, so the
        prose around it stays free to reword: what must survive is that
        the example is a real candidate with the keys the read path
        actually produces."""
        spec = TRAVERSE_TOOL_SCHEMA["parameters"]["$defs"]["rerank"]["properties"]
        description = spec["document_from"]["description"]
        example = json.loads(description[description.index("{"):description.rindex("}") + 1])
        assert set(example) == {"id", "properties", "similarity", "similarities", "boosts"}
        assert "title" in example["properties"]
        # The worked filter, and the guard that keeps a missing property
        # from producing no document at all.
        assert ".properties.title" in description and '// ""' in description
        # `.paths` is hop-only, and "you have to know that" is the kind
        # of gap this library treats as a defect.
        assert "paths" in description and "never on `start`" in description

    def test_the_rerank_spec_says_which_bound_candidates_is(self):
        """`candidates` and `keep`/`k` are the two numbers most easily
        confused, and LanceDB's own API confuses them (a `top_n` on the
        reranker beside the query's `.limit()`). The description has to
        say which end it bounds, or a model sets it to the answer size
        and reranks a pool it cannot reorder."""
        spec = TRAVERSE_TOOL_SCHEMA["parameters"]["$defs"]["rerank"]["properties"]
        assert "INPUT bound" in spec["candidates"]["description"]

    def test_direction_enum_matches_what_hop_accepts(self):
        for direction in TRAVERSE_TOOL_SCHEMA["parameters"]["properties"]["hops"]["items"] \
                ["properties"]["direction"]["enum"]:
            Hop(direction=direction)  # must not raise

    def test_an_example_from_the_schema_translates(self):
        spec = {"start": {"where": {"type": "person"}},
                "hops": [{"where": {"active": True}, "hops": [1, 4], "direction": "forward"}]}
        start, hops = spec_to_traversal(spec)
        assert start.where and len(hops) == 1

    @staticmethod
    def offgraph():
        from hopai import Graph
        return Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline")

    @staticmethod
    def declare(graph):
        from hopai import EdgeType, NodeType, Property
        graph.define_schema(
            nodes=[NodeType("person", properties=[Property("email", "string", required=True),
                                                  Property("age", "number")]),
                   NodeType("company", properties=[Property("name", "string", required=True)])],
            edges=[EdgeType("works_at", source="person", target="company",
                            properties=[Property("since", "number")]),
                   EdgeType("likes", source="person", target="person"),
                   EdgeType("likes", source="person", target="company")],
        )

    def test_tool_schemas_embed_the_declared_schema(self):
        """A model reads the description instead of documentation, so
        with a schema defined every tool must say what exists: type
        names, required markers, and endpoint pairs grouped per kind --
        and still be json.dumps-clean."""
        graph = self.offgraph()
        self.declare(graph)
        tools = graph.tool_schemas()
        assert len(tools) == 4
        for tool in tools:
            json.dumps(tool)
            description = tool["description"]
            assert "person(email*, age)" in description
            assert "company(name*)" in description
            assert "works_at: person -> company (since)" in description
            assert "likes: person -> person, person -> company" in description

    def test_tool_schemas_without_a_schema_are_the_constants_uncoupled(self):
        """Schema stays optional: no schema means the static definitions,
        equal but never SHARED -- an integration mutating its copy must
        not corrupt the module constants every other caller reads.

        `rerank=True` here so the comparison is against the constants as
        written; the default strips `rerank`, which the two tests below
        pin separately."""
        from hopai import INGEST_TOOL_SCHEMA, MUTATE_TOOL_SCHEMA
        tools = self.offgraph().tool_schemas(rerank=True)
        assert tools == [TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA, INGEST_TOOL_SCHEMA,
                         MUTATE_TOOL_SCHEMA]
        tools[0]["description"] = "vandalized"
        tools[2]["parameters"]["properties"].clear()
        assert TRAVERSE_TOOL_SCHEMA["description"] != "vandalized"
        assert INGEST_TOOL_SCHEMA["parameters"]["properties"]

    def test_a_bare_graph_does_not_advertise_rerank(self):
        """A Graph holds no reranker -- the client is the operator's and
        arrives via serve(rerank=) or traverse_json(rerank=Policy) -- so
        a spec carrying `rerank` against one is refused BY NAME. A schema
        offering a parameter its handler rejects every time is the defect
        CLAUDE.md names, and it is the judgement that already keeps the
        vector-search tool out of this list for a graph with no vectors."""
        blob = json.dumps(self.offgraph().tool_schemas())
        assert "rerank" not in blob

    def test_rerank_true_keeps_it_for_a_caller_that_has_one(self):
        """hopai.mcp builds ON these schemas rather than beside them, so
        the parameter has to survive long enough for serve(rerank=) to
        fill in its published fields and ceiling. Stripping it
        unconditionally would leave that surface with nothing to
        configure."""
        blob = json.dumps(self.offgraph().tool_schemas(rerank=True))
        assert "rerank" in blob

    def test_a_policy_where_a_bool_belongs_is_refused_not_swallowed(self):
        """`rerank` is CHECKED, not coerced -- the `all="false"` rule.
        A RerankPolicy is truthy, so tool_schemas(rerank=policy) was
        tool_schemas(True) exactly: the policy silently discarded and
        the model shown the abstract "the application may cap this"
        sentences instead of the published field list and the real
        ceiling it was handed. A truthy non-bool selecting a WEAKER
        behaviour than the caller asked for is the defect the
        invariants name."""
        from hopai import Rerank, RerankPolicy
        policy = RerankPolicy(Rerank(lambda query, documents: [0.0] * len(documents),
                                     document_from=".properties.title"),
                              fields=["properties.title"], max_candidates=120)
        with pytest.raises(TypeError) as refused:
            self.offgraph().tool_schemas(rerank=policy)
        message = str(refused.value)
        # It names where a policy DOES belong: a Graph cannot publish
        # one's fields or ceiling -- that rewrite is hopai.mcp's, and
        # core.py must not import the MCP front end to borrow it.
        assert "rerank= is True or False" in message
        assert "hopai.mcp.serve(rerank=" in message and "traverse_json" in message

    @pytest.mark.parametrize("truthy", [1, "true", ["yes"]])
    def test_no_truthy_value_stands_in_for_the_bool(self, truthy):
        """The string "true" arriving where a JSON boolean was meant is
        an ordinary tool-call failure, and every one of these used to
        mean rerank=True by accident."""
        with pytest.raises(TypeError, match="rerank= is True or False"):
            self.offgraph().tool_schemas(rerank=truthy)

    def test_the_rerank_object_says_document_from_may_be_left_out(self):
        """Served, `candidates` ends "Leave it out to use this server's
        default of N" (hopai.mcp writes it in) and `document_from` had
        no such sentence -- and the object has no "required" list, so a
        model reads the one with no note as mandatory and writes a
        filter where inheriting the application's own was both correct
        and safer. The sentence has to live HERE, in the static text
        both surfaces share; test_mcp.py pins that the server's own
        rewrite does not lose it."""
        spec = TRAVERSE_TOOL_SCHEMA["parameters"]["$defs"]["rerank"]
        assert "required" not in spec
        assert "NEITHER KEY BELOW IS REQUIRED" in spec["description"]
        assert "Leave it out" in spec["properties"]["document_from"]["description"]

    def test_the_rerank_object_states_the_step_rule_for_candidates(self):
        """`candidates` used to say only "at least keep/k". On a
        traversal step it has to be MORE than `keep` -- equality is a
        billed no-op there, because the order is discarded -- so a model
        following the old sentence wrote a call that refuses."""
        description = TRAVERSE_TOOL_SCHEMA["parameters"]["$defs"]["rerank"][
            "properties"]["candidates"]["description"]
        assert "MORE than `keep`" in description
        assert "at least `k`" in description

    def test_tool_schemas_change_descriptions_only(self):
        """The parsers accept exactly what they accepted before, so the
        parameters sections must be byte-identical to the constants --
        the DECLARED SCHEMA is presentation here, not grammar.

        Read with rerank=True, because that is the claim: summarizing a
        graph's vocabulary changes descriptions and nothing else. The one
        grammar difference tool_schemas() makes is taking `rerank` off
        when the caller has no reranker, which is a different decision
        and is pinned by its own two tests above."""
        graph = self.offgraph()
        self.declare(graph)
        from hopai import INGEST_TOOL_SCHEMA, MUTATE_TOOL_SCHEMA
        for tool, constant in zip(graph.tool_schemas(rerank=True),
                                  (TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA,
                                   INGEST_TOOL_SCHEMA, MUTATE_TOOL_SCHEMA), strict=True):
            assert tool["parameters"] == constant["parameters"]
            assert tool["name"] == constant["name"]
            assert tool["description"].startswith(constant["description"])

    def test_tool_summary_is_pinned_verbatim(self):
        """PR #25's CI surfaced this summary's string mutants one id at
        a time -- fragment-level pins let every unmatched separator and
        label drift. The hop.py rule applies: the WHOLE rendered summary
        is asserted. If you reword it, update this test."""
        graph = self.offgraph()
        self.declare(graph)
        assert graph.tool_schemas()[0]["description"].endswith(
            "This graph's declared schema (properties in parentheses, * = required, "
            "! = unique). "
            "Node types: person(email*, age); company(name*). "
            "Edge kinds (source -> target): works_at: person -> company (since); "
            "likes: person -> person, person -> company."
        )

    def test_tool_summary_is_bounded(self):
        """Prompt budget is someone else's money: a type with many
        properties lists a capped set plus an explicit overflow marker,
        never the whole inventory."""
        from hopai import NodeType, Property
        graph = self.offgraph()
        graph.define_schema(nodes=[NodeType("wide", properties=[
            Property(f"p{i:02d}", "string") for i in range(15)])])
        (description,) = {t["description"] for t in graph.tool_schemas() if t["name"] == "traverse_graph"}
        assert "+3 more" in description
        assert "p11" in description and "p12" not in description

    def test_tool_summary_edges_of_the_property_bag(self):
        """The bag's boundary cases, each a mutation-run survivor until
        pinned: the ! unique marker actually renders; exactly-at-the-cap
        shows no '+0 more'; one past the cap says '+1 more' instead of
        silently truncating; and a property-less type is its bare name,
        with nothing appended."""
        from hopai import NodeType, Property
        from hopai.schema import tool_summary
        summary = tool_summary(self.offgraph().define_schema(nodes=[
            NodeType("keyed", properties=[Property("sku", "string", unique=True)]),
            NodeType("at_cap", properties=[
                Property(f"c{i:02d}", "string") for i in range(12)]),
            NodeType("over_cap", properties=[
                Property(f"o{i:02d}", "string") for i in range(13)]),
            NodeType("bare"),
        ]))
        assert "keyed(sku!)" in summary
        assert "c11)" in summary and "more" not in summary.split("at_cap", 1)[1].split(";")[0]
        assert "+1 more" in summary
        assert summary.endswith("bare.")

    def test_no_declared_vectors_leaves_the_four_tools_unchanged(self):
        """Issue #51's other half of the same contract this class already
        pins for a schema: appending VECTOR_SEARCH_TOOL_SCHEMA must not
        become unconditional. A graph with a declared SCHEMA but no
        declared VECTORS is the case a schema-only check would miss --
        so this graph is not offgraph()'s bare handle, it is one with a
        real schema, and the count still has to stay four."""
        from hopai import INGEST_TOOL_SCHEMA, MUTATE_TOOL_SCHEMA
        graph = self.offgraph()
        self.declare(graph)
        tools = graph.tool_schemas()
        assert [t["name"] for t in tools] == [
            TRAVERSE_TOOL_SCHEMA["name"], AGGREGATE_TOOL_SCHEMA["name"],
            INGEST_TOOL_SCHEMA["name"], MUTATE_TOOL_SCHEMA["name"],
        ]

    def test_one_declared_vector_field_narrows_the_enum_to_it(self):
        """VECTOR_SEARCH_TOOL_SCHEMA's static `field` is a bare string --
        a model would have to guess a name only the application knows.
        Appended by tool_schemas(), it must narrow to exactly this
        graph's one declared field, the same treatment `search_field`
        already gets in hopai.mcp (issue #51's point: one shared answer
        to "what can be searched here")."""
        from hopai import Vector
        graph = self.offgraph()
        graph.define_vectors(nodes=[Vector("summary", 3)])
        tools = graph.tool_schemas()
        search = next(t for t in tools if t["name"] == "search_graph_by_meaning")
        assert len(tools) == 5
        field = search["parameters"]["$defs"]["near"]["properties"]["field"]
        assert field["enum"] == ["summary"]

    def test_several_fields_across_nodes_and_edges_are_all_enumerated(self):
        """One `near` $def is shared by every `target`, so the enum
        cannot be scoped per target inside the schema -- it has to be
        the union, and this pins that a node field and an edge field
        both make it in."""
        from hopai import Vector
        graph = self.offgraph()
        graph.define_vectors(nodes=[Vector("summary", 3), Vector("title", 3)],
                             edges=[Vector("rel", 3)])
        search = next(t for t in graph.tool_schemas()
                      if t["name"] == "search_graph_by_meaning")
        field = search["parameters"]["$defs"]["near"]["properties"]["field"]
        assert field["enum"] == ["rel", "summary", "title"]

    def test_the_search_tool_is_a_fresh_copy_not_the_module_constant(self):
        """The same non-negotiable test_tool_schemas_without_a_schema_
        are_the_constants_uncoupled() holds the other four to: mutating
        what tool_schemas() hands back must never corrupt
        VECTOR_SEARCH_TOOL_SCHEMA for every OTHER integration in the
        process."""
        from hopai import VECTOR_SEARCH_TOOL_SCHEMA, Vector
        before = json.dumps(VECTOR_SEARCH_TOOL_SCHEMA, sort_keys=True)
        graph = self.offgraph()
        graph.define_vectors(nodes=[Vector("summary", 3)])
        search = next(t for t in graph.tool_schemas()
                      if t["name"] == "search_graph_by_meaning")
        search["parameters"]["$defs"]["near"]["properties"]["field"]["enum"].clear()
        assert json.dumps(VECTOR_SEARCH_TOOL_SCHEMA, sort_keys=True) == before


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
