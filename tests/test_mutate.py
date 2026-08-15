"""
Test suite for hopai.mutate -- deleting and updating.

Split like the write tests: everything up to the emitted SQL needs no
database, execution does.

What is being tested is mostly damage control. A traversal that gets it
wrong returns the wrong rows and you look again; a delete that gets it
wrong is gone. So the refusals (no filter, contradictory arguments, a
Cypher shape whose meaning differs from ours) get as much attention as
the happy paths.

Graph scoping is checked twice over: the emitted SQL carries the
discriminator here (TestStatementShape), and the rows of a second graph
survive a delete in tests/test_multi_graph.py::TestScopedMutations --
the failure this design can produce is silent, so it is worth both.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy.dialects import postgresql

from hopai import (
    GT, MUTATE_TOOL_SCHEMA, NOT, OR, ConstraintViolation, CypherError, Hop,
    MutationResult, PropertyType, Required, Start, cypher_to_mutations,
    spec_to_mutations,
)
from hopai.mutate import Mutator


def norm(statement) -> str:
    return " ".join(str(statement.compile(
        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})).split())


def _plain_tables(offline_graph, graph_col):
    """A Graph over tables named nothing like hopai's own, so a
    hardcoded "start_id" or "id" shows up immediately."""
    from sqlalchemy import BigInteger, Column, MetaData, Table, Text
    from sqlalchemy.dialects.postgresql import JSONB

    from hopai import Graph
    meta = MetaData()
    scope = lambda: [] if graph_col is None else [Column(graph_col, Text)]   # noqa: E731
    nodes = Table("v", meta, Column("node_key", BigInteger, primary_key=True),
                  *scope(), Column("properties", JSONB))
    edges = Table("e", meta, Column("edge_key", BigInteger, primary_key=True),
                  *scope(), Column("src", BigInteger),
                  Column("dst", BigInteger), Column("properties", JSONB))
    return Graph(offline_graph.engine, node_table=nodes, edge_table=edges,
                 node_id_col="node_key", edge_id_col="edge_key",
                 edge_start_col="src", edge_end_col="dst", graph_col=graph_col)


def properties_of(graph, **where) -> list:
    return [n["properties"] for n in graph.traverse(Start(where=where or None)).nodes]


def names(graph) -> set:
    return {p.get("name") for p in properties_of(graph)}


def kinds(graph) -> list:
    """Every edge in the graph, by kind. Read through a traversal, so a
    delete that only *seemed* to work still shows up here."""
    return sorted(e["properties"].get("kind") for e in graph.traverse(Start(), Hop()).edges)


@pytest.fixture()
def people(fresh_graph):
    """Alice -> Bob -> Carol, plus a company nobody points at.

        (Alice)-[:knows]->(Bob)-[:knows]->(Carol)
        (Alice)-[:works_at]->(Acme)
    """
    fresh_graph.ingest({
        "nodes": [
            {"type": "person", "name": "Alice", "age": 34, "nickname": "Al"},
            {"type": "person", "name": "Bob", "age": 71},
            {"type": "person", "name": "Carol", "age": 25},
            {"type": "company", "name": "Acme"},
        ],
        "edges": [
            {"start": {"name": "Alice"}, "end": {"name": "Bob"}, "kind": "knows"},
            {"start": {"name": "Bob"}, "end": {"name": "Carol"}, "kind": "knows"},
            {"start": {"name": "Alice"}, "end": {"name": "Acme"}, "kind": "works_at"},
        ],
    })
    return fresh_graph


# ---------------------------------------------------------------------
# Deleting
# ---------------------------------------------------------------------

class TestDeleteNodes:
    def test_deletes_only_what_the_filter_matches(self, people):
        result = people.delete_nodes(where={"name": "Carol"}, detach=True)
        assert (result.deleted_nodes, result.deleted_edges) == (1, 1)
        assert names(people) == {"Alice", "Bob", "Acme"}

    def test_a_node_with_edges_is_refused_and_the_message_names_detach(self, people):
        """Postgres says "still referenced from table edges", which
        leaves the caller to work out both that edges reference nodes
        and what the flag is called."""
        with pytest.raises(ConstraintViolation) as exc:
            people.delete_nodes(where={"name": "Bob"})
        # The fix leads; the driver's text is kept as the tail because it
        # names the constraint and the row. Both halves are asserted:
        # blanking the structured fields left the suite green.
        assert str(exc.value).startswith("cannot delete a node that still has edges")
        assert "detach=True" in str(exc.value)
        assert exc.value.constraint and exc.value.detail
        assert "Bob" in names(people)

    def test_the_refusal_still_carries_the_driver_error(self, people):
        """The rewritten message is the useful half, but the original
        IntegrityError has to stay reachable -- it names the constraint
        and the row, which is what anyone debugging a foreign key
        actually needs."""
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(ConstraintViolation) as exc:
            people.delete_nodes(where={"name": "Bob"})
        assert isinstance(exc.value.__cause__, IntegrityError)

    def test_a_detached_delete_that_fails_puts_the_edges_back(self, people):
        """The edge delete and the node delete are one transaction, so a
        failure after the edges are gone has to bring them back.

        Provoked with a second operation that fails: the previous
        version of this test called delete_nodes(detach=False), which
        never runs an edge delete at all and so could not fail however
        the transaction was scoped."""
        with pytest.raises(ConstraintViolation):
            people.mutate({"operations": [
                {"op": "delete_nodes", "where": {"name": "Bob"}, "detach": True},
                {"op": "delete_nodes", "where": {"name": "Alice"}},   # still has works_at
            ]})
        assert kinds(people) == ["knows", "knows", "works_at"]
        assert names(people) == {"Alice", "Bob", "Carol", "Acme"}

    def test_a_self_loop_is_detached_from_both_sides(self, fresh_graph):
        """detach matches incident edges with OR over both endpoints; a
        self-loop is the row that a start-only simplification would
        leave behind to fail the foreign key."""
        fresh_graph.ingest({"nodes": [{"name": "solo"}],
                            "edges": [{"start": {"name": "solo"}, "end": {"name": "solo"},
                                       "kind": "self"}]})
        assert fresh_graph.delete_nodes(where={"name": "solo"},
                                        detach=True).to_dict()["deleted_edges"] == 1

    def test_detach_deletes_edges_on_both_sides(self, people):
        """Bob is the end of one edge and the start of another; missing
        either direction would leave the foreign key to fail on."""
        result = people.delete_nodes(where={"name": "Bob"}, detach=True)
        assert (result.deleted_nodes, result.deleted_edges) == (1, 2)
        assert kinds(people) == ["works_at"]

    def test_detach_does_not_delete_the_nodes_at_the_other_end(self, people):
        people.delete_nodes(where={"name": "Bob"}, detach=True)
        assert names(people) == {"Alice", "Carol", "Acme"}

    def test_a_filter_matching_nothing_deletes_nothing(self, people):
        assert people.delete_nodes(where={"name": "Nobody"}).deleted_nodes == 0
        assert len(properties_of(people)) == 4

    def test_the_whole_filter_language_is_available(self, people):
        """Same resolve() as a traversal, so this needs no separate
        filter support -- and that is the point being pinned."""
        assert people.delete_nodes(where=GT("age", 70), detach=True).deleted_nodes == 1
        assert people.delete_nodes(where=OR({"name": "Carol"},
                                            {"name": "Acme"}), detach=True).deleted_nodes == 2
        assert names(people) == {"Alice"}

    def test_a_negated_filter_deletes_rows_missing_the_key(self, people):
        """NOT is containment-based here, so rows that lack the key
        entirely match it: only Alice has a nickname, and everyone else
        is deleted. Cypher's `nickname <> 'Al'` evaluates to NULL for
        those rows and would have kept them -- a documented divergence,
        and one that decides which rows survive a delete."""
        people.delete_nodes(where=NOT({"nickname": "Al"}), detach=True)
        assert names(people) == {"Alice"}


class TestDeleteEdges:
    def test_deletes_by_property(self, people):
        assert people.delete_edges(where={"kind": "knows"}).deleted_edges == 2
        assert kinds(people) == ["works_at"]

    def test_leaves_the_nodes_it_disconnected(self, people):
        people.delete_edges(where={"kind": "knows"})
        assert names(people) == {"Alice", "Bob", "Carol", "Acme"}

    def test_an_endpoint_filter_narrows_to_one_side(self, people):
        """"Alice's knows edges" without fetching Alice's id first --
        one statement, so nothing can change between the lookup and the
        delete."""
        assert people.delete_edges(where={"kind": "knows"},
                                   start={"name": "Alice"}).deleted_edges == 1
        assert kinds(people) == ["knows", "works_at"]

    def test_both_endpoints_may_be_filtered(self, people):
        assert people.delete_edges(start={"name": "Alice"},
                                   end={"name": "Acme"}).deleted_edges == 1
        assert kinds(people) == ["knows", "knows"]

    def test_endpoint_filters_are_directional(self, people):
        """Bob -> Carol exists, Carol -> Bob does not."""
        assert people.delete_edges(start={"name": "Carol"}, end={"name": "Bob"}).deleted_edges == 0
        assert people.delete_edges(start={"name": "Bob"}, end={"name": "Carol"}).deleted_edges == 1


class TestClear:
    def test_empties_the_graph(self, people):
        result = people.clear()
        assert (result.deleted_nodes, result.deleted_edges) == (4, 3)
        assert properties_of(people) == []

    def test_needs_no_detach_argument(self, people):
        """Every edge belongs to a node being deleted, so the ordering
        clear() does internally is the whole answer."""
        people.clear()
        assert people.traverse(Start(), Hop()).edges == []

    def test_is_idempotent(self, people):
        people.clear()
        assert people.clear().to_dict()["deleted_nodes"] == 0


# ---------------------------------------------------------------------
# Updating
# ---------------------------------------------------------------------

class TestUpdateNodes:
    def test_set_merges_over_what_is_there(self, people):
        assert people.update_nodes(where={"name": "Alice"},
                                   set={"active": False}).updated_nodes == 1
        assert properties_of(people, name="Alice")[0] == {
            "type": "person", "name": "Alice", "age": 34, "nickname": "Al", "active": False}

    def test_set_overwrites_a_property_it_names(self, people):
        people.update_nodes(where={"name": "Alice"}, set={"age": 35})
        assert properties_of(people, name="Alice")[0]["age"] == 35

    def test_replace_makes_set_the_whole_bag(self, people):
        people.update_nodes(where={"name": "Alice"},
                            set={"type": "person", "name": "Alice"}, replace=True)
        assert properties_of(people, name="Alice")[0] == {"type": "person", "name": "Alice"}

    def test_remove_drops_only_the_named_keys(self, people):
        assert people.update_nodes(where={"name": "Alice"},
                                   remove=["nickname"]).updated_nodes == 1
        assert properties_of(people, name="Alice")[0] == {
            "type": "person", "name": "Alice", "age": 34}

    def test_removing_a_key_that_is_not_there_is_not_an_error(self, people):
        people.update_nodes(where={"name": "Bob"}, remove=["nickname"])
        assert properties_of(people, name="Bob")[0] == {"type": "person", "name": "Bob", "age": 71}

    def test_set_and_remove_apply_together(self, people):
        people.update_nodes(where={"name": "Alice"}, set={"active": True}, remove=["nickname"])
        assert properties_of(people, name="Alice")[0] == {
            "type": "person", "name": "Alice", "age": 34, "active": True}

    def test_every_matched_row_is_updated(self, people):
        """An update matches a set of rows, exactly like Cypher's SET --
        matching one would be a different, much smaller feature."""
        assert people.update_nodes(where={"type": "person"},
                                   set={"checked": True}).updated_nodes == 3
        assert all(p["checked"] for p in properties_of(people, type="person"))

    def test_a_filter_matching_nothing_updates_nothing(self, people):
        assert people.update_nodes(where={"name": "Nobody"}, set={"x": 1}).updated_nodes == 0
        assert all("x" not in p for p in properties_of(people))

    def test_hostile_property_keys_can_be_removed(self, fresh_graph):
        """remove= builds the one hand-written SQL fragment in the
        module (`jsonb - text[]`), so the keys go through bound
        parameters like everything else."""
        keys = ["100% \\ weird \" quotes ' -- ;", "日本語", "x'; DROP TABLE nodes; --"]
        fresh_graph.add_nodes([{"keep": 1, **dict.fromkeys(keys, "v")}])
        assert fresh_graph.update_nodes(where={"keep": 1}, remove=keys).updated_nodes == 1
        assert properties_of(fresh_graph) == [{"keep": 1}]

    def test_an_update_can_violate_a_declared_constraint(self, fresh_graph):
        """The caller declared PropertyType, and an UPDATE is as capable
        of breaking it as an INSERT -- so it gets the same named
        ConstraintViolation rather than a driver traceback."""
        fresh_graph.define_constraints(nodes=[Required("type"), PropertyType("age", "number")])
        fresh_graph.add_nodes([{"type": "person", "age": 34}])
        with pytest.raises(ConstraintViolation, match="rejected by constraint"):
            fresh_graph.update_nodes(where={"type": "person"}, set={"age": "thirty-four"})
        with pytest.raises(ConstraintViolation):
            fresh_graph.update_nodes(where={"type": "person"}, remove=["type"])
        assert properties_of(fresh_graph)[0] == {"type": "person", "age": 34}


class TestUpdateEdges:
    def test_set_merges_on_edges_too(self, people):
        assert people.update_edges(where={"kind": "knows"}, set={"weight": 2}).updated_edges == 2
        weights = [e["properties"].get("weight")
                   for e in people.traverse(Start(), Hop(via={"kind": "knows"})).edges]
        assert weights == [2, 2]

    def test_endpoint_filters_narrow_the_update(self, people):
        people.update_edges(where={"kind": "knows"}, start={"name": "Alice"}, set={"weight": 9})
        weighted = {e["properties"].get("weight")
                    for e in people.traverse(Start(), Hop(via={"kind": "knows"})).edges}
        assert weighted == {9, None}

    def test_an_end_filter_narrows_the_update(self, people):
        """`end=` was never executed by the suite -- dropping it from
        update_edges_statement left every test passing, because the only
        endpoint filter under test was `start=`."""
        people.update_edges(where={"kind": "knows"}, end={"name": "Carol"}, set={"weight": 9})
        weights = {e["properties"].get("name", e["properties"].get("weight"))
                   for e in people.traverse(Start(), Hop(via={"kind": "knows"})).edges}
        assert weights == {9, None}          # Bob->Carol only; Alice->Bob untouched

    def test_an_edge_update_can_violate_a_declared_constraint(self, fresh_graph):
        """The edge path translates its IntegrityError too -- an update
        rejected by a constraint the caller declared should name it, not
        surface as a driver error."""
        fresh_graph.define_constraints(edges=[PropertyType("weight", "number")])
        fresh_graph.ingest({"nodes": [{"n": 1}, {"n": 2}],
                            "edges": [{"start": {"n": 1}, "end": {"n": 2}, "kind": "knows"}]})
        with pytest.raises(ConstraintViolation, match="edge rejected by constraint"):
            fresh_graph.update_edges(where={"kind": "knows"}, set={"weight": "heavy"})

    def test_replace_and_all_reach_the_edge_path(self, people):
        """The Graph facade forwards seven arguments per call, and only
        the ones a test passes are really wired: dropping `replace=` and
        `all=` from update_edges (and `all=` from delete_edges) left the
        whole suite green, because no test had ever sent them through
        the edge methods."""
        people.update_edges(where={"kind": "knows"}, set={"weight": 9})
        assert people.update_edges(set={"kind": "knew"}, replace=True,
                                   all=True).updated_edges == 3
        # Every bag is exactly the map: asserting on `kind` alone could
        # not tell replace from merge, since the map overwrites it either
        # way -- the weight is the property that has to be gone.
        assert [e["properties"] for e in people.traverse(Start(), Hop()).edges] == (
            [{"kind": "knew"}] * 3)
        assert people.delete_edges(all=True).deleted_edges == 3

    def test_remove_drops_an_edge_property(self, people):
        people.update_edges(where={"kind": "works_at"}, set={"since": 2019})
        people.update_edges(where={"kind": "works_at"}, remove=["since"])
        edges = [e["properties"] for e in people.traverse(Start(), Hop(via={"kind": "works_at"})).edges]
        assert edges == [{"kind": "works_at"}]


# ---------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------

class TestRefusals:
    @pytest.mark.parametrize("where", [None, {}])
    def test_a_delete_with_no_filter_refuses(self, people, where):
        """`where` arriving empty is what an unset variable looks like,
        and matching everything on it would be unrecoverable."""
        with pytest.raises(ValueError, match="no filter"):
            people.delete_nodes(where=where)
        assert len(properties_of(people)) == 4

    def test_the_no_filter_message_names_both_ways_to_mean_it(self, people):
        with pytest.raises(ValueError) as exc:
            people.delete_edges()
        assert "all=True" in str(exc.value) and "graph.clear()" in str(exc.value)

    def test_an_unfiltered_update_refuses_the_same_way(self, people):
        with pytest.raises(ValueError, match="no filter"):
            people.update_nodes(set={"checked": True})

    def test_all_is_the_way_to_mean_every_row(self, people):
        assert people.update_nodes(set={"checked": True}, all=True).updated_nodes == 4

    def test_all_together_with_a_filter_refuses(self, people):
        """One of the two is being ignored, and the caller believes
        otherwise."""
        with pytest.raises(ValueError, match="disagree"):
            people.delete_nodes(where={"type": "person"}, all=True)

    def test_an_endpoint_filter_counts_as_a_filter(self, people):
        """delete_edges(start=...) is filtered even with no `where`, so
        requiring all=True there would be noise."""
        assert people.delete_edges(start={"name": "Alice"}).deleted_edges == 2

    def test_an_update_that_changes_nothing_refuses(self, people):
        with pytest.raises(ValueError, match="nothing to change"):
            people.update_nodes(where={"type": "person"})

    def test_replace_with_no_properties_refuses(self, people):
        with pytest.raises(ValueError, match="erase every property"):
            people.update_nodes(where={"type": "person"}, replace=True)

    @pytest.mark.parametrize("arguments", [
        {"set": {"a": 1}, "remove": ["b"], "replace": True},
        {"remove": ["age"], "replace": True},
    ])
    def test_replace_and_remove_contradict_each_other(self, people, arguments):
        """Diagnosed as the contradiction it is. Checked before the
        "nothing to change" case, which used to shadow it and answer
        `replace=True, remove=[...]` with "...or use remove=[...] to drop
        specific keys" -- telling the caller to do what they had done."""
        with pytest.raises(ValueError, match="contradicts itself"):
            people.update_nodes(where={"type": "person"}, **arguments)

    def test_an_explicitly_empty_bag_is_not_the_same_as_no_argument(self, people):
        """`set={}` with replace means Cypher's `SET a = {}` -- clear
        every property. `set=None` is the argument not given at all, and
        erasing on that would be the accident the guards exist for."""
        assert people.update_nodes(where={"name": "Alice"}, set={},
                                   replace=True).updated_nodes == 1
        assert properties_of(people, name="Alice") == []      # no properties left to match on
        assert len(properties_of(people)) == 4
        with pytest.raises(ValueError, match="erase every property"):
            people.update_nodes(where={"type": "person"}, set=None, replace=True)

    @pytest.mark.parametrize("call,arguments", [
        ("delete_nodes", {"all": "false"}),
        ("delete_nodes", {"where": {"type": "person"}, "detach": 1}),
        ("update_nodes", {"where": {"type": "person"}, "set": {"x": 1}, "replace": "false"}),
    ])
    def test_a_flag_that_decides_destruction_is_checked_not_coerced(self, people, call,
                                                                    arguments):
        """JSON booleans arriving as the strings "true"/"false" is an
        ordinary tool-call failure, and "false" is truthy in Python -- so
        coercing let `{"op": "delete_nodes", "all": "false"}` mean "yes,
        every row" and empty the graph."""
        argument = next(k for k in ("all", "detach", "replace") if k in arguments)
        with pytest.raises(TypeError) as exc:
            getattr(people, call)(**arguments)
        # The message has to name the argument it is talking about: three
        # of them reach _flag(), and "takes True or False" alone leaves
        # the caller to work out which one they got wrong.
        assert str(exc.value).startswith(f"{call}({argument}=...) takes True or False")
        assert len(properties_of(people)) == 4

    def test_a_property_value_outside_json_is_refused_before_the_transaction(self, people):
        """json.dumps emits the non-JSON tokens NaN and Infinity happily,
        so this used to reach the driver as a DataError -- raised after
        the transaction opened, and in a document after earlier
        operations had already run."""
        with pytest.raises(TypeError, match="values must be JSON"):
            people.update_nodes(where={"type": "person"}, set={"score": float("nan")})

    def test_setting_and_removing_one_property_refuses(self, people):
        with pytest.raises(ValueError, match="both sets and removes"):
            people.update_nodes(where={"type": "person"}, set={"age": 1}, remove=["age"])

    def test_an_edge_refusal_names_the_edge_call(self, people):
        """Both update methods share _new_properties(), which is handed
        the caller's name for its messages -- and every test that
        provoked one used update_nodes, so the edge path could have
        reported itself as anything at all."""
        with pytest.raises(ValueError) as exc:
            people.update_edges(where={"kind": "knows"}, set={"x": 1}, remove=["x"])
        assert str(exc.value).startswith("update_edges() both sets and removes")

    @pytest.mark.parametrize("arguments,message", [
        ({"set": ["age"]}, "must be a dict"),
        ({"remove": "age"}, "not one string"),
        ({"remove": [3]}, "property names"),
    ])
    def test_argument_shapes_that_would_do_nothing_useful(self, people, arguments, message):
        with pytest.raises(TypeError, match=message):
            people.update_nodes(where={"type": "person"}, **arguments)

    def test_a_property_value_that_is_not_json_says_so(self, people):
        """json.dumps' own TypeError names a class nobody wrote down;
        this one names the argument."""
        with pytest.raises(TypeError, match="values must be JSON"):
            people.update_nodes(where={"type": "person"}, set={"seen": object()})

    def test_a_refusal_takes_no_connection(self, offline_graph):
        """The guards run before the transaction opens, which is why
        these are testable with no database at all."""
        with pytest.raises(ValueError, match="no filter"):
            offline_graph.delete_nodes()


# ---------------------------------------------------------------------
# The emitted SQL
# ---------------------------------------------------------------------

class TestStatementShape:
    def test_every_statement_carries_the_graph_discriminator(self, offline_graph):
        """Forgetting it does not error -- it deletes another graph's
        rows. Enumerated statement by statement for that reason."""
        graph = offline_graph.in_graph("marketing")
        mutator = Mutator(graph)
        statements = [
            mutator.delete_nodes_statement({"type": "person"}),
            mutator.detach_statement({"type": "person"}),
            mutator.delete_edges_statement({"kind": "knows"}, start={"name": "Alice"}),
            mutator.update_nodes_statement({"type": "person"}, set={"x": 1}),
            mutator.update_edges_statement({"kind": "knows"}, end={"name": "Bob"}, set={"x": 1}),
            mutator.delete_nodes_statement(all=True),
        ]
        # Counted, not merely present: the endpoint subquery carries the
        # discriminator too, so a substring test stayed green with the
        # scope dropped from the outer DELETE of detach_statement.
        for statement, occurrences in zip(statements, (1, 3, 2, 1, 2, 1), strict=True):
            assert norm(statement).count("graph_id = 'marketing'") == occurrences

    def test_an_endpoint_subquery_is_scoped_too(self, offline_graph):
        """Two occurrences: the edges being deleted, and the nodes the
        endpoint filter resolves against."""
        sql = norm(Mutator(offline_graph.in_graph("support")).delete_edges_statement(
            {"kind": "knows"}, start={"name": "Alice"}))
        assert sql.count("graph_id = 'support'") == 2

    def test_set_merges_and_replace_assigns(self, offline_graph):
        mutator = Mutator(offline_graph)
        merged = norm(mutator.update_nodes_statement({"type": "person"}, set={"x": 1}))
        replaced = norm(mutator.update_nodes_statement({"type": "person"}, set={"x": 1},
                                                       replace=True))
        assert "SET properties=(nodes.properties || CAST('{\"x\": 1}' AS JSONB))" in merged
        assert "SET properties=CAST('{\"x\": 1}' AS JSONB)" in replaced

    def test_remove_is_one_typed_array_subtraction(self, offline_graph):
        """`jsonb - text[]` drops every key in one operation. Without the
        cast Postgres cannot tell it from `jsonb - text`, which takes a
        single key and would silently drop only the first."""
        sql = norm(Mutator(offline_graph).update_nodes_statement(
            {"type": "person"}, remove=["a", "b"]))
        assert "properties - CAST(ARRAY['a', 'b'] AS TEXT[])" in sql

    def test_detach_matches_edges_at_either_end(self, offline_graph):
        sql = norm(Mutator(offline_graph).detach_statement({"type": "person"}))
        assert "start_id IN" in sql and "end_id IN" in sql and " OR " in sql

    def test_a_blank_filter_is_left_out_rather_than_resolved_to_true(self, offline_graph):
        """all=True is the only way to reach this, and `AND true` is
        noise a reader has to look twice at."""
        assert "true" not in norm(Mutator(offline_graph).delete_edges_statement(all=True))

    def test_tables_without_a_discriminator_carry_no_scope_at_all(self, offline_graph):
        """graph_col=None means one graph, so the predicate is the
        filter alone -- `WHERE true AND ...` is noise, and `and_()` over
        nothing raises."""
        graph = _plain_tables(offline_graph, graph_col=None)
        mutator = Mutator(graph)
        assert norm(mutator.delete_nodes_statement({"type": "person"})) == (
            'DELETE FROM v WHERE v.properties @> CAST(\'{"type": "person"}\' AS JSONB)')
        assert norm(mutator.delete_edges_statement(all=True)) == "DELETE FROM e"
        with pytest.raises(ValueError, match=r"match every node in these tables\."):
            mutator.delete_nodes_statement()

    def test_custom_column_names_are_honoured(self, offline_graph):
        """Nothing may hardcode "start_id" -- the tables are the
        caller's."""
        graph = _plain_tables(offline_graph, graph_col="g")
        sql = norm(Mutator(graph).delete_edges_statement({"kind": "knows"}, start={"name": "A"}))
        assert "DELETE FROM e" in sql and "e.src IN (SELECT v.node_key" in sql


# ---------------------------------------------------------------------
# The JSON front end
# ---------------------------------------------------------------------

class TestJsonDocument:
    def test_operations_run_in_order_in_one_transaction(self, people):
        result = people.mutate({"operations": [
            {"op": "update_nodes", "where": {"type": "person"}, "set": {"checked": True}},
            {"op": "delete_edges", "where": {"kind": "works_at"}},
            {"op": "delete_nodes", "where": {"name": "Acme"}},
        ]})
        assert result.to_dict() == {
            "deleted_nodes": 1, "deleted_edges": 1, "updated_nodes": 3,
            "updated_edges": 0, "elapsed_ms": pytest.approx(0, abs=10_000)}
        assert names(people) == {"Alice", "Bob", "Carol"}

    def test_a_failing_operation_rolls_back_the_ones_before_it(self, people):
        """Half a plan is the state neither the caller nor a retry can
        reason about -- the same rule ingest() follows."""
        with pytest.raises(ConstraintViolation):
            people.mutate({"operations": [
                {"op": "update_nodes", "where": {"type": "person"}, "set": {"checked": True}},
                {"op": "delete_nodes", "where": {"name": "Bob"}},   # still has edges
            ]})
        assert "checked" not in properties_of(people, name="Alice")[0]
        assert "Bob" in names(people)

    def test_json_filters_are_the_json_grammar(self, people):
        """The operator forms come in as JSON and become the same filter
        objects the Python API takes, so both notations meet at
        resolve()."""
        people.mutate({"operations": [
            {"op": "delete_nodes", "where": {"gt": ["age", 70]}, "detach": True},
        ]})
        assert names(people) == {"Alice", "Carol", "Acme"}

    def test_endpoint_filters_take_the_json_grammar_too(self, people):
        """`start`/`end` are parsed like `where`, not passed through:
        with only `where` parsed, every test still passed, because plain
        dicts need no parsing and no test used an operator form here."""
        assert people.mutate({"operations": [
            {"op": "delete_edges", "start": {"gt": ["age", 70]}},
        ]}).deleted_edges == 1
        assert kinds(people) == ["knows", "works_at"]

    def test_an_edge_update_is_counted_in_the_document_total(self, people):
        """update_edges was never executed through a plan: deleting the
        line that accumulates its count left the whole suite green."""
        result = people.mutate({"operations": [
            {"op": "update_edges", "where": {"kind": "knows"}, "set": {"weight": 2}}]})
        assert result.updated_edges == 2
        assert [e["properties"].get("weight")
                for e in people.traverse(Start(), Hop(via={"kind": "knows"})).edges] == [2, 2]

    def test_all_and_replace_travel_through_a_document(self, people):
        """Neither flag was exercised by any document, so a per-operation
        key could be dropped from _OP_KEYS and only the union test
        noticed -- which it did not, being a union."""
        result = people.mutate({"operations": [
            {"op": "update_nodes", "where": {"name": "Alice"},
             "set": {"type": "person", "name": "Alice"}, "replace": True},
            {"op": "delete_edges", "all": True},
        ]})
        assert (result.updated_nodes, result.deleted_edges) == (1, 3)
        assert properties_of(people, name="Alice") == [{"type": "person", "name": "Alice"}]

    def test_a_flag_arriving_as_a_string_is_refused(self, people):
        """The tool-call failure this front end exists to survive: JSON
        booleans as the strings "true"/"false". "false" is truthy in
        Python, so this used to mean "yes, every row"."""
        with pytest.raises(TypeError, match="takes True or False"):
            people.mutate({"operations": [{"op": "delete_nodes", "all": "false"}]})
        assert len(properties_of(people)) == 4

    def test_the_plan_can_be_read_without_running_it(self, people):
        plan = spec_to_mutations({"operations": [
            {"op": "delete_nodes", "where": {"type": "draft"}, "detach": True}]})
        assert plan == [{"op": "delete_nodes", "where": {"type": "draft"}, "detach": True}]
        assert len(properties_of(people)) == 4

    @pytest.mark.parametrize("spec,message", [
        ({}, "non-empty"),
        ({"operations": []}, "non-empty"),
        ({"ops": [{"op": "delete_nodes"}]}, "unknown keys"),
        ({"operations": [{"op": "drop_everything"}]}, "unknown operation"),
        ({"operations": [{"op": "delete_nodes", "set": {"x": 1}}]}, "does not take"),
        ({"operations": [{"op": "update_nodes", "detach": True}]}, "does not take"),
        ({"operations": ["delete_nodes"]}, "must be an object"),
        ("delete everything", r"dict with an 'operations' list, got str"),
        (42, r"dict with an 'operations' list, got int"),
    ])
    def test_a_document_that_does_not_say_what_it_means_refuses(self, spec, message):
        with pytest.raises((ValueError, TypeError), match=message):
            spec_to_mutations(spec)

    def test_the_executor_refuses_an_operation_it_does_not_know(self, people):
        """spec_to_mutations() is not the only way to reach it -- a
        hand-built plan gets the same check."""
        with pytest.raises(ValueError, match="unknown operation"):
            people._mutator.execute_operations([{"op": "truncate"}])


def branches() -> dict:
    """The schema's per-operation branches, keyed by `op`."""
    return {b["properties"]["op"]["const"]: b
            for b in MUTATE_TOOL_SCHEMA["parameters"]["properties"]["operations"]
            ["items"]["oneOf"]}


class TestToolSchema:
    def test_is_json_serializable(self):
        assert json.loads(json.dumps(MUTATE_TOOL_SCHEMA)) == MUTATE_TOOL_SCHEMA

    def test_lists_exactly_the_operations_the_parser_accepts(self):
        from hopai.mutate import MUTATION_OPS
        assert set(branches()) == MUTATION_OPS

    def test_each_operation_describes_exactly_its_own_arguments(self):
        """Asserted per operation, not as a union: a union hides the
        removal of `all` from delete_nodes behind `all` still being
        listed on the three others, and a model emitting the documented
        {"op": "delete_nodes", "all": true} would then be told
        delete_nodes does not take it."""
        from hopai.mutate import _OP_KEYS
        for op, branch in branches().items():
            assert set(branch["properties"]) - {"op"} == set(_OP_KEYS[op])
            # The three keywords that make the branch mean anything to a
            # constrained decoder, none of which any other test reads.
            assert branch["type"] == "object"
            assert branch["required"] == ["op"]
            assert branch["additionalProperties"] is False

    def test_every_documented_argument_survives_the_parser(self):
        """The schema and the parser are two statements of one rule, so
        anything the schema offers has to parse."""
        for op, branch in branches().items():
            values = {"where": {"type": "x"}, "start": {"type": "x"}, "end": {"type": "x"},
                      "set": {"a": 1}, "remove": ["b"], "replace": False,
                      "detach": True, "all": False}
            operation = {"op": op, **{k: values[k] for k in branch["properties"] if k != "op"}}
            assert spec_to_mutations({"operations": [operation]})[0]["op"] == op

    def test_the_filter_grammar_is_spelled_out_not_referenced(self):
        """A caller may be handed mutate_graph and nothing else, so
        "same grammar as traverse_graph" would be a dangling pointer --
        and a model that cannot express a comparison gets a silent zero
        and reaches for the one lever left, which is `all`."""
        for key in ("where", "start", "end"):
            described = [b["properties"][key]["description"]
                         for b in branches().values() if key in b["properties"]]
            assert described and all('{"gt": [key, value]}' in d for d in described)

    def test_the_all_flag_documents_both_halves_of_its_rule(self):
        described = branches()["delete_nodes"]["properties"]["all"]["description"]
        assert "neither a filter nor this flag" in described and "with both" in described

    def test_a_document_shaped_like_the_schema_runs(self, people):
        assert people.mutate({"operations": [
            {"op": "update_nodes", "where": {"type": "person"}, "set": {"active": True},
             "remove": ["nickname"]},
            {"op": "delete_edges", "where": {"kind": "knows"}, "start": {"name": "Alice"}},
        ]}).to_dict()["updated_nodes"] == 3


class TestMutationResult:
    def test_reports_both_tables_a_detach_touched(self, people):
        assert people.delete_nodes(where={"name": "Bob"}, detach=True).to_dict() == {
            "deleted_nodes": 1, "deleted_edges": 2, "updated_nodes": 0,
            "updated_edges": 0, "elapsed_ms": pytest.approx(0, abs=10_000)}

    def test_repr_says_what_changed(self):
        assert repr(MutationResult(deleted_nodes=2, elapsed_ms=1.234)) == (
            "MutationResult(deleted_nodes=2, deleted_edges=0, updated_nodes=0, "
            "updated_edges=0, elapsed_ms=1.2)")


# ---------------------------------------------------------------------
# Cypher: DELETE, DETACH DELETE, SET, REMOVE
# ---------------------------------------------------------------------

class TestCypherTranslation:
    @pytest.mark.parametrize("query,plan", [
        ("MATCH (a:person {name: 'Alice'}) DELETE a",
         [{"op": "delete_nodes", "where": {"type": "person", "name": "Alice"}}]),
        ("MATCH (a:person) DETACH DELETE a",
         [{"op": "delete_nodes", "where": {"type": "person"}, "detach": True}]),
        ("MATCH (n) DETACH DELETE n",
         [{"op": "delete_nodes", "detach": True, "all": True}]),
        ("MATCH (a {name: 'Alice'})-[r:knows]->() DELETE r",
         [{"op": "delete_edges", "where": {"kind": "knows"}, "start": {"name": "Alice"}}]),
        ("MATCH ()-[r:knows]->(b:person) DELETE r",
         [{"op": "delete_edges", "where": {"kind": "knows"}, "end": {"type": "person"}}]),
        ("MATCH ()-[r]->() DELETE r", [{"op": "delete_edges", "all": True}]),
        ("MATCH (a:person) SET a.active = false",
         [{"op": "update_nodes", "where": {"type": "person"}, "set": {"active": False}}]),
        ("MATCH (a:person) SET a += {active: false}",
         [{"op": "update_nodes", "where": {"type": "person"}, "set": {"active": False}}]),
        ("MATCH (a:person) SET a = {type: 'person'}",
         [{"op": "update_nodes", "where": {"type": "person"}, "set": {"type": "person"},
           "replace": True}]),
        ("MATCH (a:person) REMOVE a.nickname, a.temp",
         [{"op": "update_nodes", "where": {"type": "person"},
           "remove": ["nickname", "temp"]}]),
        ("MATCH (a:person) SET a.x = 1 REMOVE a.y",
         [{"op": "update_nodes", "where": {"type": "person"}, "set": {"x": 1},
           "remove": ["y"]}]),
        ("MATCH (a:person) SET a = {type: 'person'}",
         [{"op": "update_nodes", "where": {"type": "person"}, "set": {"type": "person"},
           "replace": True}]),
        ("MATCH ()-[r:knows]->() SET r.weight = 2",
         [{"op": "update_edges", "where": {"kind": "knows"}, "set": {"weight": 2}}]),
        # An endpoint filter is a filter: deriving `all` from `where`
        # alone turned this into "every edge", which the executor then
        # refused as "all=True also got a filter".
        ("MATCH (a {name: 'Alice'})-[r]->() DELETE r",
         [{"op": "delete_edges", "start": {"name": "Alice"}}]),
        ("MATCH ()-[r]->(b:company) SET r.x = 1",
         [{"op": "update_edges", "end": {"type": "company"}, "set": {"x": 1}}]),
        # replace has to survive anything else touching the variable, or
        # a query that says "wipe the bag" quietly means "merge into it".
        ("MATCH (a:person) SET a = {type: 'person', y: 2}, a += {x: 1}",
         [{"op": "update_nodes", "where": {"type": "person"},
           "set": {"type": "person", "y": 2, "x": 1}, "replace": True}]),
        ("MATCH (a:person) SET a = {type: 'person', y: 2} SET a += {x: 1}",
         [{"op": "update_nodes", "where": {"type": "person"},
           "set": {"type": "person", "y": 2, "x": 1}, "replace": True}]),
        ("MATCH ()-[r:knows]->() SET r = {kind: 'knows', since: 2000}",
         [{"op": "update_edges", "where": {"kind": "knows"},
           "set": {"kind": "knows", "since": 2000}, "replace": True}]),
        # `= null` REMOVES a property in Cypher. Merging a JSON null
        # instead leaves a key Cypher calls absent and Required() calls
        # present -- the same intent walking past a declared constraint.
        ("MATCH (a:person) SET a.nickname = null",
         [{"op": "update_nodes", "where": {"type": "person"}, "remove": ["nickname"]}]),
        ("MATCH (a:person) SET a += {nickname: null, x: 1}",
         [{"op": "update_nodes", "where": {"type": "person"}, "set": {"x": 1},
           "remove": ["nickname"]}]),
        # Applied in order, last writer wins -- these two differ in
        # Cypher, and collecting SET and REMOVE independently made them
        # identical AND unexecutable (the executor refuses both on one key).
        ("MATCH (a:person) SET a.x = 1 REMOVE a.x",
         [{"op": "update_nodes", "where": {"type": "person"}, "remove": ["x"]}]),
        ("MATCH (a:person) REMOVE a.x SET a.x = 1",
         [{"op": "update_nodes", "where": {"type": "person"}, "set": {"x": 1}}]),
        # Last writer wins when the last writer is a null: the property
        # has to leave `set` as well as enter `remove`, or the plan both
        # sets and removes one key and the executor refuses to run it.
        ("MATCH (a:person) SET a.x = 1 SET a.x = null",
         [{"op": "update_nodes", "where": {"type": "person"}, "remove": ["x"]}]),
        # A null inside a REPLACING map is simply a key the map does not
        # carry -- recording a removal as well would hand the executor a
        # plan it refuses, since replace and remove contradict.
        ("MATCH (a:person) SET a = {type: 'person', x: null}",
         [{"op": "update_nodes", "where": {"type": "person"}, "set": {"type": "person"},
           "replace": True}]),
        # REMOVE and the start-side endpoint filter, on the edge path:
        # both were dropped by a mutant with the suite still green.
        ("MATCH ()-[r:knows]->() REMOVE r.since",
         [{"op": "update_edges", "where": {"kind": "knows"}, "remove": ["since"]}]),
        ("MATCH (a {name: 'Alice'})-[r:knows]->() SET r.weight = 2",
         [{"op": "update_edges", "where": {"kind": "knows"}, "start": {"name": "Alice"},
           "set": {"weight": 2}}]),
    ])
    def test_translates_to_the_same_operations_the_python_api_runs(self, query, plan):
        assert cypher_to_mutations(query) == plan

    def test_a_backward_relationship_swaps_the_endpoints(self, ):
        """`<-` means the edge runs the other way, so the pattern's first
        node is the one it ends at. Getting this backwards deletes real
        edges that are not the ones asked for."""
        assert cypher_to_mutations("MATCH (a:person)<-[r:knows]-(b:company) DELETE r") == [
            {"op": "delete_edges", "where": {"kind": "knows"},
             "start": {"type": "company"}, "end": {"type": "person"}}]

    def test_where_is_anded_with_the_pattern_on_the_same_variable(self):
        """The pattern and the WHERE both constrain `a`, and both have to
        survive into the filter -- keeping only one would delete rows the
        query excluded."""
        plan = cypher_to_mutations("MATCH (a:person) WHERE a.age > 65 DELETE a")
        assert plan[0]["op"] == "delete_nodes"
        assert repr(plan[0]["where"]) == "AND({'type': 'person'}, GT('age', 65))"

    def test_alternation_on_a_relationship_is_an_or(self):
        assert cypher_to_mutations("MATCH ()-[r:knows|likes]->() DELETE r") == [
            {"op": "delete_edges", "where": {"kind": ["knows", "likes"]}}]

    @pytest.mark.parametrize("query,subject", [
        ("MATCH (a:person {type: 'company'}) DELETE a", "node a"),
        ("MATCH ()-[r:knows {kind: 'likes'}]->() DELETE r", "relationship r"),
        ("MATCH (a:person {type: 'company'})-[r]->() DELETE r", "node"),
        ("MATCH ()-[r]->(b:person {type: 'company'}) DELETE r", "node"),
    ])
    def test_a_pattern_nothing_can_satisfy_is_refused_by_name(self, query, subject):
        """The same refusal the read path makes -- an AND nothing can
        satisfy would delete nothing and report success -- and it says
        which part of the pattern is unsatisfiable, on every side."""
        with pytest.raises(CypherError) as exc:
            cypher_to_mutations(query)
        assert str(exc.value).startswith(subject)
        assert "which nothing can match" in str(exc.value)

    def test_relationship_properties_join_the_filter(self):
        assert cypher_to_mutations("MATCH ()-[r:knows {since: 2000}]->() DELETE r") == [
            {"op": "delete_edges", "where": {"kind": "knows", "since": 2000}}]

    def test_labels_can_be_ignored(self):
        assert cypher_to_mutations("MATCH (a:Node {type: 'leaf'}) DELETE a",
                                   node_label_key=None) == [
            {"op": "delete_nodes", "where": {"type": "leaf"}}]


class TestCypherRefusals:
    @pytest.mark.parametrize("query,message", [
        ("MATCH (a:person)-[:knows]->(b) DELETE b", "traversal driving a write"),
        ("MATCH (a)-[r:knows*1..3]->(b) DELETE r", "variable-length"),
        ("MATCH (a)-[r:knows]->(b) DELETE a, r", "^this query changes a, r at once"),
        ("MATCH (a)-[r]->(b) DETACH DELETE r", "is a relationship"),
        ("MATCH (a:person) SET a.x = 1 DELETE a", "both changes and deletes"),
        ("MATCH (a:person) SET a.x = 1, a = {y: 2}", "would be discarded"),
        ("MATCH (a:person) SET a = {y: 2} REMOVE a.z", "nothing to remove from"),
        ("MATCH (a:person) SET a:Employee", "no labels"),
        ("MATCH (a:person) REMOVE a:Employee", "no labels"),
        ("MATCH (a:person) DELETE b", "not bound by the MATCH"),
        ("MATCH (a:person), (b:company) DELETE a",
         "^comma-separated patterns before a delete or an update"),
        ("MATCH p = (a)-[r]->(b) DELETE r", "never a path"),
        ("OPTIONAL MATCH (a:person) DELETE a", "no meaning"),
        ("MATCH (a:person) MATCH (b:company) DELETE a", "one MATCH clause"),
        ("MATCH (a)-[r]->(b)-[q]->(c) DELETE r",
         "^a delete or an update matches one node or one relationship, not a multi-hop"),
        ("MATCH (a:person) DELETE a RETURN count(a)", "MutationResult, not a number"),
        ("MATCH (a:person) CREATE (b {x: 1}) DELETE a",
         "^a query that creates or merges and also deletes or updates"),
        ("MATCH (a:person)-[r]->(b) WHERE a.x = 1 OR b.y = 2 DELETE r",
         r"several variables \(a, b\)"),
        ("MATCH (a:person) WHERE b.x = 1 DELETE a", "unknown variable"),
        ("DELETE a", "no MATCH clause"),
        ("MATCH (a)-[a:knows]->(b) DELETE a", "used twice in this pattern"),
        ("MATCH (a:person:employee) DELETE a", "multiple labels"),
        ("MATCH (a)-[r]->(b) WHERE all(x IN relationships(p) WHERE x.k = 1) DELETE r",
         r"^all\(\.\.\. IN relationships\(p\) \.\.\.\) constrains the edges of a walk"),
        ("MATCH (a:person) SET a.x = 1 SET a = {y: 2}", "would be discarded"),
        ("MATCH (a:person) REMOVE a.x SET a = {y: 2}", "would be discarded"),
        ("MATCH (a:person) SET a 5", "expected a.property = value"),
    ])
    def test_what_does_not_translate_says_why(self, query, message):
        with pytest.raises(CypherError, match=message):
            cypher_to_mutations(query)

    def test_a_constraint_the_options_discard_names_what_it_discarded(self):
        """The message has to quote the pattern that was thrown away and
        the rewrite -- "something was discarded" leaves the caller to
        guess which half of their query stopped counting."""
        with pytest.raises(CypherError) as exc:
            cypher_to_mutations("MATCH ()-[r:knows|likes]->() DELETE r",
                                edge_type_key=None)
        assert str(exc.value).startswith("[:knows|likes] is the only thing constraining")
        assert "`MATCH ()-[r {kind: 'knows'}]->()`" in str(exc.value)

    @pytest.mark.parametrize("query,options,message", [
        ("MATCH (a:person) DETACH DELETE a", {"node_label_key": None},
         "would change every row instead of the ones it names"),
        ("MATCH ()-[r:knows]->() DELETE r", {"edge_type_key": None},
         "edge_type_key=None discards it"),
    ])
    def test_a_constraint_the_options_discard_refuses_rather_than_widening(
            self, query, options, message):
        """THE dangerous one. `node_label_key=None` is a supported
        option; on the read path it widens a result set, and here it
        widened a DELETE to the whole graph -- and then auto-supplied
        the `all=True` opt-in that exists to stop exactly that."""
        with pytest.raises(CypherError, match=message):
            cypher_to_mutations(query, **options)

    @pytest.mark.parametrize("query,var,subject,keep", [
        ("MATCH (a:person) SET a = {name: 'Alicia'}", "a", "a label", "type: 'person'"),
        ("MATCH (a:person) SET a = {}", "a", "a label", "type: 'person'"),
        ("MATCH (a {email: 'a@x.com'}) SET a = {name: 'Alicia'}", "a", "a label",
         "a type property"),
        ("MATCH ()-[r:knows]->() SET r = {since: 2000}", "r", "a relationship's type",
         "kind: 'knows'"),
    ])
    def test_a_replacing_map_may_not_erase_the_label_or_the_type(self, query, var,
                                                                 subject, keep):
        """In Cypher `SET n = {map}` replaces PROPERTIES: labels survive
        it and a relationship's type cannot be changed at all. Here both
        are properties, so the same query erased the discriminator --
        leaving a node no `(a:person)` matches and an edge no
        `[:knows]` finds. Cypher guarantees this query is
        non-destructive, so translating it is not an option."""
        with pytest.raises(CypherError) as exc:
            cypher_to_mutations(query)
        assert str(exc.value).startswith(f"`SET {var} = {{...}}` replaces every property")
        assert f"and {subject} is the property" in str(exc.value)
        assert keep in str(exc.value) and f"SET {var} += " in str(exc.value)

    @pytest.mark.parametrize("query,message", [
        ("MERGE (a:person {email: 'c@x.com'}) SET a.name = 'Carol'", "ON CREATE SET"),
        ("CREATE (a:person {email: 'c@x.com'}) SET a.name = 'Carol'",
         "put the properties in the pattern"),
    ])
    def test_a_bare_set_after_a_write_keeps_its_own_message(self, query, message):
        """Both are ordinary Cypher for "and give it these properties".
        The general mixing refusal sent the caller to split a query whose
        halves cannot be split -- the SET has nothing to match on its
        own."""
        from hopai import cypher_to_operations
        with pytest.raises(CypherError, match=message):
            cypher_to_operations(query)

    def test_a_merge_cannot_remove_a_property(self):
        from hopai import cypher_to_operations
        with pytest.raises(CypherError, match="removes a property in Cypher"):
            cypher_to_operations("MERGE (a {n: 1}) ON MATCH SET a.x = null")

    @pytest.mark.parametrize("query,rewrite", [
        ("MATCH (a:person)-[:knows]->(b) DELETE b", "`MATCH (b {...}) DETACH DELETE b`"),
        ("MATCH (a:person)-[:knows]->(b) SET b.x = 1", "`MATCH (b {...}) SET b.x = ...`"),
    ])
    def test_the_rewrite_matches_the_verb_the_caller_wrote(self, query, rewrite):
        """The example used to be hardcoded to DETACH DELETE, so someone
        who wrote SET was told to write a delete -- and with the example
        pinned only by the sentence before it, that could come back."""
        with pytest.raises(CypherError) as exc:
            cypher_to_mutations(query)
        assert rewrite in str(exc.value)

    def test_a_read_query_is_refused_by_the_mutation_translator(self):
        with pytest.raises(CypherError, match="deletes and updates nothing"):
            cypher_to_mutations("MATCH (a:person) RETURN a")

    def test_merge_still_refuses_the_map_that_replaces_everything(self):
        """`a += {...}` is merging, which is what ON MATCH SET already
        does; `a = {...}` is not, and a MERGE has nothing to replace
        with."""
        from hopai import cypher_to_operations
        assert cypher_to_operations(
            "MERGE (a {n: 1}) ON MATCH SET a += {x: 2}")[0]["on_match"] == {"x": 2}
        with pytest.raises(CypherError, match="replaces every property"):
            cypher_to_operations("MERGE (a {n: 1}) ON MATCH SET a = {x: 2}")


class TestVectorsSurviveMutations:
    """Vectors live in real `vec_*` columns, not in `properties` -- so a
    replacing update rewrites the bag beside them and leaves them
    standing. Worth pinning rather than assuming: `replace=True` reads
    like "replace the row", and a future `update_nodes` that helpfully
    cleared the vector columns too would silently cost an embedding
    nobody can regenerate from what is left."""

    @pytest.fixture()
    def vectored(self, fresh_graph):
        from hopai import Vector
        fresh_graph.define_vectors(nodes=[Vector("summary", 4)])
        fresh_graph.migrate_vectors()
        fresh_graph.add_nodes([{"id": 1, "type": "doc", "title": "a"},
                               {"id": 2, "type": "doc", "title": "b"}])
        fresh_graph.set_vectors(nodes=[{"id": 1, "summary": [1.0, 0.0, 0.0, 0.0]},
                                       {"id": 2, "summary": [0.0, 1.0, 0.0, 0.0]}])
        return fresh_graph

    def test_a_replacing_update_keeps_the_stored_vector(self, vectored):
        from hopai import Near
        vectored.update_nodes(where={"title": "a"},
                              set={"type": "doc", "title": "a2"}, replace=True)
        found = vectored.vector_search(Near("summary", [1.0, 0.0, 0.0, 0.0]), k=1)
        assert found[0]["id"] == "1"
        assert found[0]["properties"] == {"type": "doc", "title": "a2"}

    def test_deleting_a_node_takes_its_vector_with_it(self, vectored):
        from hopai import Near
        vectored.delete_nodes(where={"title": "b"}, detach=True)
        found = vectored.vector_search(Near("summary", [0.0, 1.0, 0.0, 0.0]), k=2)
        assert [row["id"] for row in found] == ["1"]


class TestStrictSchema:
    """`strict_schema=True` reaches the mutation path too. It matters
    more here than on the read side: a hallucinated label returns an
    empty subgraph there, which looks like a result, while a delete that
    matched nothing reports exactly what a correct delete of an
    already-clean graph reports."""

    @pytest.fixture()
    def declared(self, people):
        from hopai import EdgeType, NodeType, Property
        people.define_schema(
            nodes=[NodeType("person", properties=[Property("name", "string"),
                                                  Property("age", "number"),
                                                  Property("nickname", "string")]),
                   NodeType("company", properties=[Property("name", "string")])],
            edges=[EdgeType("knows", "person", "person",
                            properties=[Property("weight", "number")])])
        return people

    @pytest.mark.parametrize("query,message", [
        ("MATCH (a:persson) DETACH DELETE a", "unknown label 'persson'"),
        ("MATCH (a:person) SET a.nickmane = 'Al'", r"unknown property \['nickmane'\]"),
        ("MATCH (a:person) REMOVE a.nickmane", r"unknown property \['nickmane'\]"),
        ("MATCH ()-[r:knowss]->() DELETE r", "unknown relationship kind 'knowss'"),
        ("MATCH ()-[r:knows]->(b:persson) DELETE r", "unknown label 'persson'"),
    ])
    def test_vocabulary_the_schema_does_not_declare_is_refused(self, declared, query,
                                                               message):
        with pytest.raises(CypherError, match=message):
            declared.mutate_cypher(query, strict_schema=True)
        assert len(properties_of(declared)) == 4

    def test_a_declared_query_still_runs(self, declared):
        assert declared.mutate_cypher("MATCH (a:person) SET a.nickname = 'x'",
                                      strict_schema=True).updated_nodes == 3
        assert declared.mutate_cypher("MATCH ()-[r:knows]->() SET r.weight = 2",
                                      strict_schema=True).updated_edges == 2

    def test_the_discriminator_keys_reach_the_validator(self, fresh_graph):
        """node_label_key/edge_type_key have to travel with the schema:
        dropping edge_type_key from the validate_mutations() call left
        every test passing, because they all use the default 'kind'."""
        from hopai import EdgeType, NodeType, Property
        fresh_graph.define_schema(
            nodes=[NodeType("person", properties=[Property("name", "string")])],
            edges=[EdgeType("knows", "person", "person")])
        with pytest.raises(CypherError, match="unknown relationship kind 'knowss'"):
            fresh_graph.mutate_cypher("MATCH ()-[r:knowss]->() DELETE r",
                                      strict_schema=True, edge_type_key="rel")

    def test_strict_without_a_schema_names_the_fix(self, people):
        with pytest.raises(CypherError, match="define_schema"):
            people.mutate_cypher("MATCH (a:person) DELETE a", strict_schema=True)

    def test_graph_cypher_passes_the_option_through(self, declared):
        """The dispatching entry point is the one an agent uses, so the
        option has to survive the trip to the mutation translator."""
        with pytest.raises(CypherError, match="unknown label"):
            declared.cypher("MATCH (a:persson) DETACH DELETE a", strict_schema=True)


class TestCypherExecution:
    def test_delete_runs_end_to_end(self, people):
        result = people.cypher("MATCH (a:person {name: 'Bob'}) DETACH DELETE a")
        assert (result.deleted_nodes, result.deleted_edges) == (1, 2)
        assert names(people) == {"Alice", "Carol", "Acme"}

    def test_set_runs_end_to_end(self, people):
        assert people.cypher("MATCH (a:person) WHERE a.age > 30 "
                             "SET a.senior = true").updated_nodes == 2
        assert {p["name"] for p in properties_of(people) if p.get("senior")} == {"Alice", "Bob"}

    def test_deleting_a_relationship_runs_end_to_end(self, people):
        assert people.cypher(
            "MATCH (a {name: 'Alice'})-[r:knows]->() DELETE r").deleted_edges == 1
        assert kinds(people) == ["knows", "works_at"]

    def test_setting_a_relationship_property_runs_end_to_end(self, people):
        assert people.cypher("MATCH ()-[r:knows]->() SET r.weight = 2").updated_edges == 2
        assert all(e["properties"]["weight"] == 2
                   for e in people.traverse(Start(), Hop(via={"kind": "knows"})).edges)

    def test_setting_a_property_to_null_removes_it(self, people):
        """End to end, because the whole point is the stored row: a JSON
        null would still satisfy Required('nickname') and would not be
        found by `WHERE a.nickname IS NULL`."""
        assert people.cypher("MATCH (a:person) SET a.nickname = null").updated_nodes == 3
        assert properties_of(people, name="Alice")[0] == {
            "type": "person", "name": "Alice", "age": 34}

    def test_cypher_dispatches_on_what_the_query_does(self, people):
        """Four return types, decided by the query rather than by which
        method the caller remembered."""
        assert people.cypher("MATCH (a:person) RETURN count(a)") == {"count": 3}
        assert people.cypher("CREATE (d:draft {name: 'D'})").nodes == 1
        assert people.cypher("MATCH (d:draft) DELETE d").deleted_nodes == 1
        assert len(people.cypher("MATCH (a:person) RETURN a").nodes) == 3

    def test_a_cypher_delete_that_needs_detach_says_so(self, people):
        with pytest.raises(ConstraintViolation, match="DETACH DELETE"):
            people.cypher("MATCH (a:person {name: 'Bob'}) DELETE a")

    def test_the_plan_can_be_reviewed_before_it_runs(self, people):
        cypher_to_mutations("MATCH (n) DETACH DELETE n")
        assert len(properties_of(people)) == 4
