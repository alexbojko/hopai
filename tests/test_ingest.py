"""
Test suite for hopai.ingest.

The contract being tested is mostly about being unsurprising: two row
spellings that mean the same thing, ids that may be given or generated,
edges that may reference nodes by id or by property, and a document form
an agent can emit without being told our conventions. Anything ambiguous
raises instead of guessing, and each of those refusals is tested here
too.
"""

from __future__ import annotations

import json

import pytest

from sqlalchemy import text

from hopai import (
    INGEST_TOOL_SCHEMA, ConstraintViolation, Hop, IngestResult, Required, Start, Unique, Vector,
)
from hopai.ingest import BATCH_SIZE, split_row


def properties_of(graph, **where) -> list:
    return [n["properties"] for n in graph.traverse(Start(where=where or None)).nodes]


def count(graph, table="nodes") -> int:
    from sqlalchemy import func, select
    tbl = graph.nodes_tbl if table == "nodes" else graph.edges_tbl
    with graph.engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(tbl)).scalar()


# ---------------------------------------------------------------------
# Row shapes
# ---------------------------------------------------------------------

class TestRowShapes:
    def test_flat_and_nested_mean_the_same_thing(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 1, "type": "person", "name": "Alice"}])
        fresh_graph.add_nodes([{"id": 2, "properties": {"type": "person", "name": "Alice"}}])
        stored = properties_of(fresh_graph, type="person")
        assert stored[0] == stored[1] == {"type": "person", "name": "Alice"}

    def test_a_result_can_be_fed_straight_back(self, fresh_graph, graph):
        """The nested spelling is exactly what a traversal returns, so a
        subgraph from one graph loads into another with no reshaping."""
        result = graph.traverse(Start(where={"type": "leaf"}))
        fresh_graph.add_nodes(result.nodes)
        assert count(fresh_graph) == len(result.nodes)

    def test_mixing_the_two_spellings_is_refused(self, fresh_graph):
        """Silently dropping `name` here would lose data the caller
        believes it wrote."""
        with pytest.raises(ValueError, match="mixes both spellings"):
            fresh_graph.add_nodes([{"properties": {"type": "person"}, "name": "Alice"}])

    def test_reserved_keys_may_sit_beside_properties(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        fresh_graph.add_edges([{"id": 9, "start_id": 1, "end_id": 2,
                                "properties": {"kind": "knows"}}])
        assert count(fresh_graph, "edges") == 1

    @pytest.mark.parametrize("row,message", [
        ("not a dict", "must be a dict"),
        (42, "must be a dict"),
        ({"properties": "not a dict"}, "'properties' must be a dict"),
    ])
    def test_malformed_rows(self, fresh_graph, row, message):
        with pytest.raises((TypeError, ValueError), match=message):
            fresh_graph.add_nodes([row])

    def test_empty_properties_are_allowed(self, fresh_graph):
        assert fresh_graph.add_nodes([{}, {"properties": {}}, {"properties": None}]) == 3

    def test_split_row_is_the_documented_rule(self):
        assert split_row({"id": 1, "a": 2}, frozenset({"id", "properties"}), "node") \
            == ({"id": 1}, {"a": 2})
        assert split_row({"id": 1, "properties": {"a": 2}}, frozenset({"id", "properties"}), "node") \
            == ({"id": 1}, {"a": 2})

    def test_a_declared_vector_fields_floats_are_refused_not_stored(self, fresh_graph):
        """Without the fix these land in `properties` (issue #50):
        similarity silently finds nothing for the row, the invariant
        'vectors live in vec_* columns, never in properties' is broken
        from outside the engine, and every read echoes the floats back."""
        fresh_graph.define_vectors(nodes=[Vector("summary", 3)])
        fresh_graph.migrate_vectors()
        with pytest.raises(ValueError, match="set_vectors") as excinfo:
            fresh_graph.add_nodes([{"id": 1, "title": "raft", "summary": [1.0, 0.0, 0.0]}])
        message = str(excinfo.value)
        assert "add_nodes()" in message and "'summary'" in message and "declared vector field" in message
        assert count(fresh_graph) == 0

    def test_source_text_at_a_vector_field_name_still_writes_and_leaves_vector_null(self, fresh_graph):
        """The text path this fix must not touch: a string is exactly
        what source= reads and embed_stale() later fills the vector
        from -- it keeps writing to `properties`, unindexed, until then."""
        fresh_graph.define_vectors(nodes=[Vector("summary", 3)])
        fresh_graph.migrate_vectors()
        fresh_graph.add_nodes([{"id": 1, "title": "raft", "summary": "a paper about Raft"}])
        assert properties_of(fresh_graph, title="raft")[0]["summary"] == "a paper about Raft"
        with fresh_graph.engine.connect() as conn:
            assert conn.execute(text("SELECT vec_summary FROM nodes WHERE id = 1")).scalar() is None


# ---------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------

class TestIdentity:
    def test_ids_are_generated_when_omitted(self, fresh_graph):
        fresh_graph.add_nodes([{"type": "a"}, {"type": "b"}])
        ids = {n["id"] for n in fresh_graph.traverse(Start()).nodes}
        assert len(ids) == 2 and all(i.isdigit() for i in ids)

    def test_explicit_ids_are_kept(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 100, "type": "a"}])
        assert {n["id"] for n in fresh_graph.traverse(Start()).nodes} == {"100"}

    def test_generated_ids_do_not_collide_with_explicit_ones(self, fresh_graph):
        """The classic identity-column trap: rows inserted with explicit
        ids leave the sequence behind them, and the next generated id
        collides with one that already exists. Ingestion resyncs the
        sequence; without that this test fails on a duplicate key."""
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}, {"id": 3}])
        fresh_graph.add_nodes([{"type": "generated"}])
        assert count(fresh_graph) == 4

    def test_a_batch_may_not_half_specify_ids(self, fresh_graph):
        """executemany binds one parameter shape, so a mixed batch would
        insert NULL ids rather than generating them."""
        with pytest.raises(ValueError, match="explicit id"):
            fresh_graph.add_nodes([{"id": 1}, {"type": "no id"}])

    def test_edges_get_generated_ids_too(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        fresh_graph.add_edges([{"start_id": 1, "end_id": 2}])
        assert count(fresh_graph, "edges") == 1


# ---------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------

class TestEdges:
    def test_by_id(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        assert fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}]) == 1

    def test_by_property_reference(self, fresh_graph):
        """The case that matters for agents: whatever wrote the nodes
        does not know the generated ids, and should not have to ask."""
        fresh_graph.add_nodes([{"email": "a@x.com"}, {"email": "b@x.com"}])
        fresh_graph.add_edges([{"start": {"email": "a@x.com"},
                                "end": {"email": "b@x.com"}, "kind": "knows"}])
        edges = fresh_graph.traverse(Start(), Hop()).edges
        assert len(edges) == 1 and edges[0]["properties"] == {"kind": "knows"}

    def test_mixed_reference_styles_in_one_call(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 1, "email": "a@x.com"}, {"id": 2, "email": "b@x.com"}])
        assert fresh_graph.add_edges([{"start_id": 1, "end": {"email": "b@x.com"}}]) == 1

    def test_multi_property_reference(self, fresh_graph):
        fresh_graph.add_nodes([{"tenant": "t1", "slug": "a"}, {"tenant": "t2", "slug": "a"}])
        fresh_graph.add_edges([{"start": {"tenant": "t1", "slug": "a"},
                                "end": {"tenant": "t2", "slug": "a"}}])
        assert count(fresh_graph, "edges") == 1

    def test_reference_matching_nothing_is_refused(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 1}])
        with pytest.raises(ValueError, match="no node matches"):
            fresh_graph.add_edges([{"start_id": 1, "end": {"email": "ghost@x.com"}}])

    def test_ambiguous_reference_is_refused(self, fresh_graph):
        """Picking one of two would produce a graph that is wrong in a way
        nobody would notice for months."""
        fresh_graph.add_nodes([{"id": 9}])
        fresh_graph.add_nodes([{"type": "person"}, {"type": "person"}])
        with pytest.raises(ValueError, match="nodes match"):
            fresh_graph.add_edges([{"start": {"type": "person"}, "end_id": 9}])

    def test_empty_reference_is_refused(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 1}])
        with pytest.raises(ValueError, match="empty"):
            fresh_graph.add_edges([{"start": {}, "end_id": 1}])

    def test_missing_endpoint_is_refused(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 1}])
        with pytest.raises(ValueError, match="missing end_id"):
            fresh_graph.add_edges([{"start_id": 1}])

    def test_references_are_resolved_in_one_lookup(self, fresh_graph):
        """Distinct references are batched -- one query for the call, not
        one per edge. Asserted through behavior: many edges, all correct."""
        fresh_graph.add_nodes([{"n": i} for i in range(20)])
        fresh_graph.add_edges([{"start": {"n": i}, "end": {"n": i + 1}} for i in range(19)])
        assert count(fresh_graph, "edges") == 19

    def test_foreign_key_is_enforced(self, fresh_graph):
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_edges([{"start_id": 999, "end_id": 998}])

    def test_a_declared_vector_fields_floats_are_refused_on_add_edges(self, fresh_graph):
        fresh_graph.define_vectors(edges=[Vector("relvec", 3)])
        fresh_graph.migrate_vectors()
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        with pytest.raises(ValueError, match="set_vectors") as excinfo:
            fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "relvec": [1.0, 0.0, 0.0]}])
        message = str(excinfo.value)
        assert "add_edges()" in message and "'relvec'" in message
        assert count(fresh_graph, "edges") == 0


# ---------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------

class TestMerge:
    def test_inserts_when_absent_and_updates_when_present(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("email")])
        fresh_graph.merge_nodes([{"email": "a@x.com", "name": "Alice"}], on=["email"])
        fresh_graph.merge_nodes([{"email": "a@x.com", "name": "Alicia"}], on=["email"])
        assert count(fresh_graph) == 1
        assert properties_of(fresh_graph)[0]["name"] == "Alicia"

    def test_unmentioned_properties_survive(self, fresh_graph):
        """`||` semantics, matching Cypher's ON MATCH SET: a merge updates
        what it mentions and leaves the rest alone."""
        fresh_graph.define_constraints(nodes=[Unique("email")])
        fresh_graph.merge_nodes([{"email": "a@x.com", "name": "Alice", "age": 30}], on=["email"])
        fresh_graph.merge_nodes([{"email": "a@x.com", "name": "Alicia"}], on=["email"])
        assert properties_of(fresh_graph)[0] == {"email": "a@x.com", "name": "Alicia", "age": 30}

    def test_replace_overwrites_the_whole_bag(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("email")])
        fresh_graph.merge_nodes([{"email": "a@x.com", "age": 30}], on=["email"])
        fresh_graph.merge_nodes([{"email": "a@x.com", "name": "Alice"}], on=["email"], replace=True)
        assert properties_of(fresh_graph)[0] == {"email": "a@x.com", "name": "Alice"}

    def test_composite_merge_key(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("tenant", "slug")])
        fresh_graph.merge_nodes([{"tenant": "a", "slug": "x", "v": 1}], on=["tenant", "slug"])
        fresh_graph.merge_nodes([{"tenant": "a", "slug": "x", "v": 2}], on=["tenant", "slug"])
        fresh_graph.merge_nodes([{"tenant": "b", "slug": "x", "v": 1}], on=["tenant", "slug"])
        assert count(fresh_graph) == 2

    def test_merging_edges(self, fresh_graph):
        from hopai import Col
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        fresh_graph.define_constraints(edges=[Unique(Col("start_id"), Col("end_id"), "kind")])
        for weight in (1, 2):
            fresh_graph.merge_edges([{"start_id": 1, "end_id": 2, "kind": "knows",
                                      "weight": weight}],
                                    on=[Col("start_id"), Col("end_id"), "kind"])
        assert count(fresh_graph, "edges") == 1

    def test_on_refuses_a_bare_string_naming_a_real_column(self, fresh_graph):
        """merge_edges(on=["start_id"]) meant Col("start_id") -- a bare
        string always names a JSONB property, so this can only be a
        mistake. Before the refusal, it silently compiled a conflict
        target on properties->>'start_id', a key ingestion never
        writes there, so the merge could never find its own inserts."""
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        with pytest.raises(TypeError, match="'start_id' is a real column"):
            fresh_graph.merge_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}],
                                    on=["start_id", "end_id", "kind"])

    def test_merge_is_idempotent(self, fresh_graph):
        """The property that makes merge the right call for an agent that
        may retry: running it twice changes nothing the second time."""
        fresh_graph.define_constraints(nodes=[Unique("email")])
        rows = [{"email": f"{i}@x.com", "n": i} for i in range(5)]
        fresh_graph.merge_nodes(rows, on=["email"])
        before = sorted(properties_of(fresh_graph), key=lambda p: p["n"])
        fresh_graph.merge_nodes(rows, on=["email"])
        assert sorted(properties_of(fresh_graph), key=lambda p: p["n"]) == before
        assert count(fresh_graph) == 5

    def test_merge_without_a_unique_index_says_what_to_declare(self, fresh_graph):
        with pytest.raises(ConstraintViolation, match="define_constraints"):
            fresh_graph.merge_nodes([{"email": "a@x.com"}], on=["email"])

    def test_merge_needs_keys(self, fresh_graph):
        with pytest.raises(ValueError, match="needs the keys"):
            fresh_graph.merge_nodes([{"email": "a@x.com"}], on=[])

    def test_merge_edges_needs_keys_and_names_its_own_call(self, fresh_graph):
        """The message interpolates merge_{what}s, so the edge path must
        say merge_edges -- and Graph.merge_edges must actually forward
        (mutant xǁIngestorǁmerge_edgesǁ__mutmut_22 upcased the label,
        and nothing pinned it)."""
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        with pytest.raises(ValueError, match=r"merge_edges\(on="):
            fresh_graph.merge_edges([{"start_id": 1, "end_id": 2, "kind": "x"}], on=[])

    def test_merge_edges_forwards_replace(self, fresh_graph):
        """Graph.merge_edges is a one-line delegation, and delegation
        kwargs are exactly where mutation testing keeps finding dropped
        arguments (xǁGraphǁmerge_edgesǁ__mutmut_4/7 dropped replace=):
        replace=True must swap the whole bag, and the default must
        keep unmentioned properties."""
        from hopai import Col
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        fresh_graph.define_constraints(edges=[Unique(Col("start_id"), Col("end_id"), "kind")])
        on = [Col("start_id"), Col("end_id"), "kind"]
        base = {"start_id": 1, "end_id": 2, "kind": "knows"}
        fresh_graph.merge_edges([{**base, "weight": 1, "keep": True}], on=on)
        fresh_graph.merge_edges([{**base, "weight": 2}], on=on)   # default: || merge
        with fresh_graph.engine.connect() as conn:
            props = conn.execute(text("SELECT properties FROM edges")).scalar()
        assert props == {"kind": "knows", "weight": 2, "keep": True}
        fresh_graph.merge_edges([{**base, "weight": 3}], on=on, replace=True)
        with fresh_graph.engine.connect() as conn:
            props = conn.execute(text("SELECT properties FROM edges")).scalar()
        assert props == {"kind": "knows", "weight": 3}   # 'keep' replaced away

    def test_a_declared_vector_fields_floats_are_refused_on_merge_nodes(self, fresh_graph):
        """The refusal fires before merge_nodes ever builds its ON
        CONFLICT statement, so this needs no Unique() -- the same
        row that would corrupt add_nodes() corrupts merge_nodes()."""
        fresh_graph.define_vectors(nodes=[Vector("summary", 3)])
        fresh_graph.migrate_vectors()
        with pytest.raises(ValueError, match="set_vectors") as excinfo:
            fresh_graph.merge_nodes([{"email": "a@x.com", "summary": [1.0, 0.0, 0.0]}],
                                    on=["email"])
        assert "merge_nodes()" in str(excinfo.value)

    def test_a_declared_vector_fields_floats_are_refused_on_merge_edges(self, fresh_graph):
        from hopai import Col
        fresh_graph.define_vectors(edges=[Vector("relvec", 3)])
        fresh_graph.migrate_vectors()
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        with pytest.raises(ValueError, match="set_vectors") as excinfo:
            fresh_graph.merge_edges(
                [{"start_id": 1, "end_id": 2, "kind": "knows", "relvec": [1.0, 0.0, 0.0]}],
                on=[Col("start_id"), Col("end_id"), "kind"])
        assert "merge_edges()" in str(excinfo.value)


# ---------------------------------------------------------------------
# Documents and interchange
# ---------------------------------------------------------------------

class TestDocuments:
    def test_nodes_are_written_before_edges(self, fresh_graph):
        """So one document can create a node and an edge referencing it,
        which is what any single agent turn will try to do."""
        result = fresh_graph.ingest({
            "nodes": [{"email": "a@x.com"}, {"email": "b@x.com"}],
            "edges": [{"start": {"email": "a@x.com"}, "end": {"email": "b@x.com"},
                       "kind": "knows"}],
        })
        assert (result.nodes, result.edges) == (2, 1)

    def test_either_half_may_be_missing(self, fresh_graph):
        assert fresh_graph.ingest({"nodes": [{"a": 1}]}).edges == 0
        assert fresh_graph.ingest({}).to_dict()["nodes"] == 0

    def test_merge_mode(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("email")])
        document = {"nodes": [{"email": "a@x.com", "v": 1}]}
        fresh_graph.ingest(document, merge_nodes_on=["email"])
        fresh_graph.ingest(document, merge_nodes_on=["email"])
        assert count(fresh_graph) == 1

    @pytest.mark.parametrize("bad,message", [
        ({"nodes": [], "extra": []}, "unknown keys"),
        ([], "takes a dict"),
    ])
    def test_malformed_documents(self, fresh_graph, bad, message):
        with pytest.raises((TypeError, ValueError), match=message):
            fresh_graph.ingest(bad)

    def test_result_object(self, fresh_graph):
        result = fresh_graph.ingest({"nodes": [{"a": 1}]})
        assert isinstance(result, IngestResult)
        assert json.loads(json.dumps(result.to_dict()))["nodes"] == 1
        assert "nodes=1" in repr(result)


class TestNetworkxRoundTrip:
    def test_round_trip_preserves_nodes_edges_and_properties(self, fresh_graph, graph):
        """to_networkx and add_networkx are inverses -- the property that
        makes the library usable next to the rest of the Python graph
        ecosystem instead of instead of it."""
        original = graph.traverse(Start(where={"type": "leaf"}), Hop()).to_networkx()
        fresh_graph.add_networkx(original)
        reloaded = fresh_graph.traverse(Start(), Hop()).to_networkx()
        assert set(reloaded.nodes) == set(original.nodes)
        assert set(reloaded.edges) == set(original.edges)
        for node in original.nodes:
            assert reloaded.nodes[node] == original.nodes[node]
        # edge PROPERTIES too: add_networkx builds its edge rows under a
        # literal "properties" key, and mutating that key silently
        # re-nested every edge's attributes (xǁIngestorǁadd_networkxǁ_14)
        for edge in original.edges:
            assert reloaded.edges[edge] == original.edges[edge]
        assert any(original.edges[e] for e in original.edges), \
            "fixture must carry at least one edge with properties, or this asserts nothing"


class TestToolSchema:
    def test_is_json_serializable(self):
        assert json.loads(json.dumps(INGEST_TOOL_SCHEMA)) == INGEST_TOOL_SCHEMA

    def test_describes_both_ways_to_reference_a_node(self):
        edge = INGEST_TOOL_SCHEMA["parameters"]["properties"]["edges"]["items"]["properties"]
        assert {"start_id", "end_id", "start", "end"} == set(edge)

    def test_allows_arbitrary_properties(self):
        """additionalProperties must stay true, or a model will assume the
        listed keys are the only ones allowed and stop sending data."""
        items = INGEST_TOOL_SCHEMA["parameters"]["properties"]
        assert items["nodes"]["items"]["additionalProperties"] is True
        assert items["edges"]["items"]["additionalProperties"] is True

    def test_a_document_shaped_like_the_schema_ingests(self, fresh_graph):
        assert fresh_graph.ingest({
            "nodes": [{"id": 1, "type": "person"}, {"id": 2, "type": "company"}],
            "edges": [{"start_id": 1, "end_id": 2, "kind": "works_at"}],
        }).to_dict() == {"nodes": 2, "edges": 1, "elapsed_ms": pytest.approx(0, abs=10_000)}


# ---------------------------------------------------------------------
# Scale and safety
# ---------------------------------------------------------------------

class TestAtomicity:
    """A write is one transaction, batching and documents included.

    Half-committing is the worst possible failure for an agent: it is
    told the call failed, retries, and the retry now collides with rows
    that did land. Every test here fails loudly if a transaction is ever
    opened per batch or per half-document again."""

    def test_a_failing_late_batch_rolls_back_the_earlier_ones(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("n")])
        rows = [{"n": i} for i in range(BATCH_SIZE + 10)]
        rows[-1]["n"] = 0  # collides with the first row, in a later chunk
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes(rows)
        assert count(fresh_graph) == 0

    def test_a_failing_late_merge_batch_rolls_back_the_earlier_ones(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("n"), Required("kind")])
        rows = [{"n": i, "kind": "ok"} for i in range(BATCH_SIZE + 10)]
        del rows[-1]["kind"]  # fails Required, in a later chunk
        with pytest.raises(ConstraintViolation):
            fresh_graph.merge_nodes(rows, on=["n"])
        assert count(fresh_graph) == 0

    def test_failing_edges_roll_back_the_documents_nodes(self, fresh_graph):
        """A document is one unit. Its nodes must not survive its edges
        failing."""
        with pytest.raises(ConstraintViolation):
            fresh_graph.ingest({
                "nodes": [{"id": 1, "type": "person"}, {"id": 2, "type": "person"}],
                "edges": [{"start_id": 1, "end_id": 2}, {"start_id": 1, "end_id": 999}],
            })
        assert count(fresh_graph) == 0 and count(fresh_graph, "edges") == 0

    def test_an_unresolvable_edge_reference_rolls_back_the_nodes(self, fresh_graph):
        with pytest.raises(ValueError, match="no node matches"):
            fresh_graph.ingest({
                "nodes": [{"email": "a@x.com"}],
                "edges": [{"start": {"email": "a@x.com"}, "end": {"email": "ghost@x.com"}}],
            })
        assert count(fresh_graph) == 0

    def test_edges_resolve_against_nodes_from_the_same_document(self, fresh_graph):
        """The other half of sharing one transaction: a reference must see
        the uncommitted nodes written moments earlier in the same call."""
        result = fresh_graph.ingest({
            "nodes": [{"email": "a@x.com"}, {"email": "b@x.com"}],
            "edges": [{"start": {"email": "a@x.com"}, "end": {"email": "b@x.com"}}],
        })
        assert (result.nodes, result.edges) == (2, 1)

    def test_merge_with_explicit_ids_still_advances_the_sequence(self, fresh_graph):
        """The sequence resync has to happen inside the merge's own
        transaction, not a second one opened after it."""
        fresh_graph.define_constraints(nodes=[Unique("n")])
        fresh_graph.merge_nodes([{"id": 50, "n": 1}], on=["n"])
        fresh_graph.add_nodes([{"n": 2}])
        assert count(fresh_graph) == 2


class TestScaleAndSafety:
    def test_batches_larger_than_one_statement(self, fresh_graph):
        rows = [{"n": i} for i in range(BATCH_SIZE + 250)]
        assert fresh_graph.add_nodes(rows) == len(rows)
        assert count(fresh_graph) == len(rows)

    def test_a_large_merge_spanning_batches(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("n")])
        rows = [{"n": i} for i in range(BATCH_SIZE + 250)]
        fresh_graph.merge_nodes(rows, on=["n"])
        fresh_graph.merge_nodes(rows, on=["n"])
        assert count(fresh_graph) == len(rows)

    @pytest.mark.parametrize("payload", [
        "'; DROP TABLE nodes; --",
        {"nested": {"deep": ["'; DELETE FROM nodes; --"]}},
        "100% \\ weird \" quotes '",
        "日本語",
    ])
    def test_hostile_property_values_round_trip_intact(self, fresh_graph, payload):
        """Ingested data is model output in the use case this library is
        for. It goes in as a bound parameter and must come back byte for
        byte, with the tables still standing."""
        fresh_graph.add_nodes([{"id": 1, "value": payload}])
        assert properties_of(fresh_graph)[0]["value"] == payload
        assert count(fresh_graph) == 1

    def test_hostile_property_keys_round_trip_intact(self, fresh_graph):
        fresh_graph.add_nodes([{"'; DROP TABLE nodes; --": "x"}])
        assert properties_of(fresh_graph)[0] == {"'; DROP TABLE nodes; --": "x"}

    def test_ingested_data_is_traversable(self, fresh_graph):
        """The whole point, end to end: write a graph, then walk it."""
        fresh_graph.ingest({
            "nodes": [{"id": i, "type": "person", "n": i} for i in range(1, 5)],
            "edges": [{"start_id": i, "end_id": i + 1, "kind": "knows"} for i in range(1, 4)],
        })
        result = fresh_graph.traverse(
            Start(where={"n": 1}), Hop(via={"kind": "knows"}, hops=(1, 3)))
        assert {n["properties"]["n"] for n in result.nodes} == {1, 2, 3, 4}
