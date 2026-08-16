"""
Test suite for hopai.asyncio -- AsyncGraph.

Not a second test of traversal/mutation/ingestion/vector-search
semantics: those are exhaustively covered against the sync Graph
elsewhere, and AsyncGraph shares its query builders and
execute-and-hydrate functions exactly (core.py's
_traverse_with_session()/_aggregate_with_session(), Mutator's and
Ingestor's connection=-taking methods, the vectors.py functions'
connection=) -- there is no second implementation here to disagree
with the first. What this file checks is the WIRING: that routing
through AsyncSession/AsyncConnection.run_sync() reaches those same
functions and returns the same answers, that every operation issue #45
named is reachable, and that the methods deliberately NOT covered
(schema/constraint DDL) refuse with the fix named rather than failing
opaquely against the async engine's sync facade.

No pytest-asyncio: run() below is a one-line asyncio.run() wrapper, so
this suite needs nothing beyond the `asyncio` extra's async driver.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from hopai import Count, Hop, Near, Start, Vector
from hopai.asyncio import AsyncGraph


def run(coro):
    return asyncio.run(coro)


class TestConstruction:
    def test_rejects_a_sync_engine(self, engine):
        with pytest.raises(TypeError, match="sync Engine"):
            AsyncGraph(engine)

    def test_repr_names_the_dsn_and_graph(self, async_graph):
        assert repr(async_graph).startswith("AsyncGraph(")

    def test_engine_property_is_the_async_engine(self, async_graph):
        assert isinstance(async_graph.engine, AsyncEngine)


class TestSameAnswerAsSync:
    """The core claim: AsyncGraph has no traversal/aggregation logic of
    its own, so its answer for the same query must be identical to
    Graph's -- not just close, identical, because underneath it is the
    same function call either way."""

    def test_traverse(self, graph, async_graph):
        start = Start(where={"type": "leaf"})
        hop = Hop(via={"kind": "knows"}, hops=(1, 2))
        sync_result = graph.traverse(start, hop)
        async_result = run(async_graph.traverse(start, hop))
        assert async_result.nodes == sync_result.nodes
        assert async_result.edges == sync_result.edges

    def test_aggregate(self, graph, async_graph):
        start = Start(where={"type": "leaf"})
        sync_result = graph.aggregate(start, aggregates={"n": Count()})
        async_result = run(async_graph.aggregate(start, aggregates={"n": Count()}))
        assert async_result == sync_result

    def test_cypher_traverse(self, graph, async_graph):
        query = "MATCH (a {type: 'leaf'})-[:knows]->(b) RETURN a, b"
        sync_result = graph.cypher(query)
        async_result = run(async_graph.cypher(query))
        assert async_result.nodes == sync_result.nodes
        assert async_result.edges == sync_result.edges

    def test_cypher_aggregate(self, graph, async_graph):
        query = "MATCH (a {type: 'leaf'}) RETURN count(a)"
        sync_result = graph.cypher(query)
        async_result = run(async_graph.cypher(query))
        assert async_result == sync_result


class TestIngestAndMutate:
    def test_add_nodes_edges_and_traverse(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1, "type": "person"},
                                               {"id": 2, "type": "person"}])
            await async_fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}])
            return await async_fresh_graph.traverse(
                Start(where={"type": "person"}), Hop(via={"kind": "knows"}, hops=(1, 1)))

        result = run(body())
        assert len(result.nodes) == 2
        assert len(result.edges) == 1

    def test_merge_nodes_updates_on_conflict(self, async_fresh_graph, async_admin_graph):
        from hopai import Unique

        # define_constraints() is deliberately not on AsyncGraph -- see
        # TestOutOfScope below -- so the unique index merge_nodes() needs
        # to detect a conflict is declared through the documented escape
        # hatch: a plain sync Graph on the same schema.
        async_admin_graph.define_constraints(nodes=[Unique("email")])

        async def body():
            await async_fresh_graph.merge_nodes(
                [{"email": "a@x.com", "type": "person"}], on=["email"])
            await async_fresh_graph.merge_nodes(
                [{"email": "a@x.com", "type": "person", "age": 30}], on=["email"])
            return await async_fresh_graph.traverse(Start())

        result = run(body())
        assert len(result.nodes) == 1
        assert result.nodes[0]["properties"]["age"] == 30

    def test_ingest_document(self, async_fresh_graph):
        async def body():
            return await async_fresh_graph.ingest(
                {"nodes": [{"id": 1}, {"id": 2}], "edges": [{"start_id": 1, "end_id": 2}]})

        result = run(body())
        assert result.nodes == 2
        assert result.edges == 1

    def test_delete_nodes_with_no_filter_refuses(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1}])
            await async_fresh_graph.delete_nodes()

        with pytest.raises(ValueError, match="no filter"):
            run(body())

    def test_update_and_clear(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1, "type": "draft"}])
            updated = await async_fresh_graph.update_nodes(
                where={"type": "draft"}, set={"status": "archived"})
            cleared = await async_fresh_graph.clear()
            return updated, cleared

        updated, cleared = run(body())
        assert updated.updated_nodes == 1
        assert cleared.deleted_nodes == 1

    def test_mutate_document(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1, "type": "draft"}])
            return await async_fresh_graph.mutate({"operations": [
                {"op": "update_nodes", "where": {"type": "draft"}, "set": {"status": "archived"}},
            ]})

        result = run(body())
        assert result.updated_nodes == 1

    def test_write_cypher_then_cypher_dispatches_to_traverse(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.write_cypher(
                "CREATE (a:person {email: 'a@x.com'})-[:knows]->(b:person {email: 'b@x.com'})")
            return await async_fresh_graph.cypher("MATCH (a:person) RETURN a")

        result = run(body())
        assert len(result.nodes) == 2

    def test_cypher_dispatches_to_write(self, async_fresh_graph):
        async def body():
            write_result = await async_fresh_graph.cypher(
                "CREATE (a:person {email: 'a@x.com'})")
            return write_result, await async_fresh_graph.traverse(Start())

        write_result, traversal = run(body())
        assert write_result.nodes == 1
        assert len(traversal.nodes) == 1

    def test_merge_edges_updates_on_conflict(self, async_fresh_graph, async_admin_graph):
        from hopai import Unique

        async_admin_graph.define_constraints(edges=[Unique("tag")])

        async def body():
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.merge_edges(
                [{"start_id": 1, "end_id": 2, "tag": "e1"}], on=["tag"])
            await async_fresh_graph.merge_edges(
                [{"start_id": 1, "end_id": 2, "tag": "e1", "weight": 5}], on=["tag"])
            return await async_fresh_graph.traverse(
                Start(), Hop(hops=(1, 1)))

        result = run(body())
        assert len(result.edges) == 1
        assert result.edges[0]["properties"]["weight"] == 5

    def test_delete_edges_and_update_edges(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}, {"id": 3}])
            await async_fresh_graph.add_edges([
                {"start_id": 1, "end_id": 2, "kind": "knows"},
                {"start_id": 1, "end_id": 3, "kind": "blocks"},
            ])
            updated = await async_fresh_graph.update_edges(
                where={"kind": "knows"}, set={"seen": True})
            deleted = await async_fresh_graph.delete_edges(where={"kind": "blocks"})
            remaining = await async_fresh_graph.traverse(Start(), Hop(hops=(1, 1)))
            return updated, deleted, remaining

        updated, deleted, remaining = run(body())
        assert updated.updated_edges == 1
        assert deleted.deleted_edges == 1
        assert len(remaining.edges) == 1
        assert remaining.edges[0]["properties"]["seen"] is True

    def test_mutate_cypher_then_cypher_dispatches_to_mutate(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.write_cypher("CREATE (a:person {active: true})")
            await async_fresh_graph.cypher("MATCH (a:person) SET a.active = false")
            return await async_fresh_graph.cypher("MATCH (a:person) RETURN a")

        result = run(body())
        assert result.nodes[0]["properties"]["active"] is False


class TestVectors:
    def test_migrate_set_search_get_stale_drop(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1, "type": "doc"}, {"id": 2, "type": "doc"}])
            stale_before = await async_fresh_graph.stale_vectors()
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 1, "summary": [1.0, 0.0, 0.0]},
                {"id": 2, "summary": [0.0, 1.0, 0.0]},
            ])
            hits = await async_fresh_graph.vector_search(Near("summary", [1.0, 0.0, 0.0]), k=1)
            got = await async_fresh_graph.get_vectors(node_ids=[1])
            dropped = await async_fresh_graph.drop_vectors(node_fields=["summary"])
            return stale_before, hits, got, dropped

        stale_before, hits, got, dropped = run(body())
        assert sorted(stale_before["nodes"]["summary"]["missing"]) == ["1", "2"]
        assert hits[0]["id"] == "1"
        assert got["nodes"]["1"]["summary"] == pytest.approx([1.0, 0.0, 0.0])
        assert dropped == ["nodes.vec_summary"]

    def test_vector_search_many_ranks_each_query_independently(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 2)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 1, "summary": [1.0, 0.0]}, {"id": 2, "summary": [0.0, 1.0]},
            ])
            return await async_fresh_graph.vector_search_many(
                [Near("summary", [1.0, 0.0]), Near("summary", [0.0, 1.0])], k=1)

        results = run(body())
        assert results[0][0]["id"] == "1"
        assert results[1][0]["id"] == "2"


class TestOutOfScope:
    """Schema/constraint DDL has no async override -- see hopai/asyncio.py's
    module docstring. Each must refuse LOUD, naming the fix, rather than
    silently reaching the async engine's sync facade outside the greenlet
    bridge (which would raise SQLAlchemy's own MissingGreenlet instead)."""

    @pytest.mark.parametrize("name", [
        "create_schema", "drop_schema", "define_constraints", "drop_constraints",
        "enforce_schema", "save_schema", "load_schema", "infer_schema",
        "schema_violations", "add_networkx",
    ])
    def test_admin_methods_refuse_with_the_fix_named(self, async_graph, name):
        with pytest.raises(AttributeError, match="plain Graph"):
            getattr(async_graph, name)

    def test_connection_free_methods_pass_through_unchanged(self, async_graph):
        # build_query() never connects, on Graph or AsyncGraph alike --
        # compilable with no database, exactly like Graph.build_query().
        query = async_graph.build_query(Start(where={"type": "leaf"}), [])
        assert query is not None
        assert async_graph.tool_schemas()


class TestInGraph:
    def test_scopes_to_a_different_graph(self, async_graph):
        other = async_graph.in_graph("other")
        assert isinstance(other, AsyncGraph)
        assert other.graph == "other"
        assert other.engine is async_graph.engine


class TestDispose:
    def test_dispose_closes_the_pool(self, async_fresh_graph):
        run(async_fresh_graph.dispose())
