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

from hopai import Count, CypherError, Hop, Near, Start, Unique, Vector
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

    def test_vector_search_default_k_is_ten(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 2)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": i} for i in range(1, 13)])
            await async_fresh_graph.set_vectors(
                nodes=[{"id": i, "summary": [1.0, 0.0]} for i in range(1, 13)])
            # No k= -- relies entirely on the method's own default.
            return await async_fresh_graph.vector_search(Near("summary", [1.0, 0.0]))

        results = run(body())
        assert len(results) == 10

    def test_get_vectors_with_both_node_and_edge_ids(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 2)],
                                             edges=[Vector("summary", 2)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.add_edges([{"id": 9, "start_id": 1, "end_id": 2}])
            await async_fresh_graph.set_vectors(
                nodes=[{"id": 1, "summary": [1.0, 0.0]}],
                edges=[{"id": 9, "summary": [0.0, 1.0]}])
            return await async_fresh_graph.get_vectors(node_ids=[1], edge_ids=[9])

        result = run(body())
        assert result["nodes"]["1"]["summary"] == pytest.approx([1.0, 0.0])
        # Without edge_ids reaching the call, this would come back empty.
        assert result["edges"]["9"]["summary"] == pytest.approx([0.0, 1.0])

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

    def test_vector_search_many_where_filters_candidates(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 2)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([
                {"id": 1, "type": "keep"}, {"id": 2, "type": "skip"},
            ])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 1, "summary": [1.0, 0.0]}, {"id": 2, "summary": [1.0, 0.0]},
            ])
            return await async_fresh_graph.vector_search_many(
                [Near("summary", [1.0, 0.0])], where={"type": "keep"}, k=5)

        results = run(body())
        # Without where= reaching the call, both same-vector nodes would match.
        assert [hit["id"] for hit in results[0]] == ["1"]


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


class TestColumnOverridesPropagate:
    """Every keyword AsyncGraph.__init__() takes must reach the wrapped
    sync Graph unchanged -- __init__() and in_graph() each had an
    argument mutation testing could drop with nothing noticing, since
    every other test in this file leaves every override at its default.
    No database needed: construction never connects, on Graph or
    AsyncGraph alike."""

    OVERRIDES = {
        "node_id_col": "pk", "edge_id_col": "eid", "edge_start_col": "src",
        "edge_end_col": "dst", "graph_col": "tenant",
    }
    OFFLINE_DSN = "postgresql+psycopg://offline:offline@127.0.0.1:1/offline"

    def _custom_tables(self):
        from sqlalchemy import BigInteger, Column, MetaData, Table, Text
        from sqlalchemy.dialects.postgresql import JSONB
        meta = MetaData()
        nodes = Table("t_nodes", meta, Column("pk", BigInteger, primary_key=True),
                      Column("tenant", Text), Column("properties", JSONB))
        edges = Table("t_edges", meta, Column("eid", BigInteger, primary_key=True),
                      Column("tenant", Text), Column("src", BigInteger),
                      Column("dst", BigInteger), Column("properties", JSONB))
        return nodes, edges

    def _assert_overrides(self, graph, nodes, edges):
        assert graph.nodes_tbl is nodes
        assert graph.edges_tbl is edges
        for key, value in self.OVERRIDES.items():
            assert getattr(graph, key) == value

    def test_constructor_propagates_every_override(self):
        nodes, edges = self._custom_tables()
        graph = AsyncGraph(self.OFFLINE_DSN, node_table=nodes, edge_table=edges,
                           **self.OVERRIDES)
        self._assert_overrides(graph, nodes, edges)

    def test_in_graph_propagates_every_override(self):
        nodes, edges = self._custom_tables()
        graph = AsyncGraph(self.OFFLINE_DSN, node_table=nodes, edge_table=edges,
                           **self.OVERRIDES)
        other = graph.in_graph("other")
        assert other.graph == "other"
        self._assert_overrides(other, nodes, edges)


class TestConnectionIsShared:
    """Every AsyncGraph write method must pass its already-checked-out
    connection through to the sync Mutator/Ingestor/vectors function it
    wraps -- several mutation-testing survivors on hopai/asyncio.py were
    exactly that connection=c keyword being silently dropped, which
    opens a SECOND connection instead of reusing the first. See
    conftest.py's async_fresh_graph_pool1 for the mechanism: a
    one-connection pool turns "opened an extra connection" into a pool
    timeout instead of a difference nothing observes."""

    def test_writes_never_need_a_second_connection(self, async_fresh_graph_pool1):
        g = async_fresh_graph_pool1

        async def body():
            g.define_vectors(nodes=[Vector("summary", 2)])
            await g.migrate_vectors()
            await g.add_nodes([{"id": 1, "email": "a@x.com"}, {"id": 2, "email": "b@x.com"}])
            await g.merge_nodes([{"email": "a@x.com", "age": 31}], on=["email"])
            await g.add_edges([{"start_id": 1, "end_id": 2, "tag": "e1"}])
            await g.merge_edges([{"start_id": 1, "end_id": 2, "tag": "e1", "weight": 2}],
                                on=["tag"])
            await g.ingest({"nodes": [{"id": 3, "email": "c@x.com"}],
                            "edges": [{"start_id": 1, "end_id": 3, "tag": "e2"}]})
            await g.set_vectors(nodes=[{"id": 1, "summary": [1.0, 0.0]}])
            await g.vector_search(Near("summary", [1.0, 0.0]), k=1)
            await g.vector_search_many([Near("summary", [1.0, 0.0])], k=1)
            await g.get_vectors(node_ids=[1])
            await g.stale_vectors()
            await g.drop_vectors(node_fields=["summary"])
            await g.mutate_cypher("MATCH (a {email: 'a@x.com'}) SET a.active = true")
            await g.write_cypher("CREATE (d:person {email: 'd@x.com'})")
            await g.update_nodes(where={"email": "d@x.com"}, set={"checked": True})
            await g.delete_nodes(where={"email": "d@x.com"})
            await g.delete_edges(where={"tag": "e2"})
            await g.update_edges(where={"tag": "e1"}, set={"seen": True})
            await g.mutate({"operations": [
                {"op": "update_nodes", "where": {"email": "b@x.com"}, "set": {"checked": True}},
            ]})
            await g.clear()

        run(body())   # a pool timeout (TimeoutError) means some call above needed a 2nd connection


class TestArgumentsReachTheSyncCall:
    """Each of these caught an argument mutation testing found could be
    replaced with a hardcoded value (None, or the default) with nothing
    noticing -- every prior test only ever exercised the default."""

    def test_merge_nodes_replace_true_reaches_the_call(self, async_fresh_graph, async_admin_graph):
        async_admin_graph.define_constraints(nodes=[Unique("email")])

        async def body():
            await async_fresh_graph.merge_nodes([{"email": "a@x.com", "age": 20}], on=["email"])
            await async_fresh_graph.merge_nodes(
                [{"email": "a@x.com", "name": "Alice"}], on=["email"], replace=True)
            return await async_fresh_graph.traverse(Start())

        result = run(body())
        # replace=True means the second write IS the whole properties bag --
        # "age" from the first write must be gone, not merged alongside "name".
        assert result.nodes[0]["properties"] == {"email": "a@x.com", "name": "Alice"}

    def test_merge_nodes_default_is_merge_not_replace(self, async_fresh_graph, async_admin_graph):
        async_admin_graph.define_constraints(nodes=[Unique("email")])

        async def body():
            await async_fresh_graph.merge_nodes([{"email": "a@x.com", "age": 20}], on=["email"])
            # No replace= passed -- the default (False) must still merge, not
            # wipe "age" the way replace=True does in the test above.
            await async_fresh_graph.merge_nodes([{"email": "a@x.com", "name": "Alice"}],
                                                 on=["email"])
            return await async_fresh_graph.traverse(Start())

        result = run(body())
        assert result.nodes[0]["properties"] == {"email": "a@x.com", "age": 20, "name": "Alice"}

    def test_update_nodes_remove_reaches_the_call(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1, "type": "person", "nickname": "Al"}])
            await async_fresh_graph.update_nodes(where={"type": "person"}, remove=["nickname"])
            return await async_fresh_graph.traverse(Start())

        result = run(body())
        # Without remove= reaching Mutator.update_nodes(), set= and remove= are
        # both None -- "nothing to change" -- and this raises instead of succeeding.
        assert result.nodes[0]["properties"] == {"type": "person"}

    def test_update_edges_all_true_reaches_the_call(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.add_edges([{"start_id": 1, "end_id": 2}])
            # No where/start/end -- only all=True makes this a legal call.
            return await async_fresh_graph.update_edges(all=True, set={"seen": True})

        result = run(body())
        assert result.updated_edges == 1

    def test_delete_edges_end_filter_narrows_the_delete(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2, "role": "target"},
                                               {"id": 3, "role": "other"}])
            await async_fresh_graph.add_edges([
                {"start_id": 1, "end_id": 2, "kind": "knows"},
                {"start_id": 1, "end_id": 3, "kind": "knows"},
            ])
            deleted = await async_fresh_graph.delete_edges(
                where={"kind": "knows"}, end={"role": "target"})
            remaining = await async_fresh_graph.traverse(Start(), Hop(hops=(1, 1)))
            return deleted, remaining

        deleted, remaining = run(body())
        # Without end= reaching the call, both edges match where= alone and both go.
        assert deleted.deleted_edges == 1
        assert len(remaining.edges) == 1
        assert remaining.edges[0]["end_id"] == "3"

    def test_ingest_with_merge_nodes_on(self, async_fresh_graph, async_admin_graph):
        async_admin_graph.define_constraints(nodes=[Unique("email")])

        async def body():
            await async_fresh_graph.ingest({"nodes": [{"email": "a@x.com", "age": 20}]})
            await async_fresh_graph.ingest(
                {"nodes": [{"email": "a@x.com", "age": 31}]}, merge_nodes_on=["email"])
            return await async_fresh_graph.traverse(Start())

        result = run(body())
        assert len(result.nodes) == 1
        assert result.nodes[0]["properties"]["age"] == 31

    def test_delete_nodes_with_a_real_filter(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1, "type": "draft"},
                                               {"id": 2, "type": "keep"}])
            deleted = await async_fresh_graph.delete_nodes(where={"type": "draft"})
            remaining = await async_fresh_graph.traverse(Start())
            return deleted, remaining

        deleted, remaining = run(body())
        assert deleted.deleted_nodes == 1
        assert len(remaining.nodes) == 1
        assert remaining.nodes[0]["properties"]["type"] == "keep"

    def test_delete_nodes_detach_true_removes_edges_too(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1, "type": "person"},
                                               {"id": 2, "type": "person"}])
            await async_fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}])
            return await async_fresh_graph.delete_nodes(where={"type": "person"}, detach=True)

        result = run(body())
        # Without detach=True actually reaching Mutator.delete_nodes(), the foreign
        # key refuses (edges still point at these nodes) and this raises instead.
        assert result.deleted_nodes == 2
        assert result.deleted_edges == 1


class TestStrictSchemaOptionReachesTheGraph:
    """strict_schema=True needs the real AsyncGraph passed to
    resolve_strict() -- resolve_strict(None, ...) raises AttributeError
    reaching for None.schema instead of the CypherError this refusal is
    supposed to be. Cheap to prove with no schema defined at all:
    strict_schema=True always raises, but only raises the RIGHT
    exception when the wiring is intact -- so these fail on the wrong
    exception type under a broken mutant rather than needing a whole
    declared schema to set up."""

    def test_mutate_cypher(self, async_fresh_graph):
        with pytest.raises(CypherError, match="strict_schema"):
            run(async_fresh_graph.mutate_cypher(
                "MATCH (a:person) SET a.active = true", strict_schema=True))

    def test_cypher_dispatches_write_with_options(self, async_fresh_graph):
        with pytest.raises(CypherError, match="strict_schema"):
            run(async_fresh_graph.cypher(
                "CREATE (a:person {email: 'a@x.com'})", strict_schema=True))

    def test_cypher_dispatches_aggregate_with_options(self, async_fresh_graph):
        with pytest.raises(CypherError, match="strict_schema"):
            run(async_fresh_graph.cypher("MATCH (a:person) RETURN count(a)", strict_schema=True))

    def test_cypher_dispatches_traverse_with_options(self, async_fresh_graph):
        with pytest.raises(CypherError, match="strict_schema"):
            run(async_fresh_graph.cypher("MATCH (a:person) RETURN a", strict_schema=True))
