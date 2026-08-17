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
import time

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine

from hopai import Boost, Count, CypherError, Hop, Near, Start, Unique, Vector
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

    def test_cypher_aggregate_carries_its_hops(self, graph, async_graph):
        """The case above has no hops, so it cannot see cypher()'s
        dispatch dropping `*hops` on the way to aggregate() -- with none
        to drop, both spellings count the same set. With a hop, dropping
        it counts the SEED set instead: a different number, from a query
        that still succeeds and says nothing."""
        walked = "MATCH (a {type: 'leaf'})-[:knows]->(b) RETURN count(DISTINCT b)"
        seeds = "MATCH (a {type: 'leaf'}) RETURN count(a)"
        sync_result = graph.cypher(walked)
        assert run(async_graph.cypher(walked)) == sync_result
        # Without this the assertion above is vacuous: it only means
        # something while the hop actually changes the answer.
        assert sync_result != graph.cypher(seeds)


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

    def test_ingest_forwards_merge_edges_on(self, async_fresh_graph, async_admin_graph):
        """Both merge keys have to reach the sync ingestor. A mutant
        blanking `merge_edges_on` survived every ingest test here,
        because the only one that existed passed neither -- so an
        ingest asked to merge edges would have inserted duplicates
        instead, and nothing would have said so.

        Asserted through the OUTCOME rather than the call: re-ingesting
        the same edge is one edge with the new weight if the key
        arrived, and two edges if it did not."""
        from hopai import Unique

        async_admin_graph.define_constraints(edges=[Unique("tag")])
        document = {"nodes": [{"id": 1}, {"id": 2}],
                    "edges": [{"start_id": 1, "end_id": 2, "tag": "e1", "weight": 5}]}

        async def body():
            await async_fresh_graph.ingest(document, merge_edges_on=["tag"])
            # Edges only the second time: the nodes already exist, and
            # re-sending them would test merge_nodes_on instead.
            await async_fresh_graph.ingest(
                {"edges": [{"start_id": 1, "end_id": 2, "tag": "e1", "weight": 9}]},
                merge_edges_on=["tag"])
            return await async_fresh_graph.traverse(Start(), Hop(hops=(1, 1)))

        result = run(body())
        assert len(result.edges) == 1
        assert result.edges[0]["properties"]["weight"] == 9

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

    def test_text_is_embedded_before_the_transaction_opens(self, async_fresh_graph):
        """The sync set_vectors() plans and then opens a transaction, so
        the provider call is outside it by construction. This one cannot
        get that for free: the transaction is already open by the time
        run_sync() reaches vectors.py, so planning inside would hold row
        locks for a network round trip -- the invariant set_vectors()'
        docstring states and tests/test_vectors.py pins for the sync
        side. Here the plan is built before `begin()`, and this is what
        says so."""
        embedded, opened = [], []

        def watching(texts):
            embedded.extend(texts)
            return [[1.0, 0.0, 0.0] for _ in texts]

        def on_begin(conn):
            opened.append(len(embedded))

        async def body():
            async_fresh_graph.define_vectors(
                nodes=[Vector("summary", 3, embed=watching)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1, "summary": "a paper about Raft"}])

            engine = async_fresh_graph.engine.sync_engine
            event.listen(engine, "begin", on_begin)
            try:
                await async_fresh_graph.set_vectors(
                    nodes=[{"id": 1, "summary": "a paper about Raft"}])
            finally:
                event.remove(engine, "begin", on_begin)

        run(body())
        assert embedded == ["a paper about Raft"]
        # Every transaction this call opened already had the embedding in
        # hand. A zero here is a provider round trip holding row locks.
        assert opened and all(count == 1 for count in opened)

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

    def test_define_vectors_migrate_true_is_refused_with_the_fix_named(
            self, async_fresh_graph):
        """define_vectors(migrate=True) (issue #57) is a Graph-only
        shortcut: on AsyncGraph it would run migrate_vectors() straight
        against the async engine's sync facade outside the greenlet
        bridge -- the same MissingGreenlet trap TestOutOfScope pins for
        the admin methods -- so it must refuse loud instead, naming both
        the two-call replacement and the plain-Graph escape hatch."""
        with pytest.raises(AttributeError, match="migrate=True"):
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3)], migrate=True)
        # The refusal fires before any declaration -- nothing half-applied.
        assert async_fresh_graph.vectors is None

    def test_define_vectors_without_migrate_still_passes_through(self, async_fresh_graph):
        result = async_fresh_graph.define_vectors(nodes=[Vector("summary", 3)])
        assert set(result["nodes"]) == {"summary"}


class TestTextEmbeddingStaysOffTheLoop:
    """Issue #74: every text= embedding reachable from AsyncGraph is
    resolved (and, for set_vectors(), PLANNED) awaited, BEFORE
    run_sync()/begin() opens -- so a slow provider round trip never
    runs on the event loop's own thread the way it did before this fix
    (either directly, for set_vectors(), or inside the greenlet
    bridge, for every read path).

    A passing RESULT alone cannot show that: a version that blocks the
    loop still returns the right answer, just after starving every
    other task in the process for the length of the embed call. Each
    test below runs the AsyncGraph call concurrently with an
    independent ticking task and asserts the ticker actually
    progressed -- the acceptance test issue #74 asked for."""

    @staticmethod
    async def _progress_during(coro):
        """Run `coro` concurrently with a task that only makes progress
        if the loop is free to schedule it. Returns (coro's result,
        how many ticks got in before coro finished) -- a blocked loop
        lets zero through no matter how long coro takes."""
        ticks = []

        async def ticker():
            while True:
                await asyncio.sleep(0.01)
                ticks.append(None)

        ticker_task = asyncio.create_task(ticker())
        try:
            result = await coro
        finally:
            ticker_task.cancel()
        return result, len(ticks)

    @staticmethod
    def slow_embed(vector=(1.0, 0.0, 0.0), delay: float = 0.2):
        """A fake provider client that blocks like a real HTTP call
        would -- Embedder's to_thread fallback is what has to keep this
        off the loop, since a plain callable has no native async form."""
        def embed(texts):
            time.sleep(delay)
            return [list(vector) for _ in texts]
        return embed

    def test_traverse_start_near_text_does_not_block_the_loop(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=self.slow_embed())])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 1, "summary": [1.0, 0.0, 0.0]}, {"id": 2, "summary": [0.0, 1.0, 0.0]},
            ])
            return await self._progress_during(
                async_fresh_graph.traverse(Start(near=Near("summary", text="q"), keep=1)))

        result, ticks = run(body())
        assert [n["id"] for n in result.nodes] == ["1"]
        assert ticks > 0, "an independent task made no progress -- the loop was blocked"

    def test_hop_near_text_does_not_block_the_loop(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=self.slow_embed())])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1, "role": "root"}, {"id": 2}, {"id": 3}])
            await async_fresh_graph.add_edges([
                {"start_id": 1, "end_id": 2, "kind": "knows"},
                {"start_id": 1, "end_id": 3, "kind": "knows"},
            ])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 2, "summary": [1.0, 0.0, 0.0]}, {"id": 3, "summary": [0.0, 1.0, 0.0]},
            ])
            return await self._progress_during(async_fresh_graph.traverse(
                Start(where={"role": "root"}),
                Hop(via={"kind": "knows"}, near=Near("summary", text="q"), keep=1)))

        result, ticks = run(body())
        assert sorted(n["id"] for n in result.nodes) == ["1", "2"]
        assert ticks > 0

    def test_hop_via_near_text_does_not_block_the_loop(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(edges=[Vector("rel", 3, embed=self.slow_embed())])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1, "role": "root"}, {"id": 2}, {"id": 3}])
            await async_fresh_graph.add_edges([
                {"id": 91, "start_id": 1, "end_id": 2}, {"id": 92, "start_id": 1, "end_id": 3},
            ])
            await async_fresh_graph.set_vectors(edges=[
                {"id": 91, "rel": [1.0, 0.0, 0.0]}, {"id": 92, "rel": [0.0, 1.0, 0.0]},
            ])
            return await self._progress_during(async_fresh_graph.traverse(
                Start(where={"role": "root"}),
                Hop(via_near=Near("rel", text="q"), via_keep=1)))

        result, ticks = run(body())
        assert sorted(n["id"] for n in result.nodes) == ["1", "2"]
        assert ticks > 0

    def test_aggregate_near_text_does_not_block_the_loop(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=self.slow_embed())])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 1, "summary": [1.0, 0.0, 0.0]}, {"id": 2, "summary": [0.0, 1.0, 0.0]},
            ])
            return await self._progress_during(async_fresh_graph.aggregate(
                Start(near=Near("summary", text="q"), keep=5), aggregates={"n": Count()}))

        result, ticks = run(body())
        assert result == {"n": 2}
        assert ticks > 0

    def test_vector_search_text_does_not_block_the_loop(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=self.slow_embed())])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 1, "summary": [1.0, 0.0, 0.0]}, {"id": 2, "summary": [0.0, 1.0, 0.0]},
            ])
            return await self._progress_during(
                async_fresh_graph.vector_search(Near("summary", text="q"), k=1))

        hits, ticks = run(body())
        assert [h["id"] for h in hits] == ["1"]
        assert ticks > 0

    def test_vector_search_many_text_does_not_block_the_loop(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=self.slow_embed())])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 1, "summary": [1.0, 0.0, 0.0]}, {"id": 2, "summary": [0.0, 1.0, 0.0]},
            ])
            return await self._progress_during(async_fresh_graph.vector_search_many(
                [Near("summary", text="q")], k=1))

        results, ticks = run(body())
        assert [h["id"] for h in results[0]] == ["1"]
        assert ticks > 0

    def test_set_vectors_text_does_not_block_the_loop(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=self.slow_embed())])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}])
            return await self._progress_during(
                async_fresh_graph.set_vectors(nodes=[{"id": 1, "summary": "a paper about Raft"}]))

        written, ticks = run(body())
        assert written == 1
        assert ticks > 0


class TestTextEmbeddingCorrectness:
    """The resolved vector has to be the SAME one validate_nears() would
    have produced synchronously -- aresolve_spec_texts()/aresolve_near()/
    aresolve_queries()/aplan_vector_writes() only move WHEN the provider
    is called, never WHAT is asked of it or what is done with the
    answer."""

    def test_same_answer_as_passing_the_vector_directly(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(
                nodes=[Vector("summary", 3, embed=lambda texts: [[1.0, 0.0, 0.0] for _ in texts])])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 1, "summary": [1.0, 0.0, 0.0]}, {"id": 2, "summary": [0.0, 1.0, 0.0]},
            ])
            by_text = await async_fresh_graph.vector_search(Near("summary", text="q"), k=2)
            by_vector = await async_fresh_graph.vector_search(Near("summary", [1.0, 0.0, 0.0]), k=2)
            return by_text, by_vector

        by_text, by_vector = run(body())
        assert by_text == by_vector

    def test_a_field_with_no_embedder_names_the_right_hop(self, async_fresh_graph):
        """The FULL validate_nears() error still fires, with its usual
        per-hop label -- only the provider call itself moved earlier;
        every other check stays exactly where it was, in the sync path."""
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3)])   # no embed=
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.traverse(
                Start(),
                Hop(),
                Hop(near=Near("summary", text="q"), keep=1))

        with pytest.raises(ValueError, match=r"hop 1 \(unlabeled\)"):
            run(body())

    def test_one_field_reused_across_hops_costs_one_provider_call(self, async_fresh_graph):
        """aresolve_spec_texts() batches by (target, field) across the
        WHOLE chain -- a chain that asks the same field to embed text
        twice must still cost one round trip, not one per occurrence."""
        calls = []

        def counting_embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=counting_embed)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}, {"id": 3}])
            await async_fresh_graph.add_edges([
                {"start_id": 1, "end_id": 2, "kind": "knows"},
                {"start_id": 2, "end_id": 3, "kind": "knows"},
            ])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 1, "summary": [1.0, 0.0, 0.0]}, {"id": 2, "summary": [1.0, 0.0, 0.0]},
                {"id": 3, "summary": [1.0, 0.0, 0.0]},
            ])
            return await async_fresh_graph.traverse(
                Start(near=Near("summary", text="q1"), keep=5),
                Hop(via={"kind": "knows"}, near=Near("summary", text="q2"), keep=5))

        run(body())
        # ONE call, carrying BOTH texts -- not two calls of one each.
        assert calls == [["q1", "q2"]]

    def test_different_fields_are_embedded_concurrently_not_sequentially(self, async_fresh_graph):
        """A chain ranking two DIFFERENT fields by text batches to two
        provider calls (one call per field is unavoidable -- they are
        different embedders), but those two calls must be gathered, not
        awaited one after the other: the whole resolution should cost
        roughly ONE round trip's worth of wall-clock, not their sum."""
        delay = 0.2

        def slow(vector):
            def embed(texts):
                time.sleep(delay)
                return [list(vector) for _ in texts]
            return embed

        async def body():
            async_fresh_graph.define_vectors(
                nodes=[Vector("summary", 3, embed=slow((1.0, 0.0, 0.0)))],
                edges=[Vector("rel", 3, embed=slow((1.0, 0.0, 0.0)))])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.add_edges([{"id": 91, "start_id": 1, "end_id": 2}])
            await async_fresh_graph.set_vectors(
                nodes=[{"id": 1, "summary": [1.0, 0.0, 0.0]}, {"id": 2, "summary": [1.0, 0.0, 0.0]}],
                edges=[{"id": 91, "rel": [1.0, 0.0, 0.0]}])
            t0 = time.monotonic()
            await async_fresh_graph.traverse(
                Start(near=Near("summary", text="q1"), keep=5),
                Hop(via_near=Near("rel", text="q2"), via_keep=5))
            return time.monotonic() - t0

        elapsed = run(body())
        # Sequential would be >= 2 * delay; gathered stays close to one.
        assert elapsed < delay * 1.7, f"took {elapsed:.2f}s -- the two embed calls ran sequentially"


class TestOutOfScope:
    """Schema/constraint DDL has no async override -- see hopai/asyncio.py's
    module docstring. Each must refuse LOUD, naming the fix, rather than
    silently reaching the async engine's sync facade outside the greenlet
    bridge (which would raise SQLAlchemy's own MissingGreenlet instead)."""

    @pytest.mark.parametrize("name", [
        "create_schema", "drop_schema", "define_constraints", "drop_constraints",
        "enforce_schema", "save_schema", "load_schema", "infer_schema",
        "schema_violations", "add_networkx", "load_vectors",
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

    def test_delete_edges_all_true_reaches_the_call(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.add_edges([{"start_id": 1, "end_id": 2}])
            # No where/start/end -- only all=True makes this legal, so a
            # dropped `all` turns a delete into a refusal.
            return await async_fresh_graph.delete_edges(all=True)

        assert run(body()).deleted_edges == 1

    def test_update_edges_replace_true_reaches_the_call(self, async_fresh_graph):
        async def body():
            await async_fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
            await async_fresh_graph.add_edges([{"start_id": 1, "end_id": 2,
                                                "kind": "knows", "weight": 3}])
            await async_fresh_graph.update_edges(where={"kind": "knows"},
                                                 set={"kind": "knows", "seen": True},
                                                 replace=True)
            return await async_fresh_graph.traverse(Start(), Hop(hops=(1, 1)))

        result = run(body())
        # replace=True means the map IS the properties bag: a dropped
        # `replace` merges instead, and "weight" survives.
        assert result.edges[0]["properties"] == {"kind": "knows", "seen": True}

    def test_vector_search_many_boost_reaches_the_call(self, async_fresh_graph):
        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 2)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1, "rank": 0}, {"id": 2, "rank": 100}])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 1, "summary": [1.0, 0.0]},
                # Not fully orthogonal (0.1 instead of 0.0 on the first
                # axis): with the default scale="normalized" (#55), rank 0
                # vs 100 normalizes to a boost of exactly 0 vs 1 * weight,
                # so an exactly-orthogonal node 2 would only TIE node 1's
                # similarity of 1.0 rather than beat it -- a tie the id
                # tiebreak resolves in node 1's favor, which would make
                # this assert the wrong thing for the wrong reason.
                {"id": 2, "summary": [0.1, 1.0]},
            ])
            # Node 1 is the better cosine match; node 2's `rank` is large
            # enough that the boost has to reorder them. A dropped boost=
            # leaves node 1 first and says nothing.
            return await async_fresh_graph.vector_search_many(
                [Near("summary", [1.0, 0.0])], k=2, boost=Boost("rank"))

        assert [hit["id"] for hit in run(body())[0]] == ["2", "1"]

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


def _consume(passed: list, expected: dict, name: str) -> None:
    """Match each expected value against a DISTINCT recorded argument.

    A multiset, not membership: delete_nodes passes detach=True and
    all=True, so `True in passed` still succeeds after one of them is
    dropped -- the other one answers for it. The first version of this
    test used `in` and missed exactly that mutant."""
    remaining = list(passed)
    for keyword, value in expected.items():
        for i, item in enumerate(remaining):
            if item is value or item == value:
                del remaining[i]
                break
        else:
            raise AssertionError(
                f"{name}() did not pass {keyword}={value!r} -- recorded {passed!r}")


class TestEveryWrapperForwardsEveryArgument:
    """AsyncGraph is ~25 one-line delegations, and the mutation runs have
    now picked six different ones apiece: boost off vector_search_many,
    all off delete_edges, replace off update_edges, merge_edges_on off
    ingest, hops off cypher's aggregate branch, all off update_nodes,
    and merge_edges' `replace` default flipped outright. Every prior
    test called each wrapper with the defaults for whatever it was not
    itself about, so each was invisible in turn.

    tests/test_vectors.py has the same table for Graph's vector methods;
    this is the write half, and between them the family is closed rather
    than its members picked off one run at a time.

    Both directions are covered, because they fail differently: passing
    a NON-DEFAULT value catches an argument dropped on the way through,
    and passing NOTHING catches a default rewritten in the signature --
    the shape that flipped merge_edges to replace=True."""

    #: {AsyncGraph method: (delegate to record, required args, optional args)}
    #: The delegate is patched on the class it lives on, so `self`
    #: arrives first and is ignored -- only the values matter.
    CALLS = {
        "merge_nodes": ("hopai.ingest.Ingestor.merge_nodes",
                        {"rows": [{"a": 1}], "on": ["a"]}, {"replace": True}),
        "merge_edges": ("hopai.ingest.Ingestor.merge_edges",
                        {"rows": [{"a": 1}], "on": ["a"]}, {"replace": True}),
        "ingest": ("hopai.ingest.Ingestor.ingest", {"document": {"nodes": []}},
                   {"merge_nodes_on": ["mn"], "merge_edges_on": ["me"]}),
        "delete_nodes": ("hopai.mutate.Mutator.delete_nodes", {"where": {"w": 1}},
                         {"detach": True, "all": True}),
        "delete_edges": ("hopai.mutate.Mutator.delete_edges", {"where": {"w": 1}},
                         {"start": {"s": 1}, "end": {"e": 1}, "all": True}),
        "update_nodes": ("hopai.mutate.Mutator.update_nodes", {"where": {"w": 1}},
                         {"set": {"s": 1}, "remove": ["r"], "replace": True, "all": True}),
        "update_edges": ("hopai.mutate.Mutator.update_edges", {"where": {"w": 1}},
                         {"start": {"s": 1}, "end": {"e": 1}, "set": {"v": 1},
                          "remove": ["r"], "replace": True, "all": True}),
    }

    #: What each optional argument is when the caller says nothing. A
    #: signature default rewritten in place changes behaviour for every
    #: caller who trusted it, and no test that passes the value can see it.
    DEFAULTS = {
        "merge_nodes": {"replace": False},
        "merge_edges": {"replace": False},
        "ingest": {"merge_nodes_on": None, "merge_edges_on": None},
        "delete_nodes": {"detach": False, "all": False},
        "delete_edges": {"start": None, "end": None, "all": False},
        "update_nodes": {"set": None, "remove": None, "replace": False, "all": False},
        "update_edges": {"start": None, "end": None, "set": None, "remove": None,
                         "replace": False, "all": False},
    }

    @staticmethod
    def _record(monkeypatch, target: str) -> dict:
        seen: dict = {}

        def recorder(self, *args, **kwargs):
            seen["passed"] = [*args, *kwargs.values()]
            return "recorded"

        monkeypatch.setattr(target, recorder)
        return seen

    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_non_default_values_reach_the_sync_call(self, async_graph, monkeypatch, name):
        target, required, optional = self.CALLS[name]
        seen = self._record(monkeypatch, target)
        run(getattr(async_graph, name)(**required, **optional))
        _consume(seen["passed"], optional, name)

    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_the_documented_defaults_are_what_arrive(self, async_graph, monkeypatch, name):
        target, required, _ = self.CALLS[name]
        seen = self._record(monkeypatch, target)
        run(getattr(async_graph, name)(**required))
        _consume(seen["passed"], self.DEFAULTS[name], name)
