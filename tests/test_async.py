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

import hopai.asyncio as asyncio_module
from hopai import Boost, Count, CypherError, Hop, Near, Start, Unique, Vector
from hopai.asyncio import AsyncGraph


def run(coro):
    return asyncio.run(coro)


async def _while_ticking(ticks: list, coro):
    """Await `coro` alongside an independent task that counts, into
    `ticks`, every time the event loop came back to it.

    This is the measurement issue #74 asks for and the reason these
    tests are not "probably fine": the counter is ordinary loop work,
    so it advances only while the loop is free to run something other
    than `coro`. A provider call that holds the loop's thread shows up
    as a stretch where the count does not move at all."""
    running = True

    async def tick():
        while running:
            await asyncio.sleep(0.001)
            ticks[0] += 1

    counter = asyncio.ensure_future(tick())
    try:
        return await coro
    finally:
        running = False
        await counter


#: How much of an idle loop's throughput a call must leave behind it.
#:
#: `ticks > 0` -- what these tests asserted first -- is a test for TOTAL
#: starvation and nothing else, and total starvation is not the only way
#: to block a loop: a bounded 600ms stall inside a one-second call scores
#: hundreds of ticks and passed every one of them. That is exactly how a
#: 121ms document-building block lived here unnoticed (see
#: hopai/rerankers.py on pruning). So the assertion is a RATIO against a
#: baseline measured on the same machine in the same run, which is the
#: only form that survives a loaded CI box: an absolute ticks/s would be
#: either flaky or vacuous.
#:
#: Half is deliberately generous. A free loop measured ~880-890 ticks/s
#: here; the unpruned 100KB case measured ZERO during the block and ~20%
#: over the call containing it, and 500KB measured lower still. Anything
#: between 50% and 100% is ordinary scheduling noise, and nothing this
#: library does should land there.
LOOP_FLOOR = 0.5


#: How much longer than an idle loop's own worst pause a call may hold
#: the thread in ONE stretch, and the floor below which the comparison
#: stops being meaningful.
#:
#: The rate above is the right instrument for a window of known length
#: -- a 200ms provider call -- and the wrong one for a call whose length
#: is not fixed: a traversal spends most of itself in the database, so a
#: 121ms block inside a 185ms call reads as 29% of idle while the same
#: block inside a 1s call reads as 88%. Measured here, the step-wise
#: path's rate swung between 33% and 69% run to run for that reason
#: alone.
#:
#: The LONGEST GAP between ticks does not have that problem. It is how
#: long, in one stretch, the loop was unavailable -- which is what
#: "blocking" means -- and it is unaffected by how much awaiting
#: happened either side. Measured on this path: 107ms unpruned against
#: 8ms pruned, with an idle loop's own worst pause at 5ms.
GAP_MULTIPLE = 5
GAP_FLOOR = 0.030


async def _profile(coro):
    """(result, ticks/s, longest gap in seconds) while `coro` runs.

    The gap is measured from the ticker's own timestamps rather than
    from a count, because a count cannot tell 500 evenly spread ticks
    from 500 ticks either side of a stall."""
    stamps = [time.perf_counter()]
    running = True

    async def tick():
        while running:
            await asyncio.sleep(0.001)
            stamps.append(time.perf_counter())

    counter = asyncio.ensure_future(tick())
    started = time.perf_counter()
    try:
        result = await coro
    finally:
        elapsed = time.perf_counter() - started
        running = False
        await counter
    gaps = [after - before for before, after in zip(stamps, stamps[1:], strict=False)]
    return result, len(gaps) / elapsed, max(gaps, default=elapsed)


async def _idle_profile(seconds: float = 0.3):
    """(ticks/s, longest gap) for a loop with nothing else to do.

    Measured per test rather than hard-coded: `asyncio.sleep(0.001)`
    resolves to the platform's timer granularity plus whatever else is
    running on the box, so the only honest baseline is one taken next
    to the measurement it is compared with."""
    _, rate, gap = await _profile(asyncio.sleep(seconds))
    return rate, gap


async def _idle_rate(seconds: float = 0.3) -> float:
    """Ticks per second a loop with nothing else to do achieves."""
    rate, _ = await _idle_profile(seconds)
    return rate


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

    def test_a_reranked_traversal_answers_the_same_either_way(
            self, async_fresh_graph, async_admin_graph):
        """AsyncGraph has its OWN rerank driver -- the probes go through
        run_sync() and the scoring is awaited outside it -- so unlike
        every case above there really are two orderings of the same
        steps here. They must still pin the same survivors and re-run
        the same ordinary traversal."""
        pytest.importorskip("jq")
        from hopai import Rerank

        def seed(g):
            g.define_vectors(nodes=[Vector("summary", 3, embed=lambda t: [[1.0, 0.0, 0.0]
                                                                          for _ in t])])

        def rerank():
            # `far` is the WORST cosine and the best document, so only a
            # real rerank keeps it -- and both paths must agree on that.
            return Rerank(lambda query, documents: [1.0 if d == "far" else 0.0
                                                    for d in documents],
                          document_from=".properties.name", candidates=5)

        def spec():
            return (Start(where={"type": "doc"}),
                    Hop(via={"kind": "cites"}, near=Near("summary", text="q"), keep=1,
                        rerank=rerank()))

        async def body():
            seed(async_fresh_graph)
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([
                {"id": 1, "type": "doc", "name": "seed"},
                {"id": 2, "name": "near"}, {"id": 3, "name": "far"}])
            await async_fresh_graph.set_vectors(nodes=[
                {"id": 2, "summary": [1.0, 0.0, 0.0]}, {"id": 3, "summary": [-1.0, 0.0, 0.0]}])
            await async_fresh_graph.add_edges([
                {"start_id": 1, "end_id": 2, "kind": "cites"},
                {"start_id": 1, "end_id": 3, "kind": "cites"}])
            return await async_fresh_graph.traverse(*spec())

        async_result = run(body())
        # The same rows, reached through the plain sync Graph the
        # async_admin_graph fixture points at the very same schema.
        seed(async_admin_graph)
        sync_result = async_admin_graph.traverse(*spec())
        assert {n["id"] for n in async_result.nodes} == {"1", "3"}
        assert sync_result.nodes == async_result.nodes
        assert sync_result.edges == async_result.edges

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

    def test_a_batch_of_queries_still_costs_one_provider_call_per_field(
            self, async_fresh_graph):
        """vector_search_many() exists to turn N round trips into one,
        and aresolve_queries() must not put them back on the way to the
        embedder -- the async twin of test_vectors.py's
        test_many_queries_cost_one_provider_call_per_field, which pins
        the same thing for the sync resolver."""
        calls = []

        def counting_embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=counting_embed)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}])
            await async_fresh_graph.set_vectors(nodes=[{"id": 1, "summary": [1.0, 0.0, 0.0]}])
            return await async_fresh_graph.vector_search_many(
                [Near("summary", text="apple"), Near("summary", text="banana"),
                 Near("summary", text="cherry")], k=1)

        results = run(body())
        assert len(results) == 3
        assert calls == [["apple", "banana", "cherry"]]

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


class TestValidationRunsBeforeEmbedding:
    """A call about to be refused for a reason that has NOTHING to do
    with the provider must never pay for -- or fail because of -- an
    embedding round trip first. Hoisting text resolution earlier
    (aresolve_spec_texts()/aresolve_near()/aresolve_queries() in
    hopai/vectors.py) would otherwise let exactly that happen: a
    caller mistake (duplicate field, bad k, a misplaced optional=True)
    reaching the provider before the cheap, sync-side check that would
    have caught it -- and, if the provider itself then fails, surfacing
    as a confusing EmbeddingError instead of the real ValueError/
    TypeError. Each test below asserts BOTH the right exception AND
    that the embedder was never called."""

    def test_duplicate_field_in_start_near_is_refused_before_embedding(self, async_fresh_graph):
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=embed)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.traverse(
                Start(near=[Near("summary", text="a"), Near("summary", text="b")], keep=1))

        with pytest.raises(ValueError, match="two Near specs both rank field"):
            run(body())
        assert calls == []

    def test_a_non_near_item_is_refused_before_embedding(self, async_fresh_graph):
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=embed)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.traverse(
                Start(near=[Near("summary", text="a"), "not-a-near"], keep=1))

        with pytest.raises(TypeError, match="near= takes Near"):
            run(body())
        assert calls == []

    def test_a_negative_k_on_vector_search_is_refused_before_embedding(self, async_fresh_graph):
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=embed)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.vector_search(Near("summary", text="q"), k=-5)

        with pytest.raises(ValueError, match="k must be a positive integer"):
            run(body())
        assert calls == []

    def test_a_negative_k_on_vector_search_many_is_refused_before_embedding(
            self, async_fresh_graph):
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=embed)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.vector_search_many([Near("summary", text="q")], k=-5)

        with pytest.raises(ValueError, match="k must be a positive integer"):
            run(body())
        assert calls == []

    def test_optional_on_a_non_last_hop_is_refused_before_embedding(self, async_fresh_graph):
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=embed)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.traverse(
                Start(),
                Hop(optional=True),
                Hop(near=Near("summary", text="q"), keep=1))

        with pytest.raises(ValueError, match="optional=True is only supported on the LAST hop"):
            run(body())
        assert calls == []

    def test_optional_on_an_aggregate_hop_is_refused_before_embedding(self, async_fresh_graph):
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=embed)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.aggregate(
                Start(),
                Hop(optional=True, near=Near("summary", text="q"), keep=1),
                aggregates={"n": Count()})

        with pytest.raises(ValueError, match="optional=True has no effect on an aggregation"):
            run(body())
        assert calls == []

    def test_an_empty_aggregates_dict_is_refused_before_embedding(self, async_fresh_graph):
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        async def body():
            async_fresh_graph.define_vectors(nodes=[Vector("summary", 3, embed=embed)])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.aggregate(
                Start(near=Near("summary", text="q"), keep=1), aggregates={})

        with pytest.raises(ValueError, match="aggregates must be a non-empty dict"):
            run(body())
        assert calls == []


class TestRerankingStaysOffTheLoop:
    """Issue #73's provider call, held to issue #74's rule.

    Reranking is the SECOND network call on the read path, and it lands
    in the same trap from the other end: it scores rows the database has
    already returned, so the obvious place to put it is inside the
    run_sync() block that fetched them -- which is the event loop's own
    thread. Every test below runs the reranked call next to an
    independent ticking task and asserts that task got somewhere DURING
    the provider call. On a version that scored inside the bridge, each
    one scores exactly zero: that is the measurement, not an assumption.

    Embedding is not what these measure -- TestTextEmbeddingStaysOffTheLoop
    above does that -- so the embedder here is instant on purpose, and
    the ticks are counted by the RERANKER, inside its own call."""

    DIMS = 3

    def _embed(self, texts):
        return [[1.0] + [0.0] * (self.DIMS - 1) for _ in texts]

    async def _seeded(self, graph):
        """A graph with one vector field per target, a node and an edge
        already embedded -- everything the reads below need, with the
        setup's own provider calls already spent."""
        graph.define_vectors(nodes=[Vector("summary", self.DIMS, embed=self._embed)],
                             edges=[Vector("rel", self.DIMS, embed=self._embed)])
        await graph.migrate_vectors()
        await graph.add_nodes([{"id": 1, "type": "doc"}, {"id": 2, "type": "doc"}])
        await graph.add_edges([{"id": 9, "start_id": 1, "end_id": 2, "kind": "cites"}])
        await graph.set_vectors(nodes=[{"id": 1, "summary": [1.0, 0.0, 0.0]},
                                       {"id": 2, "summary": [1.0, 0.0, 0.0]}],
                                edges=[{"id": 9, "rel": [1.0, 0.0, 0.0]}])

    #: How long each fake provider call takes. The window `progress`
    #: below is a rate over, so it has to be long enough that a handful
    #: of ticks either way is not the answer.
    CALL = 0.2

    def _rerank(self, ticks, scores=None, per_call=None, sync=False, **options):
        """A Rerank whose provider call takes a visible amount of time
        and records the RATE the loop achieved while it was inside it.

        Async by default, since that is the shape this module is built
        around. `sync=True` gives the other one on purpose: a plain
        callable takes `Rerank._aattempt`'s own `asyncio.to_thread`,
        which is a DIFFERENT thread hop from Embedder's and was only
        ever unit-tested -- nothing ran a synchronous reranker client
        through AsyncGraph and watched the loop."""
        from hopai import Rerank

        progress, spans = [], []

        def answer(query, documents, started, before):
            progress.append((ticks[0] - before) / (time.perf_counter() - started))
            spans.append((started, time.perf_counter()))
            if per_call is not None:
                per_call.append(query)
            return [(scores or {}).get(d, 0.0) for d in documents]

        async def client(query, documents):
            before, started = ticks[0], time.perf_counter()
            await asyncio.sleep(self.CALL)
            return answer(query, documents, started, before)

        def blocking_client(query, documents):
            # time.sleep, not asyncio.sleep: a synchronous client BLOCKS,
            # and the only thing that keeps the loop alive through it is
            # the worker thread hopai puts it on.
            before, started = ticks[0], time.perf_counter()
            time.sleep(self.CALL)
            return answer(query, documents, started, before)

        if sync:
            client = blocking_client

        # `.id` rather than a property: _seeded()'s nodes carry only a
        # type, and what these tests are about is the loop, not the
        # document.
        options.setdefault("document_from", ".id")
        options.setdefault("candidates", 5)
        rerank = Rerank(client, **options)
        rerank.progress = progress
        rerank.spans = spans
        return rerank

    @staticmethod
    def _assert_free(rerank, idle):
        """Every provider call left the loop most of its throughput.

        A RATE against `idle`, not `> 0`: the old form passed on a
        version that blocked the loop for 600ms, because 600ms of block
        inside a longer call still lets a 1ms ticker score hundreds of
        times. See LOOP_FLOOR."""
        assert rerank.progress, "the reranker was never called"
        assert all(rate > idle * LOOP_FLOOR for rate in rerank.progress), (
            f"the loop managed {[round(r) for r in rerank.progress]} ticks/s during the "
            f"reranker calls against an idle {idle:.0f} -- it is back on its thread")

    def test_vector_search_does_not_hold_the_loop_while_it_reranks(self, async_fresh_graph):
        """Run inside run_sync() the scoring would sit on the event
        loop's own thread for the whole round trip. It is awaited
        OUTSIDE the bridge, and this is the measurement -- on a version
        that awaited it inside, the ticker below scores zero."""
        pytest.importorskip("jq")
        ticks = [0]
        rerank = self._rerank(ticks, scores={"2": 1.0})

        async def body():
            await self._seeded(async_fresh_graph)
            idle = await _idle_rate()
            return await _while_ticking(ticks, async_fresh_graph.vector_search(
                Near("summary", text="raft consensus"), k=1, rerank=rerank)), idle

        hits, idle = run(body())
        assert [hit["id"] for hit in hits] == ["2"]
        assert hits[0]["rerank_score"] == 1.0
        self._assert_free(rerank, idle)

    def test_vector_search_many_issues_its_rerank_calls_concurrently(self, async_fresh_graph):
        """N queries mean N reranker calls -- a score is the (query,
        document) relationship, so one call cannot serve two. But this
        call exists to turn N round trips into ONE, and issuing the
        provider calls one after another hands that straight back.
        Asserted on the calls' own clocks OVERLAPPING rather than on the
        wall time of the whole search -- the embedding round trip in
        front of it would otherwise decide the number."""
        pytest.importorskip("jq")
        ticks = [0]
        queries = []
        rerank = self._rerank(ticks, per_call=queries)

        async def body():
            await self._seeded(async_fresh_graph)
            return await _while_ticking(ticks, async_fresh_graph.vector_search_many(
                [Near("summary", text="raft"), Near("summary", text="paxos")], k=1,
                rerank=rerank))

        results = run(body())
        assert len(results) == 2
        assert queries == ["raft", "paxos"]          # one call per query, its own text
        starts, ends = zip(*rerank.spans, strict=True)
        assert len(rerank.spans) == 2 and max(starts) < min(ends), \
            "the reranker calls ran one after another, not together"

    def test_a_step_wise_rerank_does_not_hold_the_loop_either(self, async_fresh_graph):
        """The traversal path has its own driver -- probe, score, pin --
        and its own chance to score inside the bridge. The probe goes
        through run_sync() because it is ordinary SQL; the scoring must
        not."""
        pytest.importorskip("jq")
        ticks = [0]
        rerank = self._rerank(ticks)

        async def body():
            await self._seeded(async_fresh_graph)
            idle = await _idle_rate()
            return await _while_ticking(ticks, async_fresh_graph.traverse(
                Start(where={"type": "doc"}),
                Hop(via={"kind": "cites"}, near=Near("summary", text="raft"), keep=1,
                    rerank=rerank))), idle

        (result, idle) = run(body())
        assert sorted(node["id"] for node in result.nodes) == ["1", "2"]
        self._assert_free(rerank, idle)

    def test_a_reranked_step_survives_the_text_resolution(self, async_fresh_graph):
        """aresolve_spec_texts() rebuilds each step it resolves, and a
        Start/Hop validates on construction -- one of those rules being
        that rerank= needs the query as TEXT, which the rebuild has just
        turned into floats. Without vectors._resolved_spec() re-attaching
        the reranker AFTER the copy, every reranked-and-text-ranked
        traversal refuses itself with "rerank= needs the query as TEXT"
        for a query that had it."""
        pytest.importorskip("jq")
        ticks = [0]
        rerank = self._rerank(ticks, scores={"2": 1.0})

        async def body():
            await self._seeded(async_fresh_graph)
            return await async_fresh_graph.traverse(
                Start(near=Near("summary", text="raft"), keep=1, rerank=rerank))

        result = run(body())
        # Both nodes carry the same vector, so similarity alone would
        # keep node 1 on the id tiebreak. Node 2 surviving is the
        # reranker having run -- with the query text it was written
        # with, read before the resolution turned it into floats.
        assert [node["id"] for node in result.nodes] == ["2"]
        assert rerank.progress

    def test_a_step_that_finds_nothing_spends_no_provider_call(self, async_fresh_graph):
        """AsyncGraph has its own rerank driver, so the sync one's "no
        candidates, no call" shortcut has to exist here too -- awaiting
        `ascore` on an empty list would be a billed round trip for an
        answer nobody can use, and it would hold the loop for the length
        of it."""
        pytest.importorskip("jq")
        ticks = [0]
        queries = []
        rerank = self._rerank(ticks, per_call=queries)

        async def body():
            await self._seeded(async_fresh_graph)
            return await async_fresh_graph.traverse(
                Start(where={"type": "nothing-matches-this"},
                      near=Near("summary", text="raft"), keep=1, rerank=rerank),
                Hop(via={"kind": "cites"}))

        result = run(body())
        assert queries == [] and result.nodes == [] and result.edges == []

    def test_an_empty_search_result_is_not_sent_to_the_reranker_either(
            self, async_fresh_graph):
        """The flat path's own version of the same shortcut, on the
        awaited side."""
        pytest.importorskip("jq")
        ticks = [0]
        queries = []
        rerank = self._rerank(ticks, per_call=queries)

        async def body():
            await self._seeded(async_fresh_graph)
            return await async_fresh_graph.vector_search(
                Near("summary", text="raft"), k=1, where={"type": "nothing-matches-this"},
                rerank=rerank)

        assert run(body()) == []
        assert queries == []

    def test_a_synchronous_reranker_client_does_not_hold_the_loop(self, async_fresh_graph):
        """The gap every other test here left: a client that BLOCKS.

        `Rerank._aattempt` puts a synchronous client on
        `asyncio.to_thread`, and that thread hop is its own -- it is not
        Embedder's, which the embedding tests cover, and it had only
        ever been unit-tested on the Rerank in isolation. Nothing ran
        one through AsyncGraph and watched the loop, which is the level
        the hop actually has to survive: `_rerank_pins()` and
        `arerank_hits()` both call `ascore()`, and either could have
        awaited a blocking call directly with no test saying so."""
        pytest.importorskip("jq")
        ticks = [0]
        rerank = self._rerank(ticks, scores={"2": 1.0}, sync=True)
        assert rerank.is_async is False                  # the shape under test

        async def body():
            await self._seeded(async_fresh_graph)
            idle = await _idle_rate()
            return await _while_ticking(ticks, async_fresh_graph.vector_search(
                Near("summary", text="raft consensus"), k=1, rerank=rerank)), idle

        hits, idle = run(body())
        assert [hit["id"] for hit in hits] == ["2"]
        self._assert_free(rerank, idle)

    def test_a_synchronous_client_stays_off_the_loop_at_a_hop_too(self, async_fresh_graph):
        """The traversal driver has its own path to `ascore()`, so the
        same thread hop has to hold there -- `_rerank_pins()` awaits it
        once per reranked step, outside the bridge."""
        pytest.importorskip("jq")
        ticks = [0]
        rerank = self._rerank(ticks, sync=True)

        async def body():
            await self._seeded(async_fresh_graph)
            idle = await _idle_rate()
            return await _while_ticking(ticks, async_fresh_graph.traverse(
                Start(where={"type": "doc"}),
                Hop(via={"kind": "cites"}, near=Near("summary", text="raft"), keep=1,
                    rerank=rerank))), idle

        (result, idle) = run(body())
        assert sorted(node["id"] for node in result.nodes) == ["1", "2"]
        self._assert_free(rerank, idle)

    def test_per_path_reranking_does_not_hold_the_loop(self, async_fresh_graph):
        """The other gap: `per_path=True` was covered only synchronously.

        It is the expensive mode -- one document per (node, route)
        rather than per node -- so it is the one whose document building
        multiplies, and it takes a WIDER probe (`.paths` is read, so the
        hop hydrates every node on every route) before it gets there.
        Every extra byte of that was, before pruning, extra time on the
        event loop's thread."""
        pytest.importorskip("jq")
        ticks = [0]
        rerank = self._rerank(ticks, per_path=True,
                              document_from='.id + " via " + (.paths | tostring)')

        async def body():
            await self._seeded(async_fresh_graph)
            idle = await _idle_rate()
            return await _while_ticking(ticks, async_fresh_graph.traverse(
                Start(where={"type": "doc"}),
                Hop(via={"kind": "cites"}, near=Near("summary", text="raft"), keep=1,
                    rerank=rerank))), idle

        (result, idle) = run(body())
        assert sorted(node["id"] for node in result.nodes) == ["1", "2"]
        self._assert_free(rerank, idle)


class TestDocumentBuildingStaysOffTheLoop:
    """The block the tick-counting tests above could not see, and the
    reason they could not.

    `_rerank()`'s progress measures the window INSIDE the provider call.
    Document building happens before it -- `build_documents()` on the
    event loop's own thread at hopai/asyncio.py's `_rerank_pins()` and
    at vectors.py's `arerank_hits()` -- so a stall there fell outside
    every window this file measured, and `ticks > 0` over the whole call
    could not distinguish it from scheduling noise. Measured, fifty
    100KB candidates held the thread for 121ms with the ticker scoring
    ZERO: the signature of a `time.sleep`, not of a slow function.

    WHY THIS CLASS DOES NOT COUNT TICKS, having every reason to. Three
    instruments were tried against this call and two of them are traps:

      - THROUGHPUT RATIO (ticks/s against an idle baseline). Wrong for a
        call whose length is not fixed. A reranked traversal spends most
        of itself in the database, so the SAME block is a different
        fraction of two runs: measured between 33% and 69% of idle with
        the fix in place, straddling any threshold worth setting. It is
        the right instrument for the 200ms provider window above, where
        the denominator is known, and it is stable there under load.
      - LONGEST GAP between ticks. Better -- it is what "blocking" means
        and it does not move with the call's length (107ms unpruned
        against 8ms pruned) -- but it is an EXTREME-VALUE statistic over
        a few dozen samples, and on a loaded box the operating system
        supplies the extreme. Run under 2x nproc busy loops it failed a
        different test of this class on most runs. CI runs four Python
        versions in parallel on shared runners, so that is the
        environment, not a pathological case.
      - WHAT IS ACTUALLY ASSERTED HERE: the two facts the fix is made
        of, neither of which a loaded machine can move.

        1. HOW MUCH jq IS HANDED. The bug was that `input_value()`
           marshals whatever it is given, so `.properties.title` was
           handed 100KB to read 60 characters out of. That is a byte
           count, and it is exact.
        2. HOW MUCH CPU THE BUILD SPENDS, by `time.process_time` --
           which counts this process's own cycles and ignores every
           other process on the box. 0.7ms pruned against 120ms
           unpruned, and descheduling cannot inflate it.

    Between them they say precisely what the wall-clock instruments were
    reaching for, with no threshold that depends on the machine. The
    wall-clock numbers themselves are still reported, by
    benchmarks/bench_documents.py, where a number is measured rather
    than gated on."""

    DIMS = 3
    ROWS = 40
    KILOBYTES = 100

    #: A pruned view of `.properties.title` is about 30 bytes; the
    #: candidate it came from is over 100,000. Anything under a kilobyte
    #: proves the row is not what was handed over, with two orders of
    #: magnitude to spare either side.
    MAX_BYTES_HANDED = 1000

    #: Seconds of THIS PROCESS's CPU the whole build may spend. Measured:
    #: 0.7ms with the fix, ~120ms without it, so 25ms sits ~35x above one
    #: and ~5x below the other.
    MAX_CPU_SECONDS = 0.025

    def _embed(self, texts):
        return [[1.0] + [0.0] * (self.DIMS - 1) for _ in texts]

    async def _seeded(self, graph):
        """Nodes carrying a big `properties` payload beside a short title
        -- the shape that made reading the TITLE cost what reading the
        body would."""
        from hopai import Vector

        graph.define_vectors(nodes=[Vector("summary", self.DIMS, embed=self._embed)])
        await graph.migrate_vectors()
        body = "x" * 1000
        await graph.add_nodes([
            dict({"id": i, "type": "doc", "title": f"node {i}"},
                 **{f"body_{b}": body for b in range(self.KILOBYTES)})
            for i in range(self.ROWS)])
        await graph.set_vectors(
            nodes=[{"id": i, "summary": [1.0, 0.0, 0.0]} for i in range(self.ROWS)])

    def _watched(self, monkeypatch):
        """A Rerank that records what its document building was handed
        and what it cost, through the REAL AsyncGraph call sites.

        The spy keeps a REFERENCE to each view rather than measuring it
        -- sizing 40 candidates of 100KB inside the window being timed
        would be most of the CPU the window is asserting about."""
        import hopai.rerankers as rerankers_module
        from hopai import Rerank

        handed, cost = [], []
        original = rerankers_module._evaluate

        def spy(program, candidate, filter_text, view=None):
            handed.append(candidate if view is None else view)
            return original(program, candidate, filter_text, view)

        monkeypatch.setattr(rerankers_module, "_evaluate", spy)

        class Watched(Rerank):
            def build_documents(self, candidates, **options):
                started = time.process_time()
                try:
                    return super().build_documents(candidates, **options)
                finally:
                    cost.append(time.process_time() - started)

        rerank = Watched(lambda query, documents: [0.0] * len(documents),
                         document_from=".properties.title", candidates=self.ROWS)
        rerank.handed, rerank.cost = handed, cost
        return rerank

    def _assert_cheap(self, rerank):
        import json

        assert rerank.handed, "build_documents() never ran"
        assert rerank.cost, "the build was never measured"
        biggest = max(len(json.dumps(view)) for view in rerank.handed)
        assert biggest <= self.MAX_BYTES_HANDED, (
            f"jq was handed {biggest} bytes for one candidate -- the filter reads a "
            f"title, so it is being handed the row again and the cost is back to being "
            f"proportional to the payload")
        spent = sum(rerank.cost)
        assert spent < self.MAX_CPU_SECONDS, (
            f"document building spent {spent * 1000:.0f}ms of CPU over "
            f"{len(rerank.handed)} candidate(s), above the {self.MAX_CPU_SECONDS * 1000:.0f}ms "
            f"budget -- on the async path every millisecond of that is the event loop's "
            f"own thread")

    def test_the_flat_search_path_hands_jq_only_the_projection(
            self, async_fresh_graph, monkeypatch):
        """vectors.arerank_hits()' call site. `.properties.title` reads
        60 characters out of a 100KB candidate, and before pruning it
        cost exactly what reading the 100KB would -- because
        `input_value()` marshalled the whole row either way."""
        pytest.importorskip("jq")
        rerank = self._watched(monkeypatch)

        async def body():
            await self._seeded(async_fresh_graph)
            return await async_fresh_graph.vector_search(
                Near("summary", text="raft"), k=5, rerank=rerank)

        assert len(run(body())) == 5
        assert len(rerank.handed) == self.ROWS      # every candidate went through jq
        self._assert_cheap(rerank)

    def test_the_step_wise_path_hands_jq_only_the_projection(
            self, async_fresh_graph, monkeypatch):
        """asyncio._rerank_pins()' call site -- the other one, and the
        one that runs once per reranked STEP rather than once per
        query."""
        pytest.importorskip("jq")
        rerank = self._watched(monkeypatch)

        async def body():
            await self._seeded(async_fresh_graph)
            return await async_fresh_graph.traverse(
                Start(where={"type": "doc"}, near=Near("summary", text="raft"),
                      keep=5, rerank=rerank))

        result = run(body())
        assert len(result.nodes) == 5
        assert len(rerank.handed) == self.ROWS
        self._assert_cheap(rerank)



class TestNoTextTakesNoNewPath:
    """The negative half of issue #74, and the async twin of
    test_defining_vectors_changes_no_near_less_query: resolving text
    earlier must be invisible to every traversal that has none. Not
    "equivalent" -- the SAME objects reach the same sync function, so
    there is nothing for the SQL to differ about."""

    def test_a_traversal_without_text_is_handed_the_very_same_spec(
            self, async_graph, monkeypatch):
        """Identity, not equality: a rebuilt-but-equal Start would pass
        an == check while proving nothing about the path taken."""
        from hopai.core import Graph

        seen = {}
        original = Graph._traverse_with_session

        def record(self, session, start, hops, pins=None):
            seen["start"], seen["hops"], seen["pins"] = start, hops, pins
            return original(self, session, start, hops, pins=pins)

        monkeypatch.setattr(Graph, "_traverse_with_session", record)
        start, hop = Start(where={"type": "leaf"}), Hop(via={"kind": "knows"})
        run(async_graph.traverse(start, hop))
        assert seen["start"] is start
        assert seen["hops"] == [hop] and seen["hops"][0] is hop
        # And no pins: a chain with no rerank= must reach build_query()
        # with pins=None, which is what keeps its SQL byte-identical to
        # what it emitted before step-wise reranking existed.
        assert seen["pins"] is None

    def test_a_vector_valued_near_never_reaches_the_embedder(self, async_fresh_graph):
        """A Near carrying floats is already resolved, so declaring an
        embedder on the field must not make the async path call it --
        that would be a provider bill (and a round trip) for a query
        that asked for none."""
        calls = []

        async def body():
            async_fresh_graph.define_vectors(nodes=[
                Vector("summary", 2, embed=lambda texts: calls.append(texts) or [[1.0, 0.0]])])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}])
            await async_fresh_graph.set_vectors(nodes=[{"id": 1, "summary": [1.0, 0.0]}])
            await async_fresh_graph.traverse(
                Start(near=Near("summary", [1.0, 0.0]), keep=1))
            await async_fresh_graph.vector_search(Near("summary", [1.0, 0.0]), k=1)
            await async_fresh_graph.vector_search_many([Near("summary", [1.0, 0.0])], k=1)

        run(body())
        assert calls == []

    def test_a_vector_carrying_near_beside_a_text_one_is_left_alone(
            self, async_fresh_graph):
        """A multivector query may mix the two. Only the text half is
        embedded -- resolving "everything in the list" would re-embed a
        vector the caller supplied, which is not text and cannot be."""
        embedded = []

        async def body():
            async_fresh_graph.define_vectors(nodes=[
                Vector("summary", 2, embed=lambda texts: embedded.extend(texts)
                       or [[1.0, 0.0] for _ in texts]),
                Vector("title", 2),          # no embedder: never asked for one
            ])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}])
            await async_fresh_graph.set_vectors(
                nodes=[{"id": 1, "summary": [1.0, 0.0], "title": [1.0, 0.0]}])
            return await async_fresh_graph.vector_search(
                Near("summary", text="raft"), Near("title", [1.0, 0.0]), k=1)

        assert [hit["id"] for hit in run(body())] == ["1"]
        assert embedded == ["raft"]

    def test_a_multivector_batch_resolves_only_its_text(self, async_fresh_graph):
        """vector_search_many() takes a LIST of queries, each of which
        may itself be a list -- so "does this need a provider" has to
        look one level deeper than a single search does."""
        embedded = []

        async def body():
            async_fresh_graph.define_vectors(nodes=[
                Vector("summary", 2, embed=lambda texts: embedded.extend(texts)
                       or [[1.0, 0.0] for _ in texts]),
                Vector("title", 2),
            ])
            await async_fresh_graph.migrate_vectors()
            await async_fresh_graph.add_nodes([{"id": 1}])
            await async_fresh_graph.set_vectors(
                nodes=[{"id": 1, "summary": [1.0, 0.0], "title": [1.0, 0.0]}])
            return await async_fresh_graph.vector_search_many(
                [[Near("summary", text="raft"), Near("title", [1.0, 0.0])]], k=1)

        assert [hit["id"] for hit in run(body())[0]] == ["1"]
        assert embedded == ["raft"]

    def test_declaring_vectors_changes_no_near_less_query(self, async_graph):
        """The pinned sync invariant, asserted through AsyncGraph's own
        pass-through builder: a traversal with no near= must compile to
        byte-identical SQL whether or not fields are declared."""
        from sqlalchemy.dialects import postgresql

        from hopai import Graph

        def sql(graph):
            return str(graph.build_query(Start(where={"type": "leaf"}), [Hop(hops=(1, 3))])
                       .compile(dialect=postgresql.dialect()))

        plain = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline")
        async_graph.define_vectors(nodes=[Vector("summary", 3)])
        assert sql(async_graph) == sql(plain)


class TestTheSameRefusalsInTheSameOrder:
    """Resolving text a step earlier moves WHERE a refusal is raised
    from, so each one has to keep saying the same thing. The caller
    labels ("Start", "hop 0 (...)", "... via_near") are built in
    core.py for the sync path and re-derived in vectors.py's
    aresolve_spec_texts() for the async one; without these, the two are
    free to drift and an async user gets told to fix a different hop.
    Stronger than asserting a label by pattern: the two messages are
    compared to each other, character for character."""

    def _defined(self, graph):
        graph.define_vectors(nodes=[Vector("summary", 3)], edges=[Vector("rel", 3)])

    @pytest.mark.parametrize("spec", [
        lambda: (Start(near=Near("nope", text="q"), keep=1), []),
        lambda: (Start(), [Hop(near=Near("nope", text="q"), keep=1)]),
        lambda: (Start(), [Hop(via_near=Near("nope", text="q"), via_keep=1)]),
    ])
    def test_an_unknown_field_names_the_same_place_either_way(
            self, async_fresh_graph, async_admin_graph, spec):
        self._defined(async_fresh_graph)
        self._defined(async_admin_graph)
        start, hops = spec()
        with pytest.raises(ValueError) as sync_failure:
            async_admin_graph.traverse(start, *hops)
        with pytest.raises(ValueError) as async_failure:
            run(async_fresh_graph.traverse(start, *hops))
        assert str(async_failure.value) == str(sync_failure.value)
        assert "no vector field 'nope'" in str(async_failure.value)

    def test_a_field_without_an_embedder_names_the_same_place_either_way(
            self, async_fresh_graph, async_admin_graph):
        """The refusal a caller actually hits: the field exists, the
        query is text, and nothing was declared to embed it."""
        self._defined(async_fresh_graph)
        self._defined(async_admin_graph)
        start = Start(near=Near("summary", text="q"), keep=1)
        with pytest.raises(ValueError) as sync_failure:
            async_admin_graph.traverse(start)
        with pytest.raises(ValueError) as async_failure:
            run(async_fresh_graph.traverse(start))
        assert str(async_failure.value) == str(sync_failure.value)
        assert "declares no embedder" in str(async_failure.value)


class TestOutOfScope:
    """Schema/constraint DDL has no async override -- see hopai/asyncio.py's
    module docstring. Each must refuse LOUD, naming the fix, rather than
    silently reaching the async engine's sync facade outside the greenlet
    bridge (which would raise SQLAlchemy's own MissingGreenlet instead)."""

    #: DERIVED from the frozenset, not restated beside it. A hand-written
    #: list is what let `graphs()` fall out of _NEEDS_SYNC_GRAPH with no
    #: test failing: an entry nothing exercises is an entry that can go
    #: missing again, and a list that has to be kept in step by hand is
    #: the thing that was not kept in step. Deriving it means a name
    #: added to the set is tested the moment it is added.
    @pytest.mark.parametrize("name", sorted(asyncio_module._NEEDS_SYNC_GRAPH))
    def test_admin_methods_refuse_with_the_fix_named(self, async_graph, name):
        with pytest.raises(AttributeError, match="plain Graph"):
            getattr(async_graph, name)

    def test_the_set_covers_every_method_that_opens_its_own_connection(self):
        """The other direction, which parametrizing over the set cannot
        see: a Graph method that connects on `self.engine` DIRECTLY --
        rather than through a session the async driver hands it -- has
        to be listed, or it reaches the async engine's sync facade
        outside the greenlet bridge and raises MissingGreenlet.

        `graphs()` is exactly that shape and was missing. This asserts
        the ones known to be, so the next one added to core.py is caught
        by a name, not by a user."""
        for name in ("graphs", "load_vectors", "embed_stale", "schema_violations"):
            assert name in asyncio_module._NEEDS_SYNC_GRAPH, (
                f"Graph.{name}() opens its own connection -- on AsyncGraph that is a "
                f"MissingGreenlet, not an AttributeError naming the fix")

    def test_graphs_refuses_rather_than_raising_missing_greenlet(self, async_graph):
        """The specific hole: `await`-free `async_graph.graphs()` used to
        reach Graph.graphs()' own `self.engine.connect()` on the sync
        facade. A raw MissingGreenlet from inside SQLAlchemy is the
        confusing, out-of-context failure this whole set exists to
        convert into a sentence naming the fix."""
        name = "graphs"                       # not a literal: B009 rewrites that to an
        with pytest.raises(AttributeError) as refused:   # expression ruff then calls useless
            getattr(async_graph, name)
        assert "AsyncGraph has no graphs()" in str(refused.value)
        assert "Graph(<dsn>).graphs(...)" in str(refused.value)

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
