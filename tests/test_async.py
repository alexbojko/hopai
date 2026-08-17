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
import threading
import time

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine

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


class _SlowClient:
    """An embedding client that takes a visible amount of time, and
    records what the rest of the loop got done while it was inside the
    call.

    Sync by default -- a `requests`/`httpx` client blocked on a socket,
    which is what nearly every hopai user passes. `progress` is one
    entry per provider call: the loop turns that happened during it.
    Zero is what every path here scored before issue #74, because the
    call ran on the event loop's own thread."""

    PAUSE = 0.2

    def __init__(self, ticks: list, dimensions: int = 3):
        self.ticks = ticks
        self.dimensions = dimensions
        self.progress: list = []
        self.threads: list = []
        self.texts: list = []

    def _answer(self, texts) -> list:
        self.texts.extend(texts)
        self.threads.append(threading.get_ident())
        return [[1.0] + [0.0] * (self.dimensions - 1) for _ in texts]

    def __call__(self, texts):
        before = self.ticks[0]
        time.sleep(self.PAUSE)                     # the provider round trip
        self.progress.append(self.ticks[0] - before)
        return self._answer(texts)


class _SlowAsyncClient(_SlowClient):
    """The same, as a plain `async def` -- what an async provider client
    (openai.AsyncOpenAI and friends) looks like to _abind()."""

    async def __call__(self, texts):
        before = self.ticks[0]
        await asyncio.sleep(self.PAUSE)
        self.progress.append(self.ticks[0] - before)
        return self._answer(texts)


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


class TestTheEventLoopKeepsRunning:
    """Issue #74: AsyncGraph was async for the database and synchronous
    for the network. Every read path taking text= embedded it INSIDE
    SQLAlchemy's greenlet bridge -- which runs on the event loop's own
    thread -- so a 200ms provider call stopped every other task in the
    process for 200ms. set_vectors() did it without the bridge at all,
    straight in its `async def` body.

    Each test below runs the operation next to an independent loop task
    and asserts that task got somewhere DURING the provider call. On
    the code these were written against, every one of them scores
    exactly zero: that is the measurement, not an assumption."""

    #: Enough to make the point without making the suite slow: the
    #: ticker runs every 1ms, so a 200ms call that frees the loop is
    #: worth ~200 turns and one that holds it is worth 0. Asserting
    #: "> 0" rather than a count keeps this about the loop being free,
    #: not about scheduler timing.
    DIMS = 3

    async def _seeded(self, graph, client, edge_client=None):
        """A graph with one vector field per target, a node and an edge
        already embedded -- everything the reads below need, with the
        provider calls it costs already spent (so only the read's own
        embed is left to measure)."""
        graph.define_vectors(nodes=[Vector("summary", self.DIMS, embed=client)],
                             edges=[Vector("rel", self.DIMS,
                                           embed=edge_client or client)])
        await graph.migrate_vectors()
        await graph.add_nodes([{"id": 1, "type": "doc"}, {"id": 2, "type": "doc"}])
        await graph.add_edges([{"id": 9, "start_id": 1, "end_id": 2, "kind": "cites"}])
        await graph.set_vectors(nodes=[{"id": 1, "summary": [1.0, 0.0, 0.0]},
                                       {"id": 2, "summary": [1.0, 0.0, 0.0]}],
                                edges=[{"id": 9, "rel": [1.0, 0.0, 0.0]}])
        client.progress.clear()
        client.texts.clear()

    def test_a_traversal_does_not_hold_the_loop_while_it_embeds(self, async_fresh_graph):
        """Start(near=Near(text=...)) -- the entry point issue #74 leads
        with. Before the fix the embed ran inside session.run_sync(),
        which is the loop's thread, and the ticker below never moved."""
        ticks = [0]
        client = _SlowClient(ticks)

        async def body():
            await self._seeded(async_fresh_graph, client)
            return await _while_ticking(ticks, async_fresh_graph.traverse(
                Start(near=Near("summary", text="raft consensus"), keep=1)))

        result = run(body())
        assert [node["id"] for node in result.nodes] == ["1"]
        assert client.texts == ["raft consensus"]
        assert client.progress and all(p > 0 for p in client.progress), \
            "the loop made no progress during the embed -- it is still on its thread"

    def test_a_hop_near_and_via_near_are_resolved_off_the_loop_too(self, async_fresh_graph):
        """The other two call sites: a hop's node ranking and its edge
        beam. They reach validate_nears() from different places in
        _walk_matches(), so covering only Start would leave two thirds
        of the traversal path still blocking."""
        ticks = [0]
        client = _SlowClient(ticks)

        async def body():
            await self._seeded(async_fresh_graph, client)
            return await _while_ticking(ticks, async_fresh_graph.traverse(
                Start(where={"type": "doc"}),
                Hop(via_near=Near("rel", text="cites"), via_keep=1,
                    near=Near("summary", text="raft consensus"), keep=1)))

        result = run(body())
        # Both endpoints of the one edge the beam followed.
        assert sorted(node["id"] for node in result.nodes) == ["1", "2"]
        assert [edge["id"] for edge in result.edges] == ["9"]
        # One provider call per site, and the loop ran during both.
        assert client.texts == ["cites", "raft consensus"]
        assert len(client.progress) == 2 and all(p > 0 for p in client.progress)

    def test_aggregate_does_not_hold_the_loop_while_it_embeds(self, async_fresh_graph):
        """aggregate() shares _walk_matches() with traverse() but has
        its own run_sync() call, so it needs its own resolution."""
        ticks = [0]
        client = _SlowClient(ticks)

        async def body():
            await self._seeded(async_fresh_graph, client)
            return await _while_ticking(ticks, async_fresh_graph.aggregate(
                Start(near=Near("summary", text="raft consensus"), keep=1),
                aggregates={"n": Count()}))

        assert run(body()) == {"n": 1}
        assert client.progress and all(p > 0 for p in client.progress)

    def test_vector_search_does_not_hold_the_loop_while_it_embeds(self, async_fresh_graph):
        ticks = [0]
        client = _SlowClient(ticks)

        async def body():
            await self._seeded(async_fresh_graph, client)
            return await _while_ticking(ticks, async_fresh_graph.vector_search(
                Near("summary", text="raft consensus"), k=1))

        hits = run(body())
        assert [hit["id"] for hit in hits] == ["1"]
        assert client.progress and all(p > 0 for p in client.progress)

    def test_vector_search_many_embeds_the_whole_batch_off_the_loop(self, async_fresh_graph):
        """The batch path resolves every query's text in ONE provider
        call per field -- the round trip vector_search_many() exists to
        save. Awaiting it must not turn that back into N calls."""
        ticks = [0]
        client = _SlowClient(ticks)

        async def body():
            await self._seeded(async_fresh_graph, client)
            return await _while_ticking(ticks, async_fresh_graph.vector_search_many(
                [Near("summary", text="raft"), Near("summary", text="paxos")], k=1))

        results = run(body())
        assert len(results) == 2
        assert client.texts == ["raft", "paxos"]
        assert len(client.progress) == 1 and client.progress[0] > 0

    def test_set_vectors_does_not_hold_the_loop_while_it_embeds(self, async_fresh_graph):
        """Out of the transaction was never the same as off the loop:
        plan_vector_writes() was already hoisted before begin() (which
        must stay -- it is what keeps the round trip off the row locks)
        and still ran on the loop's thread."""
        ticks = [0]
        client = _SlowClient(ticks)

        async def body():
            await self._seeded(async_fresh_graph, client)
            await _while_ticking(ticks, async_fresh_graph.set_vectors(
                nodes=[{"id": 1, "summary": "a paper about Raft"}]))
            return await async_fresh_graph.get_vectors(node_ids=[1])

        stored = run(body())
        assert stored["nodes"]["1"]["summary"] == pytest.approx([1.0, 0.0, 0.0])
        assert client.texts == ["a paper about Raft"]
        assert client.progress and all(p > 0 for p in client.progress)

    def test_a_sync_client_is_run_in_a_worker_thread(self, async_fresh_graph):
        """The compatibility path, and the reason the tests above pass
        for a client that cannot be awaited at all: asyncio.to_thread().
        Legitimate here because a provider SDK blocked on a socket has
        released the GIL -- the loop really does keep running -- and
        the thread is the visible cost that makes an async client the
        better answer."""
        ticks = [0]
        client = _SlowClient(ticks)

        async def body():
            await self._seeded(async_fresh_graph, client)
            await _while_ticking(ticks, async_fresh_graph.vector_search(
                Near("summary", text="raft consensus"), k=1))
            return threading.get_ident()

        loop_thread = run(body())
        assert client.threads and all(t != loop_thread for t in client.threads)

    # -- the SECOND network call on the read path: reranking -----------

    def _rerank(self, ticks, scores=None, per_call=None):
        """A Rerank whose provider call takes _SlowClient.PAUSE and
        records what the loop got done while it was inside it. Async by
        construction, since that is the shape the module is built around
        -- a sync one falls back to a worker thread and is covered above
        for embeddings."""
        from hopai import Rerank

        progress, spans = [], []

        async def client(query, documents):
            before = ticks[0]
            started = time.perf_counter()
            await asyncio.sleep(_SlowClient.PAUSE)
            progress.append(ticks[0] - before)
            spans.append((started, time.perf_counter()))
            if per_call is not None:
                per_call.append(query)
            return [(scores or {}).get(d, 0.0) for d in documents]

        # `.id` rather than a property: _seeded()'s nodes carry only a
        # type, and what these tests are about is the loop, not the
        # document.
        rerank = Rerank(client, document_from=".id", candidates=5)
        rerank.progress = progress
        rerank.spans = spans
        return rerank

    def test_vector_search_does_not_hold_the_loop_while_it_reranks(self, async_fresh_graph):
        """Reranking is a provider call on the READ path, so it lands in
        exactly the trap issue #74 documented: run inside run_sync() it
        would sit on the event loop's own thread for the whole round
        trip. The scoring is awaited OUTSIDE the bridge, and this is the
        measurement -- on a version that awaited it inside, the ticker
        below scores zero."""
        pytest.importorskip("jq")
        ticks = [0]
        client = _SlowClient(ticks)
        rerank = self._rerank(ticks, scores={"2": 1.0})

        async def body():
            await self._seeded(async_fresh_graph, client)
            return await _while_ticking(ticks, async_fresh_graph.vector_search(
                Near("summary", text="raft consensus"), k=1, rerank=rerank))

        hits = run(body())
        assert [hit["id"] for hit in hits] == ["2"]
        assert hits[0]["rerank_score"] == 1.0
        assert rerank.progress and all(p > 0 for p in rerank.progress), \
            "the loop made no progress during the rerank -- it is still on its thread"

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
        client = _SlowClient(ticks)
        queries = []
        rerank = self._rerank(ticks, per_call=queries)

        async def body():
            await self._seeded(async_fresh_graph, client)
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
        client = _SlowClient(ticks)
        rerank = self._rerank(ticks)

        async def body():
            await self._seeded(async_fresh_graph, client)
            return await _while_ticking(ticks, async_fresh_graph.traverse(
                Start(where={"type": "doc"}),
                Hop(via={"kind": "cites"}, near=Near("summary", text="raft"), keep=1,
                    rerank=rerank)))

        result = run(body())
        assert sorted(node["id"] for node in result.nodes) == ["1", "2"]
        assert rerank.progress and all(p > 0 for p in rerank.progress)

    def test_a_step_that_finds_nothing_spends_no_provider_call(self, async_fresh_graph):
        """AsyncGraph has its own rerank driver, so the sync one's "no
        candidates, no call" shortcut has to exist here too -- awaiting
        `ascore` on an empty list would be a billed round trip for an
        answer nobody can use, and it would hold the loop for the length
        of it."""
        pytest.importorskip("jq")
        ticks = [0]
        client = _SlowClient(ticks)
        queries = []
        rerank = self._rerank(ticks, per_call=queries)

        async def body():
            await self._seeded(async_fresh_graph, client)
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
        client = _SlowClient(ticks)
        queries = []
        rerank = self._rerank(ticks, per_call=queries)

        async def body():
            await self._seeded(async_fresh_graph, client)
            return await async_fresh_graph.vector_search(
                Near("summary", text="raft"), k=1, where={"type": "nothing-matches-this"},
                rerank=rerank)

        assert run(body()) == []
        assert queries == []

    def test_an_async_client_is_awaited_on_the_loops_own_thread(self, async_fresh_graph):
        """The primary path. An `async def` client is awaited directly,
        so it costs no thread at all -- which is the resource ceiling
        this module's docstring chose the greenlet bridge to avoid.

        It also fails LOUDLY before the fix rather than slowly: the old
        code called the client inside run_sync() and got a coroutine
        back, which EmbeddingError reported as an unusable answer."""
        ticks = [0]
        client = _SlowAsyncClient(ticks)

        async def body():
            await self._seeded(async_fresh_graph, client)
            hits = await _while_ticking(ticks, async_fresh_graph.vector_search(
                Near("summary", text="raft consensus"), k=1))
            return hits, threading.get_ident()

        hits, loop_thread = run(body())
        assert [hit["id"] for hit in hits] == ["1"]
        assert client.progress and all(p > 0 for p in client.progress)
        assert client.threads == [loop_thread]


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

    def test_a_batch_with_no_text_is_handed_back_as_it_came(self, async_graph):
        """The resolver's own fast path, asserted on identity: called
        with nothing to embed it must return the caller's list, not a
        rebuilt copy of it -- which is what makes "no text, no new
        path" true for the batch search as well."""
        from hopai.vectors import aresolve_query_texts

        async_graph.define_vectors(nodes=[Vector("summary", 2)])
        queries = [Near("summary", [1.0, 0.0])]
        assert run(aresolve_query_texts(async_graph._sync, "nodes", queries, 1)) is queries

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
    _near_sites() for the async one; without these, the two are free to
    drift and an async user gets told to fix a different hop."""

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

    def test_a_lazy_handle_still_recovers_its_fields_before_resolving(
            self, async_fresh_graph, async_admin_graph):
        """Resolving text outside run_sync() needs the field
        declaration EARLIER than search() would have asked for it, so
        the async path loads a lazy handle's registry first. Without
        that step a search on one is refused by name for the wrong
        reason -- "none are defined ... this call didn't go through [a
        search]", from inside a search.

        The lazy state is built by hand because AsyncGraph.in_graph()
        does not produce one today (it builds a fresh Graph, which
        starts non-lazy) -- Graph.in_graph() does, and this is the
        check search() itself makes on every call."""
        async def body():
            async_admin_graph.define_vectors(nodes=[Vector("summary", 3)])
            async_admin_graph.migrate_vectors()
            # Forget the declaration the way a fresh in_graph() handle
            # has never had it.
            async_fresh_graph._sync._vectors = None
            async_fresh_graph._sync._vectors_lazy = True
            await async_fresh_graph.vector_search(Near("summary", text="q"), k=1)

        with pytest.raises(ValueError) as failure:
            run(body())
        # Recovered from the database, so the refusal is the real one:
        # load_vectors() cannot restore embed=, which is a policy, not a
        # shape. "needs vector fields" here would mean the field was
        # never found at all.
        assert "declares no embedder" in str(failure.value)

    def test_a_meaningless_k_is_refused_before_anything_is_embedded(
            self, async_fresh_graph):
        """_check_k runs before the provider on the sync path too. A
        call that cannot succeed must not cost a round trip first."""
        calls = []
        async_fresh_graph.define_vectors(nodes=[
            Vector("summary", 2, embed=lambda texts: calls.append(texts) or [[1.0, 0.0]])])
        with pytest.raises(ValueError, match="k must be a positive integer"):
            run(async_fresh_graph.vector_search(Near("summary", text="a"), k=0))
        with pytest.raises(ValueError, match="k must be a positive integer"):
            run(async_fresh_graph.vector_search_many([Near("summary", text="a")], k=0))
        assert calls == []

    def test_a_batch_about_to_be_refused_is_never_embedded(self, async_fresh_graph):
        """Shape first, provider second -- validate_nears()' own order,
        which the async path has to keep from one step further out.
        Two Nears on one field is refused; embedding them first would
        spend a provider call to reach the same error."""
        calls = []
        async_fresh_graph.define_vectors(nodes=[
            Vector("summary", 2, embed=lambda texts: calls.append(texts) or [[1.0, 0.0]])])
        with pytest.raises(ValueError, match="two Near specs"):
            run(async_fresh_graph.vector_search(
                Near("summary", text="a"), Near("summary", text="b"), k=1))
        assert calls == []


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
