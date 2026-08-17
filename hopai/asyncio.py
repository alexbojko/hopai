"""
hopai.asyncio

AsyncGraph: the same Graph, reached through SQLAlchemy's own sync/async
bridge -- AsyncSession.run_sync() / AsyncConnection.run_sync(), backed
by greenlet plus an async DBAPI -- instead of a hand-written second
implementation. See issue #45 and docs/architecture.md.

    from hopai.asyncio import AsyncGraph

    graph = AsyncGraph("postgresql+psycopg://user:pass@host/db")
    result = await graph.traverse(
        Start(where={"type": "person"}),
        Hop(via={"kind": "friend"}, hops=(1, 4)),
    )
    await graph.mutate({"operations": [
        {"op": "update_nodes", "where": {"type": "draft"}, "set": {"status": "archived"}},
    ]})

WHY THIS SHAPE, not a second implementation: every method below opens
an AsyncSession/AsyncConnection and calls run_sync() into the EXACT
sync function Graph already runs for that operation --
core.py's _traverse_with_session()/_aggregate_with_session(),
Mutator's delete_nodes()/update_nodes()/... (already accept
connection=, added for mutate()'s one-transaction-per-document
guarantee), Ingestor's add_nodes()/merge_nodes()/... (same). Sync and
async can never answer a traversal or a mutation differently, because
there is only one traversal implementation and one mutation
implementation -- this file has none of its own.

run_sync(fn) hands `fn` a plain, ordinary sync Session/Connection,
bridged through a greenlet: the function bodies on the other end of
every call below are unaware anything async is happening, and are the
same functions the sync Graph calls directly.

WHAT MAY BLOCK INSIDE run_sync(), AND WHAT MAY NOT -- the rule the
greenlet design does NOT say out loud, and the one issue #74 was
filed for. That "constant one OS thread" below is the EVENT LOOP'S
thread. SQLAlchemy's database I/O yields back to the loop from inside
the greenlet, which is the entire point of the bridge; an ARBITRARY
blocking call in there has no such arrangement and simply holds the
thread, stalling every other task in the process. So a call is safe
inside run_sync() only if it RELEASES THE GIL while it waits.
Socket I/O does. A CPU-bound C extension does not -- and for that one
even a thread pool leaves the loop starved, so it does not belong on
an async path at all. Anything else that later wants to run inside
the bridge is decided by that question and nothing else.

The embedding provider call was the first thing to fail it: every
read path taking `text=` did a blocking HTTP round trip on the loop's
thread. Hence the shape the read methods below now have -- resolve
every embedding FIRST, awaited, then enter run_sync() with specs that
already carry vectors. That is the same "plan, then open" move
set_vectors() already made for row locks, with the plan step awaited;
hopai/vectors.py's aresolve_*/aplan_* functions are the plan, and a
call with no text= is handed back untouched, so nothing about the
sync path or the emitted SQL changes.

NEEDS AN ASYNC DRIVER. psycopg2 (hopai's base dependency) cannot run
async -- use postgresql+psycopg://... (psycopg3; `pip install
hopai[asyncio]`) or postgresql+asyncpg://.... greenlet itself needs no
separate install: SQLAlchemy 2.0's own wheel pulls it in on every
platform hopai supports.

MEASURED, NOT ASSUMED, before this was written: a throwaway benchmark
compared this design against asyncio.to_thread() (the shape
LangChain's `ainvoke` default takes -- wrap the unmodified sync call
in a thread pool). Both delivered real concurrency. The greenlet
bridge did it on a CONSTANT ONE OS THREAD regardless of how many
traversals were in flight; the thread-pool version held one thread per
in-flight call, climbing with concurrency (27 -> 46 -> 76 across the
sizes tested). Wall-clock did NOT consistently favor the greenlet
path -- the thread pool was sometimes faster, and the gap widened at
higher concurrency, most likely because each AsyncSession.run_sync()
call's own setup (session open/close, greenlet spawn) is paid
serially on the one event-loop thread. So the case for this design is
the resource ceiling a thread-pool wrapper eventually hits in a real
server (a cap to size, memory and scheduling cost per thread), not a
guaranteed speed win -- worth knowing before assuming "async" alone
answers a concurrency question.

WHAT THIS DOES NOT COVER, ON PURPOSE: schema and constraint
declaration -- create_schema(), drop_schema(), define_constraints(),
drop_constraints(), enforce_schema(), save_schema(), load_schema(),
infer_schema(), schema_violations(), add_networkx(). These are
one-time setup/admin calls with no concurrency to gain (issue #45's
own list of what needs async does not name them either), and
AsyncGraph's wrapped Graph runs on the async engine's SYNC FACADE
(AsyncEngine.sync_engine), which is only safe to execute against
INSIDE a greenlet run_sync() spawns. Calling one of these here raises
a clear error naming the fix, rather than either quietly reaching the
sync facade outside that bridge (SQLAlchemy's own MissingGreenlet,
correct but unhelpful out of context) or silently returning something
async-unsafe. Run them through a plain Graph on the same database
instead -- once, at start-up, same as today:

    from hopai import Graph
    Graph("postgresql+psycopg2://user:pass@host/db").create_schema()
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.engine import Engine as _SyncEngine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from .core import Graph, Subgraph
from .hop import Hop, Start
from .ingest import IngestResult
from .models import DEFAULT_GRAPH
from .mutate import MutationResult

#: Setup/admin calls that touch the database directly and have no async
#: override below -- see the module docstring's last section. Kept as an
#: explicit, finite list rather than inferred, so a method added to Graph
#: later fails LOUD (an AttributeError naming the fix) instead of being
#: silently passed through to SQLAlchemy's own MissingGreenlet.
_NEEDS_SYNC_GRAPH = frozenset({
    "create_schema", "drop_schema", "define_constraints", "drop_constraints",
    "enforce_schema", "save_schema", "load_schema", "infer_schema",
    "schema_violations", "add_networkx",
    # load_vectors() (#53) is a database round trip, unlike
    # define_vectors()/vectors/*_ddl(), which is why it is named here
    # rather than left to fall through __getattr__ to the wrapped sync
    # Graph -- that would run it straight against the async engine's
    # sync facade outside the greenlet bridge, same MissingGreenlet trap
    # as every other admin call in this list.
    "load_vectors",
})


class AsyncGraph:
    """Same Graph, same query builders, same filter DSL -- an async
    caller for the operations issue #45 named as needing one:
    traverse, aggregate, ingestion, mutation, and vector search/storage.
    Everything connection-free (build_query(), define_schema(),
    define_vectors(), tool_schemas(), the schema_*/vectors properties,
    *_ddl()) passes straight through to the wrapped sync Graph -- there
    is nothing async about compiling a statement or updating an
    in-memory declaration.
    """

    def __init__(
        self,
        dsn_or_engine,
        graph: str = DEFAULT_GRAPH,
        node_table=None,
        edge_table=None,
        node_id_col: str = "id",
        edge_id_col: str = "id",
        edge_start_col: str = "start_id",
        edge_end_col: str = "end_id",
        graph_col: Optional[str] = "graph_id",
    ):
        if isinstance(dsn_or_engine, _SyncEngine):
            raise TypeError(
                f"AsyncGraph needs an async engine or DSN, got a sync Engine ({dsn_or_engine!r}) "
                f"-- pass the DSN string instead (postgresql+psycopg://... or "
                f"postgresql+asyncpg://...) and AsyncGraph builds the async engine itself, or "
                f"pass an AsyncEngine from sqlalchemy.ext.asyncio.create_async_engine() directly"
            )
        self._async_engine: AsyncEngine = (
            dsn_or_engine if isinstance(dsn_or_engine, AsyncEngine)
            else create_async_engine(dsn_or_engine)
        )
        # .sync_engine is the plain-Engine facade SQLAlchemy's async
        # extension keeps around FOR the greenlet bridge -- every method
        # below reaches it only via AsyncSession/AsyncConnection.run_sync(),
        # never directly, which is the boundary _NEEDS_SYNC_GRAPH exists
        # to protect a caller from crossing by accident.
        self._sync = Graph(self._async_engine.sync_engine, graph=graph, node_table=node_table,
                           edge_table=edge_table, node_id_col=node_id_col, edge_id_col=edge_id_col,
                           edge_start_col=edge_start_col, edge_end_col=edge_end_col,
                           graph_col=graph_col)

    def __repr__(self) -> str:
        return f"AsyncGraph({self._async_engine.url!r}, graph={self._sync.graph!r})"

    @property
    def engine(self) -> AsyncEngine:
        return self._async_engine

    def __getattr__(self, name):
        if name in _NEEDS_SYNC_GRAPH:
            raise AttributeError(
                f"AsyncGraph has no {name}() -- it is a one-time setup/admin call with no "
                f"concurrency to gain, and it would run directly against the async engine's "
                f"sync facade outside the greenlet bridge, which SQLAlchemy refuses "
                f"(MissingGreenlet). Call it on a plain Graph pointed at the same database: "
                f"Graph(<dsn>).{name}(...)"
            )
        return getattr(self._sync, name)

    def in_graph(self, graph: str) -> AsyncGraph:
        """The same tables and async engine, scoped to a different graph
        -- the async counterpart of Graph.in_graph()."""
        s = self._sync
        return AsyncGraph(self._async_engine, graph=graph, node_table=s.nodes_tbl,
                          edge_table=s.edges_tbl, node_id_col=s.node_id_col,
                          edge_id_col=s.edge_id_col, edge_start_col=s.edge_start_col,
                          edge_end_col=s.edge_end_col, graph_col=s.graph_col)

    # -- reading ----------------------------------------------------

    async def _embedded(self, start: Start, hops: list) -> tuple:
        """(start, hops) with every Near(text=) already a vector.

        Awaited HERE, before run_sync() below, because inside the
        bridge the provider call runs on the event loop's own thread --
        see the module docstring. Nothing is copied when there is no
        text= to resolve: the very same Start and Hop objects go on to
        the same sync function, which is what keeps a traversal without
        text on exactly the path it was on before (issue #74)."""
        from .vectors import aresolve_traversal_texts
        return await aresolve_traversal_texts(self._sync, start, hops)

    async def _lazy_vectors_loaded(self, conn) -> None:
        """search()'s own first act, done here instead, because
        resolving text needs the field declaration that call recovers.

        A lazy handle keeps its declarations in the database until
        something needs them (see Graph.in_graph()), and resolving text
        outside run_sync() now needs them EARLIER than search() would
        ask. Doing it here keeps the two in the same order; it is
        idempotent, so search() finding a registry a moment later is
        the ordinary case, and it is paid only when there is text.

        Reachability, stated because it is not obvious: AsyncGraph's own
        in_graph() builds a fresh Graph, which starts non-lazy, so today
        this only ever finds a registry already in place. It is what
        stops the async path being the one place that skips the step --
        without it a lazy handle would be refused with "none are
        defined ... but this call didn't go through [a search]", from
        inside a search."""
        from .vectors import _ensure_lazy_vectors
        await conn.run_sync(lambda c: _ensure_lazy_vectors(self._sync, c))

    async def _rerank_pins(self, start: Start, hops: list, plan: dict) -> dict:
        """Graph._rerank_pins(), with the provider call AWAITED.

        The probe and the hydration go through run_sync() -- they are
        ordinary SQL -- and the scoring does not: inside the bridge it
        would run on the event loop's own thread and stall every other
        task for the length of the round trip, which is issue #74's
        finding applied to the second network call on the read path.
        The session is closed before the await for the reason the sync
        driver closes it: a provider round trip must never happen with a
        transaction open.

        Serial by nature, and that is not a shortcut taken here: hop
        N+1's candidates ARE hop N's survivors, so there is nothing to
        gather.

        Everything except the await is the sync driver's, imported
        rather than restated -- which document each candidate becomes,
        how a node scores across its paths, and how many survive are
        one set of rules, and a second copy is how the two paths drift
        into pinning different nodes for the same query."""
        from .core import _rerank_documents, _rerank_steps, _rerank_survivors

        pins: dict = {}
        for index, spec in _rerank_steps(start, hops):
            async with AsyncSession(self._async_engine) as session:
                candidates = await session.run_sync(
                    lambda s, i=index: self._sync._rerank_probe(s, start, hops, i, pins))
            units = _rerank_documents(candidates, spec.rerank)
            if not units:
                pins[index] = []
                continue
            documents = spec.rerank.build_documents([candidate for _, candidate in units])
            scores = await spec.rerank.ascore(plan[index], documents)
            pins[index] = _rerank_survivors(units, scores, spec.keep)
        return pins

    async def _reranked(self, start: Start, hops: list) -> tuple:
        """(start, hops, pins) -- every provider call on the traversal
        path made, awaited, before anything opens.

        The plan is read from the specs AS WRITTEN, before _embedded()
        resolves them: `Near._with_vector()` hands back a spec carrying
        floats and no text, and the reranker's query is that text."""
        from .core import rerank_plan

        plan = rerank_plan(start, hops)
        start, hops = await self._embedded(start, hops)
        pins = await self._rerank_pins(start, hops, plan) if plan else None
        return start, hops, pins

    async def traverse(self, start: Start, *hops: Hop) -> Subgraph:
        start, hops, pins = await self._reranked(start, list(hops))
        async with AsyncSession(self._async_engine) as session:
            return await session.run_sync(
                lambda s: self._sync._traverse_with_session(s, start, hops, pins=pins))

    async def aggregate(self, start: Start, *hops: Hop, aggregates: dict) -> dict:
        start, hops, pins = await self._reranked(start, list(hops))
        async with AsyncSession(self._async_engine) as session:
            return await session.run_sync(
                lambda s: self._sync._aggregate_with_session(s, start, hops, aggregates,
                                                             pins=pins))

    async def vector_search(self, *near, target: str = "nodes", k: int = 10, where=None,
                            boost=None, rerank=None) -> list:
        from .vectors import (
            arerank_hits, aresolve_near_texts, has_text, rerank_query_text, search_candidates,
        )
        near = list(near)
        # Read (and validate) the reranker's query BEFORE embedding: the
        # resolved Near carries floats and no text.
        query = rerank_query_text(near, rerank, k, "vector_search()")
        async with self._async_engine.connect() as conn:
            if has_text(near):
                await self._lazy_vectors_loaded(conn)
                near = await aresolve_near_texts(self._sync, near, target, k)
            hits = await conn.run_sync(
                lambda c: search_candidates(self._sync, near, target=target, k=k, where=where,
                                            boost=boost, connection=c, rerank=rerank))
        # Outside the connection block AND outside run_sync(): the
        # scoring is a provider round trip, and inside the bridge it
        # would run on the loop's own thread.
        if rerank is None:
            return hits
        return await arerank_hits(hits, rerank, query, k)

    async def vector_search_many(self, queries, target: str = "nodes", k: int = 10, where=None,
                                 boost=None, rerank=None) -> list:
        from .vectors import (
            arerank_many, aresolve_query_texts, has_text, rerank_query_texts,
            search_many_candidates,
        )
        texts = rerank_query_texts(queries, rerank, k, "vector_search_many()")
        async with self._async_engine.connect() as conn:
            if has_text(queries):
                await self._lazy_vectors_loaded(conn)
                queries = await aresolve_query_texts(self._sync, target, queries, k)
            grouped = await conn.run_sync(
                lambda c: search_many_candidates(self._sync, queries, target=target, k=k,
                                                 where=where, boost=boost, connection=c,
                                                 rerank=rerank))
        if rerank is None:
            return grouped
        # CONCURRENTLY: this call exists to turn N round trips into one,
        # and N sequential provider calls would hand that straight back.
        return await arerank_many(grouped, rerank, texts, k)

    async def get_vectors(self, node_ids=None, edge_ids=None, node_fields=None,
                          edge_fields=None) -> dict:
        from .vectors import get_vectors
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                lambda c: get_vectors(self._sync, node_ids, edge_ids, node_fields, edge_fields,
                                      connection=c))

    async def stale_vectors(self, node_fields=None, edge_fields=None, limit=None,
                            after=None) -> dict:
        from .vectors import stale_vectors
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                lambda c: stale_vectors(self._sync, node_fields, edge_fields, limit, after,
                                        connection=c))

    # -- vectors: writing ---------------------------------------------

    async def set_vectors(self, nodes=None, edges=None) -> int:
        """Planned BEFORE the transaction opens, unlike every other write
        here, because a row's value may be TEXT -- and turning text into
        a vector is a network round trip. The sync set_vectors() gets
        that ordering for free by planning and then opening; this one
        cannot, since the transaction is already open by the time
        run_sync() reaches into vectors.py. Planning here keeps the
        provider call off the row locks, which is the invariant, not a
        preference.

        AWAITED, not merely hoisted: out of the transaction is not the
        same as off the event loop. Called plainly in this `async def`
        body -- which is what it was -- the round trip held the loop's
        thread for its whole duration and stalled every other task in
        the process (issue #74)."""
        from .vectors import aplan_vector_writes, set_vectors
        plan = await aplan_vector_writes(self._sync, nodes, edges)
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: set_vectors(self._sync, connection=c, plan=plan))

    def define_vectors(self, nodes=None, edges=None, migrate: bool = False):
        """The declaration half of Graph.define_vectors() -- connection-
        free, so (like build_query()/define_schema()) it passes straight
        through to the wrapped sync Graph with no run_sync() needed.

        `migrate=True` is refused here rather than silently attempted:
        it would run migrate_vectors() directly against the async
        engine's sync facade, outside the greenlet bridge, which
        SQLAlchemy refuses with a raw MissingGreenlet instead of a named
        fix -- the same failure _NEEDS_SYNC_GRAPH exists to turn into an
        AttributeError for the other setup calls. Declare here, then
        `await graph.migrate_vectors()` for the DDL:

            graph.define_vectors(nodes=[Vector("summary", 1536)])
            await graph.migrate_vectors()
        """
        if migrate:
            raise AttributeError(
                "AsyncGraph.define_vectors(migrate=True) is not supported -- it would run "
                "migrate_vectors() directly against the async engine's sync facade outside "
                "the greenlet bridge, which SQLAlchemy refuses (MissingGreenlet). Call "
                "define_vectors(...) here with no migrate=, then `await "
                "graph.migrate_vectors()` separately -- or run both together on a plain "
                "Graph pointed at the same database: Graph(<dsn>).define_vectors(..., "
                "migrate=True)"
            )
        return self._sync.define_vectors(nodes, edges)

    async def migrate_vectors(self) -> list:
        from .vectors import migrate_vectors
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(lambda c: migrate_vectors(self._sync, connection=c))

    async def drop_vectors(self, node_fields=None, edge_fields=None) -> list:
        from .vectors import drop_vectors
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: drop_vectors(self._sync, node_fields, edge_fields, connection=c))

    # -- writing ------------------------------------------------------

    async def add_nodes(self, rows: list) -> int:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._ingestor.add_nodes(rows, connection=c))

    async def add_edges(self, rows: list) -> int:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._ingestor.add_edges(rows, connection=c))

    async def merge_nodes(self, rows: list, on: list, replace: bool = False) -> int:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._ingestor.merge_nodes(rows, on=on, replace=replace,
                                                            connection=c))

    async def merge_edges(self, rows: list, on: list, replace: bool = False) -> int:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._ingestor.merge_edges(rows, on=on, replace=replace,
                                                            connection=c))

    async def ingest(self, document: dict, merge_nodes_on: Optional[list] = None,
                     merge_edges_on: Optional[list] = None) -> IngestResult:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._ingestor.ingest(document, merge_nodes_on, merge_edges_on,
                                                       connection=c))

    async def write_cypher(self, query: str, **options) -> IngestResult:
        from .cypher import cypher_to_operations, resolve_strict
        operations = cypher_to_operations(query, **resolve_strict(self._sync, dict(options)))
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._ingestor.execute_operations(operations, connection=c))

    # -- deleting and updating -----------------------------------------

    async def delete_nodes(self, where=None, detach: bool = False, all: bool = False) -> MutationResult:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._mutator.delete_nodes(where, detach=detach, all=all,
                                                            connection=c))

    async def delete_edges(self, where=None, start=None, end=None, all: bool = False) -> MutationResult:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._mutator.delete_edges(where, start=start, end=end, all=all,
                                                            connection=c))

    async def update_nodes(self, where=None, set=None, remove=None, replace: bool = False,
                           all: bool = False) -> MutationResult:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._mutator.update_nodes(where, set=set, remove=remove,
                                                            replace=replace, all=all, connection=c))

    async def update_edges(self, where=None, start=None, end=None, set=None, remove=None,
                           replace: bool = False, all: bool = False) -> MutationResult:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._mutator.update_edges(
                    where, start=start, end=end, set=set, remove=remove, replace=replace,
                    all=all, connection=c))

    async def clear(self) -> MutationResult:
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(lambda c: self._sync._mutator.clear(connection=c))

    async def mutate(self, document: dict) -> MutationResult:
        from .mutate import spec_to_mutations
        operations = spec_to_mutations(document)
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._mutator.execute_operations(operations, connection=c))

    async def mutate_cypher(self, query: str, **options) -> MutationResult:
        from .cypher import cypher_to_mutations, resolve_strict
        operations = cypher_to_mutations(query, **resolve_strict(self._sync, dict(options)))
        async with self._async_engine.begin() as conn:
            return await conn.run_sync(
                lambda c: self._sync._mutator.execute_operations(operations, connection=c))

    async def cypher(self, query: str, **options):
        """Run any supported Cypher -- the async counterpart of
        Graph.cypher(). Parsing (which branch this is) is pure, so only
        the dispatch is reimplemented here; each branch calls this
        class's own async method rather than the sync free functions
        hopai.cypher.traverse_cypher()/aggregate_cypher() use, since
        those call `graph.traverse(...)` unawaited."""
        from .cypher import (
            _MutateClause, _Parser, _ReturnClause, _tokenize, _WriteClause,
            cypher_to_aggregation, cypher_to_traversal, resolve_strict,
        )
        clauses = _Parser(_tokenize(query)).parse()
        if any(isinstance(c, _MutateClause) for c in clauses):
            return await self.mutate_cypher(query, **options)
        if any(isinstance(c, _WriteClause) for c in clauses):
            return await self.write_cypher(query, **options)
        if any(isinstance(c, _ReturnClause) for c in clauses):
            start, hops, aggregates = cypher_to_aggregation(
                query, **resolve_strict(self._sync, dict(options)))
            return await self.aggregate(start, *hops, aggregates=aggregates)
        start, hops = cypher_to_traversal(query, **resolve_strict(self._sync, dict(options)))
        return await self.traverse(start, *hops)

    async def dispose(self) -> None:
        """Close the async engine's connection pool. Call this once,
        at shutdown -- not per-call, the way traverse()/mutate()/... open
        and close their own session or connection already."""
        await self._async_engine.dispose()
