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

    async def traverse(self, start: Start, *hops: Hop) -> Subgraph:
        async with AsyncSession(self._async_engine) as session:
            return await session.run_sync(
                lambda s: self._sync._traverse_with_session(s, start, list(hops)))

    async def aggregate(self, start: Start, *hops: Hop, aggregates: dict) -> dict:
        async with AsyncSession(self._async_engine) as session:
            return await session.run_sync(
                lambda s: self._sync._aggregate_with_session(s, start, list(hops), aggregates))

    async def vector_search(self, *near, target: str = "nodes", k: int = 10, where=None,
                            boost=None) -> list:
        from .vectors import search
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                lambda c: search(self._sync, list(near), target=target, k=k, where=where,
                                 boost=boost, connection=c))

    async def vector_search_many(self, queries, target: str = "nodes", k: int = 10, where=None,
                                 boost=None) -> list:
        from .vectors import search_many
        async with self._async_engine.connect() as conn:
            return await conn.run_sync(
                lambda c: search_many(self._sync, queries, target=target, k=k, where=where,
                                      boost=boost, connection=c))

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
        preference."""
        from .vectors import plan_vector_writes, set_vectors
        plan = plan_vector_writes(self._sync, nodes, edges)
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
