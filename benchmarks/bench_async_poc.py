"""
benchmarks/bench_async_poc.py

Proof of concept for issue #45 ("Async support has to arrive
library-wide"). The proposed design routes hopai's EXISTING sync
Graph.traverse() through SQLAlchemy's own sync/async bridge --
AsyncSession.run_sync(), backed by greenlet plus an async driver --
instead of writing a second async implementation. Before committing to
that shape: is it real concurrency, or a repaint of the "wrap the sync
call in a thread pool" trick libraries like LangChain default to
(ainvoke -> run_in_executor)? That trick LOOKS async from the caller's
side but is bounded by a thread-pool's worker count and pays GIL
contention on anything CPU-bound -- the trap this benchmark exists to
either confirm hopai avoids, or catch before the real refactor lands.

ONE function, _traverse_with_session(), stands in for the
core.py._traverse_with_session() split the real change would make --
copied here rather than added to hopai/core.py, since this script is
the feasibility check that decides whether that split is worth making.
It is executed three ways, same N calls, same simulated per-call
latency (SELECT pg_sleep(LATENCY) standing in for a network round trip
-- localhost is too fast to show contention on its own):

  1. sync sequential   -- N calls, one at a time, the baseline.
  2. thread-pool async -- N calls via asyncio.to_thread(), the
                           LangChain-style ainvoke default.
  3. greenlet async    -- N calls via AsyncSession.run_sync(), the
                           design this benchmark is checking.

Each run also counts the distinct OS threads that did the work --
wall-clock alone cannot tell "genuinely concurrent" from "N threads
each blocked for the same amount of time" -- and an optional
--cpu-ms flag adds a pure-Python busy-loop after each simulated query,
standing in for row hydration, to show what GIL contention between
worker threads costs the thread-pool approach that the greenlet
approach, running on one thread throughout, does not pay.

Usage:
    python bench_async_poc.py \\
        --dsn "postgresql+psycopg://user:pass@host/db" --n 30 --latency 0.05
    python bench_async_poc.py --dsn "..." --n 30 --latency 0.05 --cpu-ms 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session

from hopai import Graph, Hop, Start

GRAPH = "bench_async_poc"


# ---------------------------------------------------------------------
# Data setup -- a small graph is enough; this benchmark is about
# concurrency, not query cost (see bench_hopai.py for that).
# ---------------------------------------------------------------------

def setup_data(engine, n_nodes: int = 200) -> Graph:
    graph = Graph(engine, graph=GRAPH)
    graph.create_schema()
    graph.clear()
    graph.add_nodes([{"id": i, "type": "person", "name": f"n{i}"} for i in range(n_nodes)])
    graph.add_edges([
        {"start_id": i, "end_id": i + 1, "kind": "knows"} for i in range(n_nodes - 1)
    ])
    return graph


def _cpu_work(iters: int) -> int:
    """Pure-Python busy-loop standing in for row hydration -- CPU work
    that competes for the GIL when run concurrently across threads."""
    total = 0
    for i in range(iters):
        total += i * i
    return total


def _calibrate_cpu_iters(target_ms: float) -> int:
    if target_ms <= 0:
        return 0
    iters = 200_000
    t0 = time.perf_counter()
    _cpu_work(iters)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return max(1, int(iters * target_ms / max(elapsed_ms, 0.001)))


# ---------------------------------------------------------------------
# THE function under test -- what core.py's _traverse_with_session()
# split would be. One implementation, called from all three run modes.
# ---------------------------------------------------------------------

def _traverse_with_session(session, graph: Graph, start: Start, hops: list,
                           latency: float, cpu_iters: int, thread_log: list) -> int:
    thread_log.append(threading.get_ident())
    if latency > 0:
        session.execute(text("SELECT pg_sleep(:s)"), {"s": latency})
    query = graph.build_query(start, hops)
    rows = session.execute(query).all()
    if cpu_iters:
        _cpu_work(cpu_iters)
    return len(rows)


# ---------------------------------------------------------------------
# Three ways to run N of those calls "at once"
# ---------------------------------------------------------------------

def run_sync_sequential(engine, graph, start, hops, n, latency, cpu_iters):
    thread_log = []
    t0 = time.perf_counter()
    for _ in range(n):
        with Session(engine) as session:
            _traverse_with_session(session, graph, start, hops, latency, cpu_iters, thread_log)
    return time.perf_counter() - t0, thread_log


def run_thread_pool(engine, graph, start, hops, n, latency, cpu_iters, max_workers):
    """The LangChain ainvoke default: asyncio.to_thread() wraps the
    unmodified sync call. Every in-flight call holds an OS thread for
    its whole duration, capped at max_workers."""
    thread_log = []
    executor = ThreadPoolExecutor(max_workers=max_workers)

    def one_call():
        with Session(engine) as session:
            _traverse_with_session(session, graph, start, hops, latency, cpu_iters, thread_log)

    async def run_all():
        loop = asyncio.get_running_loop()
        await asyncio.gather(*(loop.run_in_executor(executor, one_call) for _ in range(n)))

    t0 = time.perf_counter()
    asyncio.run(run_all())
    elapsed = time.perf_counter() - t0
    executor.shutdown()
    return elapsed, thread_log


def run_greenlet_async(async_engine, graph, start, hops, n, latency, cpu_iters):
    """The design under test: AsyncSession.run_sync() bridges into the
    SAME _traverse_with_session() via greenlet -- no worker thread
    held for the wait, concurrency limited by the connection pool, not
    a thread count."""
    thread_log = []

    async def one_call():
        async with AsyncSession(async_engine) as session:
            return await session.run_sync(
                lambda sync_session: _traverse_with_session(
                    sync_session, graph, start, hops, latency, cpu_iters, thread_log))

    async def run_all():
        await asyncio.gather(*(one_call() for _ in range(n)))

    t0 = time.perf_counter()
    asyncio.run(run_all())
    elapsed = time.perf_counter() - t0
    return elapsed, thread_log


# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default="postgresql+psycopg://postgres:testpass@localhost/hopai_bench",
                        help="a psycopg3 DSN -- one driver serves both the sync and async engine")
    parser.add_argument("--n", type=int, default=30, help="concurrent traversals per run")
    parser.add_argument("--latency", type=float, default=0.05,
                        help="simulated per-call DB round trip, seconds (SELECT pg_sleep)")
    parser.add_argument("--cpu-ms", type=float, default=0.0,
                        help="simulated per-call CPU work (row hydration stand-in), milliseconds")
    parser.add_argument("--out", default="bench_async_poc_results.json")
    args = parser.parse_args()

    engine = create_engine(args.dsn, pool_size=args.n, max_overflow=0)
    async_engine = create_async_engine(args.dsn, pool_size=args.n, max_overflow=0)

    graph = setup_data(engine)
    start = Start(where={"type": "person"})
    hops = [Hop(via={"kind": "knows"}, hops=(1, 3))]
    cpu_iters = _calibrate_cpu_iters(args.cpu_ms)

    print(f"n={args.n} latency={args.latency}s cpu_ms={args.cpu_ms} "
         f"(calibrated to {cpu_iters} loop iterations)\n")

    results = {}

    elapsed, threads = run_sync_sequential(engine, graph, start, hops, args.n, args.latency, cpu_iters)
    results["sync_sequential"] = {"elapsed_s": elapsed, "distinct_threads": len(set(threads))}
    print(f"1. sync sequential   : {elapsed:6.3f}s   threads used: {len(set(threads))}")
    engine.dispose()

    elapsed, threads = run_thread_pool(engine, graph, start, hops, args.n, args.latency,
                                       cpu_iters, max_workers=args.n)
    results["thread_pool_async"] = {"elapsed_s": elapsed, "distinct_threads": len(set(threads))}
    print(f"2. thread-pool async : {elapsed:6.3f}s   threads used: {len(set(threads))}")
    engine.dispose()

    elapsed, threads = run_greenlet_async(async_engine, graph, start, hops, args.n, args.latency, cpu_iters)
    results["greenlet_async"] = {"elapsed_s": elapsed, "distinct_threads": len(set(threads))}
    print(f"3. greenlet async    : {elapsed:6.3f}s   threads used: {len(set(threads))}")

    print(f"\nsequential baseline for reference: {args.n} x {args.latency}s = "
         f"{args.n * args.latency:.3f}s of pure DB-wait time")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    engine.dispose()
    asyncio.run(async_engine.dispose())


if __name__ == "__main__":
    main()
