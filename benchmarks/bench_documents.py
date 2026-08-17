"""
benchmarks/bench_documents.py

Measures how much of the event loop's thread `Rerank.build_documents()`
holds, against ROW SIZE -- the axis nothing in this directory measured,
which is why a 121ms block lived in the reranking path with a green
suite in front of it.

The cost that matters here is not the filter's. `document_from` is
evaluated by libjq through `program.input_value(candidate)`, and
`input_value` marshals the WHOLE candidate into jq's own value
representation before the filter reads a single field of it -- so the
price was proportional to the candidate, not to the projection.
`.properties.title` (60 characters out) cost exactly what
`.properties.field_0` (1KB out) cost on the same 100KB row, and 50 such
rows held the thread for the length of a page load.

hopai now prunes the candidate to the paths `jqsafe.paths_read()` says
the filter reads, so `us_per_row` should be flat across `--sizes` and
`us_per_kb` should be ~0. A run where `us_per_kb` climbs back toward
25 is the regression this file exists to catch.

NO DATABASE AND NO NETWORK: document building happens after the SQL
returns and before the provider call, so it can be measured with
neither. That also makes this the one benchmark here that runs
anywhere.

Usage:
    python bench_documents.py [--candidates 50] [--sizes 1,10,100,500]
        [--filter .properties.title] [--out bench_document_results.json]

Numbers recorded during development (jq 1.12, one core, 50 candidates,
`.properties.title`), unpruned against pruned:

    | KB per row | before | after |
    | ---------: | -----: | ----: |
    |          1 |   40us |  10us |
    |         10 |  327us |  10us |
    |        100 | 2879us |  13us |
    |        500 |15685us |  21us |

    us_per_kb:                 31.4  ->  0.02
    title at 500KB/row:      14592us ->  23us
    the 1KB field, same rows:13147us ->  46us   (the control: the two
      projections cost the same before AND after, for opposite reasons)

The loop measurement in the same run is the one to read second: a
coroutine ticking at 1ms runs beside the build, and `longest_gap_ms` is
how long in one stretch it did not get the thread.

    | KB per row | before: build / gap | after: build / gap |
    | ---------: | ------------------: | -----------------: |
    |          1 |     2.2ms / 13.3ms  |     0.7ms / 7.9ms  |
    |         10 |    15.5ms / 61.5ms  |     0.7ms / 6.8ms  |
    |        100 |   146.1ms / 443.8ms |     0.7ms / 7.7ms  |
    |        500 |   719.1ms / 719.1ms |     1.4ms / 9.8ms  |

At 500KB the gap EQUALS the build: the ticker did not run once for the
whole call, at 0 ticks/s against an idle 970. That is total starvation
-- the signature of a `time.sleep`, not of a slow function -- and it is
what `ticks > 0` in tests/test_async.py could not tell from a 121ms
stall, which is why it went unnoticed. After pruning the gap sits at
the ticker's own granularity (the idle baseline's own longest gap
measured 2.8ms) and no longer moves with the row.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from hopai.rerankers import Rerank


def rows(count: int, kilobytes: int) -> list:
    """`count` candidates in the shape the read path produces, each
    carrying `kilobytes` of payload in `properties` beside a short
    title -- a node that stores text, which is the case the cost model
    was wrong about."""
    body = "x" * 1000
    return [{"id": str(i), "similarity": 0.5, "similarities": {"summary": 0.5},
             "boosts": {},
             "properties": dict({"title": f"node {i}", "summary": "a summary"},
                                **{f"field_{f}": body for f in range(kilobytes)})}
            for i in range(count)]


def timed(rerank: Rerank, candidates: list, repeats: int = 5) -> float:
    """Seconds per row, best of `repeats`. Best rather than mean: this
    is a CPU-bound measurement on a shared box, so the fastest run is
    the one least contaminated by everything else on it."""
    rerank.build_documents(candidates)                  # warm
    best = min(_once(rerank, candidates) for _ in range(repeats))
    return best / len(candidates)


def _once(rerank: Rerank, candidates: list) -> float:
    started = time.perf_counter()
    rerank.build_documents(candidates)
    return time.perf_counter() - started


async def _ticks_per_second(work, window: float = 0.4) -> tuple:
    """(ticks/s, seconds per call) with a 1ms coroutine running beside
    `work` -- the measurement that tells a slow call from a BLOCKING
    one.

    `work` is a plain callable, deliberately: this is what a call site
    that forgot its thread looks like, and it is what hopai/asyncio.py
    and hopai/vectors.py do at their two `build_documents()` sites.

    It is called REPEATEDLY for at least `window` seconds rather than
    once, and that is not padding: a pruned build of 50 candidates
    takes under a millisecond, so a single call finishes before the
    1ms ticker can fire even once and scores a ticks/s of zero -- the
    same number total starvation scores. Measuring over a fixed window
    makes fast and blocked distinguishable again.

    THE LONGEST GAP is the third number and the least ambiguous of the
    three: it is how long, in one stretch, the ticker was not given the
    thread. A rate can be dragged down by running builds back to back;
    a gap cannot. It is what "the loop was held for 121ms" MEANS."""
    stamps = [time.perf_counter()]
    running = True

    async def tick():
        while running:
            await asyncio.sleep(0.001)
            stamps.append(time.perf_counter())

    counter = asyncio.ensure_future(tick())
    started = time.perf_counter()
    calls = 0
    try:
        while True:
            work()
            calls += 1
            if time.perf_counter() - started >= window:
                break
            await asyncio.sleep(0)          # yield, as an await-ing caller would
    finally:
        elapsed = time.perf_counter() - started
        running = False
        await counter
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    return len(gaps) / elapsed, elapsed / calls, max(gaps, default=elapsed)


async def _loop_profile(rerank: Rerank, sizes: list, candidates: int) -> dict:
    idle, _, idle_gap = await _ticks_per_second(lambda: None)
    profile = {"idle_ticks_per_second": round(idle),
               "idle_longest_gap_ms": round(idle_gap * 1000, 1)}
    for kilobytes in sizes:
        batch = rows(candidates, kilobytes)
        rerank.build_documents(batch)                   # warm
        rate, per_call, gap = await _ticks_per_second(
            lambda rows=batch: rerank.build_documents(rows))
        profile[f"{kilobytes}KB"] = {
            "ms_per_build": round(per_call * 1000, 1),
            # The number to read: with builds running back to back, this
            # is how long the ticker went without the thread. Unpruned
            # it tracked ms_per_build exactly, which is what makes the
            # build a block rather than a cost.
            "longest_gap_ms": round(gap * 1000, 1),
            "ticks_per_second": round(rate),
        }
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=50,
                        help="documents per build, i.e. Rerank(candidates=)")
    parser.add_argument("--sizes", default="1,10,100,500",
                        help="comma-separated KB of payload per candidate")
    parser.add_argument("--filter", default=".properties.title",
                        help="the document_from filter to measure")
    parser.add_argument("--out", default="bench_document_results.json")
    args = parser.parse_args()

    sizes = [int(size) for size in args.sizes.split(",")]
    rerank = Rerank(lambda query, documents: [0.0] * len(documents),
                    document_from=args.filter, candidates=args.candidates)

    per_row = {kilobytes: timed(rerank, rows(args.candidates, kilobytes))
               for kilobytes in sizes}
    results = {
        "candidates": args.candidates,
        "document_from": args.filter,
        "us_per_row": {f"{kb}KB": round(cost * 1e6, 1) for kb, cost in per_row.items()},
    }
    # The transferable number, and the one the regression shows up in:
    # the SLOPE against payload. Pruned it is noise; unpruned it was
    # ~25us for every KB a candidate carried, whatever the filter read.
    low, high = min(sizes), max(sizes)
    if high > low:
        results["us_per_kb"] = round((per_row[high] - per_row[low]) / (high - low) * 1e6, 3)
    # The same rows read TWO ways: a 60-character title and a 1KB field.
    # Unpruned these cost the same because both marshalled the whole
    # row; pruned they cost the same because neither does. Same
    # equality, opposite reason -- the ratio is the tell.
    biggest = rows(args.candidates, high)
    title = timed(Rerank(lambda q, d: [], document_from=".properties.title"), biggest)
    field = timed(Rerank(lambda q, d: [], document_from=".properties.field_0"), biggest)
    results["narrow_vs_wide_projection"] = {
        f"title_us_per_row_at_{high}KB": round(title * 1e6, 1),
        f"field_us_per_row_at_{high}KB": round(field * 1e6, 1),
    }
    results["event_loop"] = asyncio.run(_loop_profile(rerank, sizes, args.candidates))

    print(json.dumps(results, indent=2))
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
