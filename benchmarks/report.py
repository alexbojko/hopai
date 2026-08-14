"""
benchmarks/report.py

Turns a benchmark run into `benchmarks/RESULTS.md`: the measurements, an
ASCII chart of them, and the machine they were taken on.

A latency number without its machine is not a measurement, it is a
rumour. Two people quoting "3ms" on different hardware, with different
`shared_buffers`, against different dataset sizes are not disagreeing --
they never measured the same thing. So the report records CPU, memory,
OS, Python, the PostgreSQL version AND the server settings that actually
move these numbers, next to the results themselves.

The file is REGENERATED on every run, never appended to. A benchmark
report that accumulates is a report nobody trusts, because you cannot
tell which rows describe the code in front of you.

No new dependencies -- rule one of this project. Everything here comes
from the standard library, and each probe degrades to "unknown" rather
than raising when a platform does not offer it.
"""

from __future__ import annotations

import math
import os
import statistics
import platform
import subprocess
from datetime import datetime, timezone

#: Server settings worth recording: each one materially changes traversal
#: latency, and each is the first thing a reader will ask about.
PG_SETTINGS = ("shared_buffers", "work_mem", "effective_cache_size",
               "max_parallel_workers_per_gather", "jit")

UNKNOWN = "unknown"


# ---------------------------------------------------------------------
# Machine profile
# ---------------------------------------------------------------------

def _total_memory_bytes() -> int | None:
    """Physical RAM, without pulling in psutil."""
    try:  # Linux, and anything else exposing the POSIX knobs
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=5)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _cpu_model() -> str:
    """A human-readable CPU name. platform.processor() is empty on many
    Linux builds and returns the architecture on macOS, so neither is
    trustworthy alone."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or UNKNOWN


def _human_bytes(value: int | None) -> str:
    if not value:
        return UNKNOWN
    return f"{value / 1024 ** 3:.1f} GiB"


def machine_profile(connection=None) -> dict:
    """Everything needed to reproduce -- or fairly dispute -- a number."""
    profile = {
        "cpu": _cpu_model(),
        "cores": os.cpu_count() or UNKNOWN,
        "architecture": platform.machine() or UNKNOWN,
        "memory": _human_bytes(_total_memory_bytes()),
        "os": platform.platform(),
        "python": platform.python_version(),
        "postgres": UNKNOWN,
    }
    if connection is not None:
        profile.update(postgres_profile(connection))
    return profile


def postgres_profile(connection) -> dict:
    """Server version and the settings that move traversal latency.

    Read from the live server rather than assumed, because the number in
    the report was produced by whatever the server is actually running --
    a container default, not the value in anyone's postgresql.conf."""
    from sqlalchemy import text

    found = {}
    try:
        version = connection.execute(text("SHOW server_version")).scalar()
        found["postgres"] = str(version)
        for setting in PG_SETTINGS:
            try:
                found[setting] = str(connection.execute(text(f"SHOW {setting}")).scalar())
            except Exception:  # noqa: BLE001 - an unknown GUC must not sink the report
                found[setting] = UNKNOWN
    except Exception:  # noqa: BLE001
        found["postgres"] = UNKNOWN
    return found


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def bar(value: float, peak: float, width: int = 40) -> str:
    """One horizontal bar, scaled against the slowest query in the run.

    Scaling to the peak rather than to a fixed millisecond scale is what
    makes the chart readable whether the run took microseconds or
    minutes -- the shape of the comparison is the point, and the exact
    numbers sit beside it anyway."""
    if peak <= 0 or value <= 0:
        return "░" * width
    filled = max(1, round(value / peak * width))
    return "█" * filled + "░" * (width - filled)


def chart(results: list, key: str, width: int = 40) -> str:
    """An ASCII chart of one column, slowest first."""
    if not results:
        return "(no measurements)"
    rows = sorted(results, key=lambda r: r.get(key) or 0, reverse=True)
    peak = max((r.get(key) or 0) for r in rows)
    label_width = max(len(str(r["query"])) for r in rows)
    return "\n".join(
        f"{str(r['query']):<{label_width}}  {bar(r.get(key) or 0, peak, width)}  "
        f"{(r.get(key) or 0):>9,.1f} ms"
        + ("   <- returned NO ROWS" if not r.get("nodes") else "")
        for r in rows
    )


def empty_queries(results: list) -> list:
    """Queries that matched nothing.

    Their timing is real and meaningless: a query that matched nothing
    did no work, so it will always look fast and will always be the wrong
    thing to quote.

    Emptiness comes from the runner, not from a row count -- an
    aggregate without a Count() has no row count, and treating that
    absence as zero reported queries that computed real values as having
    measured nothing."""
    return [r.get("id") or r["query"] for r in results
            if r.get("empty") if not r.get("dnf")]


def overhead(row: dict) -> float | None:
    """How many times slower traverse() is than the same SQL run raw.

    Not a criticism of the library -- the gap buys result mapping,
    property hydration and a Subgraph. It is the number to quote when
    somebody asks what the convenience costs, and quoting it honestly
    means measuring it rather than estimating it."""
    raw, warm = row.get("raw_sql_ms"), row.get("warm_ms")
    if not raw or not warm:
        return None
    return warm / raw


def _table(results: list) -> str:
    """Every measurement, including the ones that are not measurements.

    A DNF row keeps its place in the table with its outcome spelled out
    rather than a number: dropping it would hide a query, and printing a
    budget in the timing column would let non-completion read as a
    latency."""
    has_raw = any(r.get("raw_sql_ms") for r in results)
    columns = ["ID", "Query", "Feature", "Tier", "Cold (ms)", "Warm (ms), median + range"]
    if has_raw:
        columns += ["Raw SQL (ms)", "Overhead"]
    columns += ["Rows"]
    header = ("| " + " | ".join(columns) + " |\n"
              "| --- | --- | --- | --- |" + " ---: |" * (len(columns) - 4))

    rows = []
    for r in results:
        cells = [r.get("id", ""), r.get("query", ""), f"`{r.get('feature', '')}`",
                 r.get("tier", "")]
        if r.get("dnf"):
            outcome = f"**DNF** (>{r.get('budget_s', 0):.0f}s)"
            cells += [outcome, outcome] + (["-", "-"] if has_raw else []) + ["-"]
            rows.append("| " + " | ".join(cells) + " |")
            continue

        warm = f"{r.get('warm_ms', 0):,.1f}"
        if r.get("warm_min_ms") is not None and r.get("samples", 0) > 1:
            warm += f" <sub>{r['warm_min_ms']:,.1f}-{r['warm_max_ms']:,.1f}</sub>"
        cells += [f"{r.get('cold_ms', 0):,.1f}", warm]
        if has_raw:
            ratio = overhead(r)
            cells += [f"{r.get('raw_sql_ms', 0):,.1f}", f"{ratio:.1f}x" if ratio else "-"]
        # "-" not "0": an aggregate without a Count() has no row count,
        # and printing zero would read as "matched nothing".
        rows_cell = "-" if r.get("nodes") is None else f"{r['nodes']:,}"
        cells += [rows_cell]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, *rows]) if rows else "(no measurements)"


def _profile_table(profile: dict) -> str:
    labels = {
        "cpu": "CPU", "cores": "Cores", "architecture": "Architecture",
        "memory": "Memory", "os": "OS", "python": "Python", "postgres": "PostgreSQL",
    }
    rows = [f"| {labels.get(k, f'`{k}`')} | {v} |"
            for k, v in profile.items() if v not in (None, "")]
    return "\n".join(["| | |", "| --- | --- |", *rows])


def log_bar(value: float, low: float, high: float, width: int = 34) -> str:
    """A bar on a LOG scale.

    Traversal timings span orders of magnitude -- single-digit
    milliseconds for a property filter, seconds for a deep walk. On a
    linear scale every fast query collapses to one indistinguishable
    pixel and the chart only shows the outlier. Log keeps all of them
    readable, which is the entire reason to draw a chart instead of
    reading the table."""
    if value <= 0 or high <= low:
        return "·" * width
    span = math.log10(high) - math.log10(low)
    if span <= 0:
        return "█" * width
    filled = max(1, round((math.log10(max(value, low)) - math.log10(low)) / span * width))
    return "█" * min(filled, width) + "·" * max(0, width - filled)


def grouped_chart(results: list, series: list, width: int = 34) -> str:
    """One block per query, one bar per system, log scale.

    `series` is [(key, label)]. A row whose value is missing is skipped;
    a row marked dnf is drawn full width and labelled, because "never
    finished" is a different outcome from "slow" and must not read as a
    number."""
    measured = [v for r in results for k, _ in series
                if (v := r.get(k)) and v > 0]
    if not measured:
        return "(no measurements)"
    low, high = min(measured), max(measured)
    label_width = max(len(label) for _, label in series)

    blocks = []
    for r in results:
        head = f"{r.get('id', '')} {r.get('query', '')}".strip()
        lines = [f"{head}   [{r.get('feature', '')}]"]
        for key, label in series:
            value = r.get(key)
            if value is None:
                continue
            if r.get("dnf") and key in ("warm_ms", "cold_ms"):
                lines.append(f"  {label:<{label_width}}  {'▓' * width}  "
                             f"DNF (>{r.get('budget_s', 0):.0f}s)")
                continue
            lines.append(f"  {label:<{label_width}}  {log_bar(value, low, high, width)}  "
                         f"{value:>10,.1f} ms")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def headline(results: list) -> list:
    """The four numbers a reader should leave with.

    Derived, never hand-written: a headline that drifts from the table
    beneath it is worse than no headline."""
    ran = [r for r in results if not r.get("dnf") and r.get("warm_ms")]
    dnf = [r for r in results if r.get("dnf")]
    stats = []
    if ran:
        slowest = max(ran, key=lambda r: r["warm_ms"])
        fastest = min(ran, key=lambda r: r["warm_ms"])
        under_1s = sum(1 for r in ran if r["warm_ms"] < 1000)
        stats += [
            f"| **{slowest['warm_ms']:,.0f} ms** | slowest: `{slowest['id']}` "
            f"{slowest['query']} |",
            f"| **{fastest['warm_ms']:,.1f} ms** | fastest: `{fastest['id']}` "
            f"{fastest['query']} |",
            f"| **{under_1s} / {len(results)}** | queries answering in under a second |",
        ]
    stats.append(
        f"| **{len(dnf)}** | queries that did not finish"
        + (": " + ", ".join(f"`{r['id']}`" for r in dnf) if dnf else "") + " |")
    return ["| | |", "| ---: | --- |", *stats]


def by_tier(results: list) -> list:
    """Median warm time per difficulty tier.

    Aggregation cost is not one number: it depends on how much the walk
    underneath had to match, how many aggregates run, and whether
    DISTINCT forces a sort. Grouping by tier is what makes "how fast are
    aggregations" a question with an answer."""
    tiers = ("simple", "complex", "very complex")
    rows = ["| Tier | Queries | Median warm | Slowest |", "| --- | ---: | ---: | --- |"]
    seen = False
    for tier in tiers:
        group = [r for r in results if r.get("tier") == tier and not r.get("dnf")
                 and r.get("warm_ms")]
        if not group:
            continue
        seen = True
        slowest = max(group, key=lambda r: r["warm_ms"])
        rows.append(
            f"| {tier} | {len(group)} | "
            f"{statistics.median(r['warm_ms'] for r in group):,.1f} ms | "
            f"`{slowest['id']}` {slowest['warm_ms']:,.1f} ms |")
    return rows if seen else []


def twin_savings(results: list) -> list:
    """Aggregate rows measured against the traversal they mirror.

    The pair is the only honest way to quote what aggregation saves: the
    same chain, the same match set, one materialising a subgraph and one
    not."""
    index = {r.get("id"): r for r in results}
    pairs = []
    for row in results:
        twin = index.get(row.get("twin_of"))
        if twin and row.get("warm_ms") and twin.get("warm_ms"):
            pairs.append((row, twin, twin["warm_ms"] / row["warm_ms"]))
    return pairs


def findings(results: list) -> list:
    """Observations the data supports, stated only when it does.

    Every one is conditional on the measurement: no finding is printed
    for a phenomenon that did not occur in this run."""
    out = []
    ran = [r for r in results if not r.get("dnf")]

    worst = [r for r in ran if overhead(r) is not None]
    if worst:
        top = max(worst, key=overhead)
        out += [
            f"**Where the library layer costs most.** `{top['id']}` "
            f"({top['query']}) ran {overhead(top):.1f}x the raw statement "
            f"({top['warm_ms']:,.1f} ms vs {top['raw_sql_ms']:,.1f} ms) over "
            f"{top.get('nodes') or 0:,} nodes. The gap is result mapping and property "
            f"hydration, and it is roughly fixed per call -- so it dominates a small "
            f"answer and shrinks against a large one. Read it next to the row count, "
            f"never alone.",
            "",
        ]

    pairs = twin_savings(results)
    if pairs:
        best = max(pairs, key=lambda p: p[2])
        out += [
            f"**What aggregating instead of traversing saves.** `{best[0]['id']}` runs "
            f"`{best[1]['id']}`'s chain and returns a number instead of a subgraph: "
            f"{best[0]['warm_ms']:,.1f} ms against {best[1]['warm_ms']:,.1f} ms, "
            f"**{best[2]:.1f}x**. Same walk, same match set -- the difference is the "
            f"edge-reconstruction CTEs and the property hydration that a count does not "
            f"need. Pairs measured: "
            + ", ".join(f"`{a['id']}`/`{b['id']}` {r:.1f}x" for a, b, r in pairs) + ".",
            "",
        ]

    empty = empty_queries(ran)
    if empty:
        out += [
            "**Queries that measured nothing.** " + ", ".join(f"`{q}`" for q in empty)
            + " returned zero rows. Finding nothing is always fast, so these timings "
              "describe the dataset, not the engine.",
            "",
        ]

    dnf = [r for r in results if r.get("dnf")]
    if dnf:
        out += [
            "**Did not finish.** " + ", ".join(f"`{r['id']}` ({r['query']})" for r in dnf)
            + f" exceeded the {dnf[0].get('budget_s', 0):.0f}s budget and were cancelled "
              "by the server. Non-completion is reported as its own outcome rather than "
              "as a large number, which would let it average in with the rest.",
            "",
        ]

    spread = [r for r in ran if r.get("warm_max_ms") and r.get("warm_min_ms")
              and r["warm_min_ms"] > 0
              and r["warm_max_ms"] / r["warm_min_ms"] > 1.5]
    if spread:
        out += [
            "**Noisy measurements.** " + ", ".join(f"`{r['id']}`" for r in spread)
            + " varied by more than 50% between warm runs. Their medians are not "
              "comparable across commits; re-run on a quiet machine before drawing a "
              "conclusion from them.",
            "",
        ]
    return out or ["Nothing notable: every query finished, returned rows, and was stable "
                   "across runs.", ""]


def _floor_section(results: list) -> list:
    """hopai's own statement, executed straight through the driver.

    Only rendered when it was measured. The gap between this and warm
    latency is what the library layer costs -- result mapping, property
    hydration, building a Subgraph -- measured rather than estimated,
    and from ONE statement rather than from two hand-written queries
    somebody has to keep in step."""
    if not any(r.get("raw_sql_ms") for r in results):
        return []
    return [
        "## The floor: the same SQL, run raw",
        "",
        "No result mapping, no property hydration, no `Subgraph`. The ratio in the",
        "table below is what the convenience costs.",
        "",
        "```",
        chart(results, "raw_sql_ms"),
        "```",
        "",
    ]


def render(results: list, profile: dict, dataset: dict | None = None,
           generated_at: str | None = None) -> str:
    """The whole report, in the order a reader needs it: what happened,
    then the evidence, then the caveats. Deterministic given its inputs,
    so it is testable without a database or a stopwatch."""
    stamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dataset = dataset or {}
    ran = [r for r in results if not r.get("dnf")]
    series = [("warm_ms", "hopai"), ("raw_sql_ms", "raw SQL")]
    has_raw = any(r.get("raw_sql_ms") for r in results)

    parts = [
        "# hopai benchmark",
        "",
        f"{len(results)} queries covering every capability the library claims — forward, "
        "backward and mixed direction, bounded and deep multi-hop, compound chains, "
        "AND/OR/NOT, range comparisons, edge-property filtering and OPTIONAL — run "
        "against identical data.",
        "",
        f"Generated by `benchmarks/bench_hopai.py` on {stamp}.",
        "",
        "> **Rewritten on every run.** Do not edit by hand, and do not compare rows",
        "> across machines — the profile below is part of the measurement.",
        "",
        "## 01 — Headline",
        "",
        *headline(results),
        "",
        "## 02 — Every query",
        "",
        "Log scale: these timings span orders of magnitude, and on a linear scale every",
        "fast query collapses into one indistinguishable mark. Bars are comparable across",
        "the whole chart; the numbers beside them are exact.",
        "",
        "```",
        grouped_chart(results, [(key, label) for key, label in series
                        if key != "raw_sql_ms" or has_raw]),
        "```",
        "",
        "## 03 — Cost by difficulty",
        "",
        *by_tier(results),
        "",
        "## 04 — Full results",
        "",
        _table(results),
        "",
    ]
    if has_raw:
        parts += [
            "`raw SQL` is hopai's own statement executed straight through the driver — no",
            "result mapping, no property hydration, no `Subgraph`. The overhead column is",
            "what the API costs, measured rather than estimated, from one statement rather",
            "than two hand-written queries that would drift apart.",
            "",
        ]
    parts += [
        "## 05 — Findings",
        "",
        *findings(results),
        "## 06 — Environment",
        "",
        _profile_table(profile),
        "",
    ]
    if dataset:
        parts += [
            "Dataset: "
            + ", ".join(f"**{v:,}** {k}" if isinstance(v, int) else f"{k} `{v}`"
                        for k, v in dataset.items())
            + ".",
            "",
        ]
    parts += [
        f"Every timing is wall-clock around actual query execution. Warm figures are the "
        f"median of {ran[0].get('samples', 1) if ran else 1} runs with the cache primed; "
        f"cold is the first run after load. Queries exceeding the per-query budget are "
        f"cancelled by the server and reported as DNF, never as a large number.",
        "",
        "Synthetic data: a sparse random background DAG plus a deliberately structured hub "
        "subgraph with real fan-in across several depth levels — the shape that stresses a "
        "traversal engine, which a purely random graph does not.",
        "",
    ]
    return "\n".join(parts)
