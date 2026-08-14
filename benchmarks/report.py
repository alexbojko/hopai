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

import os
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

    Their timing is real and meaningless: a traversal that returns no
    rows did no work, so it will always look fast and will always be the
    wrong thing to quote. Called out rather than quietly listed."""
    return [r["query"] for r in results if not r.get("nodes")]


def _table(results: list) -> str:
    header = ("| Query | Cold (ms) | Warm (ms) | Nodes | Edges |\n"
              "| --- | ---: | ---: | ---: | ---: |")
    rows = [
        f"| `{r['query']}` | {r.get('cold_ms', 0):,.1f} | {r.get('warm_ms', 0):,.1f} "
        f"| {r.get('nodes', 0):,} | {r.get('edges', 0):,} |"
        for r in results
    ]
    return "\n".join([header, *rows]) if rows else "(no measurements)"


def _profile_table(profile: dict) -> str:
    labels = {
        "cpu": "CPU", "cores": "Cores", "architecture": "Architecture",
        "memory": "Memory", "os": "OS", "python": "Python", "postgres": "PostgreSQL",
    }
    rows = [f"| {labels.get(k, f'`{k}`')} | {v} |"
            for k, v in profile.items() if v not in (None, "")]
    return "\n".join(["| | |", "| --- | --- |", *rows])


def render(results: list, profile: dict, dataset: dict | None = None,
           generated_at: str | None = None) -> str:
    """The whole report. Deterministic given its inputs, so it is
    testable without a database or a stopwatch."""
    stamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# Benchmark results",
        "",
        f"Generated by `benchmarks/bench_hopai.py` on {stamp}.",
        "",
        "> **This file is rewritten on every run.** Do not edit it by hand, and do not",
        "> compare rows across machines -- the profile below is part of the measurement,",
        "> not decoration.",
        "",
        "## Machine",
        "",
        _profile_table(profile),
        "",
    ]
    if dataset:
        parts += [
            "## Dataset",
            "",
            "\n".join(f"- **{k}**: {v:,}" if isinstance(v, int) else f"- **{k}**: {v}"
                      for k, v in dataset.items()),
            "",
        ]
    empty = empty_queries(results)
    if empty:
        parts += [
            "> ⚠️ **Measured nothing:** " + ", ".join(f"`{q}`" for q in empty) + ".",
            "> These returned zero rows, so their timings say how fast it is to find",
            "> nothing on this dataset -- not how fast the query is. Fix the data or the",
            "> query before quoting them.",
            "",
        ]
    parts += [
        "## Warm latency",
        "",
        "Second run of each query, with the cache primed -- the number a live",
        "system actually experiences.",
        "",
        "```",
        chart(results, "warm_ms"),
        "```",
        "",
        "## Cold latency",
        "",
        "First run after the data is loaded, nothing cached.",
        "",
        "```",
        chart(results, "cold_ms"),
        "```",
        "",
        "## All measurements",
        "",
        _table(results),
        "",
    ]
    return "\n".join(parts)
