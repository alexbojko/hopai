#!/usr/bin/env python
"""
Execute the notebooks in notebooks/ and fail if any cell raises.

The notebooks are documentation that runs, which is only worth something
if something runs them: this is what CI calls, and what to call locally
before committing a change to one.

    python scripts/run_notebooks.py                  # verify all of them
    python scripts/run_notebooks.py --save           # ...and store the outputs
    python scripts/run_notebooks.py --check          # ...and fail if outputs went stale
    python scripts/run_notebooks.py 03_aggregation   # just the ones matching

Without --save or --check the executed copy is thrown away and only the
verdict -- did every cell run without raising -- matters, so a verification
run never dirties the working tree. That verdict is necessary but not
sufficient: a notebook can run cleanly and still print something the
committed output no longer shows, e.g. an API change that added a field
nobody re-ran `--save` for. --check catches that: it diffs the freshly
executed outputs against the committed ones (after normalizing execution
metadata and timing-shaped numbers -- see mask_timings()) and fails, naming
the offending notebook and cell, on any other difference. --save is the
fix for a real --check failure: it rewrites the notebook with fresh
outputs, which is how the committed outputs are regenerated after an API
change.

Needs a reachable PostgreSQL (HOPAI_DSN, defaulting to the one in
docker-compose.yml) and the notebook extra:

    pip install -e ".[notebooks]"
    docker compose up -d

A missing database is a skip, not a failure -- the same rule the test
suite follows -- unless HOPAI_REQUIRE_DB=1 is set, as it is in CI.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
#: Generous, because a cell that hangs should fail the run rather than
#: the job's global timeout, and EXPLAIN ANALYZE on a cold cache is slow.
CELL_TIMEOUT = 300

#: A cell tagged with this (Cell menu > Edit Tags in Jupyter, or
#: cell.metadata.tags in the raw JSON) is exempt from --check's diff --
#: for a cell whose output genuinely, legitimately varies run to run in a
#: way that is not timing-shaped (an id whose value depends on insertion
#: history nobody controls, say), and so cannot be normalized away without
#: also hiding a real regression in every OTHER cell. The name matches
#: nbval's tag of the same purpose, which is the convention a reader is
#: more likely to already know than one invented here -- see CLAUDE.md
#: principle 2. hopai does not depend on nbval; only the tag string is
#: borrowed.
IGNORE_TAG = "nbval-ignore-output"

#: What --check masks before diffing, because it varies for a reason that
#: has nothing to do with whether the notebook is stale: wall-clock time.
#: Two shapes appear in the notebooks' printed output --
#:   - a bare `elapsed_ms=` keyword argument inside a dataclass repr
#:     (`Subgraph(nodes=4, edges=4, elapsed_ms=11.4)`);
#:   - a number immediately followed by a time unit, with or without a
#:     space (`"9.9 ms"`, `"29.6ms"`, EXPLAIN ANALYZE's own
#:     `"Execution Time: 4.665 ms"`).
#: The lookbehind/lookahead pair around the second branch is load-bearing,
#: not decorative: without it, `\d+\.?\d*\s*(ms|s|...)` also matches the
#: literal "s" alias in emitted SQL like `near_0.s AS s` (the digit in
#: `near_0`, zero digits after a phantom decimal, zero whitespace, then
#: the "s" two tokens later) -- a false positive that would silently mask
#: a real SQL regression next to it. Requiring the character before the
#: number to be neither a word character nor "." keeps the match starting
#: on an actual number, not a suffix of an identifier; requiring the
#: character after the unit to not be a word character keeps "ms" from
#: eating into a following word.
#:
#: The leading `\s*` is the same idea one layer out: notebook 08 prints a
#: `>8.1f`-formatted table, so the *count* of space-padding in front of a
#: value is a function of how many digits the value has, not of anything
#: the notebook is documenting. Masking the number alone left that
#: padding behind at two different widths for two differently-sized
#: numbers -- a diff with nothing wrong to report. Consuming the padding
#: into the same `<T>` token removes the width along with the value it
#: was padding.
#:
#: Checked by hand against every timing-shaped string already committed in
#: notebooks/*.ipynb, with no false positive and no missed print -- that
#: corpus is what this pattern was fitted against.
TIMING_RE = re.compile(
    r"(?P<kwarg>\w*_ms=)\d[\d,]*(?:\.\d+)?"
    r"|\s*(?<![\w.])\d[\d,]*(?:\.\d+)?\s?(?:ms|µs|us|s)\b(?![\w])"
)


def mask_timings(text: str) -> str:
    """Replace every timing-shaped number in `text` with a fixed token."""

    def replace(match: re.Match) -> str:
        if match.group("kwarg"):
            return f"{match.group('kwarg')}<T>"
        return "<T>"

    return TIMING_RE.sub(replace, text)


def database_reachable() -> tuple:
    """(reachable, message). Import-light enough to run before nbclient."""
    from sqlalchemy import create_engine
    from sqlalchemy.exc import OperationalError

    sys.path.insert(0, str(NOTEBOOKS))
    from demo_graph import DSN

    engine = create_engine(DSN)
    try:
        engine.connect().close()
        return True, DSN
    except OperationalError as exc:
        return False, f"no PostgreSQL at {DSN} ({exc.orig.__class__.__name__})"
    finally:
        engine.dispose()


def _execute(path: pathlib.Path):
    """Run `path` into an in-memory copy. Returns (notebook, elapsed_seconds).

    Never touches the file on disk -- `run()` and `check()` both build on
    this and decide separately whether to write anything back.
    """
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        # The notebooks import demo_graph, so the kernel has to start in
        # notebooks/ -- exactly where a reader would open them.
        resources={"metadata": {"path": str(NOTEBOOKS)}},
    )
    started = time.perf_counter()
    client.execute()
    elapsed = time.perf_counter() - started
    return notebook, elapsed


def run(path: pathlib.Path, save: bool) -> float:
    import nbformat

    notebook, elapsed = _execute(path)
    if save:
        nbformat.write(notebook, path)
    return elapsed


def _normalize_output(output: dict) -> dict:
    """A JSON-safe copy of one cell output with the run-to-run noise gone.

    Drops `execution_count` (nbclient's kernel-restart counter on an
    execute_result, not a property of what was printed) and masks every
    timing-shaped number in its text -- see TIMING_RE. Everything else --
    the SQL, the JSON, the row counts, the ids -- is compared verbatim,
    which is the point: those are exactly what going stale would change.
    """
    normalized = dict(output)
    normalized.pop("execution_count", None)
    if "text" in normalized:
        text = normalized["text"]
        if isinstance(text, list):
            text = "".join(text)
        normalized["text"] = mask_timings(text)
    if "data" in normalized:
        masked_data = {}
        for mime, value in normalized["data"].items():
            if isinstance(value, list):
                value = "".join(value)
            if isinstance(value, str):
                value = mask_timings(value)
            masked_data[mime] = value
        normalized["data"] = masked_data
    return normalized


def _coalesce_streams(outputs: list) -> list:
    """Merge adjacent stream outputs that share a stream name into one.

    nbclient appends one 'stream' output per iopub message it receives
    (see NotebookClient.output) with no merging of its own, and how many
    iopub messages a cell's prints arrive as depends on kernel-side stdout
    flush timing -- not on what the cell printed. The same cell's `print`
    calls can turn up as one merged blob one run and several split outputs
    the next, verified by running this notebook suite twice locally.
    Comparing merged text rather than message boundaries is what keeps
    that scheduling noise from reading as a stale notebook.
    """
    coalesced: list = []
    for output in outputs:
        text = output.get("text")
        previous = coalesced[-1] if coalesced else None
        if (
            text is not None
            and previous is not None
            and previous.get("output_type") == "stream"
            and previous.get("name") == output.get("name")
        ):
            merged = dict(previous)
            merged["text"] = previous.get("text", "") + text
            coalesced[-1] = merged
        else:
            coalesced.append(dict(output))
    return coalesced


def _normalize_outputs(cell) -> list:
    return [_normalize_output(output) for output in _coalesce_streams(cell.get("outputs", []))]


def _cell_source_summary(cell) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    first_line = source.strip().splitlines()[0] if source.strip() else "(empty cell)"
    return first_line


def _render_json(outputs: list) -> list:
    return json.dumps(outputs, indent=2, sort_keys=True).splitlines()


def diff_notebooks(committed, fresh, name: str) -> list:
    """[messages], one per code cell whose normalized output changed.

    Comparing cell by cell, rather than the notebook as one blob, is what
    lets a mismatch name the ONE cell that went stale instead of leaving
    a reader to diff the whole file by hand to find it.
    """
    committed_cells = committed.get("cells", [])
    fresh_cells = fresh.get("cells", [])
    if len(committed_cells) != len(fresh_cells):
        # Can't happen from --check alone (both are read from the same
        # committed source), but a mismatched pairing below would blame
        # the wrong cell, so refuse to guess rather than mislead.
        return [f"{name}: committed has {len(committed_cells)} cells, "
                f"freshly executed has {len(fresh_cells)} -- re-run and inspect by hand"]

    mismatches = []
    paired = zip(committed_cells, fresh_cells, strict=True)
    for index, (committed_cell, fresh_cell) in enumerate(paired):
        if committed_cell.get("cell_type") != "code":
            continue
        if IGNORE_TAG in committed_cell.get("metadata", {}).get("tags", []):
            continue
        before = _normalize_outputs(committed_cell)
        after = _normalize_outputs(fresh_cell)
        if before == after:
            continue
        diff = "\n".join(difflib.unified_diff(
            _render_json(before), _render_json(after),
            fromfile="committed", tofile="freshly executed", lineterm=""))
        mismatches.append(
            f"{name} cell {index} (`{_cell_source_summary(committed_cell)}`) is stale:\n{diff}")
    return mismatches


def check(path: pathlib.Path) -> tuple:
    """(elapsed_seconds, [mismatch messages]). Never writes to `path`."""
    import nbformat

    committed = nbformat.read(path, as_version=4)
    fresh, elapsed = _execute(path)
    return elapsed, diff_notebooks(committed, fresh, path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match", nargs="*", help="only run notebooks whose name contains this")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--save", action="store_true",
                       help="write the executed outputs back into the notebooks")
    mode.add_argument("--check", action="store_true",
                       help="fail if the freshly executed outputs differ from the committed "
                            "ones, after normalizing execution metadata and timing-shaped "
                            "numbers -- naming the offending notebook and cell")
    args = parser.parse_args()

    reachable, message = database_reachable()
    if not reachable:
        if os.environ.get("HOPAI_REQUIRE_DB"):
            print(f"error: {message} -- HOPAI_REQUIRE_DB is set", file=sys.stderr)
            return 1
        print(f"skipped: {message} -- start one with `docker compose up -d`, "
              f"or set HOPAI_DSN")
        return 0
    print(f"database: {message}\n")

    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if args.match:
        paths = [p for p in paths if any(m in p.name for m in args.match)]
    if not paths:
        print("error: no notebooks matched", file=sys.stderr)
        return 1

    from nbclient.exceptions import CellExecutionError

    failures = []
    for path in paths:
        print(f"running {path.name} ... ", end="", flush=True)
        try:
            if args.check:
                elapsed, mismatches = check(path)
            else:
                elapsed = run(path, args.save)
                mismatches = []
        except CellExecutionError as exc:
            print("FAILED")
            # The traceback nbclient carries is the one from the kernel,
            # which names the cell and the line -- print it whole rather
            # than a summary that sends the reader back to re-run by hand.
            print(f"\n{exc}\n", file=sys.stderr)
            failures.append(path.name)
            continue

        if mismatches:
            print("STALE")
            # One notebook can have several stale cells; each mismatch
            # already names its own cell, so print them all rather than
            # stopping at the first and hiding the rest.
            print(f"\n{chr(10).join(mismatches)}\n", file=sys.stderr)
            failures.append(path.name)
        else:
            print(f"ok ({elapsed:.1f}s)")

    print()
    if failures:
        verb = "failed or went stale" if args.check else "failed"
        print(f"{len(failures)} notebook(s) {verb}: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"{len(paths)} notebook(s) executed cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
