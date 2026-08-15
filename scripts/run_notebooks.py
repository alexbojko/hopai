#!/usr/bin/env python
"""
Execute the notebooks in notebooks/ and fail if any cell raises.

The notebooks are documentation that runs, which is only worth something
if something runs them: this is what CI calls, and what to call locally
before committing a change to one.

    python scripts/run_notebooks.py                  # verify all of them
    python scripts/run_notebooks.py --save           # ...and store the outputs
    python scripts/run_notebooks.py 03_aggregation   # just the ones matching

Without --save the executed copy is thrown away and only the verdict
matters, so a verification run never dirties the working tree. With it,
the notebook is rewritten with fresh outputs -- that is how the committed
outputs are regenerated after an API change.

Needs a reachable PostgreSQL (HOPAI_DSN, defaulting to the one in
docker-compose.yml) and the notebook extra:

    pip install -e ".[notebooks]"
    docker compose up -d

A missing database is a skip, not a failure -- the same rule the test
suite follows -- unless HOPAI_REQUIRE_DB=1 is set, as it is in CI.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
#: Generous, because a cell that hangs should fail the run rather than
#: the job's global timeout, and EXPLAIN ANALYZE on a cold cache is slow.
CELL_TIMEOUT = 300


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


def run(path: pathlib.Path, save: bool) -> float:
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
    if save:
        nbformat.write(notebook, path)
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match", nargs="*", help="only run notebooks whose name contains this")
    parser.add_argument("--save", action="store_true",
                        help="write the executed outputs back into the notebooks")
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
            elapsed = run(path, args.save)
        except CellExecutionError as exc:
            print("FAILED")
            # The traceback nbclient carries is the one from the kernel,
            # which names the cell and the line -- print it whole rather
            # than a summary that sends the reader back to re-run by hand.
            print(f"\n{exc}\n", file=sys.stderr)
            failures.append(path.name)
        else:
            print(f"ok ({elapsed:.1f}s)")

    print()
    if failures:
        print(f"{len(failures)} notebook(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"{len(paths)} notebook(s) executed cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
