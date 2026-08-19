#!/usr/bin/env python3
"""
Run mutmut with mutation restricted to the lines a PR's diff changed.

    python scripts/mutation_run.py <changed-lines.json> [MUTANT_NAME ...]

where the JSON is `{"hopai/core.py": [11, 12, 40], ...}` as produced by
`scripts/mutation_scope.py --changed-lines`.

WHY THIS EXISTS. mutmut has no "only these lines" CLI flag, and without
one every run mutates whole files: ~10000 mutants over `hopai/`, of
which a small PR's diff accounts for a few dozen. Checking a mutant
reruns the test suite, so the rest is not a slower report, it is a run
that never finishes -- CI kept hitting its wall-clock budget with
thousands of mutants unchecked, and the report had to lead with "no
evidence" instead of a score.

mutmut DOES have the machinery, one layer down, because
`mutate_only_covered_lines` needs exactly the same thing:

    mutmut._covered_lines           {abs path under mutants/: {line, ...}}
    store_lines_covered_by_tests()  populates it, before create_mutants()
    _should_mutate_node()           skips any node whose start line is
                                    not in that file's set
    get_covered_lines_for_file()    returns an EMPTY set for a file the
                                    map does not mention -- so a file
                                    left out gets no mutants at all

So the narrowing is a set intersection on that map, applied in the
window between the two calls, which is what wrapping
`store_lines_covered_by_tests` gets us.

That is mutmut's internals, not its API, so `_assert_shape()` below
checks every attribute this depends on and refuses with the version in
the message rather than silently mutating whole files again -- a silent
fallback here reads as "mutation ran and found nothing", which is the
one report this must never produce. The MUTANT_NAMES arguments are the
supported CLI's own filter and stay in place as a second gate: they
bound what gets CHECKED even if a `mutants/` tree survives from an
earlier run with a wider scope.
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

# mutmut is the `mutation` extra, not `dev`. Imported inside the functions
# that need it so that restrict_to()/narrow() -- pure set arithmetic, and
# where the subtle mistakes live -- stay importable, and testable, in the
# test matrix that does not install it.


def _assert_shape() -> None:
    """Refuse rather than silently mutate everything if mutmut moved the
    internals this depends on."""
    import mutmut
    import mutmut.__main__ as mutmut_main
    from mutmut.code_coverage import get_covered_lines_for_file

    version = getattr(mutmut, "__version__", "unknown")
    problems = [
        name
        for obj, name in (
            (mutmut, "_covered_lines"),
            (mutmut_main, "store_lines_covered_by_tests"),
            (mutmut_main, "_run"),
        )
        if not hasattr(obj, name)
    ]
    # The key format is the part most likely to drift silently, and the
    # one whose failure looks like success: a map mutmut cannot match
    # yields an empty set per file, i.e. zero mutants and a green report.
    # Ask mutmut's own reader what it makes of a key we built.
    probe = "hopai/_mutation_scope_probe.py"
    if get_covered_lines_for_file(probe, restrict_to({probe: [7]})) != {7}:
        problems.append("get_covered_lines_for_file key format")
    if problems:
        raise SystemExit(
            f"scripts/mutation_run.py drives mutmut internals that mutmut {version} "
            f"no longer matches: {', '.join(problems)}. Restricting mutation to the "
            f"changed lines is not possible through the public CLI, so this refuses "
            f"rather than mutate whole files and report the result as a clean sweep. "
            f"Re-check mutmut's code_coverage.py/__main__.py and update the wrapper."
        )


def restrict_to(changed: dict[str, list[int]]) -> dict[str, set[int]]:
    """The changed-lines map keyed the way mutmut keys `_covered_lines` --
    absolute, and under the `mutants/` tree it copies sources into."""
    return {
        str((Path("mutants") / path).absolute()): set(lines)
        for path, lines in changed.items()
    }


def covered_lines_from_coverage_xml(xml_path: str, paths: list[str]) -> dict[str, set[int]]:
    """`{path: covered lines}` for `paths`, read off a coverage.xml.

    The point is what this AVOIDS. mutmut's own
    `store_lines_covered_by_tests()` gets the same answer by running the
    entire suite again under `coverage` -- and the `quality` job already
    ran it, with `--cov-report=xml`, one step earlier. On a suite this
    size that duplicate run was a large part of the fixed cost mutmut
    paid before checking a single mutant, which is how a budget got spent
    with nothing to show.

    coverage.xml names each file relative to its `<source>` root
    (`hopai/core.py` is `filename="core.py"` under
    `<source>/abs/path/hopai</source>`), so the two have to be rejoined
    before anything matches.
    """
    root = ElementTree.parse(xml_path).getroot()
    sources = [source.text or "" for source in root.findall("sources/source")]
    wanted = {os.path.normpath(path) for path in paths}
    covered: dict[str, set[int]] = {}
    for element in root.iter("class"):
        filename = element.get("filename")
        if not filename:
            continue
        for source in sources or [""]:
            relative = os.path.normpath(os.path.relpath(os.path.join(source, filename), os.getcwd()))
            if relative in wanted:
                lines = {
                    int(line.get("number", 0))
                    for line in element.iter("line")
                    if int(line.get("hits", 0)) > 0
                }
                covered.setdefault(relative, set()).update(lines)
                break
    missing = sorted(wanted - set(covered))
    if missing:
        # Not fatal -- a genuinely untested new file looks like this --
        # but silent-zero is the failure mode that reads as a clean
        # sweep, so it gets said out loud.
        print(f"warning: no coverage rows for {', '.join(missing)}; "
              f"mutants on their changed lines will not be generated", file=sys.stderr)
    return covered


def narrow(covered: dict[str, set[int]] | None, restrict: dict[str, set[int]]) -> dict[str, set[int]]:
    """Intersect mutmut's covered-lines map with the changed lines.

    `covered is None` means `mutate_only_covered_lines` is off, so the
    changed lines stand alone -- an uncovered changed line then gets a
    mutant that no test can kill, which is that setting's own trade-off,
    not one to re-decide here. With it on, the intersection is the honest
    scope: a line that is changed AND exercised is the only one where a
    surviving mutant says something about this PR.
    """
    if covered is None:
        return restrict
    return {path: lines & restrict.get(path, set()) for path, lines in covered.items()}


def main(argv: list[str]) -> int:
    args = argv[1:]
    coverage_xml: str | None = None
    if len(args) >= 2 and args[0] == "--coverage-xml":
        coverage_xml = args[1]
        args = args[2:]
    if not args:
        print("usage: mutation_run.py [--coverage-xml FILE] <changed-lines.json> [MUTANT_NAME ...]",
              file=sys.stderr)
        return 2
    _assert_shape()

    import mutmut
    import mutmut.__main__ as mutmut_main

    changed = json.loads(Path(args[0]).read_text())
    mutant_names = args[1:]
    restrict = restrict_to(changed)

    original = mutmut_main.store_lines_covered_by_tests

    def store_lines_covered_by_tests() -> None:
        if coverage_xml:
            # Deliberately NOT calling `original()`: it would run the whole
            # suite again under coverage to learn what the `quality` job's
            # own --cov run already wrote to this file.
            covered = restrict_to(
                {path: sorted(lines) for path, lines
                 in covered_lines_from_coverage_xml(coverage_xml, list(changed)).items()}
            )
        else:
            original()
            covered = mutmut._covered_lines
        mutmut._covered_lines = narrow(covered, restrict)
        total = sum(len(lines) for lines in mutmut._covered_lines.values())
        print(f"mutating {total} changed, covered line(s) across {len(changed)} file(s)",
              file=sys.stderr)

    mutmut_main.store_lines_covered_by_tests = store_lines_covered_by_tests
    mutmut_main._run(mutant_names, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
