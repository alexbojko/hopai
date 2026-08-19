#!/usr/bin/env python3
"""
Render the sticky PR comment: line coverage, and the mutation report
beside it.

    python scripts/pr_report.py \
        --coverage-xml coverage.xml \
        --threshold 85 \
        --mutmut-stats mutants/mutmut-cicd-stats.json \
        --mutmut-results survivors.txt \
        --mutmut-diffs diffs.txt \
        --mutation-note "3 changed files" \
        --run-url https://... > comment.md

`--mutmut-diffs` is the concatenated output of `mutmut show <id>` for
every survivor, which puts each mutation IN the comment: a survivor's
name says where to look, its diff says whether it is a real gap, an
equivalent mutant or a message nobody pinned. Without it triage starts
by running `mutmut show` forty times, which is the job the comment
exists to save. `--list-survivors` prints the ids to feed it, so which
mutants matter is decided here and not re-implemented in the workflow.

Two numbers that answer different questions, which is why they share a
comment rather than a job:

  coverage   how much of the code the tests EXECUTE. It gates the PR at
             --threshold, because a line no test runs is a line nobody
             has checked at all.
  mutation   how much of the code the tests actually PIN. A surviving
             mutant is a change to the source that no test objected to,
             which is the part coverage cannot see -- a test can execute
             a line and assert nothing about it. Never a gate: mutmut is
             slow, partial on a time budget, and produces equivalent
             mutants no test could ever kill. Advisory does not mean
             ignorable; see CLAUDE.md for the triage rule.

Kept a script rather than a heredoc in the workflow so it can be read,
run locally and tested -- see tests/test_pr_report.py.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path

#: First line of the comment. The workflow finds its previous comment by
#: this exact prefix and PATCHes it, so one PR keeps one comment instead
#: of a new one per push. Changing it orphans every existing comment.
MARKER = "<!-- hopai-pr-report -->"

_SURVIVOR_RE = re.compile(r"^\s*(?P<id>[\w.]+):\s*(?P<status>\w+)\s*$")

#: Each `mutmut show` block opens with `# <id>: <status>`.
_SHOW_HEADER_RE = re.compile(r"^#\s*(?P<id>\S+):\s*\w")

#: Per side of one change. Bounded because this lands in a table cell --
#: enough to recognize the edit, and the mutant id is there for the
#: reader who needs the whole hunk.
CHANGE_LIMIT = 90
#: Rows, and a character budget: GitHub rejects a comment body over
#: 65536 characters and this table is the only part that grows with the
#: size of the diff. Whatever is dropped is always named.
MAX_ROWS = 200
TABLE_BUDGET = 40_000


def coverage_percent(xml_path: Path) -> float:
    """Overall line coverage, as a percentage."""
    root = ElementTree.parse(xml_path).getroot()
    return float(root.attrib["line-rate"]) * 100


def coverage_by_file(xml_path: Path) -> list:
    """[(filename, percent, uncovered_lines), ...], worst first."""
    root = ElementTree.parse(xml_path).getroot()
    rows = []
    for element in root.iter("class"):
        lines = element.find("lines")
        missing = sum(1 for line in (lines if lines is not None else [])
                      if line.attrib.get("hits") == "0")
        rows.append((element.attrib["filename"],
                     float(element.attrib["line-rate"]) * 100, missing))
    return sorted(rows, key=lambda row: (row[1], -row[2]))


def mutation_stats(path: Path) -> dict:
    return json.loads(path.read_text())


def survivors(text: str) -> list:
    """Parse `mutmut results`, which prints one indented
    `module.func__mutmut_N: status` line per mutant it did not kill."""
    found = []
    for line in text.splitlines():
        match = _SURVIVOR_RE.match(line)
        if match and match.group("status") != "killed":
            found.append((match.group("id"), match.group("status")))
    return found


def mutation_changes(text: str) -> dict:
    """{mutant id: (removed, added)} from concatenated `mutmut show`
    output.

    Only the -/+ lines are kept: the context lines are the same source
    the reader already has, and the removed/added pair IS the mutation.
    A hunk that changes several lines joins them, so a multi-line
    mutation never renders as a partial one."""
    changes: dict = {}
    current, removed, added = None, [], []

    def flush() -> None:
        if current is not None and (removed or added):
            changes[current] = (" ⏎ ".join(removed), " ⏎ ".join(added))

    for line in text.splitlines():
        header = _SHOW_HEADER_RE.match(line)
        if header:
            flush()
            current, removed, added = header.group("id"), [], []
        elif line.startswith(("---", "+++", "@@")):
            continue                      # file headers and hunk ranges
        elif line.startswith("-"):
            removed.append(line[1:].strip())
        elif line.startswith("+"):
            added.append(line[1:].strip())
    flush()
    return changes


def _cell(text: str, limit: int = CHANGE_LIMIT) -> str:
    """Source as a table cell: one line, bounded, and unable to break
    the table.

    An unescaped `|` ends the cell early -- and `properties || incoming`
    is ordinary code here, so this is not hypothetical. Backticked
    content falls back to <code>, since a code span cannot contain the
    character that delimits it."""
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[:limit - 1] + "…"
    if "`" in flat:
        flat = (flat.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace("`", "&#96;").replace("|", "&#124;"))
        return f"<code>{flat}</code>"
    return "`" + flat.replace("|", "\\|") + "`"


def _survivor_rows(found: list, changes: dict) -> list:
    """The table body, plus the line naming anything left out."""
    header = (["| Mutant | Status | Change |", "| --- | --- | --- |"] if changes
              else ["| Mutant | Status |", "| --- | --- |"])
    rows, budget = [], TABLE_BUDGET
    for index, (name, status) in enumerate(found):
        if changes:
            removed, added = changes.get(name, ("", ""))
            change = (f"{_cell(removed)} → {_cell(added)}" if removed or added
                      else "—")
            row = f"| `{name}` | {status} | {change} |"
        else:
            row = f"| `{name}` | {status} |"
        budget -= len(row)
        if index >= MAX_ROWS or budget <= 0:
            # Never a silent cap: a truncated table that looks complete
            # would read as "these are all the survivors".
            rows.append(f"| … and {len(found) - index} more, not shown "
                        f"(comment size limit) | | |" if changes
                        else f"| … and {len(found) - index} more, not shown "
                             f"(comment size limit) | |")
            break
        rows.append(row)
    return header + rows


def _bar(percent: float, width: int = 20) -> str:
    filled = round(percent / 100 * width)
    return "█" * filled + "░" * (width - filled)


def render_coverage(percent: float, threshold: float, by_file: list) -> str:
    verdict = "✅" if percent >= threshold else "❌"
    lines = [
        f"### {verdict} Coverage {percent:.1f}%  `{_bar(percent)}`",
        "",
        f"Required: **{threshold:.0f}%** — "
        + ("passing." if percent >= threshold
           else f"**{threshold - percent:.1f} points short; this PR is blocked.**"),
    ]
    weakest = [row for row in by_file if row[1] < 100][:5]
    if weakest:
        lines += ["", "<details><summary>Least-covered files</summary>", "",
                  "| File | Coverage | Uncovered lines |", "| --- | ---: | ---: |"]
        lines += [f"| `{name}` | {pct:.0f}% | {missing} |" for name, pct, missing in weakest]
        lines += ["", "</details>"]
    return "\n".join(lines)


def render_mutation(stats: dict, found: list, note: str, attempted: bool = False,
                    changes: dict = None, timed_out: str = "") -> str:
    """Mutation half of the comment. `note` explains the scope, or why
    there was nothing to run.

    `attempted` separates the two ways there can be no numbers, which
    must never read alike: nothing to mutate (fine) versus mutmut ran and
    produced nothing (a broken harness). The second is the case the
    triage rule calls out -- it looks like a clean sweep and is the
    opposite.

    `timed_out` splits that second case again, because the two halves
    need OPPOSITE fixes and the message used to assert one of them. A
    budget spent before the first mutant means the run needs longer (or
    a cheaper baseline); a mutants tree that cannot run the suite means
    a file is missing from `also_copy`. Guessing sends the reader to
    the wrong one, which is how a broken harness stays broken."""
    if stats is None:
        if attempted and timed_out:
            return "\n".join([
                "### ⚠️ Mutation testing — BUDGET SPENT BEFORE ANY MUTANT", "",
                f"{note}  mutmut was stopped by its {timed_out}s wall-clock budget "
                f"having checked **nothing**. This is not a clean sweep, and it is "
                f"not a broken mutants tree either: before the first mutant, mutmut "
                f"runs the suite to establish a baseline and again under tracing to "
                f"map tests to lines, and that fixed cost grows with the suite. "
                f"Raise `MUTATION_BUDGET_SECONDS`, or make the baseline cheaper.",
            ])
        if attempted:
            return "\n".join([
                "### ⚠️ Mutation testing — HARNESS FAILED", "",
                f"{note}  mutmut ran, was not stopped by its budget, and still "
                f"produced no results, so **nothing was checked**. This is not a "
                f"clean sweep: the usual cause is a file the suite reads that "
                f"`also_copy` does not put in the mutants tree, which fails the "
                f"baseline run before any mutant is tried. See the job log.",
            ])
        return "\n".join(["### 🧬 Mutation testing — not run", "", note])

    total, killed = stats.get("total", 0), stats.get("killed", 0)

    odd = {name: stats.get(name, 0) for name in
           ("timeout", "suspicious", "no_tests", "skipped", "segfault")}
    odd = {name: count for name, count in odd.items() if count}

    # Mutants mutmut generated but never reached a verdict on. A run
    # stopped by its budget exports a `total` with every count at zero,
    # and the difference is the whole story: "0 killed, 0 survived" is
    # not a clean sweep, it is nothing having run. Reporting it as one
    # ("every mutant was caught") is the same silently-different-answer
    # this project refuses everywhere else, and it is worse here because
    # a reader takes it as evidence the tests are strong.
    unchecked = total - killed - len(found) - sum(odd.values())

    if unchecked > 0:
        # A percentage computed only over the mutants that happened to
        # finish is not a mutation score -- "63% (4084/6477 killed)"
        # skims as "roughly passing" when a third of the scope (2121
        # mutants here) has no verdict at all, and two runs of the same
        # commit can print different percentages depending only on
        # where the harness stopped. So the warning IS the headline;
        # there is no percentage left to skim past it with.
        lines = [
            f"### ⚠️ Mutation testing incomplete — {unchecked} of {total} "
            f"mutant(s) never reached a verdict",
            "",
            f"{note}  {killed}/{total} killed so far. That is a harness that "
            f"stopped, not a suite that caught everything: treat this run as "
            f"**no evidence** about the unreached mutants rather than as a "
            f"pass. Advisory either way — this never blocks the merge, but "
            f"every survivor still needs triage (see CLAUDE.md). See the job log.",
        ]
    else:
        score = (killed / total * 100) if total else 0.0
        lines = [
            f"### 🧬 Mutation score {score:.0f}%  ({killed}/{total} killed)",
            "",
            f"{note}  Advisory — this never blocks the merge, "
            f"but every survivor needs triage (see CLAUDE.md).",
        ]

    if odd:
        lines += ["", "Not a verdict either way: "
                  + ", ".join(f"**{count} {name}**" for name, count in odd.items())
                  + "."]

    if found:
        # Folded by default: on a large PR this is the longest thing in
        # the comment, and it is reference material for whoever triages
        # rather than something every reader needs open.
        lines += ["", f"<details><summary>{len(found)} survivor(s) — "
                  f"each is a change no test objected to</summary>", ""]
        lines += _survivor_rows(found, changes or {})
        lines += ["", "Each row is the source before → after the mutation. "
                  "`mutmut show <mutant>` prints the surrounding hunk; "
                  "CLAUDE.md has the triage rule.", "", "</details>"]
    elif total and unchecked <= 0:
        lines += ["", "No survivors — every mutant on the changed lines was caught."]

    return "\n".join(lines)


def render(percent, threshold, by_file, stats, found, note, run_url,
           attempted: bool = False, changes: dict = None,
           timed_out: str = "") -> str:
    parts = [MARKER, "## Test quality report", "",
             render_coverage(percent, threshold, by_file), "",
             render_mutation(stats, found, note, attempted, changes, timed_out)]
    if run_url:
        parts += ["", f"<sub>[CI run]({run_url})</sub>"]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-xml", type=Path)
    parser.add_argument("--threshold", type=float, default=85.0)
    parser.add_argument("--mutmut-stats", type=Path)
    parser.add_argument("--mutmut-results", type=Path)
    parser.add_argument("--mutmut-diffs", type=Path,
                        help="concatenated `mutmut show <id>` output for the survivors; "
                             "without it the table falls back to names alone")
    parser.add_argument("--list-survivors", action="store_true",
                        help="print the survivor ids and exit -- what the workflow "
                             "loops over to collect the diffs")
    parser.add_argument("--mutation-note", default="")
    parser.add_argument("--mutation-attempted", action="store_true",
                        help="mutmut was run; missing results therefore mean a broken "
                             "harness rather than an empty scope")
    parser.add_argument("--mutation-timed-out", default="", metavar="SECONDS",
                        help="the wall-clock budget, when timeout(1) killed the run -- "
                             "a spent budget and a broken mutants tree both report zero "
                             "checked and need opposite fixes")
    parser.add_argument("--run-url", default="")
    args = parser.parse_args()

    found = (survivors(args.mutmut_results.read_text())
             if args.mutmut_results and args.mutmut_results.is_file() else [])
    if args.list_survivors:
        print("\n".join(name for name, _ in found))
        return

    stats = (mutation_stats(args.mutmut_stats)
             if args.mutmut_stats and args.mutmut_stats.is_file() else None)
    changes = (mutation_changes(args.mutmut_diffs.read_text())
               if args.mutmut_diffs and args.mutmut_diffs.is_file() else {})

    print(render(coverage_percent(args.coverage_xml), args.threshold,
                 coverage_by_file(args.coverage_xml), stats, found,
                 args.mutation_note, args.run_url, args.mutation_attempted,
                 changes, args.mutation_timed_out), end="")


if __name__ == "__main__":
    main()
