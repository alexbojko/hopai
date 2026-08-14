#!/usr/bin/env python3
"""
Render the sticky PR comment: line coverage, and the mutation report
beside it.

    python scripts/pr_report.py \
        --coverage-xml coverage.xml \
        --threshold 85 \
        --mutmut-stats mutants/mutmut-cicd-stats.json \
        --mutmut-results survivors.txt \
        --mutation-note "3 changed files" \
        --run-url https://... > comment.md

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


def render_mutation(stats: dict, found: list, note: str, attempted: bool = False) -> str:
    """Mutation half of the comment. `note` explains the scope, or why
    there was nothing to run.

    `attempted` separates the two ways there can be no numbers, which
    must never read alike: nothing to mutate (fine) versus mutmut ran and
    produced nothing (a broken harness). The second is the case the
    triage rule calls out -- it looks like a clean sweep and is the
    opposite."""
    if stats is None:
        if attempted:
            return "\n".join([
                "### ⚠️ Mutation testing — HARNESS FAILED", "",
                f"{note}  mutmut ran but produced no results, so **nothing was "
                f"checked**. This is not a clean sweep: the usual cause is a file "
                f"the suite reads that `also_copy` does not put in the mutants "
                f"tree, which fails the baseline run before any mutant is tried. "
                f"See the job log.",
            ])
        return "\n".join(["### 🧬 Mutation testing — not run", "", note])

    total, killed = stats.get("total", 0), stats.get("killed", 0)
    score = (killed / total * 100) if total else 0.0
    lines = [
        f"### 🧬 Mutation score {score:.0f}%  ({killed}/{total} killed)",
        "",
        f"{note}  Advisory — this never blocks the merge, "
        f"but every survivor needs triage (see CLAUDE.md).",
    ]

    odd = {name: stats.get(name, 0) for name in
           ("timeout", "suspicious", "no_tests", "skipped", "segfault")}
    odd = {name: count for name, count in odd.items() if count}
    if odd:
        lines += ["", "Not a verdict either way: "
                  + ", ".join(f"**{count} {name}**" for name, count in odd.items())
                  + "."]

    if found:
        lines += ["", f"<details><summary>{len(found)} survivor(s) — "
                  f"each is a change no test objected to</summary>", "",
                  "| Mutant | Status |", "| --- | --- |"]
        lines += [f"| `{name}` | {status} |" for name, status in found[:40]]
        if len(found) > 40:
            lines.append(f"| … and {len(found) - 40} more | |")
        lines += ["", "Inspect one with `mutmut show <mutant>`.", "", "</details>"]
    elif total:
        lines += ["", "No survivors — every mutant on the changed lines was caught."]
    return "\n".join(lines)


def render(percent, threshold, by_file, stats, found, note, run_url,
           attempted: bool = False) -> str:
    parts = [MARKER, "## Test quality report", "",
             render_coverage(percent, threshold, by_file), "",
             render_mutation(stats, found, note, attempted)]
    if run_url:
        parts += ["", f"<sub>[CI run]({run_url})</sub>"]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-xml", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=85.0)
    parser.add_argument("--mutmut-stats", type=Path)
    parser.add_argument("--mutmut-results", type=Path)
    parser.add_argument("--mutation-note", default="")
    parser.add_argument("--mutation-attempted", action="store_true",
                        help="mutmut was run; missing results therefore mean a broken "
                             "harness rather than an empty scope")
    parser.add_argument("--run-url", default="")
    args = parser.parse_args()

    stats = (mutation_stats(args.mutmut_stats)
             if args.mutmut_stats and args.mutmut_stats.is_file() else None)
    found = (survivors(args.mutmut_results.read_text())
             if args.mutmut_results and args.mutmut_results.is_file() else [])

    print(render(coverage_percent(args.coverage_xml), args.threshold,
                 coverage_by_file(args.coverage_xml), stats, found,
                 args.mutation_note, args.run_url, args.mutation_attempted), end="")


if __name__ == "__main__":
    main()
