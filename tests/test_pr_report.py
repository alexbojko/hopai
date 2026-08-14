"""
Tests for the PR-comment renderer.

It runs in CI where nobody reads its output until it is wrong, and it
decides whether a PR reads as passing or blocked -- so the parsing and
the verdict are pinned here rather than discovered on a pull request.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pr_report import (  # noqa: E402
    MARKER, coverage_by_file, coverage_percent, render, render_coverage,
    render_mutation, survivors,
)

COVERAGE_XML = """<?xml version="1.0" ?>
<coverage line-rate="0.9123" lines-valid="100" lines-covered="91">
  <packages><package><classes>
    <class filename="hopai/core.py" line-rate="1.0">
      <lines><line number="1" hits="1"/></lines>
    </class>
    <class filename="hopai/cypher.py" line-rate="0.5">
      <lines><line number="1" hits="1"/><line number="2" hits="0"/></lines>
    </class>
  </classes></package></packages>
</coverage>
"""


@pytest.fixture()
def coverage_xml(tmp_path) -> Path:
    path = tmp_path / "coverage.xml"
    path.write_text(COVERAGE_XML)
    return path


class TestCoverageParsing:
    def test_percent_is_a_percentage_not_a_rate(self, coverage_xml):
        """coverage.xml stores 0.9123; a comment saying '0.9% covered'
        would read as catastrophic rather than fine."""
        assert coverage_percent(coverage_xml) == pytest.approx(91.23)

    def test_by_file_is_worst_first(self, coverage_xml):
        rows = coverage_by_file(coverage_xml)
        assert [name for name, _, _ in rows] == ["hopai/cypher.py", "hopai/core.py"]
        assert rows[0][2] == 1  # one uncovered line


class TestVerdict:
    def test_passing_above_threshold(self, coverage_xml):
        body = render_coverage(91.23, 85, [])
        assert "✅" in body and "passing" in body

    def test_blocked_below_threshold_says_by_how_much(self, coverage_xml):
        body = render_coverage(80.0, 85, [])
        assert "❌" in body and "5.0 points short" in body and "blocked" in body

    def test_exactly_at_threshold_passes(self):
        assert "✅" in render_coverage(85.0, 85, [])

    def test_weakest_files_are_listed_and_capped(self):
        rows = [(f"hopai/m{i}.py", float(i), i) for i in range(10)]
        body = render_coverage(50, 85, rows)
        assert body.count("| `hopai/m") == 5          # capped at five
        assert "hopai/m0.py" in body                   # worst included

    def test_fully_covered_files_are_not_listed(self):
        assert "<details>" not in render_coverage(100, 85, [("hopai/a.py", 100.0, 0)])


class TestSurvivorParsing:
    def test_parses_mutmut_results_lines(self):
        text = ("    hopai.hop.x__normalize_hops__mutmut_12: survived\n"
                "    hopai.core.x_build__mutmut_3: timeout\n")
        assert survivors(text) == [
            ("hopai.hop.x__normalize_hops__mutmut_12", "survived"),
            ("hopai.core.x_build__mutmut_3", "timeout"),
        ]

    def test_killed_mutants_are_not_survivors(self):
        assert survivors("    hopai.a.x_b__mutmut_1: killed\n") == []

    def test_noise_is_ignored(self):
        assert survivors("Running mutation testing\n\n1.5 mutations/second\n") == []


class TestMutationSection:
    def test_score_and_counts(self):
        body = render_mutation({"total": 13, "killed": 12, "survived": 1}, [], "scope")
        assert "92%" in body and "(12/13 killed)" in body

    def test_always_says_it_is_advisory(self):
        """The one thing a reader must not conclude is that a red
        mutation section blocks their merge."""
        body = render_mutation({"total": 2, "killed": 1}, [("m", "survived")], "scope")
        assert "never blocks" in body and "triage" in body

    def test_survivors_are_listed_and_capped(self):
        found = [(f"m{i}", "survived") for i in range(60)]
        body = render_mutation({"total": 60, "killed": 0}, found, "scope")
        assert "and 20 more" in body

    def test_no_survivors_says_so(self):
        body = render_mutation({"total": 5, "killed": 5}, [], "scope")
        assert "No survivors" in body

    def test_inconclusive_outcomes_are_called_out_separately(self):
        """A timeout or a segfault is neither a kill nor a survival, and
        folding it into the score would flatter or damn it wrongly."""
        body = render_mutation({"total": 3, "killed": 1, "timeout": 1, "segfault": 1},
                               [], "scope")
        assert "1 timeout" in body and "1 segfault" in body

    def test_not_run_explains_why(self):
        body = render_mutation(None, [], "no hopai/ files changed")
        assert "not run" in body and "no hopai/ files changed" in body

    def test_a_broken_harness_does_not_read_like_a_clean_sweep(self):
        """The two ways to have no numbers must not look alike. mutmut
        aborting before it checks anything once rendered as a mild
        'not run', which is the single most misleading thing this comment
        could say."""
        body = render_mutation(None, [], "4 changed files", attempted=True)
        assert "HARNESS FAILED" in body
        assert "nothing was checked" in body
        assert "not a clean sweep" in body

    def test_an_empty_scope_is_not_reported_as_a_failure(self):
        body = render_mutation(None, [], "nothing changed", attempted=False)
        assert "HARNESS FAILED" not in body

    def test_zero_mutants_does_not_divide_by_zero(self):
        assert "0%" in render_mutation({"total": 0, "killed": 0}, [], "scope")


class TestWholeComment:
    def test_starts_with_the_sticky_marker(self, coverage_xml):
        """The workflow finds its previous comment by this prefix. If it
        is not the first thing in the body, every push posts a new
        comment instead of updating one."""
        body = render(91.2, 85, [], {"total": 1, "killed": 1}, [], "scope", "http://run")
        assert body.startswith(MARKER)

    def test_contains_both_halves_and_the_run_link(self):
        body = render(91.2, 85, [], {"total": 1, "killed": 1}, [], "scope", "http://run")
        assert "Coverage" in body and "Mutation" in body and "http://run" in body

    def test_is_valid_json_payload_material(self):
        """The workflow posts the body as JSON; backticks, pipes and
        emoji must survive the round trip."""
        body = render(91.2, 85, [("hopai/a.py", 50.0, 3)],
                      {"total": 1, "killed": 0}, [("m|x", "survived")], "n", "")
        assert json.loads(json.dumps({"body": body}))["body"] == body
