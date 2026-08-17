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
    MARKER, MAX_ROWS, coverage_by_file, coverage_percent, mutation_changes, render,
    render_coverage, render_mutation, survivors,
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

    def test_every_survivor_is_listed(self):
        """The list used to stop at 40. Triage needs all of them -- the
        table is folded, so length costs nothing to a reader who is not
        triaging, and a hidden survivor is one nobody sorts into gap,
        equivalent or out of scope. Only the comment's size limit caps
        it now, and that says what it dropped."""
        found = [(f"m{i}", "survived") for i in range(60)]
        body = render_mutation({"total": 60, "killed": 0}, found, "scope")
        assert "| `m59` | survived |" in body
        assert "not shown" not in body

    def test_no_survivors_says_so(self):
        body = render_mutation({"total": 5, "killed": 5}, [], "scope")
        assert "No survivors" in body

    def test_zero_killed_and_zero_survivors_is_not_a_clean_sweep(self):
        """The state a budget-killed run exports: a `total`, and every
        count at zero. It used to render as "every mutant on the changed
        lines was caught" -- a reader takes that as evidence the tests
        are strong, when in fact nothing ran. A harness that stopped and
        a suite that caught everything must never look alike."""
        body = render_mutation({"total": 2145, "killed": 0}, [], "scope")
        assert "No survivors" not in body
        assert "2145 of 2145 mutant(s) never reached a verdict" in body
        assert "no evidence" in body

    def test_a_partial_run_reports_the_remainder(self):
        """A budget that expires PART way through is the common case, and
        the mutants it did check are real results -- so the survivors
        still list, and only the unchecked remainder is disclaimed."""
        body = render_mutation({"total": 10, "killed": 3}, [("m1", "survived")], "scope")
        assert "6 of 10 mutant(s) never reached a verdict" in body
        assert "| `m1` | survived |" in body

    def test_a_complete_run_carries_no_warning(self):
        """Every mutant accounted for as killed, survived or inconclusive
        means the run finished; a warning there would cry wolf on every
        healthy PR."""
        body = render_mutation({"total": 5, "killed": 3, "timeout": 1}, [("m", "survived")],
                               "scope")
        assert "never reached a verdict" not in body

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

    def test_a_spent_budget_is_not_reported_as_a_broken_tree(self):
        """Zero checked has two causes needing OPPOSITE fixes, and the
        message used to assert one of them: a reader whose budget ran
        out was sent hunting for a missing `also_copy` entry that was
        never missing. Before the first mutant mutmut runs the suite for
        a baseline and again under tracing, and that fixed cost grows
        with the suite -- so this is the one that arrives as a project
        succeeds."""
        body = render_mutation(None, [], "7 changed files", attempted=True,
                               timed_out="900")
        assert "BUDGET SPENT" in body
        assert "900s wall-clock budget" in body
        assert "MUTATION_BUDGET_SECONDS" in body
        # The other diagnosis must be absent, not merely outweighed.
        assert "also_copy" not in body

    def test_a_broken_tree_still_says_also_copy(self):
        """And the reverse: no budget was hit, so the missing-file
        diagnosis is the right one and must survive."""
        body = render_mutation(None, [], "7 changed files", attempted=True)
        assert "also_copy" in body
        assert "BUDGET SPENT" not in body
        assert "was not stopped by its budget" in body

    def test_zero_mutants_does_not_divide_by_zero(self):
        assert "0%" in render_mutation({"total": 0, "killed": 0}, [], "scope")


SHOW_OUTPUT = """\
# hopai.mutate.xǁMutatorǁ_guard__mutmut_8: survived
--- hopai/mutate.py
+++ hopai/mutate.py
@@ -6,7 +6,7 @@
         them is.\"\"\"
-    _flag(all, call, "all")
+    _flag(all, call, "ALL")
     unfiltered = all_(self._blank(f) for f in filters)
# hopai.mutate.x_merge__mutmut_1: survived
--- hopai/mutate.py
+++ hopai/mutate.py
@@ -1 +1 @@
-        value = value.op("||")(incoming)
+        value = value.op("&&")(incoming)
"""


class TestMutationChanges:
    """The diff is what makes a survivor triageable in the comment
    instead of at a checkout."""

    def test_pairs_each_mutant_with_its_edit(self):
        changes = mutation_changes(SHOW_OUTPUT)
        assert changes["hopai.mutate.xǁMutatorǁ_guard__mutmut_8"] == (
            '_flag(all, call, "all")', '_flag(all, call, "ALL")')

    def test_file_headers_are_not_read_as_the_change(self):
        """`--- hopai/mutate.py` and `+++ hopai/mutate.py` start with the
        diff markers and are not the mutation; taking them would make
        every row read as "the file changed into itself"."""
        removed, added = mutation_changes(SHOW_OUTPUT)["hopai.mutate.x_merge__mutmut_1"]
        assert "hopai/mutate.py" not in removed + added
        assert removed == 'value = value.op("||")(incoming)'

    def test_a_multi_line_hunk_keeps_every_line(self):
        """A mutation spanning two lines rendered as one of them would
        be a diff that does not say what changed."""
        text = ("# m: survived\n--- a.py\n+++ a.py\n@@ -1,2 +1,2 @@\n"
                "-one\n-two\n+uno\n+dos\n")
        assert mutation_changes(text)["m"] == ("one ⏎ two", "uno ⏎ dos")

    def test_output_for_no_mutants_is_empty_not_an_error(self):
        assert mutation_changes("") == {}


class TestSurvivorTable:
    def test_the_change_column_appears_when_diffs_are_available(self):
        body = render_mutation({"total": 10, "killed": 9},
                               [("hopai.mutate.xǁMutatorǁ_guard__mutmut_8", "survived")],
                               "note", changes=mutation_changes(SHOW_OUTPUT))
        assert "| Mutant | Status | Change |" in body
        assert '`_flag(all, call, "all")` → `_flag(all, call, "ALL")`' in body

    def test_it_stays_folded_and_two_columns_without_diffs(self):
        """The diffs are best-effort -- `mutmut show` failing must cost
        the column, not the report."""
        body = render_mutation({"total": 10, "killed": 9},
                               [("m1", "survived")], "note")
        assert "<details><summary>" in body and "<details open" not in body
        assert "| Mutant | Status |" in body and "Change" not in body

    def test_a_pipe_in_the_source_cannot_break_the_table(self):
        """`properties || incoming` is ordinary code here, and an
        unescaped pipe ends the cell early -- the row after it silently
        loses its columns."""
        body = render_mutation({"total": 2, "killed": 1},
                               [("hopai.mutate.x_merge__mutmut_1", "survived")],
                               "note", changes=mutation_changes(SHOW_OUTPUT))
        row = next(line for line in body.splitlines() if line.startswith("| `hopai"))
        assert row.count("|") - row.count("\\|") == 4      # the four real delimiters
        assert "\\|\\|" in row

    def test_a_mutant_with_no_diff_still_gets_a_row(self):
        body = render_mutation({"total": 2, "killed": 1}, [("m1", "timeout")],
                               "note", changes={"other": ("a", "b")})
        assert "| `m1` | timeout | — |" in body

    def test_a_long_change_is_bounded_and_marked(self):
        long = "x" * 400
        body = render_mutation({"total": 2, "killed": 1}, [("m1", "survived")],
                               "note", changes={"m1": (long, long)})
        row = next(line for line in body.splitlines() if line.startswith("| `m1`"))
        assert "…" in row and len(row) < 300

    def test_a_capped_table_says_what_it_dropped(self):
        """A truncated table that looked complete would read as "these
        are all the survivors" -- the one thing the report must not
        imply."""
        found = [(f"m{i}", "survived") for i in range(MAX_ROWS + 20)]
        body = render_mutation({"total": 999, "killed": 1}, found, "note",
                               changes={"m0": ("a", "b")})
        assert "and 20 more, not shown" in body
        assert f"{len(found)} survivor(s)" in body      # the count is still honest

    def test_the_comment_stays_under_the_github_limit(self):
        """GitHub rejects a body over 65536 characters, and this table is
        the only part that grows with the diff."""
        found = [(f"hopai.module.x_function_with_a_long_name__mutmut_{i}", "survived")
                 for i in range(400)]
        changes = {name: ("a" * 200, "b" * 200) for name, _ in found}
        body = render_mutation({"total": 999, "killed": 1}, found, "note", changes=changes)
        assert len(body) < 65_000


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
