"""
Tests for the mutmut scope resolver.

It decides which mutants CI actually checks -- too narrow and a real
regression never gets a mutant to catch it, too wide and the changed-line
scoping (the whole point) does nothing. Both failure modes are silent
from inside a PR, so they are pinned here with a real git repo rather
than discovered on one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mutation_run import narrow, restrict_to  # noqa: E402
from mutation_scope import changed_lines_by_file, resolve_scope  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout


@pytest.fixture()
def repo(tmp_path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "test@example.com")
    _git(r, "config", "user.name", "Test")
    return r


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


class TestFunctionScoping:
    def test_a_body_change_is_scoped_to_that_function_only(self, repo):
        (repo / "pkg.py").write_text(
            "def alpha():\n"
            "    return 1\n"
            "\n"
            "\n"
            "def beta():\n"
            "    return 2\n"
        )
        base = _commit(repo, "base")
        (repo / "pkg.py").write_text(
            "def alpha():\n"
            "    return 1\n"
            "\n"
            "\n"
            "def beta():\n"
            "    return 20\n"
        )
        head = _commit(repo, "touch beta")

        patterns = resolve_scope(base, head, ["pkg.py"], cwd=str(repo))

        assert patterns == ["pkg.x_beta__mutmut_*"]

    def test_a_method_change_uses_the_class_name_separator(self, repo):
        (repo / "pkg.py").write_text(
            "class Widget:\n"
            "    def spin(self):\n"
            "        return 1\n"
        )
        base = _commit(repo, "base")
        (repo / "pkg.py").write_text(
            "class Widget:\n"
            "    def spin(self):\n"
            "        return 2\n"
        )
        head = _commit(repo, "touch spin")

        patterns = resolve_scope(base, head, ["pkg.py"], cwd=str(repo))

        assert patterns == ["pkg.xǁWidgetǁspin__mutmut_*"]

    def test_a_docstring_only_change_outside_any_function_yields_nothing(self, repo):
        (repo / "pkg.py").write_text(
            '"""Old docstring."""\n'
            "\n"
            "def alpha():\n"
            "    return 1\n"
        )
        base = _commit(repo, "base")
        (repo / "pkg.py").write_text(
            '"""New docstring."""\n'
            "\n"
            "def alpha():\n"
            "    return 1\n"
        )
        head = _commit(repo, "touch module docstring")

        assert resolve_scope(base, head, ["pkg.py"], cwd=str(repo)) == []

    def test_a_new_function_is_scoped_like_any_other_change(self, repo):
        (repo / "pkg.py").write_text("def alpha():\n    return 1\n")
        base = _commit(repo, "base")
        (repo / "pkg.py").write_text(
            "def alpha():\n"
            "    return 1\n"
            "\n"
            "\n"
            "def gamma():\n"
            "    return 3\n"
        )
        head = _commit(repo, "add gamma")

        assert resolve_scope(base, head, ["pkg.py"], cwd=str(repo)) == ["pkg.x_gamma__mutmut_*"]

    def test_patterns_from_multiple_files_are_combined_and_deduplicated(self, repo):
        (repo / "a.py").write_text("def alpha():\n    return 1\n")
        (repo / "b.py").write_text("def alpha():\n    return 1\n")
        base = _commit(repo, "base")
        (repo / "a.py").write_text("def alpha():\n    return 2\n")
        (repo / "b.py").write_text("def alpha():\n    return 2\n")
        head = _commit(repo, "touch both")

        patterns = resolve_scope(base, head, ["a.py", "b.py"], cwd=str(repo))

        assert patterns == ["a.x_alpha__mutmut_*", "b.x_alpha__mutmut_*"]

    def test_module_path_uses_dotted_form_not_the_file_path(self, repo):
        (repo / "sub").mkdir()
        (repo / "sub" / "mod.py").write_text("def alpha():\n    return 1\n")
        base = _commit(repo, "base")
        (repo / "sub" / "mod.py").write_text("def alpha():\n    return 2\n")
        head = _commit(repo, "touch")

        assert resolve_scope(base, head, ["sub/mod.py"], cwd=str(repo)) == ["sub.mod.x_alpha__mutmut_*"]


class TestChangedLines:
    """The per-line map is the one that bounds how many mutants get
    CREATED, so an over-wide answer here is the whole cost back."""

    def test_only_the_touched_line_is_reported(self, repo):
        (repo / "pkg.py").write_text("a = 1\nb = 2\nc = 3\n")
        base = _commit(repo, "base")
        (repo / "pkg.py").write_text("a = 1\nb = 20\nc = 3\n")
        head = _commit(repo, "touch line 2")

        assert changed_lines_by_file(base, head, ["pkg.py"], cwd=str(repo)) == {"pkg.py": [2]}

    def test_a_file_the_diff_left_unchanged_is_absent_entirely(self, repo):
        """Present-with-empty-list would mutate nothing, but absent is what
        mutmut reads as 'no mutants for this file' -- keep them the same."""
        (repo / "a.py").write_text("x = 1\n")
        (repo / "b.py").write_text("y = 1\n")
        base = _commit(repo, "base")
        (repo / "a.py").write_text("x = 2\n")
        head = _commit(repo, "touch a only")

        assert changed_lines_by_file(base, head, ["a.py", "b.py"], cwd=str(repo)) == {"a.py": [1]}


class TestCoveredLinesNarrowing:
    """mutmut skips any node whose start line is missing from this map, and
    hands back an empty set for a file the map omits -- so getting the
    intersection wrong silently mutates everything or nothing."""

    def test_the_intersection_keeps_only_lines_that_are_both_covered_and_changed(self):
        covered = {"/w/mutants/pkg.py": {1, 2, 3, 4}}
        restrict = {"/w/mutants/pkg.py": {3, 4, 99}}

        assert narrow(covered, restrict) == {"/w/mutants/pkg.py": {3, 4}}

    def test_a_covered_file_the_diff_never_touched_is_emptied_not_dropped(self):
        """Dropping the key makes get_covered_lines_for_file return an empty
        set anyway, but only because it defaults that way -- pin the value."""
        covered = {"/w/mutants/touched.py": {1}, "/w/mutants/other.py": {1, 2}}

        assert narrow(covered, {"/w/mutants/touched.py": {1}}) == {
            "/w/mutants/touched.py": {1},
            "/w/mutants/other.py": set(),
        }

    def test_no_coverage_map_means_the_changed_lines_stand_alone(self):
        """mutate_only_covered_lines off leaves _covered_lines None; the
        changed lines are then the whole restriction, not a no-op."""
        restrict = {"/w/mutants/pkg.py": {5}}

        assert narrow(None, restrict) == restrict

    def test_keys_are_absolute_and_under_the_mutants_tree(self):
        """mutmut looks the map up by the path it copied the source to, not
        the repo-relative one -- a mismatch yields zero mutants and a report
        that reads like a clean sweep."""
        key = next(iter(restrict_to({"hopai/core.py": [1]})))

        assert key == str((Path("mutants") / "hopai/core.py").absolute())


class TestMutmutInternalsStillMatch:
    """Restricting mutation to changed lines drives mutmut internals, not
    its CLI. If an upgrade moves them, this must fail HERE -- the failure
    mode in CI is a run that mutates nothing and reports a clean sweep."""

    def test_the_shape_guard_passes_against_the_installed_mutmut(self):
        pytest.importorskip("mutmut", reason="mutmut is the [mutation] extra, not [dev]")
        from mutation_run import _assert_shape

        _assert_shape()  # raises SystemExit naming what moved

    def test_mutmut_reads_back_a_key_this_wrapper_writes(self):
        """The one assumption whose breakage looks like success: mutmut's
        own reader must find the lines under the key restrict_to() builds."""
        pytest.importorskip("mutmut", reason="mutmut is the [mutation] extra, not [dev]")
        from mutmut.code_coverage import get_covered_lines_for_file

        path = "hopai/core.py"

        assert get_covered_lines_for_file(path, restrict_to({path: [4, 9]})) == {4, 9}

    def test_a_file_absent_from_the_map_gets_no_lines(self):
        """This is what makes an untouched file generate zero mutants."""
        pytest.importorskip("mutmut", reason="mutmut is the [mutation] extra, not [dev]")
        from mutmut.code_coverage import get_covered_lines_for_file

        assert get_covered_lines_for_file("hopai/untouched.py", restrict_to({"hopai/core.py": [1]})) == set()
