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

from mutation_scope import resolve_scope  # noqa: E402


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
