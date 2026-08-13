"""
Packaging invariants.

The version is written in three places -- `pyproject.toml`,
`hopai.__version__` and `.release-please-manifest.json` -- and
release-please updates all three from its config. If that config ever
stops covering one of them, nothing fails: the release goes out, and the
installed package just reports a version it isn't. These tests are the
thing that fails instead.

No database, no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import hopai

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def pyproject_version() -> str:
    """Read it with a regex rather than tomllib: this project supports
    3.10, which has no tomllib in the standard library."""
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', text, flags=re.M)
    assert match, "pyproject.toml has no [project] version"
    return match.group(1)


def manifest_version() -> str:
    return json.loads((ROOT / ".release-please-manifest.json").read_text())["."]


class TestVersion:
    def test_module_matches_pyproject(self):
        assert hopai.__version__ == pyproject_version()

    def test_release_please_manifest_matches_pyproject(self):
        """release-please treats the manifest as the source of truth for
        the NEXT bump. If it disagrees with pyproject, the next release
        jumps or repeats a version."""
        assert manifest_version() == pyproject_version()

    def test_version_is_semver(self):
        assert SEMVER.match(hopai.__version__), hopai.__version__


class TestReleaseConfig:
    @pytest.fixture()
    def config(self) -> dict:
        return json.loads((ROOT / "release-please-config.json").read_text())["packages"]["."]

    def test_targets_this_package(self, config):
        assert config["release-type"] == "python"
        assert config["package-name"] == hopai.__name__

    def test_bumps_the_module_version_too(self, config):
        """Without this extra-files entry release-please updates
        pyproject.toml and leaves hopai/__init__.py behind, which is
        exactly the drift test_module_matches_pyproject would then catch
        -- one release too late."""
        paths = {entry if isinstance(entry, str) else entry["path"]
                 for entry in config.get("extra-files", [])}
        assert "hopai/__init__.py" in paths

    def test_extra_files_use_a_supported_updater(self, config):
        """`{"type": "python"}` is not a thing: release-please v4 rejects
        the whole config with `unsupported extraFile type: python`, and
        the release job dies before opening a PR. A bare string selects
        the generic updater, which is the one that handles a
        `__version__` line."""
        for entry in config.get("extra-files", []):
            assert isinstance(entry, str) or entry.get("type") in {
                "json", "yaml", "toml", "xml", "generic"}, entry

    def test_release_as_is_removed_once_it_has_done_its_job(self, config):
        """`release-as` pins EVERY release to that version until deleted.

        It is here because nothing else produced 0.0.1: from a 0.0.0
        baseline release-please proposes 0.1.0 for a feature, and neither
        `initial-version` nor the pre-1.0 bump flags changed that (checked
        against `release-please manifest-pr --dry-run`, which is the mode
        the action runs).

        So it is allowed only while it still matches where the project
        is: before the first release (manifest 0.0.0), and on the release
        PR itself (manifest == release-as). The moment anything is
        released past it, this fails -- which is the first PR where
        leaving it in would silently re-release the same version."""
        pinned = config.get("release-as")
        if pinned is None:
            return
        released = manifest_version()
        assert released in ("0.0.0", pinned), (
            f"release-as is pinned to {pinned} but {released} is already released; "
            f"remove `release-as` from release-please-config.json or every future "
            f"release will claim {pinned} again"
        )

    def test_the_module_carries_the_generic_updater_annotation(self):
        """The generic updater rewrites only lines marked with this
        comment. Miss it and nothing fails -- the file is simply never
        touched, and the package ships reporting the previous version."""
        source = (ROOT / "hopai" / "__init__.py").read_text()
        line = next(row for row in source.splitlines() if row.startswith("__version__"))
        assert "x-release-please-version" in line, line


class TestDistributionMetadata:
    """What PyPI shows and what pip resolves. Wrong here is visible to
    every user and cannot be edited after upload."""

    @pytest.fixture()
    def pyproject(self) -> str:
        return (ROOT / "pyproject.toml").read_text()

    def test_has_a_readme_a_license_and_an_author(self, pyproject):
        assert 'readme = "README.md"' in pyproject
        assert "license = " in pyproject
        assert "authors = " in pyproject

    def test_declares_its_runtime_dependencies(self, pyproject):
        """An install that imports and then dies on `import sqlalchemy`
        is worse than one that refuses to install."""
        for requirement in ("sqlmodel", "sqlalchemy", "psycopg2-binary"):
            assert requirement in pyproject

    def test_networkx_stays_optional(self, pyproject):
        """to_networkx() imports it lazily on purpose; making it a hard
        dependency would put a graph library in the install path of a
        project that only ever traverses."""
        required = pyproject.split("[project.optional-dependencies]")[0]
        assert "networkx" not in required

    def test_requires_python_matches_the_tested_versions(self, pyproject):
        assert 'requires-python = ">=3.10"' in pyproject
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert '"3.10"' in workflow, "CI no longer tests the oldest supported Python"

    def test_only_the_package_is_shipped(self, pyproject):
        """`include = ["hopai*"]` keeps tests/, scripts/ and benchmarks/
        out of the wheel."""
        assert 'include = ["hopai*"]' in pyproject
