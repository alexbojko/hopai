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

    def test_release_as_is_gone_once_anything_has_been_released(self, config):
        """`release-as` pins EVERY release to that version until deleted.

        It existed only to force the first release to 0.0.1, because from
        a 0.0.0 baseline release-please proposes 0.1.0 and neither
        `initial-version` nor the pre-1.0 bump flags changed that.

        The rule is: the moment the manifest leaves 0.0.0, it must be
        gone. An earlier version of this test also allowed
        `released == pinned`, which is the exact state right after the
        first release -- so it passed while the pipeline was pinned to
        re-release 0.0.1 forever, which PyPI would reject."""
        pinned = config.get("release-as")
        if pinned is None:
            return
        assert manifest_version() == "0.0.0", (
            f"release-as is still pinned to {pinned} but {manifest_version()} is "
            f"released; remove `release-as` from release-please-config.json or every "
            f"future release will claim {pinned} again and PyPI will reject it"
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

    def test_pydantic_stays_optional(self, pyproject):
        """schema_pydantic imports it lazily and the hopai[pydantic]
        extra pins v2. pydantic must never become a DIRECT required
        dependency: sqlmodel already brings it transitively at ITS
        chosen floor, and a direct pin here would silently take over
        every install's resolution."""
        required = pyproject.split("[project.optional-dependencies]")[0]
        assert "pydantic" not in required

    def test_requires_python_matches_the_tested_versions(self, pyproject):
        assert 'requires-python = ">=3.10"' in pyproject
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert '"3.10"' in workflow, "CI no longer tests the oldest supported Python"

    def test_only_the_package_is_shipped(self, pyproject):
        """`include = ["hopai*"]` keeps tests/, scripts/ and benchmarks/
        out of the wheel."""
        assert 'include = ["hopai*"]' in pyproject
