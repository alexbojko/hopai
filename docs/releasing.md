# Releasing and publishing

Releases are cut by [release-please](https://github.com/googleapis/release-please)
from Conventional Commits, and published to PyPI by `.github/workflows/release.yml`.

## The loop

1. Merge PRs to `main` with conventional-commit subjects.
2. release-please keeps one open PR, `chore(main): release x.y.z`, carrying the
   version bump and the CHANGELOG entry.
3. **Merging that PR is the release.** It tags, creates the GitHub Release, and the
   same run builds and publishes to PyPI.

## Commit subjects decide the version

| Prefix | Changelog | Bump while below 1.0 |
| --- | --- | --- |
| `feat:` | Features | patch |
| `fix:` | Bug Fixes | patch |
| `perf:` `docs:` `refactor:` | own heading | patch |
| `test:` `ci:` `build:` `chore:` | hidden | none |
| `feat!:` / `BREAKING CHANGE:` footer | Features | minor |

An unprefixed subject releases nothing and appears nowhere — the work merges, CI is
green, and the release PR silently does not mention it.

## Things that bit us

- **`version` lives in three files** (`pyproject.toml`, `hopai/__init__.py`,
  `.release-please-manifest.json`) and release-please owns all three. Never edit by
  hand; `tests/test_packaging.py` fails on drift.
- **`hopai/__init__.py` uses the *generic* updater**, which only rewrites lines
  carrying `# x-release-please-version`. There is no `python` extra-files type —
  passing one aborts the whole release job with `unsupported extraFile type: python`.
  A missing annotation fails silently instead: the file is simply never touched.
- **`release-as: "0.0.1"` is temporary.** From a `0.0.0` baseline release-please
  proposes `0.1.0` for a feature; neither `initial-version` nor the pre-1.0 bump
  flags changed that. Delete it after the first release —
  `test_release_as_is_removed_once_it_has_done_its_job` fails once leaving it in
  would start re-releasing the same version.
- **Publishing is OIDC Trusted Publishing**, so no API token exists in this repo.
  PyPI trusts repo + workflow filename + the `pypi` environment; renaming any of
  them breaks the upload until the publisher is updated.
- **A Release created with `GITHUB_TOKEN` does not fire the `release` event**, which
  is why publishing is chained onto the same run rather than triggered by it. The way
  back in is *Actions → Release → Run workflow* with `publish_tag: v0.0.1`. PyPI will
  not accept re-uploading a version that already landed — bump instead.

## Checking a config change before merging it

```bash
npx release-please@16 manifest-pr --repo-url=https://github.com/alexbojko/hopai \
  --target-branch=<your-branch> --config-file=release-please-config.json \
  --manifest-file=.release-please-manifest.json --token="$(gh auth token)" --dry-run
```

Use `manifest-pr`, not `release-pr` — the legacy command answers differently and is
not what the action runs.
