# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`hopai` compiles multi-hop graph traversals into a single recursive CTE against two ordinary
PostgreSQL tables (`nodes`, `edges`, each with a JSONB `properties` bag). No graph database, no
extension.

## The goal this project is measured against

**Let a developer building an AI project keep a real knowledge graph in the database they already
run — no Neo4j, no graph extension, no new operational dependency — and make every interface
obvious to an LLM and a human on first read.**

That goal is not decoration; it decides arguments. When a design question comes up, these are the
tiebreakers, in order:

1. **No new dependency, ever.** Postgres and SQLAlchemy are the whole stack. A feature that needs
   an extension, a sidecar service, or a background worker is the wrong feature — find the version
   that a well-indexed table can do. Optional extras (`networkx`) may only ever be optional.
2. **An LLM must get it right with no custom instructions.** Prefer a protocol a model has already
   seen ten thousand times — Cypher, JSON node/edge lists, JSON Schema tool definitions, networkx,
   SQLAlchemy's own idioms — over anything invented here, even when the invention is tidier. If
   using the API correctly would require a paragraph of prompt explaining our conventions, the API
   is wrong. Ship a JSON Schema alongside anything an agent is meant to call.
3. **Obvious to a human on first read.** The same property, from the other side: names that say
   what they do, one way to do each thing, errors that name the fix. This library is meant to be
   handed to teammates and recommended to other teams; "you have to know that..." is a defect.
4. **Refuse, don't approximate.** Where our semantics differ from what a caller likely expects,
   raise and name the rewrite. A silently different answer is the worst outcome this library can
   produce, and the hardest for an agent to notice.
5. **Postgres features are the advantage — use them.** Constraints (unique, composite unique,
   partial, CHECK) are enterprise-only in Neo4j and free for us. Anything Postgres gives us that a
   graph database charges for or lacks is worth surfacing as a first-class feature.
6. **No performance regression.** Traversal speed is the reason to believe the premise. Any change
   that could touch the query path needs its SQL inspected (see `build_query` below) and, for
   anything structural, a benchmark run before and after. "Probably fine" is not a measurement.

## Commands

```bash
pip install -e ".[dev]"

# Database-backed tests skip cleanly when nothing is listening; this starts one
# matching the default DSN, so no configuration is needed.
docker compose up -d
pytest tests/ -v
pytest "tests/test_hopai.py::TestCoreTraversal::test_simple_forward_hop" -v   # one test
pytest tests/ -k "optional or cycle"                                          # by keyword
ruff check .

# Points at your own instead: export HOPAI_TEST_DSN=postgresql+psycopg2://...
# HOPAI_REQUIRE_DB=1 turns a missing database into an error rather than skips.
# CI sets it, because a suite that skips everything is a suite that passes
# having tested nothing.
```

Fixtures: `graph` is the seeded 7-node read fixture; `fresh_graph` is an empty
graph in its own schema, rebuilt per test (constraints outlive TRUNCATE, so write
tests need a real drop); `offline_graph` needs no database at all.

```bash
pytest tests/ --cov=hopai --cov-report=term-missing    # coverage; CI requires >= 85%

pip install -e ".[dev,mutation]"
pytest tests/ --cov=hopai --cov-report=          # mutmut needs a .coverage file first
mutmut run hopai/hop.py                          # one module — the usual local loop
mutmut results                                   # survivors
mutmut show <mutant-id>                          # the diff of one survivor
```

Config is `setup.cfg` (mutmut's only config surface). Two settings there exist
because of bugs, not taste: `hopai` is in `also_copy` as well as `source_paths`
(a narrowed scope otherwise leaves the package without its `__init__.py` in the
mutants tree, and every test dies on an import error), and `scripts` is copied
because a test imports from it. The test engines set `gssencmode=disable` for
the same class of reason — on macOS libpq's Kerberos probe goes through XPC,
which is not fork-safe, and mutmut's forked children segfault before reaching
Postgres.

## Commit messages are load-bearing

Releases are cut by **release-please** from **Conventional Commits**, so the subject
line of every commit to `main` is an input to the version number and the changelog —
not prose. Use `type: summary`, and `type(scope): summary` when a scope helps:

| Prefix | Changelog | Bump while below 1.0 |
| --- | --- | --- |
| `feat:` | Features | patch (`0.0.1` → `0.0.2`) |
| `fix:` | Bug Fixes | patch |
| `perf:` `docs:` `refactor:` | own heading | patch |
| `test:` `ci:` `build:` `chore:` | hidden | none |
| `feat!:` or a `BREAKING CHANGE:` footer | Features | minor while below 1.0 |

An unprefixed subject releases nothing and appears nowhere. That is the failure mode
to watch for: the work merges, CI is green, and the release PR silently does not
mention it.

**Everything is a patch bump on purpose while the version is `0.0.x`** — that series
says "anything may change". `bump-patch-for-minor-pre-major` in
`release-please-config.json` is what does it; flip it to `bump-minor-pre-major` when
the API is worth promising, or land a `feat!:` to move to `0.1.0`.

**Never edit `version` in `pyproject.toml`, `hopai/__init__.py` or
`.release-please-manifest.json` by hand.** release-please owns all three;
`tests/test_packaging.py` fails if they drift apart. `0.0.0` in the manifest means
nothing has been published yet — the first release PR turns it into `0.0.1`.

## Releasing

1. Merge normal PRs to `main` with conventional-commit subjects.
2. release-please keeps one open PR titled `chore(main): release x.y.z` with the
   bump and the CHANGELOG entry. Review it like any other PR.
3. **Merging that PR is the release.** It tags the commit, creates the GitHub
   Release, and the same workflow builds and publishes to PyPI via Trusted
   Publishing (OIDC — there is no API token anywhere in this repo).

The publish job checks out the *tag*, runs `twine check --strict`, and asserts the
built artifact's version equals the tag before uploading. A PyPI upload cannot be
undone or replaced, so those checks come first.

If the upload itself fails, re-run it from Actions → Release → *Run workflow* with
`publish_tag: v0.0.1`. That path exists because a GitHub Release created with
`GITHUB_TOKEN` does not fire the `release` event, so publishing can only be chained
onto the run that created it — and a chain that has already failed needs a door back
in. Note that PyPI will reject a re-upload of a version that already landed; bump
instead.

## Coverage and mutation testing

**Coverage gates; mutation informs. Neither may be skimmed.**

Line coverage must stay **at or above 85%**, enforced by CI after the PR comment
is posted, so a failing PR still says by how much it fell short. Coverage answers
"did any test execute this line" — nothing more.

**Mutation testing never blocks a merge, and that exempts the merge, not the
triage.** A surviving mutant is a change to the source that the entire suite
accepted in silence: the line was executed and nothing asserted on it. That is
precisely the gap coverage cannot see, and it is the reason to read the report
rather than note its colour. On any PR whose sticky comment lists survivors,
sort **every** one into exactly one class:

1. **Real gap** — write the test that kills it, in the same PR. This is the
   default; assume a survivor is real until shown otherwise.
2. **Equivalent mutant** — the change cannot alter behaviour (a reordering with
   no observable effect, a constant only used in a message nobody depends on).
   Record the one-line proof in the PR. "Probably fine" is not a proof.
3. **Out of scope** — the mutated line belongs to a path this PR did not touch
   and a follow-up is genuinely warranted. Say so explicitly; do not let it pass
   unmentioned.

The first survivor this repo ever produced was class 1: three mutants replaced
`ValueError(...)` messages in `_normalize_hops` with `None` and nothing failed,
because the tests asserted the exception type and not the message. The fix was
to assert the message — which the "errors name the fix" principle above already
required. That is the normal outcome; treat class 2 as the rare one.

A run reporting `0 checked`, or every mutant `segfault`, is a broken harness and
not a clean sweep — investigate it rather than reading it as a pass.

No linter, formatter, or type checker is configured — don't invent one, and don't reformat files
you aren't otherwise changing.

**Inspecting generated SQL needs no database.** `create_engine` doesn't connect, so query building
works fully offline — the fastest way to check a change to `build_query` when no Postgres is around:

```python
from sqlalchemy.dialects import postgresql
from hopai import Graph, Start, Hop
g = Graph("postgresql+psycopg2://u:p@localhost/db")          # never connects
print(g.build_query(Start(where={"type": "person"}), [Hop(hops=(1, 3))])
       .compile(dialect=postgresql.dialect()))
```

Benchmarks need a large generated dataset and their own database; see `benchmarks/README.md`.

## Architecture

The pipeline spans three modules and is easier to follow as one flow than file by file:

1. **`hop.py`** — `Start(where=)` and `Hop(where=, via=, hops=, direction=, optional=)` are plain
   dataclasses. `Hop.__post_init__` normalizes `hops` (an int or `(min, max)`) into `min_hops` /
   `max_hops`, which is what `core.py` actually reads.
2. **`filters.py`** — `resolve(column, filt)` compiles the filter DSL into a SQLAlchemy boolean
   expression. Equality is JSONB containment (`@>`), not `->>` comparison. `parse_filter()` turns
   the JSON operator form into the same Python objects, and `cypher.py` produces those same
   objects too, so **all three front ends compile through the one `resolve()`** — never add a
   second compilation path.
3. **`core.py:build_query`** — per hop `i`, emits a recursive CTE `walk_i` (anchored on the `seed`
   CTE for the first hop, on `match_{i-1}` after that), then two CTEs derived from it: `match_i`
   (distinct reached nodes that pass
   `where`, feeding the next hop) and `hop_edges_i` (every real edge id used). All `hop_edges_*` are
   unioned into `all_edges` → `edge_rows`, and the final statement returns tagged
   `(kind, id)` rows — `"node"` / `"edge"` — in one round trip.
4. **`core.py:traverse`** — splits those rows by `kind` and issues two follow-up `SELECT`s to
   hydrate properties. `elapsed_ms` on the returned `Subgraph` times all three queries.

Writes are a separate path: `Graph` exposes them, `ingest.py` and `constraints.py` implement
them, and `core.py` stays the traversal engine. The one place they meet is
`constraints.key_sql()`, which renders both `CREATE INDEX` and the `ON CONFLICT` target — index
inference only works when the two are spelled identically, so they must come from one function.

### Invariants — breaking any of these reintroduces a bug the tests were written for

- **Path tracking is per-hop, not per-chain.** Each `walk_i` carries its own `local_path` array,
  reset at the hop boundary. A global path per destination node silently drops fan-in when two
  parents feed one intermediate node. Guarded by `test_fan_in_both_parents_preserved`.
- **"Which nodes continue the walk" and "which edges to report" are separate queries** (`match_i` vs
  `hop_edges_i`). Conflating them is what caused the above.
- **Reported nodes are derived from the edges found, never from the seed set.** That is what prunes
  dead ends automatically, with no separate pass. Guarded by
  `test_dead_end_excluded_when_edge_kind_filtered`.
- **A hop can span several real edges, so all of them must be reconstructed** — not one fabricated
  edge between hop endpoints. Guarded by `test_multi_hop_edge_reconstruction`.
- **`optional=True` is rejected anywhere but the last hop**, at `build_query` entry. Mid-chain
  support means every downstream hop tolerating a missing anchor — a real feature, not a flag.
- **`NOT` negates a containment test on purpose**, so rows *missing* the key are included. Naive
  `<> value` treats a missing key as SQL `NULL` and drops it under negation. Guarded by
  `test_not_includes_missing_key`.
- **A bare top-level list raises `TypeError`.** It reads as "AND these" but would mean OR; callers
  must write `OR(...)`.

### Two gotchas that surprise readers of the result

- **Ids come back as strings.** `build_query` casts them to `String` so node and edge ids can share
  one union'd result column, and `traverse` keeps them that way — assertions compare `{"1", "2"}`,
  not `{1, 2}`. The hydration queries then cast the indexed BIGINT to text to match; removing that
  cast as an optimization breaks the string-id contract the tests assert on.
- **Table and column names are configurable** via `Graph(engine, node_table=…, edge_start_col=…)`.
  Query building must go through `self.node_id_col` / `self.edge_start_col` / etc. and the
  `nodes_tbl` / `edges_tbl` attributes — never hardcode `Node.__table__` or a literal `"start_id"`.
  The SQLModel classes in `models.py` are the default, not a dependency of the engine.

## Where the reasoning lives

Every module opens with a docstring explaining *why* it is shaped that way, written against bugs
actually hit during development. Read the relevant one before changing behavior:

| Read this | For |
| --- | --- |
| `hopai/core.py` header | Why local paths, split queries, edge-derived nodes, last-hop-only `optional` |
| `hopai/filters.py` header | The full DSL in both forms, and why `OR`/`AND`/`NOT` are explicit classes |
| `hopai/hop.py` header | Why `Start` and `Hop` are separate types rather than one |
| `hopai/models.py` header | The expected DDL, and the typed-columns / JSONB-bag split |
| `hopai/cypher.py` header | The translatable Cypher subset (read and write), and why each refusal is a refusal |
| `hopai/ingest.py` header | The two row spellings, edge-by-property references, and merge semantics |
| `hopai/constraints.py` header | What each constraint compiles to, and the two SQL semantics that surprise people |
| `tests/conftest.py` | The 7-node fixture graph — it deliberately contains a dead end, a fan-in, and a cycle |
| `README.md` "What this doesn't do" | Committed-to limitations: one linear chain, sync only, path-array cost past ~10 hops |
| `benchmarks/README.md` | Measured numbers, including where raw CTEs beat this library 2-5x |

## Conventions

- Comments here explain *why*, and cite the bug or trade-off behind a decision. Match that when
  touching non-obvious code; skip it for mechanical changes.
- New tests belong to an existing `TestX` class in `tests/test_hopai.py` and get a docstring saying
  what would break without the fix.
- `json_api.py` is a translation layer only. Anything it needs to *decide* belongs in `filters.py`
  or `core.py`, and `TRAVERSE_TOOL_SCHEMA` must stay in step with what `spec_to_traversal` accepts.
- `cypher.py` is the same: a front end that emits `(Start, [Hop])` for reads and a list of
  ingestion operations for writes, holding no query logic of its own. Its rule is **refuse, don't
  approximate** — a Cypher construct with no hopai equivalent, or with a *different meaning* here
  (`<>` and `NOT x = y` versus containment-based `NOT`; whole-path `MERGE` versus per-row upsert),
  raises `CypherError` naming the rewrite. Widening the subset means adding a translation, never
  loosening one of those refusals into a near-enough mapping.
- Writes are one transaction per call, batching and multi-clause Cypher included. Half-committing
  is the worst failure mode this library has: the caller is told it failed, retries, and the retry
  collides with rows that landed. `tests/test_ingest.py::TestAtomicity` guards this.
