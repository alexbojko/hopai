# Testing

```bash
pip install -e ".[dev]"
docker compose up -d                    # throwaway Postgres on the default DSN
pytest tests/ -v
pytest tests/ -k "optional or cycle"
ruff check .
```

`HOPAI_TEST_DSN` points at your own instance. `HOPAI_REQUIRE_DB=1` turns a missing
database into an error rather than skips — CI sets it, because a suite that skips
everything is a suite that passes having tested nothing.

## Fixtures

| Fixture | What it is |
| --- | --- |
| `graph` | The seeded 7-node read fixture (dead end, fan-in, a real cycle) |
| `fresh_graph` | An empty graph in its own schema, rebuilt per test |
| `offline_graph` | No database at all — for query-shape and translation tests |

`fresh_graph` drops and recreates the schema rather than truncating: constraints are
schema-level and outlive a `TRUNCATE`, so one test's `Unique()` would leak into the
next.

Most of the suite needs no database. Query shape, filter compilation, the JSON/Python
grammar equivalence and the whole Cypher translator are tested against compiled SQL,
which is also how a change that alters the emitted query gets noticed — checking that
rows come back cannot tell a fast query from a slow one returning the same answer.

## Coverage

Line coverage must stay **at or above 85%**, enforced by CI *after* the PR comment is
posted so a failing PR still says by how much it fell short. The gate is on **total**
package coverage, not on the lines a PR adds — a PR adding a large untested module can
still pass on the strength of everything else.

## Mutation testing

```bash
pip install -e ".[dev,mutation]"
pytest tests/ --cov=hopai --cov-report=    # mutmut needs a .coverage file first
mutmut run "hopai.hop.*"                   # one module — the usual local loop
mutmut results
mutmut show <mutant-id>
```

The run filter is an `fnmatch` glob over dotted MUTANT names
(`hopai.hop.x_foo__mutmut_1`), so it is spelled `"hopai.hop.*"` and never
`hopai/hop.py` — a path silently matches nothing and mutmut aborts with "Filtered for
specific mutants, but nothing matches". The same trap one level up: `source_paths` in
`setup.cfg` (and CI's narrowed rewrite of it) is a NEWLINE-separated ini list; a
space-separated line is read as one nonexistent path and mutmut "succeeds" having
mutated zero files.

Config is `setup.cfg`, mutmut's only config surface. Three settings there exist
because of bugs, not taste:

- **`hopai` is in `also_copy` as well as `source_paths`.** A narrowed scope otherwise
  leaves the package without its `__init__.py` in the mutants tree, every test dies on
  an import error, and the run reads as "every mutant survived".
- **`scripts` is copied** because a test imports from it.
- **The test engines set `gssencmode=disable`** (`tests/conftest.py`). mutmut runs each
  mutant in a forked child; on macOS libpq probes the Kerberos credential cache through
  XPC, which is not fork-safe, so children segfaulted inside `pg_GSS_have_cred_cache`
  before reaching Postgres — and survivors were reported as `segfault` rather than as
  findings. Linux never hit it. `NullPool` is there for the same class of reason.

- **Write-side mutants can come back `timeout` rather than killed.** mutmut runs mutants
  concurrently against the one Postgres the suite uses, and `fresh_graph` starts each
  test with `DROP SCHEMA ... CASCADE` — so two mutant processes can sit waiting on each
  other's locks and be reported as timeouts. Before treating one as a finding, apply the
  mutation by hand and run the suite: `hopai.mutate.xǁMutatorǁdelete_edges__mutmut_18`
  came back `timeout` and fails the suite in nine seconds on its own.

- **A `survived` verdict is only as fresh as the last run against that source.** mutmut
  caches per mutant and re-runs one only when its source changed — so adding the test
  that kills a mutant leaves the old verdict in `mutmut results` until that function is
  edited. Three verdicts here were stale in exactly that way; applying the mutation by
  hand is what showed it. `rm -rf mutants/` for a clean read.
- **A module-level constant's builder comes back `no tests`.** `MUTATE_TOOL_SCHEMA` is
  built by `_operation_schema()` at import time, so no *test* executes those lines and
  `mutate_only_covered_lines` skips them — they are unchecked, not survived, and CI's
  report may list them either way. The assertions in `TestToolSchema` do cover the
  result; apply a mutation by hand if you want the proof.

CI scopes mutation to the files a PR changed and caps it with a wall-clock budget; a
full run over `hopai/` is thousands of mutants.
