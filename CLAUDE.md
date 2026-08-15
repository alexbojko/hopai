# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`hopai` compiles multi-hop graph traversals into a single recursive CTE against two
ordinary PostgreSQL tables (`nodes`, `edges`, each with a JSONB `properties` bag).
No graph database, no extension.

## What this project is for

**Let a developer building an AI project keep a real knowledge graph in the database
they already run, and make every interface obvious to an LLM and a human on first
read.** When a design question comes up, these decide it, in order:

1. **No new dependency, ever.** Postgres and SQLAlchemy are the whole stack. A feature
   needing an extension, sidecar or worker is the wrong feature. Extras stay optional.
   `embeddings.py` is the one place hopai makes a **network call**, and it holds the
   line rather than breaking it: no provider package is imported, the client is the
   caller's to construct and configure, and the extras (`hopai[openai]` and friends)
   name what you were installing anyway. It retries transient failures because a
   network call that gives up on one 429 is not finished work — but the policy is
   `retries=`/`backoff=` on the Embedder, defaults documented against the client's
   own, since the two multiply. A cache belongs to the application; a rate limiter
   belongs to the client.
2. **An LLM must get it right with no custom instructions.** Prefer a protocol a model
   has already seen ten thousand times — Cypher, JSON node/edge lists, JSON Schema tool
   definitions, SQLAlchemy idioms — over anything invented here, even when the invention
   is tidier. If correct use would need a paragraph of prompt, the API is wrong.
3. **Obvious to a human on first read.** One way to do each thing, errors that name the
   fix. "You have to know that..." is a defect.
4. **Refuse, don't approximate.** Where our semantics differ from what a caller expects,
   raise and name the rewrite. A silently different answer is the worst thing this
   library can produce and the hardest for an agent to notice.
5. **Postgres features are the advantage.** Constraints are enterprise-only in Neo4j and
   free for us. Surface anything in that category as a first-class feature.
6. **No performance regression.** Traversal speed is why the premise is believable.
   Inspect the emitted SQL for anything touching the query path; "probably fine" is not
   a measurement.

## Invariants — breaking any of these reintroduces a bug the tests were written for

- **Path tracking is per-hop, not per-chain.** A global path per destination node
  silently drops fan-in when two parents feed one intermediate node.
  (`test_fan_in_both_parents_preserved`)
- **"Which nodes continue the walk" and "which edges to report" are separate queries.**
  Conflating them caused the above.
- **Reported nodes derive from the edges found, never from the seed set** — that is what
  prunes dead ends. (`test_dead_end_excluded_when_edge_kind_filtered`)
- **A hop spanning several edges must report all of them**, not one fabricated edge
  between endpoints. (`test_multi_hop_edge_reconstruction`)
- **`optional=True` is rejected anywhere but the last hop.**
- **`NOT` negates a containment test on purpose**, so rows *missing* the key are
  included. Naive `<> value` drops them. (`test_not_includes_missing_key`)
- **A bare top-level list raises `TypeError`** — it reads as AND but would mean OR.
- **Aggregates run over the LAST step's matched nodes only.** A mid-chain match includes
  nodes with no continuation to the chain's end, so aggregating one would count nodes
  Cypher would not — and bare `count(b)`/`sum(b.x)` with hops mean per-*path* in Cypher,
  which hopai cannot express. `cypher.py`'s AGGREGATION docstring holds the acceptance
  matrix; loosening a refusal into a near-enough mapping is the bug, not the fix.
- **A delete or update with no filter refuses.** `where=None`/`{}` is what an empty
  variable looks like, and matching everything on it is unrecoverable. `all=True` is the
  opt-in; `all=True` *with* a filter refuses too, since one of them is being ignored.
  (`test_a_delete_with_no_filter_refuses`)
- **`all` / `detach` / `replace` are checked, not coerced.** `all="false"` is truthy in
  Python, and JSON booleans arriving as the strings `"true"`/`"false"` is an ordinary
  tool-call failure — coercing let the string `"false"` mean *every row*.
- **A constraint the options discard is not "no filter".** `node_label_key=None` throws
  labels away; on the read path that widens a result set, in front of a `DELETE` it
  widened it to the whole graph *and* auto-supplied the `all=True` opt-in. An operation
  left with nothing refuses instead.
- **Deleting a node that still has edges refuses and the message names `detach=True`.**
  Cascading instead would leave the corruption the composite FK exists to prevent.
- **A mutating `MATCH` binds a set of rows; an ingesting one binds a single node.** That
  is Cypher's own asymmetry — `SET` applies to every match, an edge attaches to exactly
  one — so neither may be "fixed" into the other.
- **`SET x = {...}` refuses unless the map carries the label/type property, and
  `SET x.k = null` means REMOVE.** Cypher's `SET n = {map}` never erases a label and a
  relationship's type cannot be changed at all; `= null` removes a property, where a
  stored JSON null is absent to Cypher and *present* to `Required`. Both are the same
  spelling meaning something else, which is the one thing this library must not do.
- **Every read and write goes through `Graph._scoped()`.** Forgetting the graph
  discriminator does not error; it silently touches another graph's rows.
- **The dimension CHECK's name is `_graph_token`-based, never `_auto_name()` +
  `scope_name()`.** Independent 63-char truncation let two graphs share one constraint —
  silently disabling one graph's enforcement and letting its `drop_vectors()` remove the
  other's. (`test_field_and_graph_can_never_share_a_constraint_name`)
- **Similarity is a LATERAL, never a correlated scalar subquery.** The planner pulls a
  scalar subquery up and re-evaluates it at every site the outer query names it — filter,
  score, `ORDER BY` — so the `unnest` ran 2–3× per candidate for identical results. It
  reads tidier as a subquery and costs 2×; `benchmarks/README.md` has the measurement.
- **A similarity is NULL — "missing" — when the stored vector is NULL, all zeros, or the
  wrong length.** `unnest(a, b)` pads the shorter side, so a mis-sized vector would
  otherwise score a confident cosine over the shared prefix.
- **Vectors live in `vec_*` real columns, never in `properties`.** JSONB storage would
  bloat the GIN index and every result. Similarity is exact (unnest+sum cosine, float8
  accumulation); a traversal without `near=` must emit byte-identical SQL to the
  pre-vector engine. (`test_defining_vectors_changes_no_near_less_query`)
- **A model may send `"text"`; a `"vector"` never reaches a tool schema.** Text is
  embedded by the field itself, with the application's own client, so the query
  embedding comes from the model that wrote the stored ones — advertise it. Floats
  asked of a model are invented, and an invented embedding finds confidently wrong
  neighbors, so `"vector"` is the single key parsed and never advertised, refused
  without `allow_vectors=True`. Widening the refusal back over `text`/`keep`/`boost`
  puts semantic search out of a tool call's reach entirely, which is what it was.
  (`TestToolSchemasStayVectorFree` pins both halves)
- **Embedding happens outside the transaction, batched per field.** `set_vectors()`
  resolves every string before it takes a connection: an HTTP call inside an open
  transaction holds row locks for a network round trip, and a provider dying halfway
  leaves a half-written batch for a retry to collide with.
  (`test_the_provider_is_called_before_the_transaction_opens`)
- **`embeddings.py` imports no provider package, ever** — not even to recognize one.
  Clients are matched by `type(client).__module__` and attribute shape; `isinstance`
  would need the import and would make `hopai[openai]` a coupling instead of a
  convenience. (`TestNoProviderIsImported`)
- **Writes are one transaction**, batching included. A half-committed write makes a
  retry collide with rows that landed.

## Conventions

- Commit subjects are **Conventional Commits** — they decide the version and the
  changelog. An unprefixed subject releases nothing and appears nowhere.
- **`main` is protected**: everything goes through a PR with green CI. No direct pushes.
- Coverage gates at **85%**. Mutation testing never blocks a merge — *and that exempts
  the merge, not the triage*. A surviving mutant is a change the whole suite accepted in
  silence, which is exactly what coverage cannot see. Sort every survivor into one of:
  **real gap** (write the killing test, same PR — assume this until shown otherwise),
  **equivalent** (record the one-line proof; "probably fine" is not a proof), or
  **out of scope** (say so explicitly). A run reporting `0 checked`, or all `segfault`,
  is a broken harness, not a clean sweep.
- `json_api.py`, `mutate.py`'s spec parser and `cypher.py` are front ends that emit
  `(Start, [Hop])`, an aggregation triple, or ingestion/mutation operations, and hold no
  query logic. Widening a subset means adding a translation, never loosening a refusal
  into a near-enough mapping. The tool schemas (`TRAVERSE_TOOL_SCHEMA` /
  `AGGREGATE_TOOL_SCHEMA` / `INGEST_TOOL_SCHEMA` / `MUTATE_TOOL_SCHEMA` /
  `VECTOR_SEARCH_TOOL_SCHEMA`) must stay in step with what the parsers accept — with
  exactly two pinned exceptions, and no third without a reason of the same kind. A near
  spec's **`"vector"`** is parsed and never advertised (the invariant above); **`label`**
  is not advertised because it names result groups for the caller's own bookkeeping and
  builds no SQL, so there is nothing for a model to decide.
  `test_advertises_the_keys_the_parser_reads` derives both sides from
  `_HOP_KEYS`/`_START_KEYS`, so a widened parser and a widened schema fail apart.
  `cypher.py` has no vector spelling and is not expected to grow one — Cypher has no
  portable similarity syntax to translate, so there is nothing to refuse by name.
- `notebooks/` is documentation that **runs**, executed by CI on every PR
  (`python scripts/run_notebooks.py`). A change to a public API means re-running
  them with `--save` and reading the output diff — a stale notebook is a broken
  build, not a cosmetic lag.
- **The documentation site has no sources of its own.** `mkdocs.yml` publishes
  `README.md` and `notebooks/` through symlinks in `docs/`, so editing either one
  *is* editing the site — there is never a second copy to keep in step, and adding
  one is the defect. CI builds it with `--strict` on every PR; **Read the Docs**
  publishes it at <https://hopai.readthedocs.io/> (`.readthedocs.yaml`), built
  **from the release tag** — so the published site describes the version that is
  on PyPI rather than whatever landed on `main`. Nothing in this repository
  pushes the site, so there is no deploy credential; the release-only rule lives
  in an RTD Automation Rule matching tags, not in a workflow.
  Links written *inside* notebooks are invisible to `--strict` (mkdocs-jupyter
  hands MkDocs finished HTML), so `scripts/mkdocs_hooks.py` rewrites them and
  `scripts/check_docs_links.py` fails the build if one stops resolving.
  **A new notebook must be added to `mkdocs.yml`'s `nav`.** It reaches the site
  on its own through the symlink, but an unlisted page is built and then left out
  of the navigation — reachable only by URL. MkDocs calls that INFO, so
  `validation.nav.omitted_files: warn` promotes it to a `--strict` failure.
- Comments explain *why*, citing the bug or trade-off. Match that for non-obvious code;
  skip it for mechanical changes.
- New tests join an existing `TestX` class and say what would break without the fix.

## Commands

```bash
pip install -e ".[dev]"
docker compose up -d      # Postgres matching the default DSN
pytest tests/ -v
ruff check .

pip install -e ".[docs]" && mkdocs serve    # the documentation site on :8000
```

Query building never connects, so the emitted SQL can be inspected with no database —
the fastest check for anything touching `build_query`:

```python
from sqlalchemy.dialects import postgresql
from hopai import Graph, Start, Hop
g = Graph("postgresql+psycopg2://u:p@localhost/db")   # never connects
print(g.build_query(Start(where={"type": "person"}), [Hop(hops=(1, 3))])
       .compile(dialect=postgresql.dialect()))
```

Deletes and updates build the same way — `Mutator`'s `*_statement` methods take no
connection, which is how the graph discriminator is asserted on with nothing running:

```python
from hopai.mutate import Mutator
print(Mutator(g).delete_nodes_statement({"type": "draft"})
      .compile(dialect=postgresql.dialect()))
```

## Deeper detail

| Read | For |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | The read and write pipelines, multi-graph internals, result gotchas, and which module docstring explains what |
| [notebooks/README.md](notebooks/README.md) | The nine runnable notebooks, how they are executed in CI, and how to regenerate their outputs |
| [docs/testing.md](docs/testing.md) | Fixtures, the database-free suite, coverage gate, mutmut config and its fork-safety quirks |
| [docs/releasing.md](docs/releasing.md) | release-please, PyPI trusted publishing, and the traps already hit |
| [README.md](README.md) | The user-facing API |
