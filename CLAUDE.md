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
- **Every read and write goes through `Graph._scoped()`.** Forgetting the graph
  discriminator does not error; it silently touches another graph's rows.
- **Vectors live in `vec_*` real columns, never in `properties`, and never pass
  through an LLM tool schema.** JSONB storage would bloat the GIN index and every
  result; a tool-schema `"vector"` parameter invites a model to invent an embedding,
  and an invented embedding finds confidently wrong neighbors. Similarity is exact
  (unnest+sum cosine, float8 accumulation); a traversal without `near=` must emit
  byte-identical SQL to the pre-vector engine. (`test_defining_vectors_changes_no_near_less_query`,
  `TestToolSchemasStayVectorFree`)
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
- `json_api.py` and `cypher.py` are front ends that emit `(Start, [Hop])`, an
  aggregation triple, or ingestion operations, and hold no query logic. Widening a
  subset means adding a translation, never loosening a refusal into a near-enough
  mapping. The tool schemas (`TRAVERSE_TOOL_SCHEMA` / `AGGREGATE_TOOL_SCHEMA` /
  `INGEST_TOOL_SCHEMA`) must stay in step with what the parsers accept — with one
  pinned exception: the vector keys (`near`/`k`) are parsed but deliberately never
  advertised to a model (see `vectors.py`).
- Comments explain *why*, citing the bug or trade-off. Match that for non-obvious code;
  skip it for mechanical changes.
- New tests join an existing `TestX` class and say what would break without the fix.

## Commands

```bash
pip install -e ".[dev]"
docker compose up -d      # Postgres matching the default DSN
pytest tests/ -v
ruff check .
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

## Deeper detail

| Read | For |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | The read and write pipelines, multi-graph internals, result gotchas, and which module docstring explains what |
| [docs/testing.md](docs/testing.md) | Fixtures, the database-free suite, coverage gate, mutmut config and its fork-safety quirks |
| [docs/releasing.md](docs/releasing.md) | release-please, PyPI trusted publishing, and the traps already hit |
| [README.md](README.md) | The user-facing API |
