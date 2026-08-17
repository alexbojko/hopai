<div align="center">

# 🐘 hopai

**A knowledge graph in the Postgres you already run — no graph database required.**

[![CI](https://github.com/alexbojko/hopai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alexbojko/hopai/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/coverage-%E2%89%A585%25-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

hopai compiles multi-hop graph traversals into a single recursive CTE against
two ordinary PostgreSQL tables (`nodes`, `edges`, each with a JSONB
`properties` bag) — no extension, no sidecar service. Traversal, ingestion,
updates, real constraints and vector search, through a Python API, a JSON
one, and a Cypher subset, so an LLM agent and a human developer can both use
it with nothing new to learn.

> **Early and moving.** hopai is pre-1.0 (`0.0.x`). Every `feat`/`fix`
> release may change an interface — a method signature, a JSON key, a
> refusal's exact wording — without a deprecation cycle; pin a version if
> that would break you. Schema migrations don't exist yet either:
> `drop_schema()` + `create_schema()` is the upgrade path for now. See the
> [changelog](https://github.com/alexbojko/hopai/blob/main/CHANGELOG.md)
> before bumping.

## ⚡ Quick start

```bash
pip install hopai
```

```python
from sqlalchemy import create_engine
from hopai import Graph, Start, Hop

graph = Graph(create_engine("postgresql+psycopg2://user:pass@host/db"))

# "Which companies do Alice's friends work for — counting friends of
#  friends, up to four hops out, and only people who are still active?"
result = graph.traverse(
    Start(where={"name": "Alice"}),
    Hop(via={"kind": "friend"}, hops=(1, 4), where={"active": True}),
    Hop(via={"kind": "works_at"}, where={"type": "company"}),
)

result.nodes            # [{"id": "1", "properties": {...}}, ...]
result.edges            # [{"id": "7", "start_id": "1", "end_id": "2", "properties": {...}}, ...]
result.to_networkx()    # in-memory graph, if you have networkx installed
```

Read a traversal left to right as a sentence: **start here → follow these
edges this many times → land on nodes like this → then again.** You get
back the whole matching subgraph, not just the endpoints. The full walkthrough
is [`01_quickstart`](notebooks/01_quickstart.ipynb).

## 🔭 Every way to ask

One engine underneath all of it — the same "Alice's friends" question as
above, other front ends that ask it, and the other things this engine
answers:

```python
# JSON -- for an LLM tool call, an HTTP handler, or config-driven traversal
traverse_json(graph, {"start": {"where": {"name": "Alice"}},
                      "hops": [{"via": {"kind": "friend"}, "hops": [1, 4]}]})

# Cypher -- for a caller, or a model, that already thinks in it
graph.cypher("MATCH (a:person {name: 'Alice'})-[:friend*1..4]->(b) RETURN b")

# Aggregate -- a number instead of a subgraph, computed in the database
graph.aggregate(Start(where={"name": "Alice"}),
                Hop(via={"kind": "friend"}, hops=(1, 4)),
                aggregates={"count": Count()})               # {"count": 12}

# Change and delete -- the same filters, selecting rows to update or remove
graph.update_nodes(where={"name": "Alice"}, set={"active": False})
```

**Similarity and traversal compose** — find the nodes closest in meaning to
some text, then walk the graph from there. No separate vector database, no
gluing two systems together with application code:

```python
# Vector search on its own -- exact cosine similarity, no pgvector, no extension
graph.vector_search(Near("summary", "distributed consensus"), k=10,
                    where={"type": "paper"})

# Start a traversal from similarity instead of a property match: the 5 papers
# most similar to the text, then everything they cite, up to 3 hops out.
graph.traverse(
    Start(near=Near("summary", "distributed consensus"), keep=5),
    Hop(via={"kind": "cites"}, hops=(1, 3)),
)
```

`Hop(near=, keep=)` and `Hop(via_near=, via_keep=)` do the same thing
mid-chain — rank what a hop reaches, or beam over each node's edges, by
similarity instead of only filtering by property. See
[Vector search](https://hopai.readthedocs.io/en/latest/reference/vector-search/)
in the table below.

**Reranking is the third stage** — retrieve wide and cheap with the cosine
above, then actually read each candidate against the query before keeping
`k`. It attaches to a flat search and to a traversal step alike:

```python
import cohere
from hopai import Rerank

reader = Rerank(cohere.ClientV2(), model="rerank-v3.5",
                document_from='.properties.title + ": " + .properties.summary',
                candidates=50)                                # pool the reranker sees

graph.vector_search(Near("summary", "distributed consensus"), rerank=reader, k=10)
```

No provider package imported here either — a Cohere/Voyage client, a
`sentence-transformers` `CrossEncoder`, or a plain callable all work. See
[Reranking](https://hopai.readthedocs.io/en/latest/reference/reranking/) in
the table below.

Every one of these compiles through the same query builder, so the SQL, the
semantics and the invariants are identical no matter which front end wrote
the call. The ["Learn more"](#-learn-more) table below is where each one's
full reference lives.

## ✨ Highlights

- 🐘 **Plain PostgreSQL** — two tables, a recursive CTE, no extension, no
  new operational dependency.
- 🧭 **Real multi-hop traversal** — bounded and unbounded hops, per-hop
  direction, `OPTIONAL`, rich JSONB filtering, one round trip.
- ✏️ **Update and delete by filter** — `SET` / `REMOVE` / `DETACH DELETE`
  semantics, with a filterless call refusing rather than emptying the graph.
- 🧮 **In-database aggregation** — `count` / `sum` / `avg` / `min` / `max`
  computed where the data lives.
- 🧬 **Many graphs, one database** — a graph is a string, not a schema;
  cross-graph edges are impossible by construction (composite FK).
- 🤖 **Three front ends, one engine** — Python, JSON (with a ready-made LLM
  tool schema), and a Cypher subset all compile through the same builder.
- 🔌 **An MCP server in one command** — `hopai-mcp` exposes reading, writing,
  schema and similarity tools, with permissions deciding which tools exist.
- 🔐 **Constraints Neo4j puts behind an enterprise licence** — unique,
  composite, partial, existence, type and CHECK constraints on JSONB.
- 🧲 **Similarity-seeded traversal** — find the nodes closest in meaning to
  some text, then walk the graph from there, in one call. Exact cosine
  similarity on plain `real[]` columns, no pgvector, multivector queries,
  and a field-level `embed=` so you can hand it text instead of floats.
- 🎯 **Reranking, including inside the walk** — `Rerank(client,
  document_from='<jq>')` adds a third retrieval stage to a search *and* to
  a traversal step, where a candidate is a node plus how it was reached. A
  model may write the projection: it's validated against a total jq subset
  in which `env` doesn't parse.
- 🧪 **Tested like it matters** — an 85% coverage gate and mutation testing
  in CI, and real benchmark numbers in `benchmarks/`.

## 🗄️ Schema

```python
graph.create_schema()   # idempotent; safe to call on every start-up
```

```sql
CREATE TABLE nodes (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    properties JSONB NOT NULL DEFAULT '{}'
);
CREATE TABLE edges (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    start_id BIGINT NOT NULL REFERENCES nodes(id),
    end_id   BIGINT NOT NULL REFERENCES nodes(id),
    properties JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX ON edges (start_id);
CREATE INDEX ON edges (end_id);
CREATE INDEX ON nodes USING GIN (properties);
CREATE INDEX ON edges USING GIN (properties);
```

Custom table/column names, and extra real columns alongside the JSONB bag
(a foreign key to your own `users` table, with the collision refusals that
keep it distinct from a JSONB property) are covered in full in
[Schema](https://hopai.readthedocs.io/en/latest/reference/schema/); multi-graph
isolation on one connection pool in
[`07_many_graphs`](notebooks/07_many_graphs.ipynb) and
[Many graphs](https://hopai.readthedocs.io/en/latest/reference/multi-graph/).

## 📚 Learn more

Nothing below is summarized away — every section the README used to spell
out inline now has a full write-up in one of three places, and each links
to the others: a **runnable notebook** for the topics that have one
(executed in CI on every PR, so it can't drift from the API), a **guide**
under Reference explaining the semantics and the gotchas a notebook doesn't
narrate, and a generated **API reference** — [`hopai.core`](https://hopai.readthedocs.io/en/latest/api/core/)
for `Graph` itself, and one page per module for everything else — built
straight from the library's own docstrings and signatures, so it is the one
tier of this table that is structurally unable to go stale:

| Topic | Notebook | Full reference |
| --- | --- | --- |
| Schema — two tables, extending the model with real columns | [`08_under_the_hood`](notebooks/08_under_the_hood.ipynb) | [Schema](https://hopai.readthedocs.io/en/latest/reference/schema/) |
| Many graphs, one database | [`07_many_graphs`](notebooks/07_many_graphs.ipynb) | [Many graphs](https://hopai.readthedocs.io/en/latest/reference/multi-graph/) |
| Getting data in — `add_nodes`/`add_edges`/`merge_*`/`ingest` | — | [Getting data in](https://hopai.readthedocs.io/en/latest/reference/ingestion/) |
| Changing and deleting — `update_*`/`delete_*`/`clear`/`mutate` | — | [Changing and deleting](https://hopai.readthedocs.io/en/latest/reference/mutations/) |
| Constraints — unique, composite, partial, existence, type, CHECK | [`05_constraints`](notebooks/05_constraints.ipynb) | [Constraints](https://hopai.readthedocs.io/en/latest/reference/constraints/) |
| Declaring, inferring and enforcing a graph schema | [`06_graph_schema`](notebooks/06_graph_schema.ipynb) | [Graph schema](https://hopai.readthedocs.io/en/latest/reference/graph-schema/) |
| Filters — `AND`/`OR`/`NOT`/`GT`/`BETWEEN`, the escape hatch | — | [Filters](https://hopai.readthedocs.io/en/latest/reference/filters/) |
| Traversal: direction, hop count, `OPTIONAL` | [`02_traversal`](notebooks/02_traversal.ipynb) | [Traversal](https://hopai.readthedocs.io/en/latest/reference/traversal/) |
| Aggregation | [`03_aggregation`](notebooks/03_aggregation.ipynb) | [Aggregation](https://hopai.readthedocs.io/en/latest/reference/aggregation/) |
| Vector search, hybrid ranking, text-to-vector embedding | [`09_vector_search`](notebooks/09_vector_search.ipynb) | [Vector search](https://hopai.readthedocs.io/en/latest/reference/vector-search/) |
| Reranking — the third retrieval stage, including step-wise | [`10_reranking`](notebooks/10_reranking.ipynb) | [Reranking](https://hopai.readthedocs.io/en/latest/reference/reranking/) |
| The JSON interface | [`04_json_and_cypher`](notebooks/04_json_and_cypher.ipynb) | [JSON interface](https://hopai.readthedocs.io/en/latest/reference/json-interface/) |
| Cypher as input syntax | [`04_json_and_cypher`](notebooks/04_json_and_cypher.ipynb) | [Cypher](https://hopai.readthedocs.io/en/latest/reference/cypher/) |
| What this doesn't do (yet), and why each refusal is a refusal | — | [Limits](https://hopai.readthedocs.io/en/latest/reference/limits/) |
| MCP server — client setup, every tool, every flag | — | [Full guide](https://hopai.readthedocs.io/en/latest/mcp/) |
| Read/write pipelines, multi-graph internals, gotchas | — | [architecture.md](https://hopai.readthedocs.io/en/latest/architecture/) |
| Fixtures, coverage gate, mutation testing | — | [testing.md](https://hopai.readthedocs.io/en/latest/testing/) |
| release-please, PyPI trusted publishing | — | [releasing.md](https://hopai.readthedocs.io/en/latest/releasing/) |
| Measured traversal and vector-search costs | — | `benchmarks/README.md` |

See [`notebooks/README.md`](notebooks/README.md) for how to run the
notebooks yourself against a throwaway database.

## 🔌 MCP server

The same graph as an [MCP](https://modelcontextprotocol.io/) server, so
Claude Desktop, Claude Code, an IDE or an agent framework can use it with
nothing to write:

```bash
pip install "hopai[mcp]"
hopai-mcp --dsn postgresql+psycopg2://user:pass@localhost/db --read-only
```

Eleven tools — traverse, aggregate, Cypher, ingest, update/delete, schema
inference/declaration, and similarity search. **Permissions decide which
tools exist**: `--read-only` registers reading only, the default adds
writing, `--allow-mutations` adds deleting, `--allow-ddl` adds
`enforce_schema`. 📖 **[Full guide](https://hopai.readthedocs.io/en/latest/mcp/)**.

## ⏱️ Async

`AsyncGraph` (`pip install hopai[asyncio]`) covers traversal, aggregation,
ingestion, mutation and vector search for an async app — the same query
builders `Graph` runs, reached through SQLAlchemy's sync/async bridge.
Schema and constraint declaration stay on the sync `Graph` — one-time setup
calls with no concurrency to gain. See `hopai/asyncio.py` and the
[Async](https://hopai.readthedocs.io/en/latest/architecture/#async) section
of architecture.md for the bridge design and the benchmark behind it.

## 🚧 What this doesn't do (yet)

- No disjoint multi-pattern matching — one linear chain of hops only.
- `OPTIONAL` only on the last hop, not mid-chain.
- Aggregation covers `count`/`sum`/`avg`/`min`/`max` over the last step's
  matched nodes, numeric properties only — no grouping, no `stddev`/percentiles.
- Deletes and updates select rows by their properties, never by where a
  traversal arrived.
- Vector search is exact and unindexed by design — no ANN, no
  late-interaction multivectors.
- Embedding retries transient failures but does not cache or rate-limit —
  that's the application's and the client's job, respectively.
- A cycle-protection path array on every recursive row is measurably
  not-cheap past roughly 10 hops on a single-segment traversal.

Each refusal names the rewrite rather than approximating — see
[the full list](https://hopai.readthedocs.io/en/latest/reference/limits/)
for the reasoning behind each one, and
[architecture.md](https://hopai.readthedocs.io/en/latest/architecture/) /
`hopai/vectors.py`/`hopai/cypher.py` for the implementation.

## 🛠️ Development

```bash
pip install -e ".[dev]"
docker compose up -d      # throwaway PostgreSQL matching the default DSN
pytest tests/ -v
ruff check .
```

Most of the suite needs no database — query shape, filter compilation and
the Cypher translator are tested against compiled SQL. CI enforces an 85%
line coverage floor and runs mutation testing on every PR. See
[testing.md](https://hopai.readthedocs.io/en/latest/testing/).

## 📄 License

MIT — see [LICENSE](LICENSE).
