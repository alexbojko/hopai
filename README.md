<div align="center">

# 🐘 hopai

**A knowledge graph in the Postgres you already run — no graph database required.**

[![CI](https://github.com/alexbojko/hopai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alexbojko/hopai/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/coverage-%E2%89%A585%25-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Multi-hop traversal, ingestion, updates and real constraints — with a
Python API, a JSON one, and Cypher, so an agent and a developer can both
use it without being taught anything new.

## ✨ Highlights

- 🐘 **Plain PostgreSQL** — two ordinary tables and a recursive CTE. No
  extension, no sidecar service, no new operational dependency.
- 🧭 **Real multi-hop traversal** — bounded and unbounded hops, per-hop
  direction, `OPTIONAL`, rich JSONB filtering, one round trip.
- ✏️ **Update and delete by filter** — `SET` / `REMOVE` /
  `DETACH DELETE` semantics through the same filters a traversal uses,
  with a filterless call refusing rather than emptying the graph.
- 🧮 **In-database aggregation** — `count` / `sum` / `avg` / `min` /
  `max` over what a traversal matches, computed where the data lives
  instead of hydrating a subgraph to count it.
- 🤖 **Three front ends, one engine** — Python, JSON (with a ready-made
  LLM tool schema), and a Cypher subset all compile through the same
  query builder.
- 🔌 **An MCP server in one command** — `hopai-mcp` exposes reading,
  writing, schema and similarity tools over stdio or HTTP, with
  permissions that decide which tools exist at all.
- 🔐 **Constraints Neo4j puts behind an enterprise licence** — unique,
  composite, partial, existence, type and CHECK constraints on JSONB
  properties.
- 🧲 **Vector search without pgvector** — exact cosine similarity on
  plain `real[]` columns, many named fields per node/edge, weighted
  multivector queries, and similarity-seeded traversals. Hand a field
  your OpenAI/Cohere/Voyage/SentenceTransformer client and store text
  instead of floats. Exact means no recall tuning and no index to
  rebuild — and an unfiltered scan is linear in rows × dimensions (see
  [What this costs](#what-this-costs)); `pgvector_exit_ddl()` prints
  the migration for when you outgrow it.
- 🧪 **Tested like it matters** — SQL-level assertions, a live-Postgres
  suite, an 85% coverage gate and mutation testing in CI.
- 📊 **Measured, not claimed** — real benchmark numbers in `benchmarks/`,
  including where raw SQL still wins.

## ⚡ Quick start

```bash
pip install hopai
```

```python
from sqlalchemy import create_engine
from hopai import Graph, Start, Hop, OR, AND, NOT, GT, BETWEEN

graph = Graph(create_engine("postgresql+psycopg2://user:pass@host/db"))

# "Which companies do Alice's friends work for — counting friends of
#  friends, up to four hops out, and only people who are still active?"
result = graph.traverse(
    Start(where={"name": "Alice"}),                        # begin at Alice
    Hop(via={"kind": "friend"},                            # follow `friend` edges
        hops=(1, 4),                                       #   between 1 and 4 of them
        where={"active": True}),                           #   landing only on active people
    Hop(via={"kind": "works_at"},                          # then one `works_at` edge
        where={"type": "company"}),                        #   landing on a company
)

result.nodes            # [{"id": "1", "properties": {...}}, ...]
result.edges            # [{"id": "7", "start_id": "1", "end_id": "2",
                        #   "properties": {...}}, ...]
result.to_networkx()    # in-memory graph, if you have networkx installed
```

Read a traversal left to right as a sentence: **start here → follow these
edges this many times → land on nodes like this → then again**. `via`
filters the *edges* you walk, `where` filters the *node* you arrive at.

You get back the whole matching subgraph — Alice, the friends in between,
the companies, and the real edges connecting them — not just the endpoints.

## 💡 Why

Most "I need graph queries" projects reach for a dedicated graph
database before checking whether they need to. This library is the
other answer: if your data already lives in Postgres, a well-indexed
recursive CTE handles bounded and unbounded traversal, compound
multi-hop patterns, and rich filtering — often faster than a bolted-on
graph extension, and competitively with a real graph database, without
adding an operational dependency. See `benchmarks/` for real, measured
numbers, not a claim.

## 🗄️ Schema

```python
graph.create_schema()   # idempotent; safe to call on every start-up
```

Two tables — a typed identity column plus a JSONB properties bag on
each, and the indexes traversal depends on:

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

`BY DEFAULT`, not `ALWAYS`: ids may be supplied or generated.

`graph_id` arrived after 0.0.1. While the version is `0.0.x` the schema
may change without a migration path — `drop_schema()` and
`create_schema()` are the upgrade. Built-in migrations come later.

Different table or column names? `Graph(engine, node_table=..., edge_table=..., node_id_col=..., ...)`.

**Extending the model** — a field no JSONB property can give you, like a
foreign key to a `users` table, is an ordinary `Column()` on your own
table:

```python
nodes = Table(
    "nodes", metadata,
    Column("id", BigInteger, Identity(always=False), primary_key=True),
    Column("graph_id", Text, nullable=False, server_default="default"),
    Column("properties", JSONB, nullable=False, server_default="{}"),
    Column("user_id", BigInteger, ForeignKey("users.id"), nullable=False),
)
graph = Graph(engine, node_table=nodes)
graph.create_schema()   # emits user_id and its foreign key too

graph.add_nodes([{"id": 1, "user_id": 7, "type": "task"}])   # writes the real column
graph.traverse(Start()).nodes[0]     # {"id": "1", "properties": {"type": "task"}, "user_id": 7}
```

Graph() finds every such **extra column** on its own — nothing to
declare — and it behaves like `id` from then on: written by
`add_nodes`/`add_edges`/`merge_nodes`/`merge_edges` when a row names it,
returned by `traverse()`, and `Col("user_id")` names it for
`Unique`/`Index`/merge's `on=`, the same way `Col("start_id")` already
does. A real `NOT NULL`/`FOREIGN KEY` validates it, for free; what does
*not* extend to it is `update_nodes`/`update_edges` — `set=`/`remove=`
stay JSONB-only, so change an extra column with `UPDATE` through the
engine, or `merge_nodes`/`merge_edges`, which does refresh it on
conflict. See `hopai/models.py`'s "EXTENDING THE MODEL" note.

Naming a property, a constraint key, or a merge `on=` entry the same as
an extra column — easy to do by accident, since `user_id` is your own
project's name, not a universal convention like `id` — is refused, not
guessed:

```python
graph.define_schema(nodes=[NodeType("task", properties=[Property("user_id", "number")])])
# ValueError: NodeType('task'): ['user_id'] already name real column(s) on 'nodes' -- ...

graph.define_constraints(nodes=[Required("user_id")])
# TypeError: 'user_id' is a real column on nodes (...) -- say Col('user_id') if you meant it

graph.merge_nodes([{"user_id": 7, "name": "..."}], on=["user_id"])
# TypeError: 'user_id' is a real column on nodes -- say Col('user_id')
```

Every one of these would otherwise compile a rule against
`properties->>'user_id'`, a JSONB key that name can never reach once
it is an extra column — a correct row would fail forever, or worse for
merge: `ON CONFLICT` would never match, so every "merge" silently
inserts a duplicate instead of updating one. `Col("user_id")` is always
the fix when the real column genuinely is what you mean.

## 🧬 Many graphs, one database

```python
# Three completely separate graphs, one database, one connection pool.
marketing = Graph(engine, graph="marketing")
support   = Graph(engine, graph="support")
tenant    = graph.in_graph(f"tenant-{id}")     # same engine and tables

marketing.add_nodes([{"type": "person", "name": "Alice"}])
support.traverse(Start(where={"name": "Alice"}))   # finds nothing — different graph
```

Every read and every write carries `graph_id = ...`, so the graphs are
invisible to each other. A new graph is a **string, not a schema** — it
costs a row, not DDL, so thousands of them are ordinary and one
connection pool serves all of them.

- `graph_id` **leads** both endpoint indexes, so the discriminator is
  indexed away rather than paid for on every hop.
- **Cross-graph edges are impossible**, not merely discouraged: edges
  carry a composite foreign key `(start_id, graph_id) → nodes(id, graph_id)`.
  Postgres rejects the write.
- **Constraints are per graph.** `Unique("email")` puts `graph_id` first,
  so each graph may have its own `a@x.com`; `Required`/`Check`/`PropertyType`
  are guarded so one graph's rules never bind another's rows.

Bringing your own tables with no discriminator? `Graph(engine, graph_col=None)`
runs a single unscoped graph against them.

## 📥 Getting data in

```python
# "Alice is a person, Acme is a company, and Alice has worked at Acme
#  since 2019."
graph.add_nodes([
    {"type": "person", "name": "Alice"},          # ids assigned by Postgres
    {"type": "company", "name": "Acme"},
])
graph.add_edges([
    {"start": {"name": "Alice"},    # endpoints looked up by property, so you
     "end": {"name": "Acme"},       # never have to juggle generated ids
     "kind": "works_at", "since": 2019},
])
```

A row is written one of two ways, and the rule is one line: **a row with
a `properties` key is nested; any other row is flat, and every key that
isn't an identity key is a property.**

```python
{"id": 1, "type": "person"}                    # flat — what you write by hand
{"id": 1, "properties": {"type": "person"}}    # nested — what a traversal returns
```

The nested form is exactly `result.nodes`, so a subgraph loads into
another graph without reshaping.

Supply `id` yourself when it means something to you, or leave it out and
let Postgres assign one — but **not both in the same call**. A batch where
some rows have an id and others don't is refused, because it would insert
`NULL` for the rest instead of generating them; split it into two calls.

Edges take endpoints as `start_id`/`end_id`, or as `start`/`end`
property dicts matching one existing node each — because whatever just
wrote the nodes usually doesn't know their generated ids. References are
resolved in one batched lookup; matching nothing or several raises.

```python
graph.merge_nodes([{"email": "a@x.com", "name": "Alice"}], on=["email"])
```

`INSERT ... ON CONFLICT DO UPDATE`, needing a `Unique` on the `on` keys.
A match merges the new properties over the old ones and leaves the rest
alone (Cypher's `ON MATCH SET`); `replace=True` overwrites the bag.
Merging is idempotent, which is what makes it the right call for an
agent that might retry.

For agents and HTTP handlers, one document, and one schema to hand a
model:

```python
from hopai import INGEST_TOOL_SCHEMA

graph.ingest({
    "nodes": [{"id": 1, "type": "person"}],
    "edges": [{"start_id": 1, "end_id": 2, "kind": "knows"}],
})
```

Nodes are written before edges, so a single document can create a node
and an edge that references it. `graph.add_networkx(g)` loads a networkx
graph — the inverse of `result.to_networkx()`.

## ✏️ Changing and deleting

The other half of a graph an agent maintains rather than only fills.
`where=` is the same filter language a traversal uses, and it selects a
**set** of rows — every row it matches is changed, exactly as Cypher's
`SET` and `DELETE` do.

```python
# "Everyone over 65 is retired."
graph.update_nodes(where=GT("age", 65), set={"retired": True})

# `set` merges over what's there; `remove` drops keys; `replace=True`
# makes `set` the whole property bag.
graph.update_nodes(where={"email": "a@x.com"}, remove=["nickname"])

# "Alice doesn't know Bob any more." Endpoint filters, so you never have
# to look an id up first.
graph.delete_edges(where={"kind": "knows"},
                   start={"email": "a@x.com"}, end={"email": "b@x.com"})

# "Forget Alice." detach=True deletes her edges with her.
graph.delete_nodes(where={"email": "a@x.com"}, detach=True)

graph.clear()          # this graph, and no other, in one transaction
```

Every call returns a `MutationResult` — `deleted_nodes`,
`deleted_edges`, `updated_nodes`, `updated_edges`, `elapsed_ms` — four
counters because one delete touches both tables.

`start`/`end` are filters, not references: any number of nodes may match
one, and every edge touching any of them goes. That is the opposite of
`add_edges()`, where `start`/`end` must identify exactly one node and
ambiguity raises.

**A call with no filter raises rather than matching everything.**
`where=None` and `where={}` are what an empty variable looks like, and
the cost of being wrong here is the data. Say it on purpose with
`all=True`, or call `clear()`. `all`, `detach` and `replace` must be
real booleans — `all="false"` raises rather than being read as truthy,
because JSON booleans arriving as strings is an ordinary tool-call
failure and this one would empty the graph.

**Deleting a node that still has edges fails**, and the error names
`detach=True` — the composite foreign key is doing its job, and an edge
pointing at a node that no longer exists is exactly the corruption it
was added to prevent.

The same thing in Cypher, and as one JSON document for a tool-calling
model (`MUTATE_TOOL_SCHEMA`):

```python
graph.cypher("MATCH (a:person) WHERE a.age > 65 SET a.retired = true")
graph.cypher("MATCH (a {email: 'a@x.com'})-[r:knows]->() DELETE r")
graph.cypher("MATCH (a {email: 'a@x.com'}) DETACH DELETE a")

graph.mutate({"operations": [                    # in order, one transaction
    {"op": "update_nodes", "where": {"type": "draft"}, "set": {"status": "archived"}},
    {"op": "delete_nodes", "where": {"type": "spam"}, "detach": True},
]})
```

## 🔐 Constraints

Neo4j puts uniqueness, composite and existence constraints behind an
enterprise licence. Postgres has always had them, and a JSONB property
is as constrainable as a column once the expression is indexed:

```python
from hopai import Unique, Required, Check, Index, PropertyType, Col, GT

graph.define_constraints(
    nodes=[
        Required("type"),                            # the key must be present
        Unique("email"),                             # no two nodes share one
        Unique("tenant", "slug"),                    # composite
        Unique("email", where={"type": "person"}),   # only among people
        PropertyType("age", "number"),               # not the string "42"
        Check(GT("age", 0), name="age_positive"),    # any filter, as a CHECK
        Index("type"),                               # plain lookup index
    ],
    edges=[
        Unique(Col("start_id"), Col("end_id"), "kind"),   # one edge of a kind per pair
    ],
)
```

Idempotent, so it belongs next to `create_schema()`. A violation raises
`ConstraintViolation` naming the constraint and the offending row rather
than a driver error. `graph.constraint_ddl(...)` returns the exact SQL
without running it; `graph.drop_constraints(...)` is the inverse.

Every constraint here is a real `sqlalchemy.Index`/`CheckConstraint`
attached to `graph.nodes_tbl`/`edges_tbl`, not DDL kept off to the side
-- so if your project points its own Alembic `target_metadata` at those
tables (or a custom `node_table=`/`edge_table=` you pass to `Graph()`),
`alembic revision --autogenerate` sees hopai's constraints as declared
schema instead of drift to propose dropping. Call `create_schema()` and
`define_constraints(...)` (or their `_ddl` previews, which attach
without running anything) from `env.py` before autogenerate runs, the
same way you would import your own models.

`PropertyType` is worth the line when a model writes your data: an LLM
emitting `"42"` where you expected `42` breaks every numeric comparison
downstream, silently and much later.

`where=` is the one with no Neo4j equivalent at any price — "email is
unique among people" is a partial index, and a partial index is just an
index.

Two SQL semantics to know, both of which are what you want once stated:

- A unique index doesn't constrain rows where the property is **missing**
  (`->>'email'` is NULL, and NULLs repeat). `Unique("email")` means "no
  two share an email", not "everyone has one" — pair it with
  `Required("email")` for both. Neo4j's uniqueness constraint behaves
  the same way.
- Postgres evaluates `CHECK` **before** resolving `ON CONFLICT`, so a
  merge row must satisfy every check on its own even when it is destined
  to update a row that already does.

## 📐 Graph schema

Declare the *shape* of the graph — which node types exist, which
properties each carries, and which edge kinds connect which node types —
as plain dataclasses. An edge class names its endpoints as fields
annotated with node classes:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Person:
    email: str                  # no default -> required
    age: Optional[int] = None   # optional, may be null

@dataclass
class Company:
    name: str

@dataclass
class WorksAt:
    source: Person              # endpoint, not a property
    target: Company             # endpoint, not a property
    since: int                  # property

graph.define_schema(nodes=[Person, Company], edges=[WorksAt])
```

Pydantic v2 models work anywhere a dataclass does, and the same schema
can be spelled with explicit primitives instead — `NodeType`,
`EdgeType`, `Property` — when there is no class to hand over. All three
inputs normalize to the identical canonical form.

Read it back in whichever representation the consumer speaks:

```python
graph.schema           # canonical dataclasses; None until defined
graph.schema_json      # JSON Schema vocabulary — paste it into a system prompt
graph.schema_networkx  # nx.MultiDiGraph meta-graph (pip install hopai[networkx])
graph.schema_pydantic  # generated pydantic models  (pip install hopai[pydantic])
graph.schema_mermaid   # Mermaid flowchart — paste into a ```mermaid fence
```

`schema_mermaid` needs no dependency at all: GitHub, GitLab and most doc
tooling render the string as a picture. For the schema above it draws:

```mermaid
flowchart LR
    person["person (email*, age)"]
    company["company (name*)"]
    person -- works_at --> company
```

(`*` marks required, `!` marks unique — the same markers
`tool_summary()` uses, capped with `+N more` on wide types.)

Declaring is in-memory and instant. **Enforcing** is a separate,
explicit step that compiles the schema to CHECK constraints, so the
database itself rejects a person without an email or an `age` of
`"42"` — whichever door the write came through, Cypher and raw SQL
included:

```python
graph.schema_ddl()      # the exact SQL, without running it
graph.enforce_schema()  # idempotent; re-running after a schema change
                        # drops the rules the schema no longer has
graph.add_nodes([{"type": "person"}])   # ConstraintViolation: email required
```

Same SQLAlchemy-metadata attachment as [Constraints](#-constraints)
above: `enforce_schema()`'s CHECK constraints land as real
`CheckConstraint` objects on `graph.nodes_tbl`/`edges_tbl`, visible to
Alembic `--autogenerate` the same way.

Enforcing on a graph that grew **before** the schema did? `ADD
CONSTRAINT` validates every existing row and fails opaquely on the
first bad one. Ask first:

```python
report = graph.schema_violations()   # read-only; falsy when clean
print(report)   # per rule: row counts + sample ids -- the work list
```

Property schemas go further than presence and type: `values=` (or an
`Enum` annotation) enforces an allowed set, `unique=True` compiles to a
*partial* unique index — unique among rows of that type only — nested
dataclasses become nested object schemas, and `datetime`/`date` map to
strings with a JSON Schema format. And with a schema defined, the
Cypher front end can refuse hallucinated vocabulary outright:

```python
graph.cypher("MATCH (a:persn) RETURN a", strict_schema=True)
# CypherError: unknown label 'persn' -- the schema declares: company, person
```

Enforcement covers property presence and JSON type per node type and
edge kind. Endpoint types ("`works_at` connects only person → company")
need a look at the endpoint *nodes*, which a CHECK can't do — so that
one is an explicit opt-in backed by a constraint trigger, priced
per edge write:

```python
graph.enforce_schema(endpoints=True)
graph.add_edges([{"start_id": robot, "end_id": acme, "kind": "works_at"}])
# ConstraintViolation: works_at connects robot -> company,
#                      but the schema declares: works_at: person -> company
```

**Grew the graph first, never declared anything?** The schema is
sitting in the data, and Postgres can compute it:

```python
inferred, report = graph.infer_schema()   # a few GROUP BYs over JSONB
print(report)      # per-type row counts, untyped rows, 42-vs-"42" conflicts

graph.define_schema(schema=inferred)      # adopt it — your call, not automatic
graph.enforce_schema()                    # chaotic graph, now server-validated
```

Inference stays honest: a property on *every* row of its type infers
required, missing-on-some infers optional, an observed null infers
nullable, and a key holding both `42` and `"42"` infers the type set
`["number", "string"]` plus a report entry — never a silently picked
winner. Rows with no `type`/`kind` can't be invented into a type; they
are counted in the report and left alone. An inferred schema is an
**observation** — nothing is registered or enforced until you adopt it,
which is exactly why `infer_schema()` is a method and not a silent
`.schema` fallback.

Inference is a sequential scan. When the graph is too large for one,
`infer_schema(sample_percent=5)` reads a `TABLESAMPLE SYSTEM (5)` slice
instead — counts become estimates, a rare property can be missed, and
`required` means "on every *sampled* row", so the report leads with
`sampled 5% of rows — counts are estimates` rather than letting an
estimate read as truth.

**Sharing the contract.** A schema declared on one handle is invisible
to every other process on the same database. Persist it, and the
database becomes the single source of truth:

```python
graph.save_schema()    # upserts this graph's schema into hopai_schema
                       # (a small metadata table, created on first save)

# ...meanwhile, in another service:
graph = Graph(same_dsn)
graph.load_schema()    # reads it back, validates, ADOPTS it
graph.enforce_schema() # and the contract enforces from the loaded copy
```

Unlike an inferred schema, a loaded one is adopted on the spot — it was
explicitly declared a contract by whoever saved it. The stored document
is `schema_json` verbatim (readable in `psql`), and loading rebuilds it
through the same validation `define_schema()` runs, so a corrupted row
raises loudly instead of half-loading.

## 🔎 Filters

Anywhere a `where=` or `via=` is accepted:

```python
{"type": "person"}                    # people
{"type": "person", "active": True}    # people who are ALSO active      (AND of keys)
{"type": ["person", "company"]}       # people OR companies             (IN-like)

OR({"type": "person"}, {"type": "company"})            # the same, spelled out
AND(OR({"type": "person"}, {"type": "company"}),
    {"active": True})                                  # ...and active

NOT({"type": "person"})               # everything that is not a person — INCLUDING
                                      # rows with no `type` key at all

GT("age", 18)                         # age > 18        (GTE, LT, LTE likewise)
BETWEEN("age", 18, 65)                # 18 <= age <= 65

lambda col: col.op("->>")("name").op("~")("^A")   # escape hatch: any SQLAlchemy
                                                  # expression — here, names
                                                  # starting with "A"
```

A bare list at the top level (`[{"a": 1}, {"b": 2}]`) raises `TypeError`
rather than being guessed at — it reads ambiguously as "both of these"
to a human, when it would have meant OR. Use `OR(...)` explicitly.

`NOT` is built on JSONB containment specifically because it handles a
missing property correctly (excluded from the positive filter → included
under `NOT`), unlike naive equality-based negation, which treats a
missing property as SQL `NULL` and silently drops it under `NOT` too.
Verified during development to be a real trap, not a hypothetical one —
see `tests/test_hopai.py::test_not_includes_missing_key`.

## 🧭 Direction and hop count

```python
Hop(hops=3)                 # exactly 3 edges away
Hop(hops=(1, 6))            # anywhere from 1 to 6 edges away
Hop(direction="backward")   # walk edges INTO the node instead of out of it
```

Forward follows `start_id → end_id`; backward follows `end_id → start_id`,
which is how you ask "what points at this?":

```python
# "Who works at Acme?" — start at the company and walk the works_at
#  edges backwards, to the people on the other end.
graph.traverse(
    Start(where={"name": "Acme"}),
    Hop(via={"kind": "works_at"}, direction="backward", where={"type": "person"}),
)
```

Direction is per hop, so one chain can mix both:

```python
# "Who are Alice's colleagues?" — out to her employer, then back down to
#  everyone else who works there.
graph.traverse(
    Start(where={"name": "Alice"}),
    Hop(via={"kind": "works_at"}),                            # up to the company
    Hop(via={"kind": "works_at"}, direction="backward"),      # back down to its people
)
```

## 🧩 OPTIONAL

```python
# "List every active person, and the company they work for IF they have
#  one." Without optional=True, the unemployed drop out of the answer
#  entirely; with it, they come back with no company attached.
graph.traverse(
    Start(where={"type": "person", "active": True}),
    Hop(via={"kind": "works_at"}, where={"type": "company"}, optional=True),
)
```

Cypher's `OPTIONAL MATCH`, equivalent: nodes that reach this point in the
chain are kept even if this hop finds nothing for them. **Only valid on
the last hop** — supporting it mid-chain would mean every downstream hop
tolerating a missing anchor, a materially larger feature this library
hasn't built.

## 🧮 Aggregation

A number instead of a subgraph — computed in the database, in one round
trip, with none of the edge-reconstruction or hydration work a traversal
pays for:

```python
from hopai import Count, Sum, Avg, Min, Max

graph.aggregate(
    Start(where={"type": "person"}),
    Hop(via={"kind": "friend"}, hops=(1, 4)),
    aggregates={"friends": Count(), "avg_age": Avg("age"), "oldest": Max("age")},
)
# {"friends": 42, "avg_age": 31.5, "oldest": 87}
```

The aggregates run over the **distinct nodes the last hop matched** (the
seed set when there are no hops), each node counted once however many
paths reach it. `distinct=True` on `Count`/`Sum`/`Avg` collapses equal
property values first — `Sum("age", distinct=True)` adds each age once.
`Count("age")` counts nodes carrying the property; bare `Count()` counts
the nodes themselves.

The values come back as plain `int`/`float`, JSON-ready. Over an empty
match: `count` → `0`, `sum` → `0`, `avg`/`min`/`max` → `None`. A missing,
`null` or non-numeric property value is ignored, the way both SQL and
Cypher aggregates skip `NULL` — one node carrying `"high"` where you
expected a number doesn't error the query (`PropertyType("age", "number")`
is the constraint that keeps such rows out entirely).

The same call in JSON (`AGGREGATE_TOOL_SCHEMA` is the ready-made LLM tool
definition):

```python
from hopai import aggregate_json

aggregate_json(graph, {
    "start": {"where": {"type": "person"}},
    "hops": [{"via": {"kind": "friend"}, "hops": [1, 4]}],
    "aggregates": {"friends": {"fn": "count"},
                   "avg_age": {"fn": "avg", "property": "age"}},
})
```

…and in Cypher, where it earns its own subtlety — see the Cypher section
below:

```python
graph.cypher("MATCH (a:person)-[:friend*1..4]->(b) RETURN count(DISTINCT b)")
# {"count": 42}
```

## 🧲 Vector search

Exact cosine similarity over nodes and edges — computed by Postgres
itself, on plain `real[]` columns. No pgvector, no extension, no
approximate index, and (because it is exact) metadata filtering costs
nothing extra:

```python
from hopai import Vector, Near

graph.define_vectors(nodes=[Vector("summary", 1536), Vector("title", 384)],
                     edges=[Vector("relation", 384)],
                     migrate=True)   # ALTER TABLE, idempotent; vector_ddl() previews it
                                     # (the dimension CHECK it adds is real SA metadata too)

graph.set_vectors(nodes=[{"id": 1, "summary": embedding}])

graph.vector_search(Near("summary", query_embedding), k=10,
                    where={"type": "person"})
# [{"id": "1", "similarity": 0.93, "properties": {...}}, ...]
```

`migrate=True` runs `migrate_vectors()` for you and hands back what it
returned, so the DDL stays visible instead of happening invisibly. Skip it
(`migrate=False`, the default) and call `migrate_vectors()` yourself when
migrations run separately from application code — a deploy step with
schema-changing credentials, kept apart from the read-only process that
declares fields and calls `vector_search()`.

Declare as many fields as you need — each is one migration-managed
column, dimension-checked by the server **per graph**, so two graphs
sharing the tables can give the same field different dimensionality.

Several fields can rank one query together (multivector search), each
with a weight, an optional `min_similarity` floor, and a say over rows
missing its vector:

```python
graph.vector_search(
    Near("summary", q_summary, weight=0.7),
    Near("title",   q_title,   weight=0.3, missing="zero"),  # title optional
    k=10,
)
```

And similarity composes with traversal — seed a walk with the most
similar nodes, keep only the most similar of what a hop reaches, or
follow only the most similar **edges** out of each node:

```python
graph.traverse(
    Start(near=Near("summary", query_embedding), keep=25),  # 25 nearest seeds
    Hop(via={"kind": "cites"}, hops=(1, 3)),
)
graph.traverse(                                     # keep the nearest reached
    Start(where={"type": "paper"}),
    Hop(via={"kind": "cites"}, near=Near("summary", q), keep=10),
)
graph.traverse(                                     # a beam over edges
    Start(where={"type": "paper"}),
    Hop(via_near=Near("relation", q), via_keep=3),  # 3 nearest edges per node
)
graph.aggregate(Start(near=Near("summary", q), keep=100),
                aggregates={"avg_score": Avg("score")})
```

**Several queries at once.** A question expanded into sub-queries is
one statement, not one per query:

```python
graph.vector_search_many([Near("summary", q1), Near("summary", q2)], k=5)
# -> [[...5 hits for q1...], [...5 hits for q2...]]
```

This buys **round trips, not arithmetic** — every query still scores
every candidate, so against a local database it measured 1.08×. The
saving is `N-1` network round trips, which is the real cost when
Postgres isn't on localhost.

**Hybrid ranking.** A numeric property can contribute to the score
alongside similarity. A boost cannot push a row past a
`min_similarity` floor — but it does reorder, so with `k` it changes
which rows come back, and each result's `similarity` is then the
combined score, no longer a cosine in `[-1, 1]`:

```python
graph.vector_search(Near("summary", q), boost=Boost("importance", 0.2), k=10)
```

By default the property is rescaled into `[0, 1]` — similarity's own
scale, not its sign — with a min-max window function over the candidate rows — the ones
still in play after `where=` — before `weight` is applied, so
`Boost("importance", 0.2)` means "20% weight" whatever raw range
`importance` holds: a raw view count in the thousands would otherwise
overwhelm a cosine that never exceeds 1, and the "boost" would replace
the ranking instead of nudging it. `Boost("importance", 0.2,
scale="raw")` opts back into the unscaled coefficient — for a property
you already normalized, or when you want the per-query window function
off the query path.

Two things worth knowing about the traversal forms. A traversal returns
a **subgraph, not a ranking** — the scores and their order don't survive
into the result, so use `vector_search()` when you need them. And with
no hops, `Start(near=…, keep=N)` selects exactly what `vector_search()
` would, minus the score: `near=` on `Start` earns its place because a
traversal cannot be seeded from a list of ids.

### What this costs

Exact means no ANN index, so every candidate row is scored, and the cost is
a measured constant, not a guess: **0.13 µs per vector element** per
candidate row (Postgres 16, one core — `benchmarks/README.md`
has the methodology, including why the similarity is a LATERAL and not a
scalar subquery: the naive form re-evaluates it at every site the query
names it and was measured at 2× the cost for identical results). Worked
through at that rate, a candidate costs about `dimensions × 0.13 µs`:

| rows | 384-dim | 1536-dim |
| ---: | ---: | ---: |
| 10k | ~0.5 s | ~2 s |
| 100k | ~5 s | ~20 s |
| 1M | ~50 s | ~200 s |

That's fine for the shape this is built for — a knowledge graph, filtered
by `where=` down to a manageable candidate set before ranking — but it
should never be a surprise found in production, which is why it's a number
and not a vibe. Two knobs look like they both narrow the search; they do
opposite things to that bill:

- **`where=` reduces cost.** It removes rows *before* they reach the
  LATERAL, the same index-backed filter every other read here uses.
  Measured: a 20k × 384-dim search filtered to 25% of rows dropped from
  ~1.0s to ~0.25s.
- **`min_similarity=` reduces results, not cost.** Every candidate is
  scanned and scored regardless; the bound only drops rows from the
  output *after* scoring — unlike an ANN index's search radius, it never
  skips a candidate. See `Near.min_similarity` in `hopai/vectors.py`.

Outgrowing this is a planned move, not a rewrite: `pgvector_exit_ddl()`
(below) prints the migration onto pgvector whenever the numbers above say
it's time.

### Text in, vectors out

You do not have to produce the floats. Give a field the embedding
client you already have and hopai calls it for you — on the way in, on
the way out, and for the backfill in between:

```python
import openai
from hopai import Vector, Near

graph.define_vectors(nodes=[
    Vector("summary", 1536, source="abstract",
           embed=openai.OpenAI()),          # or cohere, voyage, google...
], migrate=True)

graph.set_vectors(nodes=[{"id": 1, "summary": "a paper about Raft"}])
graph.vector_search(Near("summary", "how do nodes agree?"), k=10)

graph.embed_stale()      # embed every row that has no vector yet
# -> {"nodes": {"summary": {"embedded": ["2", "3"], "skipped": []}}, "edges": {}}
```

`Near`'s second argument takes **either** a vector or the text to embed
into one — a string and a sequence of numbers can never be confused for
one another, so there is no keyword to remember:

```python
Near("summary", "how do nodes agree?")   # embedded by the field's client
Near("summary", [0.12, 0.44, ...])       # you already have the floats
Near("summary", text="[0.1, 0.2]")       # explicit, for a string that
                                         # looks like a serialized vector
```

A string that looks like a serialized vector (`"[0.1, 0.2]"`) is
**refused** rather than embedded, since embedding those characters
ranks against whatever the phrase means and attaches a confident score
to it. `text=` is how you say you meant it.

`source=` names the **property** holding the text and defaults to the
field's own name, so `Vector("title", 768, embed=…)` embeds each row's
`title`. `embed_stale()` reads that property for every row
`stale_vectors()` reports, embeds them, and writes them; rows whose
property is missing or blank come back under `skipped` rather than
raising, because a paper with no abstract legitimately has no abstract
vector.

It is a **backfill, not a one-shot**: one call walks the whole field in
pages of `batch` (default 1000), each its own embed call and its own
transaction, so a million rows cost bounded memory and a run that dies
partway resumes instead of restarting. That paging is a keyset cursor
rather than a `LIMIT` window on purpose — rows that can never be filled
in stay stale forever, and a window would hand back those same rows on
every pass and never reach the work behind them.

**No new dependency.** hopai imports no provider package — not even to
recognize one; clients are matched by module name and duck typing. The
extras are a convenience: `pip install "hopai[openai]"`, `[cohere]`,
`[voyageai]`, `[google]`, `[sentence-transformers]`, or `[embeddings]`
for all of them. Anything with `embed_documents`/`embed_query`
(LangChain), `get_text_embedding_batch`/`get_query_embedding`
(LlamaIndex), a `SentenceTransformer`, or a plain
`callable(texts) -> vectors` works with no extra at all. `Embedder`
wraps whatever you pass and handles the parts that are easy to get
wrong: per-provider batch caps, and the document/query asymmetry that
several providers score differently and that silently costs recall.

This is the one place hopai makes a **network call** — always to the
client you constructed and configured, and always outside the write
transaction, so a provider failure never leaves a half-written batch.

**Transient failures are retried, terminal ones are not**, and the
difference is the point. A 429 or a 503 is the provider saying "later"
and is retried with exponential backoff plus full jitter; a 401 or a
400 fails identically forever, so retrying it only burns your rate
limit to reach the same error more slowly. Which is which is decided by
the HTTP status the exception carries, or its class name when it has
none — hopai imports no provider package, so it cannot name
`RateLimitError`, but it can read a `429`. A `Retry-After` header wins
over the computed backoff, being the only number involved that isn't a
guess.

```python
Embedder(openai.OpenAI(), model="text-embedding-3-small",
         retries=2, backoff=0.5)      # the defaults: 3 attempts, 0.5s doubling
```

Your client almost certainly retries too, and **the two policies
multiply** — three attempts inside three is nine calls. Pick a side:
`Embedder(retries=0)` leaves it to the client, `openai.OpenAI(max_retries=0)`
leaves it to hopai.

When the retries are spent, `EmbeddingError` still carries the
provider's own exception as `__cause__`, for classifying more precisely
than the heuristic can:

```python
try:
    graph.embed_stale()
except EmbeddingError as failed:
    if isinstance(failed.__cause__, openai.RateLimitError):
        ...          # back off and re-run -- embed_stale() resumes
```

Every provider call is logged to the `hopai.embeddings` logger: the
size at `DEBUG`, each retry and every final failure at `WARNING` — a
retry that succeeded is not an error and does not claim to be.

**Re-embedding and the exit door.** `stale_vectors()` lists the rows
with no vector or a vector the current declaration no longer fits (the
window a dimension change opens) — the report behind `embed_stale()`,
and what you loop over yourself for fields you fill in by hand. And if
the exact scan is outgrown, `pgvector_exit_ddl()` prints the migration
onto pgvector — generated without importing or requiring the extension:

```python
for node_id in graph.stale_vectors()["nodes"]["summary"]["missing"]:
    graph.set_vectors(nodes=[{"id": node_id, "summary": embed(text_for(node_id))}])

print("\n".join(graph.pgvector_exit_ddl()))   # one-way; read vectors.py first
```

One honest limit, documented in depth in `hopai/vectors.py`: the search
is an exact scan, linear in the candidates left after filtering — see
[What this costs](#what-this-costs) above for the numbers, and why
`where=` and `min_similarity=` do opposite things to the bill.

And one rule, in one key: a model may send `"text"`, never `"vector"`.
Text is embedded by the field itself, with your client, so the query
embedding comes from the model that wrote the stored ones. A `"vector"`
asked of a model is invented, and an invented embedding finds
confidently wrong neighbors — so it is the single thing the tool
schemas never advertise, and the JSON front ends refuse it unless you
pass `allow_vectors=True` from your own code.

## 🤖 The JSON interface

For callers that shouldn't or can't write Python — an LLM tool call, an
HTTP handler, config-driven traversal:

```python
from hopai import traverse_json

# The same question as the Quick start, in JSON: "which companies do
# Alice's friends — up to four hops out, active only — work for?"
traverse_json(graph, {
    "start": {"where": {"name": "Alice"}},
    "hops": [
        {"via": {"kind": "friend"}, "hops": [1, 4], "where": {"active": True}},
        {"via": {"kind": "works_at"}, "where": {"type": "company"}},
    ],
})
```

Same keys, same meaning, same engine — JSON in, JSON out.

Filters accept the same grammar, spelled as JSON operators:
`{"and": [...]}`, `{"or": [...]}`, `{"not": ...}`, `{"gt": [key, value]}`,
`{"gte": [...]}`, `{"lt": [...]}`, `{"lte": [...]}`, `{"between": [key, lo, hi]}`.

A traversal can also select by **meaning**, with the same `near` a
Python caller writes — a model sends the words, and the field embeds
them with the client your application declared:

```python
traverse_json(graph, {
    "start": {"near": {"field": "summary", "text": "distributed consensus"},
              "keep": 25},
    "hops": [{"via": {"kind": "cites"}, "hops": [1, 3]}],
})
```

`hopai.TRAVERSE_TOOL_SCHEMA` is a ready-to-use JSON Schema for wiring
this into an LLM function-calling definition directly, alongside
`AGGREGATE_TOOL_SCHEMA`, `INGEST_TOOL_SCHEMA`, `MUTATE_TOOL_SCHEMA` and
`VECTOR_SEARCH_TOOL_SCHEMA` — and with a
[graph schema](#-graph-schema) defined, `graph.tool_schemas()` returns
the four traversal/write definitions with *your* node types, edge kinds
and properties summarized into the descriptions, so the model stops
hallucinating labels:

```python
tools = graph.tool_schemas()   # traverse / aggregate / ingest / mutate,
                               # each describing what this graph holds
```

That is every front end, `mutate_graph` included — hand over the subset
you actually want the model to have.

## 🗣️ Cypher as input syntax

For callers who already think in Cypher — reading and writing:

```python
# Create two people and the friendship between them.
graph.cypher("""
    CREATE (a:person {email: 'alice@x.com'})-[:friend]->(b:person {email: 'bob@x.com'})
""")

# "Make sure Alice exists." Creates her the first time, and on every run
# after that just stamps when she was last seen.
graph.cypher("""
    MERGE (a:person {email: 'alice@x.com'})
    ON CREATE SET a.name = 'Alice'
    ON MATCH SET  a.last_seen = 2026
""")

# "Which of Alice's friends, up to four hops out, are active adults?"
graph.cypher("""
    MATCH (a:person {email: 'alice@x.com'})-[:friend*1..4]->(b {active: true})
    WHERE b.age > 18
    RETURN b
""")
```

`graph.cypher()` returns a `Subgraph` for a query that reads, an
`IngestResult` for one that writes, a `MutationResult` for one that
deletes or updates, and a plain `dict` of numbers for one whose `RETURN`
aggregates; `traverse_cypher`, `write_cypher`, `mutate_cypher` and
`aggregate_cypher` are the same thing when you'd rather be explicit.
`cypher_to_traversal`, `graph.cypher_operations` and
`cypher_to_mutations` show the translation — a `(Start, [Hop])` pair,
the ingestion plan, or the mutation plan — without running anything.

Writes compile to the same `add_nodes` / `merge_nodes` / `add_edges` the
Python API calls, in one transaction, with ids from the insert wiring the
edges. Three places writes stop short of Cypher:

- **`MERGE` on a whole path is refused.** Cypher's
  `MERGE (a {…})-[:x]->(b {…})` matches the *entire* pattern and creates
  all of it when it doesn't match, duplicating nodes that already exist.
  Bind the endpoints first, then `MERGE (a)-[:x]->(b)`.
- **`MERGE` needs a unique index** over every property in the pattern —
  those are the keys Cypher matches on. Anything that shouldn't take part
  in matching goes in `ON CREATE SET`. (Cypher needs no index and races
  instead; the error here names the `Unique(...)` to declare.)
- **`MATCH` before a write binds single nodes** by property, one lookup
  each. It doesn't traverse.

`SET`, `REMOVE`, `DELETE` and `DETACH DELETE` compile to the same
`update_nodes` / `delete_edges` / … the Python API calls — all three
`SET` spellings included: `a.x = 1` and `a += {…}` merge, `a = {…}`
replaces. What a `MATCH` means shifts with what follows it, and the
difference is deliberate: before a `CREATE` it names **one** node (an
edge has to attach to exactly one row, so an ambiguous match is an
error), before a `DELETE` or a `SET` it names the **set** of rows to
change, which is what Cypher means by it too.

Three places where Cypher's meaning decides ours, because labels are
properties here and are not in Neo4j:

- **`SET x = {…}` refuses unless the map carries the label or type
  property.** Cypher's `SET n = {map}` replaces *properties* — labels
  survive it, and a relationship's type cannot be changed at all. Here
  both are ordinary properties, so the same query would leave a node no
  `(a:person)` can match and an edge no `[:knows]` can find. Put
  `type: 'person'` in the map, or write `+=`.
- **`SET a.x = null` removes the property**, as it does in Cypher. A
  stored JSON null is *absent* to Cypher and *present* to `Required`, so
  merging one would walk a constraint you declared.
- **`SET` and `REMOVE` apply in order, last writer wins** —
  `SET a.x = 1 REMOVE a.x` and `REMOVE a.x SET a.x = 1` differ.

And one refusal that only exists because deleting is not reading: with
`node_label_key=None`, `MATCH (a:person) DELETE a` has had its only
constraint translated away. On the read path that widens a result set;
here it would empty the graph, so it raises instead.

`strict_schema=True` reaches mutations as well, and is worth more here
than on the read side — a hallucinated label there returns an empty
subgraph, which at least looks like a result, while a delete that
matched nothing reports exactly what a correct delete of an
already-clean graph reports.

The pattern is one node or one relationship. Changing the rows a
multi-hop pattern reached (`MATCH (a)-[:knows]->(b) DELETE b`) is a
traversal driving a write, and refuses — match the rows by their
properties instead. A relationship pattern can still filter both ends:
`MATCH (a {name: 'Alice'})-[r:knows]->(b:person) DELETE r` compiles to
one statement with an endpoint filter on each side. `DELETE a, r`, a
query that both creates and deletes, and `SET a = {…}` after another
assignment to `a` all refuse and say why.

hopai has no label concept, so labels compile to property tests:
`(a:person)` → `{"type": "person"}`, `[:friend]` → `{"kind": "friend"}`.
Change the keys with `node_label_key=` / `edge_type_key=`, or pass
`None` to ignore labels entirely.

Translates: linear `MATCH` chains (including several `MATCH` clauses
joined end to end), `*min..max`, `->` / `<-` per hop, `[:A|B]`, inline
property maps, `WHERE` with `AND`/`OR`/comparisons/`IN`/`IS NULL`,
`all(r IN relationships(p) WHERE ...)` → `via`, `OPTIONAL MATCH` as
the last clause, and aggregating `RETURN` — with rules worth reading:

**Aggregation translates only when it means the same thing.** Cypher
aggregates over result *rows* — one per path — while hopai aggregates
over the distinct nodes the last step matched. So:

- `count(DISTINCT b)`, `sum(DISTINCT b.age)`, `avg(DISTINCT b.age)`,
  `count(DISTINCT b.age)` translate exactly (distinct values are
  distinct values, however many paths there are), as do bare
  `min(b.age)` / `max(b.age)` (an extremum is immune to multiplicity)
  and any bare aggregate on a hopless single-node pattern
  (`MATCH (a:person) RETURN count(a)` — one node, one row).
- `WITH DISTINCT b RETURN avg(b.age)` — Cypher's spelling of hopai's
  native per-matched-node aggregation — is recognized as a unit.
- Bare `count(b)` / `sum(b.age)` / `avg(b.age)` / `count(*)` with hops
  involved count **per path** — a node reachable two ways counts twice —
  which hopai deliberately cannot express. They raise, naming both exact
  rewrites, instead of quietly answering the per-node question.
- Only the *last* node of the chain can be aggregated; grouping
  (`RETURN b.city, count(b)`), relationship-variable aggregates and
  `collect()`/`stdev()`/percentiles raise for now.

Everything else raises `CypherError` naming the rewrite, rather than
translating into something that answers a different question:

- **A plain `RETURN` has no target.** A traversal returns the whole
  matching subgraph, so non-aggregating projections are parsed and
  ignored.
- **`x.k <> v` and `NOT x.k = v` raise.** Cypher evaluates these to
  `NULL` when `k` is missing and drops the row; hopai's containment-based
  `NOT` keeps it. Same spelling, different result set. Write the
  NULL-safe idiom `x.k IS NULL OR x.k <> v`, which maps exactly onto
  `NOT({"k": v})`.
- Also refused: cross-variable `OR` (`a.x = 1 OR b.y = 2`), unbounded
  `*` (pass `max_var_length=N` to cap it), undirected `-[]-`,
  comma-separated patterns, `WITH` (except the `WITH DISTINCT` unit
  above) / `ORDER BY` / `LIMIT`, and `OPTIONAL MATCH` anywhere but last.

## 🔌 MCP server

The same graph as an [MCP](https://modelcontextprotocol.io/) server, so
Claude Desktop, Claude Code, an IDE or an agent framework can use it with
nothing to write:

```bash
pip install "hopai[mcp]"
hopai-mcp --dsn postgresql+psycopg2://user:pass@localhost/db --read-only
```

```jsonc
// claude_desktop_config.json — or any other MCP client's config
{
  "mcpServers": {
    "hopai": {
      "command": "hopai-mcp",
      "args": ["--dsn", "postgresql+psycopg2://user:pass@localhost/db", "--read-only"]
    }
  }
}
```

Eleven tools — traverse, aggregate, Cypher, ingest, update/delete, schema
inference and declaration, and similarity search — each one a call this
README already documents. **Permissions decide which tools exist**:
`--read-only` registers the reading tools only, the default adds writing,
`--allow-mutations` adds deleting, `--allow-ddl` adds `enforce_schema`. A
tool a model cannot see is one it cannot be talked into calling.

Every graph in the database is served unless `--graph` narrows it, and
**search by meaning takes text, never vectors** — you supply the embedding
function, so no tool has anywhere for a model to put invented floats.

📖 **[Full guide](https://hopai.readthedocs.io/en/latest/mcp/)** — client
setup, every tool, every flag, and troubleshooting.

## 🚧 What this doesn't do (yet)

- No disjoint multi-pattern matching (`MATCH (a)-[]->(b), (c)-[]->(d)`
  joined on shared variables) — one linear chain of hops only.
- `OPTIONAL` only on the last hop, not mid-chain.
- Aggregation covers `count`/`sum`/`avg`/`min`/`max` over the last
  step's matched nodes, numeric properties only. No grouping
  (`RETURN b.city, count(b)`), no edge-property aggregates, no
  `stddev`/percentiles, no lexicographic `min`/`max` on strings — each
  refuses with a message rather than approximating.
- Deletes and updates select rows by their properties, not by where a
  traversal arrived: `MATCH (a)-[:knows]->(b) DELETE b` refuses rather
  than guessing which of the two readings you meant. There is no way to
  target a row by its `id` column either — `where=` filters the JSONB
  properties, so give the rows you care about a property you can name
  (and a `Unique` on it).
- `enforce_schema(endpoints=True)` polices endpoint types with a trigger
  on the *edges* table, so retyping a node with `update_nodes` (or
  `merge_nodes(replace=True)`, which could always do it) can leave a
  declared edge connecting types the schema forbids without raising.
  Re-run `enforce_schema()` after retyping nodes.
- Vector search is exact and unindexed by design — no ANN (HNSW/IVF)
  and no late-interaction/ColBERT multivectors; `hopai/vectors.py`
  spells out the cost model and why each refusal is a refusal. Cosine
  is the only metric — on the unit-normalized vectors every current
  embedding API ships, dot and euclidean rank identically anyway.
- Embedding is a thin seam, not a framework: transient failures are
  retried with backoff and jitter, but there is no caching and no rate
  limiting. A cache belongs to the application and a rate limiter
  belongs to the client, which already has one configured the way you
  wanted it.
- A cycle-protection path array is carried on every recursive row. Cheap
  at moderate depth, measurably not-cheap on single-segment traversals
  past roughly 10 hops — see `benchmarks/` for the actual numbers rather
  than a guess.

## ⏱️ Async

`AsyncGraph` covers traversal, aggregation, ingestion, mutation and vector
search/storage for an async app — not a second implementation, but the
same query builders and execute-and-hydrate functions `Graph` already
runs, reached through SQLAlchemy's own sync/async bridge:

```python
from hopai.asyncio import AsyncGraph

graph = AsyncGraph("postgresql+psycopg://user:pass@host/db")   # pip install hopai[asyncio]
result = await graph.traverse(
    Start(where={"type": "person"}),
    Hop(via={"kind": "friend"}, hops=(1, 4)),
)
await graph.mutate({"operations": [
    {"op": "update_nodes", "where": {"type": "draft"}, "set": {"status": "archived"}},
]})
```

Schema and constraint declaration (`create_schema()`, `enforce_schema()`,
`define_constraints()`, ...) stay on the sync `Graph` — one-time setup
calls with no concurrency to gain — and `AsyncGraph` refuses them by name
if called, pointing at the fix. See `hopai/asyncio.py` for the design
(and the benchmark that checked it isn't `asyncio.to_thread()` wearing a
different name before it was written).

## 📓 Runnable documentation

Everything above, as nine notebooks you can execute against a throwaway
database — quick start, traversal semantics, aggregation, the JSON and
Cypher front ends, constraints, graph schema, multi-graph, the SQL
underneath, and vector search:

```bash
pip install -e ".[dev,notebooks]"
docker compose up -d
jupyter lab notebooks/
```

They are executed in CI on every PR, and they assert on the behaviours
that matter, so a notebook cannot quietly document an API this library
no longer has. See [`notebooks/README.md`](notebooks/README.md).

## 🛠️ Development

```bash
pip install -e ".[dev]"
docker compose up -d      # throwaway PostgreSQL matching the default DSN
pytest tests/ -v
ruff check .
```

Most of the suite needs no database at all — query shape, filter
compilation and the Cypher translator are all tested against compiled
SQL. Those that do need one skip cleanly when it isn't there; set
`HOPAI_REQUIRE_DB=1` (as CI does) to make a missing database an error
instead.

CI enforces a **line coverage floor of 85%** and runs **mutation
testing** (`mutmut`) on every PR — a surviving mutant is triaged, not
ignored, because a line a mutation can change in silence is a line no
test is really asserting on.

## 📊 Benchmarking

See `benchmarks/README.md`.

## 📄 License

MIT — see [LICENSE](LICENSE).
