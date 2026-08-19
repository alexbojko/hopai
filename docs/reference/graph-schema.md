# Graph schema

Declare the *shape* of the graph — node types, their properties, and which
edge kinds connect which node types — as plain dataclasses, Pydantic v2
models, or `NodeType`/`EdgeType`/`Property` primitives; all three normalize
to one canonical form. [`06_graph_schema`](../notebooks/06_graph_schema.ipynb)
is the full, worked walkthrough: declaring, reading it back as JSON Schema /
networkx / pydantic / Mermaid, **enforcing** it as real CHECK constraints
(`enforce_schema()`, idempotent and reconciling), asking what would break
first (`schema_violations()`), the endpoint-type opt-in
(`enforce_schema(endpoints=True)`), **inferring** a schema from a graph that
grew first (`infer_schema()`, including sampling on a table too large for a
full scan), and **sharing** the contract across processes
(`save_schema()`/`load_schema()`, backed by a `hopai_schema` metadata table).

Two things that notebook doesn't cover:

- With a schema defined, the Cypher front end can refuse hallucinated
  vocabulary outright: `graph.cypher("MATCH (a:persn) RETURN a",
  strict_schema=True)` raises `CypherError: unknown label 'persn' -- the
  schema declares: company, person`.
- `enforce_schema()`'s CHECK constraints are real SQLAlchemy
  `CheckConstraint` objects attached to `graph.nodes_tbl`/`edges_tbl`, the
  same metadata attachment [Constraints](constraints.md) describes — so
  Alembic `--autogenerate` sees them as declared schema rather than drift
  to propose dropping.

See [hopai.schema](../api/schema.md) for every type's exact fields.

## Relations with their own identity

Every edge here still connects exactly two nodes and carries no identity of
its own beyond its properties — a property-graph limit, not a hopai one; the
same restriction holds in Neo4j and in every `MATCH (a)-[r]->(b)` Cypher
dialect. A relational "link table" recording who referred whom, with a bonus
amount and a date on the referral itself, has no single-edge equivalent here
for the same reason it wouldn't fit a two-column foreign-key pair either: the
relation has its own data, and a third participant that isn't one of the two
endpoints being connected.

The property-graph answer is to **reify the relation as a node** — give it
its own type, put its properties on it, and connect it to every participant
with a plain edge naming that participant's role. See the "Relations with
their own identity" section of
[06 · Graph schema](../notebooks/06_graph_schema.ipynb) for the worked
example: no new API, because there is nothing hopai-specific to reify.
