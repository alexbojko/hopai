# hopai

Graph traversal on plain PostgreSQL. Bounded and unbounded multi-hop
queries, `AND`/`OR`/`NOT`/range filters, and a JSON interface built for
tool-calling agents — no separate graph database required.

```python
from sqlalchemy import create_engine
from hopai import Graph, Start, Hop, OR, AND, NOT, GT, BETWEEN

graph = Graph(create_engine("postgresql+psycopg2://user:pass@host/db"))

result = graph.traverse(
    Start(where={"type": "person"}),
    Hop(where={"active": True}, via={"kind": "friend"}, hops=(1, 4)),
    Hop(where={"type": "company"}, hops=3),
)

result.nodes            # [{"id": ..., "properties": {...}}, ...]
result.edges            # [{"start_id": ..., "end_id": ..., "properties": {...}}, ...]
result.to_networkx()    # in-memory graph, if you have networkx installed
```

## Why

Most "I need graph queries" projects reach for a dedicated graph
database before checking whether they need to. This library is the
other answer: if your data already lives in Postgres, a well-indexed
recursive CTE handles bounded and unbounded traversal, compound
multi-hop patterns, and rich filtering — often faster than a bolted-on
graph extension, and competitively with a real graph database, without
adding an operational dependency. See `benchmarks/` for real, measured
numbers, not a claim.

## Schema

Two tables — a typed identity column plus a JSONB properties bag on
each:

```sql
CREATE TABLE nodes (id BIGINT PRIMARY KEY, properties JSONB NOT NULL DEFAULT '{}');
CREATE TABLE edges (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    start_id BIGINT NOT NULL REFERENCES nodes(id),
    end_id   BIGINT NOT NULL REFERENCES nodes(id),
    properties JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX ON edges (start_id);
CREATE INDEX ON edges (end_id);
CREATE INDEX ON nodes USING GIN (properties);
CREATE INDEX ON edges USING GIN (properties);
```

Different table or column names? `Graph(engine, node_table=..., edge_table=..., node_id_col=..., ...)`.

## Filters

```python
{"type": "person"}                          # equality
{"type": "person", "active": True}          # AND of keys, same dict
{"type": ["person", "company"]}             # OR of values, one key (IN-like)
OR({"type": "person"}, {"type": "company"})
AND(OR(...), {"active": True})
NOT({"type": "person"})                     # includes rows missing the key entirely
GT("age", 18) / GTE / LT / LTE
BETWEEN("age", 18, 65)
lambda col: col.op("~")("^A")               # escape hatch: any real SQLAlchemy expression
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

## Direction and hop count

```python
Hop(hops=3)                 # exactly 3 hops
Hop(hops=(1, 6))            # 1 to 6 hops
Hop(direction="backward")   # follow end_id -> start_id ("what points to this")
```

Direction is per-hop — a chain can mix forward and backward steps (a
"who else does X's dependents depend on" query, for instance).

## OPTIONAL

```python
Hop(where=..., optional=True)
```

Cypher's `OPTIONAL MATCH`, equivalent: nodes that reach this point in the
chain are kept even if this hop finds nothing for them. **Only valid on
the last hop** — supporting it mid-chain would mean every downstream hop
tolerating a missing anchor, a materially larger feature this library
hasn't built.

## The JSON interface

For callers that shouldn't or can't write Python — an LLM tool call, an
HTTP handler, config-driven traversal:

```python
from hopai import traverse_json

traverse_json(graph, {
    "start": {"where": {"type": "person"}},
    "hops": [
        {"where": {"active": True}, "via": {"kind": "friend"}, "hops": [1, 4]},
        {"where": {"type": "company"}, "hops": 3, "optional": True},
    ],
})
```

Filters accept the same grammar, spelled as JSON operators:
`{"and": [...]}`, `{"or": [...]}`, `{"not": ...}`, `{"gt": [key, value]}`,
`{"gte": [...]}`, `{"lt": [...]}`, `{"lte": [...]}`, `{"between": [key, lo, hi]}`.

`hopai.TRAVERSE_TOOL_SCHEMA` is a ready-to-use JSON Schema for wiring
this into an LLM function-calling definition directly.

## Cypher as input syntax

For callers who already think in Cypher:

```python
from hopai import traverse_cypher

traverse_cypher(graph, """
    MATCH (a:person)-[:friend*1..4]->(b {active: true})
    WHERE b.age > 18
    RETURN b
""")
```

`cypher_to_traversal(query)` returns the `(Start, [Hop, ...])` pair
instead, if you want to inspect or adjust the translation before running
it.

hopai has no label concept, so labels compile to property tests:
`(a:person)` → `{"type": "person"}`, `[:friend]` → `{"kind": "friend"}`.
Change the keys with `node_label_key=` / `edge_type_key=`, or pass
`None` to ignore labels entirely.

Translates: linear `MATCH` chains (including several `MATCH` clauses
joined end to end), `*min..max`, `->` / `<-` per hop, `[:A|B]`, inline
property maps, `WHERE` with `AND`/`OR`/comparisons/`IN`/`IS NULL`,
`all(r IN relationships(p) WHERE ...)` → `via`, and `OPTIONAL MATCH` as
the last clause.

Everything else raises `CypherError` naming the rewrite, rather than
translating into something that answers a different question:

- **`RETURN` has no target.** A traversal returns the whole matching
  subgraph, so projections are parsed and ignored — and aggregations
  (`RETURN count(a)`) raise, since the caller clearly wanted a number.
- **`x.k <> v` and `NOT x.k = v` raise.** Cypher evaluates these to
  `NULL` when `k` is missing and drops the row; hopai's containment-based
  `NOT` keeps it. Same spelling, different result set. Write the
  NULL-safe idiom `x.k IS NULL OR x.k <> v`, which maps exactly onto
  `NOT({"k": v})`.
- Also refused: cross-variable `OR` (`a.x = 1 OR b.y = 2`), unbounded
  `*` (pass `max_var_length=N` to cap it), undirected `-[]-`,
  comma-separated patterns, `WITH` / `ORDER BY` / `LIMIT`, and
  `OPTIONAL MATCH` anywhere but last.

## What this doesn't do (yet)

- No disjoint multi-pattern matching (`MATCH (a)-[]->(b), (c)-[]->(d)`
  joined on shared variables) — one linear chain of hops only.
- `OPTIONAL` only on the last hop, not mid-chain.
- Synchronous only — every call blocks; no `AsyncSession` support yet.
- A cycle-protection path array is carried on every recursive row. Cheap
  at moderate depth, measurably not-cheap on single-segment traversals
  past roughly 10 hops — see `benchmarks/` for the actual numbers rather
  than a guess.

## Development

```bash
pip install -e ".[dev]"
export HOPAI_TEST_DSN="postgresql+psycopg2://user:pass@localhost/db"
pytest tests/ -v
```

## Benchmarking

See `benchmarks/README.md`.

## License

MIT
