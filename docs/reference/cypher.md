# Cypher as input syntax

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

