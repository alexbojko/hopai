# Many graphs, one database

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

