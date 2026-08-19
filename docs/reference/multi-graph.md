# Many graphs, one database

A new graph is a **string, not a schema** (`Graph(engine, graph="marketing")`,
or `graph.in_graph("tenant-42")` off an existing handle) — it costs a row,
not DDL, so thousands of them are ordinary and one connection pool serves
all of them. Every read and write carries `graph_id = ...`, so graphs are
invisible to each other, cross-graph edges are refused by a composite
foreign key rather than merely discouraged, and constraints (`Unique`,
`Required`, `Check`, `PropertyType`) are scoped per graph.
[`07_many_graphs`](../notebooks/07_many_graphs.ipynb) is the worked demo.
Bringing your own tables with no discriminator column?
`Graph(engine, graph_col=None)` runs a single unscoped graph against them.

A graph is still a string, not a schema — that does not change. What an
optional **`graphs` registry table** (`id`, `name`, `description`) adds is
somewhere to say what a string *means*: a human-readable name and a free-text
description for a `graph_id` that would otherwise only ever appear as an
opaque tenant slug. It is purely descriptive — there is no foreign key from
`nodes.graph_id`/`edges.graph_id` into it, on purpose, so a graph with rows
and no registered name is not an error, just unnamed, and
`Graph(engine, graph="anything")` keeps working exactly as before for a
caller who never opts in. Register one with
`graph.create_graph(name="Marketing", description="Campaign data")` — an
upsert, so calling it again to rename a graph is normal, not a conflict — and
bring your own table, extended past `id`/`name`/`description`, the same way
`node_table=`/`edge_table=` already work: `Graph(engine, graph_table=...)`.
`hopai.mcp`'s `list_graphs` tool reads it with `registry=True`.

See the [Multi-graph](https://hopai.readthedocs.io/en/latest/architecture/#multi-graph)
section of architecture.md for why `graph_id` leads the endpoint indexes and
how `save_schema()`'s metadata table and the `graphs` registry each fit the
same per-graph discipline.
