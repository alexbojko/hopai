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

See the [Multi-graph](https://hopai.readthedocs.io/en/latest/architecture/#multi-graph)
section of architecture.md for why `graph_id` leads the endpoint indexes and
how `save_schema()`'s metadata table fits the same per-graph discipline.
