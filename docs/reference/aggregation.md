# Aggregation

A number instead of a subgraph — computed in the database, over the
**distinct nodes the last hop matched** (the seed set when there are no
hops), each node counted once however many paths reach it.
[`03_aggregation`](../notebooks/03_aggregation.ipynb) works through
`Count`/`Sum`/`Avg`/`Min`/`Max`, `distinct=True`, and why
`PropertyType("age", "number")` is what keeps a stray `"high"` from ever
reaching the aggregate; [`04_json_and_cypher`](../notebooks/04_json_and_cypher.ipynb)
covers the same call as JSON (`aggregate_json`/`AGGREGATE_TOOL_SCHEMA`) and
as Cypher (`RETURN count(DISTINCT b)`).

One thing neither shows: values over an **empty match**. `count` → `0`,
`sum` → `0`, `avg`/`min`/`max` → `None` — plain `int`/`float`/`None`,
JSON-ready, the same way SQL and Cypher aggregates treat an empty set.

See [hopai.aggregates](../api/aggregates.md) for every function's exact
signature.
