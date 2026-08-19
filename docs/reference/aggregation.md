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

## Grouping

`group_by="<property>"` runs every aggregate once per distinct value of a
property, read off the **same last-hop nodes** the aggregates already run
over — Cypher's `RETURN b.city, count(b)`. The result becomes a **list** of
per-group dicts instead of one:

```python
graph.aggregate(
    Start(where={"type": "person"}), aggregates={"n": Count()}, group_by="city",
)
# -> [{"city": "Berlin", "n": 2}, {"city": None, "n": 1}, ...]
```

A node missing the property groups with every other node that is also
missing it, under `None` — matching `Count(property)`'s own "missing counts
as absent" judgement rather than dropping the row. An **empty match**
returns `[]`, not one row of zeros: GROUP BY produces no rows when there is
nothing to group, unlike the single-row shape a plain (ungrouped) aggregate
returns over an empty match.

The same restriction that keeps a plain aggregate off a mid-chain node
applies to the grouping key too — it can only name a property on the last
step, never an earlier one or an edge, and Cypher's translator (`cypher.py`)
enforces that itself since it is the one front end that lets a query name
any step. `aggregate_json()` takes the identical `"group_by"` key in its
spec. `04_json_and_cypher` and `03_aggregation` both run a grouped example.

See [hopai.aggregates](../api/aggregates.md) for every function's exact
signature.
