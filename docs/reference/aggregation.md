# Aggregation

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

