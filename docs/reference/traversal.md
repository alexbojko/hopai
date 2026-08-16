# Traversal: direction, hop count, and OPTIONAL

```python
Hop(hops=3)                 # exactly 3 edges away
Hop(hops=(1, 6))            # anywhere from 1 to 6 edges away
Hop(direction="backward")   # walk edges INTO the node instead of out of it
```

Forward follows `start_id → end_id`; backward follows `end_id → start_id`
("what points at this?"), and direction is per hop, so one chain can mix
both. [`02_traversal`](../notebooks/02_traversal.ipynb) works through both,
plus `OPTIONAL` — Cypher's `OPTIONAL MATCH` equivalent, valid **only on the
last hop**, and the `ValueError` a mid-chain `optional=True` raises rather
than half-working. See [hopai.hop](../api/hop.md) for `Start`/`Hop`'s exact
fields.
