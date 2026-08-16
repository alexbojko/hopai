# Traversal: direction, hop count, and OPTIONAL

```python
Hop(hops=3)                 # exactly 3 edges away
Hop(hops=(1, 6))            # anywhere from 1 to 6 edges away
Hop(direction="backward")   # walk edges INTO the node instead of out of it
```

Forward follows `start_id → end_id`; backward follows `end_id → start_id`,
which is how you ask "what points at this?":

```python
# "Who works at Acme?" — start at the company and walk the works_at
#  edges backwards, to the people on the other end.
graph.traverse(
    Start(where={"name": "Acme"}),
    Hop(via={"kind": "works_at"}, direction="backward", where={"type": "person"}),
)
```

Direction is per hop, so one chain can mix both:

```python
# "Who are Alice's colleagues?" — out to her employer, then back down to
#  everyone else who works there.
graph.traverse(
    Start(where={"name": "Alice"}),
    Hop(via={"kind": "works_at"}),                            # up to the company
    Hop(via={"kind": "works_at"}, direction="backward"),      # back down to its people
)
```


## OPTIONAL

```python
# "List every active person, and the company they work for IF they have
#  one." Without optional=True, the unemployed drop out of the answer
#  entirely; with it, they come back with no company attached.
graph.traverse(
    Start(where={"type": "person", "active": True}),
    Hop(via={"kind": "works_at"}, where={"type": "company"}, optional=True),
)
```

Cypher's `OPTIONAL MATCH`, equivalent: nodes that reach this point in the
chain are kept even if this hop finds nothing for them. **Only valid on
the last hop** — supporting it mid-chain would mean every downstream hop
tolerating a missing anchor, a materially larger feature this library
hasn't built.

