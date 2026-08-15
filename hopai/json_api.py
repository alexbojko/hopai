"""
hopai.json_api

A JSON-in / JSON-out interface, for callers that can't or shouldn't write
Python: an LLM tool call, a future HTTP endpoint or MCP server, a config
file. Everything here is a thin translation layer over Graph.traverse()
and hopai.filters.parse_filter() -- there is no separate logic to
trust here, just a different way to describe the same traversal.

Spec shape:

    {
      "start": {"where": {"type": "person"}},
      "hops": [
        {"where": {"active": true}, "via": {"kind": "friend"},
         "hops": [1, 4], "direction": "forward"},
        {"where": {"type": "company"}, "hops": 3, "optional": true}
      ]
    }

`where` and `via` accept the same JSON filter grammar as
hopai.filters.parse_filter(): plain objects for equality/AND, plus
{"and": [...]}, {"or": [...]}, {"not": ...}, {"gt": [key, value]},
{"gte": [...]}, {"lt": [...]}, {"lte": [...]}, {"between": [key, lo, hi]}.

`hops` accepts either an integer (exact hop count) or a two-element
array [min, max].

`start` and each hop also accept `near` (a {"field", "vector", ...}
similarity spec or a list of them -- hopai.vectors.parse_near) and `k`,
mirroring Start/Hop's near=/k=. DELIBERATELY absent from the tool
schemas below: a tool-calling model asked to fill in "vector" will
invent plausible floats, and an invented embedding finds confidently
wrong neighbors. These keys exist for HTTP/config callers that hold
real vectors; hopai/vectors.py has the full reasoning, and a test pins
the omission.

An aggregation spec is the same thing plus an `aggregates` object (see
hopai.aggregates for what the forms mean), run with aggregate_json():

    {
      "start": {"where": {"type": "person"}},
      "hops": [{"via": {"kind": "friend"}, "hops": [1, 4]}],
      "aggregates": {"friends": {"fn": "count"},
                     "avg_age": {"fn": "avg", "property": "age"}}
    }
"""

from __future__ import annotations


from .aggregates import parse_aggregate
from .core import Graph, Subgraph
from .filters import parse_filter
from .hop import Hop, Start
from .vectors import parse_boost, parse_near


def _near_of(spec: dict, key: str = "near"):
    return parse_near(spec[key]) if key in spec else None


def _boost_of(spec: dict):
    return parse_boost(spec["boost"]) if "boost" in spec else None


def spec_to_traversal(spec: dict) -> tuple:
    """Convert a JSON spec into (Start, [Hop, ...]). Exposed on its own
    in case you want to inspect or modify the parsed traversal before
    running it."""
    if "start" not in spec:
        raise ValueError("spec must have a 'start' key, e.g. {\"start\": {\"where\": {...}}}")

    start_spec = spec["start"]
    start = Start(
        where=parse_filter(start_spec.get("where")),
        label=start_spec.get("label"),
        near=_near_of(start_spec),
        k=start_spec.get("k"),
        boost=_boost_of(start_spec),
    )

    hops = []
    for h in spec.get("hops", []):
        hops.append(
            Hop(
                where=parse_filter(h.get("where")),
                via=parse_filter(h.get("via")),
                hops=tuple(h["hops"]) if isinstance(h.get("hops"), list) else h.get("hops", 1),
                direction=h.get("direction", "forward"),
                optional=h.get("optional", False),
                label=h.get("label"),
                near=_near_of(h),
                k=h.get("k"),
                via_near=_near_of(h, "via_near"),
                via_k=h.get("via_k"),
                boost=_boost_of(h),
            )
        )
    return start, hops


def traverse_json(graph: Graph, spec: dict) -> dict:
    """Run a traversal described entirely in JSON and return a
    JSON-serializable dict -- the single call an LLM tool integration or
    HTTP handler needs: no Python objects in, no Python objects out.

        result = traverse_json(graph, {
            "start": {"where": {"type": "person"}},
            "hops": [{"where": {"active": True}, "hops": [1, 4]}],
        })
        result["nodes"], result["edges"], result["elapsed_ms"]
    """
    start, hops = spec_to_traversal(spec)
    subgraph: Subgraph = graph.traverse(start, *hops)
    return subgraph.to_dict()


def spec_to_aggregation(spec: dict) -> tuple:
    """Convert a JSON spec into (Start, [Hop, ...], {name: aggregate}).
    The traversal half is exactly spec_to_traversal(); `aggregates` maps
    result names to JSON-form aggregates for hopai.aggregates.parse_aggregate()."""
    if not spec.get("aggregates"):
        raise ValueError(
            'spec must have a non-empty "aggregates" object, e.g. '
            '{"aggregates": {"n": {"fn": "count"}}}'
        )
    start, hops = spec_to_traversal({k: v for k, v in spec.items() if k != "aggregates"})
    aggregates = {name: parse_aggregate(a) for name, a in spec["aggregates"].items()}
    return start, hops, aggregates


def aggregate_json(graph: Graph, spec: dict) -> dict:
    """Run an aggregation described entirely in JSON and return a
    JSON-serializable dict of the named results -- the aggregation
    counterpart of traverse_json(), and the call behind
    AGGREGATE_TOOL_SCHEMA.

        aggregate_json(graph, {
            "start": {"where": {"type": "person"}},
            "hops": [{"via": {"kind": "friend"}, "hops": [1, 4]}],
            "aggregates": {"friends": {"fn": "count"},
                           "avg_age": {"fn": "avg", "property": "age"}},
        })
        # -> {"friends": 42, "avg_age": 31.5}
    """
    start, hops, aggregates = spec_to_aggregation(spec)
    return graph.aggregate(start, *hops, aggregates=aggregates)


def vector_search_json(graph: Graph, spec: dict) -> dict:
    """Run a vector search described entirely in JSON -- the
    vector_search() counterpart of traverse_json(), for HTTP/config
    callers that hold real embeddings. There is deliberately NO tool
    schema for this (see the module docstring): a model cannot supply
    the "vector" values truthfully.

        vector_search_json(graph, {
            "near": {"field": "summary", "vector": [...]},
            "k": 10,
            "target": "nodes",
            "where": {"type": "person"},
        })
        # -> {"results": [{"id": "1", "similarity": 0.93, ...}, ...]}
    """
    if "near" not in spec:
        raise ValueError('spec must have a "near" key, e.g. '
                         '{"near": {"field": "summary", "vector": [...]}}')
    results = graph.vector_search(
        *_as_list(parse_near(spec["near"])),
        target=spec.get("target", "nodes"),
        k=spec.get("k", 10),
        where=parse_filter(spec.get("where")),
        boost=_boost_of(spec),
    )
    return {"results": results}


def _as_list(near) -> list:
    return near if isinstance(near, list) else [near]


# A JSON Schema description of the spec format above, ready to hand
# directly to an LLM function-calling / tool definition. Kept here
# rather than hand-duplicated wherever hopai gets wired into an
# agent framework or MCP server later.
#
# NO "near"/"k" here, and no vector-search schema, ON PURPOSE: an LLM
# filling a "vector" parameter invents floats, and invented embeddings
# find confidently wrong neighbors. Vector search reaches an agent as
# results (via application code that embeds real text), never as a
# tool the model fills in. tests/test_vectors.py pins this omission.
TRAVERSE_TOOL_SCHEMA: dict = {
    "name": "traverse_graph",
    "description": (
        "Traverse a property graph stored in PostgreSQL. Follow edges from a "
        "starting set of nodes through one or more filtered hops, forward or "
        "backward, bounded or ranged depth, and return every node and edge "
        "on a complete matching path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start": {
                "type": "object",
                "description": "The seed set of nodes to begin from.",
                "properties": {
                    "where": {"type": "object", "description": "Filter on starting nodes."}
                },
                "required": ["where"],
            },
            "hops": {
                "type": "array",
                "description": "Ordered list of traversal steps, applied one after another.",
                "items": {
                    "type": "object",
                    "properties": {
                        "where": {"type": "object", "description": "Filter on the node reached by this hop."},
                        "via": {"type": "object", "description": "Filter on edges traversed during this hop."},
                        "hops": {
                            "description": "Exact hop count (integer) or [min, max] range.",
                            "anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}],
                        },
                        "direction": {"type": "string", "enum": ["forward", "backward"]},
                        "optional": {
                            "type": "boolean",
                            "description": "If true, keep prior matches even if this hop finds nothing. Only valid on the last hop.",
                        },
                    },
                },
            },
        },
        "required": ["start"],
    },
}


# The aggregation counterpart. Spelled out rather than sharing objects
# with TRAVERSE_TOOL_SCHEMA because the two genuinely differ: hops here
# have no `optional` (it cannot change an aggregate, so aggregate()
# refuses it) -- a test keeps the shared parts in step instead.
AGGREGATE_TOOL_SCHEMA: dict = {
    "name": "aggregate_graph",
    "description": (
        "Aggregate over the nodes a graph traversal matches: count them, or "
        "sum/average/min/max a numeric property -- computed in the database in one "
        "round trip, without returning the nodes themselves. Takes the same "
        "start/hops traversal spec as traverse_graph. Aggregates run over the "
        "distinct nodes matched by the LAST hop (the starting set when there are "
        "no hops), each node counted once however many paths reach it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start": {
                "type": "object",
                "description": "The seed set of nodes to begin from.",
                "properties": {
                    "where": {"type": "object", "description": "Filter on starting nodes."}
                },
                "required": ["where"],
            },
            "hops": {
                "type": "array",
                "description": "Ordered list of traversal steps, applied one after another.",
                "items": {
                    "type": "object",
                    "properties": {
                        "where": {"type": "object", "description": "Filter on the node reached by this hop."},
                        "via": {"type": "object", "description": "Filter on edges traversed during this hop."},
                        "hops": {
                            "description": "Exact hop count (integer) or [min, max] range.",
                            "anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}],
                        },
                        "direction": {"type": "string", "enum": ["forward", "backward"]},
                    },
                },
            },
            "aggregates": {
                "type": "object",
                "description": (
                    "Aggregates to compute, keyed by the name each result should "
                    "come back under."
                ),
                "minProperties": 1,
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "fn": {"type": "string", "enum": ["count", "sum", "avg", "min", "max"]},
                        "property": {
                            "type": "string",
                            "description": (
                                "Node property to aggregate. Required for "
                                "sum/avg/min/max; count without it counts the "
                                "matched nodes themselves."
                            ),
                        },
                        "distinct": {
                            "type": "boolean",
                            "description": (
                                "If true, equal property values collapse before "
                                "aggregating. Applies to count/sum/avg only."
                            ),
                        },
                    },
                    "required": ["fn"],
                },
            },
        },
        "required": ["start", "aggregates"],
    },
}
