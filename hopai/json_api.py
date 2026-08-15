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

`start` accepts `near` (a {"field", "vector", ...} similarity spec or
a list of them -- hopai.vectors.parse_near), `keep`, and `boost`
(hopai.vectors.parse_boost); each hop accepts those plus `via_near`
and `via_keep`. They mirror Start/Hop exactly. Unknown keys are
REFUSED rather than ignored, because `top_k`/`limit`/`filter` are the
names a model reaches for and silently dropping one answers a
different question than the one asked.

Every vector key is DELIBERATELY absent from the tool schemas below:
a tool-calling model asked to fill in "vector" will invent plausible
floats, and an invented embedding finds confidently wrong neighbors.
They exist for HTTP/config callers that hold real vectors --
traverse_json()/aggregate_json() refuse them unless the caller says
allow_vectors=True, so the invariant is enforced and not merely
advertised. hopai/vectors.py has the full reasoning; a test pins it.

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


#: Keys each spec object accepts. Unknown keys are refused rather than
#: ignored: `top_k`, `limit` and `filter` are exactly the names a model
#: has seen ten thousand times, and silently ignoring them answered a
#: different question than the one asked -- unfiltered, or with the
#: default k. parse_near/parse_aggregate are strict one level down;
#: this is the same rule at the top.
_START_KEYS = {"where", "label", "near", "keep", "boost"}
_HOP_KEYS = {"where", "via", "hops", "direction", "optional", "label",
             "near", "keep", "via_near", "via_keep", "boost"}
_SEARCH_KEYS = {"near", "target", "k", "where", "boost"}
_VECTOR_KEYS = {"near", "keep", "via_near", "via_keep", "boost"}


def _check_keys(spec: dict, allowed: set, what: str) -> None:
    if not isinstance(spec, dict):
        raise TypeError(f"{what} must be an object -- got {type(spec).__name__}")
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(
            f"unknown {what} keys {unknown} -- {what} accepts {sorted(allowed)}"
        )


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

    _check_keys(spec, {"start", "hops"}, "traversal spec")
    start_spec = spec["start"]
    _check_keys(start_spec, _START_KEYS, '"start"')
    start = Start(
        where=parse_filter(start_spec.get("where")),
        label=start_spec.get("label"),
        near=_near_of(start_spec),
        keep=start_spec.get("keep"),
        boost=_boost_of(start_spec),
    )

    hops = []
    for h in spec.get("hops", []):
        _check_keys(h, _HOP_KEYS, '"hops" entry')
        hops.append(
            Hop(
                where=parse_filter(h.get("where")),
                via=parse_filter(h.get("via")),
                hops=tuple(h["hops"]) if isinstance(h.get("hops"), list) else h.get("hops", 1),
                direction=h.get("direction", "forward"),
                optional=h.get("optional", False),
                label=h.get("label"),
                near=_near_of(h),
                keep=h.get("keep"),
                via_near=_near_of(h, "via_near"),
                via_keep=h.get("via_keep"),
                boost=_boost_of(h),
            )
        )
    return start, hops


def _refuse_vectors(spec: dict, caller: str) -> None:
    """The one place the "vectors never travel through the LLM"
    invariant is ENFORCED rather than advertised.

    Omitting the keys from the tool schemas is not a defence: the
    schemas carry no additionalProperties:false, the JSON grammar
    documents `near` for HTTP callers, and this function is what an
    agent integration actually calls. A model that emits a `near` gets
    a plausible subgraph built from invented floats and no complaint --
    a confidently wrong answer, which rule 4 says is the worst thing
    this library can produce. Callers holding REAL vectors say so."""
    for scope in (spec.get("start") or {}, *(spec.get("hops") or [])):
        if not isinstance(scope, dict):
            continue
        present = sorted(_VECTOR_KEYS & set(scope))
        if present:
            raise ValueError(
                f"{caller}: {present} cannot come from a tool call -- a model cannot supply "
                f"a real embedding, and an invented one finds confidently wrong neighbors. "
                f"Embed the text in application code and call graph.vector_search(), or "
                f"pass allow_vectors=True if this spec came from your own code"
            )


def traverse_json(graph: Graph, spec: dict, allow_vectors: bool = False) -> dict:
    """Run a traversal described entirely in JSON and return a
    JSON-serializable dict -- the single call an LLM tool integration or
    HTTP handler needs: no Python objects in, no Python objects out.

        result = traverse_json(graph, {
            "start": {"where": {"type": "person"}},
            "hops": [{"where": {"active": True}, "hops": [1, 4]}],
        })
        result["nodes"], result["edges"], result["elapsed_ms"]

    Similarity keys (`near`/`keep`/`via_near`/`via_keep`/`boost`) are
    REFUSED unless allow_vectors=True: this is the call an agent
    integration wires up, and a model cannot supply a real embedding.
    Application code holding real vectors opts in explicitly.
    """
    if not allow_vectors:
        _refuse_vectors(spec, "traverse_json()")
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


def aggregate_json(graph: Graph, spec: dict, allow_vectors: bool = False) -> dict:
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

    Refuses similarity keys unless allow_vectors=True, for the reason
    traverse_json() gives.
    """
    if not allow_vectors:
        _refuse_vectors(spec, "aggregate_json()")
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
    _check_keys(spec, _SEARCH_KEYS, "vector search spec")
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
        "on a complete matching path. Filters are EXACT property matches: this "
        "tool cannot find nodes by meaning or semantic closeness. If the question "
        "needs that, say so rather than guessing property values -- the "
        "application must run the search and give you node ids to start from."
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
        "no hops), each node counted once however many paths reach it. Filters are "
        "EXACT property matches: this tool cannot select nodes by meaning."
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
