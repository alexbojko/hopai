"""
hopai.json_api

A JSON-in / JSON-out interface, for callers that can't or shouldn't write
Python: an LLM tool call, an HTTP endpoint, a config file. It is what
hopai/mcp.py serves over MCP, and what an agent framework wires up
directly. Everything here is a thin translation layer over Graph.traverse()
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

`start` accepts `near` (a {"field", "text"} similarity spec or a list
of them -- hopai.vectors.parse_near), `keep`, and `boost`
(hopai.vectors.parse_boost); each hop accepts those plus `via_near`
and `via_keep`. They mirror Start/Hop exactly. Unknown keys are
REFUSED rather than ignored, because `top_k`/`limit`/`filter` are the
names a model reaches for and silently dropping one answers a
different question than the one asked.

A near spec takes "text" OR "vector", and the difference is the whole
LLM story. "text" is embedded by the field itself, with the client the
application declared -- so the query embedding comes from the same
model that wrote the stored ones, and a tool-calling model can fill it
in as truthfully as any filter. "vector" is DELIBERATELY the one key
parsed and never advertised: asked for floats a model invents
plausible ones, and an invented embedding finds confidently wrong
neighbors. It exists for HTTP/config callers holding real vectors, and
traverse_json()/aggregate_json()/vector_search_json() refuse it unless
the caller says allow_vectors=True -- so the invariant is enforced and
not merely advertised. hopai/vectors.py has the full reasoning; a test
pins it.

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
#: The keys whose values are near specs, and so the only ones that can
#: carry a literal embedding. Everything else in the similarity family
#: -- keep, via_keep, boost -- holds integers and property names a
#: model can supply as truthfully as any filter.
_NEAR_KEYS = ("near", "via_near")


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


def refuse_vectors(spec: dict, caller: str) -> None:
    """The one place the "a VECTOR never travels through the LLM"
    invariant is ENFORCED rather than advertised.

    Only the literal floats are refused. `text` is the model's way in
    and is meant to be used: the field embeds it with the client the
    application declared, so the embedding comes from the same model
    that wrote the stored ones. An invented `vector` cannot -- it finds
    confidently wrong neighbors and reports no error, which rule 4 says
    is the worst thing this library can produce.

    Omitting the key from the tool schemas is not a defence on its own:
    the schemas carry no additionalProperties:false, the JSON grammar
    documents `vector` for callers that hold real floats, and this
    function is what an agent integration actually calls.

    Public, and named without the underscore, because it is not only
    this module's business: hopai/mcp.py embeds the model's TEXT into
    a real vector itself and then passes allow_vectors=True, so it has
    to make this exact refusal first -- against the same keys, with the
    same message. A second copy of the rule is a second place for it to
    rot.

    `spec` itself is the third scope, not just start/hops: a
    vector_search_json() spec carries its near at the top level."""
    for scope in (spec.get("start") or {}, *(spec.get("hops") or []), spec):
        if not isinstance(scope, dict):
            continue
        for key in _NEAR_KEYS:
            for one in _as_list(scope.get(key) or []):
                if isinstance(one, dict) and "vector" in one:
                    raise ValueError(
                        f"{caller}: {key}={{...\"vector\": [...]}} cannot come from a tool "
                        f"call -- a model cannot supply a real embedding, and an invented "
                        f"one finds confidently wrong neighbors. Use "
                        # A placeholder rather than an example field name
                        # when the spec named none: inventing one reads
                        # as "you asked for summary", which they did not.
                        f'{{"field": {one.get("field", "<your field>")!r}, "text": "..."}} '
                        f'and let '
                        f"the field embed it, or pass allow_vectors=True if this spec came "
                        f"from your own code"
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
        refuse_vectors(spec, "traverse_json()")
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
        refuse_vectors(spec, "aggregate_json()")
    start, hops, aggregates = spec_to_aggregation(spec)
    return graph.aggregate(start, *hops, aggregates=aggregates)


def vector_search_json(graph: Graph, spec: dict, allow_vectors: bool = False) -> dict:
    """Run a vector search described entirely in JSON -- the
    vector_search() counterpart of traverse_json(), and the call behind
    VECTOR_SEARCH_TOOL_SCHEMA.

        vector_search_json(graph, {
            "near": {"field": "summary", "text": "how do nodes agree?"},
            "k": 10,
            "target": "nodes",
            "where": {"type": "person"},
        })
        # -> {"results": [{"id": "1", "similarity": 0.93, ...}, ...]}

    Same gate as traverse_json(): a literal "vector" is refused unless
    allow_vectors=True, while "text" is embedded by the field itself
    and is what a model is meant to send.
    """
    _check_keys(spec, _SEARCH_KEYS, "vector search spec")
    if not allow_vectors:
        refuse_vectors(spec, "vector_search_json()")
    if "near" not in spec:
        raise ValueError('spec must have a "near" key, e.g. '
                         '{"near": {"field": "summary", "text": "..."}}')
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
# EXACTLY ONE key is parsed and never advertised: a near spec's
# "vector". A model filling it invents floats, and invented embeddings
# find confidently wrong neighbors -- it exists for callers that hold
# real ones. "text" is its replacement here and is meant to be used:
# the field embeds it with the application's own client, so the query
# embedding comes from the model that wrote the stored ones.
# tests/test_vectors.py pins the omission.


def _near_schema(what: str) -> dict:
    """The near spec as JSON Schema -- everything parse_near() accepts
    except "vector". Built by a function because it appears four times
    and a hand-copied fifth would be the one that drifts."""
    return {
        "description": what,
        "anyOf": [{"$ref": "#/$defs/near"}, {"type": "array", "items": {"$ref": "#/$defs/near"}}],
    }


_NEAR_DEF: dict = {
    "type": "object",
    "properties": {
        "field": {
            "type": "string",
            "description": "Name of the vector field to compare against.",
        },
        "text": {
            "type": "string",
            "description": (
                "The text to search for. The application embeds this with the same "
                "model that produced the stored vectors -- write what you are looking "
                "for in words, and never numbers."
            ),
        },
        "weight": {
            "type": "number",
            "description": (
                "This field's coefficient when several near specs are combined into "
                "one score. Leave it out unless you are ranking on more than one field."
            ),
        },
        "min_similarity": {
            "type": "number",
            "description": (
                "Drop rows scoring below this cosine similarity on this field, "
                "between -1 and 1. A filter, applied before the row limit."
            ),
        },
        "missing": {
            "type": "string",
            "enum": ["exclude", "zero"],
            "description": (
                "What to do with rows that have no vector for this field: drop them "
                "(default), or score them 0 here and let other fields carry the row."
            ),
        },
    },
    "required": ["field", "text"],
}
#: A non-similarity term in a ranked score: hybrid retrieval. Only
#: meaningful alongside `near`, which is what Boost itself refuses
#: without -- said here so a model does not reach for it alone.
_BOOST_DEF: dict = {
    "type": "object",
    "description": (
        "Add a numeric property to the similarity score, so ranking is not by "
        "meaning alone. Only valid alongside `near`."
    ),
    "properties": {
        "property": {"type": "string", "description": "Numeric node property to add in."},
        "weight": {"type": "number", "description": "Its coefficient in the combined score."},
        "missing": {
            "type": "number",
            "description": "Value to use for rows lacking the property. Defaults to 0.",
        },
    },
    "required": ["property", "weight"],
}


TRAVERSE_TOOL_SCHEMA: dict = {
    "name": "traverse_graph",
    "description": (
        "Traverse a property graph stored in PostgreSQL. Follow edges from a "
        "starting set of nodes through one or more filtered hops, forward or "
        "backward, bounded or ranged depth, and return every node and edge "
        "on a complete matching path. `where`/`via` are EXACT property matches; "
        "to select nodes by MEANING instead, give `near` a field and the text to "
        "look for, with `keep` to say how many to keep."
    ),
    "parameters": {
        "type": "object",
        "$defs": {"near": _NEAR_DEF, "boost": _BOOST_DEF},
        "properties": {
            "start": {
                "type": "object",
                "description": "The seed set of nodes to begin from.",
                "properties": {
                    "where": {"type": "object", "description": "Filter on starting nodes."},
                    "near": _near_schema(
                        "Rank the starting nodes by similarity to some text instead of "
                        "(or as well as) filtering them. Needs `keep`."),
                    "keep": {
                        "type": "integer",
                        "description": "How many of the highest-scoring starting nodes to keep.",
                    },
                    "boost": {"$ref": "#/$defs/boost"},
                },
                # A seed set has to come from SOMEWHERE. Before `near`
                # existed this was `required: ["where"]`; dropping it to
                # let a purely semantic seed through advertised `{}` as
                # a valid start, which is "every node in the graph" --
                # the unbounded result of #47, handed to a model as a
                # legal call. Either way in is fine; neither is not.
                "anyOf": [{"required": ["where"]}, {"required": ["near"]}],
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
                        "direction": {"type": "string", "enum": ["forward", "backward"],
                                     "description": "Which way to walk this hop's edges: "
                                                    "forward follows start->end, backward "
                                                    "follows end->start. Default forward."},
                        "optional": {
                            "type": "boolean",
                            "description": "If true, keep prior matches even if this hop finds nothing. Only valid on the last hop.",
                        },
                        "near": _near_schema(
                            "Rank the nodes this hop reaches by similarity to some text. "
                            "Needs `keep`."),
                        "keep": {
                            "type": "integer",
                            "description": "How many of the highest-scoring reached nodes to keep.",
                        },
                        "via_near": _near_schema(
                            "Rank the EDGES this hop traverses by similarity to some text. "
                            "Needs `via_keep`."),
                        "via_keep": {
                            "type": "integer",
                            "description": "How many of the highest-scoring edges to follow.",
                        },
                        "boost": {"$ref": "#/$defs/boost"},
                    },
                },
            },
        },
        "required": ["start"],
    },
}



VECTOR_SEARCH_TOOL_SCHEMA: dict = {
    "name": "search_graph_by_meaning",
    "description": (
        "Find the nodes or edges whose stored text is closest in MEANING to what "
        "you describe -- semantic search, not an exact property match. Returns each "
        "hit with its similarity score. Use this when you do not know the exact "
        "property values to filter on, and traverse_graph when you do."
    ),
    "parameters": {
        "type": "object",
        "$defs": {"near": _NEAR_DEF, "boost": _BOOST_DEF},
        "properties": {
            "near": _near_schema("The field to search and the text to search for."),
            "target": {
                "type": "string",
                "enum": ["nodes", "edges"],
                "description": "What to search. Defaults to nodes.",
            },
            "k": {
                "type": "integer",
                "description": "How many results to return, best first. Defaults to 10.",
            },
            "where": {
                "type": "object",
                "description": (
                    "Optional exact-match filter applied before ranking -- narrowing the "
                    "candidates is what makes this fast."
                ),
            },
            "boost": {"$ref": "#/$defs/boost"},
        },
        "required": ["near"],
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
        "no hops), each node counted once however many paths reach it. "
        "`where`/`via` are EXACT property matches; `near` selects by meaning."
    ),
    "parameters": {
        "type": "object",
        "$defs": {"near": _NEAR_DEF, "boost": _BOOST_DEF},
        "properties": {
            "start": {
                "type": "object",
                "description": "The seed set of nodes to begin from.",
                "properties": {
                    "where": {"type": "object", "description": "Filter on starting nodes."},
                    "near": _near_schema(
                        "Rank the starting nodes by similarity to some text instead of "
                        "(or as well as) filtering them. Needs `keep`."),
                    "keep": {
                        "type": "integer",
                        "description": "How many of the highest-scoring starting nodes to keep.",
                    },
                    "boost": {"$ref": "#/$defs/boost"},
                },
                # A seed set has to come from SOMEWHERE. Before `near`
                # existed this was `required: ["where"]`; dropping it to
                # let a purely semantic seed through advertised `{}` as
                # a valid start, which is "every node in the graph" --
                # the unbounded result of #47, handed to a model as a
                # legal call. Either way in is fine; neither is not.
                "anyOf": [{"required": ["where"]}, {"required": ["near"]}],
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
                        "direction": {"type": "string", "enum": ["forward", "backward"],
                                     "description": "Which way to walk this hop's edges: "
                                                    "forward follows start->end, backward "
                                                    "follows end->start. Default forward."},
                        "near": _near_schema(
                            "Rank the nodes this hop reaches by similarity to some text. "
                            "Needs `keep`."),
                        "keep": {
                            "type": "integer",
                            "description": "How many of the highest-scoring reached nodes to keep.",
                        },
                        "via_near": _near_schema(
                            "Rank the EDGES this hop traverses by similarity to some text. "
                            "Needs `via_keep`."),
                        "via_keep": {
                            "type": "integer",
                            "description": "How many of the highest-scoring edges to follow.",
                        },
                        "boost": {"$ref": "#/$defs/boost"},
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
                        "fn": {"type": "string", "enum": ["count", "sum", "avg", "min", "max"],
                              "description": "Which aggregate to compute. count needs no "
                                             "`property`; sum/avg/min/max do."},
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
