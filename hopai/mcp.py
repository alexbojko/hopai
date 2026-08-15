"""
hopai.mcp

The graph as an MCP server: one opt-in command that hands any client
speaking the protocol -- Claude Desktop, an IDE, an agent framework --
a set of tools for reading, writing and shaping this graph.

    pip install "hopai[mcp]"

    hopai-mcp --dsn postgresql+psycopg2://user:pass@host/db
    hopai-mcp --dsn ... --transport http --port 8000

...or from Python, which is what you want as soon as the graph needs
setting up first -- or an embedding function (see SIMILARITY):

    from hopai import Graph
    from hopai.mcp import serve

    graph = Graph(dsn)
    graph.define_schema(nodes=[Person, Company], edges=[WorksAt])
    serve(graph, embed=my_embedder)          # stdio; transport="http" for HTTP

THE TOOLS, and the call each one is:

    describe_graph    what exists: the declared schema, the vector
                      fields, what this server will and will not do
    traverse_graph    traverse_json()      -- multi-hop, filtered
    aggregate_graph   aggregate_json()     -- count/sum/avg/min/max
    cypher            Graph.cypher()       -- read or write, one syntax
    search_similar    Graph.vector_search()          (needs `embed`)
    ingest_graph      Graph.ingest()       -- create, or merge on keys
    infer_schema      Graph.infer_schema() -- observe, adopt nothing
    define_schema     Graph.define_schema() + save_schema()
    enforce_schema    Graph.schema_violations() / enforce_schema()

PERMISSIONS. Three levels, because "an agent may read this graph" and
"an agent may run ALTER TABLE on it" are not the same sentence:

    read_only=True    the four read tools only
    (default)         the above plus ingest_graph, define_schema, and
                      writing Cypher
    allow_ddl=True    the above plus enforce_schema, which runs DDL

The gate is which tools get REGISTERED, not a check inside a handler:
a tool the model cannot see is a tool it cannot be talked into calling.

SIMILARITY, and the rule it must not break. Vectors never pass through
a tool schema -- a model asked for an embedding invents one, and an
invented embedding finds confidently wrong neighbors (hopai/vectors.py
argues this at length). So the model here sends TEXT, and this server
embeds it with the `embed` callable the operator wired up:

    serve(graph, embed=lambda text: openai_embedding(text))

That is the whole of it: `search_similar` appears only when `embed` is
configured, `traverse_graph`/`aggregate_graph` grow a `start.search`
string that seeds the walk from the most similar nodes, and no tool
anywhere accepts a list of floats. json_api.refuse_vectors() -- the one
place that invariant is enforced -- still runs on every spec BEFORE the
embedding is injected, so a model that invents a `near` is refused by
the same code that refuses it in traverse_json().

NOT HERE: delete. hopai has no delete API, in Python or in Cypher, so
this server has no delete tool and `cypher` refuses DELETE with the
message cypher.py already gives. A tool that quietly deleted "just the
node" and left its edges, or approximated the missing feature any other
way, is the failure mode this whole library is written against.

NO AUTHENTICATION. Over stdio the client is the process that spawned
this one, which is the trust boundary. Over HTTP it binds 127.0.0.1 by
default and anyone who can reach the port gets the tools that are
registered -- put it behind something that authenticates before you
give it an address, and give it a read_only server unless writes are
genuinely the point.

This module is a FRONT END, like json_api.py and cypher.py: every tool
is a translation into a call one of those already exposes, and there is
no query logic here to review. What IS here and worth reading: the
1.x/2.x SDK adapter in _sdk(), and _register(), which advertises hopai's
hand-written JSON Schemas instead of the ones the SDK would derive from
these handlers' Python signatures.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Optional, get_type_hints

from .core import Graph
from .json_api import aggregate_json, refuse_vectors, traverse_json, vector_search_json
from .models import DEFAULT_GRAPH

#: Server-level guidance, sent to the client at connection time. What a
#: model has to know before its first call and cannot see in any single
#: tool's schema.
SERVER_INSTRUCTIONS = """\
This server exposes one property graph stored in PostgreSQL.

Call describe_graph first. It returns the node types, edge kinds and
property names that actually exist here, and guessing those is the main
way these tools return an empty result instead of an answer.

Filters are EXACT property matches, never substring or fuzzy matches.
A traversal returns a SUBGRAPH -- every node and edge on a matching
chain -- not a ranked list, and ids come back as strings.

Errors from these tools name the fix. Read the message and correct the
call rather than retrying it unchanged.\
"""

#: The sentence TRAVERSE_TOOL_SCHEMA/AGGREGATE_TOOL_SCHEMA use to tell a
#: model that meaning-based lookup is not available. It stops being true
#: the moment an `embed` callable is wired up, so it is REPLACED (not
#: appended to) rather than leaving the description arguing with itself.
_NO_MEANING = (
    "Filters are EXACT property matches: this tool cannot find nodes by meaning "
    "or semantic closeness. If the question needs that, say so rather than guessing "
    "property values -- the application must run the search and give you node ids "
    "to start from."
)
_NO_MEANING_AGGREGATE = (
    "Filters are EXACT property matches: this tool cannot select nodes by meaning."
)
_WITH_MEANING = (
    "Filters are EXACT property matches. To select nodes by MEANING instead, put the "
    "question's own text in start.search: this server embeds it and seeds the "
    "traversal with the most similar nodes. Never invent a vector; there is nowhere "
    "to put one."
)

#: Default seed size for start.search. Big enough that a real answer is
#: usually inside it, small enough that the walk stays cheap.
DEFAULT_KEEP = 25


@dataclass(frozen=True)
class ToolSpec:
    """One MCP tool: what the model is told, and what runs.

    `call` is an ordinary synchronous function whose parameters ARE the
    tool's arguments, so a test (and any other front end) can call it
    with no SDK installed and no server running. `parameters` is the
    JSON Schema the model actually receives -- hopai's own, not one
    derived from `call`'s annotations; _register() explains why."""
    name: str
    description: str
    parameters: dict
    call: Callable[..., Any]


# ---------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------

def _object(properties: dict, required: Optional[list] = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


def _described(graph: Graph, description: str) -> str:
    """A description with this graph's declared vocabulary appended --
    the same treatment Graph.tool_schemas() gives the three static
    schemas, applied to the tools this module defines itself."""
    if graph.schema is None:
        return description
    from .schema import tool_summary
    return f"{description} {tool_summary(graph.schema)}"


def _vector_fields(graph: Graph, target: str) -> dict:
    return (graph.vectors or {}).get(target) or {}


def _resolve_field(graph: Graph, target: str, requested: Optional[str], caller: str) -> str:
    """Which vector field a text query ranks against.

    One declared field needs no argument (and the tool schema does not
    offer one); several make the choice the caller's, because ranking
    against the wrong one returns confident nonsense rather than an
    error."""
    fields = _vector_fields(graph, target)
    if not fields:
        raise ValueError(
            f"{caller}: no vector fields are defined for {target} in this graph -- the "
            f"server was started without define_vectors({target}=[Vector(name, dims)])"
        )
    if requested is not None:
        if requested not in fields:
            raise ValueError(
                f"{caller}: no vector field {requested!r} for {target} -- "
                f"defined: {sorted(fields)}"
            )
        return requested
    if len(fields) > 1:
        raise ValueError(
            f"{caller}: this graph defines several {target} vector fields "
            f"({sorted(fields)}) -- name the one to search in `field`"
        )
    return next(iter(fields))


def _seed(graph: Graph, embed: Optional[Callable], start: Any, caller: str) -> dict:
    """The model's `start` object, with `search` text turned into a real
    `near` seed.

    The order matters and is the point: the caller's keys are refused by
    json_api.refuse_vectors() BEFORE anything is injected, so `search`
    is the only route from a model to a similarity search, and an
    invented `near` is rejected by the same function that rejects it in
    traverse_json()."""
    if not isinstance(start, dict):
        raise TypeError(f"{caller}: `start` must be an object, got {type(start).__name__}")
    start = dict(start)
    search = start.pop("search", None)
    field = start.pop("search_field", None)
    keep = start.pop("keep", None)
    refuse_vectors({"start": start}, caller)

    if search is None:
        if keep is not None or field is not None:
            raise ValueError(
                f"{caller}: start.keep/start.search_field only mean something with "
                f"start.search -- they say how many of the most similar nodes to seed "
                f"from, and how to rank them"
            )
        return start
    if embed is None:
        raise ValueError(
            f"{caller}: start.search needs an embedding function and this server has "
            f"none -- start it with serve(graph, embed=...) to search by meaning, or "
            f"filter on properties with start.where"
        )
    if not isinstance(search, str) or not search.strip():
        raise ValueError(f"{caller}: start.search must be a non-empty string, got {search!r}")
    name = _resolve_field(graph, "nodes", field, caller)
    start["near"] = {"field": name, "vector": embed(search)}
    start["keep"] = DEFAULT_KEEP if keep is None else keep
    return start


def _seeds_from_text(graph: Graph, embed: Optional[Callable]) -> bool:
    """Whether `start.search` can work at all.

    An embedder is not enough: a seed ranks NODE vectors, so a graph
    that declares only edge vector fields can offer `search_similar`
    over its edges and still have nothing to seed a traversal from.
    Advertising the key anyway would put a parameter in front of a
    model that fails on every use -- found by a surviving mutant, which
    is what mutation triage is for."""
    return embed is not None and bool(_vector_fields(graph, "nodes"))


def _search_keys(graph: Graph) -> dict:
    """The `search` half of a start object's schema, offered only when
    this server can actually embed text."""
    fields = _vector_fields(graph, "nodes")
    keys = {
        "search": {
            "type": "string",
            "description": (
                "Text to seed the traversal from by MEANING: the server embeds it and "
                "starts from the most similar nodes. Combine with `where` to restrict "
                "which nodes may be seeded. Send the question's own words -- never a vector."
            ),
        },
        "keep": {
            "type": "integer",
            "description": f"How many of the most similar nodes to seed from. "
                           f"Default {DEFAULT_KEEP}. Only with `search`.",
        },
    }
    if len(fields) > 1:
        keys["search_field"] = {
            "type": "string",
            "enum": sorted(fields),
            "description": "Which vector field `search` ranks against.",
        }
    return keys


def _with_search(schema: dict, graph: Graph, sentence: str) -> dict:
    """A traversal/aggregation tool schema, taught about start.search.

    The `cannot find nodes by meaning` sentence is replaced rather than
    added to: a description that says both things is worse than either,
    and a model reading the refusal will not try the feature that is
    right there."""
    if sentence not in schema["description"]:
        raise RuntimeError(
            f"hopai.mcp expected to replace this sentence in {schema['name']}'s "
            f"description and did not find it: {sentence!r}. It moved or was reworded "
            f"-- update _NO_MEANING/_NO_MEANING_AGGREGATE in hopai/mcp.py to match, or "
            f"the two descriptions will contradict each other"
        )
    schema["description"] = schema["description"].replace(sentence, _WITH_MEANING)
    start = schema["parameters"]["properties"]["start"]
    start["properties"].update(_search_keys(graph))
    # `where` alone is required on start in the static schema. With
    # `search` as a second way in, keeping that would forbid a purely
    # semantic seed -- and dropping it without a replacement would stop
    # saying that a seed set has to come from SOMEWHERE.
    start.pop("required", None)
    start["anyOf"] = [{"required": ["where"]}, {"required": ["search"]}]
    return schema


# ---------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------

def _describe_tool(graph: Graph, read_only: bool, allow_ddl: bool,
                   embed: Optional[Callable]) -> ToolSpec:
    def describe_graph(counts: bool = False) -> dict:
        schema = graph.schema
        vectors = {target: {name: field.dimensions
                            for name, field in _vector_fields(graph, target).items()}
                   for target in ("nodes", "edges")}
        result = {
            "graph": graph.graph,
            "schema": schema.to_json() if schema is not None else None,
            "schema_mermaid": schema.to_mermaid() if schema is not None else None,
            "vector_fields": vectors,
            "search_by_meaning": bool(embed) and bool(vectors["nodes"]),
            "writes_allowed": not read_only,
            "ddl_allowed": allow_ddl,
            "refusals": [
                "No delete: hopai has no delete API, in Python or in Cypher. Deleting "
                "rows is a database operation someone does with SQL, not a tool here.",
                "No fuzzy or substring matching: property filters are exact.",
                "No grouping (RETURN b.city, count(b)) and no edge-property aggregates.",
            ],
        }
        if schema is None:
            result["note"] = (
                "No schema is declared on this server, so node types and property names "
                "are unknown to it. infer_schema observes what the rows actually contain."
            )
        if counts:
            from .aggregates import Count
            from .hop import Start
            result["counts"] = graph.aggregate(Start(), aggregates={"nodes": Count()})
        return result

    return ToolSpec(
        name="describe_graph",
        description=(
            "Describe this graph before querying it: its declared schema (node types, "
            "edge kinds, property names and types), its vector fields, what this server "
            "is permitted to do, and what it refuses. Call this first -- the other tools "
            "match property values EXACTLY, so a guessed type name returns an empty "
            "result rather than an error."
        ),
        parameters=_object({
            "counts": {
                "type": "boolean",
                "description": "Also count the nodes in this graph. Costs a scan; "
                               "leave it off unless the size is the question.",
            },
        }),
        call=describe_graph,
    )


def _traverse_tool(graph: Graph, schema: dict, embed: Optional[Callable]) -> ToolSpec:
    if _seeds_from_text(graph, embed):
        schema = _with_search(schema, graph, _NO_MEANING)

    def traverse_graph(start: dict, hops: Optional[list] = None) -> dict:
        spec = {"start": _seed(graph, embed, start, "traverse_graph"), "hops": hops or []}
        refuse_vectors({"hops": spec["hops"]}, "traverse_graph")
        # allow_vectors: every vector in this spec was put there by
        # _seed() from an embedding of the model's TEXT, and both
        # refusals above have already run against what the model sent.
        return traverse_json(graph, spec, allow_vectors=True)

    return ToolSpec(schema["name"], schema["description"], schema["parameters"], traverse_graph)


def _aggregate_tool(graph: Graph, schema: dict, embed: Optional[Callable]) -> ToolSpec:
    if _seeds_from_text(graph, embed):
        schema = _with_search(schema, graph, _NO_MEANING_AGGREGATE)

    def aggregate_graph(start: dict, aggregates: dict, hops: Optional[list] = None) -> dict:
        spec = {"start": _seed(graph, embed, start, "aggregate_graph"),
                "hops": hops or [], "aggregates": aggregates}
        refuse_vectors({"hops": spec["hops"]}, "aggregate_graph")
        return aggregate_json(graph, spec, allow_vectors=True)

    return ToolSpec(schema["name"], schema["description"], schema["parameters"], aggregate_graph)


def _search_tool(graph: Graph, embed: Callable) -> ToolSpec:
    node_fields = sorted(_vector_fields(graph, "nodes"))
    edge_fields = sorted(_vector_fields(graph, "edges"))

    def search_similar(query: str, k: int = 10, target: str = "nodes",
                       where: Optional[dict] = None, field: Optional[str] = None) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"search_similar: query must be a non-empty string, got {query!r}")
        name = _resolve_field(graph, target, field, "search_similar")
        return vector_search_json(graph, {
            "near": {"field": name, "vector": embed(query)},
            "target": target,
            "k": k,
            "where": where,
        })

    properties = {
        "query": {
            "type": "string",
            "description": "The text to find matches for, in its own words. The server "
                           "embeds it -- never send numbers or a vector.",
        },
        "k": {"type": "integer", "description": "How many results to return. Default 10."},
        "target": {"type": "string", "enum": ["nodes", "edges"],
                   "description": "Search node vectors (default) or edge vectors."},
        "where": {"type": "object",
                  "description": "Exact-match property filter applied BEFORE ranking, in the "
                                 "same filter language as traverse_graph."},
    }
    if len(node_fields) + len(edge_fields) > 1:
        properties["field"] = {
            "type": "string",
            "enum": sorted(set(node_fields) | set(edge_fields)),
            "description": f"Which vector field to rank against. Node fields: "
                           f"{node_fields or 'none'}. Edge fields: {edge_fields or 'none'}.",
        }
    return ToolSpec(
        name="search_similar",
        description=_described(graph, (
            "Find the nodes (or edges) whose stored embedding is closest in MEANING to a "
            "piece of text -- the way in when you do not know the exact property values to "
            "filter on. Returns each match with a `similarity` score, most similar first, "
            "and no vectors. Use traverse_graph's start.search to walk outward from such "
            "matches instead of just listing them."
        )),
        parameters=_object(properties, ["query"]),
        call=search_similar,
    )


def _cypher_tool(graph: Graph, read_only: bool, strict_schema: bool) -> ToolSpec:
    from .cypher import classify_cypher

    def cypher(query: str) -> dict:
        if read_only and classify_cypher(query) == "write":
            raise ValueError(
                "this server is read-only and that query writes -- run a MATCH instead, "
                "or ask the operator for a server started without read_only=True"
            )
        result = graph.cypher(query, **({"strict_schema": True} if strict_schema else {}))
        # Subgraph and IngestResult both know how to serialize themselves;
        # an aggregating RETURN already comes back as a plain dict.
        return result.to_dict() if hasattr(result, "to_dict") else result

    reading = (
        "MATCH with one linear chain of hops, `*1..4` ranges, WHERE, and an aggregating "
        "RETURN (count/sum/avg/min/max, DISTINCT). A non-aggregating RETURN is ignored: a "
        "read returns the whole matching subgraph."
    )
    writing = (
        " CREATE and MERGE write; MERGE on a relationship needs both endpoints bound first, "
        "and MERGE needs a unique index on exactly the properties it matches."
        if not read_only else
        " This server is READ-ONLY: CREATE and MERGE are refused."
    )
    return ToolSpec(
        name="cypher",
        description=_described(graph, (
            f"Run a Cypher query against this graph. Supported: {reading}{writing} "
            f"Labels compile to the `type` property and relationship types to `kind`. "
            f"Anything outside the supported subset is REFUSED with a message naming the "
            f"rewrite -- including DELETE, which does not exist here at all, and bare `<>` "
            f"negation, whose Cypher meaning differs from this engine's. Read the refusal "
            f"and rewrite; do not retry the same query."
        )),
        parameters=_object({
            "query": {"type": "string", "description": "The Cypher query."},
        }, ["query"]),
        call=cypher,
    )


def _ingest_tool(graph: Graph, schema: dict) -> ToolSpec:
    def ingest_graph(nodes: Optional[list] = None, edges: Optional[list] = None,
                     merge_nodes_on: Optional[list] = None,
                     merge_edges_on: Optional[list] = None) -> dict:
        document = {"nodes": nodes or [], "edges": edges or []}
        result = graph.ingest(document, merge_nodes_on=merge_nodes_on,
                              merge_edges_on=merge_edges_on)
        return result.to_dict()

    schema["parameters"]["properties"]["merge_nodes_on"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "UPDATE instead of insert: property keys identifying an existing node. A node "
            "matching on these keys has the new properties merged over its old ones; one "
            "that matches nothing is created. Needs a unique index on exactly these keys."
        ),
    }
    schema["parameters"]["properties"]["merge_edges_on"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": "The same for edges.",
    }
    return ToolSpec(schema["name"], schema["description"], schema["parameters"], ingest_graph)


def _infer_schema_tool(graph: Graph) -> ToolSpec:
    def infer_schema(sample_percent: Optional[float] = None) -> dict:
        schema, report = graph.infer_schema(sample_percent=sample_percent)
        return {"schema": schema.to_json(), "report": asdict(report), "adopted": False,
                "note": "Nothing was declared: this is an observation of the rows that "
                        "exist. define_schema adopts it as the contract."}

    return ToolSpec(
        name="infer_schema",
        description=(
            "Observe what this graph actually contains and derive a schema from it: node "
            "types from the `type` property, edge kinds and their endpoint pairs, and which "
            "properties are always present. Returns the schema plus a report of what the "
            "schema cannot say -- row counts per type, untyped rows, and properties seen "
            "with conflicting JSON types (42 versus \"42\"). Declares NOTHING; pass the "
            "result to define_schema to adopt it. Reads every row, so it is a scan: use "
            "sample_percent on a large graph."
        ),
        parameters=_object({
            "sample_percent": {
                "type": "number",
                "description": "Read a random sample of this percentage of rows instead of "
                               "all of them (e.g. 5). Counts become estimates and rare "
                               "properties can be missed.",
            },
        }),
        call=infer_schema,
    )


def _define_schema_tool(graph: Graph) -> ToolSpec:
    from .schema import schema_from_document

    def define_schema(schema: dict, save: bool = True) -> dict:
        declared = graph.define_schema(schema=schema_from_document(schema))
        if save:
            graph.save_schema()
        return {"schema": declared.to_json(), "saved": bool(save),
                "mermaid": declared.to_mermaid(),
                "note": "Declared, not enforced: existing rows were not checked and future "
                        "writes are not yet validated by the database. enforce_schema does "
                        "that. The tool descriptions in this session still describe the "
                        "schema as it was when the server started; describe_graph reads "
                        "the live one."}

    property_schema = {
        "type": "object",
        "description": "JSON Schema for one type's properties: {\"type\": \"object\", "
                       "\"properties\": {\"email\": {\"type\": \"string\", \"unique\": true}}, "
                       "\"required\": [\"email\"]}. Property types are the JSON type names "
                       "(string, number, boolean, object, array, null), or a list of them "
                       "when a property is nullable.",
    }
    return ToolSpec(
        name="define_schema",
        description=_described(graph, (
            "Declare what this graph is supposed to contain: node types with typed "
            "properties, and edge kinds with the (source type -> target type) pairs they "
            "connect. The declaration is the contract other tools describe and validate "
            "against, and it is saved into the database so other processes load the same "
            "one. It REPLACES any current declaration -- send the whole schema, not a "
            "patch, and call describe_graph first if you mean to extend what exists. "
            "Declaring does not change or check a single row; enforce_schema does that."
        )),
        parameters=_object({
            "schema": _object({
                "nodes": {
                    "type": "object",
                    "description": "Node types keyed by type name.",
                    "additionalProperties": property_schema,
                },
                "edges": {
                    "type": "array",
                    "description": "Edge kinds. The same kind between two different type "
                                   "pairs is two entries.",
                    "items": _object({
                        "kind": {"type": "string"},
                        "source": {"type": "string", "description": "A declared node type."},
                        "target": {"type": "string", "description": "A declared node type."},
                        "properties": property_schema,
                    }, ["kind", "source", "target"]),
                },
            }),
            "save": {
                "type": "boolean",
                "description": "Persist the declaration for other processes. Default true.",
            },
        }, ["schema"]),
        call=define_schema,
    )


def _enforce_schema_tool(graph: Graph) -> ToolSpec:
    def enforce_schema(dry_run: bool = True, endpoints: bool = False) -> dict:
        if dry_run:
            violations = graph.schema_violations()
            return {"dry_run": True, "clean": not violations,
                    "summary": str(violations), "rules": asdict(violations)["rules"]}
        return {"dry_run": False, "constraints": graph.enforce_schema(endpoints=endpoints)}

    return ToolSpec(
        name="enforce_schema",
        description=(
            "Have PostgreSQL enforce the declared schema, so EVERY write path is validated "
            "by the database itself and a violation is an error rather than a bad row. "
            "Runs as a dry run by default: that reads, reports every row the enforcement "
            "would reject, and changes nothing. Run the dry run first and fix what it "
            "lists -- with dry_run false this runs DDL, and adding a constraint over rows "
            "that violate it fails on the first bad row without telling you how many more "
            "there are."
        ),
        parameters=_object({
            "dry_run": {
                "type": "boolean",
                "description": "True (the default) reports what enforcement would reject "
                               "without running any DDL.",
            },
            "endpoints": {
                "type": "boolean",
                "description": "Also police the declared (kind, source type, target type) "
                               "triples with a trigger that fires on every edge write.",
            },
        }),
        call=enforce_schema,
    )


def tools(graph: Graph, *, read_only: bool = False, allow_ddl: bool = False,
          embed: Optional[Callable] = None, strict_schema: bool = False) -> list:
    """Every tool this server would register, as ToolSpecs -- the single
    source of what hopai offers over MCP.

    Separated from build_server() so the tools can be inspected, tested
    and called with no SDK installed and no server running: `call` is an
    ordinary function whose parameters are the tool's arguments."""
    if embed is not None:
        if not callable(embed):
            raise TypeError(f"embed must be a callable taking text and returning a vector, "
                            f"got {type(embed).__name__}")
        if not _vector_fields(graph, "nodes") and not _vector_fields(graph, "edges"):
            raise ValueError(
                "embed= was given but this graph has no vector fields, so there is nothing "
                "to search -- call graph.define_vectors(nodes=[Vector('summary', 1536)]) "
                "(or pass --vector nodes:summary:1536) before serving"
            )
    if strict_schema and graph.schema is None:
        raise ValueError(
            "strict_schema=True needs a declared schema and this graph has none -- call "
            "define_schema() before serving, or start the server without it"
        )
    if allow_ddl and read_only:
        raise ValueError(
            "allow_ddl=True and read_only=True contradict each other -- enforce_schema "
            "runs ALTER TABLE, which is not a read. Pick one"
        )

    static = {tool["name"]: tool for tool in graph.tool_schemas()}
    specs = [
        _describe_tool(graph, read_only, allow_ddl, embed),
        _traverse_tool(graph, static["traverse_graph"], embed),
        _aggregate_tool(graph, static["aggregate_graph"], embed),
        _cypher_tool(graph, read_only, strict_schema),
        _infer_schema_tool(graph),
    ]
    if embed is not None:
        specs.append(_search_tool(graph, embed))
    if not read_only:
        specs.append(_ingest_tool(graph, static["ingest_graph"]))
        specs.append(_define_schema_tool(graph))
    if allow_ddl:
        specs.append(_enforce_schema_tool(graph))
    return specs


# ---------------------------------------------------------------------
# The MCP SDK
# ---------------------------------------------------------------------

def _sdk() -> tuple:
    """(server class, tool class, major version) for whichever MCP SDK
    is installed.

    The SDK renamed its high-level server from FastMCP to MCPServer in
    2.0 and moved the module with it. Everything this file uses --
    `tools=` on the constructor, `Tool.from_function`, `run(transport=)`
    -- is spelled identically on both sides of that rename, so the whole
    difference is these imports plus where host/port are accepted
    (constructor on 1.x, run() on 2.0; see serve())."""
    try:
        from mcp.server.mcpserver import MCPServer                    # mcp >= 2.0
        from mcp.server.mcpserver.tools.base import Tool
        return MCPServer, Tool, 2
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP                        # mcp 1.x
        from mcp.server.fastmcp.tools.base import Tool
        return FastMCP, Tool, 1
    except ImportError as exc:
        raise ImportError(
            "the hopai MCP server needs the MCP SDK, which is an optional extra -- "
            "pip install 'hopai[mcp]'"
        ) from exc


def _register(spec: ToolSpec, tool_class):
    """One ToolSpec as an SDK tool, advertising hopai's OWN JSON Schema.

    Two things are going on, and both are deliberate:

    The SDK derives a tool's input schema from its handler's Python
    annotations, which for `start: dict` yields `{"type": "object"}` --
    true, and useless to a model that then has to guess `where`, `via`,
    `hops` and `direction`. hopai already ships hand-written schemas
    that spell those out, kept in step with what the parsers accept, so
    the derived one is replaced with the hand-written one. Argument
    VALIDATION still runs against the handler's signature, which is why
    a test asserts the two agree about the top-level parameters -- an
    advertised parameter no handler accepts would be a lie the model
    finds out about at call time.

    And the handler is wrapped to run in a worker thread: hopai is
    synchronous (a traversal blocks until PostgreSQL answers), and a
    blocking call in an async server stalls every other request on the
    connection, not just its own."""
    import anyio

    async def run(**arguments):
        return await anyio.to_thread.run_sync(functools.partial(spec.call, **arguments))

    # func_metadata() reads the signature to build the validation model,
    # and inspect.signature() honours __signature__ -- so the wrapper is
    # validated exactly like the function it wraps.
    #
    # eval_str is load-bearing, not tidiness: this module uses
    # `from __future__ import annotations`, so the handlers' annotations
    # are STRINGS, and a stored __signature__ is handed back verbatim
    # (the SDK's own eval_str=True cannot re-resolve it). Pydantic then
    # tries to resolve "Optional[list]" in ITS module and fails with
    # `is not fully defined`. Evaluating here, where Optional is in
    # scope, is what makes the wrapper's types real.
    run.__signature__ = inspect.signature(spec.call, eval_str=True)
    run.__name__ = spec.name
    run.__annotations__ = get_type_hints(spec.call)
    tool = tool_class.from_function(run, name=spec.name, description=spec.description)
    return tool.model_copy(update={"parameters": spec.parameters})


def build_server(graph: Graph, *, name: str = "hopai", read_only: bool = False,
                 allow_ddl: bool = False, embed: Optional[Callable] = None,
                 strict_schema: bool = False, http: Optional[dict] = None):
    """The configured MCP server object, not yet running.

    For mounting hopai's tools inside an application that owns the
    transport (an existing ASGI app, a server that also serves other
    tools); serve() is the whole thing when it does not.

    `http` carries the HTTP bind settings, which mcp 1.x takes on the
    constructor and mcp 2.0 takes on run() -- serve() fills it in, and
    nothing else needs to pass it."""
    server_class, tool_class, era = _sdk()
    registered = [_register(spec, tool_class)
                  for spec in tools(graph, read_only=read_only, allow_ddl=allow_ddl,
                                    embed=embed, strict_schema=strict_schema)]
    settings = http if (http and era == 1) else {}
    return server_class(name, instructions=SERVER_INSTRUCTIONS, tools=registered, **settings)


def serve(graph: Graph, *, transport: str = "stdio", host: str = "127.0.0.1",
          port: int = 8000, path: str = "/mcp", **options) -> None:
    """Build the server and run it until interrupted.

        serve(graph)                                    # stdio
        serve(graph, transport="http", port=8000)       # HTTP on /mcp

    Takes build_server()'s options (read_only, allow_ddl, embed,
    strict_schema, name). HTTP binds 127.0.0.1 unless told otherwise:
    there is no authentication here, and a graph an agent may write to
    is not a thing to put on 0.0.0.0 by accident."""
    if transport not in ("stdio", "http"):
        raise ValueError(f"transport must be 'stdio' or 'http', got {transport!r}")
    _, _, era = _sdk()
    http = {"host": host, "port": port, "streamable_http_path": path}
    server = build_server(graph, http=http, **options)
    if transport == "stdio":
        server.run("stdio")
    else:
        # "streamable-http" is the SDK's name for it; "http" is what an
        # operator types. 1.x took the bind settings at construction.
        server.run("streamable-http", **(http if era == 2 else {}))


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

def _vector(value: str):
    """--vector nodes:summary:1536 -> ("nodes", Vector("summary", 1536)).

    Vector fields are declared per handle rather than stored in the
    database (see Graph.define_vectors), so a server started from the
    command line has to be told about them or it cannot search."""
    from .vectors import Vector
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--vector takes TARGET:NAME:DIMENSIONS, e.g. nodes:summary:1536 -- got {value!r}")
    target, field, dimensions = parts
    if target not in ("nodes", "edges"):
        raise argparse.ArgumentTypeError(
            f"--vector target must be 'nodes' or 'edges', got {target!r}")
    if not dimensions.isdigit():
        raise argparse.ArgumentTypeError(
            f"--vector dimensions must be a positive integer, got {dimensions!r}")
    return target, Vector(field, int(dimensions))


def _callable(value: str):
    """--embed mypackage.embeddings:embed -> the function itself."""
    module_name, _, attribute = value.partition(":")
    if not module_name or not attribute:
        raise argparse.ArgumentTypeError(
            f"--embed takes MODULE:FUNCTION, e.g. myapp.embeddings:embed -- got {value!r}")
    import importlib
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise argparse.ArgumentTypeError(
            f"--embed: cannot import {module_name!r} -- {exc}") from exc
    try:
        function = getattr(module, attribute)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(
            f"--embed: {module_name!r} has no attribute {attribute!r}") from exc
    if not callable(function):
        raise argparse.ArgumentTypeError(f"--embed: {value} is not callable")
    return function


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hopai-mcp",
        description="Serve a hopai graph over MCP, on stdio or HTTP.",
        epilog="Example: hopai-mcp --dsn postgresql+psycopg2://u:p@localhost/db --read-only",
    )
    parser.add_argument("--dsn", default=os.environ.get("HOPAI_DSN"),
                        help="PostgreSQL DSN. Defaults to $HOPAI_DSN.")
    parser.add_argument("--graph", default=DEFAULT_GRAPH,
                        help=f"Which graph in those tables to serve (default {DEFAULT_GRAPH!r}).")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio",
                        help="stdio (default) for a client that spawns this process; "
                             "http for a long-running server.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument("--path", default="/mcp", help="HTTP path (default /mcp).")
    parser.add_argument("--name", default="hopai", help="Server name shown to the client.")
    parser.add_argument("--read-only", action="store_true",
                        help="Register only the reading tools.")
    parser.add_argument("--allow-ddl", action="store_true",
                        help="Also register enforce_schema, which runs DDL.")
    parser.add_argument("--strict-schema", action="store_true",
                        help="Refuse Cypher naming a label or kind the declared schema "
                             "does not have, instead of matching nothing.")
    parser.add_argument("--vector", action="append", type=_vector, default=[],
                        metavar="TARGET:NAME:DIMENSIONS",
                        help="Declare a vector field, e.g. nodes:summary:1536. Repeatable.")
    parser.add_argument("--embed", type=_callable, metavar="MODULE:FUNCTION",
                        help="A function taking text and returning a vector. Without it "
                             "there is no search by meaning -- a model cannot supply an "
                             "embedding itself.")
    parser.add_argument("--load-schema", action=argparse.BooleanOptionalAction, default=True,
                        help="Adopt the schema saved in the database, if there is one "
                             "(default). --no-load-schema skips the lookup.")
    return parser


def main(argv: Optional[list] = None) -> int:
    """`hopai-mcp`. Builds a Graph from the arguments and serves it."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("no database to serve -- pass --dsn or set HOPAI_DSN")

    graph = Graph(args.dsn, graph=args.graph)
    if args.vector:
        graph.define_vectors(
            nodes=[field for target, field in args.vector if target == "nodes"],
            edges=[field for target, field in args.vector if target == "edges"],
        )
    if args.load_schema:
        try:
            graph.load_schema()
        except ValueError as exc:
            # Absent is the normal case (nothing ever called save_schema);
            # a corrupted document raises here too, and both leave the
            # server usable without a schema -- so it says so on stderr
            # rather than either dying or going quiet.
            print(f"hopai-mcp: serving without a declared schema: {exc}", file=sys.stderr)

    serve(graph, transport=args.transport, host=args.host, port=args.port, path=args.path,
          name=args.name, read_only=args.read_only, allow_ddl=args.allow_ddl,
          embed=args.embed, strict_schema=args.strict_schema)
    return 0


if __name__ == "__main__":       # pragma: no cover - `python -m hopai.mcp`
    raise SystemExit(main())
