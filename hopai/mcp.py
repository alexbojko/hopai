"""
hopai.mcp

The graph as an MCP server: one opt-in command that hands any client
speaking the protocol -- Claude Desktop, an IDE, an agent framework --
a set of tools for reading, writing and shaping this graph.

    pip install "hopai[mcp]"

    hopai-mcp --dsn postgresql+psycopg2://user:pass@host/db
    hopai-mcp --dsn ... --transport http --port 8000
    hopai-mcp --dsn ... --graph docs --graph crm      # several, one pool

...or from Python, which is what you want as soon as the graphs need
setting up first -- or an embedding function (see SIMILARITY):

    from hopai import Graph
    from hopai.mcp import serve

    graph = Graph(dsn)
    graph.define_schema(nodes=[Person, Company], edges=[WorksAt])
    serve(graph, embed=my_embedder)          # stdio; transport="http" for HTTP
    serve({"docs": graph, "crm": graph.in_graph("crm")})

MANY GRAPHS, ONE SERVER. `graph=` takes one Graph or a {name: Graph}
mapping, because that is what the library already does: `Graph` is a
cheap handle and `in_graph()` shares the engine and its pool, so N
graphs cost N handles rather than N connection pools. Serving one graph
per process would have made an operator run N processes to get back
something hopai gives away.

With several, every tool REQUIRES a `graph` argument (an enum of the
served names) and `list_graphs` appears to say what those names are --
there is no default, because an omitted `graph` has no safe reading:
falling back to one graph answers a question about another, and for a
write it puts the rows there. With one graph, no tool mentions graphs
at all. Each graph keeps its OWN schema and vector fields --
`in_graph()` carries neither on purpose, since a different graph is
allowed a different shape.

WHICH graphs, from the command line: by default ALL of them.
`hopai-mcp --dsn ...` calls Graph.graphs(), which lists the graph_ids
that have rows, and serves every one. `--graph NAME` restricts it and is
repeatable.

That way round because the DSN is the boundary. A process holding these
credentials can already read every graph in the database, so declining
to enumerate them protects nothing. The alternative default -- the one
this replaced -- served only the graph literally named 'default', which
had a server pointed at a database whose rows live in 'docs' and 'crm'
answer "nothing here": confidently, and about graphs it had simply not
been told to look at. That is the failure this library is written
against, and it is worse than the exposure it was avoiding.

So an agent that must not see `crm` gets `--graph docs` -- a server that
does not serve it, rather than one that declines to admit it exists.

`list_graphs` still reports what this server SERVES, discovered or
named. It is never a live query, so a graph created after start-up is
not silently in scope.

What one server cannot do is give two graphs DIFFERENT permissions:
read_only, allow_mutations and allow_ddl are properties of the server,
not of a graph.
An agent that may read `docs` and write `crm` is two servers, and that
is the honest boundary -- a per-call argument was never one.

THE TOOLS, and the call each one is:

    list_graphs       the graphs this server serves (only when >1)
    describe_graph    what exists: the declared schema, the vector
                      fields, what this server will and will not do
    traverse_graph    traverse_json()      -- multi-hop, filtered
    aggregate_graph   aggregate_json()     -- count/sum/avg/min/max
    cypher            Graph.cypher()       -- read or write, one syntax
    search_similar    Graph.vector_search()          (needs `embed`)
    ingest_graph      Graph.ingest()       -- create, or merge on keys
    mutate_graph      Graph.mutate()       -- update or delete by filter
    infer_schema      Graph.infer_schema() -- observe, adopt nothing
    define_schema     Graph.define_schema() + save_schema()
    enforce_schema    Graph.schema_violations() / enforce_schema()

PERMISSIONS. Four levels, because "an agent may read this graph", "may
add to it", "may delete from it" and "may run ALTER TABLE on it" are
four different sentences, and an operator who can only say one of them
ends up saying the largest:

    read_only=True         the read tools only
    (default)              the above plus ingest_graph, define_schema,
                           and writing Cypher
    allow_mutations=True   the above plus mutate_graph, and Cypher
                           DELETE / DETACH DELETE / SET / REMOVE
    allow_ddl=True         adds enforce_schema, which runs DDL

allow_mutations is its own flag rather than part of the write level
because creating rows and destroying them are not the same power. A
delete matches by filter and is unrecoverable; that belongs to a
sentence someone said on purpose. The library agrees with the
distinction from the other side -- a filterless delete refuses, and
`all=True` is the opt-in.

The gate is which tools get REGISTERED, not a check inside a handler:
a tool the model cannot see is a tool it cannot be talked into calling.
The one exception is `cypher`, which is one tool covering all four
kinds -- it classifies the query first (cypher.classify_cypher) and
refuses by permission before opening a connection.

SIMILARITY, and the rule it must not break. A VECTOR never passes
through a tool schema -- a model asked for an embedding invents one, and
an invented embedding finds confidently wrong neighbors (hopai/vectors.py
argues this at length). TEXT is the opposite: a model says what it is
looking for as truthfully as it writes a filter, so text is advertised
and only the floats are refused. json_api.refuse_vectors() is the one
place that is enforced, and it runs on every spec BEFORE anything is
injected.

There are two ways in, and which applies depends on where the embedder
lives:

  - `near: {"field": ..., "text": ...}` on start or any hop, the same
    spelling traverse_json() takes. The FIELD embeds it, with the client
    the application declared in Vector(name, dims, embed=...), so the
    query embedding comes from the same model that wrote the stored
    ones. This is the better answer wherever it is available, and it
    needs nothing from this server.
  - `start.search`, a bare string, for graphs whose fields carry no
    embedder of their own. This server embeds it with the operator's
    callable:

        serve(graph, embed=lambda text: openai_embedding(text))

    One callable for every field, which is why it is the fallback: it
    cannot know which model wrote which field.

`search_similar` appears only when `embed` is configured. `near` needs
no such gate -- it is advertised whenever the underlying schema
advertises it, and a field with no embedder refuses by name when the
text is resolved. Sending both `search` and a `start.near` is refused
rather than resolved: one of the two would have to be silently dropped.

SIZE. `traverse_graph` and a Cypher MATCH both return the ENTIRE
matching subgraph, whatever its size -- fine for a Python caller, not
fine for a model: a broad traversal returns everything, the MCP client
silently truncates the result to fit its context window, and the model
never learns it got a partial answer. That is rule 4's "silently
different answer", produced by the CLIENT on hopai's behalf rather than
by hopai -- and letting it happen is no better than causing it.

`max_nodes` (`--max-nodes` on the CLI, default 500; `None`/`none`
disables it) is a server-set ceiling, not a per-call argument -- a model
choosing its own would defeat the point. Crossing it REFUSES the whole
call, naming the real count and the real ceiling, rather than returning
a silently truncated subgraph:

    traverse_graph: this matches 4,312 nodes, over the 500-node ceiling
    this server was started with. Nothing was returned, because a
    truncated subgraph is not a subgraph. Narrow it: add properties to
    `where`, reduce `hops`, or call aggregate_graph if a count is what
    you wanted. To rank by relevance instead, use start.search with
    `keep` (needs an embedder; describe_graph says whether this server
    has one).

NODES, not edges or bytes: nodes are the unit a model actually reads
one at a time, and Count() -- the existing counting machinery -- already
counts nodes, so a node ceiling is what that machinery can enforce
without inventing a second one. A `properties` bag heavy enough to blow
a context window at a low node count is a real failure mode this does
not catch, but it is a differently-shaped problem (bound the
properties, not the traversal) and conflating the two would make the
ceiling answer neither question precisely.

WHY THE CHECK RUNS AFTER THE READ, NOT BEFORE IT. The obvious design
counts first -- run `aggregate_graph`'s `Count()` over the same
start/hops, refuse before touching the traversal at all if it is over
the ceiling. That was measured and rejected, and not on performance: it
is WRONG. `Count()` aggregates the LAST hop's matched nodes only
(hopai/core.py's build_aggregate_query, and the CLAUDE.md invariant
that names it), while the subgraph a traversal actually reports is the
union of every node on every hop's matching edges -- deliberately
bigger, so a hop spanning several edges reports all of them and dead
ends still show the nodes that led to them. The two numbers are not the
same question. Measured on benchmarks/generate_graph.py's hub data,
`forward_bounded_4hop` returns 10,350 nodes while `Count()` over the
IDENTICAL chain reports 225 -- a 46x undercount, and unboundedly worse
with a broad early hop narrowed by a late filter (a synthetic hub-then-
rare-leaf chain measured a 26x undercount on 53 actual nodes). A
pre-count gate built on `Count()` would wave through exactly the
oversized result this feature exists to catch, which is worse than no
gate at all -- an approximation rule 4 forbids, not a shortcut it
allows.

So the traversal runs to completion (through the same traverse_json()/
Graph.traverse() a Python caller uses) and the check happens on its
actual, exact node count, before the result reaches the client. The
cost lands on the rare call that turns out to be oversized -- the one
already about to be refused -- and a call that stays under the ceiling,
the common case, pays nothing beyond what it already paid to run. That
oversized call is NOT new cost this feature introduces: the identical
query, hydrating the identical full result in Python, already ran
before max_nodes existed -- the client just threw the result away
after. This feature adds the refusal; it does not add the work.

The oversized result is fully materialized in Python before the check
discards it, so a match in the millions still costs the memory a match
in the millions costs -- max_nodes bounds what reaches the caller, not
peak memory during the call that gets refused. A server expecting
traversals that large regularly should narrow `where=` at the query
level, not rely on this ceiling to keep them cheap.

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
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional, get_type_hints

from .core import Graph
from .json_api import aggregate_json, refuse_vectors, traverse_json, vector_search_json
from .models import DEFAULT_GRAPH

#: Server-level guidance, sent to the client at connection time. What a
#: model has to know before its first call and cannot see in any single
#: tool's schema.
_ONE_GRAPH = "This server exposes one property graph stored in PostgreSQL."
_ADVICE = """\
Call describe_graph first. It returns the node types, edge kinds and
property names that actually exist here, and guessing those is the main
way these tools return an empty result instead of an answer.

Filters are EXACT property matches, never substring or fuzzy matches.
A traversal returns a SUBGRAPH -- every node and edge on a matching
chain -- not a ranked list, and ids come back as strings.

Errors from these tools name the fix. Read the message and correct the
call rather than retrying it unchanged.\
"""
SERVER_INSTRUCTIONS = f"{_ONE_GRAPH}\n\n{_ADVICE}"

#: The sentence TRAVERSE_TOOL_SCHEMA/AGGREGATE_TOOL_SCHEMA use to point a
#: model at `near` for meaning-based lookup. `near` is always available;
#: what an `embed` callable adds is the SECOND way in -- start.search,
#: for graphs whose fields carry no embedder of their own. So the
#: sentence is REPLACED (not appended to) rather than leaving the
#: description listing one way and the parameters offering two.
_NO_MEANING = (
    "`where`/`via` are EXACT property matches; to select nodes by MEANING instead, "
    "give `near` a field and the text to look for, with `keep` to say how many to keep."
)
_NO_MEANING_AGGREGATE = (
    "`where`/`via` are EXACT property matches; `near` selects by meaning."
)
_WITH_MEANING = (
    "`where`/`via` are EXACT property matches. To select nodes by MEANING instead, "
    "either give `near` a field and the text to look for (the field embeds it), or "
    "put the question's own text in start.search and this server will embed it and "
    "seed the traversal with the most similar nodes. Use one or the other, not both. "
    "Never invent a vector; there is nowhere to put one."
)

#: Default seed size for start.search. Big enough that a real answer is
#: usually inside it, small enough that the walk stays cheap.
DEFAULT_KEEP = 25

#: Default node ceiling for traverse_graph and a Cypher MATCH -- see the
#: module docstring's SIZE section for what it counts and why. Chosen as
#: "comfortably more than a model reads in one tool result, comfortably
#: less than blows a context window" rather than measured against any
#: one client; an operator with a bigger (or no) budget passes their own.
DEFAULT_MAX_NODES = 500


#: Per-caller narrowing advice, in each one's own vocabulary --
#: traverse_graph's `where`/`hops`/aggregate_graph versus Cypher's
#: WHERE/`*min..max`/aggregating RETURN. Keyed by the same caller name
#: the refusal message opens with.
_NARROW = {
    "traverse_graph": (
        "Narrow it: add properties to `where`, reduce `hops`, or call aggregate_graph "
        "if a count is what you wanted."
    ),
    "cypher": (
        "Narrow it: add properties to WHERE, reduce the `*min..max` hop range, or "
        "write an aggregating RETURN (e.g. RETURN count(...)) if a count is what you "
        "wanted."
    ),
}


def _max_nodes_message(caller: str, count: int, max_nodes: int, can_search: bool) -> str:
    """The max_nodes refusal, shared by traverse_graph and Cypher's
    MATCH path so the wording (and the reasoning behind it -- the module
    docstring's SIZE section) lives in exactly one place.

    `can_search` decides whether the `start.search` sentence is present
    at all -- the same conditional-sentence pattern _with_search() uses
    for _NO_MEANING/_WITH_MEANING. Pointing a model at a lever this
    server cannot actually pull (no embedder, or Cypher, which has no
    `start.search` spelling at all) sends it at a dead end instead of a
    fix, which is worse than not mentioning it."""
    message = (
        f"{caller}: this matches {count:,} nodes, over the {max_nodes:,}-node ceiling "
        f"this server was started with. Nothing was returned, because a truncated "
        f"subgraph is not a subgraph. {_NARROW[caller]}"
    )
    if can_search:
        message += (
            " To rank by relevance instead, use start.search with `keep` (needs an "
            "embedder; describe_graph says whether this server has one)."
        )
    return message


def _enforce_max_nodes(caller: str, count: int, max_nodes: Optional[int], can_search: bool) -> None:
    """Raise the max_nodes refusal if `count` -- the ALREADY-COMPUTED,
    exact size of a traversal's result -- is over the ceiling.

    Called on a result that has already been built (traverse_json()'s
    dict, or a Cypher read's Subgraph), never used to decide whether to
    run the traversal in the first place -- see the module docstring's
    SIZE section for why a pre-count would be wrong, not just slower."""
    if max_nodes is not None and count > max_nodes:
        raise ValueError(_max_nodes_message(caller, count, max_nodes, can_search))


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

    def __post_init__(self):
        """A tool that cannot describe itself does not get registered.

        The SDK ships whatever it is handed, so an empty description or
        a `parameters` that is not an object schema reaches the model as
        a tool it has to guess at -- which is the failure this module
        hand-writes its schemas to avoid. Checked here rather than in a
        test because it is a property of the thing, and a mutant that
        blanked a description got all the way to a registered tool."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"a tool needs a name, got {self.name!r}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(
                f"tool {self.name!r} has no description -- a model chooses tools by "
                f"reading them, so an undescribed tool is one it can only guess at")
        if not isinstance(self.parameters, dict) or self.parameters.get("type") != "object":
            raise ValueError(
                f"tool {self.name!r}: `parameters` must be a JSON Schema object "
                f"(\"type\": \"object\"), got {self.parameters!r}")
        if not callable(self.call):
            raise TypeError(f"tool {self.name!r}: `call` must be callable, "
                            f"got {type(self.call).__name__}")


# ---------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------

def _object(properties: dict, required: Optional[list] = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


class Served:
    """The graphs one server exposes, in the order they were given --
    the first is the default, and the only one a single-graph server
    ever mentions.

    One process serves many graphs because that is what the library
    already does: `Graph` is a cheap handle and `in_graph()` shares the
    engine and its pool, so N graphs cost N handles rather than N
    connection pools (README, "Many graphs, one database"). Pinning a
    server to one graph would have made the operator run N processes to
    get back what hopai gives away.

    Each graph keeps its OWN schema and vector fields, because
    `in_graph()` deliberately carries neither: a different graph is
    allowed a different shape. That is why this holds handles rather
    than one handle and a list of names."""

    def __init__(self, graphs):
        if isinstance(graphs, Served):
            graphs = graphs.graphs
        if isinstance(graphs, Graph):
            graphs = {graphs.graph: graphs}
        if not isinstance(graphs, Mapping) or not graphs:
            raise TypeError(
                "serve a Graph, or a non-empty {name: Graph} mapping of the graphs this "
                f"server exposes -- got {graphs!r}"
            )
        for name, graph in graphs.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"a graph name must be a non-empty string, got {name!r}")
            if not isinstance(graph, Graph):
                raise TypeError(f"graph {name!r} must be a Graph, got {type(graph).__name__}")
        self.graphs = dict(graphs)

    @property
    def names(self) -> list:
        return list(self.graphs)

    @property
    def first_name(self) -> str:
        """The first name given. There is no DEFAULT graph when several
        are served -- `graph` is required on every call then -- so this
        is only what a single-graph server answers with, plus whose
        vocabulary the static schemas summarize in that case."""
        return self.names[0]

    @property
    def first(self) -> Graph:
        return self.graphs[self.first_name]

    @property
    def many(self) -> bool:
        return len(self.graphs) > 1

    def pick(self, name: Optional[str], caller: str) -> Graph:
        """The named graph. With several served there is no default and
        an unnamed call is REFUSED, because a default is precisely what
        writes a model's rows into the graph it did not mean: `graph`
        is an argument a model can forget, and forgetting it would
        otherwise be silent and successful. With one served graph there
        is nothing to name and nothing to forget.

        A name this server does not serve is refused with the list, and
        never answered from another graph -- reporting one graph's rows
        as another's is the single bug multi-graph scoping produces."""
        if name is None:
            if self.many:
                raise ValueError(
                    f"{caller}: this server serves several graphs, so every call names "
                    f"one -- pass graph=<name>. It serves {self.names}; list_graphs "
                    f"describes them"
                )
            return self.first
        if name not in self.graphs:
            raise ValueError(
                f"{caller}: this server does not serve a graph named {name!r} -- "
                f"it serves {self.names}"
            )
        return self.graphs[name]

    def vector_fields(self, target: str) -> set:
        """Every vector field name declared for `target` by ANY served
        graph. The union is what a tool schema can advertise; picking
        one the CHOSEN graph does not declare is refused per call by
        _resolve_field(), naming that graph's own fields.

        Names only. This carried which graphs declared each field until
        a mutant replaced those names with None and nothing noticed --
        because nothing read them. Data no caller uses is where mutants
        live, so it is gone rather than asserted into relevance.

        Built from vectors.field_names() -- the same "what can be
        searched here" answer Graph.tool_schemas() reads for a single
        graph, so a widened enum on one side can never drift from the
        other (issue #51)."""
        from .vectors import field_names
        return {name for graph in self.graphs.values()
                for name in field_names(graph.vectors, target)}

    def seeds_from_text(self, embed: Optional[Callable]) -> bool:
        return any(_seeds_from_text(graph, embed) for graph in self.graphs.values())


def _static_schemas(served: Served) -> dict:
    """hopai's four hand-written tool schemas, keyed by name, with
    their descriptions fitted to what this server serves.

    One graph: `Graph.tool_schemas()`, which is the library's own way of
    summarizing THIS graph's declared vocabulary into each description.
    Several: the module constants instead, because tool_schemas() would
    stamp the default graph's type names onto tools that also answer
    for every other graph -- a summary that is wrong for all but one of
    them is worse than none. _described() names the graphs instead."""
    if not served.many:
        return {tool["name"]: tool for tool in served.first.tool_schemas()}

    import copy

    from .ingest import INGEST_TOOL_SCHEMA
    from .json_api import AGGREGATE_TOOL_SCHEMA, TRAVERSE_TOOL_SCHEMA
    from .mutate import MUTATE_TOOL_SCHEMA
    # Their descriptions are left exactly as written: _described() adds a
    # single graph's vocabulary and has nothing to add here, and tools()
    # appends the line that names the graphs to every spec. Running them
    # through _described() anyway was a no-op that read like a step --
    # a mutant broke the assignment and no test could tell.
    return {tool["name"]: copy.deepcopy(tool)
            for tool in (TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA, INGEST_TOOL_SCHEMA,
                         MUTATE_TOOL_SCHEMA)}


def _graph_key(served: Served) -> dict:
    """The `graph` parameter, offered only when there is a choice --
    the same rule as `search_field`. A single-graph server's tools look
    exactly as they did before this existed."""
    if not served.many:
        return {}
    return {
        "graph": {
            "type": "string",
            "enum": served.names,
            "description": "Which graph this call is about. Required: the graphs are "
                           "separate, nothing in one is visible from another, and there "
                           "is no default to fall back on. list_graphs names them.",
        },
    }


def _described(served: Served, description: str) -> str:
    """A description with the served graph's declared vocabulary
    appended -- the same treatment Graph.tool_schemas() gives the three
    static schemas, applied to the tools this module defines itself.

    Nothing is appended when several graphs are served: there is no
    single vocabulary, and the default's would read as every graph's.
    _named_graphs() says what IS true of all of them instead."""
    if served.many or served.first.schema is None:
        return description
    from .schema import tool_summary
    return f"{description} {tool_summary(served.first.schema)}"


def _instructions(served: Served) -> str:
    """The server-level guidance a client sees at connection time. The
    opening sentence is the only part a second graph changes, so it is
    a separate string rather than something to rewrite by search."""
    if not served.many:
        return SERVER_INSTRUCTIONS
    return f"{_named_graphs(served)} They share one PostgreSQL database.\n\n{_ADVICE}"


def _named_graphs(served: Served) -> str:
    """The line every tool description carries when there is more than
    one graph. Naming them beats summarizing them: this library
    supports thousands of graphs, so a per-graph schema summary in
    every tool description is not a thing that scales -- and
    describe_graph returns one graph's schema in full."""
    return (f"This server exposes {len(served.names)} separate graphs "
            f"({', '.join(repr(n) for n in served.names)}), each with its own schema and "
            f"invisible to the others. There is no default: every call names its graph. "
            f"list_graphs lists them, describe_graph says what one contains.")


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
    json_api.refuse_vectors() BEFORE anything is injected, so a model
    that invents floats is rejected by the same function that rejects
    them in traverse_json(). What it may legitimately send is `near`
    with TEXT -- that needs no help from this server, because the field
    embeds it -- so this function only has to handle `search`, the
    fallback for fields carrying no embedder of their own.

    `keep` is NOT this function's to claim: it belongs to whichever of
    the two produced the ranking. Popping it was right when `search` was
    the only way in and is a bug now, since `{"near": ..., "keep": 3}`
    is an ordinary spec traverse_json() has always accepted."""
    if not isinstance(start, dict):
        raise TypeError(f"{caller}: `start` must be an object, got {type(start).__name__}")
    start = dict(start)
    search = start.pop("search", None)
    field = start.pop("search_field", None)
    refuse_vectors({"start": start}, caller)

    if search is not None and "near" in start:
        raise ValueError(
            f"{caller}: start.search and start.near both rank the seed set, and only "
            f"one of them can -- drop start.search to let the field embed your "
            f"`near` text, or drop start.near to have this server embed the search "
            f"string"
        )
    if search is None:
        if field is not None:
            raise ValueError(
                f"{caller}: start.search_field only means something with start.search "
                f"-- it says which vector field to rank the search text against. Name "
                f"the field in `near` instead: {{\"field\": ..., \"text\": ...}}"
            )
        return start
    if embed is None:
        raise ValueError(
            f"{caller}: start.search needs an embedding function and this server has "
            f"none -- start it with serve(graph, embed=...), give `near` a field and "
            f"text so the field embeds it itself, or filter on properties with "
            f"start.where"
        )
    if not isinstance(search, str) or not search.strip():
        raise ValueError(f"{caller}: start.search must be a non-empty string, got {search!r}")
    name = _resolve_field(graph, "nodes", field, caller)
    start["near"] = {"field": name, "vector": embed(search)}
    start.setdefault("keep", DEFAULT_KEEP)
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


def _search_keys(served: Served) -> dict:
    """The `search` half of a start object's schema, offered only when
    this server can actually embed text.

    The field enum is the union across served graphs: a schema is one
    document however many graphs are behind it, and naming a field the
    CHOSEN graph does not declare is refused per call by
    _resolve_field(), which lists that graph's own fields."""
    fields = served.vector_fields("nodes")
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


def _with_search(schema: dict, served: Served, sentence: str) -> dict:
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
    start["properties"].update(_search_keys(served))
    # A seed set has to come from SOMEWHERE, and `search` is a third way
    # in beside the static schema's `where` and `near`. EXTENDED, not
    # replaced: overwriting the pair would drop `near` from the list of
    # legal starts on exactly the servers that can search by meaning --
    # advertising the fallback while un-advertising the better route.
    start["anyOf"] = [*start.get("anyOf", []), {"required": ["search"]}]
    return schema


# ---------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------

def _list_graphs_tool(served: Served) -> ToolSpec:
    def list_graphs() -> dict:
        return {
            "graphs": [
                {
                    "graph": name,
                    "schema_declared": graph.schema is not None,
                    "node_types": sorted(nt.name for nt in graph.schema.node_types)
                                  if graph.schema is not None else None,
                    "vector_fields": {
                        target: sorted(_vector_fields(graph, target))
                        for target in ("nodes", "edges")
                    },
                }
                for name, graph in served.graphs.items()
            ],
            "note": "Every other tool takes a `graph` argument naming one of these. "
                    "They are separate graphs: nothing in one is visible from another, "
                    "and there is no default.",
        }

    return ToolSpec(
        name="list_graphs",
        description=(
            "List the graphs this server exposes, with the node types and vector fields "
            "each one declares. Call this first: every other tool requires a `graph` "
            "argument naming one of them, and the graphs are separate -- a question asked "
            "of the wrong one is answered, emptily, rather than refused. describe_graph "
            "returns one graph's full schema."
        ),
        parameters=_object({}),
        call=list_graphs,
    )


def _describe_tool(served: Served, read_only: bool, allow_ddl: bool, allow_mutations: bool,
                   embed: Optional[Callable], max_nodes: Optional[int]) -> ToolSpec:
    def describe_graph(graph: Optional[str] = None, counts: bool = False) -> dict:
        graph = served.pick(graph, "describe_graph")
        schema = graph.schema
        from .schema import vector_field_json
        vectors = {target: {name: vector_field_json(name, field)
                            for name, field in _vector_fields(graph, target).items()}
                   for target in ("nodes", "edges")}
        result = {
            "graph": graph.graph,
            "graphs": served.names,
            "schema": schema.to_json() if schema is not None else None,
            "schema_mermaid": schema.to_mermaid() if schema is not None else None,
            "vector_fields": vectors,
            # Two capabilities, not one: search_similar needs an
            # embedder (tools() refuses one without a field to rank),
            # while seeding a traversal additionally needs a NODE
            # field. Reporting them as a single flag told a model with
            # edge-only vectors that it could not search at all -- and
            # a mutation that made the two interchangeable survived,
            # which is how the conflation surfaced.
            "search_by_meaning": bool(embed) and bool(
                _vector_fields(graph, "nodes") or _vector_fields(graph, "edges")),
            "seed_traversal_by_meaning": _seeds_from_text(graph, embed),
            "writes_allowed": not read_only,
            "deletes_and_updates_allowed": allow_mutations,
            "ddl_allowed": allow_ddl,
            # None means disabled -- reported as such rather than left
            # out, so a model can tell "no ceiling" from "didn't ask"
            # without a second call. See the module docstring's SIZE
            # section for what this counts and why traverse_graph/cypher
            # enforce it after the read rather than before.
            "max_nodes": max_nodes,
            # What the library refuses, plus -- when mutations are off
            # -- the fact that nothing here can delete. A model that
            # finds no delete tool and is not told why emulates one, by
            # "updating" a node to look deleted or by trying a Cypher
            # DELETE it will only be refused for.
            "refusals": [
                "No fuzzy or substring matching: property filters are exact.",
                "No grouping (RETURN b.city, count(b)) and no edge-property aggregates.",
            ] + ([
                "A delete or update with no filter refuses rather than matching the whole "
                "graph; say all=true if that is genuinely what you mean.",
                "Deleting a node that still has edges refuses and names detach=true, "
                "rather than cascading silently.",
            ] if allow_mutations else [
                "No delete and no update on this server: mutate_graph is not registered "
                "and Cypher DELETE / DETACH DELETE / SET / REMOVE are refused. Do not "
                "emulate one by marking a node deleted -- ask the operator instead.",
            ]),
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
            "result rather than an error. It also lists every graph this server exposes; "
            "each is a separate graph with its own schema, and nothing in one is visible "
            "from another."
        ),
        parameters=_object({
            **_graph_key(served),
            "counts": {
                "type": "boolean",
                "description": "Also count the nodes in this graph. Costs a scan; "
                               "leave it off unless the size is the question.",
            },
        }),
        call=describe_graph,
    )


def _traverse_tool(served: Served, schema: dict, embed: Optional[Callable],
                   max_nodes: Optional[int]) -> ToolSpec:
    if served.seeds_from_text(embed):
        schema = _with_search(schema, served, _NO_MEANING)
    schema["parameters"]["properties"].update(_graph_key(served))

    def traverse_graph(start: dict, hops: Optional[list] = None,
                       graph: Optional[str] = None) -> dict:
        chosen = served.pick(graph, "traverse_graph")
        spec = {"start": _seed(chosen, embed, start, "traverse_graph"), "hops": hops or []}
        refuse_vectors({"hops": spec["hops"]}, "traverse_graph")
        # allow_vectors: every vector in this spec was put there by
        # _seed() from an embedding of the model's TEXT, and both
        # refusals above have already run against what the model sent.
        result = traverse_json(chosen, spec, allow_vectors=True)
        # AFTER the traversal, on its exact node count -- not a
        # aggregate_graph pre-count. See the module docstring's SIZE
        # section for why a pre-count is wrong rather than merely
        # slower: this call site is the reason it stays wrong for
        # traverse_graph specifically, since traverse_json() already ran
        # exactly what a Python caller's graph.traverse() runs, and
        # `result` IS the answer -- there is nothing left to guess at.
        _enforce_max_nodes("traverse_graph", len(result["nodes"]), max_nodes,
                          can_search=_seeds_from_text(chosen, embed))
        return result

    return ToolSpec(schema["name"], schema["description"], schema["parameters"], traverse_graph)


def _aggregate_tool(served: Served, schema: dict, embed: Optional[Callable]) -> ToolSpec:
    if served.seeds_from_text(embed):
        schema = _with_search(schema, served, _NO_MEANING_AGGREGATE)
    schema["parameters"]["properties"].update(_graph_key(served))

    def aggregate_graph(start: dict, aggregates: dict, hops: Optional[list] = None,
                        graph: Optional[str] = None) -> dict:
        chosen = served.pick(graph, "aggregate_graph")
        spec = {"start": _seed(chosen, embed, start, "aggregate_graph"),
                "hops": hops or [], "aggregates": aggregates}
        refuse_vectors({"hops": spec["hops"]}, "aggregate_graph")
        return aggregate_json(chosen, spec, allow_vectors=True)

    return ToolSpec(schema["name"], schema["description"], schema["parameters"], aggregate_graph)


def _search_tool(served: Served, embed: Callable) -> ToolSpec:
    node_fields = sorted(served.vector_fields("nodes"))
    edge_fields = sorted(served.vector_fields("edges"))

    def search_similar(query: str, k: int = 10, target: str = "nodes",
                       where: Optional[dict] = None, field: Optional[str] = None,
                       graph: Optional[str] = None) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"search_similar: query must be a non-empty string, got {query!r}")
        chosen = served.pick(graph, "search_similar")
        name = _resolve_field(chosen, target, field, "search_similar")
        # allow_vectors, for the same reason traverse_graph passes it:
        # this `near` is the server's own, built from the model's TEXT a
        # line above. Without it refuse_vectors() rejects the embedding
        # this tool exists to make -- a vector_search_json() spec carries
        # its near at the TOP level, which is a scope the refusal walks.
        return vector_search_json(chosen, {
            "near": {"field": name, "vector": embed(query)},
            "target": target,
            "k": k,
            "where": where,
        }, allow_vectors=True)

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
    properties.update(_graph_key(served))
    if len(node_fields) + len(edge_fields) > 1:
        properties["field"] = {
            "type": "string",
            "enum": sorted(set(node_fields) | set(edge_fields)),
            "description": f"Which vector field to rank against. Node fields: "
                           f"{node_fields or 'none'}. Edge fields: {edge_fields or 'none'}.",
        }
    return ToolSpec(
        name="search_similar",
        description=_described(served, (
            "Find the nodes (or edges) whose stored embedding is closest in MEANING to a "
            "piece of text -- the way in when you do not know the exact property values to "
            "filter on. Returns each match with a `similarity` score, most similar first, "
            "and no vectors. Use traverse_graph's start.search to walk outward from such "
            "matches instead of just listing them."
        )),
        parameters=_object(properties, ["query"]),
        call=search_similar,
    )


def _cypher_tool(served: Served, read_only: bool, allow_mutations: bool,
                 strict_schema: bool, max_nodes: Optional[int]) -> ToolSpec:
    from .cypher import classify_cypher

    def cypher(query: str, graph: Optional[str] = None) -> dict:
        chosen = served.pick(graph, "cypher")
        # Classified before running, and against the permission that
        # matches: DELETE is not the same power as CREATE, so a server
        # allowed to write is not thereby allowed to delete.
        kind = classify_cypher(query)
        if kind == "mutate" and not allow_mutations:
            raise ValueError(
                "this server does not allow deleting or updating rows, and that query "
                "does (DELETE/DETACH DELETE/SET/REMOVE) -- ask the operator for a server "
                "started with allow_mutations=True, or rewrite it as a read"
            )
        if read_only and kind == "write":
            raise ValueError(
                "this server is read-only and that query writes -- run a MATCH instead, "
                "or ask the operator for a server started without read_only=True"
            )
        result = chosen.cypher(query, **({"strict_schema": True} if strict_schema else {}))
        # The SAME ceiling traverse_graph enforces, on the SAME kind of
        # result: a plain MATCH (kind == "read") is the only one of the
        # four kinds classify_cypher() reports that returns a Subgraph,
        # and it goes through the identical traversal engine
        # traverse_graph does -- so it is exactly as capable of returning
        # an unbounded result, and a model reaching for Cypher instead of
        # traverse_graph must not find that a way around the ceiling.
        # "aggregate" already returns a number, not a subgraph; "write"
        # and "mutate" return an IngestResult/MutationResult -- capping
        # either would cap a write, which the issue this exists for
        # never asked for and CLAUDE.md's mutation invariants would not
        # want silently reinterpreted as a node count.
        #
        # can_search=False always: Cypher has no `start.search` spelling
        # to point a model at (cypher.py's own docstring says so), so
        # the sentence would send it at a dead end.
        if kind == "read":
            _enforce_max_nodes("cypher", len(result.nodes), max_nodes, can_search=False)
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
    # Named separately from `writing` because they are separate
    # permissions -- a write-enabled server still refuses DELETE unless
    # the operator said allow_mutations, and a description that folded
    # the two together would promise one of them wrongly either way.
    mutating = (
        " SET, REMOVE, DELETE and DETACH DELETE change or remove existing rows; a DELETE "
        "whose MATCH has no filter refuses rather than emptying the graph, and deleting a "
        "node that still has edges needs DETACH DELETE."
        if allow_mutations else
        " DELETE, DETACH DELETE, SET and REMOVE are refused on this server: it is not "
        "permitted to change or remove existing rows."
    )
    return ToolSpec(
        name="cypher",
        description=_described(served, (
            f"Run a Cypher query against this graph. Supported: {reading}{writing}"
            f"{mutating} "
            f"Labels compile to the `type` property and relationship types to `kind`. "
            f"Anything outside the supported subset is REFUSED with a message naming the "
            f"rewrite -- including bare `<>` negation, whose Cypher meaning differs from "
            f"this engine's. Read the refusal and rewrite; do not retry the same query."
        )),
        parameters=_object({
            "query": {"type": "string", "description": "The Cypher query."},
            **_graph_key(served),
        }, ["query"]),
        call=cypher,
    )


def _ingest_tool(served: Served, schema: dict) -> ToolSpec:
    def ingest_graph(nodes: Optional[list] = None, edges: Optional[list] = None,
                     merge_nodes_on: Optional[list] = None,
                     merge_edges_on: Optional[list] = None,
                     graph: Optional[str] = None) -> dict:
        document = {"nodes": nodes or [], "edges": edges or []}
        result = served.pick(graph, "ingest_graph").ingest(
            document, merge_nodes_on=merge_nodes_on, merge_edges_on=merge_edges_on)
        return result.to_dict()

    schema["parameters"]["properties"].update(_graph_key(served))

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


def _mutate_tool(served: Served, schema: dict) -> ToolSpec:
    def mutate_graph(operations: list, graph: Optional[str] = None) -> dict:
        chosen = served.pick(graph, "mutate_graph")
        return chosen.mutate({"operations": operations}).to_dict()

    schema["parameters"]["properties"].update(_graph_key(served))
    return ToolSpec(schema["name"], schema["description"], schema["parameters"], mutate_graph)


def _infer_schema_tool(served: Served) -> ToolSpec:
    def infer_schema(sample_percent: Optional[float] = None,
                     graph: Optional[str] = None) -> dict:
        schema, report = served.pick(graph, "infer_schema").infer_schema(
            sample_percent=sample_percent)
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
            **_graph_key(served),
            "sample_percent": {
                "type": "number",
                "description": "Read a random sample of this percentage of rows instead of "
                               "all of them (e.g. 5). Counts become estimates and rare "
                               "properties can be missed.",
            },
        }),
        call=infer_schema,
    )


def _define_schema_tool(served: Served) -> ToolSpec:
    from .schema import schema_from_document

    def define_schema(schema: dict, save: bool = True,
                      graph: Optional[str] = None) -> dict:
        chosen = served.pick(graph, "define_schema")
        declared = chosen.define_schema(schema=schema_from_document(schema))
        if save:
            chosen.save_schema()
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
        description=_described(served, (
            "Declare what this graph is supposed to contain: node types with typed "
            "properties, and edge kinds with the (source type -> target type) pairs they "
            "connect. The declaration is the contract other tools describe and validate "
            "against, and it is saved into the database so other processes load the same "
            "one. It REPLACES any current declaration -- send the whole schema, not a "
            "patch, and call describe_graph first if you mean to extend what exists. "
            "Declaring does not change or check a single row; enforce_schema does that."
        )),
        parameters=_object({
            "schema": {
                "type": "object",
                "description": (
                    "The whole schema as one document: `nodes` keyed by type name, `edges` "
                    "a list of (kind, source, target) entries. It REPLACES the current "
                    "declaration, so send everything you want declared -- describe_graph "
                    "returns the current one in exactly this shape."
                ),
                "required": [],
                "properties": {
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
            }},
            "save": {
                "type": "boolean",
                "description": "Persist the declaration for other processes. Default true.",
            },
            **_graph_key(served),
        }, ["schema"]),
        call=define_schema,
    )


def _enforce_schema_tool(served: Served) -> ToolSpec:
    def enforce_schema(dry_run: bool = True, endpoints: bool = False,
                       graph: Optional[str] = None) -> dict:
        chosen = served.pick(graph, "enforce_schema")
        if dry_run:
            violations = chosen.schema_violations()
            return {"dry_run": True, "clean": not violations,
                    "summary": str(violations), "rules": asdict(violations)["rules"]}
        return {"dry_run": False, "constraints": chosen.enforce_schema(endpoints=endpoints)}

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
            **_graph_key(served),
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


def tools(graph, *, read_only: bool = False, allow_ddl: bool = False,
          allow_mutations: bool = False, embed: Optional[Callable] = None,
          strict_schema: bool = False, max_nodes: Optional[int] = DEFAULT_MAX_NODES) -> list:
    """Every tool this server would register, as ToolSpecs -- the single
    source of what hopai offers over MCP.

    `graph` is one Graph, or a {name: Graph} mapping to serve several
    over one connection pool. With several,
    every tool grows an optional `graph` parameter naming which to use;
    with one, no tool mentions graphs at all.

    `max_nodes` caps how many nodes traverse_graph and a Cypher MATCH
    may return before refusing instead -- see the module docstring's
    SIZE section. `None` disables it. A SERVER setting, not a per-call
    argument a model could widen for itself.

    Separated from build_server() so the tools can be inspected, tested
    and called with no SDK installed and no server running: `call` is an
    ordinary function whose parameters are the tool's arguments."""
    served = Served(graph)
    if embed is not None:
        if not callable(embed):
            raise TypeError(f"embed must be a callable taking text and returning a vector, "
                            f"got {type(embed).__name__}")
        if not served.vector_fields("nodes") and not served.vector_fields("edges"):
            raise ValueError(
                f"embed= was given but no served graph has vector fields, so there is "
                f"nothing to search -- call define_vectors(nodes=[Vector('summary', 1536)]) "
                f"on {served.names} (or pass --vector nodes:summary:1536) before serving"
            )
    if max_nodes is not None and (isinstance(max_nodes, bool)
                                  or not isinstance(max_nodes, int) or max_nodes <= 0):
        # bool is checked explicitly: isinstance(True, int) is True in
        # Python, and max_nodes=True would otherwise silently become a
        # 1-node ceiling instead of the type error it actually is --
        # the same "checked, not coerced" rule CLAUDE.md names for
        # all=/detach=/replace=, applied to the one place this flag
        # arrives as a Python value rather than JSON.
        raise TypeError(
            f"max_nodes must be a positive integer or None (to disable the ceiling), "
            f"got {max_nodes!r}"
        )
    if strict_schema:
        # Every graph, not just the default: a per-call `graph` argument
        # would otherwise reach one that cannot be strict, and the
        # refusal would name Cypher rather than the missing schema.
        without = [name for name, g in served.graphs.items() if g.schema is None]
        if without:
            raise ValueError(
                f"strict_schema=True needs a declared schema and {without} "
                f"{'has' if len(without) == 1 else 'have'} none -- call define_schema() "
                f"before serving, or start the server without it"
            )
    if allow_ddl and read_only:
        raise ValueError(
            "allow_ddl=True and read_only=True contradict each other -- enforce_schema "
            "runs ALTER TABLE, which is not a read. Pick one"
        )
    if allow_mutations and read_only:
        raise ValueError(
            "allow_mutations=True and read_only=True contradict each other -- deleting "
            "and updating rows is not a read. Pick one"
        )

    static = _static_schemas(served)
    specs = [
        *([_list_graphs_tool(served)] if served.many else []),
        _describe_tool(served, read_only, allow_ddl, allow_mutations, embed, max_nodes),
        _traverse_tool(served, static["traverse_graph"], embed, max_nodes),
        _aggregate_tool(served, static["aggregate_graph"], embed),
        _cypher_tool(served, read_only, allow_mutations, strict_schema, max_nodes),
        _infer_schema_tool(served),
    ]
    if embed is not None:
        specs.append(_search_tool(served, embed))
    if not read_only:
        specs.append(_ingest_tool(served, static["ingest_graph"]))
        specs.append(_define_schema_tool(served))
    if allow_mutations:
        specs.append(_mutate_tool(served, static["mutate_graph"]))
    if allow_ddl:
        specs.append(_enforce_schema_tool(served))
    if served.many:
        # On every tool, not a chosen few: each one takes `graph`, and a
        # model that has to hunt for which graphs exist will guess.
        line = _named_graphs(served)
        for spec in specs:
            if "graph" in spec.parameters["properties"]:
                # REQUIRED, not defaulted. Serving several graphs, an
                # omitted `graph` has no safe reading: falling back to
                # one of them answers a question about another, and for
                # a write it puts the rows there. Advertised here rather
                # than in each builder because it is a property of the
                # server, and enforced in Served.pick() as well -- the
                # handler signature is shared by both configurations and
                # cannot say it.
                spec.parameters["required"] = sorted(
                    {*spec.parameters.get("required", []), "graph"})
        specs = [replace(spec, description=f"{spec.description} {line}") for spec in specs]
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


def build_server(graph, *, name: str = "hopai", read_only: bool = False,
                 allow_ddl: bool = False, allow_mutations: bool = False,
                 embed: Optional[Callable] = None,
                 strict_schema: bool = False, max_nodes: Optional[int] = DEFAULT_MAX_NODES,
                 http: Optional[dict] = None):
    """The configured MCP server object, not yet running.

    For mounting hopai's tools inside an application that owns the
    transport (an existing ASGI app, a server that also serves other
    tools); serve() is the whole thing when it does not.

    `graph` is one Graph or a {name: Graph} mapping -- see tools().
    `max_nodes` is tools()'s own argument, forwarded unchanged.

    `http` carries the HTTP bind settings, which mcp 1.x takes on the
    constructor and mcp 2.0 takes on run() -- serve() fills it in, and
    nothing else needs to pass it."""
    server_class, tool_class, era = _sdk()
    registered = [_register(spec, tool_class)
                  for spec in tools(graph, read_only=read_only, allow_ddl=allow_ddl,
                                    allow_mutations=allow_mutations, embed=embed,
                                    strict_schema=strict_schema, max_nodes=max_nodes)]
    settings = http if (http and era == 1) else {}
    return server_class(name, instructions=_instructions(Served(graph)), tools=registered,
                        **settings)


def serve(graph, *, transport: str = "stdio", host: str = "127.0.0.1",
          port: int = 8000, path: str = "/mcp", **options) -> None:
    """Build the server and run it until interrupted.

        serve(graph)                                    # stdio
        serve(graph, transport="http", port=8000)       # HTTP on /mcp
        serve({"docs": docs, "crm": crm})               # several graphs, one pool

    Takes build_server()'s options (read_only, allow_ddl, allow_mutations,
    embed, strict_schema, max_nodes, name). HTTP binds 127.0.0.1 unless told otherwise:
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
    """--vector nodes:summary:1536      -> every served graph gets it
       --vector crm:nodes:summary:1536  -> only the graph named `crm`

    Returns (graph or None, target, Vector).

    Vector fields are declared per handle rather than stored in the
    database (see Graph.define_vectors), so a server started from the
    command line has to be told about them or it cannot search. The
    optional graph prefix exists because in_graph() carries no vector
    fields on purpose: two graphs in one server are allowed different
    shapes, and applying one declaration to all of them by default
    would be a guess -- so say which, or say it once for all."""
    from .vectors import Vector
    parts = value.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"--vector takes TARGET:NAME:DIMENSIONS (every served graph) or "
            f"GRAPH:TARGET:NAME:DIMENSIONS (one of them), e.g. nodes:summary:1536 "
            f"-- got {value!r}")
    graph = parts.pop(0) if len(parts) == 4 else None
    target, field, dimensions = parts
    if target not in ("nodes", "edges"):
        raise argparse.ArgumentTypeError(
            f"--vector target must be 'nodes' or 'edges', got {target!r}")
    if not dimensions.isdigit():
        raise argparse.ArgumentTypeError(
            f"--vector dimensions must be a positive integer, got {dimensions!r}")
    return graph, target, Vector(field, int(dimensions))


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


def _max_nodes(value: str) -> Optional[int]:
    """--max-nodes 500 (a positive count), or --max-nodes none to
    disable the ceiling entirely -- see tools()/the module docstring's
    SIZE section for what it caps and why."""
    if value.strip().lower() in ("none", "unlimited"):
        return None
    if not value.isdigit() or int(value) <= 0:
        raise argparse.ArgumentTypeError(
            f"--max-nodes must be a positive integer or 'none', got {value!r}")
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hopai-mcp",
        description="Serve a hopai graph over MCP, on stdio or HTTP.",
        epilog="Example: hopai-mcp --dsn postgresql+psycopg2://u:p@localhost/db --read-only",
    )
    parser.add_argument("--dsn", default=os.environ.get("HOPAI_DSN"),
                        help="PostgreSQL DSN. Defaults to $HOPAI_DSN.")
    parser.add_argument("--graph", action="append", default=[], metavar="NAME",
                        help="RESTRICT the server to this graph. Repeatable. Without it "
                             "every graph in the database is served, which is what the "
                             "DSN already grants. With more than one graph served, every "
                             "tool REQUIRES a `graph` argument naming one of them, and "
                             "list_graphs is added to say what the names are.")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio",
                        help="stdio (default) for a client that spawns this process; "
                             "http for a long-running server.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument("--path", default="/mcp", help="HTTP path (default /mcp).")
    parser.add_argument("--name", default="hopai", help="Server name shown to the client.")
    parser.add_argument("--read-only", action="store_true",
                        help="Register only the reading tools.")
    parser.add_argument("--allow-mutations", action="store_true",
                        help="Also register mutate_graph, and allow Cypher DELETE / "
                             "DETACH DELETE / SET / REMOVE. Separate from writing "
                             "because deleting by filter is unrecoverable.")
    parser.add_argument("--allow-ddl", action="store_true",
                        help="Also register enforce_schema, which runs DDL.")
    parser.add_argument("--strict-schema", action="store_true",
                        help="Refuse Cypher naming a label or kind the declared schema "
                             "does not have, instead of matching nothing.")
    parser.add_argument("--vector", action="append", type=_vector, default=[],
                        metavar="[GRAPH:]TARGET:NAME:DIMENSIONS",
                        help="Declare a vector field, e.g. nodes:summary:1536 for every "
                             "served graph, or crm:nodes:summary:1536 for one. Repeatable.")
    parser.add_argument("--embed", type=_callable, metavar="MODULE:FUNCTION",
                        help="A function taking text and returning a vector. Without it "
                             "there is no search by meaning -- a model cannot supply an "
                             "embedding itself.")
    parser.add_argument("--load-schema", action=argparse.BooleanOptionalAction, default=True,
                        help="Adopt the schema saved in the database, if there is one "
                             "(default). --no-load-schema skips the lookup.")
    parser.add_argument("--max-nodes", type=_max_nodes, default=DEFAULT_MAX_NODES,
                        metavar="N|none",
                        help=f"Refuse a traverse_graph call or Cypher MATCH whose result "
                             f"would exceed this many nodes, naming the real count instead "
                             f"of letting the MCP client silently truncate it (default "
                             f"{DEFAULT_MAX_NODES}). 'none' disables the ceiling.")
    return parser


def main(argv: Optional[list] = None) -> int:
    """`hopai-mcp`. Builds the graphs from the arguments and serves them."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("no database to serve -- pass --dsn or set HOPAI_DSN")

    duplicates = sorted({n for n in args.graph if args.graph.count(n) > 1})
    if duplicates:
        parser.error(f"--graph {duplicates} given more than once -- name each graph once")

    # One engine, one pool: this handle opens it and in_graph() shares
    # it, which is the whole reason a server can hold many graphs
    # without holding many pools. Its OWN scope never matters -- graphs()
    # asks about the tables rather than about the caller, and every
    # served handle comes from in_graph() below. Scoping it to the first
    # name and then reusing it for that one saved an object and cost a
    # branch that could not be wrong either way; a mutant deleted the
    # scope entirely and nothing could tell, which is what said so.
    base = Graph(args.dsn)
    if args.graph:
        names = args.graph
    else:
        # No --graph: serve what is in the database. The DSN is the
        # boundary -- anything this process can enumerate it can already
        # read -- and defaulting to the single graph named 'default'
        # instead meant a server pointed at a database whose rows live
        # in 'docs' and 'crm' answered "nothing here", confidently and
        # wrongly, about graphs it simply had not been told to look at.
        try:
            names = base.graphs()
        except Exception as exc:            # noqa: BLE001 - any driver error, same fix
            parser.error(f"cannot list the graphs in this database: {exc}. Pass --graph "
                         f"to name the ones to serve without looking them up")
        # An empty database has no graphs to find, and refusing to start
        # would make the server useless exactly when it is being set up.
        names = names or [DEFAULT_GRAPH]
        print(f"hopai-mcp: serving every graph in this database: {names}. "
              f"Pass --graph to serve fewer.", file=sys.stderr)

    unknown = {g for g, _, _ in args.vector if g is not None} - set(names)
    if unknown:
        parser.error(f"--vector names graph(s) {sorted(unknown)} that this server does "
                     f"not serve: {names}")

    graphs = {name: base.in_graph(name) for name in names}

    for name, graph in graphs.items():
        fields = [(target, field) for chosen, target, field in args.vector
                  if chosen in (None, name)]
        if fields:
            graph.define_vectors(
                nodes=[field for target, field in fields if target == "nodes"],
                edges=[field for target, field in fields if target == "edges"],
            )
        if args.load_schema:
            try:
                graph.load_schema()
            except ValueError as exc:
                # Absent is the normal case (nothing ever called save_schema);
                # a corrupted document raises here too, and both leave the
                # server usable without a schema -- so it says so on stderr
                # rather than either dying or going quiet.
                print(f"hopai-mcp: serving {name!r} without a declared schema: {exc}",
                      file=sys.stderr)

    serve(graphs, transport=args.transport, host=args.host, port=args.port, path=args.path,
          name=args.name, read_only=args.read_only, allow_ddl=args.allow_ddl,
          allow_mutations=args.allow_mutations, embed=args.embed,
          strict_schema=args.strict_schema, max_nodes=args.max_nodes)
    return 0


if __name__ == "__main__":       # pragma: no cover - `python -m hopai.mcp`
    raise SystemExit(main())
