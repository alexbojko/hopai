"""
hopai.http_api

A JSON-over-HTTP front end, for a browser. `hopai-api` serves the same
operations hopai/mcp.py serves over MCP -- to a page, a curl, a fetch()
-- because an HTML UI cannot reasonably speak MCP: that is JSON-RPC with
session negotiation and a streaming transport, and a `<script>` tag is
not going to implement it.

This is a FRONT END in the same sense json_api.py, cypher.py and mcp.py
are: every handler is one call into hopai.json_api or one Graph method,
and there is no query logic here to review. What IS here: routing,
CORS, the permission gate, and turning hopai's refusals into status
codes.

    pip install "hopai[http]"
    hopai-api --dsn postgresql+psycopg2://user:pass@localhost/db --read-only

PERMISSIONS DECIDE WHICH ROUTES EXIST, exactly as they decide which
tools mcp.py registers. A route that is not permitted is not mounted, so
it 404s rather than 403s -- there is nothing to talk past, and no
handler that has to remember to check. The default is the same as
hopai-mcp's: reading and writing, but no filter-matched deletes and no
DDL, both of which are opt-in.

THERE IS NO AUTHENTICATION, and the CORS default is deliberately narrow
because of it. `--cors '*'` is one flag, and it is one flag because on a
server with no auth in front of it, allowing any origin to call a write
endpoint is a decision an operator should make out loud rather than
inherit from a default.

ERRORS ARE THE PRODUCT HERE TOO. hopai refuses a lot on purpose -- a
delete with no filter, an unknown spec key, a near carrying invented
floats -- and every one of those refusals is a sentence naming the fix.
They come back as 400 with that sentence intact, because a UI that shows
the user "Bad Request" has thrown away the only useful part.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Callable
from typing import Optional

from .core import Graph
from .json_api import aggregate_json, traverse_json, vector_search_json
from .providers import provider_names
from .vectors import Vector

#: Refusals hopai raises on purpose. These are the caller's mistake and
#: come back as 400 with the message -- which names the fix -- rather
#: than as a 500 with a stack trace the browser cannot use.
_BAD_REQUEST = (ValueError, TypeError, KeyError)


def _sdk():
    """Starlette, or a message naming the extra.

    Imported here rather than at module scope for the same reason
    mcp.py defers the MCP SDK: `import hopai` must not require a web
    stack for the callers who only ever write Python."""
    try:
        from starlette.applications import Starlette
        from starlette.middleware.cors import CORSMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route
    except ImportError as exc:      # pragma: no cover - exercised by the extra's absence
        raise ImportError(
            'the hopai HTTP API needs Starlette -- pip install "hopai[http]"'
        ) from exc
    return Starlette, CORSMiddleware, Request, JSONResponse, Route


class Served:
    """The graphs this API exposes, and what it will do to them.

    A near-copy of mcp.py's Served in intent and deliberately not an
    import of it: that one is shaped by the MCP SDK's tool registry,
    this one by routing. What they share is the rule -- an unserved name
    is refused WITH THE LIST, because "not found" leaves the caller
    guessing between a typo and a permission."""

    def __init__(self, graphs, read_only=False, allow_mutations=False, allow_ddl=False):
        if isinstance(graphs, Graph):
            graphs = {graphs.graph: graphs}
        if not graphs:
            raise ValueError("serve a Graph, or a non-empty {name: Graph} mapping")
        if read_only and (allow_mutations or allow_ddl):
            raise ValueError(
                "read_only=True contradicts allow_mutations/allow_ddl -- a read-only "
                "API has nothing to allow. Pick one")
        self.graphs = dict(graphs)
        self.read_only = read_only
        self.allow_mutations = allow_mutations
        self.allow_ddl = allow_ddl

    def pick(self, name: Optional[str]) -> Graph:
        if name is None:
            if len(self.graphs) == 1:
                return next(iter(self.graphs.values()))
            raise ValueError(
                f"this API serves several graphs, so `graph` is required: "
                f"{sorted(self.graphs)}")
        if name not in self.graphs:
            raise ValueError(
                f"no graph named {name!r} on this server -- served: {sorted(self.graphs)}")
        return self.graphs[name]

    @property
    def writes(self) -> bool:
        return not self.read_only


def build_app(graphs, embed: Optional[Callable] = None, cors: Optional[list] = None,
              read_only: bool = False, allow_mutations: bool = False,
              allow_ddl: bool = False):
    """The Starlette application. Separate from serve() so a caller can
    mount it inside their own ASGI app, behind their own authentication
    -- which is the deployment this module cannot provide and should not
    pretend to."""
    Starlette, CORSMiddleware, Request, JSONResponse, Route = _sdk()
    served = Served(graphs, read_only=read_only, allow_mutations=allow_mutations,
                    allow_ddl=allow_ddl)

    async def read(request) -> dict:
        try:
            body = await request.json()
        except Exception as exc:        # noqa: BLE001 - any malformed body, one answer
            raise ValueError(f"the request body must be JSON -- {exc}") from exc
        if not isinstance(body, dict):
            raise TypeError(f"the request body must be a JSON object, got "
                            f"{type(body).__name__}")
        return body

    def graph_of(body: dict, request) -> Graph:
        """`graph` from the body or the query string. Both, because a
        POST names it in the document it is already sending and a GET has
        nowhere else to put it."""
        return served.pick(body.get("graph") or request.query_params.get("graph"))

    def handler(run: Callable):
        """Every route's error contract in one place: hopai's own
        refusals become 400 with the sentence that names the fix,
        anything else is a 500 that is logged rather than leaked."""
        async def route(request):
            try:
                return JSONResponse(await run(request))
            except _BAD_REQUEST as exc:
                return JSONResponse({"error": str(exc), "type": type(exc).__name__},
                                    status_code=400)
            except Exception as exc:    # noqa: BLE001 - the 500 path, deliberately broad
                traceback.print_exc(file=sys.stderr)
                return JSONResponse(
                    {"error": "internal error -- see the server log",
                     "type": type(exc).__name__}, status_code=500)
        return route

    # -- reading ------------------------------------------------------

    async def health(request):
        return {"status": "ok", "graphs": sorted(served.graphs)}

    async def graphs_(request):
        out = []
        for name, graph in sorted(served.graphs.items()):
            vectors = {target: {field.name: field.dimensions
                                for field in (graph.vectors or {}).get(target, {}).values()}
                       for target in ("nodes", "edges")}
            out.append({"graph": name, "vector_fields": vectors})
        return {"graphs": out,
                "permissions": {"writes": served.writes,
                                "mutations": served.allow_mutations,
                                "ddl": served.allow_ddl}}

    async def schema(request):
        graph = graph_of({}, request)
        declared = graph.schema
        return {"graph": graph.graph,
                "schema": declared.to_json() if declared is not None else None}

    async def traverse(request):
        body = await read(request)
        graph = graph_of(body, request)
        return traverse_json(graph, _spec(body), allow_vectors=False)

    async def aggregate(request):
        body = await read(request)
        graph = graph_of(body, request)
        return aggregate_json(graph, _spec(body), allow_vectors=False)

    async def search(request):
        body = await read(request)
        graph = graph_of(body, request)
        spec = _spec(body)
        if embed is not None and "near" in spec and isinstance(spec["near"], dict) \
                and "text" in spec["near"] and "vector" not in spec["near"]:
            # The server's own embedder, for a graph whose FIELDS carry
            # none. Same shape and same order as mcp.py's start.search:
            # the model's text is refused-or-accepted as text first, and
            # only then replaced by floats this process made.
            from .json_api import refuse_vectors
            refuse_vectors(spec, "search")
            near = dict(spec["near"])
            near["vector"] = embed(near.pop("text"))
            spec = {**spec, "near": near}
            return vector_search_json(graph, spec, allow_vectors=True)
        return vector_search_json(graph, spec, allow_vectors=False)

    routes = [
        Route("/health", handler(health)),
        Route("/graphs", handler(graphs_)),
        Route("/schema", handler(schema)),
        Route("/traverse", handler(traverse), methods=["POST"]),
        Route("/aggregate", handler(aggregate), methods=["POST"]),
        Route("/search", handler(search), methods=["POST"]),
    ]

    # -- writing, gated by registration rather than by a check ---------

    async def ingest(request):
        body = await read(request)
        graph = graph_of(body, request)
        result = graph.ingest(body.get("document") or {},
                              merge_nodes_on=body.get("merge_nodes_on"),
                              merge_edges_on=body.get("merge_edges_on"))
        return {"nodes": result.nodes, "edges": result.edges}

    async def mutate(request):
        body = await read(request)
        graph = graph_of(body, request)
        result = graph.mutate(body.get("document") or {})
        return {"deleted_nodes": result.deleted_nodes, "deleted_edges": result.deleted_edges,
                "updated_nodes": result.updated_nodes, "updated_edges": result.updated_edges}

    async def cypher(request):
        body = await read(request)
        graph = graph_of(body, request)
        query = body.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("cypher: `query` must be a non-empty string")
        # One endpoint covers all four kinds, so it cannot be gated by
        # routing -- it classifies first and refuses by permission before
        # opening a connection, exactly as the MCP tool does.
        from .cypher import classify_cypher
        kind = classify_cypher(query)
        if kind in ("write", "mutate") and served.read_only:
            raise ValueError("this API is read-only and that query writes -- run a MATCH")
        if kind == "mutate" and not served.allow_mutations:
            raise ValueError("this API does not allow deleting or updating rows, and that "
                             "query does -- start it with --allow-mutations")
        result = graph.cypher(query)
        return result.to_dict() if hasattr(result, "to_dict") else result

    if served.writes:
        routes.append(Route("/ingest", handler(ingest), methods=["POST"]))
    if served.allow_mutations:
        routes.append(Route("/mutate", handler(mutate), methods=["POST"]))
    routes.append(Route("/cypher", handler(cypher), methods=["POST"]))

    app = Starlette(routes=routes)
    # Narrow by default, and `*` is one flag away -- see the module
    # docstring. allow_credentials stays off: with `*` the browser
    # refuses the combination anyway, and this server has no session to
    # send.
    app.add_middleware(CORSMiddleware, allow_origins=cors or [],
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
    return app


def _spec(body: dict) -> dict:
    """The traversal/search spec, minus the routing key.

    `graph` names which graph to run against and is not part of the
    query, so it is removed rather than passed on -- json_api refuses
    unknown keys, on purpose, and would refuse this one."""
    return {key: value for key, value in body.items() if key != "graph"}


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

def serve(graphs, host: str = "127.0.0.1", port: int = 8080, **options) -> None:
    """Build the app and run it until interrupted.

    Binds 127.0.0.1 unless told otherwise, for the same reason
    hopai-mcp does: there is no authentication here, and a graph a
    browser can write to is not a thing to put on 0.0.0.0 by accident."""
    try:
        import uvicorn
    except ImportError as exc:      # pragma: no cover - exercised by the extra's absence
        raise ImportError(
            'hopai-api needs uvicorn to run -- pip install "hopai[http]"') from exc
    uvicorn.run(build_app(graphs, **options), host=host, port=port)


def build_parser() -> argparse.ArgumentParser:
    """`hopai-api`'s arguments. Deliberately the same spellings as
    hopai-mcp's where they mean the same thing -- --dsn, --graph,
    --vector, --embed-provider, --read-only -- so one compose file can
    configure both services from one set of variables."""
    # From mcp.py rather than duplicated: these are pure argument
    # parsing with no MCP in them, and importing hopai.mcp costs nothing
    # -- the SDK is loaded inside _sdk(), not at module scope.
    from .mcp import _callable, _resolve_embed, _vector

    parser = argparse.ArgumentParser(
        prog="hopai-api",
        description="Serve a hopai graph as a JSON HTTP API, for a browser or a script.",
        epilog="Example: hopai-api --dsn postgresql+psycopg2://u:p@localhost/db --cors '*'",
    )
    parser.add_argument("--dsn", default=os.environ.get("HOPAI_DSN"),
                        help="PostgreSQL DSN. Defaults to $HOPAI_DSN.")
    parser.add_argument("--graph", action="append", default=[], metavar="NAME",
                        help="RESTRICT the API to this graph. Repeatable. Without it "
                             "every graph in the database is served.")
    parser.add_argument("--host", default=os.environ.get("HOPAI_API_HOST", "127.0.0.1"),
                        help="Bind address. Defaults to $HOPAI_API_HOST, then 127.0.0.1.")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("HOPAI_API_PORT", "8080")),
                        help="Port. Defaults to $HOPAI_API_PORT, then 8080.")
    parser.add_argument("--cors", action="append", default=None, metavar="ORIGIN",
                        help="Allow this browser origin. Repeatable. '*' allows any, "
                             "which on a server with no authentication is a decision "
                             "to make out loud. Defaults to $HOPAI_API_CORS "
                             "(comma-separated), then to allowing none.")
    parser.add_argument("--read-only", action="store_true",
                        help="Mount only the reading routes.")
    parser.add_argument("--allow-mutations", action="store_true",
                        help="Also mount /mutate, and allow Cypher DELETE / SET / REMOVE.")
    parser.add_argument("--allow-ddl", action="store_true",
                        help="Allow schema DDL.")
    parser.add_argument("--vector", action="append", default=[], type=_vector,
                        metavar="[GRAPH:]TARGET:NAME:DIMENSIONS",
                        help="Declare a vector field. Repeatable.")
    parser.add_argument("--embed", type=_callable, metavar="MODULE:FUNCTION",
                        help="A function taking text and returning a vector.")
    parser.add_argument("--embed-provider", metavar="NAME",
                        default=os.environ.get("HOPAI_EMBED_PROVIDER"),
                        help="Build the embedding client from environment variables. "
                             f"One of: {', '.join(provider_names())}. Defaults to "
                             "$HOPAI_EMBED_PROVIDER.")
    parser.add_argument("--embed-model", metavar="NAME",
                        default=os.environ.get("HOPAI_EMBED_MODEL"),
                        help="The embedding model, or on Azure the DEPLOYMENT name. "
                             "Defaults to $HOPAI_EMBED_MODEL. Never guessed.")
    parser.add_argument("--load-schema", action=argparse.BooleanOptionalAction,
                        default=True, help="Adopt the schema saved in the database.")
    return parser, _resolve_embed


def _origins(args) -> list:
    if args.cors is not None:
        return args.cors
    configured = os.environ.get("HOPAI_API_CORS", "")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def main(argv: Optional[list] = None) -> int:
    """`hopai-api`."""
    parser, resolve_embed = build_parser()
    args = parser.parse_args(argv)
    embed, embedder = resolve_embed(parser, args)
    if not args.dsn:
        parser.error("no database to serve -- pass --dsn or set HOPAI_DSN")

    base = Graph(args.dsn)
    if args.graph:
        names = args.graph
    else:
        try:
            names = base.graphs()
        except Exception as exc:        # noqa: BLE001 - any driver error, same fix
            parser.error(f"cannot list the graphs in this database: {exc}. Pass --graph "
                         f"to name the ones to serve without looking them up")
        names = names or [base.graph]

    unknown = {g for g, _, _ in args.vector if g is not None} - set(names)
    if unknown:
        parser.error(f"--vector names graph(s) {sorted(unknown)} that this API does "
                     f"not serve: {names}")

    graphs = {name: base.in_graph(name) for name in names}
    for name, graph in graphs.items():
        fields = [(target, field) for chosen, target, field in args.vector
                  if chosen in (None, name)]
        if embedder is not None:
            fields = [(target, Vector(field.name, field.dimensions, embed=embedder))
                      for target, field in fields]
        if fields:
            graph.define_vectors(
                nodes=[field for target, field in fields if target == "nodes"],
                edges=[field for target, field in fields if target == "edges"],
            )
        if args.load_schema:
            try:
                graph.load_schema()
            except ValueError as exc:
                print(f"hopai-api: serving {name!r} without a declared schema: {exc}",
                      file=sys.stderr)

    origins = _origins(args)
    if "*" in origins:
        print("hopai-api: CORS is open to any origin and there is no authentication "
              "here. Put it behind something before giving it an address.", file=sys.stderr)
    print(f"hopai-api: serving {sorted(graphs)} on http://{args.host}:{args.port}",
          file=sys.stderr)
    serve(graphs, host=args.host, port=args.port, embed=embed, cors=origins,
          read_only=args.read_only, allow_mutations=args.allow_mutations,
          allow_ddl=args.allow_ddl)
    return 0


if __name__ == "__main__":       # pragma: no cover - `python -m hopai.http_api`
    raise SystemExit(main())
