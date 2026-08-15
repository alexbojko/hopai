# MCP server

Serve a hopai graph to any MCP client — Claude Desktop, Claude Code, Cursor, an IDE,
an agent framework — over **stdio or HTTP**, as one command.

```bash
pip install "hopai[mcp]"
hopai-mcp --dsn postgresql+psycopg2://user:pass@localhost/db --read-only
```

## Connect a client

Most clients take the same JSON. Add this to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/`, Windows:
`%APPDATA%\Claude\`), or to your client's equivalent, and restart it:

```jsonc
{
  "mcpServers": {
    "hopai": {
      "command": "hopai-mcp",
      "args": ["--dsn", "postgresql+psycopg2://user:pass@localhost/db", "--read-only"]
    }
  }
}
```

For **Claude Code**, one line does it:

```bash
claude mcp add hopai -- hopai-mcp --dsn postgresql+psycopg2://user:pass@localhost/db --read-only
```

The DSN can come from the environment instead of the command line, which keeps the
password out of the config file:

```jsonc
{
  "mcpServers": {
    "hopai": {
      "command": "hopai-mcp",
      "args": ["--read-only"],
      "env": {"HOPAI_DSN": "postgresql+psycopg2://user:pass@localhost/db"}
    }
  }
}
```

### Over HTTP

For a long-running server rather than one the client spawns:

```bash
hopai-mcp --dsn ... --transport http --port 8000        # streamable HTTP on /mcp
```

```jsonc
{"mcpServers": {"hopai": {"url": "http://127.0.0.1:8000/mcp"}}}
```

!!! warning "There is no authentication"
    Over stdio the client is the process that spawned this one, which is the trust
    boundary. Over HTTP it binds `127.0.0.1` and **anyone who can reach the port gets
    every registered tool**. Put it behind something that authenticates before giving
    it an address, and keep it `--read-only` unless writes are the point.

## The tools

| Tool | What it does | Registered when |
| --- | --- | --- |
| `describe_graph` | The declared schema, the vector fields, what this server will and won't do. **The first call a model should make.** | always |
| `traverse_graph` | Multi-hop traversal — filters, hop ranges, direction, `OPTIONAL`. Returns a subgraph. | always |
| `aggregate_graph` | `count` / `sum` / `avg` / `min` / `max` over what a traversal matches. | always |
| `cypher` | A Cypher query — read, write, or mutate, depending on permissions. | always |
| `infer_schema` | Observes what the rows actually contain. Declares nothing. | always |
| `list_graphs` | The graphs this server serves, and what each declares. | 2+ graphs |
| `search_similar` | Nearest neighbours by meaning. Takes **text**, not vectors. | `--embed` |
| `ingest_graph` | Create nodes and edges, or merge on keys to update. | writes |
| `define_schema` | Declare the graph's shape and persist it. | writes |
| `mutate_graph` | Update or delete by filter. | `--allow-mutations` |
| `enforce_schema` | Dry-run schema violations, or run the DDL. | `--allow-ddl` |

## Permissions

Permissions decide **which tools exist**, rather than being checked inside a handler —
a tool the model cannot see is one it cannot be talked into calling.

| Flag | Adds |
| --- | --- |
| `--read-only` | nothing — the reading tools only |
| *(default)* | `ingest_graph`, `define_schema`, and writing Cypher |
| `--allow-mutations` | `mutate_graph`, and Cypher `DELETE` / `DETACH DELETE` / `SET` / `REMOVE` |
| `--allow-ddl` | `enforce_schema`, which runs `ALTER TABLE` |

Deleting is separate from writing because a delete matches by filter and does not come
back. `--read-only` with either of the other two refuses at start-up rather than at the
first call.

The `cypher` tool covers all four kinds in one tool, so it cannot be gated by
registration: it classifies the query first and refuses by permission **before opening a
connection**.

## Which graphs

By default, **every graph in the database** — `hopai-mcp` lists them at start-up and
serves them all. `--graph` restricts it:

```bash
hopai-mcp --dsn ...                              # every graph in those tables
hopai-mcp --dsn ... --graph docs                 # only `docs`
hopai-mcp --dsn ... --graph docs --graph crm     # only these two
```

`--graph` is the restriction because the DSN is the boundary: a process holding those
credentials can already read every graph. An agent that must not see `crm` gets
`--graph docs` — and separate credentials if that matters.

With more than one graph served, every tool **requires** a `graph` argument and
`list_graphs` appears to say what the names are. With one, no tool mentions graphs at
all. One server cannot give two graphs different permissions — that is two servers.

## Search by meaning

Similarity takes **text**, never vectors. A model asked to fill in an embedding invents
one, and an invented embedding finds confidently wrong neighbours — so no tool here has
anywhere to put floats. You supply the embedding function; the server embeds the model's
words with it.

```bash
hopai-mcp --dsn ... --vector nodes:summary:1536 --embed myapp.embeddings:embed
```

`--embed` takes `MODULE:FUNCTION`, resolving to a callable that takes text and returns a
vector. `--vector` declares a field as `[GRAPH:]TARGET:NAME:DIMENSIONS` and is
repeatable — vector fields are per-handle, so a CLI server has no other way to know
about them.

With both set, `search_similar` is registered and the traversal tools grow a
`start.search` that seeds the walk from the most similar nodes:

```jsonc
{"start": {"search": "retrieval augmented generation", "keep": 25},
 "hops":  [{"via": {"kind": "wrote"}, "direction": "backward"}]}
```

Without `--embed` there is no search by meaning, and the tool descriptions say so rather
than leaving a model to guess property values for a semantic question.

## All flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--dsn` | `$HOPAI_DSN` | PostgreSQL DSN. Required, from either source. |
| `--graph NAME` | *every graph* | Restrict to this graph. Repeatable. |
| `--transport` | `stdio` | `stdio` or `http`. |
| `--host` | `127.0.0.1` | HTTP bind address. |
| `--port` | `8000` | HTTP port. |
| `--path` | `/mcp` | HTTP path. |
| `--name` | `hopai` | Server name shown to the client. |
| `--read-only` | off | Register only the reading tools. |
| `--allow-mutations` | off | Add `mutate_graph` and mutating Cypher. |
| `--allow-ddl` | off | Add `enforce_schema`, which runs DDL. |
| `--strict-schema` | off | Refuse Cypher naming a label the schema does not have. |
| `--vector` | none | Declare a vector field. Repeatable. |
| `--embed` | none | `MODULE:FUNCTION` returning a vector for text. |
| `--no-load-schema` | loads | Skip adopting the schema saved in the database. |

## From Python

When the graph needs setting up first — a schema, vector fields, an embedder:

```python
from hopai import Graph, Vector
from hopai.mcp import serve, build_server, tools

graph = Graph(dsn)
graph.define_vectors(nodes=[Vector("summary", 1536)])
graph.define_schema(nodes=[Person, Company], edges=[WorksAt])

serve(graph, embed=my_embedder)                          # stdio
serve(graph, transport="http", port=8000)                # HTTP
serve({"docs": graph, "crm": graph.in_graph("crm")})     # several, one pool
serve({name: graph.in_graph(name) for name in graph.graphs()})   # all of them

app = build_server(graph)      # mount it in an app that owns the transport
specs = tools(graph)           # just the tool definitions, no SDK needed
```

`serve()` takes the same options as the flags: `read_only`, `allow_mutations`,
`allow_ddl`, `strict_schema`, `embed`, `name`.

## Try it

Ask the client something the graph can answer. A model that has called
`describe_graph` knows your node types and edge kinds, so it can name them:

- *"What's in this graph?"* → `describe_graph`
- *"Which companies do Alice's friends work for, up to four hops out?"* → `traverse_graph`
- *"How many active people are there?"* → `aggregate_graph`
- *"Find papers about retrieval augmented generation"* → `search_similar` (needs `--embed`)

## Troubleshooting

**The client shows no tools.** Check the server starts on its own first —
`hopai-mcp --dsn ... --read-only` should sit there waiting on stdin rather than exiting.
A traceback usually means the DSN is wrong or the database is unreachable. Most clients
keep a log; `hopai-mcp` writes its own diagnostics to stderr, which is where they land.

**`the hopai MCP server needs the MCP SDK`.** The extra is not installed:
`pip install "hopai[mcp]"`. Both SDK eras work — 1.10+ and 2.x.

**`cannot list the graphs in this database`.** Start-up could not reach the database to
see which graphs exist. Fix the DSN, or pass `--graph NAME` to start without the lookup.

**The model gets empty results.** Filters are exact — no substring or fuzzy matching — so
a guessed type name matches nothing rather than erroring. Have it call `describe_graph`
first. If the server was started with `--graph`, check the data is in *that* graph.

**A tool the model wants isn't there.** That is the permission gate, and it is
deliberate: `mutate_graph` needs `--allow-mutations`, `enforce_schema` needs
`--allow-ddl`, `search_similar` needs `--embed` plus a declared vector field.
`describe_graph` reports what this server will and won't do.

**A write says the server is read-only.** `--read-only` was passed. Note that it cannot
be combined with `--allow-mutations` or `--allow-ddl`; that combination refuses at
start-up.

## Why it is built this way

The design decisions — why the advertised JSON Schemas are hopai's own rather than
derived from Python signatures, why permissions gate registration instead of handlers,
why similarity never accepts floats — are in
[the architecture notes](architecture.md#the-mcp-server) and in `hopai/mcp.py`'s module
docstring.
