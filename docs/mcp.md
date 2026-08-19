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
anywhere to put floats. Text is the opposite: a model says what it is looking for as
truthfully as it writes a filter, so text is advertised and only the floats are refused.

There are two ways in, and which one applies depends on where the embedder lives.

**1. `near`, embedded by the field.** The same spelling `traverse_json()` takes, available
whether or not this server has an embedder of its own:

```jsonc
{"start": {"near": {"field": "summary", "text": "retrieval augmented generation"},
           "keep": 25},
 "hops":  [{"via": {"kind": "wrote"}, "direction": "backward"}]}
```

The **field** embeds that text, using the client declared in
`Vector("summary", 1536, embed=…)`, so the query embedding comes from the same model that
wrote the stored ones. This is the better answer wherever it is available. `keep`,
`via_near`, `via_keep` and `boost` come with it — none of them is a place to put floats.

**2. `start.search`, embedded by the server.** The fallback for graphs whose fields carry
no embedder of their own:

```bash
hopai-mcp --dsn ... --vector nodes:summary:1536 --embed myapp.embeddings:embed
```

`--embed` takes `MODULE:FUNCTION`, resolving to a callable that takes text and returns a
vector. `--vector` declares a field as `[GRAPH:]TARGET:NAME:DIMENSIONS` and is
repeatable — vector fields are per-handle, so a CLI server has no other way to know
about them.

### Or name a provider and let the environment supply the rest

`--embed` needs a Python function to point at, which a container does not have. Name a
provider instead and hopai builds the client from environment variables:

```bash
export OPENAI_API_KEY=sk-...
hopai-mcp --dsn ... --vector nodes:summary:1536 \
          --embed-provider openai --embed-model text-embedding-3-small
```

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT=my-embedding-deployment
hopai-mcp --dsn ... --vector nodes:summary:3072 --embed-provider azure-openai
```

| Provider | Reads |
| --- | --- |
| `openai` | `$OPENAI_API_KEY`, `$OPENAI_EMBEDDING_MODEL`. Optional `$OPENAI_BASE_URL`, `$OPENAI_ORG_ID` |
| `azure-openai` | `$AZURE_OPENAI_API_KEY`, `$AZURE_OPENAI_ENDPOINT`, `$AZURE_OPENAI_EMBEDDING_DEPLOYMENT`. Optional `$AZURE_OPENAI_API_VERSION` |
| `cohere` | `$COHERE_API_KEY`, `$COHERE_EMBEDDING_MODEL` |
| `voyage` | `$VOYAGE_API_KEY`, `$VOYAGE_EMBEDDING_MODEL` |
| `google` | `$GOOGLE_API_KEY` (or `$GEMINI_API_KEY`), `$GOOGLE_EMBEDDING_MODEL` |
| `sentence-transformers` | `$SENTENCE_TRANSFORMERS_MODEL`. Runs locally, no credentials |

`hopai-mcp --embed-provider-help` prints that table. The provider and model can also come
from `$HOPAI_EMBED_PROVIDER` and `$HOPAI_EMBED_MODEL`, so a container is configured
entirely by environment with no command to change.

!!! warning "On Azure, the model is the DEPLOYMENT name"
    `$AZURE_OPENAI_EMBEDDING_DEPLOYMENT` is the deployment you created in the portal,
    which is often **not** the same string as the model it serves. Passing the model
    name where Azure wants the deployment is the usual way this configuration fails.

**Everything that can be wrong is wrong at start-up**, out loud, naming the fix — an
unknown provider, a package that is not installed, a variable that is not set, a model
that was never chosen. A server that starts without a working embedder and only finds
out on the first search has turned a configuration mistake into somebody's failed query,
somewhere else, later.

```
hopai-mcp: error: --embed-provider azure-openai needs $AZURE_OPENAI_ENDPOINT and it is
unset or empty. Export it before starting. It is the resource URL, e.g.
https://my-resource.openai.azure.com
```

The model is never defaulted, for the reason the whole vector surface exists to protect:
the model answering a query has to be the model that wrote the stored vectors, and
picking one on your behalf returns confidently wrong neighbours with nothing to see.

`--embed-provider` also **attaches its embedder to the fields `--vector` declares**, so
`near: {"field": ..., "text": ...}` works from a container too — not only `start.search`.

With both set, `search_similar` is registered and the traversal tools grow a
`start.search` that seeds the walk from the most similar nodes:

```jsonc
{"start": {"search": "retrieval augmented generation", "keep": 25},
 "hops":  [{"via": {"kind": "wrote"}, "direction": "backward"}]}
```

One callable serves every field here, which is why it is the fallback: it cannot know
which model wrote which field. Send `search` **or** `start.near`, never both — they both
rank the seed set, so accepting both would mean silently dropping one.

Without `--embed`, `search_similar` is not registered and `start.search` is not
advertised; `near` still works for any field that can embed its own text.

## Reranking

Retrieval that works is three stages, not two: retrieve wide and cheap, **rerank**
a bounded top-N by *reading* each candidate against the query, then keep what
survives. Stage two is a call to a reranking model — accurate, expensive, and
billed per document — so the client is the **operator's**, exactly as `embed` is.

From a command line, in the same three spellings `--embed` has:

```bash
hopai-mcp --dsn ... --vector nodes:summary:384 \
  --rerank-provider sentence-transformers \
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank-document-from '.properties.title + ": " + (.properties.summary // "")' \
  --rerank-field properties.title --rerank-field properties.summary
```

That one runs **locally and needs no credentials** — any cross-encoder on the
Hugging Face hub works, since the name is passed straight to `CrossEncoder()`.
For a hosted reranker, name the provider and export its key:

```bash
export COHERE_API_KEY=...
hopai-mcp --dsn ... --rerank-provider cohere --rerank-model rerank-v3.5 \
  --rerank-document-from '.properties.title + ": " + (.properties.summary // "")' \
  --rerank-field properties.title --rerank-field properties.summary
```

**Give the filter something to read.** `document_from` looks like a formatting
detail and is not: a cross-encoder scores the string it is handed, so a document
too thin to judge produces a *flat* ranking rather than an obviously broken one.
Measured on `ms-marco-MiniLM-L-6-v2`, query *"how do distributed nodes agree?"*
over three nodes:

| `document_from` | Paxos | Raft | Banana bread |
| --- | ---: | ---: | ---: |
| `.properties.title` | −11.03 | −11.21 | −11.27 |
| `.properties.title + ": " + (.properties.summary // "")` | **−2.81** | **−8.85** | −11.35 |

Titles alone put banana bread within 0.24 of Raft — noise wearing a ranking's
clothes. The same nodes with their summaries separate by 8.5. MS MARCO models
are trained on passage-shaped text; hand them a passage. `//  ""` keeps a node
whose summary is missing scoreable instead of dropping it.

`--rerank-provider` accepts **fewer names than `--embed-provider`** —
`cohere`, `voyage`, `sentence-transformers`. A reranker reads a query and a
document *together*, which is a different model from the one that embeds, and
only some vendors sell both; naming one that only embeds refuses saying exactly
that rather than reporting an unknown name. The model variables are separate for
the same reason (`$COHERE_RERANK_MODEL`, not `$COHERE_EMBEDDING_MODEL`), and
`hopai-mcp --rerank-provider-help` prints the table.

Or in Python, which stays the better answer when the client itself needs
configuring — a timeout, a proxy, credential rotation:

```python
import cohere
from hopai import Rerank
from hopai.mcp import serve

serve(graph,
      rerank=Rerank(cohere.ClientV2(), model="rerank-v3.5",
                    document_from='.properties.title + ": " + (.properties.summary // "")',
                    candidates=50),
      rerank_fields=["properties.title", "properties.summary"],
      max_candidates=100)
```

**`--rerank-document-from` and `--rerank-field` are required** with a reranker
and have no defaults — the CLI saying out loud what the Python path already
refuses to guess. Everything below applies to both spellings.

Every one of these has an environment spelling, so a container handed its
configuration by an orchestrator needs no command line at all:

```bash
HOPAI_RERANK_PROVIDER=sentence-transformers
HOPAI_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
HOPAI_RERANK_DOCUMENT_FROM='.properties.title + ": " + (.properties.summary // "")'
HOPAI_RERANK_FIELDS=properties.title,properties.summary
```

`$HOPAI_RERANK_FIELDS` is comma-separated because a variable cannot repeat the
way `--rerank-field` can; passing the flag **replaces** the variable rather than
adding to it, like every other flag/variable pair here.

What a model supplies is never the client. It is `document_from` — a jq filter
projecting one candidate into the one string the reranker reads — and
`candidates`, how many to spend. Both are retrieval decisions a model is well
placed to make, unlike an embedding it would have to invent.

**`rerank_fields` is required with `rerank`**, and it publishes what that filter
may read. The filter's *output is the document*, and the document is POSTed to a
vendor, so `.properties.ssn` parses perfectly well in the safe jq subset and
would ship straight out; a path outside the list refuses, naming it. The list is
not just a gate — it is written into the advertised description of
`document_from`, so a model picks from published paths rather than guessing. It
binds your own template too — the list is checked against the `document_from` you
passed when the server is built, so a `rerank_fields` that forgets a path your
own filter reads refuses at start-up rather than at the first tool call.
`rerank_fields=` **without** `rerank=` refuses rather than looking like a
narrowing that is not there.

**`max_candidates` bounds the bill** (default 100; `None` disables it). A
`candidates` above the ceiling **refuses naming it** rather than being quietly
lowered — the same judgement a `candidates` gets when it leaves `keep`/`k`
nothing to choose from. Silently serving 100 where 500 was asked for hides the
disagreement in the one place it costs money.

The parameter lives on `traverse_graph` and `aggregate_graph`, on `start` and on
every hop, beside that step's `near`. With **no reranker configured, no tool
advertises it at all** — the whole `rerank` object is stripped from both schemas,
which is the rule permissions already follow here: a parameter a model cannot see
is one it cannot be talked into sending. A spec that asks for reranking on such a
server refuses, since a reranker client cannot travel in JSON. `describe_graph`
reports `rerank_available`, `rerank_fields` and `max_candidates`, so a model can
tell *this server cannot rerank* from *I asked for it wrong*.

### `search_similar` deliberately has no `rerank`

That is a property of reranking rather than of this server. `search_similar`
embeds the model's text into a raw vector itself, and a reranker scores a query
against a document by **reading both** — a raw-vector `near` with a `rerank=`
refuses in hopai's core, so the parameter would be one whose every use failed.
Rerank through `traverse_graph` instead, whose `start.near` carries `{field,
text}` and whose text the **field** embeds, so the dense stage and the rerank
stage see the same query rather than two spellings of it:

```jsonc
{"start": {"near": {"field": "summary", "text": "how do nodes agree?"},
           "keep": 10,
           "rerank": {"document_from": ".properties.title", "candidates": 50}}}
```

Be aware of what that costs you, because nothing warns about it: a traversal
returns a **subgraph, not a ranking**. Similarity scores never survived into one
and rerank scores get no exception, so what a rerank changed there is *which*
nodes come back — never a score you can read. `rerank_score` exists on
`vector_search()` hits in Python, and no tool returns those.
**Reranked results with their scores are not reachable over MCP today.**

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
| `--max-nodes N\|none` | `500` | Refuse a `traverse_graph` call or Cypher `MATCH` whose result would exceed this many nodes — naming the real count, rather than letting the client silently truncate the subgraph. `none` disables the ceiling. |
| `--rerank` | none | `MODULE:FUNCTION` scoring `(query, documents)`. |
| `--rerank-provider` | none | Build the reranking client from the environment: `cohere`, `voyage`, `sentence-transformers`. |
| `--rerank-model` | none | The reranking model — on `sentence-transformers`, the CrossEncoder to load. |
| `--rerank-document-from` | none | jq filter projecting one candidate into the string the reranker reads. **Required** with a reranker. |
| `--rerank-field` | `$HOPAI_RERANK_FIELDS` | A property path that filter may read. Repeatable. **Required** with a reranker — there is no "everything" spelling. The variable is comma-separated, since an environment variable cannot repeat; a flag replaces it rather than adding to it. |
| `--max-candidates N\|none` | `100` | Ceiling on candidates one call may send to the reranker. Rerankers bill per document. |

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

`serve()` takes every flag that is not about *building* the graph — `read_only`,
`allow_mutations`, `allow_ddl`, `strict_schema`, `embed`, `max_nodes`, `name`,
`rerank`, `rerank_fields` and `max_candidates`. `build_server()` takes the same
set. All of them have a flag; reranking's are [below](#reranking).

Python is still the better answer when the client needs configuring — a timeout,
a proxy, a base URL, credential rotation — for exactly the reason `Vector(...,
embed=client)` is: a flag can name a provider, but only your code can hand hopai
a client it already built.

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

**The model cannot find `rerank`.** No reranker is configured, so the parameter
is not advertised anywhere — that is the same gate the tools themselves are
behind. Configure one with `--rerank-provider` (or `serve(rerank=…,
rerank_fields=[…])`) and it appears. `describe_graph` answers
`rerank_available: false` for exactly this question.

**A write says the server is read-only.** `--read-only` was passed. Note that it cannot
be combined with `--allow-mutations` or `--allow-ddl`; that combination refuses at
start-up.

## Why it is built this way

The design decisions — why the advertised JSON Schemas are hopai's own rather than
derived from Python signatures, why permissions gate registration instead of handlers,
why similarity never accepts floats — are in
[the architecture notes](architecture.md#the-mcp-server) and in `hopai/mcp.py`'s module
docstring.
