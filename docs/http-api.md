# HTTP API

A JSON API over the graph, for a browser. `hopai-api` serves the same
operations the [MCP server](mcp.md) serves — to a page, a `fetch()`, a `curl`.

```bash
pip install "hopai[http]"
hopai-api --dsn postgresql+psycopg2://user:pass@localhost/db --cors 'http://localhost:3000'
```

!!! question "Why not just point the page at the MCP server?"
    Because MCP is JSON-RPC over a streaming transport with session negotiation, and
    a `<script>` tag is not going to implement it. This is the same graph and the same
    `hopai.json_api` functions underneath, spelled as ordinary HTTP.

## Endpoints

| Method | Path | Body | Mounted when |
| --- | --- | --- | --- |
| `GET` | `/health` | — | always |
| `GET` | `/graphs` | — | always |
| `GET` | `/schema` | `?graph=` | always |
| `POST` | `/traverse` | a traversal spec | always |
| `POST` | `/aggregate` | a spec plus `aggregates` | always |
| `POST` | `/search` | a `near` spec, `k`, `where` | always |
| `POST` | `/cypher` | `{"query": "..."}` | always (gated by classification) |
| `POST` | `/ingest` | `{"document": {...}}` | writes allowed |
| `POST` | `/mutate` | `{"document": {"operations": [...]}}` | `--allow-mutations` |

The body is exactly the [JSON interface](reference/json-interface.md) spec, plus an
optional `graph` key naming which graph to run against — that one is routing and is
removed before the spec is parsed.

```bash
curl -s localhost:8080/traverse -H 'content-type: application/json' -d '{
  "start": {"where": {"type": "person"}},
  "hops":  [{"via": {"kind": "knows"}, "hops": [1, 2]}]
}'
```

```json
{"nodes": [{"id": "1", "properties": {"name": "Alice"}}],
 "edges": [{"id": "7", "start_id": "1", "end_id": "2", "properties": {"kind": "knows"}}],
 "elapsed_ms": 13.1}
```

## Refusals come back whole

hopai refuses a lot on purpose, and every refusal is a sentence naming the fix. They
arrive as **400 with that sentence intact**, because a UI that shows the user
"Bad Request" has thrown away the useful part.

```json
{"error": "unknown \"start\" keys ['limit'] -- \"start\" accepts ['boost', 'keep', 'label', 'near', 'rerank', 'where']",
 "type": "ValueError"}
```

Anything that is *not* a deliberate refusal is a 500 with `"internal error -- see the
server log"`, and the traceback goes to stderr rather than to the browser.

## Permissions

Permissions decide **which routes exist**, exactly as they decide which tools the MCP
server registers. An unmounted route 404s: there is nothing to talk past, and no handler
that has to remember to check.

| Flag | Adds |
| --- | --- |
| `--read-only` | nothing — the reading routes only |
| *(default)* | `/ingest`, and writing Cypher |
| `--allow-mutations` | `/mutate`, and Cypher `DELETE` / `DETACH DELETE` / `SET` / `REMOVE` |
| `--allow-ddl` | schema DDL |

`/cypher` covers all four kinds in one route, so it cannot be gated by mounting: it
classifies the query and refuses by permission **before opening a connection**.

## CORS

No origin is allowed by default. Name the ones you serve your page from:

```bash
hopai-api --dsn ... --cors http://localhost:3000 --cors http://localhost:5173
hopai-api --dsn ... --cors '*'          # any origin
```

!!! warning "There is no authentication"
    `--cors '*'` on a server with no auth in front of it means any page in any tab can
    `POST /ingest`. That is fine on a laptop with the port bound to loopback, and not
    fine anywhere else. Put it behind something that authenticates before giving it an
    address.

## Search by meaning

`/search` takes the same `near` spec everything else does, and the same two ways in
[the MCP server](mcp.md#search-by-meaning) has:

```jsonc
{"near": {"field": "summary", "text": "retrieval augmented generation"}, "k": 10}
```

The **field** embeds that text when it was declared with an embedder — either in Python
(`Vector("summary", 1536, embed=client)`) or by starting the API with
`--vector nodes:summary:1536 --embed-provider openai`, which attaches one to the fields
it declares. Failing that, the server's own `--embed`/`--embed-provider` callable answers.
With neither, a `text` query refuses by name rather than returning nothing.

A `"vector"` key is refused here exactly as it is everywhere else: floats that did not
come from an embedding model find confidently wrong neighbours.

## Mounting it yourself

`build_app()` returns a plain Starlette application, so the deployment this module
cannot provide — authentication — is yours to add:

```python
from starlette.applications import Starlette
from starlette.routing import Mount
from hopai import Graph
from hopai.http_api import build_app

app = Starlette(routes=[
    Mount("/graph", app=build_app(Graph(dsn), cors=["https://app.example.com"]),
          middleware=[my_auth_middleware]),
])
```

## All flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dsn` | `$HOPAI_DSN` | PostgreSQL DSN |
| `--graph` | every graph | Restrict to this graph. Repeatable |
| `--host` | `$HOPAI_API_HOST`, then `127.0.0.1` | Bind address |
| `--port` | `$HOPAI_API_PORT`, then `8080` | Port |
| `--cors` | `$HOPAI_API_CORS`, then none | Allowed origin. Repeatable |
| `--read-only` | off | Reading routes only |
| `--allow-mutations` | off | Mount `/mutate` |
| `--allow-ddl` | off | Allow schema DDL |
| `--vector` | none | Declare a vector field. Repeatable |
| `--embed` | none | `MODULE:FUNCTION` returning a vector for text |
| `--embed-provider` | `$HOPAI_EMBED_PROVIDER` | Build the client from the environment |
| `--embed-model` | `$HOPAI_EMBED_MODEL` | Model, or on Azure the deployment name |
| `--no-load-schema` | loads | Skip the saved schema |

## With Docker

[`docker-compose.yml`](https://github.com/alexbojko/hopai/blob/main/docker-compose.yml)
runs Postgres, this API and the MCP server together:

```bash
cp .env.example .env      # put your embedding credentials in it
docker compose up -d
curl localhost:8080/health
```

Both services read the same `HOPAI_EMBED_PROVIDER` and credential variables, so one
`.env` configures the pair. `docker compose up -d db` starts just the database, which is
what the test suite wants.
