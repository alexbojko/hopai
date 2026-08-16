# Architecture

## The read path

Spanning three modules, easier as one flow than file by file:

1. **`hop.py`** — `Start(where=)` and `Hop(where=, via=, hops=, direction=, optional=)`
   are plain dataclasses. `__post_init__` normalizes `hops` into `min_hops`/`max_hops`,
   which is what `core.py` actually reads.
2. **`filters.py`** — `resolve(column, filt)` compiles the filter DSL to a SQLAlchemy
   boolean. Equality is JSONB containment (`@>`), not `->>` comparison. `parse_filter()`
   turns the JSON operator form into the same objects, and `cypher.py` produces those
   objects too, so **all three front ends compile through one `resolve()`**.
3. **`core.py:build_query`** — per hop `i`, a recursive CTE `walk_i` (anchored on `seed`
   for the first hop, `match_{i-1}` after), then `match_i` (distinct reached nodes
   passing `where`) and `hop_edges_i` (every real edge id used). All `hop_edges_*` union
   into `all_edges` → `edge_rows`, and the statement returns tagged `(kind, id)` rows in
   one round trip.
4. **`core.py:traverse`** — splits by `kind`, then two follow-up `SELECT`s hydrate
   properties. `elapsed_ms` times all three queries.

## The aggregation path

`Count`/`Sum`/`Avg`/`Min`/`Max` in `aggregates.py` mirror the filter DSL:
`parse_aggregate()` reads the JSON form, `cypher.py`'s `RETURN` translation emits the
same objects, and **all three front ends compile through one `resolve_aggregate()`** —
the same single-path rule as filters.

`core.py:build_aggregate_query` reuses the seed/walk/match chain via `_walk_matches`
(shared with `build_query`, so the two can never disagree about what a traversal
matches) and aggregates over the **final** match CTE — the last hop's distinct nodes,
or the seed set when there are no hops. It emits none of the edge CTEs and
`Graph.aggregate()` does no hydration, which is why an aggregate answers in a fraction
of its traversal twin's time (see `benchmarks/`). Numeric extraction is guarded with
`jsonb_typeof`, so a stray non-numeric value reads as NULL (ignored, like Cypher and
PG both do) instead of aborting the whole statement.

Why only the final match may be aggregated — and which Cypher spellings translate
exactly versus refuse — is the AGGREGATION section of `hopai/cypher.py`'s docstring.

## The vector path

`vectors.py` owns everything similarity-shaped; `core.py` only decides *where* a ranked
set plugs into the walk.

- A vector field is a `vec_<name> real[]` column **beside** `properties`, never inside
  it — JSONB storage would bloat the GIN index and every result. Dimensions are a
  per-graph scoped CHECK (`scope_check`), which is what lets two graphs give one shared
  column different dimensionality — the reason the column is `real[]` and not a typed
  `vector(d)`.
- Similarity is exact cosine as an `unnest`+`sum` **LATERAL**, one per field. LATERAL
  rather than a correlated scalar subquery is a measurement: a scalar subquery gets
  pulled up and re-evaluated at every site the outer query names it (filter, score,
  `ORDER BY`), which cost 2× for identical results. It also makes the value readable as
  a column, which is what the missing/wrong-length guards need. The query vector's norm
  is precomputed in Python (one `sqrt` in the SQL, not two), and the products are cast
  to float8 **before** summing — `sum(real)` accumulates in float4 and drifts over
  embedding-length arrays. All pinned by shape tests.
- A similarity is NULL — read everywhere as "missing" — when the stored vector is NULL,
  all zeros, or **not the declared length**. That last one is not defensive noise:
  `unnest(a, b)` pads the shorter array with NULLs, so a mis-sized vector would
  otherwise score a confident cosine over the shared prefix.
- The dimension CHECK's name carries a `_graph_token` (schema.py's, reused) rather than
  a slugged graph suffix. Independent truncation of base and suffix let two graphs share
  one constraint — silently disabling one graph's enforcement and letting its
  `drop_vectors()` remove the other's.
- `ranked_ids()` is the one shape behind `Start(near=)` and `Hop(near=)`: an inner
  select computing labeled per-field similarities over deduplicated candidates, an
  outer select filtering/ordering/limiting. In `_walk_matches` it simply **becomes**
  the `seed` / `match_i` CTE, so everything downstream (edge reconstruction, dead-end
  pruning, aggregation) is unchanged — and a traversal without `near` emits
  byte-identical SQL to the pre-vector code, which a test asserts.
- `search_many()` puts the queries in a VALUES list and hangs the per-query top-k off it
  as a LATERAL, so N queries are one round trip. `_similarity()` therefore takes its
  query vector and norm either as Python constants (single search) or as expressions
  from that VALUES row. Both the inner cosine and the beam use `correlate`/
  `correlate_except`: left to infer, SQLAlchemy copies the outer FROM element into the
  subquery, which cross-joins the VALUES list and makes Postgres reject a recursive
  reference outright.
- `Boost` adds property terms to the score. They are coalesced, which is what keeps
  `combined IS NOT NULL` meaning "some similarity had a direction" — but a boost is not
  consequence-free: it reorders, and with a `keep`/`k` limit reordering decides
  membership, so a boosted `similarity` is the combined score and can exceed 1.
- `Hop(via_near=)` compiles to `edge_beam()`: a LATERAL, per anchor row, yielding the
  `(edge_id, move_id)` pair the plain join produced — so depth, the local path and edge
  reconstruction are untouched. The cycle guard goes *inside* the beam, or a top-`via_keep`
  beam would spend slots on edges leading back into the path.
- Writes go through `set_vectors()` only (UPDATE … FROM VALUES … RETURNING, one
  transaction, missing ids fail the call); ingestion rows never carry vectors.
  `stale_vectors()` reports what needs re-embedding; `pgvector_exit_ddl()` emits the one-way
  migration off this engine without importing the extension.
- The JSON forms exist for the whole family, and only `"vector"` is refused without
  `allow_vectors=True` — `"text"` is the model's way in, since the field embeds it with
  the application's own client, while an invented `"vector"` finds confidently wrong
  neighbors. `_refuse_vectors()` in `json_api.py` is the single enforcement point;
  `tests/test_vectors.py::TestToolSchemasStayVectorFree` pins both halves — that no
  schema advertises `"vector"`, and that all of them advertise `"text"`.

## The embedding seam

`embeddings.py` is the only part of hopai that makes a network call, and it makes none
of its own decisions about how: you construct the client, hopai calls one method on it.

- **No provider package is ever imported**, not even to recognize one. `_provider()`
  reads `type(client).__module__`'s first segment and dispatch is duck-typed on
  attribute shape — `isinstance` would need the import and would make the coupling
  real. `tests/test_embeddings.py::TestNoProviderIsImported` asserts `sys.modules`
  stayed clean after every adapter ran.
- `Embedder` owns the two things that are easy to get wrong: per-provider batch caps
  (`_BATCH_CAPS`, chunked here so a provider refusing 200 inputs never becomes the
  caller's problem) and the **document/query asymmetry** — several providers embed
  stored text and query text differently, and getting it wrong raises nothing and
  quietly costs recall. `embed_documents`/`embed_query`/`embed_queries` are that split.
- `Vector(embed=, source=)` binds a field to a client and to the **property** holding
  its text, defaulting to the field's own name. Three paths consume it: `set_vectors()`
  when a value is a string, `Near(text=)` at query build (resolved in `validate_nears()`,
  the first point where the spec and the graph are both in hand), and `embed_stale()`
  for the backfill.
- **Transient failures are retried; terminal ones are refused immediately.** A 429 or
  5xx is the provider saying "later"; a 401 or 400 fails identically forever, so
  retrying it only spends the caller's rate limit to reach the same error more slowly.
  `_retryable()` decides on the HTTP status the exception carries, falling back to its
  class NAME — the only provider vocabulary available to a module that imports no
  provider. Backoff is exponential with **full jitter**, because a backfill fanning out
  over several fields fails at one instant and would otherwise retry in lockstep,
  rebuilding the burst that caused the 429. A `Retry-After` header wins over the
  computed window, capped, since it is the only number involved that is not a guess.
  `retries=0` disables it: the client almost certainly retries too and the two policies
  multiply. Provider calls log to the `hopai.embeddings` logger — size at DEBUG, each
  retry and each final failure at WARNING; a retry that succeeded is not an error.
  `EmbeddingError` still carries the provider's exception as `__cause__` for a caller
  who wants to classify more precisely.
- **Async has to arrive library-wide or not at all.** An async client used inside this
  sync module means `asyncio.run()` inside `set_vectors()`, which raises
  `RuntimeError: asyncio.run() cannot be called from a running event loop` — in exactly
  the async application that motivated it. The Graph API and the SQLAlchemy engine have
  to move together.
- **Embedding always happens outside the transaction**, batched per field.
  `set_vectors()` validates and resolves every row before it takes a connection: an
  HTTP call inside an open transaction holds row locks for a network round trip, and a
  provider dying halfway would leave a half-written batch for a retry to collide with.
- `vector_search_many()` resolves every query's text in one call per field
  (`_resolve_query_texts()`), because N round trips to the provider would undo the N-1
  round trips to Postgres that call exists to save.

## The write path

`Graph` exposes the writes; `ingest.py` and `constraints.py` implement them; `core.py`
stays the traversal engine. They meet at `constraints.key_sql()`, which renders both
`CREATE INDEX` and the `ON CONFLICT` target — index inference only works when the two
are spelled identically, so they must come from one function.

Every write is one transaction, batching and multi-clause Cypher included. Half
committing is the worst failure this library has: the caller is told it failed, retries,
and the retry collides with rows that landed. `tests/test_ingest.py::TestAtomicity`
guards it.

## The delete/update path

`mutate.py` is ingestion's counterpart and deliberately shares its plumbing:
`one_transaction()` and `constraint_violation()` live in `ingest.py` and are used by
both, so "one transaction per call" and "an IntegrityError names the constraint the
caller declared" are one implementation, not two that drift.

- **`where=` is `filters.resolve()`**, the same compiler a traversal's `where=` goes
  through — a fourth filter dialect for deletes would be a fourth thing to get wrong.
- **A blank filter raises** (`Mutator._guard`), in one place, so the Python API, the JSON
  document and the Cypher translator obey the same rule. `all=True` is the explicit
  opt-in; passing it *with* a filter also raises, since one of the two is being ignored.
- **`detach=True` deletes incident edges first, in the same transaction.** Between two
  transactions an edge inserted in the gap would fail the node delete with the error
  detach was meant to prevent. Without it, the composite foreign key refuses and the
  driver's error is rewritten to name `detach=True` (`_detach_hint`).
- **The `*_statement` methods build without executing**, like `build_query` — which is
  how the graph discriminator is asserted on with no database
  (`tests/test_mutate.py::TestStatementShape`).
- **Both front ends emit the same operation dicts** (`MUTATION_OPS`), run by one
  executor: `spec_to_mutations()` for JSON, `cypher_to_mutations()` for Cypher.
- **`all=True` is decided by the front end, never by the executor** — only the front end
  knows whether the caller wrote no filter or wrote one that translation discarded
  (`node_label_key=None`). `_MutateTranslator._operation` refuses in the second case;
  the executor only sees a flag that was meant.
- **`strict_schema` applies to mutations too**, via `schema.validate_mutations()` —
  the delete/update twin of `validate_operations()`, checking the filter's vocabulary
  and the properties an update writes against the type its filter names.
- **The booleans go through `_flag()`**, which rejects anything that is not `True` or
  `False`. `"false"` is truthy, and these arguments decide how many rows survive.

A mutating `MATCH` binds a *set* of rows; the ingesting one binds a single node. That
asymmetry is Cypher's own — `MATCH (a:person) SET a.x = 1` updates every person, while an
edge has to attach to exactly one row — and `_MutateTranslator`'s docstring is where it is
written down.

## Multi-graph

One pair of tables holds every graph, discriminated by `graph_id`.

- Every read and write goes through `Graph._scoped()`. **Any new query path must too** —
  forgetting it does not error, it silently returns or writes another graph's rows.
- `graph_id` **leads** `ix_edges_graph_start_id` / `..._end_id`. A trailing position
  would make them useless the moment a second graph exists, which is the entire cost
  model of the design.
- Cross-graph edges are prevented by a composite foreign key
  `(start_id, graph_id) → nodes(id, graph_id)`, not by Python.
- `Unique`/`Index` prepend `graph_id` (`_Target.scope_index`), so one index serves every
  graph with per-graph semantics. `Required`/`Check`/`PropertyType` compile to
  `graph_id <> '<g>' OR <predicate>` (`_Target.scope_check`) — an unguarded CHECK covers
  the whole table and would make one graph's rules law everywhere. Schema enforcement
  (`enforce_schema()`) rides the same `scope_check`, and its reconcile-on-re-enforce
  only ever touches constraints carrying its own graph's `ck_schema_*` name prefix.
- `merge_*` conflict targets go through the same `scope_index()`, and so does every
  delete and update — including the subquery behind `delete_edges(start=...)`, which
  would otherwise resolve an endpoint against another graph's node.
- `graph_col=None` disables all of it for callers bringing their own tables.
- `save_schema()` adds a third table, `hopai_schema(graph_id, document, saved_at)`, and
  it sits **outside** the two-table invariant on purpose: it is metadata about a graph,
  not graph data. It is never on the query path, traversal and ingestion do not know it
  exists, it is created lazily on first save (never by `create_schema()`), and its one
  row per `graph_id` is the same per-graph discipline as everything above.

## The MCP server

`hopai/mcp.py` is a front end in the same sense `json_api.py` and `cypher.py` are: each
tool is one call into those, and there is no query logic in it. Three things about it
are not obvious from the outside.

- **The advertised JSON Schemas are hopai's own.** The MCP SDK derives a tool's input
  schema from its handler's Python annotations, which turns `start: dict` into
  `{"type": "object"}` and leaves the model to guess `where`/`via`/`hops`/`direction`.
  `_register()` builds the tool and then replaces that schema with the hand-written one
  (`TRAVERSE_TOOL_SCHEMA` and friends, via `graph.tool_schemas()` so the descriptions
  carry this graph's vocabulary). Argument *validation* still runs against the handler
  signature, so `tests/test_mcp.py` asserts the two agree about every top-level
  parameter — an advertised parameter no handler accepts is a promise broken at call
  time.
- **Similarity arrives as text.** Vectors never travel through a tool schema, so
  `search_similar` and `start.search` take words, and the server embeds them with the
  operator's `embed` callable. The refusal is not re-implemented here:
  `json_api.refuse_vectors()` runs on what the model sent *before* the embedding is
  injected, which is why that function lost its underscore.
- **Permissions decide which tools are registered**, rather than being checked inside a
  handler. `read_only` drops every write tool, `allow_mutations` is what adds
  `mutate_graph`, `allow_ddl` is what adds `enforce_schema`, and `serve()` never mixes
  `read_only` with either. They belong to the server, not to a graph — two graphs needing
  different permissions are two servers. `allow_mutations` is separate from the write
  level because creating rows and destroying them are not the same power: a delete matches
  by filter and does not come back. The one tool that cannot be gated by registration is
  `cypher`, since one tool covers all four kinds — it calls `cypher.classify_cypher()` and
  refuses by permission *before* opening a connection, which is why that function is in
  `cypher.py` rather than inlined in `Graph.cypher()`. Classifying `"mutate"` apart from
  `"write"` is what keeps a write-enabled server from picking up `DETACH DELETE` with
  `CREATE`.
- **One server holds many graphs**, because `Graph` is a handle and `in_graph()` shares
  the engine's pool: `Served` keeps `{name: Graph}`, every tool then *requires* a `graph`,
  and `list_graphs` is registered to answer the chicken-and-egg that creates. `Served.pick()`
  is the enforcement — an unserved name is refused with the list, and an omitted one is
  refused rather than defaulted, because falling back to one graph answers a question
  about another (and for a write, puts the rows there). The handler signatures keep
  `graph` optional since one handler serves both configurations; `tools()` marks it
  required in the advertised schema, and a test pins the resulting rule: everything
  advertised is accepted, everything the signature demands is advertised. Each handle
  keeps its own schema and vector fields, which is why the registry holds handles rather
  than one handle and a list of names. With a single graph nothing is advertised — the
  same rule as `search_field`, and the reason the single-graph tool surface is
  byte-identical to what it was before this existed.
- **Which graphs is decided in `main()`, not in `Served`.** With no `--graph`, the CLI
  calls `Graph.graphs()` — `SELECT DISTINCT graph_id FROM nodes` — and serves every one;
  `--graph` restricts it and skips the lookup entirely, so a restricted server never
  enumerates. `Served` itself only ever receives a finished `{name: Graph}`, which keeps
  discovery out of the request path: `list_graphs` reports the served set rather than
  re-querying, so a graph created after start-up is not silently in scope. The default is
  full access because the DSN already is — the alternative served only the graph literally
  named `default`, which is how a server pointed at a database of `docs` and `crm` came to
  answer "nothing here". An empty database still starts, on `DEFAULT_GRAPH`, since refusing
  would break the server exactly when someone is setting it up.

Handlers are synchronous and are wrapped to run in a worker thread: a blocking database
call in an async server would stall every other request on the connection. The SDK's
1.x/2.x split (`FastMCP` → `MCPServer`) is contained entirely in `_sdk()`, plus the one
place the eras disagree about where HTTP bind settings go — the constructor on 1.x,
`run()` on 2.0.

## Three gotchas in the results

- **Ids come back as strings.** `build_query` casts them so node and edge ids share one
  union'd column. The hydration queries then cast the indexed BIGINT to text to match;
  removing that cast as an optimization breaks the contract the tests assert on.
- **Both lists carry the row's real id**, so writing a result straight back with
  `add_nodes`/`add_edges` asks the primary key for ids that already exist and is
  refused. Strip `id` to copy a subgraph elsewhere. Edges carry one for the same reason
  nodes always did: `set_vectors(edges=…)` takes edge ids and a traversal is where a
  caller finds edges, and it is what tells two parallel edges apart.
- **Table and column names are configurable.** Query building must go through
  `self.node_id_col` / `self.edge_start_col` / `self.graph_col` and the `nodes_tbl` /
  `edges_tbl` attributes — never a hardcoded `"start_id"`.

## Where each module explains itself

Every module opens with a docstring saying *why* it is shaped that way, written against
bugs actually hit. Read the relevant one before changing behavior.

| Read | For |
| --- | --- |
| `hopai/core.py` | Local paths, split queries, edge-derived nodes, last-hop-only `optional` |
| `hopai/filters.py` | The DSL in both forms, and why `OR`/`AND`/`NOT` are explicit classes |
| `hopai/aggregates.py` | The aggregation DSL in both forms, the three aggregation semantics, numeric edge cases |
| `hopai/hop.py` | Why `Start` and `Hop` are separate types |
| `hopai/models.py` | The DDL, the typed-columns / JSONB split, the composite FK |
| `hopai/ingest.py` | The two row spellings, edge-by-property references, merge semantics |
| `hopai/mutate.py` | What `where=` selects, why a blank filter refuses, the three update semantics and what `detach` does |
| `hopai/constraints.py` | What each constraint compiles to, and the SQL semantics that surprise people |
| `hopai/vectors.py` | Why no pgvector, the storage and cost model, cosine-only, multivector semantics, and why a vector never passes through a tool schema |
| `hopai/embeddings.py` | Which clients are accepted and how they are recognized without an import, the document/query asymmetry, and what this seam deliberately does not do |
| `hopai/schema.py` | The graph-schema notations, the annotation mapping, what enforcement compiles to and the endpoint-type limit, and why inference is an observation rather than a `.schema` fallback |
| `hopai/cypher.py` | The translatable subset — read, write and aggregate — and why each refusal is a refusal |
| `hopai/mcp.py` | The tool inventory, the four permission levels, why similarity takes text, and the 1.x/2.x SDK adapter |
| `tests/conftest.py` | The fixture graph's shape |
| `benchmarks/README.md` | Measured numbers, including where raw CTEs beat this library 2-5x |
