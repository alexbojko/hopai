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
  `scale="normalized"` (the default) rescales the coalesced value into similarity's
  own range with a min-max window function over the candidate set, at the same
  select level `_boost_columns()` builds the column at — a zero-spread candidate
  set coalesces the result to 0 rather than propagating the window function's NULL,
  which is what keeps the invariant true even when a boost carries no signal.
  `scale="raw"` is the unscaled, unbounded escape hatch: today's only behavior
  before this normalization existed.
- `vector_search()`/`vector_search_many()` report each Near's own similarity and each
  Boost's own value alongside the combined score, keyed by name (`similarities`,
  `boosts`) rather than by the `sim_i`/`boost_j` SQL alias. `_report_columns()` reads
  each LATERAL's `s` column a SECOND time in the outward SELECT rather than the `sim_i`
  `_combined()`/`_thresholds()` use — `sim_i` is coalesced to 0 for `missing="zero"`,
  and the report has to keep saying "missing" (`None`) even where the combined score
  scores it 0. Nothing is computed twice: both are ordinary column projections off the
  one LATERAL join. `_format_hit()` is the single place both `search()` and
  `search_many()` turn the flat row into that shape, so `json_api.py`'s
  `vector_search_json()` inherits it with no code of its own.
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
- **Async arrived library-wide, so it arrived here too.** `aembed_query()`/
  `aembed_queries()`/`aembed_documents()` are the awaited half of the three methods
  above, and `AsyncGraph` resolves every text through them *before* it enters the
  greenlet bridge (issue #74 — a blocking provider call inside `run_sync()` runs on the
  event loop's own thread). What has not changed is the rule that made the old refusal
  right: nothing here ever calls `asyncio.run()`, because that raises
  `RuntimeError: asyncio.run() cannot be called from a running event loop` in exactly
  the async application that motivated it. An async client reaching a *sync* call is
  refused by name instead. `_abind()` recognizes one the same way `_bind()` recognizes
  any client — module name plus attribute shape, no provider import: `AsyncOpenAI`,
  `AsyncClientV2`, `google.genai`'s `client.aio`, `aembed_documents`/`aembed_query`,
  `aget_text_embedding_batch`/`aget_query_embedding`, or a plain `async def`. A sync
  client on an async path runs in `asyncio.to_thread()` — honest *here* because a
  provider SDK blocked on a socket has released the GIL.
- **Embedding always happens outside the transaction**, batched per field.
  `set_vectors()` validates and resolves every row before it takes a connection: an
  HTTP call inside an open transaction holds row locks for a network round trip, and a
  provider dying halfway would leave a half-written batch for a retry to collide with.
- `vector_search_many()` resolves every query's text in one call per field
  (`_resolve_query_texts()`), because N round trips to the provider would undo the N-1
  round trips to Postgres that call exists to save.

## The reranking pipeline

`rerankers.py` is the second network seam, built like the first: you construct the
client, hopai calls one method on it — `score(query, documents) -> list[float]`, one
float per document in the order given, deliberately **not** factored by modality (a
reranker cannot tell a cosine's candidate from a lexical one, and a hop's candidate is
neither). The retry policy is not restated: `_retryable`/`_retry_after` are imported
from `embeddings.py`, so hopai has one classifier for provider failures.

**Flat search** (`vectors.py:search()`) is four steps in a fixed order:

1. `search_candidates()` fetches `rerank.candidates` rows instead of `k` — that is the
   only change to the statement, `_prepare_search_query(candidates=)`, and `k` still
   validates everything else so a rerank cannot change which refusals a search produces.
2. The connection **closes**. Everything after this point runs with nothing open, for
   the reason `set_vectors()` resolves embeddings before it takes one: a provider round
   trip inside an open transaction holds a snapshot for the length of an HTTP call.
3. `build_documents()` evaluates `document_from` once per candidate against the hit dict
   the caller would have received, and `score()` makes one provider call (chunked at the
   provider's own per-call cap).
4. `_reranked()` attaches `rerank_score`, sorts by it (id as tiebreak, so two equal
   scores never swap between runs) and truncates to `k`. `similarity` is left exactly as
   retrieval set it — dropping a stage's input score is never the more useful answer.

Cohere's and Voyage's answers come back **sorted by relevance**, so `_indexed_scores()`
places each score by its own `.index` rather than by arrival order; a result set that
does not cover every document refuses instead of being completed with guesses. Zipping a
relevance-sorted answer against the documents that were sent is a plausible, confidently
wrong ranking that nothing reports, which is the failure this module is written against.

**Step-wise reranking** is the part that could not be a wrapper. A traversal compiles to
*one* recursive CTE, and a reranker is a network call in the middle of the walk, so a
reranked traversal cannot be one statement. The shape is **probe → rerank → re-run the
ordinary traversal with each step's survivors pinned** (`core.py:_rerank_pins`), never
stitching partial results together:

- `_rerank_probe()` runs the chain **truncated at that step**, with the step's own `keep`
  widened to `rerank.candidates` and every earlier step's survivors already pinned. It
  reuses `_walk_matches` — the same seed/walk/match chain `build_query` and
  `build_aggregate_query` share — so a candidate set is exactly what that step would have
  reached, cut off at the number the caller agreed to pay for.
- Path context is fetched only when the filter asks for it: `jqsafe.paths_read()` decides
  whether `document_from` reads `.paths` (over-reporting when it cannot parse the filter,
  never under-reporting). When it does, the probe reads `full_walk.local_path` — the same
  (walk, match) pair `hop_edges_i` is built from — giving one row per (candidate, route);
  routes are canonically ordered so the same graph always builds the same document, and
  `max_paths` caps how many one document may quote.
- Each step's session **closes before** the provider call, and `_rerank_survivors()`
  reduces the scores to an id list: the max over a node's documents, so `per_path=True`
  keeps a node when one route is strong.
- `_pinned()` turns that list into `id IN (...)` on the step's own CTE — and expands to
  *nothing* when a step has no pins, which is what keeps a rerank-less traversal's SQL
  byte-identical.

That last point is the whole safety argument. Because the final execution is an ordinary
traversal, **fan-in, multi-hop edge reconstruction and dead-end pruning are preserved for
free**: the edges reported for a hop are still derived from `full_walk` against `match_i`,
and `match_i` still feeds the next hop. `Hop(near=, keep=N)` already narrows `match_i` to
a subset through `ranked_ids()`; a pin is the same narrowing with a different source for
the list. A node survives or is dropped as a unit, so every in-edge from a surviving
parent is still reported — `test_fan_in_both_parents_preserved` sees no difference.
Stitching would have had to re-derive those edges by hand, which is exactly what "reported
nodes derive from the edges found" exists to prevent. `aggregate()` inherits pinning for
the same reason it inherits everything else: it shares `_walk_matches`, and the last
step's matched nodes after a rerank *are* the survivors.

**The cost, measured rather than assumed** (rule 6;
`tests/test_vectors.py::test_a_reranked_traversal_costs_one_plus_two_per_reranked_step`
pins it): a baseline traversal is **3 statements** — the walk plus its two hydrations —
and each reranked step adds **2** (a probe and one hydration for the properties its
documents mention), plus one provider call. Those calls are **serial by nature**: hop
N+1's candidates are whatever hop N's rerank left behind, so unlike
`vector_search_many()`'s they cannot be issued together — which is also why pruning
compounds and is unrecoverable, and why `candidates` should be generous at early steps.
`vector_search_many(rerank=)` stays **one statement**; its N reranker calls are per query
(one call cannot serve two, since the score is the (query, document) relationship) and
`arerank_many()` issues them concurrently, because that call exists to turn N round trips
into one.

**Scores do not survive into a traversal.** `Start` and `Hop` both document that a
traversal returns a subgraph rather than a ranking; similarity scores never survived it
and rerank scores get no exception, so `rerank_score` appears on `vector_search()` /
`vector_search_many()` hits only. What a rerank changed in a traversal is *which* nodes
are there, which is what `keep=` has always meant.

**The security posture is a grammar, not a blacklist.** `document_from` is
model-supplyable on purpose — field selection is a retrieval decision a model can make,
unlike an embedding it would have to invent — but the filter's output *is* the document
and the document is posted to a third party. `jqsafe.py` validates it against a **total
subset** of jq, and both claims are structural:

- **Sound**, because jq has no dynamic dispatch to builtins by name (`builtins` yields
  the *string* `"env/0"`) and no `eval`: the functions a program can call are exactly the
  literal names in its source, which a parser enumerates completely. `env`, `$ENV`,
  `input`, `include`/`import` are absent from the grammar rather than blacklisted.
- **Terminating**, because no unbounded-iteration construct parses — no `def`, `while`,
  `until`, `repeat`, `recurse`, `range`, `reduce`, `foreach`, no `label`/`break`. That is
  not a nicety: libjq holds the GIL through an infinite filter (no watchdog can fire, and
  SIGINT is ignored) and calls `abort()` on memory exhaustion, so neither failure can be
  caught in-process and both have to be made unreachable instead. A statically computed
  growth factor (`MAX_GROWTH`) bounds output size, which totality does not.

It is a **gate, not an interpreter**: what it accepts is handed to libjq unchanged, since
reimplementing jq's semantics for `//`, `?` and null handling is how an edge case becomes
a silently different document. `build_documents()` validates by default and
`trusted=True` is the opt-in — the same polarity as `allow_vectors=`/`all=`/`detach=`,
because an `untrusted=True` you must remember fails open. That opt-in belongs to that
call alone: `Rerank` carries no `trusted=`, so every query path — `rerank_hits()` on a
flat search, `_rerank_pins()` in a traversal, and their async twins — validates
unconditionally. By the time a filter reaches a query there is nothing left to tell who
wrote it. `fields=` narrows it further to
an operator's allowlist of readable paths (at or beneath an allowed path; a read *above*
one hands back the siblings the allowlist withholds), which is what keeps
`.properties.ssn` out of a vendor's logs.

## Extra columns

A custom `node_table=`/`edge_table=` may carry columns beyond `id`/`start_id`/`end_id`/
`graph_id`/`properties` — a foreign key to a `users` table is the motivating case.
`Graph.__init__` diffs the table's columns against the ones it already has a use for
(`models.NODE_IDENTITY_KEYS`/`EDGE_IDENTITY_KEYS`, plus the configured id/start/end/
graph columns, plus anything `vec_*` — vector fields keep their own write path and must
never be mistaken for one of these) and calls what is left over `node_extra_cols` /
`edge_extra_cols`, computed once and never revisited.

- **Write**: `Ingestor.__init__` widens the identity-key set `split_row()` uses by these
  names, so a flat row addressing one is pulled into `identity`, not `properties` —
  `ingest.py:_node_payload`/`_edge_payload` then copy it onto the insert record exactly
  like `id`. `_require_uniform_columns` is `_require_uniform`'s generalization: a batch
  where some rows name an extra column and others do not would bind NULL for the rest
  (the same executemany trap `id` has), so it is refused per-column, not just for `id`.
  `merge_nodes`/`merge_edges` (`_merge_payload`) also add each extra column the payload
  actually carries to the `ON CONFLICT DO UPDATE` `set_=` as `EXCLUDED.<col>` — a plain
  column has no `||` merge the way `properties` does, so a re-import simply overwrites
  it, in step with `properties`.
- **Read**: `Graph.traverse()`'s two hydration `SELECT`s (not `build_query()`, which only
  ever returns tagged `(kind, id)` rows) add the extra columns and fold each into its
  node/edge dict by name. With no extra columns declared the statement and the dict
  shape are unchanged from before this existed — the same "byte-identical when unused"
  contract vectors.py's `near=` keeps.
- **Constraints**: no new vocabulary — `Col("user_id")` (constraints.py) already named a
  real column before this existed (`Col("start_id")`), so `Unique`/`Index`/merge's `on=`
  need nothing extra to reach one.
- **Column collisions are refused, not guessed.** Naming an extra column in a place that
  means a JSONB property — `Property('user_id', ...)`/a dataclass field of that name,
  `Required("user_id")`/`Unique("user_id")` etc. with no `Col(...)`, or a bare
  `on=["user_id"]` — used to compile silently onto `properties->>'user_id'`, a key
  ingestion never populates for that name. `constraints._reject_column_collision()`
  (checked in `_key_expression`, so `Unique`/`Index` **and** `merge_nodes`/`merge_edges`'s
  `on=` share one guard — both render through `key_sql()`) and
  `schema.check_no_column_collisions()` (checked by `Graph.define_schema()` **and**
  `load_schema()`, since a saved schema can be adopted onto a different table) both raise
  naming the exact collision. Neither guesses a `Col(...)` on the caller's behalf —
  refusing beats guessing, same as everywhere else in this library — and neither touches
  `Check(...)`, whose filter tree has no fixed key list to check without re-deriving
  `schema.py`'s own `_filter_vocabulary` walk.
- **Deliberately out of scope**: `update_nodes()`/`update_edges()` stay `properties`-only
  (`set=`/`remove=` are a JSONB merge with no equivalent for a plain column) and Cypher
  writes never populate one (a property map literal has nowhere else to go but
  `properties`) — both documented in ingest.py/models.py rather than half-supported.

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

## Async

`hopai/asyncio.py`'s `AsyncGraph` is not a second traversal/mutation/ingestion
implementation — it is SQLAlchemy's own sync/async bridge, `AsyncSession`/
`AsyncConnection.run_sync()`, aimed at the exact sync functions `Graph` already calls:
`core.py`'s `_traverse_with_session()`/`_aggregate_with_session()`, `Mutator`'s and
`Ingestor`'s methods (already `connection=`-taking, for `mutate()`/`ingest()`'s
one-transaction-per-document guarantee — the same seam the greenlet bridge needs, so
most of it existed before `AsyncGraph` did), and `vectors.py`'s functions (which picked
up the same `connection=` parameter for this). `run_sync(fn)` hands `fn` a plain sync
`Session`/`Connection`, bridged through a greenlet — the function bodies on the other
side are unaware anything async is happening.

Schema and constraint declaration — `create_schema()`, `enforce_schema()`,
`define_constraints()`, `save_schema()`/`load_schema()`, `infer_schema()`,
`schema_violations()`, `add_networkx()`, `load_vectors()` — has no async override. These
are one-time setup calls with no concurrency to gain, and `AsyncGraph`'s wrapped `Graph`
runs on `AsyncEngine.sync_engine`, a facade only safe to execute against *inside* a
greenlet `run_sync()` spawns. `AsyncGraph.__getattr__` refuses these by name, pointing at
a plain `Graph` on the same database, rather than letting the facade fail with
SQLAlchemy's own `MissingGreenlet` outside the context that explains it.
`embed_stale()` gets the same refusal for a different reason: it is a bulk backfill that
opens its own transaction PER PAGE rather than resolving everything and opening one, so
it cannot take the `aplan_vector_writes()` treatment issue #74 gave every other write
without a second, divergent backfill implementation.

A throwaway benchmark (not committed) compared this design against `asyncio.to_thread()`
— the shape a naive async wrapper defaults to, and the same one `LangChain`'s `ainvoke`
falls back to when a `Runnable` has no native async implementation. Both delivered real
concurrency; the greenlet bridge did it on a **constant one OS thread** regardless of
concurrency, while the thread-pool version held one thread per in-flight call, climbing
with load. Wall-clock did **not** consistently favor the greenlet path — sometimes the
thread pool was faster, and the gap widened at higher concurrency, most likely because
each `AsyncSession.run_sync()` call's own setup (session open/close, greenlet spawn) is
paid serially on the one event-loop thread. The case for this design is the resource
ceiling a thread-pool wrapper eventually hits in a real server, not a guaranteed speed
win — see `hopai/asyncio.py`'s module docstring for the numbers.

The bridge only covers *database* I/O — SQLAlchemy's own calls yield back to the loop
from inside the greenlet, but an arbitrary blocking call made from `fn` does not, and
just holds the one event-loop thread until it returns. An embedding provider's HTTP call
is exactly that: reachable from `Near(text=...)` on every read path and from a text row
in `set_vectors()` (issue #74).

`hopai/embeddings.py`'s `Embedder` grew an `aembed_*()`
twin of every `embed_*()` method for this — native async when the client has one
(`openai.AsyncOpenAI()` and siblings, matched by the same module-name-plus-shape rule as
the sync client, `inspect.iscoroutinefunction` telling a client's async method apart from
its sync namesake of the same name) and `asyncio.to_thread()` otherwise, since a provider
SDK blocked on a socket releases the GIL (the same reasoning does **not** extend to a
CPU-bound C extension, which a thread pool would leave the loop starved for anyway).

`AsyncGraph.traverse()`/`aggregate()`/`vector_search()`/`vector_search_many()` await
`vectors.py`'s `aresolve_*()` helpers and `set_vectors()` awaits `aplan_vector_writes()`
— all of it *before* `run_sync()`/`begin()` runs, so every `Near`/row `fn` ever sees
already carries a plain vector and the sync functions below make no provider call of
their own. A call with no `text=` is handed back untouched, so the sync path and the
emitted SQL are unchanged.

`tests/test_async.py::TestTextEmbeddingStaysOffTheLoop` is the measurement: an
independent loop task made zero progress during a provider call before this.

**Reranking is the second network call on this path** and gets the same treatment from
the other end: it scores rows the database has already returned, so the obvious place
for it is inside the `run_sync()` block that fetched them — which is the loop's thread
again. `AsyncGraph.vector_search()`/`vector_search_many()` therefore call
`search_candidates()`/`search_many_candidates()` through the bridge and await
`arerank_hits()`/`arerank_many()` *after* the connection block closes; a reranked
traversal runs `AsyncGraph._rerank_pins()`, which is probe-through-the-bridge then
score-outside-it, once per reranked step. The rerank query is read from the specs **as
written**, before `aresolve_spec_texts()` resolves them — a resolved `Near` carries
floats and no text — and `vectors._resolved_spec()` re-attaches `rerank=` to each copy
that resolution rebuilds, since a `Start`/`Hop` validates on construction and would
otherwise refuse itself for the text it just had.
`tests/test_async.py::TestRerankingStaysOffTheLoop` is that measurement.

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
- The optional **`graphs` registry** (`models.GraphRegistry`, issue #85) sits
  **alongside** `hopai_schema`, not merged into it — one answers "what is this graph
  called and what is it for" (`id`/`name`/`description`, meant to be listed and
  skimmed), the other "what is the declared node/edge contract for it" (one JSON
  document); genuinely different questions. It follows `node_table=`/`edge_table=`'s
  PATTERN (a real `Table`, swappable via `Graph(graph_table=...)`, extended past
  `id`/`name`/`description` the same way `node_extra_cols`/`edge_extra_cols` work) but
  `hopai_schema`'s TIMING — created lazily by `Graph.create_graph()` on first use, never
  by `create_schema()`, so an existing caller who never opts in sees no schema change.
  `create_graph()` upserts the calling handle's own graph id, so calling it again to
  rename a graph is the normal way to use it. **No foreign key** ties
  `nodes.graph_id`/`edges.graph_id` to it — deliberately, since a hard FK would need
  every `graph_id` an existing database already has to gain a row here before the
  constraint could be added (a migration this feature is not allowed to require), and
  would turn `Graph(engine, graph="anything")` into a write that can fail for a caller
  who never registered anything. So the registry stays purely descriptive: a graph with
  rows and no registry entry is not an error anywhere, it is simply unnamed, and
  `Graph.graph_info()` answers `None` rather than guessing. `list_graphs`
  (`hopai/mcp.py`) reads it opt-in, via `registry=True` — off by default, since that
  tool is otherwise the one call that answers with no database connection at all
  (`Served.pick()`'s docstring calls this out: a graph created after start-up is not
  silently in scope either way, but a registry entry added after start-up is exactly
  the kind of thing the next call should see).

## The MCP server

`hopai/mcp.py` is a front end in the same sense `json_api.py` and `cypher.py` are: each
tool is one call into those, and there is no query logic in it. What is not obvious from
the outside:

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
- **The reranker is the operator's, and so is its budget.** `serve(rerank=…)` takes a
  built `Rerank` — Python-only, since a client object has no command-line spelling —
  and `tools()` wraps it in a `json_api.RerankPolicy` so the JSON and MCP front ends
  enforce one set of rules. What a tool call may supply is `document_from` and
  `candidates`; `rerank_fields` (required beside `rerank`) is the allowlist that filter
  may read, and `max_candidates` the ceiling it may spend, both refusing rather than
  clamping. `_with_rerank()` is the surface half: with a policy it rewrites the two
  advertised sentences that could otherwise only be written in the abstract (via
  `_reworded()`, which fails loudly if `json_api`'s wording moved), and **with no policy
  it deletes the `rerank` object from both step schemas** — the same "which surface
  exists" gate permissions use. `search_similar` never carries one, because it embeds
  the model's text into a raw vector itself and a raw-vector `Near` with `rerank=`
  refuses; the reranked route is `traverse_graph` with `near: {field, text}`, which
  means a reranked result *with scores* is not something MCP can return at all — a
  traversal is a subgraph.
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
| `hopai/rerankers.py` | The one-method contract and why it is not split by modality, `document_from` as a rule, why scores are re-paired by index, and why a spent call raises instead of degrading |
| `hopai/jqsafe.py` | What was measured about jq in a server, the soundness and totality claims the subset rests on, and why each excluded family is excluded |
| `hopai/schema.py` | The graph-schema notations, the annotation mapping, what enforcement compiles to and the endpoint-type limit, and why inference is an observation rather than a `.schema` fallback |
| `hopai/cypher.py` | The translatable subset — read, write and aggregate — and why each refusal is a refusal |
| `hopai/asyncio.py` | Why `AsyncGraph` is a bridge, not a second implementation, what it deliberately does not cover, and the benchmark numbers behind the design |
| `hopai/mcp.py` | The tool inventory, the four permission levels, why similarity takes text, and the 1.x/2.x SDK adapter |
| `tests/conftest.py` | The fixture graph's shape |
| `benchmarks/benchmarks.ipynb` | hopai vs. a hand-written recursive CTE and vs. Neo4j, measured, charted; `benchmarks/README.md` indexes the rest (vector search, reranking documents, Apache AGE) |
