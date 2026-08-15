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
- The JSON forms exist (`"near"`/`"keep"`/`"via_near"`/`"via_keep"`/`"boost"` in specs,
  `vector_search_json`) but the LLM tool schemas deliberately omit them, and
  `traverse_json`/`aggregate_json` **refuse** them without `allow_vectors=True` — a model
  fills a `"vector"` parameter by inventing floats, so the invariant is enforced rather
  than advertised. `tests/test_vectors.py::TestToolSchemasStayVectorFree` pins both the
  omission and the per-graph `tool_schemas()` staying vector-free.

## The write path

`Graph` exposes the writes; `ingest.py` and `constraints.py` implement them; `core.py`
stays the traversal engine. They meet at `constraints.key_sql()`, which renders both
`CREATE INDEX` and the `ON CONFLICT` target — index inference only works when the two
are spelled identically, so they must come from one function.

Every write is one transaction, batching and multi-clause Cypher included. Half
committing is the worst failure this library has: the caller is told it failed, retries,
and the retry collides with rows that landed. `tests/test_ingest.py::TestAtomicity`
guards it.

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
- `merge_*` conflict targets go through the same `scope_index()`.
- `graph_col=None` disables all of it for callers bringing their own tables.
- `save_schema()` adds a third table, `hopai_schema(graph_id, document, saved_at)`, and
  it sits **outside** the two-table invariant on purpose: it is metadata about a graph,
  not graph data. It is never on the query path, traversal and ingestion do not know it
  exists, it is created lazily on first save (never by `create_schema()`), and its one
  row per `graph_id` is the same per-graph discipline as everything above.

## Two gotchas in the results

- **Ids come back as strings.** `build_query` casts them so node and edge ids share one
  union'd column. The hydration queries then cast the indexed BIGINT to text to match;
  removing that cast as an optimization breaks the contract the tests assert on.
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
| `hopai/constraints.py` | What each constraint compiles to, and the SQL semantics that surprise people |
| `hopai/vectors.py` | Why no pgvector, the storage and cost model, cosine-only, multivector semantics, and why vectors never pass through a tool schema |
| `hopai/schema.py` | The graph-schema notations, the annotation mapping, what enforcement compiles to and the endpoint-type limit, and why inference is an observation rather than a `.schema` fallback |
| `hopai/cypher.py` | The translatable subset — read, write and aggregate — and why each refusal is a refusal |
| `tests/conftest.py` | The fixture graph's shape |
| `benchmarks/README.md` | Measured numbers, including where raw CTEs beat this library 2-5x |
