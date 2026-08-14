"""
hopai.vectors

Vector similarity for nodes and edges, in the PostgreSQL you already
run -- no pgvector, no extension, no approximate index.

    from hopai import Vector, Near

    graph.define_vectors(nodes=[Vector("summary", 1536), Vector("title", 384)])
    graph.migrate_vectors()          # ALTER TABLE: one real[] column per field

    graph.set_vectors(nodes=[{"id": 1, "summary": embedding, "title": other}])

    graph.vector_search(Near("summary", query_embedding), k=10,
                        where={"type": "person"})
    # -> [{"id": "1", "similarity": 0.93, "properties": {...}}, ...]

    graph.traverse(                  # similarity as a traversal seed
        Start(near=Near("summary", query_embedding), k=25),
        Hop(via={"kind": "cites"}, hops=(1, 3)),
    )

WHY NOT PGVECTOR: this library's first rule is that Postgres and
SQLAlchemy are the whole stack -- a feature needing an extension is the
wrong feature, however popular the extension. So a vector field is an
ordinary `real[]` column and similarity is computed by Postgres itself,
once per candidate row, as a LATERAL:

    FROM nodes JOIN LATERAL (
        SELECT sum(x*y) / nullif(sqrt(sum(x*x)) * <query norm>, 0) AS s
        FROM unnest(vec_summary, :query) AS t(x, y)) AS near_0 ON true

That is EXACT cosine similarity over every candidate row -- no index
can serve an exact ORDER BY similarity, so none is created and none is
pretended. The cost model, measured rather than hoped
(benchmarks/bench_vectors.py; Postgres 16, one core): the executor
pays roughly 0.13 microseconds per vector element, so one candidate
costs about dimensions x 0.13us -- ~0.05ms per 384-dim row, ~0.2ms per
1536-dim row, and an unfiltered scan of 20,000 384-dim vectors lands
near one second. "Candidates" is what is left AFTER the `where=`
filter and the graph discriminator, both served by the existing
indexes -- and because the search is exact, filtering costs nothing
extra: the filtered-vector-search that approximate indexes struggle
with (filter first, rank the survivors) is simply how every search
here runs. A few thousand filtered candidates answer interactively;
tens of thousands per query is the practical ceiling. Past it, this
feature is the wrong tool: the columns are ordinary Postgres columns,
and a manual `ALTER TABLE ... USING vec_x::vector(d)` moves them to
pgvector without this library's involvement.

WHY THESE STORAGE CHOICES, each visible in the DDL:

  - `real[]`, not JSONB: a vector inside `properties` would bloat the
    GIN index with thousands of meaningless keys and drag 6KB of floats
    through every traversal result. Vector columns sit BESIDE the
    properties bag, so the read path and its indexes are untouched --
    vectors never appear in traversal results or `properties`; read
    them back with get_vectors().
  - `real` (float4), not float8: embeddings are produced as float32,
    so double storage would be 2x the bytes for made-up precision.
    The similarity arithmetic casts to float8 BEFORE summing, because
    Postgres's sum(real) accumulates in float4 and loses real
    precision over a long vector.
  - dimensions are a per-graph CHECK constraint, not part of the
    column type: one shared column can then hold 1536-dim vectors for
    one graph and 768-dim for another, each enforced server-side --
    the same scope_check() machinery every other constraint uses.
    (pgvector's `vector(d)` fixes d in the type, which multi-graph
    tables could not express.)
  - SET STORAGE EXTERNAL: float noise does not compress, so the
    default TOAST compression would burn CPU on every read for nothing.

COSINE ONLY, on purpose. Every current embedding API ships vectors
normalized to unit length, and on unit vectors cosine, dot product and
euclidean distance produce the same ranking -- so one metric covers
ranking, and scores stay in [-1, 1], which is what makes weighted
multivector combination meaningful. A `metric=` knob would double the
test surface to change answers only for unnormalized vectors.

MULTIVECTOR SEARCH means several named fields combined in one ranked
query -- pass several Near specs and the score is the weighted sum:

    graph.vector_search(Near("summary", q1, weight=0.7),
                        Near("title", q2, weight=0.3), k=10)

`missing=` on each Near decides rows lacking THAT field: "exclude"
(default) drops them, "zero" scores that field 0 and lets the others
carry the row. A stored all-zero vector has no cosine direction and
ranks as missing. The OTHER meaning of "multivector" -- late
interaction / ColBERT, many vectors per row with a maxsim -- is
refused, not approximated: pure SQL would make it O(rows x tokens^2 x
dims) per query, which is not a feature, it is a timeout.

VECTORS NEVER TRAVEL THROUGH THE LLM. There is deliberately no `near`
in TRAVERSE_TOOL_SCHEMA and no vector-search tool schema: a
tool-calling model asked for an embedding will invent 1536 plausible
floats, and an invented embedding produces confidently wrong
neighbors. Embedding text is the application's job; the JSON forms
(`"near"` in a traversal spec, vector_search_json()) exist for HTTP
and config callers that hold real vectors. This is the one place a
parser accepts more than the tool schema advertises, and it is pinned
by a test.

WRITES: set_vectors() only -- add_nodes/merge rows do not carry
vectors, because embeddings are computed after the entity exists and a
reserved key would silently change the meaning of a property someone
already stores. One transaction per call, like every other write; a
row whose id does not exist fails the whole call by name.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import (
    Column, and_, case, cast, column as sa_column, func, literal, or_, select, text,
    update, values,
)
from sqlalchemy import String as SAString
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION, REAL
from sqlalchemy.exc import IntegrityError

from .constraints import ConstraintViolation, _literal, _slug, _Target
from .filters import resolve

#: Every vector field lives in a real column named after itself with
#: this prefix -- a fixed prefix rather than the bare field name, so a
#: field can never collide with id/start_id/properties or any column a
#: caller's own table brought along.
VECTOR_COLUMN_PREFIX = "vec_"

#: A field name becomes a column identifier, so it is held to
#: identifier rules up front instead of failing as mangled DDL later.
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")

_MISSING_MODES = ("exclude", "zero")

_TARGETS = ("nodes", "edges")


@dataclass
class Vector:
    """One named vector field: `Vector("summary", 1536)`.

    Declared per target in define_vectors(nodes=[...], edges=[...]);
    the migration it implies is applied by migrate_vectors()."""
    name: str
    dimensions: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise ValueError(
                f"a vector field name must match [a-z][a-z0-9_]* -- it becomes a real "
                f"column ({VECTOR_COLUMN_PREFIX}<name>), not a JSONB key. Got {self.name!r}"
            )
        if len(self.name) > 59:
            raise ValueError(
                f"vector field name {self.name!r} is longer than 59 characters, and "
                f"{VECTOR_COLUMN_PREFIX}<name> must fit Postgres's 63-character identifier limit"
            )
        if not isinstance(self.dimensions, int) or isinstance(self.dimensions, bool) \
                or self.dimensions < 1:
            raise ValueError(
                f"Vector({self.name!r}): dimensions must be a positive integer, "
                f"got {self.dimensions!r}"
            )

    @property
    def column_name(self) -> str:
        return f"{VECTOR_COLUMN_PREFIX}{self.name}"


class Near:
    """A similarity spec: one field, one query vector.

    Where it appears decides what it does -- rank candidates in
    vector_search(), pick the top-k seed set on Start, prune to the
    top-k reached nodes on a Hop. How many to keep (`k`) always
    belongs to the surrounding call/Start/Hop, never to Near itself:
    one Near per field, one k per ranked set.

    weight:          this field's coefficient in the combined score
                     (only meaningful when several Near are combined).
    min_similarity:  drop rows whose similarity ON THIS FIELD is below
                     the bound -- a filter, applied before k.
    missing:         "exclude" (default) drops rows lacking this
                     field's vector; "zero" scores them 0 here and
                     lets other fields carry the row.
    """

    #: Marker for filters.resolve(), which must refuse a Near in
    #: where=/via= by name without importing this module.
    _is_near = True

    def __init__(self, field: str, vector, weight: float = 1.0,
                 min_similarity: Optional[float] = None, missing: str = "exclude"):
        if not isinstance(field, str) or not field:
            raise TypeError(f"Near field must be a vector field name, got {field!r}")
        self.field = field
        self.vector = _clean_vector(vector, f"Near({field!r})")
        if _norm(self.vector) == 0.0:
            raise ValueError(
                f"Near({field!r}): the query vector is all zeros, which has no direction -- "
                f"cosine similarity to it is undefined for every row"
            )
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) \
                or not math.isfinite(weight) or weight == 0:
            raise ValueError(
                f"Near({field!r}): weight must be a non-zero finite number -- a zero weight "
                f"contributes nothing; drop the Near instead. Got {weight!r}"
            )
        self.weight = float(weight)
        if min_similarity is not None:
            if not isinstance(min_similarity, (int, float)) or isinstance(min_similarity, bool) \
                    or not -1 <= min_similarity <= 1:
                raise ValueError(
                    f"Near({field!r}): min_similarity is a cosine similarity bound and must "
                    f"be between -1 and 1, got {min_similarity!r}"
                )
            min_similarity = float(min_similarity)
        self.min_similarity = min_similarity
        if missing not in _MISSING_MODES:
            raise ValueError(
                f"Near({field!r}): missing must be one of {_MISSING_MODES}, got {missing!r}"
            )
        self.missing = missing

    def __repr__(self) -> str:
        parts = [repr(self.field), f"{len(self.vector)} dims"]
        if self.weight != 1.0:
            parts.append(f"weight={self.weight!r}")
        if self.min_similarity is not None:
            parts.append(f"min_similarity={self.min_similarity!r}")
        if self.missing != "exclude":
            parts.append(f"missing={self.missing!r}")
        return f"Near({', '.join(parts)})"


def _clean_vector(vector, owner: str) -> tuple:
    """Validate and normalize a vector to a tuple of finite floats.

    tolist() first: embeddings usually arrive as numpy float32 arrays,
    and refusing them over the container type would be pedantry --
    while NaN/Inf are refused loudly, because one NaN turns every
    similarity involving that vector into ranked garbage."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if not isinstance(vector, (list, tuple)) or not vector:
        raise TypeError(
            f"{owner}: the vector must be a non-empty list of numbers, "
            f"got {type(vector).__name__}"
        )
    cleaned = []
    for i, value in enumerate(vector):
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(value):
            raise ValueError(
                f"{owner}: element {i} is {value!r} -- every element must be a finite number "
                f"(NaN or Infinity would silently poison every similarity it touches)"
            )
        cleaned.append(float(value))
    return tuple(cleaned)


def _norm(vector: tuple) -> float:
    return math.sqrt(sum(value * value for value in vector))


def parse_near(spec: Any):
    """The JSON form of Near, mirroring parse_filter()/parse_aggregate():
    an object {"field": ..., "vector": [...]} plus the optional keys
    weight / min_similarity / missing, or a list of such objects."""
    if isinstance(spec, list):
        if not spec:
            raise ValueError('"near" is an empty list -- give at least one {"field", "vector"}')
        return [parse_near(one) for one in spec]
    if not isinstance(spec, dict):
        raise TypeError(f'"near" must be an object or a list of objects -- '
                        f'got {type(spec).__name__}')
    unknown = set(spec) - {"field", "vector", "weight", "min_similarity", "missing"}
    if unknown:
        raise ValueError(f'unknown "near" keys {sorted(unknown)} -- a near spec has "field", '
                         f'"vector" and optionally weight, min_similarity, missing')
    missing_keys = {"field", "vector"} - set(spec)
    if missing_keys:
        raise ValueError(f'a near spec needs {sorted(missing_keys)} -- '
                         f'e.g. {{"field": "summary", "vector": [...]}}')
    return Near(spec["field"], spec["vector"], weight=spec.get("weight", 1.0),
                min_similarity=spec.get("min_similarity"), missing=spec.get("missing", "exclude"))


# ---------------------------------------------------------------------
# Registry: which fields exist, per graph handle
# ---------------------------------------------------------------------

def build_registry(nodes, edges) -> dict:
    """What Graph.define_vectors() stores: {"nodes": {name: Vector},
    "edges": {...}}, validated. In memory only, like define_schema()."""
    registry: dict = {}
    for target, entries in (("nodes", nodes), ("edges", edges)):
        fields: dict = {}
        for entry in entries or ():
            if not isinstance(entry, Vector):
                raise TypeError(
                    f"{target}=[...] takes Vector(name, dimensions) entries, got {entry!r}"
                )
            if entry.name in fields:
                raise ValueError(f"duplicate vector field {entry.name!r} for {target} -- "
                                 f"declare each field once")
            fields[entry.name] = entry
        registry[target] = fields
    return registry


def _attach(table, column_name: str):
    """The vec_* column as SQLAlchemy metadata, adding it if this
    handle never declared it.

    drop_vectors() accepts bare field names and probes the catalog
    rather than the registry -- deliberately, since a fresh handle
    (in_graph(), or a teardown script that never declares anything)
    legitimately has no declaration for a column that exists in the
    database. Without this the UPDATE fails to compile with
    "Unconsumed column names", which names nothing the caller can act
    on."""
    if column_name not in table.c:
        table.append_column(Column(column_name, ARRAY(REAL), nullable=True))
    return table.c[column_name]


def attach_columns(graph) -> None:
    """Make the declared columns visible to SQLAlchemy so queries can
    reference them. Table metadata is shared between Graph handles on
    the same tables, and that is safe: an extra Column changes no
    behavior on its own -- every use is gated on this handle's registry."""
    for target, table in (("nodes", graph.nodes_tbl), ("edges", graph.edges_tbl)):
        for field in graph._vectors[target].values():
            _attach(table, field.column_name)


def _defined(graph, target: str, caller: str) -> dict:
    if target not in _TARGETS:
        raise ValueError(f"target must be one of {_TARGETS}, got {target!r}")
    registry = graph._vectors
    if registry is None or not registry.get(target):
        raise ValueError(
            f"{caller} needs vector fields and none are defined for {target} on this Graph -- "
            f"call define_vectors({target}=[Vector('name', dimensions)]) first (a handle from "
            f"in_graph() starts without them, like a schema)"
        )
    return registry[target]


def _field(graph, target: str, name: str, caller: str) -> Vector:
    fields = _defined(graph, target, caller)
    if name not in fields:
        # Led by the caller, like every other message here: in a chain
        # of hops "no vector field 'body'" alone leaves the reader
        # counting Hops by hand to find which one said it.
        raise ValueError(
            f"{caller}: no vector field {name!r} is defined for {target} in this graph -- "
            f"defined: {sorted(fields)}. define_vectors() declares a new one"
        )
    return fields[name]


def _table(graph, target: str):
    return graph.nodes_tbl if target == "nodes" else graph.edges_tbl


def validate_nears(graph, target: str, near, k, caller: str) -> list:
    """Normalize near= (one Near or a list) into a validated list:
    every entry a Near, every field defined for `target`, every query
    vector of the declared dimensions -- so a typo fails here with the
    fix named, not at execution as an undefined-column error."""
    nears = list(near) if isinstance(near, (list, tuple)) else [near]
    if not nears:
        raise ValueError(f"{caller}: near=[] is empty -- pass a Near(...) or a list of them")
    for one in nears:
        if not isinstance(one, Near):
            raise TypeError(
                f"{caller}: near= takes Near(field, vector) specs, got {one!r}"
            )
        field = _field(graph, target, one.field, caller)
        if len(one.vector) != field.dimensions:
            raise ValueError(
                f"{caller}: the query vector for {one.field!r} has {len(one.vector)} "
                f"dimensions, the field is defined with {field.dimensions}"
            )
    if k is None and all(one.min_similarity is None for one in nears):
        raise ValueError(
            f"{caller}: near= without k= and without any min_similarity changes nothing -- "
            f"ranking with no limit and no bound keeps every row. Pass k=<how many to keep>, "
            f"or min_similarity on a Near, or drop near="
        )
    return nears


# ---------------------------------------------------------------------
# Similarity as SQL
# ---------------------------------------------------------------------

def _similarity(table, near: Near, index: int):
    """Exact cosine similarity of one row's stored vector against the
    query, as a LATERAL join yielding one column.

    LATERAL, not a correlated scalar subquery, and that is a
    measurement rather than a preference: a scalar subquery sitting in
    a plain sub-SELECT gets pulled up by the planner and re-evaluated
    at EVERY place the outer query names it -- the filter, the score,
    the ORDER BY -- so the expensive unnest ran twice per candidate
    (three times with min_similarity), for identical results. A LATERAL
    is computed once per row and referenced as an ordinary column;
    measured at 1.9-2.2x faster on 4k x 384-dim, and it is what lets
    the guards below read the value at all.

    The query vector's own norm is a Python constant, so only the dot
    product and the STORED vector's norm are computed per row. The
    float8 casts are load-bearing: sum(real) accumulates in float4 and
    drifts over long vectors.

    NULL -- which every caller reads as "missing" -- when the stored
    vector is NULL, all zeros (no direction), or NOT THE DECLARED
    LENGTH. That last guard is not paranoia: unnest(a, b) pads the
    shorter array with NULLs and sum() skips them, so a stored vector
    of the wrong size would otherwise score a confident cosine over
    whatever prefix the two share -- a silently wrong answer, which is
    the worst thing this library can produce. It is reachable whenever
    a field's declared dimensions and its stored rows disagree (a
    redefinition not yet re-migrated, say)."""
    column = table.c[VECTOR_COLUMN_PREFIX + near.field]
    zipped = func.unnest(
        column, literal(list(near.vector), type_=ARRAY(REAL))
    ).table_valued("x", "y").render_derived()
    x = cast(zipped.c.x, DOUBLE_PRECISION)
    y = cast(zipped.c.y, DOUBLE_PRECISION)
    stored_norm = func.sqrt(func.sum(x * x))
    # type_ is not decoration: without it SQLAlchemy types nullif() as
    # NUMERIC, and a float8 numerator over a numeric denominator is
    # numeric division -- slower, and a different arithmetic than the
    # float8 the rest of this expression is careful to stay in.
    denominator = func.nullif(
        stored_norm * literal(_norm(near.vector), type_=DOUBLE_PRECISION),
        literal(0.0, type_=DOUBLE_PRECISION),
        type_=DOUBLE_PRECISION,
    )
    comparable = func.array_length(column, 1) == len(near.vector)
    # No else_: a CASE with no ELSE is NULL, which is the "missing"
    # every caller already reads. Spelling it literal(None) makes
    # SQLAlchemy warn about rendering NULL as a bound parameter.
    body = select(
        case((comparable, func.sum(x * y) / denominator)).label("s")
    ).select_from(zipped)
    return body.lateral(f"near_{index}")


def _with_laterals(from_obj, laterals: list):
    """Attach the per-field LATERALs to a FROM clause.

    An explicit `JOIN LATERAL (...) ON true` rather than the comma
    form: they mean the same thing to Postgres, but SQLAlchemy reads a
    comma with no ON clause as an accidental cartesian product and
    warns on every single search -- a warning that is wrong here (a
    LATERAL is correlated by construction) and would teach readers to
    ignore the ones that are right."""
    for lateral in laterals:
        from_obj = from_obj.join(lateral, literal(True))
    return from_obj


def _similarity_terms(table, nears: list) -> tuple:
    """(laterals, labeled columns, candidate guards) for one query.

    Each field is computed once, in a LATERAL; `sim_i` is that value,
    coalesced to 0 for missing="zero" so a threshold and the combined
    score read the same number for the same row."""
    laterals, columns, guards, has_direction = [], [], [], []
    for i, near in enumerate(nears):
        lateral = _similarity(table, near, i)
        laterals.append(lateral)
        has_direction.append(lateral.c.s.isnot(None))
        value = func.coalesce(lateral.c.s, 0.0) if near.missing == "zero" else lateral.c.s
        columns.append(value.label(f"sim_{i}"))

    # A cheap prefilter, so rows carrying no vector at all never reach
    # the arithmetic. It is an optimization, NOT the correctness
    # boundary -- a present-but-directionless vector passes it, and is
    # caught by the NULL similarity downstream.
    present = [table.c[VECTOR_COLUMN_PREFIX + one.field].isnot(None)
               for one in nears if one.missing == "exclude"]
    if present:
        guards.extend(present)
    else:
        # Every field says "zero", so nothing NULLs the combined score
        # and the row must be shown to have at least one usable vector
        # here. `IS NOT NULL` on the columns is not enough: an all-zero
        # or wrong-length vector is present and still means "missing",
        # and such a row would otherwise rank at a meaningless 0 --
        # above a row whose real vector points the other way.
        guards.append(or_(*(table.c[VECTOR_COLUMN_PREFIX + one.field].isnot(None)
                            for one in nears)))
        guards.append(or_(*has_direction))
    return laterals, columns, guards


def _combined(inner, nears: list):
    total = None
    for i, near in enumerate(nears):
        term = inner.c[f"sim_{i}"] if near.weight == 1.0 else inner.c[f"sim_{i}"] * near.weight
        total = term if total is None else total + term
    return total


def _thresholds(inner, nears: list) -> list:
    return [inner.c[f"sim_{i}"] >= one.min_similarity
            for i, one in enumerate(nears) if one.min_similarity is not None]


def ranked_ids(graph, table, id_expr, from_obj, condition, nears: list, k: Optional[int]):
    """A Select of the ids that survive similarity: the shared shape
    behind a near= seed CTE and a near= match CTE. `condition` is the
    caller's full predicate, graph scope included (None when the scope
    already lives in from_obj's join) -- this function only adds the
    similarity layer."""
    conditions = [] if condition is None else [condition]
    laterals, columns, guards = _similarity_terms(table, nears)
    inner = (
        select(id_expr.label("node_id"), *columns)
        .select_from(_with_laterals(from_obj, laterals))
        .where(*conditions, *guards)
        .subquery()
    )
    combined = _combined(inner, nears)
    # `combined IS NOT NULL` belongs to BOTH shapes, not just the
    # ranked one: it is what drops a row whose exclude-mode vector has
    # no direction. Inside `if k is not None` it meant one set of Near
    # specs matched different rows depending on whether k was passed --
    # a threshold-only query kept rows that the same query with k
    # correctly rejected.
    query = select(inner.c.node_id).where(combined.isnot(None), *_thresholds(inner, nears))
    if k is not None:
        query = query.order_by(combined.desc(), inner.c.node_id).limit(k)
    return query


def build_search_query(graph, near, target: str = "nodes", k: int = 10, where: Any = None):
    """The single statement vector_search() runs. Exposed so the SQL
    can be inspected with no database, like build_query()."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    nears = validate_nears(graph, target, near, k, "vector_search()")
    table = _table(graph, target)
    id_column = getattr(table.c, graph.node_id_col if target == "nodes" else graph.edge_id_col)

    identity = [id_column.label("_id")]
    if target == "edges":
        identity.append(getattr(table.c, graph.edge_start_col).label("_start"))
        identity.append(getattr(table.c, graph.edge_end_col).label("_end"))

    laterals, sim_columns, guards = _similarity_terms(table, nears)
    inner = (
        select(*identity, table.c.properties.label("properties"), *sim_columns)
        .select_from(_with_laterals(table, laterals))
        .where(and_(graph._scoped(table), resolve(table.c.properties, where), *guards))
        .subquery()
    )
    combined = _combined(inner, nears)
    similarity = combined.label("similarity")
    # Ids are cast to text to match the contract traversal results
    # already follow; the ORDER BY tiebreak uses the RAW id, because
    # '10' sorts before '9' as text and determinism should not change
    # the neighbors' order.
    columns = [cast(inner.c._id, SAString).label("id")]
    if target == "edges":
        columns.append(cast(inner.c._start, SAString).label("start_id"))
        columns.append(cast(inner.c._end, SAString).label("end_id"))
    columns += [inner.c.properties, similarity]
    return (
        select(*columns)
        .where(combined.isnot(None), *_thresholds(inner, nears))
        .order_by(similarity.desc(), inner.c._id)
        .limit(k)
    )


def search(graph, near, target: str = "nodes", k: int = 10, where: Any = None) -> list:
    query = build_search_query(graph, near, target=target, k=k, where=where)
    with graph.engine.connect() as connection:
        rows = connection.execute(query).mappings().all()
    return [{**row, "similarity": float(row["similarity"])} for row in rows]


# ---------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------

def _targets(graph) -> dict:
    return {
        "nodes": _Target(graph.nodes_tbl, "nodes", graph.graph, graph.graph_col or "graph_id"),
        "edges": _Target(graph.edges_tbl, "edges", graph.graph, graph.graph_col or "graph_id"),
    }


def _target_for(graph, target_name: str) -> _Target:
    built = _targets(graph)[target_name]
    if graph.graph_col is None:
        built.graph = None
    return built


def _constraint_name(target: _Target, field: Vector) -> str:
    """The dimension CHECK's name: unique per (graph, table, field),
    deterministic, and within Postgres's 63-character limit.

    NOT _auto_name() + scope_name(), which is what this used to be and
    which collides two different ways -- both silent, both producing
    wrong answers rather than errors:

      - Each truncates to 63 INDEPENDENTLY, so once the base name fills
        the budget the graph suffix is cut off entirely and every graph
        shares one constraint. The second graph's migrate_vectors()
        then finds the first graph's constraint, matches the dimension
        regex, and reports success having added nothing -- so that
        graph accepts any dimensionality, and unnest() pads the short
        side with NULLs into a confident wrong score. Worse,
        drop_vectors() in one graph drops the other graph's constraint.
        Vector() cannot catch this itself: it validates the 59-char
        COLUMN budget, while the constraint budget depends on the table
        name and the graph, neither of which it knows.
      - _slug'd names contain '_', so field 'v_b' in graph 'a' and
        field 'v' in graph 'b_a' produce one name with no truncation at
        all. schema.py hit this exact ambiguity and answered it with
        _graph_token, a fixed-charset token carrying no '_'; reusing it
        keeps one answer in the codebase instead of two.
    """
    from .schema import _graph_token

    token = _graph_token(target.graph)
    name = f"ck_vec_dims_{token}_{target.table.name}_{_slug(field.name)}"
    if len(name) <= 63:
        return name
    # Too long to stay readable: the tail becomes a digest of exactly
    # the parts that would otherwise be cut, so distinctness survives.
    digest = hashlib.sha256(
        f"{target.graph}\0{target.table.name}\0{field.name}".encode()).hexdigest()[:12]
    return f"{name[:50]}_{digest}"


def _dims_check_body(target: _Target, field: Vector) -> str:
    column = sa_column(field.column_name, ARRAY(REAL))
    shape = or_(
        column.is_(None),
        and_(func.array_ndims(column) == 1, func.array_length(column, 1) == field.dimensions),
    )
    return _literal(target.scope_check(shape))


def _field_ddl(target: _Target, field: Vector) -> list:
    return [
        f'ALTER TABLE {target.qualified} ADD COLUMN IF NOT EXISTS '
        f'"{field.column_name}" real[]',
        # Float noise does not compress; EXTERNAL skips the TOAST
        # compression attempt, so similarity scans read raw floats
        # instead of decompressing every row first.
        f'ALTER TABLE {target.qualified} ALTER COLUMN "{field.column_name}" '
        f'SET STORAGE EXTERNAL',
        f'ALTER TABLE {target.qualified} ADD CONSTRAINT "{_constraint_name(target, field)}" '
        f'CHECK ({_dims_check_body(target, field)})',
    ]


def vector_ddl(graph) -> list:
    """The exact SQL migrate_vectors() would run, without running it --
    the same contract as constraint_ddl() and schema_ddl()."""
    statements = []
    for target_name in _TARGETS:
        fields = graph._vectors or {}
        target = _target_for(graph, target_name)
        for field in (fields.get(target_name) or {}).values():
            statements.extend(_field_ddl(target, field))
    return statements


_COLUMN_TYPE = text("""
    SELECT udt_name FROM information_schema.columns
    WHERE table_name = :table AND column_name = :column
      AND table_schema = COALESCE(CAST(:schema AS text), current_schema())
""")

_CONSTRAINT_DEF = text("""
    SELECT pg_get_constraintdef(oid) FROM pg_constraint
    WHERE conname = :name AND conrelid = CAST(:table AS regclass)
""")


def migrate_vectors(graph) -> list:
    """Apply the declared vector fields: add each column and its
    per-graph dimension CHECK. Idempotent, one transaction, and it
    REFUSES drift instead of papering over it -- a column of the wrong
    type, or an existing constraint declaring different dimensions,
    names drop_vectors() (or the conflicting column) rather than
    quietly serving two incompatible definitions. Returns
    "table.column" for every field ensured, in order."""
    if graph._vectors is None:
        raise ValueError("migrate_vectors() needs vector fields and none are defined -- "
                         "call define_vectors(...) first")
    ensured = []
    with graph.engine.begin() as connection:
        for target_name in _TARGETS:
            table = _table(graph, target_name)
            target = _target_for(graph, target_name)
            for field in graph._vectors[target_name].values():
                existing = connection.execute(_COLUMN_TYPE, {
                    "table": table.name, "column": field.column_name, "schema": table.schema,
                }).scalar()
                if existing is not None and existing not in ("_float4", "_float8"):
                    raise ValueError(
                        f"{target_name}.{field.column_name} already exists with type "
                        f"{existing!r}, not a float array -- rename the vector field, or "
                        f"drop the conflicting column"
                    )
                add_column, storage, add_check = _field_ddl(target, field)
                connection.execute(text(add_column))
                connection.execute(text(storage))
                name = _constraint_name(target, field)
                definition = connection.execute(_CONSTRAINT_DEF, {
                    "name": name, "table": target.qualified,
                }).scalar()
                if definition is not None:
                    declared = re.search(
                        rf'array_length\("?{re.escape(field.column_name)}"?, 1\) = (\d+)',
                        definition,
                    )
                    if declared is None or int(declared.group(1)) != field.dimensions:
                        raise ValueError(
                            f"vector field {field.name!r} on {target_name} is already migrated "
                            f"with different dimensions (constraint {name!r}: {definition}) -- "
                            f"changing dimensions invalidates stored vectors; run "
                            f"drop_vectors({target_name}=[{field.name!r}]) first, then migrate "
                            f"and re-embed"
                        )
                else:
                    try:
                        connection.execute(text(add_check))
                    except IntegrityError as exc:
                        raise ConstraintViolation(
                            f"existing {target_name} rows in graph {graph.graph!r} carry "
                            f"{field.name!r} vectors that are not {field.dimensions}-dimensional, "
                            f"so the dimension constraint cannot be added -- run "
                            f"drop_vectors({target_name}=[{field.name!r}]) to clear them, "
                            f"then migrate and re-embed",
                            constraint=name,
                        ) from exc
                ensured.append(f"{table.name}.{field.column_name}")
    return ensured


def drop_vectors(graph, nodes=None, edges=None) -> list:
    """The inverse of migrate_vectors() for THIS graph: drop each
    field's dimension constraint and NULL its values in this graph's
    rows. The column itself stays -- it is shared by every graph in
    the table, so removing it is a deliberate manual ALTER, not a side
    effect of one graph cleaning up. Missing fields are ignored, like
    drop_constraints(). Returns the field names processed."""
    dropped = []
    with graph.engine.begin() as connection:
        for target_name, names in (("nodes", nodes), ("edges", edges)):
            table = _table(graph, target_name)
            target = _target_for(graph, target_name)
            for entry in names or ():
                field = entry if isinstance(entry, Vector) else Vector(entry, 1)
                connection.execute(text(
                    f'ALTER TABLE {target.qualified} DROP CONSTRAINT IF EXISTS '
                    f'"{_constraint_name(target, field)}"'
                ))
                exists = connection.execute(_COLUMN_TYPE, {
                    "table": table.name, "column": field.column_name, "schema": table.schema,
                }).scalar()
                if exists is not None:
                    column = _attach(table, field.column_name)
                    connection.execute(
                        update(table)
                        .where(graph._scoped(table), column.isnot(None))
                        .values({field.column_name: None})
                    )
                if graph._vectors:
                    graph._vectors[target_name].pop(field.name, None)
                dropped.append(field.name)
    return dropped


# ---------------------------------------------------------------------
# Reading and writing vectors
# ---------------------------------------------------------------------

def _coerce_id(value):
    """Traversal and search results carry ids as strings; accept them
    back without making the caller remember to int() them."""
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def set_vectors(graph, nodes=None, edges=None) -> int:
    """UPDATE existing rows' vector columns. Each row is
    {"id": ..., <field>: <vector or None>}; every non-id key must be a
    defined field of the right dimensionality. One transaction for the
    whole call: an id that matches no row in this graph fails
    everything by name, so a retry never collides with half a write.
    Returns the number of rows updated."""
    from .ingest import BATCH_SIZE, _chunks

    total = 0
    with graph.engine.begin() as connection:
        for target_name, rows in (("nodes", nodes), ("edges", edges)):
            if not rows:
                continue
            fields = _defined(graph, target_name, "set_vectors()")
            table = _table(graph, target_name)
            id_name = graph.node_id_col if target_name == "nodes" else graph.edge_id_col
            id_column = getattr(table.c, id_name)

            groups: dict = {}
            seen_ids = set()
            for row in rows:
                if not isinstance(row, dict) or "id" not in row:
                    raise ValueError(
                        f"each {target_name} row must be a dict with an 'id' plus vector "
                        f"fields, got {row!r}"
                    )
                row_id = _coerce_id(row["id"])
                if row_id in seen_ids:
                    raise ValueError(
                        f"{target_name} id {row['id']!r} appears twice in one set_vectors() "
                        f"call -- the second update would race the first; merge the rows"
                    )
                seen_ids.add(row_id)
                names = sorted(set(row) - {"id"})
                if not names:
                    raise ValueError(f"{target_name} row {row['id']!r} names no vector fields")
                cleaned = {}
                for name in names:
                    if name not in fields:
                        raise ValueError(
                            f"no vector field {name!r} is defined for {target_name} in this "
                            f"graph -- defined: {sorted(fields)}. define_vectors() declares "
                            f"a new one"
                        )
                    value = row[name]
                    if value is not None:
                        value = list(_clean_vector(value, f"{target_name} {row['id']!r} {name!r}"))
                        if len(value) != fields[name].dimensions:
                            raise ValueError(
                                f"{target_name} {row['id']!r}: vector for {name!r} has "
                                f"{len(value)} dimensions, the field is defined with "
                                f"{fields[name].dimensions}"
                            )
                    cleaned[name] = value
                groups.setdefault(tuple(names), []).append((row_id, cleaned))

            for names, group in groups.items():
                for chunk in _chunks(group, BATCH_SIZE):
                    incoming = values(
                        sa_column("id", id_column.type),
                        *(sa_column(f"v{i}", ARRAY(REAL)) for i in range(len(names))),
                        name="incoming",
                    ).data([
                        (row_id, *(cleaned[name] for name in names))
                        for row_id, cleaned in chunk
                    ])
                    statement = (
                        update(table)
                        .where(id_column == incoming.c.id, graph._scoped(table))
                        .values({
                            # The cast is for the all-None case: a VALUES
                            # column holding only NULLs is inferred as
                            # text, and text does not assign to real[].
                            VECTOR_COLUMN_PREFIX + name: cast(incoming.c[f"v{i}"], ARRAY(REAL))
                            for i, name in enumerate(names)
                        })
                        .returning(id_column)
                    )
                    updated = {row[0] for row in connection.execute(statement)}
                    missing = [row_id for row_id, _ in chunk if row_id not in updated]
                    if missing:
                        raise ValueError(
                            f"no {target_name[:-1]} with id {missing[0]!r} in graph "
                            f"{graph.graph!r} -- set_vectors() updates existing rows; "
                            f"nothing from this call was written"
                        )
                    total += len(chunk)
    return total


def get_vectors(graph, nodes=None, edges=None, fields=None) -> dict:
    """Read stored vectors back, since traversal results never carry
    them. Returns {"nodes": {id: {field: [floats] | None}}, "edges":
    {...}} with string ids, matching every other result. Ids that
    match no row are simply absent. `fields` narrows which fields are
    read (for both targets); default is all defined."""
    result: dict = {"nodes": {}, "edges": {}}
    for target_name, ids in (("nodes", nodes), ("edges", edges)):
        if not ids:
            continue
        defined = _defined(graph, target_name, "get_vectors()")
        wanted = list(fields) if fields is not None else sorted(defined)
        for name in wanted:
            if name not in defined:
                raise ValueError(
                    f"no vector field {name!r} is defined for {target_name} in this graph -- "
                    f"defined: {sorted(defined)}"
                )
        table = _table(graph, target_name)
        id_column = getattr(
            table.c, graph.node_id_col if target_name == "nodes" else graph.edge_id_col)
        query = select(
            cast(id_column, SAString).label("id"),
            *(table.c[VECTOR_COLUMN_PREFIX + name].label(name) for name in wanted),
        ).where(graph._scoped(table), id_column.in_([_coerce_id(i) for i in ids]))
        with graph.engine.connect() as connection:
            for row in connection.execute(query).mappings():
                result[target_name][row["id"]] = {name: row[name] for name in wanted}
    return result
