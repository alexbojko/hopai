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
    # -> [{"id": "1", "similarity": 0.93, "properties": {...},
    #      "similarities": {"summary": 0.93}, "boosts": {}}, ...]

    graph.traverse(                  # similarity as a traversal seed
        Start(near=Near("summary", query_embedding), keep=25),
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
pretended.

The cost model, measured rather than hoped
(benchmarks/bench_vectors.py; Postgres 16, one core): the executor
pays roughly 0.13 microseconds per vector element, so one candidate
costs about dimensions x 0.13us -- ~0.05ms per 384-dim row, ~0.2ms per
1536-dim row, and an unfiltered scan of 20,000 384-dim vectors lands
near one second.

"Candidates" is what is left AFTER the `where=`
filter and the graph discriminator, both served by the existing
indexes -- and because the search is exact, filtering costs nothing
extra: the filtered-vector-search that approximate indexes struggle
with (filter first, rank the survivors) is simply how every search
here runs.

A few thousand filtered candidates answer interactively;
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

Tuning `weight=` per field is only as good as your ability to see
which field actually drove a result, so every hit also carries
`similarities`, keyed by FIELD NAME (never `sim_0` -- that is a SQL
alias, an implementation detail no caller should have to read):

    hits = graph.vector_search(Near("summary", q1, weight=0.7),
                               Near("title", q2, weight=0.3), k=10)
    hits[0]["similarities"]     # {"summary": 0.91, "title": 0.64}

A field a row is missing reports `None` there, even under
`missing="zero"` -- the COMBINED `similarity` still scores that field 0
(that is what `missing="zero"` means), but the per-field report keeps
saying "missing" honestly rather than making a skipped field look like
a poor match. `similarities` is always a dict, even for one Near, so
`hit["similarities"][field]` never needs to branch on how many Near
clauses a search used.

BATCH SEARCH ranks many queries in ONE statement -- the shape
retrieval actually has, since a question is usually expanded into
several sub-queries:

    graph.vector_search_many([Near("summary", q1), Near("summary", q2)], k=5)
    # -> [[...hits for q1...], [...hits for q2...]]

The queries become a VALUES list and the per-query top-k hangs off it
as a LATERAL, so N searches cost one round trip instead of N. Every
query must share a SHAPE (same fields, weights, thresholds, modes) --
only the vectors may differ, because one statement can only express
one shape; a mixed batch is refused rather than silently re-ranked.

HYBRID RANKING adds a non-similarity term with Boost(property,
weight): the score becomes sum(w_i * sim_i) + sum(w_j * boost_j).

A boost cannot push a row past a min_similarity floor (thresholds
read each field's own similarity, not the combined score) -- but it
DOES reorder, so with k it changes which rows come back, and the
`similarity` in each result is then the combined score, which can
exceed 1.

By default each boost is rescaled into [0, 1] -- similarity's
own scale, not its sign -- with a min-max window function over the
candidate set before it is weighted, so `weight` means what it says
regardless of the property's own scale -- a raw view count would otherwise not
boost a cosine ranking, it would replace it.

`Boost(..., scale="raw")`
is the unbounded, unscaled escape hatch. See Boost.

A boost that dominates the ranking should be visible, not discovered
by staring at scores that don't line up with similarity -- so each hit
also carries `boosts`, keyed by the boosted property (a callable
Boost, having no property name, falls back to its own `boost_j`
slot):

    hits = graph.vector_search(Near("summary", q), k=10,
                               boost=Boost("importance", 0.2))
    hits[0]["boosts"]           # {"importance": 0.18}

Like `similarities`, `boosts` is always a dict -- empty when no
boost= was given -- so a caller never has to check whether one was
passed before reading it.

EDGE SIMILARITY is the `via` of ranking: Hop(via_near=...,
via_keep=N) follows the N most similar EDGES out of each node reached
so far -- a beam per source node, not a global truncation, because
"the N most similar edges" only means something relative to where you
are standing. Without via_keep, a Near's min_similarity decides which
edges are worth walking at all.

VECTORS NEVER TRAVEL THROUGH THE LLM. There is deliberately no `near`
in TRAVERSE_TOOL_SCHEMA and no vector-search tool schema: a
tool-calling model asked for an embedding will invent 1536 plausible
floats, and an invented embedding produces confidently wrong
neighbors. Embedding text is the application's job; the JSON forms
(`"near"` in a traversal spec, vector_search_json()) exist for HTTP
and config callers that hold real vectors. This is the one place a
parser accepts more than the tool schema advertises, and it is pinned
by a test.

`migrate_`, not `enforce_`: this is the one declaration in the
library that changes the TABLES' SHAPE rather than adding a rule to
them, so it does not belong with enforce_schema()'s vocabulary. And
define_vectors() is not purely in-memory the way define_schema() is
-- it attaches the vec_* columns to the shared SQLAlchemy Table
metadata, so create_schema() on ANY handle for these tables emits
them from then on. The database is otherwise untouched until
migrate_vectors().

WRITES: set_vectors() only -- add_nodes/merge rows do not carry
vectors, because embeddings are computed after the entity exists and a
reserved key would silently change the meaning of a property someone
already stores. One transaction per call, like every other write; a
row whose id does not exist fails the whole call by name.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Optional

from sqlalchemy import (
    Column, and_, case, cast, column as sa_column, func, literal, or_, select, text,
    update, values,
)
from sqlalchemy import String as SAString
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION, REAL
from sqlalchemy.exc import IntegrityError, ProgrammingError

from .constraints import ConstraintViolation, _compile_check, _slug, _Target
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
    the migration it implies is applied by migrate_vectors().

    embed:   an embedding client -- anything Embedder accepts, or an
             Embedder itself. Given one, hopai can turn text into
             vectors for this field: set_vectors() takes strings,
             Near(field, text=...) resolves against it, and
             embed_stale() fills the gaps. Without one the field is
             still perfectly usable; you supply the floats.

    source:  which PROPERTY holds the text to embed, defaulting to the
             field's own name. `Vector("title", 768, embed=...)` reads
             each row's "title" property, which is what the name says
             and used to be untrue -- the field name and the property
             were unrelated. Point it elsewhere when they differ:
             `Vector("summary", 768, source="abstract", embed=...)`.
    """
    name: str
    dimensions: int
    embed: Any = None
    source: Optional[str] = None

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
        if self.source is None:
            self.source = self.name
        elif not isinstance(self.source, str) or not self.source:
            raise ValueError(
                f"Vector({self.name!r}): source= is the PROPERTY name holding the text to "
                f"embed, got {self.source!r}"
            )
        if self.embed is not None:
            from .embeddings import as_embedder
            self.embed = as_embedder(self.embed)
            # An Embedder built with its own dimensions= and pointed at a
            # field of another size is a disagreement the caller can only
            # lose: whichever wins, half their configuration was ignored.
            # One Embedder with dimensions=None across several fields of
            # different sizes stays legal -- each field checks its own.
            if self.embed.dimensions is not None \
                    and self.embed.dimensions != self.dimensions:
                raise ValueError(
                    f"Vector({self.name!r}, {self.dimensions}): its embedder is built with "
                    f"dimensions={self.embed.dimensions} -- they must agree. Drop one of "
                    f"the two numbers"
                )

    @property
    def column_name(self) -> str:
        return f"{VECTOR_COLUMN_PREFIX}{self.name}"


class Near:
    """A similarity spec: one field, one query -- as a vector, or as
    text the field's own embedder turns into one.

        Near("summary", [0.1, 0.4, ...])       # you embedded it
        Near("summary", "raft consensus")      # the field's embed= will
        Near("summary", text="raft consensus")  # the same, said explicitly

    The second argument takes either, because the two can never be
    confused for one another: a str is text, a sequence of numbers is a
    vector, and nothing is both. `text=` stays for the caller who wants
    to be explicit and for the JSON form, where the distinction is a
    key rather than a type.

    Where it appears decides what it does -- rank candidates in
    vector_search(), pick the top-k seed set on Start, prune to the
    top-k reached nodes on a Hop. How many to keep (`k`) always
    belongs to the surrounding call/Start/Hop, never to Near itself:
    one Near per field, one k per ranked set.

    NEVER inside where=/via=: those are boolean filters and a Near
    ranks, so passing one there raises with the rewrite named. It
    looks like GT/BETWEEN, which is exactly why the guard exists.

    query:           the vector to rank against, or the text to embed
                     into one. Text is embedded as a QUERY rather than
                     as a document -- several providers score the two
                     differently and getting it wrong quietly costs
                     recall. Resolved when the query is built, because
                     only then is the graph (and so the field's
                     embedder) known.

    text:            the same thing, said explicitly. Use it when the
                     string might otherwise be mistaken for something
                     to parse -- it is the only way to embed a string
                     that looks like a serialized vector.

    weight:          this field's coefficient in the combined score
                     (only meaningful when several Near are combined).

    min_similarity:  drop rows whose similarity ON THIS FIELD is below
                     the bound -- a filter, applied before k, but AFTER
                     every candidate has already been scanned and
                     scored (`WHERE ... sim_0 >= min_similarity` runs
                     on the LATERAL's output). It shrinks the RESULT,
                     not the work done to get there -- unlike an ANN
                     index's search radius, this never skips a
                     candidate. where= is what actually cuts cost, by
                     removing rows before they reach the LATERAL; see
                     the module docstring's cost model.

    missing:         "exclude" (default) drops rows lacking this
                     field's vector; "zero" scores them 0 here and
                     lets other fields carry the row.
    """

    #: Marker for filters.resolve(), which must refuse a Near in
    #: where=/via= by name without importing this module.
    _is_near = True

    def __init__(self, field: str, query=None, weight: float = 1.0,
                 min_similarity: Optional[float] = None, missing: str = "exclude",
                 text: Optional[str] = None):
        if not isinstance(field, str) or not field:
            raise TypeError(f"Near field must be a vector field name, got {field!r}")
        self.field = field
        if (query is None) == (text is None):
            raise TypeError(
                f"Near({field!r}) takes a query vector or text to embed, not "
                f"{'both' if text is not None else 'neither'}"
            )
        if text is None and isinstance(query, str):
            # A string in the query slot is text -- except when it is a
            # vector someone forgot to parse. `"[0.1, 0.2]"` would embed
            # the LITERAL BRACKETS and rank against whatever that means,
            # which is a confidently wrong answer rather than an error.
            # Refused by name; text= is the way to embed such a string
            # on purpose.
            # startswith/endswith rather than slicing into a membership
            # test: `"" in "[("` is True, so a blank string took this
            # branch and was refused as a serialized vector.
            stripped = query.strip()
            if stripped.startswith(("[", "(")) and stripped.endswith(("]", ")")):
                raise ValueError(
                    f"Near({field!r}): {query[:40]!r} looks like a serialized vector, not "
                    f"text to embed -- parse it into a list of numbers first, or pass "
                    f"text= if you really mean to embed those characters"
                )
            text, query = query, None
        if text is not None:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Near({field!r}): the text to embed must be a non-empty string, "
                    f"got {text!r}"
                )
            # Deferred on purpose: the embedder belongs to the field,
            # which belongs to the graph, which a Near does not know.
            # validate_nears() resolves it and hands back a Near with a
            # vector -- this one is never mutated, so the same spec can
            # be reused across graphs.
            self.text = text
            self.vector = ()
        else:
            self.text = None
            self.vector = _clean_vector(query, f"Near({field!r})")
            if _norm(self.vector) == 0.0:
                raise ValueError(
                    f"Near({field!r}): the query vector is all zeros, which has no "
                    f"direction -- cosine similarity to it is undefined for every row"
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
        parts = [repr(self.field),
                 f"text={self.text!r}" if self.text is not None else f"{len(self.vector)} dims"]
        if self.weight != 1.0:
            parts.append(f"weight={self.weight!r}")
        if self.min_similarity is not None:
            parts.append(f"min_similarity={self.min_similarity!r}")
        if self.missing != "exclude":
            parts.append(f"missing={self.missing!r}")
        return f"Near({', '.join(parts)})"

    def _with_vector(self, vector) -> Near:
        """This spec with text= resolved to floats. A new Near rather
        than a mutation: caching the vector on the original would make
        one reused spec answer with a stale embedding, and silently
        share it across graphs whose fields embed differently."""
        return Near(self.field, vector, weight=self.weight,
                    min_similarity=self.min_similarity, missing=self.missing)


class Boost:
    """A non-similarity term in a ranked score: hybrid retrieval.

        graph.vector_search(Near("summary", q),
                            boost=Boost("importance", 0.2), k=10)

    The score becomes `sum(weight_i * similarity_i) + sum(weight_j *
    boost_j)`, and that sum is what comes back as `similarity` -- so a
    boosted result's score is no longer a cosine and can exceed 1. A
    boost cannot lift a row past a min_similarity floor (those read
    each field's own similarity), but it reorders, so with `k` it
    changes which rows you get. A boost reads a NUMERIC property; a row where the
    property is absent, null, or non-numeric contributes `default`
    (0.0) rather than dropping out -- a boost is a nudge, not a
    filter. Use `where=` to filter. (`default`, not `missing`: Near's
    `missing=` picks a MODE, this picks a VALUE, and one word for two
    kinds of thing is how you get a caller passing "zero" here.)

    NORMALIZED BY DEFAULT: a cosine similarity lives in [-1, 1], and a
    raw property does not -- `Boost("importance", 0.2)` on view counts
    in the thousands does not nudge the ranking, `0.2 * 4200` overwhelms
    a similarity that never exceeds 1 and the "boost" becomes the whole
    sort with the cosine reduced to rounding-error noise. So by default
    each boost is rescaled into similarity's own range with min-max
    normalization computed OVER THE CANDIDATE SET -- the rows still in
    play after `where=` and the graph scope, the same set the similarity
    itself is scored against:

        (value - min(value) OVER ()) / nullif(max(value) OVER () - min(value) OVER (), 0)

    That is a window function, not a guess at your data's distribution:
    it reads the actual spread of THIS query's candidates, so
    `Boost("importance", 0.2)` really does mean "20% weight" against a
    same-scale similarity, whatever range `importance` happens to hold.
    When every candidate's value is identical there is no spread to
    normalize against (the denominator is 0) -- the normalized term
    contributes NOTHING rather than divide-by-zero into NULL, since a
    property with no variance across the candidates carries no ranking
    signal to add. Recomputed per query, because "the candidate set"
    changes with every `where=`.

    `scale="raw"` is the escape hatch, and it is the FULL previous
    behavior, unchanged: the coefficient multiplies the property as
    stored, unbounded, and the caller owns scaling it (into [0, 1] by
    hand, or with a callable that does it in SQL). Reach for it when you
    already store a normalized value, or want to avoid the window
    function's cost, or when the default's per-query rescaling itself is
    the thing you do not want (it makes a boosted `similarity` above 1
    depend on what else was in the candidate set, not just on this row)
    -- see `benchmarks/README.md` for the measured cost of the default
    against `raw`.

    A boost cannot lift a row past a min_similarity floor (those read
    each field's own similarity), but it reorders, so with `k` it
    changes which rows you get. A boost reads a NUMERIC property; a row where the
    property is absent, null, or non-numeric contributes `default`
    (0.0) rather than dropping out -- a boost is a nudge, not a
    filter. Use `where=` to filter. (`default`, not `missing`: Near's
    `missing=` picks a MODE, this picks a VALUE, and one word for two
    kinds of thing is how you get a caller passing "zero" here.)

    The callable form is the same escape hatch the filter DSL has: it
    receives the real `properties` column and returns any numeric
    SQLAlchemy expression -- it too is normalized by default, since the
    library cannot tell from the expression alone whether it is already
    scaled.

        Boost(lambda p: func.ln(1 + cast(p["views"].astext, Float)), 0.1)
    """

    _SCALES = ("normalized", "raw")

    def __init__(self, key, weight: float = 1.0, default: float = 0.0,
                 scale: str = "normalized"):
        if not (isinstance(key, str) and key) and not callable(key):
            raise TypeError(
                f"Boost takes a property name or a callable receiving the properties "
                f"column, got {key!r}"
            )
        self.key = key
        for name, value in (("weight", weight), ("default", default)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(value):
                raise ValueError(f"Boost {name} must be a finite number, got {value!r}")
        if weight == 0:
            raise ValueError(
                "Boost weight must be non-zero -- a zero weight contributes nothing; "
                "drop the Boost instead"
            )
        if scale not in self._SCALES:
            raise ValueError(
                f"Boost scale must be one of {self._SCALES}, got {scale!r} -- "
                f"'normalized' rescales the property into [0, 1] "
                f"over the candidate set (the new default), 'raw' is today's unbounded "
                f"behavior"
            )
        self.weight, self.default, self.scale = float(weight), float(default), scale

    def __repr__(self) -> str:
        key = self.key if isinstance(self.key, str) else "<callable>"
        parts = [repr(key) if isinstance(self.key, str) else key, f"weight={self.weight!r}"]
        if self.default != 0.0:
            parts.append(f"default={self.default!r}")
        if self.scale != "normalized":
            parts.append(f"scale={self.scale!r}")
        return f"Boost({', '.join(parts)})"


def validate_boosts(boost, caller: str) -> list:
    """Normalize boost= (one Boost, a list, or None) into a list."""
    if boost is None:
        return []
    boosts = list(boost) if isinstance(boost, (list, tuple)) else [boost]
    if not boosts:
        raise ValueError(f"{caller}: boost=[] is empty -- pass a Boost(...) or drop the argument")
    for one in boosts:
        if not isinstance(one, Boost):
            raise TypeError(f"{caller}: boost= takes Boost(key, weight) specs, got {one!r}")
    return boosts


def _boost_columns(table, boosts: list) -> list:
    """One labeled term per boost, over the row's properties.

    The jsonb_typeof guard is the same judgement aggregates.py makes:
    a property carrying "high" where a number was expected reads as
    absent instead of aborting the statement.

    scale="normalized" (the default) wraps the coalesced value in a
    min-max window function -- computed HERE, at the same select level
    that lists these columns, because a window function's OVER () sees
    exactly the rows this SELECT is producing: the candidates that
    survived where=/via= and the graph scope, never the whole table.
    It normalizes the COALESCED value, not the raw nullable one, so a
    missing property's `default` takes part in the min/max the same way
    it takes part in the sum in _combined() -- normalizing the raw
    value first and coalescing a default into an already-[0,1] range
    would let one absent property silently outrank real data at either
    end of it.
    See Boost's docstring for the SQL this renders and why it is the
    default; benchmarks/README.md has the measured cost."""
    columns = []
    for i, boost in enumerate(boosts):
        if callable(boost.key):
            value = boost.key(table.c.properties)
        else:
            value = case((
                func.jsonb_typeof(table.c.properties[boost.key]) == "number",
                cast(table.c.properties[boost.key].astext, DOUBLE_PRECISION),
            ))
        # The type_ is documentation, not machinery, and measured so:
        # Boost coerces `default` with float() at construction, so
        # SQLAlchemy infers Float from the value and the compiled SQL and
        # bound parameter types are identical without it. Kept because it
        # says what the column holds; recorded so the next mutation run,
        # which flags dropping it, does not re-derive that it is inert.
        coalesced = func.coalesce(value, literal(boost.default, type_=DOUBLE_PRECISION))
        if boost.scale == "normalized":
            lo = func.min(coalesced).over()
            hi = func.max(coalesced).over()
            # nullif(hi - lo, 0): every candidate shares one value, so
            # there is no spread to normalize against. Left as a bare
            # division the NULL denominator would make the WHOLE term
            # NULL, and _combined() adds boosts straight into the total
            # -- one boost with no variance would then zero out a row's
            # SIMILARITY too, breaking "combined IS NOT NULL means some
            # similarity had a direction" for every row whenever a boost
            # happened to be constant across the candidates. coalesce(.,
            # 0) is what keeps a no-signal boost a no-op instead.
            # type_ on nullif() is load-bearing here, not documentation
            # (contrast the comment above): left to infer, SQLAlchemy
            # types it NUMERIC, and a float8 numerator over a numeric
            # denominator is numeric division -- slower, and a different
            # arithmetic than the float8 this expression stays in
            # otherwise. Same fix _similarity()'s own nullif() needed.
            span = func.nullif(hi - lo, literal(0.0, type_=DOUBLE_PRECISION),
                               type_=DOUBLE_PRECISION)
            normalized = func.coalesce(
                (coalesced - lo) / span, literal(0.0, type_=DOUBLE_PRECISION)
            )
            columns.append(normalized.label(f"boost_{i}"))
        else:
            columns.append(coalesced.label(f"boost_{i}"))
    return columns


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


def parse_boost(spec: Any):
    """The JSON form of Boost: {"property": "score", "weight": 0.2}
    plus the optional "default"/"scale", or a list of such objects. No
    callable form here -- JSON cannot carry one, and a string that
    became SQL would be an injection, not a feature."""
    if isinstance(spec, list):
        if not spec:
            raise ValueError('"boost" is an empty list -- give at least one {"property"}')
        return [parse_boost(one) for one in spec]
    if not isinstance(spec, dict):
        raise TypeError(f'"boost" must be an object or a list of objects -- '
                        f'got {type(spec).__name__}')
    unknown = set(spec) - {"property", "weight", "default", "scale"}
    if unknown:
        raise ValueError(f'unknown "boost" keys {sorted(unknown)} -- a boost spec has '
                         f'"property" and optionally weight, default, scale')
    if "property" not in spec:
        raise ValueError('a boost spec needs "property" -- e.g. {"property": "score"}')
    return Boost(spec["property"], weight=spec.get("weight", 1.0),
                 default=spec.get("default", 0.0), scale=spec.get("scale", "normalized"))


def parse_near(spec: Any):
    """The JSON form of Near, mirroring parse_filter()/parse_aggregate():
    an object {"field": ..., "text": "..."} -- or "vector": [...] if you
    hold the floats -- plus the optional keys weight / min_similarity /
    missing, or a list of such objects."""
    if isinstance(spec, list):
        if not spec:
            raise ValueError('"near" is an empty list -- give at least one {"field", "text"}')
        return [parse_near(one) for one in spec]
    if not isinstance(spec, dict):
        raise TypeError(f'"near" must be an object or a list of objects -- '
                        f'got {type(spec).__name__}')
    unknown = set(spec) - {"field", "vector", "text", "weight", "min_similarity", "missing"}
    if unknown:
        raise ValueError(f'unknown "near" keys {sorted(unknown)} -- a near spec has "field", '
                         f'"text" or "vector", and optionally weight, min_similarity, missing')
    if "field" not in spec:
        raise ValueError('a near spec needs "field" -- '
                         'e.g. {"field": "summary", "text": "how do nodes agree?"}')
    if ("vector" in spec) == ("text" in spec):
        raise ValueError(
            f'a near spec needs "text" to embed OR "vector" if you already have the floats, '
            f'not {"both" if "text" in spec else "neither"} -- '
            f'e.g. {{"field": {spec["field"]!r}, "text": "how do nodes agree?"}}'
        )
    return Near(spec["field"], spec.get("vector"), text=spec.get("text"),
                weight=spec.get("weight", 1.0),
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


def field_names(vectors: Optional[dict], target: Optional[str] = None) -> list:
    """Sorted declared vector field names -- the one place that answers
    "what can be searched here", so a tool schema's enum can never drift
    from the registry it describes.

    `vectors` is the {"nodes": {name: Vector}, "edges": {...}} shape
    build_registry()/Graph.vectors produces; None (define_vectors()
    never called) reads like an empty registry rather than raising --
    every caller here is building an OPTIONAL enum, not requiring the
    feature.

    `target` narrows to "nodes" or "edges"; omitted, it is the union of
    both. That union is what a schema whose `target` argument is picked
    by the CALLER, not fixed at schema-build time, has to enumerate --
    VECTOR_SEARCH_TOOL_SCHEMA's one `near` $def is shared by both, and
    mcp.py's `search_similar` takes the same union when it cannot yet
    tell which side a call will search.

    Shared by Graph.tool_schemas() (this graph's own registry) and
    hopai.mcp's Served.vector_fields() (unioned across every graph one
    server serves) -- CLAUDE.md's "one place that answers this
    question," restated for code instead of prose."""
    if vectors is None:
        return []
    if target is not None:
        return sorted(vectors.get(target) or {})
    names: set = set()
    for fields in vectors.values():
        names.update(fields)
    return sorted(names)


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
        # nullable=True is SQLAlchemy's own default for a non-primary-key
        # column, so stating it is unobservable -- an equivalent mutant
        # when dropped. It is stated anyway because it is load-bearing
        # for the reader: vectors are written only by set_vectors()
        # UPDATEing rows that already exist, so NOT NULL here would make
        # every insert impossible.
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
            f"in_graph() starts lazy -- a search or load_vectors() recovers it from the "
            f"database, but this call didn't go through either)"
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


def _group_query_texts(parsed: list) -> dict:
    """{field name: [text, ...]} across a batch of queries -- the
    grouping half of _resolve_query_texts()/_aresolve_query_texts(),
    shared because it is pure (no provider call, so no sync/async
    split to make)."""
    waiting: dict = {}
    for one in parsed:
        for near in one:
            if isinstance(near, Near) and near.text is not None:
                waiting.setdefault(near.field, []).append(near.text)
    return waiting


def _apply_resolved_texts(parsed: list, ready: dict) -> list:
    """parsed, with every Near(text=) swapped for the vector its field's
    embedder returned -- the reassembly half shared by the sync/async
    resolvers below."""
    return [[near._with_vector(next(ready[near.field]))
             if isinstance(near, Near) and near.text is not None else near
             for near in one]
            for one in parsed]


def _resolve_query_texts(graph, target: str, parsed: list, caller: str) -> list:
    """Every Near(text=) across a batch of queries, embedded in one
    call per field. validate_nears() resolves one query's texts on its
    own; this exists because vector_search_many() has N of them, and N
    round trips would undo the round trip it saves."""
    waiting = _group_query_texts(parsed)
    if not waiting:
        return parsed
    ready = {}
    for name, texts in waiting.items():
        field = _field(graph, target, name, caller)
        ready[name] = iter(_embedder(field, target, caller).embed_queries(texts))
    return _apply_resolved_texts(parsed, ready)


async def _aresolve_query_texts(graph, target: str, parsed: list, caller: str) -> list:
    """Async twin of _resolve_query_texts() -- see hopai/asyncio.py's
    module docstring and issue #74. Used by AsyncGraph.vector_search()/
    vector_search_many() to embed every Near(text=) BEFORE the query
    reaches conn.run_sync(): validate_nears()/search() then see
    near.text is None and make no provider call of their own, so the
    embedding round trip never runs inside the greenlet bridge and
    blocks the event loop."""
    waiting = _group_query_texts(parsed)
    if not waiting:
        return parsed
    names = list(waiting)
    embedders = [_embedder(_field(graph, target, name, caller), target, caller) for name in names]
    # gather, not one await per field in a loop: a query ranking two or
    # three fields at once should cost the SLOWEST provider round trip,
    # not their sum.
    answers = await asyncio.gather(*(
        embedder.aembed_queries(waiting[name])
        for embedder, name in zip(embedders, names, strict=True)))
    ready = dict(zip(names, (iter(answer) for answer in answers), strict=True))
    return _apply_resolved_texts(parsed, ready)


async def aresolve_near(graph, target: str, nears: list, k, caller: str) -> list:
    """Async pre-resolution for ONE vector_search() call's near= --
    always a list here, matching how AsyncGraph.vector_search() already
    normalizes *near before this runs. Mirrors _prepare_search_query()'s
    own order -- k, then shape, and only THEN a provider call -- so a
    call about to be refused for either reason is never embedded first
    (issue #74's own review). See _aresolve_query_texts()."""
    _check_k(k, caller)
    nears = _normalized_nears(nears, caller)
    return (await _aresolve_query_texts(graph, target, [nears], caller))[0]


async def aresolve_queries(graph, target: str, queries: list, k, caller: str) -> list:
    """Async pre-resolution for vector_search_many()'s queries= -- every
    entry embedded in one batched call per field, same as the sync path
    inside _prepare_search_many_query(). k is checked first, matching
    that function's own order. Per-query Near shape is NOT re-checked
    here before embedding -- _prepare_search_many_query() itself embeds
    every query's text (_resolve_query_texts()) before running
    validate_nears() on any of them, so this is not a new ordering,
    only its async twin. The result is already the list-of-list-of-Near
    shape search_many() accepts, so AsyncGraph hands it straight to the
    sync call with no further reshaping."""
    _check_k(k, caller)
    return await _aresolve_query_texts(graph, target, _as_query_list(queries, caller), caller)


def _resolved_spec(owner, **changes):
    """dataclasses.replace(), with `rerank` re-attached AFTER the copy
    has been validated -- the one thing aresolve_spec_texts() below
    cannot do with a plain replace().

    Start/Hop validate on construction, and one of those rules is that
    a rerank= needs the query as TEXT (hop.py's _validate_rerank) --
    which is precisely what this copy has just resolved away, since
    Near._with_vector() hands back floats and no text. Rebuilding
    without the rerank and putting it back keeps that rule where it
    belongs, on the line the caller wrote, instead of the async path
    re-deciding it against a spec it resolved itself. The original spec
    has already passed it."""
    if owner.rerank is None:
        return replace(owner, **changes)
    rebuilt = replace(owner, **changes, rerank=None)
    rebuilt.rerank = owner.rerank
    return rebuilt


async def aresolve_spec_texts(graph, start, hops: list) -> tuple:
    """Every Near(text=) in a Start/Hop chain -- start.near, each hop's
    near and via_near -- embedded before AsyncGraph.traverse()/
    aggregate() enter the greenlet bridge. Returns a NEW (Start, [Hop])
    with each resolved Near swapped in via Near._with_vector(); the
    sync build_query()/build_aggregate_query() path is completely
    unchanged downstream of that -- validate_nears() there sees
    near.text is None and skips the provider call it would otherwise
    make. See hopai/asyncio.py's module docstring and issue #74.

    Field/embedder lookups use the SAME per-hop caller label
    core.py's _walk_matches() does ("Start", "hop 2 (label) via_near",
    ...), so a bad field name or a field with no embedder still names
    the right hop.

    EVERY near=/via_near= in the chain is shape-checked -- non-empty,
    every entry a Near, no two ranking the same field -- with
    _normalized_nears(), BEFORE any of them is embedded: a chain about
    to be refused for one of these reasons must never pay for, or fail
    because of, a provider round trip first (issue #74's own review;
    AsyncGraph.traverse()/aggregate() run the hop-position/aggregate-
    spec checks earlier still, before this function is even called).
    Everything validate_nears() checks AFTER shape -- dimensions,
    missing='zero' with one Near, k-required -- still runs once, in the
    sync path, unchanged: those already run after embedding on the sync
    side too, so moving the embed call earlier changes nothing about
    when they fire relative to it.

    Batched by (target, field) across the WHOLE chain, not per hop: a
    multi-hop walk that embeds the same field's text more than once
    costs one provider round trip for it, not one per occurrence."""
    def normalized(near, caller):
        return None if near is None else _normalized_nears(near, caller)

    start_norm = normalized(start.near, "Start")
    hop_norms = []
    for i, hop in enumerate(hops):
        label = hop.label or "unlabeled"
        hop_norms.append((
            normalized(hop.near, f"hop {i} ({label})"),
            normalized(hop.via_near, f"hop {i} ({label}) via_near"),
        ))

    # Every position's shape check already ran above, in full, before
    # any of what follows -- which only collects and embeds TEXT.
    waiting: dict = {}

    def collect(target, caller, items):
        if items is None:
            return
        for one in items:
            if isinstance(one, Near) and one.text is not None:
                key = (target, one.field)
                if key not in waiting:
                    field = _field(graph, target, one.field, caller)
                    waiting[key] = (caller, field, [])
                waiting[key][2].append(one.text)

    collect("nodes", "Start", start_norm)
    for i, (near_norm, via_norm) in enumerate(hop_norms):
        label = hops[i].label or "unlabeled"
        collect("edges", f"hop {i} ({label}) via_near", via_norm)
        collect("nodes", f"hop {i} ({label})", near_norm)

    if not waiting:
        return start, hops

    keys = list(waiting)
    # gather, not one await per field in a loop: a chain ranking several
    # DIFFERENT fields by text should cost the slowest provider round
    # trip, not their sum -- the same reasoning as _aresolve_query_texts().
    answers = await asyncio.gather(*(
        _embedder(field, key[0], caller).aembed_queries(texts)
        for key, (caller, field, texts) in ((k, waiting[k]) for k in keys)))
    ready = dict(zip(keys, (iter(answer) for answer in answers), strict=True))

    def has_text(items):
        return items is not None and any(
            isinstance(one, Near) and one.text is not None for one in items)

    def resolve(target, original, items):
        if items is None:
            return None
        out = [one._with_vector(next(ready[(target, one.field)]))
               if isinstance(one, Near) and one.text is not None else one
               for one in items]
        return out if isinstance(original, (list, tuple)) else out[0]

    # _resolved_spec(), not replace(): a step carrying rerank= would be
    # refused by its own constructor for the text this copy has just
    # resolved away. See that function.
    new_start = (_resolved_spec(start, near=resolve("nodes", start.near, start_norm))
                 if has_text(start_norm) else start)
    new_hops = []
    for i, hop in enumerate(hops):
        near_norm, via_norm = hop_norms[i]
        if has_text(near_norm) or has_text(via_norm):
            new_hops.append(_resolved_spec(
                hop,
                near=resolve("nodes", hop.near, near_norm),
                via_near=resolve("edges", hop.via_near, via_norm),
            ))
        else:
            new_hops.append(hop)
    return new_start, new_hops


def _embedder(field: Vector, target: str, caller: str):
    """The field's embedder, or a refusal naming the declaration to
    change. Every text-to-vector path lands here, so "this field takes
    floats" is said once and in the same words."""
    if field.embed is None:
        raise ValueError(
            f"{caller}: field {field.name!r} on {target} was given text, but declares no "
            f"embedder -- redeclare it as Vector({field.name!r}, {field.dimensions}, "
            f"embed=<your embedding client>), or pass a {field.dimensions}-dimensional "
            f"vector here"
        )
    return field.embed


@contextmanager
def _read_connection(graph, connection=None):
    """A connection for a read-only vector query: the caller's own, when
    one is passed in (the seam AsyncGraph's greenlet bridge needs -- see
    hopai/asyncio.py), or a fresh one otherwise. Mirrors ingest.py's
    one_transaction(), minus the transaction: nothing here writes."""
    if connection is not None:
        yield connection
    else:
        with graph.engine.connect() as opened:
            yield opened


def _normalized_nears(near, caller: str) -> list:
    """near= (one Near or a list) normalized into a list, with every
    check that needs no field lookup and no provider call already done:
    non-empty, every entry a Near, no two ranking the same field.

    Split out of validate_nears() so it can run FIRST there -- and, for
    AsyncGraph, be run again standalone before any embedding starts
    (aresolve_spec_texts()/aresolve_near() in this module): a spec about
    to be refused for one of these reasons must never be embedded
    first, on either path (issue #74's own review)."""
    nears = list(near) if isinstance(near, (list, tuple)) else [near]
    if not nears:
        raise ValueError(f"{caller}: near=[] is empty -- pass a Near(...) or a list of them")
    seen = set()
    for one in nears:
        if not isinstance(one, Near):
            raise TypeError(
                f"{caller}: near= takes Near(field, vector) specs, got {one!r}"
            )
        if one.field in seen:
            # Two Nears on ONE field blend into a single score, so a row
            # similar to neither query can outrank a row identical to
            # one of them. That is what query expansion looks like when
            # it is written as one search by mistake, and it comes back
            # confidently wrong rather than empty.
            raise ValueError(
                f"{caller}: two Near specs both rank field {one.field!r}, which blends them "
                f"into ONE score -- a row similar to neither query can outrank a row "
                f"identical to one. If these are separate queries, use "
                f"vector_search_many([...]); if you meant one query, average the vectors "
                f"yourself so the blend is visible in your code"
            )
        seen.add(one.field)
    return nears


def validate_nears(graph, target: str, near, k, caller: str,
                   limit_name: str = "k") -> list:
    """Normalize near= (one Near or a list) into a validated list:
    every entry a Near, every field defined for `target`, every query
    vector of the declared dimensions -- so a typo fails here with the
    fix named, not at execution as an undefined-column error.

    Any Near carrying text= is resolved here, against the embedder its
    FIELD declares: this is the first point where the spec and the
    graph are both in hand. The returned list is a new one, so the
    caller's specs are left as they were written."""
    # Shape first, in full, and only then anything that costs a provider
    # call: a batch that is about to be refused should never be embedded.
    nears = _normalized_nears(near, caller)

    resolved = []
    for one in nears:
        field = _field(graph, target, one.field, caller)
        if one.text is not None:
            one = one._with_vector(
                _embedder(field, target, caller).embed_query(one.text))
        if len(one.vector) != field.dimensions:
            raise ValueError(
                f"{caller}: the query vector for {one.field!r} has {len(one.vector)} "
                f"dimensions, the field is defined with {field.dimensions}"
            )
        resolved.append(one)
    nears = resolved
    if len(nears) == 1 and nears[0].missing == "zero":
        # With no other field to carry the row, "zero" is provably a
        # no-op: the guards already require this vector to have a
        # direction, so coalesce(s, 0) only ever sees a non-NULL s.
        # Same class as boost= without near= -- a knob that reads as a
        # working feature and changes nothing.
        raise ValueError(
            f"{caller}: missing='zero' lets OTHER fields carry a row this field is missing, "
            f"and there is no other field here -- with one Near it changes nothing. Drop it, "
            f"or add the Near it is meant to defer to"
        )
    if k is None and all(one.min_similarity is None for one in nears):
        raise ValueError(
            f"{caller}: near= without {limit_name}= and without any min_similarity changes "
            f"nothing -- ranking with no limit and no bound keeps every row. Pass "
            f"{limit_name}=<how many to keep>, or min_similarity on a Near, or drop near="
        )
    return nears


# ---------------------------------------------------------------------
# Similarity as SQL
# ---------------------------------------------------------------------

def _similarity(table, near: Near, index: int, query=None):
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
    product and the STORED vector's norm are computed per row -- or,
    when `query` is given, both come from a correlated VALUES row so
    that one statement can rank many queries (see search_many). The
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
    if query is None:
        # The single-query case: the vector and its norm are Python
        # constants, so the norm is computed once here rather than per
        # row, and the length check compares against a plain integer.
        query_expr = literal(list(near.vector), type_=ARRAY(REAL))
        norm_expr = literal(_norm(near.vector), type_=DOUBLE_PRECISION)
        comparable = func.array_length(column, 1) == len(near.vector)
    else:
        # The batch case: both come from the VALUES row this LATERAL is
        # correlated with, so one statement can rank many queries.
        query_expr, norm_expr = query
        comparable = func.array_length(column, 1) == func.array_length(query_expr, 1)
    zipped = func.unnest(
        column, query_expr
    ).table_valued("x", "y").render_derived()
    x = cast(zipped.c.x, DOUBLE_PRECISION)
    y = cast(zipped.c.y, DOUBLE_PRECISION)
    stored_norm = func.sqrt(func.sum(x * x))
    # type_ is not decoration: without it SQLAlchemy types nullif() as
    # NUMERIC, and a float8 numerator over a numeric denominator is
    # numeric division -- slower, and a different arithmetic than the
    # float8 the rest of this expression is careful to stay in.
    denominator = func.nullif(
        stored_norm * norm_expr,
        literal(0.0, type_=DOUBLE_PRECISION),
        type_=DOUBLE_PRECISION,
    )
    # No else_: a CASE with no ELSE is NULL, which is the "missing"
    # every caller already reads. Spelling it literal(None) makes
    # SQLAlchemy warn about rendering NULL as a bound parameter.
    # correlate_except: everything this body names EXCEPT the unnest
    # comes from the enclosing query. Without it the batch form drags a
    # second copy of the queries VALUES list into this FROM clause and
    # cross-joins it -- every row scored against every query at once,
    # which is not slower, it is wrong.
    body = select(
        case((comparable, func.sum(x * y) / denominator)).label("s")
    ).select_from(zipped).correlate_except(zipped)
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


def _similarity_terms(table, nears: list, query=None) -> tuple:
    """(laterals, labeled columns, candidate guards) for one query.

    Each field is computed once, in a LATERAL; `sim_i` is that value,
    coalesced to 0 for missing="zero" so a threshold and the combined
    score read the same number for the same row."""
    laterals, columns, guards, has_direction = [], [], [], []
    for i, near in enumerate(nears):
        lateral = _similarity(table, near, i, query=None if query is None else query[i])
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


def _combined(inner, nears: list, boosts: list = ()):
    """The ranked score: weighted similarities, plus weighted boosts.

    Boosts are added AFTER the similarity terms and never NULL --
    `_boost_columns()` coalesces a missing property to `default`, and
    (scale="normalized") coalesces a zero-spread candidate set to 0 on
    top of that -- which is what keeps `combined IS NOT NULL` meaning
    "some similarity had a direction". It does NOT mean a boost is free
    of consequence: it reorders, and with a `k` limit reordering
    decides membership."""
    total = None
    for i, near in enumerate(nears):
        term = inner.c[f"sim_{i}"] if near.weight == 1.0 else inner.c[f"sim_{i}"] * near.weight
        total = term if total is None else total + term
    for j, boost in enumerate(boosts):
        term = inner.c[f"boost_{j}"] if boost.weight == 1.0 \
            else inner.c[f"boost_{j}"] * boost.weight
        total = term if total is None else total + term
    return total


def _thresholds(inner, nears: list) -> list:
    return [inner.c[f"sim_{i}"] >= one.min_similarity
            for i, one in enumerate(nears) if one.min_similarity is not None]


def ranked_ids(graph, table, id_expr, from_obj, condition, nears: list, k: Optional[int],
               boosts: list = ()):
    """A Select of the ids that survive similarity: the shared shape
    behind a near= seed CTE and a near= match CTE. `condition` is the
    caller's full predicate, graph scope included (None when the scope
    already lives in from_obj's join) -- this function only adds the
    similarity layer."""
    conditions = [] if condition is None else [condition]
    laterals, columns, guards = _similarity_terms(table, nears)
    inner = (
        select(id_expr.label("node_id"), *columns, *_boost_columns(table, boosts))
        .select_from(_with_laterals(from_obj, laterals))
        .where(*conditions, *guards)
        .subquery()
    )
    combined = _combined(inner, nears, boosts)
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


def _identity_columns(graph, table, target: str) -> list:
    id_column = getattr(table.c, graph.node_id_col if target == "nodes" else graph.edge_id_col)
    identity = [id_column.label("_id")]
    if target == "edges":
        identity.append(getattr(table.c, graph.edge_start_col).label("_start"))
        identity.append(getattr(table.c, graph.edge_end_col).label("_end"))
    return identity


def _result_columns(inner, target: str) -> list:
    # Ids are cast to text to match the contract traversal results
    # already follow; the ORDER BY tiebreak uses the RAW id, because
    # '10' sorts before '9' as text and determinism should not change
    # the neighbors' order.
    columns = [cast(inner.c._id, SAString).label("id")]
    if target == "edges":
        columns.append(cast(inner.c._start, SAString).label("start_id"))
        columns.append(cast(inner.c._end, SAString).label("end_id"))
    return columns


def edge_beam(graph, edge_alias, join_expr, anchor_expr, move_col, id_col, via,
              nears: list, k: Optional[int], name: str, extra: list = (), correlate=()):
    """The edges to follow from ONE anchor row, ranked by similarity --
    a LATERAL yielding (edge_id, move_id).

    This is what `Hop(via_near=)` compiles to, and the per-anchor
    shape is the point: "follow the k most similar edges" only means
    something relative to a source node. A global top-k over a
    recursive walk would silently starve every node after the first
    few, which is not a beam, it is a truncation.

    With `k=None` the same shape filters by threshold and follows
    everything that passes, so both forms share one code path -- and
    one set of guards, including the wrong-length and no-direction
    rules that apply to edge vectors exactly as they do to nodes."""
    laterals, columns, guards = _similarity_terms(edge_alias, nears)
    inner = (
        select(id_col.label("edge_id"), move_col.label("move_id"), *columns)
        .select_from(_with_laterals(edge_alias, laterals))
        .where(join_expr == anchor_expr, graph._scoped(edge_alias),
               resolve(edge_alias.c.properties, via), *guards, *extra)
        # The anchor (the seed CTE, or the walk's own recursive
        # reference) belongs to the ENCLOSING query. Left to infer,
        # SQLAlchemy copies it into this FROM clause instead -- which
        # cross-joins the seed, and makes Postgres reject the walk
        # outright: "recursive reference to query walk_0 must not
        # appear more than once".
        .correlate(*correlate)
        .subquery()
    )
    combined = _combined(inner, nears)
    beam = (
        select(inner.c.edge_id, inner.c.move_id)
        .where(combined.isnot(None), *_thresholds(inner, nears))
    )
    if k is not None:
        beam = beam.order_by(combined.desc(), inner.c.edge_id).limit(k)
    return beam.lateral(name)


#: Prefix for the per-field similarity report columns build_search_query()
#: and build_search_many_query() add outward, alongside boost_j -- both
#: named distinctly from `sim_i` (which stays the coalesced value
#: _combined()/_thresholds() use) so a mutation on one prefix cannot
#: silently satisfy the other's tests. See _report_columns().
_SIM_REPORT_PREFIX = "sim_report_"


def _report_columns(laterals: list) -> list:
    """Each field's RAW similarity, never coalesced -- read again off the
    same LATERAL `sim_i` already reads, so nothing is computed twice.

    This is what makes `similarities` in the result honest for
    missing="zero": `sim_i` (used for the combined score and
    min_similarity) coalesces a missing field to 0.0 so it does not NULL
    the row's total, but the PER-FIELD report must keep saying "missing"
    -- None, not a score -- or a field with no vector would look
    identical to one that matched poorly (issue #54)."""
    return [lateral.c.s.label(f"{_SIM_REPORT_PREFIX}{i}")
            for i, lateral in enumerate(laterals)]


def _search_result_columns(inner, target: str, similarity, n_near: int, n_boost: int) -> list:
    """The outward SELECT list vector_search() and vector_search_many()
    share: identity + properties + the combined score, plus each near's
    own similarity and each boost's own term so a caller can see what
    actually drove the combined number, not just the sum (issue #54)."""
    return (
        _result_columns(inner, target) + [inner.c.properties, similarity]
        + [inner.c[f"{_SIM_REPORT_PREFIX}{i}"] for i in range(n_near)]
        + [inner.c[f"boost_{j}"] for j in range(n_boost)]
    )


def _format_hit(row, nears: list, boosts: list) -> dict:
    """One search()/search_many() row, reshaped from the flat SQL row
    (id/properties/similarity plus the private sim_report_i/boost_j
    columns) into the public shape: `similarities` keyed by the field
    NAME the caller passed to Near, never by the `sim_i` SQL alias,
    which is an implementation detail (issue #54). Always present, even
    for a single Near, so `row["similarities"][field]` never needs to
    branch on how many Near clauses a search used.

    `boosts` is keyed by Boost.key the same way; a callable boost has no
    property name, so it falls back to its own `boost_j` slot. Two
    Boosts naming the same property is unusual but legal -- the combined
    score already adds both terms, so the report SUMS them under one key
    rather than the second silently overwriting the first, which would
    under-report the exact thing this feature exists to surface."""
    hit = {key: value for key, value in row.items()
           if not key.startswith(_SIM_REPORT_PREFIX) and not key.startswith("boost_")}
    hit["similarity"] = float(hit["similarity"])
    hit["similarities"] = {
        near.field: (None if row[f"{_SIM_REPORT_PREFIX}{i}"] is None
                    else float(row[f"{_SIM_REPORT_PREFIX}{i}"]))
        for i, near in enumerate(nears)
    }
    per_boost: dict = {}
    for j, boost in enumerate(boosts):
        key = boost.key if isinstance(boost.key, str) else f"boost_{j}"
        per_boost[key] = per_boost.get(key, 0.0) + float(row[f"boost_{j}"])
    hit["boosts"] = per_boost
    return hit


def _prepare_search_query(graph, near, target: str, k: Optional[int], where: Any,
                          boost, candidates: Optional[int] = None) -> tuple:
    """(nears, boosts, query) for vector_search()'s single statement --
    build_search_query() is the public, query-only view of this;
    search() needs the resolved nears/boosts too, to key `similarities`/
    `boosts` by name rather than by their SQL column position.

    `candidates` is the ROW LIMIT when a Rerank is in play, and `k` is
    still what everything else is validated against: the two are
    different bounds -- input and output -- and collapsing them would
    make a rerank silently change which refusals a search produces.

    The caller label stays a literal "vector_search()" at each call
    below, not a parameter -- TestVectorCallerNamesArePinned reads
    every caller label straight out of the source AST, and a `caller`
    variable at the call site would make this function's two labels
    invisible to it (a wrapped-and-forwarded label is a bug this
    library refuses to reproduce silently)."""
    _check_k(k, "vector_search()")
    nears = validate_nears(graph, target, near, k, "vector_search()")
    boosts = validate_boosts(boost, "vector_search()")
    table = _table(graph, target)

    laterals, sim_columns, guards = _similarity_terms(table, nears)
    inner = (
        select(*_identity_columns(graph, table, target),
               table.c.properties.label("properties"),
               *sim_columns, *_report_columns(laterals), *_boost_columns(table, boosts))
        .select_from(_with_laterals(table, laterals))
        .where(and_(graph._scoped(table), resolve(table.c.properties, where), *guards))
        .subquery()
    )
    combined = _combined(inner, nears, boosts)
    similarity = combined.label("similarity")
    columns = _search_result_columns(inner, target, similarity, len(nears), len(boosts))
    query = (
        select(*columns)
        .where(combined.isnot(None), *_thresholds(inner, nears))
        .order_by(similarity.desc(), inner.c._id)
    )
    rows = k if candidates is None else candidates
    return nears, boosts, (query if rows is None else query.limit(rows))


def build_search_query(graph, near, target: str = "nodes", k: Optional[int] = 10,
                       where: Any = None, boost=None):
    """The single statement vector_search() runs. Exposed so the SQL
    can be inspected with no database, like build_query()."""
    _, _, query = _prepare_search_query(graph, near, target=target, k=k, where=where, boost=boost)
    return query


#: Postgres's SQLSTATE for "column does not exist" -- what a compiled
#: query referencing a declared-but-never-migrated vec_* column raises
#: as. Comparing this rather than the exception's Python class name
#: keeps _raise_if_unmigrated() driver-agnostic (psycopg2's pgcode and
#: psycopg3's sqlstate both carry it), the same reason embeddings.py
#: never imports a provider package to recognize one.
_UNDEFINED_COLUMN = "42703"


def _ensure_lazy_vectors(graph, connection) -> None:
    """in_graph() marks its handle `_vectors_lazy` instead of starting
    it permanently blank (see Graph.in_graph()): the vec_* columns are
    SHARED storage, so a field another handle already migrated for
    THIS graph is usable here too, without an explicit load_vectors()
    call from the caller. Fires at most meaningfully once -- a handle
    that already has a registry, lazy or explicitly declared, is left
    alone."""
    if graph._vectors is None and getattr(graph, "_vectors_lazy", False):
        load_vectors(graph, connection=connection)


def _near_field_names(near) -> list:
    """Every Near.field in one search's near= argument, in the shape
    it was given (single Near or a list) -- used only to name fields
    on the error path below, so it does not need validate_nears()'s
    normalization."""
    items = near if isinstance(near, (list, tuple)) else [near]
    return [one.field for one in items if isinstance(one, Near)]


def _raise_if_unmigrated(graph, target: str, field_names: list, conn,
                         exc: ProgrammingError, caller: str) -> None:
    """Turn a raw UndefinedColumn on a vec_* column into a refusal
    naming migrate_vectors() -- the one gap validate_nears() cannot
    close by itself. A field can be DECLARED (define_vectors() ran)
    while its column was never added (migrate_vectors() never ran):
    the registry says "declared" either way, so only the catalog
    itself can tell "undeclared" and "declared but not migrated"
    apart. That catalog check runs ONLY here, on the error path a
    genuine query failure already took, so the ordinary search that
    never hits it pays nothing for it -- proactively probing
    information_schema before every search would cost a round trip
    for a case that almost never happens.

    Silently returns (letting the caller re-raise the original
    ProgrammingError unchanged) when the SQLSTATE is not "undefined
    column", or when every named field's column turns out to exist --
    this diagnosis is specific to vec_* columns and must not swallow
    an unrelated undefined-column error."""
    orig = getattr(exc, "orig", None)
    sqlstate = getattr(orig, "pgcode", None) or getattr(orig, "sqlstate", None)
    if sqlstate != _UNDEFINED_COLUMN:
        return
    # Postgres aborts the whole transaction on the failed statement, so
    # this connection refuses every further query ("current transaction
    # is aborted") until it is rolled back -- the diagnostic queries
    # below are read-only and change nothing this rollback could lose.
    conn.rollback()
    table = _table(graph, target)
    missing = sorted(
        name for name in field_names
        if conn.execute(_COLUMN_TYPE, {
            "table": table.name, "column": VECTOR_COLUMN_PREFIX + name, "schema": table.schema,
        }).scalar() is None
    )
    if not missing:
        return
    raise ValueError(
        f"{caller}: vector field(s) {missing} are declared for {target} in this graph but "
        f"never migrated -- call migrate_vectors() to add the column(s). load_vectors() "
        f"reads the same catalog this check just did and would find nothing either, since "
        f"the column genuinely does not exist yet for ANY graph -- it only helps once "
        f"migrate_vectors() has actually run, on some other handle this one forgot to match"
    ) from exc


def search_candidates(graph, near, target: str = "nodes", k: Optional[int] = 10,
                      where: Any = None, boost=None, connection=None, rerank=None) -> list:
    """The rows one search fetches, BEFORE any reranking: `k` of them
    normally, `rerank.candidates` when a Rerank is in play.

    Split out of search() because the provider call must happen with
    nothing open, and on the async path the connection belongs to
    AsyncGraph across a run_sync() bridge that runs on the event loop's
    own THREAD -- so an async caller takes these rows, lets the bridge
    close, and awaits the scoring itself. Same reason set_vectors()
    resolves its embeddings before it takes a connection."""
    with _read_connection(graph, connection) as conn:
        _ensure_lazy_vectors(graph, conn)
        nears, boosts, query = _prepare_search_query(
            graph, near, target=target, k=k, where=where, boost=boost,
            candidates=None if rerank is None else rerank.candidates)
        try:
            rows = conn.execute(query).mappings().all()
        except ProgrammingError as exc:
            _raise_if_unmigrated(graph, target, _near_field_names(near), conn, exc,
                                 "vector_search()")
            raise
    return [_format_hit(row, nears, boosts) for row in rows]


def search(graph, near, target: str = "nodes", k: Optional[int] = 10, where: Any = None,
           boost=None, connection=None, rerank=None, rerank_query: Optional[str] = None) -> list:
    """One search, reranked when a Rerank is given.

    `rerank_query` is the text the reranker scores against. It is
    normally derived from the Near right here, but the async path has
    to derive it BEFORE it embeds -- `Near._with_vector()` hands back a
    spec carrying floats and no text -- so it passes the text it
    already read rather than letting this look for one that is gone."""
    if rerank is not None and rerank_query is None:
        rerank_query = rerank_query_text(near, rerank, k, "vector_search()")
    hits = search_candidates(graph, near, target=target, k=k, where=where, boost=boost,
                             connection=connection, rerank=rerank)
    if rerank is None:
        return hits
    return rerank_hits(hits, rerank, rerank_query, k)


# ---------------------------------------------------------------------
# Batch: many queries, one round trip
# ---------------------------------------------------------------------

def _check_k(k, caller: str) -> None:
    """k=None means "every row that passes the thresholds".

    Start/Hop have always meant that, and vector_search defaulting to
    10 while refusing None made one spelling of "give me everything
    above 0.85" silently return the first ten -- the same Near, the
    same floor, two different answers depending on which call you
    reached for. validate_nears() then enforces the other half of the
    rule everywhere: ranking with neither a limit nor a floor keeps
    every row, so one of them is required."""
    if k is None:
        return
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"{caller}: k must be a positive integer or None, got {k!r}")


def _as_query_list(queries, caller: str) -> list:
    """queries -> [[Near, ...], ...]: one entry per query, each itself
    one Near or a list of them (a multivector query)."""
    if not isinstance(queries, (list, tuple)) or not queries:
        raise TypeError(
            f"{caller}: queries must be a non-empty list, each entry a Near(...) or a list "
            f"of them -- e.g. [Near('summary', q1), Near('summary', q2)]"
        )
    return [list(one) if isinstance(one, (list, tuple)) else [one] for one in queries]


def _prepare_search_many_query(graph, queries, target: str, k: Optional[int], where: Any,
                               boost, candidates: Optional[int] = None) -> tuple:
    """(parsed, template, boosts, query) for vector_search_many()'s
    single statement -- build_search_many_query() is the public,
    query-only view of this; search_many() needs `template`/`boosts`
    too, to key each hit's `similarities`/`boosts` by name.

    The caller label stays a literal "vector_search_many()" at each
    call below, like _prepare_search_query() -- see its docstring for
    why a `caller` parameter would defeat the AST-based pin.

    The queries become a VALUES list -- (index, vector, norm) per
    query -- and the per-query top-k hangs off it as a LATERAL.

    WHAT THIS BUYS, measured rather than assumed: one round trip
    instead of N. It does NOT reduce the arithmetic -- every query
    still scores every candidate, so 8 queries over 4k x 384-dim
    measured 1.08x against a Python loop on a local database, where a
    round trip is nearly free. The win scales with LATENCY, not with
    N: at a realistic 5-20ms each, N-1 saved round trips is the whole
    benefit, and it is worth having for exactly that reason. If the
    arithmetic is what hurts, fewer candidates (`where=`) is the lever,
    not batching.

    Each query keeps its own field set, weights, thresholds and
    missing-modes; only the vectors vary per row. Queries must
    therefore agree on the SHAPE -- same fields in the same order --
    or they would need different SQL, which one statement cannot be.
    """
    _check_k(k, "vector_search_many()")
    parsed = _as_query_list(queries, "vector_search_many()")
    # Every text= across every query, embedded in one call per field --
    # otherwise batching N searches into one statement would still cost
    # N provider round trips, which is the cost this call exists to save.
    parsed = _resolve_query_texts(graph, target, parsed, "vector_search_many()")
    validated = [validate_nears(graph, target, one, k, "vector_search_many()")
                 for one in parsed]
    shapes = {tuple((n.field, n.weight, n.min_similarity, n.missing) for n in one)
              for one in validated}
    if len(shapes) > 1:
        raise ValueError(
            "vector_search_many() ranks every query with ONE statement, so the queries must share "
            "a shape -- the same fields, in the same order, with the same weights, "
            "min_similarity and missing modes. Only the vectors may differ. Group the "
            "queries by shape and call search_many() once per group"
        )
    boosts = validate_boosts(boost, "vector_search_many()")
    template = validated[0]
    table = _table(graph, target)

    # One VALUES row per query per field: the vectors travel as bound
    # parameters, like every other caller value in this library.
    #
    # `q`'s SAString is documentation, not machinery: measured, the
    # compiled SQL is byte-identical without it and the bound parameter
    # types are unchanged, because this column renders as a plain bind
    # with no cast -- unlike v{i}, which renders ::REAL[]. Kept because
    # it says what the column holds; recorded here so the next reader
    # (or the next mutation run, which flags dropping it) does not
    # re-derive that it is inert.
    # n{i}'s DOUBLE_PRECISION is inert for the same reason and measured
    # the same way -- identical compiled SQL, and identical similarities
    # to twelve decimal places against a live database, since the norm
    # arrives as a Python float and Postgres divides in float8 either
    # way. v{i} is the one that matters: it renders ::REAL[].
    columns = [sa_column("q", SAString)]
    for i in range(len(template)):
        columns.append(sa_column(f"v{i}", ARRAY(REAL)))
        columns.append(sa_column(f"n{i}", DOUBLE_PRECISION))
    rows = []
    for index, one in enumerate(validated):
        row = [str(index)]
        for near in one:
            row.append(list(near.vector))
            row.append(_norm(near.vector))
        rows.append(tuple(row))
    queries_values = values(*columns, name="queries").data(rows)

    laterals, sim_columns, guards = _similarity_terms(
        table, template,
        query=[(queries_values.c[f"v{i}"], queries_values.c[f"n{i}"])
               for i in range(len(template))],
    )
    inner = (
        select(*_identity_columns(graph, table, target),
               table.c.properties.label("properties"),
               *sim_columns, *_report_columns(laterals), *_boost_columns(table, boosts))
        .select_from(_with_laterals(table, laterals))
        .where(and_(graph._scoped(table), resolve(table.c.properties, where), *guards))
        .subquery()
    )
    combined = _combined(inner, template, boosts)
    similarity = combined.label("similarity")
    per_query = (
        select(*_search_result_columns(inner, target, similarity, len(template), len(boosts)))
        .where(combined.isnot(None), *_thresholds(inner, template))
        .order_by(similarity.desc(), inner.c._id)
    )
    # `candidates` is PER QUERY, exactly where `k` sits -- inside the
    # LATERAL. One statement still serves every query; each just hands
    # back a wider list for its own reranker call to narrow.
    rows = k if candidates is None else candidates
    per_query = (per_query if rows is None else per_query.limit(rows)).lateral("hits")
    query = select(queries_values.c.q.label("q"), *per_query.c).select_from(
        queries_values.join(per_query, literal(True))
    )
    return parsed, template, boosts, query


def build_search_many_query(graph, queries, target: str = "nodes", k: Optional[int] = 10,
                            where: Any = None, boost=None):
    """The single statement search_many() runs: every query ranked in
    one round trip.

    WHAT THIS BUYS, measured rather than assumed: one round trip
    instead of N. It does NOT reduce the arithmetic -- every query
    still scores every candidate, so 8 queries over 4k x 384-dim
    measured 1.08x against a Python loop on a local database, where a
    round trip is nearly free. The win scales with LATENCY, not with
    N: at a realistic 5-20ms each, N-1 saved round trips is the whole
    benefit, and it is worth having for exactly that reason. If the
    arithmetic is what hurts, fewer candidates (`where=`) is the lever,
    not batching.

    Each query keeps its own field set, weights, thresholds and
    missing-modes; only the vectors vary per row. Queries must
    therefore agree on the SHAPE -- same fields in the same order --
    or they would need different SQL, which one statement cannot be.
    """
    _, _, _, query = _prepare_search_many_query(graph, queries, target=target, k=k, where=where,
                                                boost=boost)
    return query


def search_many_candidates(graph, queries, target: str = "nodes", k: Optional[int] = 10,
                           where: Any = None, boost=None, connection=None, rerank=None) -> list:
    """The rows the one statement fetches per query, BEFORE any
    reranking -- search_candidates()' batch twin, split for the same
    reason: the provider calls happen with the connection closed."""
    with _read_connection(graph, connection) as conn:
        _ensure_lazy_vectors(graph, conn)
        parsed, template, boosts, query = _prepare_search_many_query(
            graph, queries, target=target, k=k, where=where, boost=boost,
            candidates=None if rerank is None else rerank.candidates)
        grouped: dict = {str(i): [] for i in range(len(parsed))}
        try:
            rows = conn.execute(query).mappings().all()
        except ProgrammingError as exc:
            names = [name for one in parsed for name in _near_field_names(one)]
            _raise_if_unmigrated(graph, target, names, conn, exc, "vector_search_many()")
            raise
        for row in rows:
            hit = _format_hit({key: value for key, value in row.items() if key != "q"},
                              template, boosts)
            grouped[row["q"]].append(hit)
    return [grouped[str(i)] for i in range(len(parsed))]


def search_many(graph, queries, target: str = "nodes", k: Optional[int] = 10,
                where: Any = None, boost=None, connection=None, rerank=None,
                rerank_queries: Optional[list] = None) -> list:
    """Results per query, in the order the queries were given -- an
    empty list for a query nothing matched, so index i always answers
    query i.

    `rerank=` is PER CALL, where `boost=` already sits, and it produces
    N reranker calls: one call cannot serve two queries, because every
    rerank API takes `query` (singular) and the score IS the
    (query, document) relationship. They run SEQUENTIALLY here and
    concurrently on the async path (arerank_many) -- this call exists to
    turn N round trips into one, and N sequential provider calls would
    hand that back."""
    if rerank is not None and rerank_queries is None:
        rerank_queries = rerank_query_texts(queries, rerank, k, "vector_search_many()")
    grouped = search_many_candidates(graph, queries, target=target, k=k, where=where,
                                     boost=boost, connection=connection, rerank=rerank)
    if rerank is None:
        return grouped
    return [rerank_hits(hits, rerank, query, k)
            for hits, query in zip(grouped, rerank_queries, strict=True)]


# ---------------------------------------------------------------------
# Reranking a fetched candidate list
#
# The stage that must NOT run with a connection open. Everything here
# happens after the SQL round trip has closed, for the reason
# set_vectors() resolves its embeddings before it takes a connection: a
# provider call inside an open transaction holds a snapshot -- and on
# the write path row locks -- for a network round trip, and a provider
# dying halfway leaves that transaction open behind it.
#
# The result is ADDITIVE. A hit gains `rerank_score` and the list is
# ordered by it; `similarity` keeps the value the retrieval stage gave
# it rather than being overwritten, so a caller can see what the
# reranker actually changed. Without rerank= nothing here runs and the
# results are byte-identical to before it existed.
# ---------------------------------------------------------------------

def rerank_query_text(near, rerank, k, caller: str, k_name: str = "k") -> Optional[str]:
    """The text this Rerank scores against, plus the refusals that go
    with asking for one.

    hop.py's `_validate_rerank` is IMPORTED rather than restated: the
    three ways a rerank= cannot mean anything (nothing to reorder, a
    query that cannot be read, two numbers that disagree) are one rule,
    and a second copy is how a Start refusal and a vector_search()
    refusal drift into disagreeing about the same query."""
    if rerank is None:
        return None
    from .hop import _validate_rerank
    _validate_rerank(caller, near, k, rerank, k_name=k_name)
    # _validate_rerank has just proved every Near carries text -- that is
    # what makes this an assert rather than a second check. A raw-vector
    # Near never reaches here.
    texts = [one.text for one in (near if isinstance(near, (list, tuple)) else [near])]
    if not texts:
        # `near=[]` is not None, so _validate_rerank's "nothing to
        # reorder" test does not see it -- but an empty near ranks
        # nothing either, and that is the same refusal.
        raise ValueError(
            f"{caller}: rerank= reorders the candidates near= ranks, and near=[] is empty "
            f"-- pass a Near(...) to rank by, or drop rerank="
        )
    assert all(text is not None for text in texts), texts
    unique = list(dict.fromkeys(texts))
    if len(unique) > 1:
        # A multivector query is ONE question asked of several fields, so
        # it has one query string. Picking the first would rerank against
        # a query the caller never asked -- a plausible, confidently
        # wrong ranking, which is the answer this library refuses to
        # produce quietly.
        raise ValueError(
            f"{caller}: rerank= scores ONE query against each document, but the Near specs "
            f"carry {len(unique)} different texts ({unique[0]!r} and {unique[1]!r}) -- there "
            f"is no single query to score with. Give every Near the same text=, or drop "
            f"rerank= and rank on similarity alone"
        )
    return unique[0]


def rerank_query_texts(queries, rerank, k, caller: str) -> Optional[list]:
    """One query text per entry of vector_search_many()'s queries."""
    if rerank is None:
        return None
    return [rerank_query_text(one, rerank, k, caller)
            for one in _as_query_list(queries, caller)]


def _reranked(hits: list, scores: list, k: Optional[int]) -> list:
    """The hits with `rerank_score` attached, ordered by it, truncated
    to k.

    `similarity` is left exactly as the retrieval stage set it: dropping
    a stage's input score is never the more useful answer, and a caller
    tuning a hybrid query needs to see what the reranker moved. The
    tiebreak is the id, so two identical scores never come back in a
    different order for the same graph."""
    ranked = [{**hit, "rerank_score": float(score)}
              for hit, score in zip(hits, scores, strict=True)]
    ranked.sort(key=lambda hit: (-hit["rerank_score"], hit["id"]))
    return ranked if k is None else ranked[:k]


def rerank_hits(hits: list, rerank, query: str, k: Optional[int]) -> list:
    """One candidate list, reranked. The connection is already closed."""
    if not hits:
        # No documents, no provider call: an empty request is billed by
        # some providers and refused by others, and neither is an answer.
        return hits
    return _reranked(hits, rerank.score(query, rerank.build_documents(hits)), k)


async def arerank_hits(hits: list, rerank, query: str, k: Optional[int]) -> list:
    """rerank_hits(), awaited -- the provider call never on the loop."""
    if not hits:
        return hits
    return _reranked(hits, await rerank.ascore(query, rerank.build_documents(hits)), k)


async def arerank_many(grouped: list, rerank, queries: list, k: Optional[int]) -> list:
    """vector_search_many()'s N reranker calls, CONCURRENTLY.

    One provider call cannot serve two queries -- the score is the
    (query, document) relationship and every rerank API takes `query`
    singular -- but this call exists to turn N round trips into one, and
    issuing N provider calls one after another would hand the whole
    saving straight back."""
    return list(await asyncio.gather(*(
        arerank_hits(hits, rerank, query, k)
        for hits, query in zip(grouped, queries, strict=True))))


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


def _dims_shape(target: _Target, field: Vector):
    """The dimension CHECK's unscoped body, as a BOUND expression: NULL
    passes (a vector is optional until set_vectors() writes one), a 1-D
    array of exactly field.dimensions passes, anything else does not.

    Built on `_attach(target.table, ...)` rather than an unbound
    sa_column(), so the CheckConstraint compile_constraint()'s sibling
    _compile_check() builds from it can infer target.table and
    self-register -- the same real-metadata attachment every other
    constraint in this library goes through (see hopai.constraints)."""
    column = _attach(target.table, field.column_name)
    return or_(
        column.is_(None),
        and_(func.array_ndims(column) == 1, func.array_length(column, 1) == field.dimensions),
    )


def _field_ddl(target: _Target, field: Vector) -> list:
    name = _constraint_name(target, field)
    expression = target.scope_check(_dims_shape(target, field))
    return [
        f'ALTER TABLE {target.qualified} ADD COLUMN IF NOT EXISTS '
        f'"{field.column_name}" real[]',
        # Float noise does not compress; EXTERNAL skips the TOAST
        # compression attempt, so similarity scans read raw floats
        # instead of decompressing every row first.
        f'ALTER TABLE {target.qualified} ALTER COLUMN "{field.column_name}" '
        f'SET STORAGE EXTERNAL',
        _compile_check(target.table, name, expression),
    ]


#: The one pgvector index this emits, and the operator class it needs.
#: Cosine because that is the metric this library computes -- an
#: exported index ranking by a different one would answer a different
#: question than the code it replaces. HNSW only: IVFFlat's recall
#: depends on a `lists` value chosen from the table's size, and a
#: number this function cannot know is a number it should not guess.
PGVECTOR_INDEXES = {"hnsw": "vector_cosine_ops"}


def pgvector_exit_ddl(graph, index: Optional[str] = "hnsw") -> list:
    """The migration OFF this library's exact search and onto pgvector,
    as SQL you can read before running -- generated without importing,
    requiring, or checking for the extension.

        for statement in graph.pgvector_exit_ddl():
            print(statement)

    Emitted per declared field: convert the `real[]` column to
    `vector(d)`, drop this graph's dimension CHECK (the type enforces
    it now), and build an approximate index for cosine.

    THREE THINGS TO KNOW BEFORE RUNNING IT, because they are one-way:

      - **The column is shared by every graph in the table.** A
        `vector(d)` fixes d in the TYPE, so after this migration every
        graph must use d dimensions for that field. If two graphs
        declared it differently, this DDL will fail on the second
        graph's rows -- which is the honest outcome, since pgvector
        cannot represent what the CHECK could.
      - **The search stops being exact.** The HNSW index this emits is
        approximate: it answers fast and sometimes wrongly, which is
        the trade this library declines to make silently on your
        behalf. Recall becomes a tuning problem (`ef_search`), and
        `hopai`'s own search no longer runs on these
        columns.
      - **hopai does not drive pgvector.** After this, queries are
        yours to write (`ORDER BY vec_x <=> $1`). This function exists
        so outgrowing the library is a documented door rather than a
        rewrite -- not so hopai can pretend to support both.

    `index=None` emits the conversion without an index. HNSW is the
    only method offered, and deliberately: IVFFlat's recall depends on
    a `lists` value derived from the table's size, and a number this
    function cannot know is a number it should not guess.
    """
    if index is not None and index not in PGVECTOR_INDEXES:
        raise ValueError(
            f"index must be None or one of {sorted(PGVECTOR_INDEXES)}, got {index!r}"
        )
    if graph._vectors is None:
        raise ValueError("pgvector_exit_ddl() needs vector fields and none are defined -- "
                         "call define_vectors(...) first")
    statements = ["CREATE EXTENSION IF NOT EXISTS vector"]
    for target_name in _TARGETS:
        target = _target_for(graph, target_name)
        for field in graph._vectors[target_name].values():
            column, dims = field.column_name, field.dimensions
            statements.append(
                f'ALTER TABLE {target.qualified} DROP CONSTRAINT IF EXISTS '
                f'"{_constraint_name(target, field)}"')
            statements.append(
                f'ALTER TABLE {target.qualified} ALTER COLUMN "{column}" '
                f'TYPE vector({dims}) USING "{column}"::vector({dims})')
            if index is not None:
                statements.append(
                    f'CREATE INDEX IF NOT EXISTS "ix_{target.table.name}_{column}_{index}" '
                    f'ON {target.qualified} USING {index} '
                    f'("{column}" {PGVECTOR_INDEXES[index]})')
    return statements


def stale_vectors(graph, node_fields=None, edge_fields=None,
                  limit: Optional[int] = None, after=None, connection=None) -> dict:
    """Which rows need (re-)embedding, per field:

        {"nodes": {"summary": {"missing": ["4"], "wrong_dimensions": ["7"]}},
         "edges": {}}

    Two categories, because they need the same action for different
    reasons: `missing` never had a vector, `wrong_dimensions` has one
    the current declaration no longer fits -- which is exactly the
    window a dimension change opens, since migrate_vectors() refuses
    to reinterpret stored vectors and set_vectors() refuses to write
    the wrong size. Without this, closing that window means writing
    the catalog query by hand.

    Ids come back as strings, like every other result, so they feed
    straight back into set_vectors(). `node_fields`/`edge_fields` name
    FIELDS (not ids, not rows) and default to every declared field of
    that target.

    `limit` caps each field's lists, and `after` reads only ids beyond
    the one given -- a keyset cursor over the same id order this always
    returned. Paging needs BOTH: a row that can never be filled in (no
    source text to embed) stays stale forever, so `limit` alone hands
    back the same leading rows on every call and never reaches the work
    behind them. Walk with `after=<the largest id you saw>` instead."""
    if graph._vectors is None:
        raise ValueError("stale_vectors() needs vector fields and none are defined -- "
                         "call define_vectors(...) first")
    result: dict = {"nodes": {}, "edges": {}}
    for target_name, names in (("nodes", node_fields), ("edges", edge_fields)):
        if names is None and node_fields is None and edge_fields is None:
            names = sorted(_defined(graph, target_name, "stale_vectors()")) \
                if (graph._vectors or {}).get(target_name) else []
        for name in names or ():
            field = _field(graph, target_name, name, "stale_vectors()")
            table = _table(graph, target_name)
            id_column = getattr(
                table.c, graph.node_id_col if target_name == "nodes" else graph.edge_id_col)
            column = _attach(table, field.column_name)
            query = (
                select(cast(id_column, SAString).label("id"), column.is_(None).label("missing"))
                .where(graph._scoped(table),
                       or_(column.is_(None),
                           func.array_length(column, 1).is_distinct_from(field.dimensions)))
                .order_by(id_column)
            )
            if after is not None:
                # On the RAW id, matching the ORDER BY: compared as text
                # '10' sorts before '9', and a cursor that disagrees with
                # the order it walks skips rows and repeats others.
                query = query.where(id_column > _coerce_id(after))
            if limit is not None:
                query = query.limit(limit)
            missing, wrong = [], []
            with _read_connection(graph, connection) as conn:
                for row in conn.execute(query):
                    (missing if row.missing else wrong).append(row.id)
            result[target_name][name] = {"missing": missing, "wrong_dimensions": wrong}
    return result


def embed_stale(graph, node_fields=None, edge_fields=None, limit=None,
                batch: int = 1000) -> dict:
    """Fill in every stale vector from its source property:

        {"nodes": {"summary": {"embedded": ["1", "2"], "skipped": ["7"]}},
         "edges": {}}

    stale_vectors() says which rows need a vector; this reads each
    one's `source` property, embeds them, and writes them. `skipped`
    is the rows whose source property is absent, null, blank, or not a
    JSON string -- not an error, since a node with no abstract
    legitimately has no abstract vector, but reported so it is never
    mistaken for work done.

    Fields with no embed= are not touched by the default sweep and
    cannot be named: this call is only about the ones that declare how
    to embed themselves. Naming one that does not raises rather than
    returning a zero for it.

    WALKS THE FIELD IN PAGES of `batch` rows, each its own embed call
    and its own transaction, so a backfill of any size costs bounded
    memory and resumes where it stopped -- re-running simply finds
    what is still stale. That is a deliberate exception to "writes are
    one transaction": a backfill is many independent writes, and a
    retry after a failure collides with nothing, it continues.

    The paging is a keyset cursor rather than a window, because rows
    that can NEVER be filled in are stale forever: a plain LIMIT hands
    back the same unembeddable leading rows on every pass and never
    reaches the work behind them, reporting success while doing
    nothing. `limit` caps the rows one call takes on per field; the
    default is the whole field."""
    if graph._vectors is None:
        raise ValueError("embed_stale() needs vector fields and none are defined -- "
                         "call define_vectors(...) first")
    chosen = {}
    for target_name, names in (("nodes", node_fields), ("edges", edge_fields)):
        defined = (graph._vectors or {}).get(target_name) or {}
        if names is None:
            names = sorted(name for name, field in defined.items() if field.embed is not None)
        else:
            for name in names:
                _embedder(_field(graph, target_name, name, "embed_stale()"),
                          target_name, "embed_stale()")
        chosen[target_name] = list(names)
    if not chosen["nodes"] and not chosen["edges"] and node_fields is None \
            and edge_fields is None:
        # Returning {} here would read as "nothing was stale", which is
        # the opposite of "nothing here can embed itself".
        raise ValueError(
            "embed_stale(): no vector field on this graph declares an embedder, so there "
            "is nothing it can fill in -- redeclare a field as Vector(name, dimensions, "
            "embed=<your embedding client>), or write vectors with set_vectors()"
        )

    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError(f"embed_stale(): batch must be a positive integer, got {batch!r}")

    result: dict = {"nodes": {}, "edges": {}}
    for target_name, names in chosen.items():
        key = "node_fields" if target_name == "nodes" else "edge_fields"
        for name in names:
            field = _field(graph, target_name, name, "embed_stale()")
            embedded, skipped, cursor = [], [], None
            while limit is None or len(embedded) + len(skipped) < limit:
                page = batch if limit is None \
                    else min(batch, limit - len(embedded) - len(skipped))
                report = stale_vectors(graph, limit=page, after=cursor,
                                       **{key: [name]})[target_name][name]
                ids = report["missing"] + report["wrong_dimensions"]
                if not ids:
                    break
                # The two lists are ordered within themselves but
                # concatenated out of order, so the cursor is the
                # largest id in the page, not its last element.
                cursor = max(ids, key=_coerce_id)
                texts = _source_texts(graph, target_name, field.source, ids)
                fillable = [row_id for row_id in ids if texts.get(row_id)]
                embedded.extend(fillable)
                skipped.extend(row_id for row_id in ids if not texts.get(row_id))
                if fillable:
                    # set_vectors() batches the embedding itself and keeps
                    # it outside the transaction, so handing it the text is
                    # both shorter and the one place that ordering lives.
                    set_vectors(graph, **{target_name: [
                        {"id": row_id, name: texts[row_id]} for row_id in fillable
                    ]})
            result[target_name][name] = {"embedded": embedded, "skipped": skipped}
    return result


def _source_texts(graph, target: str, source: str, ids: list) -> dict:
    """{id: text} for the rows named, reading the source property.

    STRINGS ONLY, which jsonb_typeof enforces rather than ->> alone:
    ->> renders every scalar as text, so a numeric property would
    embed as "5" and a boolean as "true" -- an embedding of the
    rendering, not of anything the caller wrote. Those rows come back
    None and land in `skipped`, where they can be seen.

    ->> and not -> for the strings themselves: -> leaves the JSON
    quotes on, and every embedding would carry them."""
    if not ids:
        return {}
    from .ingest import BATCH_SIZE, _chunks
    table = _table(graph, target)
    id_column = getattr(table.c, graph.node_id_col if target == "nodes" else graph.edge_id_col)
    found = {}
    with graph.engine.connect() as connection:
        for chunk in _chunks(list(ids), BATCH_SIZE):
            query = select(
                cast(id_column, SAString).label("id"),
                case((func.jsonb_typeof(table.c.properties[source]) == "string",
                      table.c.properties[source].astext),
                     else_=None).label("text"),
            ).where(graph._scoped(table),
                    cast(id_column, SAString).in_([str(one) for one in chunk]))
            for row in connection.execute(query):
                found[row.id] = row.text if isinstance(row.text, str) and row.text.strip() \
                    else None
    return found


def vector_ddl(graph) -> list:
    """The exact SQL migrate_vectors() would run, without running it --
    the same contract as constraint_ddl() and schema_ddl(), including
    refusing an undeclared handle rather than returning an empty list
    that reads as "nothing to migrate"."""
    if graph._vectors is None:
        raise ValueError("vector_ddl() needs vector fields and none are defined -- "
                         "call define_vectors(...) first")
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

#: Every column name on a table, for load_vectors() to sift for
#: vec_* prefixes in Python -- LIKE 'vec\\_%' ESCAPE '\\' would do the
#: same filtering in SQL, but a table has few enough columns that
#: fetching them all and testing str.startswith() avoids getting the
#: escape right for a query that runs once, not per row.
_ALL_COLUMNS = text("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = :table AND table_schema = COALESCE(CAST(:schema AS text), current_schema())
""")


def migrate_vectors(graph, connection=None) -> list:
    """Apply the declared vector fields: add each column and its
    per-graph dimension CHECK. Idempotent, one transaction, and it
    REFUSES drift instead of papering over it -- a column of the wrong
    type, or an existing constraint declaring different dimensions,
    names drop_vectors() (or the conflicting column) rather than
    quietly serving two incompatible definitions. Returns
    "table.column" for every field ensured, in order."""
    from .ingest import one_transaction

    if graph._vectors is None:
        raise ValueError("migrate_vectors() needs vector fields and none are defined -- "
                         "call define_vectors(...) first")
    ensured = []
    with one_transaction(graph, connection) as conn:
        for target_name in _TARGETS:
            table = _table(graph, target_name)
            target = _target_for(graph, target_name)
            for field in graph._vectors[target_name].values():
                existing = conn.execute(_COLUMN_TYPE, {
                    "table": table.name, "column": field.column_name, "schema": table.schema,
                }).scalar()
                if existing is not None and existing not in ("_float4", "_float8"):
                    raise ValueError(
                        f"{target_name}.{field.column_name} already exists with type "
                        f"{existing!r}, not a float array -- rename the vector field, or "
                        f"drop the conflicting column"
                    )
                add_column, storage, add_check = _field_ddl(target, field)
                conn.execute(text(add_column))
                conn.execute(text(storage))
                name = _constraint_name(target, field)
                definition = conn.execute(_CONSTRAINT_DEF, {
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
                        conn.execute(text(add_check))
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


def load_vectors(graph, connection=None) -> dict:
    """The inverse of a lost declaration: read every vec_* column
    already migrated for THIS graph back out of the database, and
    populate this handle's registry from it -- issue #53's fix for the
    handle (a fresh process, a second Graph object, an in_graph() that
    forgot to redeclare) that finds only a raw UndefinedColumn where
    every other refusal here names the fix.

    THE COLUMN IS SHARED, THE DIMENSION IS NOT: a vec_* column can be
    present because SOME graph in this table migrated it, but the
    dimension CHECK migrate_vectors() adds is scoped to one graph's
    token (see schema._graph_token / _constraint_name). So a column
    with no matching constraint FOR THIS GRAPH is skipped rather than
    guessed at -- it means this graph never migrated the field, even
    though the physical column already exists for another graph's
    sake. This is also why the check is read back from
    information_schema/pg_constraint directly rather than from
    graph.nodes_tbl/edges_tbl's SQLAlchemy metadata: a fresh handle
    that never called define_vectors() has no vec_* Column attached to
    that metadata at all (attach_columns() is what puts it there), so
    the only place the shape still exists is the database itself.

    RECOVERING THE SHAPE IS NOT RECOVERING THE POLICY: embed= is an
    application's own embedding client and source= is a choice of
    which property holds the text to embed -- neither is stored in
    SQL, so every recovered field comes back with embed=None and
    source=<field name>, Vector's own defaults. A field that embeds
    text needs `define_vectors(nodes=[Vector('summary', 1536,
    embed=<your client>)])` after this call to write or Near(text=...)
    against it again; reading and vector_search() with a vector= need
    nothing more than what this call already recovers.

    Populates and returns the registry exactly like define_vectors()
    does (attach_columns() included), so the result is immediately
    usable:

        g2 = Graph(engine)     # a second handle; define_vectors() never ran here
        g2.load_vectors()      # -> {"nodes": {"summary": Vector("summary", 1536)}, "edges": {}}
        g2.vector_search(Near("summary", q))    # now works
    """
    registry: dict = {"nodes": {}, "edges": {}}
    with _read_connection(graph, connection) as conn:
        for target_name in _TARGETS:
            table = _table(graph, target_name)
            target = _target_for(graph, target_name)
            columns = conn.execute(_ALL_COLUMNS, {
                "table": table.name, "schema": table.schema,
            }).scalars().all()
            for column_name in columns:
                if not column_name.startswith(VECTOR_COLUMN_PREFIX):
                    continue
                name = column_name[len(VECTOR_COLUMN_PREFIX):]
                if not _NAME.match(name):
                    # A vec_*-prefixed column this library did not name --
                    # someone else's column, not a field it forgot.
                    continue
                # dimensions=1 is a placeholder: _constraint_name() only
                # reads .name off the field, and the real dimensions are
                # what this lookup exists to recover.
                probe_name = _constraint_name(target, Vector(name, 1))
                definition = conn.execute(_CONSTRAINT_DEF, {
                    "name": probe_name, "table": target.qualified,
                }).scalar()
                if definition is None:
                    continue
                declared = re.search(
                    rf'array_length\("?{re.escape(column_name)}"?, 1\) = (\d+)', definition)
                if declared is None:
                    continue
                registry[target_name][name] = Vector(name, int(declared.group(1)))
    graph._vectors = registry
    attach_columns(graph)
    return {target: dict(fields) for target, fields in registry.items()}


def drop_vectors(graph, node_fields=None, edge_fields=None, connection=None) -> list:
    """The inverse of migrate_vectors() for THIS graph: drop each
    field's dimension constraint and NULL its values in this graph's
    rows. The column itself stays -- it is shared by every graph in
    the table, so removing it is a deliberate manual ALTER, not a side
    effect of one graph cleaning up. `node_fields`/`edge_fields` name
    FIELDS. Missing fields are ignored, like drop_constraints().

    This is the ONE call here that works without define_vectors(): it
    probes the catalog rather than the registry, because a teardown
    script or a fresh in_graph() handle legitimately has no
    declaration for a column that exists. It is also destructive, so
    that combination is deliberate rather than an oversight -- an
    undeclared handle can still clear this graph's vectors.

    The DECLARATION survives, deliberately: drop_constraints() does not
    mutate declarations either, and the recipe migrate_vectors() prints
    for a dimension change ("drop_vectors(...), then migrate and
    re-embed") only works if the field is still declared when you get
    to step two. Returns "table.column" per field, the vocabulary
    migrate_vectors() returns."""
    from .constraints import detach_constraint
    from .ingest import one_transaction

    dropped = []
    with one_transaction(graph, connection) as conn:
        for target_name, names in (("nodes", node_fields), ("edges", edge_fields)):
            table = _table(graph, target_name)
            target = _target_for(graph, target_name)
            for entry in names or ():
                field = entry if isinstance(entry, Vector) else Vector(entry, 1)
                name = _constraint_name(target, field)
                conn.execute(text(
                    f'ALTER TABLE {target.qualified} DROP CONSTRAINT IF EXISTS "{name}"'
                ))
                detach_constraint(table, "check", name)
                exists = conn.execute(_COLUMN_TYPE, {
                    "table": table.name, "column": field.column_name, "schema": table.schema,
                }).scalar()
                if exists is not None:
                    column = _attach(table, field.column_name)
                    conn.execute(
                        update(table)
                        .where(graph._scoped(table), column.isnot(None))
                        .values({field.column_name: None})
                    )
                dropped.append(f"{table.name}.{field.column_name}")
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


def _group_vector_rows(graph, nodes, edges) -> list:
    """The pure half of plan_vector_writes()/aplan_vector_writes():
    validate every row and split it into per-field-signature groups,
    WITHOUT calling the embedder -- that is the one part sync and
    async must do differently (embed_documents() vs aembed_documents()),
    so it is left to each of their finishing passes below.

    Returns one (target_name, table, id_column, groups, fields, pending)
    per target that has rows. `pending` is {field name: [(cleaned row,
    text)]}, still needing a provider call; `groups` already holds every
    row that carried no text at all."""
    parts = []
    for target_name, rows in (("nodes", nodes), ("edges", edges)):
        if not rows:
            continue
        fields = _defined(graph, target_name, "set_vectors()")
        table = _table(graph, target_name)
        id_name = graph.node_id_col if target_name == "nodes" else graph.edge_id_col
        id_column = getattr(table.c, id_name)

        groups: dict = {}
        seen_ids = set()
        #: {field name: [(cleaned row, text)]} -- filled in below and
        #: embedded once per field, after every row has been checked.
        pending: dict = {}
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
                where = f"{target_name} {row['id']!r} {name!r}"
                value = row[name]
                if isinstance(value, str):
                    _embedder(fields[name], target_name, "set_vectors()")
                    if not value.strip():
                        raise ValueError(
                            f"{where}: the text to embed is empty -- pass None to clear "
                            f"this field's vector, or text with something in it"
                        )
                    pending.setdefault(name, []).append((cleaned, value))
                    value = None                      # filled in by the embed pass
                elif value is not None:
                    value = list(_clean_vector(value, where))
                    if len(value) != fields[name].dimensions:
                        raise ValueError(
                            f"{target_name} {row['id']!r}: vector for {name!r} has "
                            f"{len(value)} dimensions, the field is defined with "
                            f"{fields[name].dimensions}"
                        )
                cleaned[name] = value
            groups.setdefault(tuple(names), []).append((row_id, cleaned))
        parts.append((target_name, table, id_column, groups, fields, pending))
    return parts


def _apply_embedded_rows(target_name: str, name: str, dimensions: int, waiting: list,
                         vectors: list) -> None:
    """Write each embedded vector back into the row `_group_vector_rows()`
    set aside for it, in place -- shared by plan_vector_writes()'s and
    aplan_vector_writes()'s finishing pass."""
    for (cleaned, _), vector in zip(waiting, vectors, strict=True):
        if len(vector) != dimensions:
            raise ValueError(
                f"set_vectors(): the embedder for {target_name} field {name!r} "
                f"returned {len(vector)} dimensions, the field is defined with "
                f"{dimensions} -- nothing was written"
            )
        cleaned[name] = list(vector)


def plan_vector_writes(graph, nodes=None, edges=None) -> list:
    """Everything set_vectors() does BEFORE it takes a connection:
    validate every row, and turn every string into a vector.

    Separate so it can be called from outside a transaction. The
    sync path gets that for free -- it plans, then opens one. The
    async one cannot: AsyncGraph opens the transaction and reaches
    this module through run_sync(), so planning inside set_vectors()
    would put a provider round trip inside an open transaction with
    the row locks already held. AsyncGraph.set_vectors() calls
    aplan_vector_writes() -- the awaited twin below -- first, and
    passes the finished plan in.
    """
    # Validation is pure, so it all happens before a connection is
    # taken -- including the embedding, which is the slow part.
    plan = []
    for target_name, table, id_column, groups, fields, pending in _group_vector_rows(
            graph, nodes, edges):
        # One provider call per field for the whole set_vectors(), not
        # one per row: 500 rows of text is 500 HTTP round trips done
        # naively, and the Embedder already chunks to the provider's cap.
        for name, waiting in pending.items():
            vectors = fields[name].embed.embed_documents([text for _, text in waiting])
            _apply_embedded_rows(target_name, name, fields[name].dimensions, waiting, vectors)
        plan.append((target_name, table, id_column, groups))
    return plan


async def aplan_vector_writes(graph, nodes=None, edges=None) -> list:
    """Async twin of plan_vector_writes() -- AsyncGraph.set_vectors()
    awaits this BEFORE opening its transaction (see its docstring),
    which is what keeps a text row's provider call off the event loop's
    own thread entirely, rather than merely off the greenlet bridge the
    way traverse()/aggregate()/vector_search() need to (issue #74).

    Every field's embed call is gathered, not awaited one at a time: a
    call writing text to two or three DIFFERENT fields should cost the
    slowest provider round trip, not their sum."""
    plan = []
    for target_name, table, id_column, groups, fields, pending in _group_vector_rows(
            graph, nodes, edges):
        names = list(pending)
        answers = await asyncio.gather(*(
            fields[name].embed.aembed_documents([text for _, text in pending[name]])
            for name in names))
        for name, vectors in zip(names, answers, strict=True):
            _apply_embedded_rows(target_name, name, fields[name].dimensions, pending[name], vectors)
        plan.append((target_name, table, id_column, groups))
    return plan


def set_vectors(graph, nodes=None, edges=None, connection=None, plan=None) -> int:
    """UPDATE existing rows' vector columns. Each row is
    {"id": ..., <field>: <vector, text, or None>}; every non-id key
    must be a defined field of the right dimensionality. One
    transaction for the whole call: an id that matches no row in this
    graph fails everything by name, so a retry never collides with
    half a write. Returns the number of rows updated.

    A string value is embedded with the field's declared embedder --
    every string in the call, per field, in ONE batched provider call,
    resolved BEFORE the transaction opens. That ordering is the point:
    an HTTP call inside an open transaction holds row locks for the
    length of a network round trip, and a provider failure halfway
    through would leave the write half-done."""
    from .ingest import BATCH_SIZE, _chunks, one_transaction

    if plan is None:
        plan = plan_vector_writes(graph, nodes, edges)

    total = 0
    with one_transaction(graph, connection) as conn:
        for target_name, table, id_column, groups in plan:
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
                    updated = {row[0] for row in conn.execute(statement)}
                    missing = [row_id for row_id, _ in chunk if row_id not in updated]
                    if missing:
                        raise ValueError(
                            f"no {target_name[:-1]} with id {missing[0]!r} in graph "
                            f"{graph.graph!r} -- set_vectors() updates existing rows; "
                            f"nothing from this call was written"
                        )
                    total += len(chunk)
    return total


def get_vectors(graph, node_ids=None, edge_ids=None, node_fields=None,
                edge_fields=None, connection=None) -> dict:
    """Read stored vectors back, since traversal results never carry
    them. Returns {"nodes": {id: {field: [floats] | None}}, "edges":
    {...}} with string ids, matching every other result. Ids that
    match no row are simply absent. `node_fields`/`edge_fields` narrow
    which fields are read for that target; the default is all defined."""
    result: dict = {"nodes": {}, "edges": {}}
    for target_name, ids, fields in (("nodes", node_ids, node_fields),
                                     ("edges", edge_ids, edge_fields)):
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
        with _read_connection(graph, connection) as conn:
            for row in conn.execute(query).mappings():
                result[target_name][row["id"]] = {name: row[name] for name in wanted}
    return result
