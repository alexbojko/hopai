"""
hopai.pgvector

The optional pgvector backend: `vector(d)` columns and an HNSW index,
chosen once per Graph, for the field set that has outgrown exact search.

    graph = Graph(dsn, vector_backend="pgvector")
    graph.define_vectors(nodes=[Vector("summary", 1536)])
    graph.migrate_vectors()          # vector(1536) + HNSW, not real[]

    graph.vector_search(Near("summary", q), k=10, where={"type": "person"})

WHY THIS EXISTS, given that vectors.py argues at length for the
opposite. The default backend computes EXACT cosine over every
candidate row, and its cost is linear in candidates x dimensions --
~0.13us per element, so 100k x 1536-dim is ~20 seconds. That is the
right trade for a knowledge graph filtered down to a few thousand
candidates, and the wrong one past it. Before this module the only
answer was pgvector_exit_ddl(): migrate the columns and leave hopai
behind. This backend is the same migration with hopai still driving,
for callers who want the index without giving up traversal, filtering
and the rest of the library. pgvector_exit_ddl() remains, and is still
the answer for anyone who wants to write their own SQL afterwards.

IT IS OPT-IN, AND IT IS A REAL DEPENDENCY. Every other line in this
library holds to "Postgres and SQLAlchemy are the whole stack"; this
backend needs the `vector` extension installed in the server. Nothing
imports it, nothing requires it, and no default reaches it -- a Graph
built without `vector_backend="pgvector"` emits byte-identical SQL to
the pre-pgvector engine. No PYTHON package is needed either: the type
is rendered as SQL text (`'[0.1,0.2]'::vector`) and the operator as
`<=>`, so `pip install pgvector` is not part of this. The extension in
the database is the whole dependency.

WHAT YOU GIVE UP, stated plainly because the point of the exact
backend was never having to: an HNSW index answers APPROXIMATELY. It
can miss a true nearest neighbor, and no `ef_search` makes that
guarantee absolute. Opting into this backend is opting into that. The
library's job is to make sure it is the only thing you gave up --
which is what the next paragraph is about.

WHY pgvector >= 0.8 IS REQUIRED, and it is not a version-currency
preference. Below 0.8 an HNSW scan visits a fixed candidate window
(`ef_search`) and any WHERE filter is applied to what comes back. With
a selective filter that silently returns FEWER ROWS THAN EXIST:
measured on 20,000 rows, a filter matching exactly 2 of them, k=10 --
`iterative_scan = off` returns ONE of the two matching rows, and
reports success. Not "approximately ranked": a row that matched the
filter, that the caller asked for, missing, with nothing to indicate
it. pgvector 0.8's `hnsw.iterative_scan` fixes it by resuming the scan
until k rows survive the filter, and this backend sets
`strict_order` on every search for exactly that reason. So the version
floor buys the one guarantee this library will not trade away: `where=`
still means `where=`. A server below it is refused by name at
migrate/search time rather than served a quietly lossy answer.

SINGLE-FIELD SEARCH ONLY, and the refusal is not a scoping shortcut.
An HNSW index accelerates one query shape: `ORDER BY column <op> query
LIMIT k`, the exact ordering the index was built for. hopai's
multivector search ranks by a WEIGHTED SUM of several fields'
similarities (`w0*sim_summary + w1*sim_title`), and no index encodes
"nearest under a blend of two independent distance spaces" -- Postgres
can only answer it by scanning every row and sorting the sum. So a
multivector search under this backend would pay for the extension and
get the exact backend's cost anyway, while quietly having become
approximate. `Boost` is the same shape (a non-similarity term added
into the ranked score). Both are refused here and named, rather than
served at full cost under a name that promises an index. The exact
backend answers them, correctly and without an extension; use it, or
issue #96 for the retrieve-then-fuse design that would make them
index-backed.

COSINE ONLY, matching the exact backend: `vector_cosine_ops`, and
similarity reported as `1 - (a <=> b)` so a hit's `similarity` means
the same number under either backend. A caller comparing the two
should see the same scale, not two conventions.

MISSING IS STILL MISSING. The exact backend reports NULL similarity
for a vector that is NULL, all zeros, or the wrong length; this one
keeps the first two (the third is impossible -- `vector(d)` enforces
length in the TYPE, which is the one thing pgvector does better here).
An all-zero vector has no direction and pgvector's cosine distance
returns NaN for it; NaN is not filtered as "far away", it is reported
as missing, because a zero vector matches nothing rather than matching
badly.
"""

from __future__ import annotations

import re
from typing import Optional

from sqlalchemy import Float, cast, func, literal, text
from sqlalchemy.types import UserDefinedType

#: The two values `Graph(vector_backend=)` accepts. "exact" is the
#: default and the pre-pgvector engine, unchanged.
VECTOR_BACKENDS = ("exact", "pgvector")

DEFAULT_VECTOR_BACKEND = "exact"

#: See the module docstring: below this, a filtered search silently
#: returns fewer rows than match. Measured, not inferred from a
#: changelog -- `hnsw.iterative_scan` is the 0.8 feature that makes
#: `where=` mean `where=`, and there is no way to emulate it from SQL.
MINIMUM_PGVECTOR = (0, 8, 0)

#: What this backend sets before every search it runs. `strict_order`
#: rather than `relaxed_order`: relaxed lets pgvector return rows
#: slightly out of distance order to fill k faster, and a caller
#: reading `similarity` down a result list must never see it go back
#: up. The exact backend cannot produce that, so neither may this one.
ITERATIVE_SCAN = "strict_order"

#: The one index this backend builds, and the operator class it needs.
#: Cosine because that is the metric hopai reports. HNSW only, for the
#: reason pgvector_exit_ddl() already gives: IVFFlat's recall depends
#: on a `lists` value derived from the table's size, and a number this
#: library cannot know is a number it should not guess.
HNSW_OPS = "vector_cosine_ops"

#: pgvector's cosine DISTANCE operator. Similarity is `1 - distance`.
COSINE_DISTANCE = "<=>"


class Vector(UserDefinedType):
    """`vector(d)` as SQLAlchemy sees it.

    A UserDefinedType rather than pgvector's own SQLAlchemy integration,
    and that is the same judgement embeddings.py makes about provider
    packages: importing `pgvector.sqlalchemy` would turn a Postgres
    extension into a Python dependency for everyone who merely imports
    hopai. The type renders as `vector(d)` and values render as
    pgvector's own text form, which is all this backend ever needs.
    """

    cache_ok = True

    def __init__(self, dimensions: Optional[int] = None):
        self.dimensions = dimensions

    def get_col_spec(self, **kw) -> str:
        if self.dimensions is None:
            return "vector"
        return f"vector({self.dimensions})"

    def bind_processor(self, dialect):
        """A Python sequence goes to the server in pgvector's text form.

        Without this a list binds as a Postgres array and the server
        refuses it against a `vector` parameter -- the same reason the
        exact backend's ARRAY(REAL) works and this cannot borrow it."""
        def process(value):
            if value is None or isinstance(value, str):
                return value
            return literal_vector(value)
        return process

    def result_processor(self, dialect, coltype):
        """And comes back as a list of floats, not as pgvector's text.

        get_vectors() promises `[floats] | None` under either backend,
        and a caller doing arithmetic on what it returns must not have
        to discover that one backend hands back the string
        '[1,0,0]'. Parsed here, at the type, so every read site is
        covered by construction rather than one at a time."""
        def process(value):
            if value is None or not isinstance(value, str):
                return value
            inner = value.strip().strip("[]")
            if not inner:
                return []
            return [float(part) for part in inner.split(",")]
        return process


def literal_vector(vector) -> str:
    """A vector in pgvector's text form: `[0.1,0.2,0.3]`.

    repr() would render numpy scalars and Python floats differently and
    embed `np.float32(...)` into SQL; float() first keeps one spelling.
    The caller has already validated finiteness (Near/_clean_vector), so
    this does not re-check it -- a NaN reaching here would be a bug
    upstream, not user input.
    """
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def bind_vector(vector, dimensions: Optional[int] = None):
    """A query vector as a bound parameter cast to `vector(d)`.

    A bound parameter, never an interpolated literal: the values are
    floats and could not carry SQL, but a 1536-float literal inlined
    into every statement defeats Postgres's plan cache and bloats the
    logs for nothing.
    """
    return cast(literal(literal_vector(vector)), Vector(dimensions))


def distance(column, vector, dimensions: Optional[int] = None):
    """`column <=> :query` -- cosine distance, in [0, 2], NULL when the
    column is NULL and NaN when either side has no direction.

    This expression, verbatim and un-wrapped, is what the ORDER BY must
    contain for the HNSW index to serve the query. Wrapping it (in a
    CASE that maps NaN to NULL, say, or in `1 - x` to make it a
    similarity) produces the same rows and a sequential scan -- the
    index is on the operator, not on any expression built over it. That
    is why the query shape below computes distance in a subquery and
    converts to similarity outside it, rather than the other way round.
    """
    return column.op(COSINE_DISTANCE, return_type=Float)(bind_vector(vector, dimensions))


def similarity(distance_expr):
    """Cosine similarity from cosine distance, as the caller's `1 - d`.

    NaN -- an all-zero stored vector, which has no direction -- becomes
    NULL, which is what every other part of this library means by
    "missing". Postgres considers NaN equal to itself and greater than
    every real number (unlike IEEE), so `d <> d` does NOT detect it and
    a plain `d = 'NaN'` does; getting this backwards would rank a
    directionless vector as the WORST match rather than as no match,
    which is a different answer, not a rounding difference.
    """
    return func.nullif(distance_expr, float("nan"))


def is_missing(distance_expr):
    """The predicate for "this row has no usable vector": NULL column,
    or NaN distance from a zero-length stored vector. Used to drop such
    rows from a ranked result, matching the exact backend, where a NULL
    similarity never survives `combined IS NOT NULL`."""
    return distance_expr.is_(None) | (distance_expr == float("nan"))


def index_name(table_name: str, column_name: str) -> str:
    """The HNSW index's name. Not graph-scoped, deliberately, and not
    for brevity: the COLUMN is shared by every graph in the table (the
    same reason drop_vectors() NULLs values instead of dropping the
    column), so its index is shared too. One graph creating a second
    index over the same column under its own name would double the
    write cost of every other graph's inserts to buy nothing.

    Truncated to Postgres's 63-character identifier limit. The column
    name is already capped at 59 by Vector's own validation and table
    names here are short, so this is a backstop rather than a case the
    caller can reach.
    """
    return f"ix_{table_name}_{column_name}_hnsw"[:63]


def column_ddl(qualified: str, column_name: str, dimensions: int) -> str:
    """ALTER TABLE ... ADD COLUMN "vec_x" vector(d).

    No `SET STORAGE EXTERNAL` twin to the exact backend's: pgvector's
    type is already stored uncompressed for the sizes that matter, and
    an HNSW index reads from the index, not from the heap column, on
    the path this backend exists to make fast.
    """
    return (f'ALTER TABLE {qualified} ADD COLUMN IF NOT EXISTS '
            f'"{column_name}" vector({dimensions})')


def index_ddl(qualified: str, table_name: str, column_name: str) -> str:
    """CREATE INDEX ... USING hnsw ("vec_x" vector_cosine_ops)."""
    return (f'CREATE INDEX IF NOT EXISTS "{index_name(table_name, column_name)}" '
            f'ON {qualified} USING hnsw ("{column_name}" {HNSW_OPS})')


def drop_index_ddl(qualified: str, table_name: str, column_name: str) -> str:
    """The index alone. drop_vectors() NULLs a graph's values and leaves
    the shared column in place, so it leaves the shared index too --
    this exists for the caller dropping the column itself."""
    del qualified
    return f'DROP INDEX IF EXISTS "{index_name(table_name, column_name)}"'


#: Reads the installed extension's version. `extversion` rather than
#: `pg_available_extensions`: what matters is what is installed IN THIS
#: DATABASE, not what could be.
_EXTENSION_VERSION = text(
    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
)

#: Whether the server has the extension's files at all -- the difference
#: between "run CREATE EXTENSION" and "install the extension first",
#: which are different problems with different fixes and must not share
#: one error message.
_EXTENSION_AVAILABLE = text(
    "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'"
)


def parse_version(raw: str) -> tuple:
    """'0.8.0' -> (0, 8, 0). Trailing non-numeric parts are dropped, so
    a packaged '0.8.0-1' or '0.8.0devel' compares as 0.8.0 rather than
    failing to parse and refusing a server that is actually fine."""
    parts = []
    for chunk in str(raw).split("."):
        match = re.match(r"\d+", chunk.strip())
        if match is None:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def _version_text(version: tuple) -> str:
    return ".".join(str(part) for part in version)


def ensure_available(conn) -> tuple:
    """The installed pgvector version, or a refusal naming the fix.

    Three distinct failures, three distinct messages, because they need
    three different actions and a single "pgvector unavailable" would
    send the reader to the wrong one:

      - the extension's files are not on the server  -> install it
      - installed but not created in this database   -> CREATE EXTENSION
      - created but older than 0.8                   -> upgrade, and
        here is what the old one silently gets wrong

    Called at migrate time and (cached on the handle) before the first
    search, so a misconfigured server is named once, up front, instead
    of surfacing as an undefined-operator error from deep inside a
    compiled query.
    """
    installed = conn.execute(_EXTENSION_VERSION).scalar()
    if installed is None:
        available = conn.execute(_EXTENSION_AVAILABLE).scalar()
        if available is None:
            raise RuntimeError(
                "vector_backend='pgvector' needs the pgvector extension, and this server "
                "does not have it available -- install it (e.g. the pgvector/pgvector "
                "Docker image, or a postgresql-<version>-pgvector package), or build the "
                "Graph without vector_backend='pgvector' to use hopai's exact search, "
                "which needs no extension"
            )
        raise RuntimeError(
            f"vector_backend='pgvector' needs the pgvector extension, which this server has "
            f"available (version {available}) but which is not created in this database -- "
            f"run CREATE EXTENSION vector; migrate_vectors() does this for you when it has "
            f"permission, so this usually means the connecting role is not a superuser"
        )
    version = parse_version(installed)
    if version < MINIMUM_PGVECTOR:
        raise RuntimeError(
            f"vector_backend='pgvector' needs pgvector "
            f">= {_version_text(MINIMUM_PGVECTOR)} and this database has {installed} -- "
            f"below that, an HNSW scan applies where= to a fixed candidate window, so a "
            f"selective filter silently returns FEWER rows than match it (measured: 2 rows "
            f"matching, k=10, one returned). {_version_text(MINIMUM_PGVECTOR)}'s "
            f"hnsw.iterative_scan is what makes where= mean where= here. Upgrade pgvector, "
            f"or drop vector_backend='pgvector' to use hopai's exact search"
        )
    return version


def create_extension_ddl() -> str:
    return "CREATE EXTENSION IF NOT EXISTS vector"


def ensure_available_or_create(conn) -> None:
    """Create the extension if the server has it and the role may.

    migrate_vectors() is already the DDL call, so creating the
    extension there is in character -- but a non-superuser cannot, and
    failing the whole migration on that would hide the real, fixable
    situation behind a permissions error. So a failure here is
    swallowed deliberately: ensure_available() runs next and produces
    the message that names what to do, whether the cause was a missing
    extension, a missing CREATE privilege, or a version below the
    floor.
    """
    if conn.execute(_EXTENSION_VERSION).scalar() is not None:
        return
    try:
        conn.execute(text(create_extension_ddl()))
    except Exception:  # noqa: BLE001 -- ensure_available() names the fix
        conn.rollback()


def apply_scan_settings(conn) -> None:
    """Turn on the iterative scan for this connection.

    SET, not SET LOCAL, and the two were measured rather than reasoned
    about: both do reach the SELECT here (SQLAlchemy 2.0 keeps an
    implicit transaction open on a Connection, so SET LOCAL survives to
    it), and plain SET does NOT leak to the next pooled checkout --
    verified, it reads back `off`. What decides it is the FAILURE mode.
    Outside a transaction block SET LOCAL is a WARNING and a silent
    no-op, which would put the filtered search back to under-returning
    with nothing to show for it; plain SET applied where it was not
    needed is at worst a setting that makes another query more correct.
    Between a silent wrong answer and a harmless one, this library
    picks the harmless one every time.

    See the module docstring for why this is not optional: without it a
    filtered search under-returns silently.
    """
    conn.execute(text(f"SET hnsw.iterative_scan = {ITERATIVE_SCAN}"))


def refuse_unsupported(nears: list, boosts: list, caller: str) -> None:
    """Every ranking shape this backend cannot serve, refused by name.

    Called after validate_nears()/validate_boosts(), so the specs here
    are already well-formed -- what is left is whether an HNSW index
    can answer the ordering they describe, which is this module's
    question rather than theirs.
    """
    refuse_combined(nears, boosts, caller)
    for near in nears:
        if near.weight < 0:
            # A negative weight asks for the LEAST similar rows first.
            # HNSW is a nearest-neighbor index and indexes exactly one
            # direction; there is no "farthest" scan to fall back on, so
            # this would silently become a sequential scan returning the
            # opposite of what the index is for.
            raise ValueError(
                f"{caller}: vector_backend='pgvector' needs a positive Near weight and "
                f"{near.field!r} has weight={near.weight!r} -- a negative weight ranks the "
                f"LEAST similar rows first, which an HNSW index cannot serve (it indexes "
                f"nearest neighbors, in one direction). Build the Graph without "
                f"vector_backend='pgvector' if you mean to rank by dissimilarity"
            )


def refuse_combined(nears: list, boosts: list, caller: str) -> None:
    """Multivector and Boost are refused under this backend, by name.

    Not a scoping shortcut -- see the module docstring. An HNSW index
    can only answer `ORDER BY one_column <=> query`; a weighted sum
    over several fields (or a boost added into the score) is an
    ordering no index encodes, so Postgres would scan every row and
    sort the sum. The caller would pay for the extension, get the exact
    backend's cost, and have silently traded exactness for nothing. The
    refusal names the two ways forward rather than picking one.
    """
    if len(nears) > 1:
        fields = sorted(near.field for near in nears)
        raise ValueError(
            f"{caller}: vector_backend='pgvector' ranks ONE field per search, and this one "
            f"ranks {fields} -- their weighted sum is an ordering no HNSW index can serve, so "
            f"this would scan every row (the exact backend's cost) while still answering "
            f"approximately. Search one field, combine the results yourself, or build the "
            f"Graph without vector_backend='pgvector' -- hopai's exact search answers "
            f"multivector queries correctly and needs no extension"
        )
    if boosts:
        raise ValueError(
            f"{caller}: vector_backend='pgvector' does not support boost= -- a boost adds a "
            f"non-similarity term into the ranked score, which is an ordering no HNSW index "
            f"can serve, so this would scan every row while still answering approximately. "
            f"Rank by similarity here and apply the boost to the returned hits yourself, or "
            f"build the Graph without vector_backend='pgvector' -- hopai's exact search "
            f"answers hybrid ranking correctly and needs no extension"
        )


def validate_backend(name: str) -> str:
    """`vector_backend=` as Graph accepts it."""
    if name not in VECTOR_BACKENDS:
        raise ValueError(
            f"vector_backend must be one of {VECTOR_BACKENDS}, got {name!r} -- 'exact' "
            f"(the default) computes exact cosine over plain real[] columns and needs no "
            f"extension; 'pgvector' uses the pgvector extension's vector(d) type and an "
            f"approximate HNSW index"
        )
    return name
