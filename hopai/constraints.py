"""
hopai.constraints

Data integrity for a property graph, declared the way SQLAlchemy
declares it and enforced by PostgreSQL itself.

    from hopai import Unique, Required, Check, Index, PropertyType, Col, GT

    graph.define_constraints(
        nodes=[
            Required("type"),                              # the key must be present
            Unique("email"),                               # no two nodes share one
            Unique("tenant", "slug"),                      # composite
            Unique("email", where={"type": "person"}),     # only among people
            PropertyType("age", "number"),                 # not the string "42"
            Check(GT("age", 0), name="age_positive"),      # any filter, as a CHECK
            Index("type"),                                 # plain lookup index
        ],
        edges=[
            Unique(Col("start_id"), Col("end_id"), "kind"),  # one edge of a kind per pair
        ],
    )

WHY THIS EXISTS: Neo4j puts uniqueness, composite and property-existence
constraints behind an enterprise licence. Postgres has had them since
forever, and a JSONB property is as constrainable as any column once you
index the expression. Running the graph on Postgres means getting them
for free, so they are a first-class feature here rather than a footnote.

WHAT EACH ONE COMPILES TO -- worth knowing, because the semantics are
Postgres's, not an abstraction over it:

  Unique(*keys)        CREATE UNIQUE INDEX ON t ((properties->>'k'), ...)
  Unique(..., where=)  ...the same, with a partial WHERE. This is the one
                       Neo4j has no equivalent for at any price: "email is
                       unique among nodes of type person" is a partial
                       index, and a partial index is just an index.
  Required(*keys)      CHECK (properties ?& array['k', ...])
  PropertyType(k, t)   CHECK (jsonb_typeof(properties->'k') = t)
  Check(filter)        CHECK (<the filter, compiled by resolve()>)
  Index(*keys)         CREATE INDEX ON t ((properties->>'k'), ...)

TWO SEMANTICS TO KNOW, both inherited from SQL and both the behavior you
want once stated:

  - A unique index does not constrain rows where the property is MISSING.
    `properties->>'email'` is NULL for such a row, and SQL uniqueness
    lets NULLs repeat. So Unique("email") means "no two nodes share an
    email", not "every node has an email" -- pair it with
    Required("email") when you mean both. This matches Neo4j's
    uniqueness constraint, which also only applies to nodes that have the
    property.
  - PropertyType passes when the key is absent, for the same reason:
    jsonb_typeof(NULL) is NULL, and a CHECK passes on NULL. It constrains
    the type of a value that is there, not its presence.

A bare string names a JSONB property. A real table column has to say so
with Col("start_id") -- and since a bare string that happens to match a
real column name can only ever be a mistake (that column is written and
read by name already, never through `properties`), it is refused rather
than silently reinterpreted: `Required("start_id")` raises TypeError
naming the fix, the same way a typo'd Col(...) raises for a column that
does not exist. This is checked wherever a bare string could compile to
`properties->>'...'` -- Unique/Index/Required/PropertyType, and
merge_nodes()/merge_edges()'s `on=`, since key_sql() is what both the
index and the ON CONFLICT target are rendered through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from sqlalchemy import column as sa_column
from sqlalchemy import func, or_, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB

from .filters import resolve

JSON_TYPES = ("string", "number", "boolean", "object", "array", "null")


class ConstraintViolation(Exception):
    """A write rejected by a constraint this library defined.

    Raised in place of the driver's IntegrityError so the message names
    the constraint, the table and the offending values, rather than a
    Postgres index name the caller never chose. `.constraint` is the
    constraint name, `.detail` the server's DETAIL line when there is
    one, and `.__cause__` the original driver error."""

    def __init__(self, message: str, constraint: Optional[str] = None,
                 detail: Optional[str] = None):
        super().__init__(message)
        self.constraint = constraint
        self.detail = detail


@dataclass(frozen=True)
class Col:
    """A real table column, as opposed to a JSONB property.

    Only needed where the two could be confused, which is exactly why it
    is explicit: Unique(Col("start_id"), "kind") constrains the column
    start_id and the property kind, and nothing has to guess."""
    name: str

    def __repr__(self) -> str:
        # Error messages quote the keys back so they can be pasted into a
        # define_constraints() call; the generated dataclass repr would
        # spell that Col(name='start_id').
        return f"Col({self.name!r})"


Key = Union[str, Col]


@dataclass
class Unique:
    """No two rows may share these values. Composite when given several
    keys; partial when given `where`."""
    keys: tuple
    where: Any = None
    name: Optional[str] = None

    def __init__(self, *keys: Key, where: Any = None, name: Optional[str] = None):
        if not keys:
            raise ValueError("Unique() needs at least one key")
        self.keys, self.where, self.name = keys, where, name


@dataclass
class Index:
    """A plain lookup index on one or more properties. Not integrity --
    speed, for a property you filter on constantly and don't want the GIN
    index's generality for."""
    keys: tuple
    where: Any = None
    name: Optional[str] = None

    def __init__(self, *keys: Key, where: Any = None, name: Optional[str] = None):
        if not keys:
            raise ValueError("Index() needs at least one key")
        self.keys, self.where, self.name = keys, where, name


@dataclass
class Required:
    """These properties must be present on every row. Says nothing about
    their value -- pair with PropertyType or Check for that."""
    keys: tuple
    name: Optional[str] = None

    def __init__(self, *keys: str, name: Optional[str] = None):
        if not keys:
            raise ValueError("Required() needs at least one key")
        if any(isinstance(k, Col) for k in keys):
            raise TypeError("Required() constrains JSONB properties; a real column's "
                            "presence is already NOT NULL in the table definition")
        self.keys, self.name = keys, name


@dataclass
class PropertyType:
    """The property, when present, must be of this JSON type.

    Worth having when models write your data: an LLM that emits "42"
    where you expected 42 breaks every numeric comparison downstream,
    silently and much later."""
    key: str
    json_type: str
    name: Optional[str] = None

    def __post_init__(self):
        if self.json_type not in JSON_TYPES:
            raise ValueError(f"json_type must be one of {JSON_TYPES}, got {self.json_type!r}")


@dataclass
class Check:
    """An arbitrary predicate, written in the same filter language as
    `where=` on a traversal -- so GT("age", 0), OR(...), NOT(...) and the
    callable escape hatch all work, and compile through the same
    resolve()."""
    filter: Any
    name: Optional[str] = None


CONSTRAINT_TYPES = (Unique, Index, Required, PropertyType, Check)


# ---------------------------------------------------------------------
# Compilation to DDL
# ---------------------------------------------------------------------

@dataclass
class _Target:
    """The table a set of constraints applies to, and the graph within it.

    `properties_col` is deliberately an UNBOUND column rather than
    table.c.properties: an unbound one renders as `properties`, a bound
    one as `nodes.properties`. Both work in DDL, but a qualified name is
    not accepted in an ON CONFLICT target -- and index inference only
    works when the conflict target is spelled exactly as the index was.
    Generating both from this one column is what keeps merge_nodes() and
    Unique() in step."""
    table: Any
    label: str
    graph: Optional[str] = None
    graph_col: str = "graph_id"
    properties_col: Any = field(init=False)

    def __post_init__(self):
        self.properties_col = sa_column("properties", JSONB)

    def scope_index(self, keys: tuple) -> tuple:
        """Put graph_id in front of an index's columns.

        One index serves every graph and uniqueness becomes per-graph for
        free: two graphs may each have their own node with the same
        email, which is the only sane reading of "these are separate
        graphs"."""
        if self.graph is None:
            return keys
        return (Col(self.graph_col), *keys)

    def scope_check(self, expression):
        """A CHECK applies to the whole table, so an unscoped one would
        make this graph's rules bind every other graph's rows too. The
        guard makes it vacuously true elsewhere."""
        if self.graph is None:
            return expression
        return or_(sa_column(self.graph_col) != self.graph, expression)

    def scope_name(self, name: str) -> str:
        """Two graphs declaring the same check need two constraints, so
        the generated name carries the graph."""
        if self.graph is None or self.graph == "default":
            return name
        return f"{name}_{_slug(self.graph)}"[:63]

    @property
    def qualified(self) -> str:
        if self.table.schema:
            return f'"{self.table.schema}"."{self.table.name}"'
        return f'"{self.table.name}"'


def _literal(expression) -> str:
    """Compile an expression with its values inlined.

    DDL cannot carry bind parameters, so constraint bodies are rendered
    literally. The values here come from constraint definitions written
    by a developer, not from ingested data, and SQLAlchemy still does the
    quoting -- ingestion itself never renders values into SQL."""
    return str(expression.compile(
        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
    ))


def _reject_column_collision(target: _Target, key: str) -> None:
    """A bare string that names a real column on this table can only be
    a mistake: the CHECK/index it would compile tests `properties->>
    '<key>'`, and that column is written and read by name already --
    directly by add_nodes()/add_edges(), or as an EXTRA COLUMN
    (models.py's "EXTENDING THE MODEL") -- never through `properties`.
    Refusing beats guessing, the same "Refuse, don't approximate" rule
    every other silently-different-meaning trap in this library follows."""
    if key in target.table.c:
        raise TypeError(
            f"{key!r} is a real column on {target.label} ({sorted(c.name for c in target.table.c)}), "
            f"not a JSONB property -- a bare string always means a property inside "
            f"`properties`, so this can only be a mistake. Say Col({key!r}) if you meant "
            f"the column (accepted by Unique/Index/merge's on=), or use a different "
            f"property name"
        )


def _key_expression(target: _Target, key: Key):
    if isinstance(key, Col):
        if key.name not in target.table.c:
            raise ValueError(
                f"{target.label} has no column {key.name!r} -- "
                f"columns are {sorted(c.name for c in target.table.c)}"
            )
        return sa_column(key.name)
    if not isinstance(key, str):
        raise TypeError(f"a constraint key must be a property name or Col(...), got {key!r}")
    _reject_column_collision(target, key)
    return target.properties_col[key].astext


def key_sql(target: _Target, key: Key) -> str:
    """One key as index-ready SQL: `start_id`, or `(properties ->> 'k')`.

    Both CREATE INDEX and ON CONFLICT go through here, so a conflict
    target can never drift from the index it needs to infer."""
    rendered = _literal(_key_expression(target, key))
    return rendered if isinstance(key, Col) else f"({rendered})"


def _slug(key: Key) -> str:
    name = key.name if isinstance(key, Col) else key
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def _auto_name(prefix: str, target: _Target, parts) -> str:
    """A deterministic name, so re-running define_constraints() is a
    no-op instead of a pile of duplicates. Truncated to Postgres's
    63-character identifier limit."""
    name = f"{prefix}_{target.table.name}_{'_'.join(_slug(p) for p in parts)}"
    return name[:63]


def compile_constraint(constraint: Any, target: _Target) -> tuple:
    """Return (kind, name, ddl) for one constraint.

    kind is 'index' or 'check' -- they are created and dropped by
    different statements. Exposed so `graph.constraint_ddl()` can show a
    caller exactly what will run before it runs."""
    if isinstance(constraint, (Unique, Index)):
        unique = isinstance(constraint, Unique)
        keys = target.scope_index(constraint.keys)
        name = constraint.name or _auto_name("uq" if unique else "ix", target, constraint.keys)
        columns = ", ".join(key_sql(target, k) for k in keys)
        ddl = (f'CREATE {"UNIQUE " if unique else ""}INDEX IF NOT EXISTS "{name}" '
               f'ON {target.qualified} ({columns})')
        if constraint.where is not None:
            ddl += f" WHERE ({_literal(resolve(target.properties_col, constraint.where))})"
        return "index", name, ddl

    if isinstance(constraint, Required):
        for key in constraint.keys:
            _reject_column_collision(target, key)
        name = constraint.name or target.scope_name(
            _auto_name("ck_required", target, constraint.keys))
        body = _literal(target.scope_check(
            target.properties_col.has_all(postgresql.array(constraint.keys))))
        return "check", name, _add_check(target, name, body)

    if isinstance(constraint, PropertyType):
        _reject_column_collision(target, constraint.key)
        name = constraint.name or target.scope_name(
            _auto_name(f"ck_{constraint.json_type}", target, [constraint.key]))
        expression = func.jsonb_typeof(target.properties_col[constraint.key]) == constraint.json_type
        return "check", name, _add_check(target, name, _literal(target.scope_check(expression)))

    if isinstance(constraint, Check):
        # No _reject_column_collision here: unlike Required/PropertyType,
        # a Check's filter is an arbitrary AND/OR/NOT/GT/... tree with a
        # callable escape hatch (see filters.py), so there is no fixed
        # set of "the keys" to check without re-deriving schema.py's own
        # _filter_vocabulary walk -- and that walk already treats a
        # callable as opaque for the same reason. Left to the author.
        if constraint.name is None:
            raise ValueError(
                "Check(...) needs an explicit name= -- a filter has no short, stable "
                "spelling to derive one from, and an unnamed constraint cannot be "
                "dropped or made idempotent"
            )
        name = target.scope_name(constraint.name)
        body = _literal(target.scope_check(resolve(target.properties_col, constraint.filter)))
        return "check", name, _add_check(target, name, body)

    raise TypeError(
        f"expected one of {', '.join(t.__name__ for t in CONSTRAINT_TYPES)}, "
        f"got {type(constraint).__name__}"
    )


def _add_check(target: _Target, name: str, body: str) -> str:
    return f'ALTER TABLE {target.qualified} ADD CONSTRAINT "{name}" CHECK ({body})'


_CHECK_EXISTS = text("""
    SELECT 1 FROM pg_constraint
    WHERE conname = :name AND conrelid = CAST(:table AS regclass)
""")


def constraint_exists(connection, target: _Target, name: str) -> bool:
    """CHECK constraints have no ADD ... IF NOT EXISTS in PostgreSQL, so
    idempotency is a lookup. (Indexes do, and use it.)"""
    return connection.execute(
        _CHECK_EXISTS, {"name": name, "table": target.qualified}
    ).first() is not None
