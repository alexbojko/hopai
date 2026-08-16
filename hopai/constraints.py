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

EVERY CONSTRAINT DECLARED HERE IS REAL SQLALCHEMY METADATA, not a hand-
built DDL string kept off to the side: compile_constraint() attaches an
actual `sqlalchemy.Index`/`CheckConstraint` to `graph.nodes_tbl`/
`edges_tbl` and derives its returned DDL from that same object. That is
what lets a project's own Alembic `--autogenerate` -- run against
`target_metadata` that includes these tables -- see hopai's indexes and
checks as declared schema instead of drift to propose dropping. It costs
nothing extra to call: `constraint_ddl()`'s preview and
`define_constraints()`'s apply both attach, so either one run once is
enough for the shape to be visible.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Union

from sqlalchemy import CheckConstraint as SACheckConstraint
from sqlalchemy import Index as SAIndex
from sqlalchemy import PrimaryKeyConstraint as SAPrimaryKeyConstraint
from sqlalchemy import UniqueConstraint as SAUniqueConstraint
from sqlalchemy import column as sa_column
from sqlalchemy import func, or_, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import AddConstraint, CreateIndex
from sqlalchemy.sql.elements import quoted_name

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
    """The table a set of constraints applies to, and the graph within it."""
    table: Any
    label: str
    graph: Optional[str] = None
    graph_col: str = "graph_id"

    @property
    def properties_col(self):
        """The properties column, BOUND to this target's table.

        Building it from table.c.properties -- rather than an unbound
        sa_column("properties", JSONB), which is what this used to be --
        is what lets compile_constraint() hand it to sqlalchemy.Index/
        CheckConstraint: both infer which table they belong to from a
        bound column inside their expression, and an unbound one carries
        no table to infer. The one place that still needs the UNBOUND
        rendering -- merge()'s ON CONFLICT target -- builds it directly
        in key_sql() rather than through this property, since a bound
        column renders table-qualified when compiled standalone and
        Postgres rejects a qualified conflict target."""
        return self.table.c.properties

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
        return or_(self.table.c[self.graph_col] != self.graph, expression)

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


def _validate_key(target: _Target, key: Key) -> None:
    if isinstance(key, Col):
        if key.name not in target.table.c:
            raise ValueError(
                f"{target.label} has no column {key.name!r} -- "
                f"columns are {sorted(c.name for c in target.table.c)}"
            )
        return
    if not isinstance(key, str):
        raise TypeError(f"a constraint key must be a property name or Col(...), got {key!r}")
    _reject_column_collision(target, key)


def _key_expression(target: _Target, key: Key):
    """This key as a BOUND expression, for building the actual
    sqlalchemy.Index/CheckConstraint objects compile_constraint()
    attaches to target.table. See key_sql() for the unbound counterpart
    ON CONFLICT needs."""
    _validate_key(target, key)
    if isinstance(key, Col):
        return target.table.c[key.name]
    return target.properties_col[key].astext


def key_sql(target: _Target, key: Key) -> str:
    """One key as UNQUALIFIED index-ready SQL: `start_id`, or
    `(properties ->> 'k')`.

    Both CREATE INDEX (through compile_constraint()) and ON CONFLICT go
    through here, so a conflict target can never drift from the index it
    needs to infer -- which is also why this builds its own unbound
    expression rather than reusing _key_expression()'s bound one: a bound
    column renders table-qualified (`nodes.properties`) when compiled
    standalone outside a DDL-construct context, and Postgres rejects a
    qualified ON CONFLICT target."""
    _validate_key(target, key)
    if isinstance(key, Col):
        return _literal(sa_column(key.name))
    return f"({_literal(sa_column('properties', JSONB)[key].astext)})"


def _slug(key: Key) -> str:
    name = key.name if isinstance(key, Col) else key
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def _auto_name(prefix: str, target: _Target, parts) -> str:
    """A deterministic name, so re-running define_constraints() is a
    no-op instead of a pile of duplicates. Truncated to Postgres's
    63-character identifier limit."""
    name = f"{prefix}_{target.table.name}_{'_'.join(_slug(p) for p in parts)}"
    return name[:63]


#: Marks an Index/CheckConstraint as attached by THIS module, in
#: `.info` -- the SchemaItem-standard place for a library to stash its
#: own bookkeeping without colliding with SQLAlchemy's own attributes.
#: Everything that hides, replaces or removes an object reads it first,
#: so a table's own structural constraints (its primary key, foreign
#: keys, and any Index/UniqueConstraint/CheckConstraint a project wrote
#: directly into a custom node_table=/edge_table=) are never touched.
_DECLARED = "hopai_declared"


def _container(table: Any, kind: str):
    return table.indexes if kind == "index" else table.constraints


def _declared(table: Any, kind: str, name: str):
    """The hopai-declared Index/CheckConstraint of this name, or None.

    OWNERSHIP IS CHECKED, NOT JUST THE NAME, and that is the whole point
    of this function. `table.constraints` also holds the table's primary
    key, its foreign keys, and anything a project put in its own
    node_table=/edge_table= -- and models.py names the composite foreign
    keys `fk_edges_start_same_graph`/`fk_edges_end_same_graph` in
    cleartext. Matching on name alone let
    `Check(..., name="fk_edges_start_same_graph")` silently REPLACE that
    foreign key with an ordinary CheckConstraint, on the module-level
    Edge singleton every default Graph() shares -- so a later
    create_schema() anywhere in the same process built `edges` without
    the constraint that makes a cross-graph edge impossible. It was
    reachable through constraint_ddl() alone, which is documented as
    showing the SQL without running anything.
    (test_a_declaration_cannot_replace_the_composite_foreign_key)"""
    wanted = SAIndex if kind == "index" else SACheckConstraint
    return next((obj for obj in _container(table, kind)
                 if getattr(obj, "name", None) == name
                 and isinstance(obj, wanted)
                 and getattr(obj, "info", {}).get(_DECLARED)), None)


def _foreign_holders(table: Any, kind: str):
    """Every object on `table` this library did not declare whose name
    an object of `kind` would collide with.

    An index collides with more than the other indexes: a PRIMARY KEY or
    UNIQUE constraint owns a backing index of its own name, so they share
    one namespace in Postgres. `models.py` names one -- uq_nodes_id_graph,
    the composite key the edges foreign key points at -- and
    `Unique(..., name="uq_nodes_id_graph")` therefore compiled a
    CREATE UNIQUE INDEX IF NOT EXISTS that Postgres skipped as
    already-present, leaving the declared rule silently unenforced.
    A CHECK owns no index, so it is not part of that namespace and is not
    listed here."""
    yield from (obj for obj in _container(table, kind)
                if not getattr(obj, "info", {}).get(_DECLARED))
    if kind == "index":
        yield from (obj for obj in table.constraints
                    if isinstance(obj, (SAPrimaryKeyConstraint, SAUniqueConstraint))
                    and not getattr(obj, "info", {}).get(_DECLARED))


def _reject_foreign_name(table: Any, kind: str, name: str) -> None:
    """Refuse a name already taken by an object this library did not
    declare, rather than shadowing it with a second one of the same name.

    The alternative to refusing is two same-named objects on one table --
    SQLAlchemy permits it, Alembic then sees competing definitions, and
    Postgres either rejects the second CREATE or, with the IF NOT EXISTS
    this library emits, skips it and leaves the rule unenforced. Naming
    the collision beats any of those. Silently REPLACING it is what
    _declared()'s docstring describes and is worse still."""
    clash = next((obj for obj in _foreign_holders(table, kind)
                  if getattr(obj, "name", None) == name), None)
    if clash is not None:
        raise ValueError(
            f"{name!r} is already the name of a {type(clash).__name__} on "
            f"{table.name!r} that hopai did not declare -- naming a constraint "
            f"after one of the table's own would shadow it. Pass a different "
            f"name= (every constraint takes one)"
        )


def detach_constraint(table: Any, kind: str, name: str) -> None:
    """Remove the hopai-declared Index/CheckConstraint of this name from
    `table`'s own SQLAlchemy metadata. A name this library did not
    declare is left alone (see _declared).

    _attach_index()/_attach_check() call this before constructing a
    replacement: SQLAlchemy does not dedupe two same-named objects
    attached to one table, so constructing without detaching first
    leaves both sitting in `table.indexes`/`table.constraints` -- and a
    tool reading that metadata (Alembic included) then sees two
    competing definitions for one name. drop_constraints() and
    enforce_schema()'s reconciliation call it too, so an object dropped
    from the database does not linger as a phantom in Python metadata.

    Takes the table directly, not a _Target: nothing here needs a
    graph's scoping rules, only which table's metadata to touch --
    create_schema()'s baseline indexes and vectors.py's dimension checks
    use it too, and neither one has a constraint _Target of its own."""
    match = _declared(table, kind, name)
    if match is not None:
        _container(table, kind).discard(match)


def _attach_index(table: Any, name: str, expressions, *, unique: bool = False,
                  where=None, **options) -> SAIndex:
    """A real sqlalchemy.Index, attached to `table`.

    Built from BOUND expressions (see _key_expression), which is what
    lets Index infer `table` and self-register on construction -- the
    same mechanism that makes it visible to Alembic --autogenerate.
    `**options` passes through dialect-specific kwargs like
    postgresql_using="gin"."""
    _reject_foreign_name(table, "index", name)
    detach_constraint(table, "index", name)
    if where is not None:
        options["postgresql_where"] = where
    idx = SAIndex(quoted_name(name, True), *expressions, unique=unique, **options)
    idx.info[_DECLARED] = True
    return idx


def _attach_check(table: Any, name: str, expression) -> SACheckConstraint:
    """A real sqlalchemy.CheckConstraint, attached to `table`."""
    _reject_foreign_name(table, "check", name)
    detach_constraint(table, "check", name)
    ck = SACheckConstraint(expression, name=quoted_name(name, True))
    ck.info[_DECLARED] = True
    table.append_constraint(ck)
    return ck


@contextmanager
def suspended_declarations(*tables: Any):
    """Temporarily detach every Index/CheckConstraint _attach_index()/
    _attach_check() put on `tables`, restoring them on exit.

    Table.create() (and the CreateTable DDL it emits) renders EVERY
    index/constraint currently attached to a table, not just the ones
    native to its own definition. create_schema()'s contract is "the
    table's own structure, plus its five baseline indexes" -- so if
    this process already called define_constraints()/enforce_schema()/
    migrate_vectors() on this table (on this Graph, or on another one
    sharing the default Node/Edge table) before create_schema() ever
    ran, those declarations must not get baked into the CREATE TABLE/
    CREATE INDEX statements create_schema() is about to issue. A
    table's own structural constraints -- its primary key, foreign
    keys, and anything a project wrote directly into a custom
    node_table=/edge_table= -- are untouched, since only objects
    carrying _DECLARED in `.info` (see _attach_index/_attach_check) are
    ever hidden."""
    hidden = [
        ([ix for ix in table.indexes if getattr(ix, "info", {}).get(_DECLARED)],
         [c for c in table.constraints if getattr(c, "info", {}).get(_DECLARED)])
        for table in tables
    ]
    for table, (indexes, checks) in zip(tables, hidden, strict=True):
        for ix in indexes:
            table.indexes.discard(ix)
        for ck in checks:
            table.constraints.discard(ck)
    try:
        yield
    finally:
        for table, (indexes, checks) in zip(tables, hidden, strict=True):
            table.indexes.update(indexes)
            table.constraints.update(checks)


def _compile_check(table: Any, name: str, expression) -> str:
    """Attach `expression` (already scoped) as a CheckConstraint named
    `name`, and return its ADD CONSTRAINT DDL.

    The one seam every CHECK this library emits goes through --
    compile_constraint()'s Required/PropertyType/Check branches, and
    schema.py's per-type presence/type/value rules -- so a CHECK is never
    hand-built DDL invisible to SQLAlchemy's own metadata."""
    ck = _attach_check(table, name, expression)
    return str(AddConstraint(ck).compile(
        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


def compile_constraint(constraint: Any, target: _Target) -> tuple:
    """Return (kind, name, ddl) for one constraint -- attaching it to
    target.table's SQLAlchemy metadata as a real Index/CheckConstraint
    along the way, so a tool reading target_metadata (Alembic's
    --autogenerate, first and foremost) sees exactly what this call
    declares rather than proposing to drop it. Re-declaring under the
    same name replaces the prior object rather than accumulating a
    duplicate (see detach_constraint), which is also why calling this
    more than once -- constraint_ddl() previewing, then
    define_constraints() applying -- always reflects the CURRENT
    declaration rather than whichever one ran first.

    kind is 'index' or 'check' -- they are created and dropped by
    different statements. Exposed so `graph.constraint_ddl()` can show a
    caller exactly what will run before it runs."""
    if isinstance(constraint, (Unique, Index)):
        unique = isinstance(constraint, Unique)
        keys = target.scope_index(constraint.keys)
        name = constraint.name or _auto_name("uq" if unique else "ix", target, constraint.keys)
        expressions = [_key_expression(target, key) for key in keys]
        where = (resolve(target.properties_col, constraint.where)
                 if constraint.where is not None else None)
        idx = _attach_index(target.table, name, expressions, unique=unique, where=where)
        ddl = str(CreateIndex(idx, if_not_exists=True).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
        return "index", name, ddl

    if isinstance(constraint, Required):
        for key in constraint.keys:
            _reject_column_collision(target, key)
        name = constraint.name or target.scope_name(
            _auto_name("ck_required", target, constraint.keys))
        expression = target.scope_check(
            target.properties_col.has_all(postgresql.array(constraint.keys)))
        return "check", name, _compile_check(target.table, name, expression)

    if isinstance(constraint, PropertyType):
        _reject_column_collision(target, constraint.key)
        name = constraint.name or target.scope_name(
            _auto_name(f"ck_{constraint.json_type}", target, [constraint.key]))
        expression = target.scope_check(
            func.jsonb_typeof(target.properties_col[constraint.key]) == constraint.json_type)
        return "check", name, _compile_check(target.table, name, expression)

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
        expression = target.scope_check(resolve(target.properties_col, constraint.filter))
        return "check", name, _compile_check(target.table, name, expression)

    raise TypeError(
        f"expected one of {', '.join(t.__name__ for t in CONSTRAINT_TYPES)}, "
        f"got {type(constraint).__name__}"
    )


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
