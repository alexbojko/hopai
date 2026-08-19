"""
hopai.mutate

Changing and removing what is already in the graph -- the counterpart to
ingest.py, and the other half of a knowledge graph an agent maintains
rather than only fills.

    graph.update_nodes(where={"type": "person"}, set={"active": False})
    graph.update_nodes(where={"email": "a@x.com"}, remove=["nickname"])
    graph.delete_edges(where={"kind": "knows"}, start={"name": "Alice"})
    graph.delete_nodes(where={"email": "a@x.com"}, detach=True)

...or the same thing in Cypher, which is what a model reaches for:

    graph.cypher("MATCH (a:person) SET a.active = false")
    graph.cypher("MATCH (a {email: 'a@x.com'}) REMOVE a.nickname")
    graph.cypher("MATCH (a {name: 'Alice'})-[r:knows]->() DELETE r")
    graph.cypher("MATCH (a {email: 'a@x.com'}) DETACH DELETE a")

`where` IS THE TRAVERSAL FILTER LANGUAGE. Same hopai.filters.resolve(),
so OR/NOT/GT/BETWEEN and the JSON operator forms all work, and a filter
that selects rows for a traversal selects the same rows here. It matches
a SET of rows, not one: an update or a delete changes every row it
matches, exactly as Cypher's does.

NO FILTER IS A REFUSAL, NOT "EVERYTHING". `where=None` and `where={}`
raise instead of matching the whole graph, because both are what a
caller's empty variable looks like, and the cost of being wrong is the
data. Saying it on purpose is `all=True` -- or `graph.clear()`, which is
the same thing with a name you cannot type by accident.

TARGETING A ROW BY ITS ID, not just its properties: `ids=[...]` selects
rows by the real id column, ANDed with `where` when both are given.
`where` compiles through hopai.filters.resolve() against `properties`,
and an id is not a property -- `where={"id": 7}` is a JSONB containment
test that matches nothing and says nothing while doing it. `ids=` is
the deliberate second parameter that closes that trap, the same way
Start(ids=...) does for a traversal's seed set (hopai/hop.py). An empty
`ids=[]` is treated exactly like an empty `where`: it is what a
caller's empty variable looks like, so it counts toward the no-filter
refusal above rather than silently matching nothing (contrast
Start(ids=[]), a read with no such danger, where an explicit empty
selection matches nothing instead).

DELETING A NODE THAT STILL HAS EDGES fails on the foreign key, and that
failure is translated into a message naming `detach=True` (Cypher's
DETACH DELETE, which deletes the incident edges with the node). The
error is worth keeping rather than cascading silently: an edge pointing
at a node that no longer exists is the corruption this schema's
composite foreign key was added to make impossible.

UPDATE SEMANTICS -- three of them, one per Cypher spelling, because
"update the properties" means three different things:

    set={...}                    merge over what is there (JSONB `||`),
                                 leaving unmentioned keys alone
                                 -- SET a.x = 1, SET a += {...}
    set={...}, replace=True      the new bag IS the properties
                                 -- SET a = {...}
    remove=["x"]                 drop those keys
                                 -- REMOVE a.x

A tool-calling model drives the same operations through JSON, one
document, one transaction -- see MUTATE_TOOL_SCHEMA:

    graph.mutate({"operations": [
        {"op": "update_nodes", "where": {"type": "person"}, "set": {"active": False}},
        {"op": "delete_nodes", "where": {"type": "draft"}, "detach": True},
    ]})

SEEING THE SQL BEFORE IT RUNS, the same way build_query() is inspected,
with no database and nothing executed:

    from sqlalchemy.dialects import postgresql
    from hopai.mutate import Mutator

    print(Mutator(graph).delete_nodes_statement({"type": "draft"})
          .compile(dialect=postgresql.dialect()))

ONE THING THIS DOES NOT RE-CHECK: `enforce_schema(endpoints=True)`
polices an edge's endpoint types with a trigger on the EDGES table, so
changing a NODE's type here (or with merge_nodes(replace=True), which
could already do it) can leave a declared edge connecting types the
schema forbids, and nothing raises. `graph.schema_violations()` does not
see it either -- it reads the same declarations the trigger does.
Re-run enforce_schema() after retyping nodes, or check the edges you
know point at them.
"""

from __future__ import annotations

import json as _json
import time
from dataclasses import dataclass

from sqlalchemy import ARRAY, Text, cast, delete, literal, or_, select, update
from sqlalchemy.dialects.postgresql import JSONB, array
from sqlalchemy.exc import IntegrityError

from .constraints import ConstraintViolation
# The same coercion the vector writes use: results carry ids as
# strings, and a caller handing one straight back should not have to
# remember to int() it.
from .vectors import _coerce_id
from .filters import parse_filter, resolve
from .ingest import constraint_violation, one_transaction

# `set` and `all` are parameter names throughout this module on purpose:
# they are what Cypher calls these things and what the JSON schema
# spells, and an argument a model has to translate is an argument it
# gets wrong. The two builtins they shadow are aliased here so the code
# below can still reach them.
all_ = all
set_ = set

#: What each operation accepts. Both front ends -- the JSON document and
#: the Cypher translator -- emit these and nothing else, so there is one
#: executor to trust rather than one per notation.
_OP_KEYS = {
    "delete_nodes": {"where", "detach", "all", "ids"},
    "delete_edges": {"where", "start", "end", "all", "ids"},
    "update_nodes": {"where", "set", "remove", "replace", "all", "ids"},
    "update_edges": {"where", "start", "end", "set", "remove", "replace", "all", "ids"},
}
MUTATION_OPS = frozenset(_OP_KEYS)

#: Operation keys holding a filter, which the JSON front end parses and
#: the Python API is handed already built.
_FILTER_KEYS = ("where", "start", "end")


@dataclass
class MutationResult:
    """What a delete or update did.

    Counts are rows MATCHED, not rows whose properties ended up
    different -- an update setting a property to the value it already
    had still counts, the same way IngestResult counts a no-op upsert.

    Four counters rather than one number because a single delete touches
    both tables: `detach=True` removes a node's edges with it, and
    hiding that in one total would make "3" mean three of something
    unstated."""
    deleted_nodes: int = 0
    deleted_edges: int = 0
    updated_nodes: int = 0
    updated_edges: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {"deleted_nodes": self.deleted_nodes, "deleted_edges": self.deleted_edges,
                "updated_nodes": self.updated_nodes, "updated_edges": self.updated_edges,
                "elapsed_ms": self.elapsed_ms}

    def __repr__(self) -> str:
        return (f"MutationResult(deleted_nodes={self.deleted_nodes}, "
                f"deleted_edges={self.deleted_edges}, updated_nodes={self.updated_nodes}, "
                f"updated_edges={self.updated_edges}, elapsed_ms={self.elapsed_ms:.1f})")


def _flag(value, call: str, name: str) -> bool:
    """A boolean argument that decides whether rows are destroyed is
    checked, never coerced.

    `all="false"` is truthy in Python, and JSON booleans arriving as the
    strings "true"/"false" is an ordinary tool-call failure -- so
    coercing would let the string "false" mean "yes, every row", which
    is the exact inversion of the refusal this library is built
    around."""
    if value is not True and value is not False:
        raise TypeError(
            f"{call}({name}=...) takes True or False, got {value!r}. Anything else is a "
            f"guess about what you meant, and this argument decides how many rows change"
        )
    return value


def _detach_hint(exc: IntegrityError) -> ConstraintViolation:
    """The foreign-key refusal, rewritten as the fix.

    Postgres says "still referenced from table edges", which is true and
    useless: the caller has to know that edges reference nodes and that
    the flag is spelled `detach`. Cypher's own error for this is the
    thing to imitate -- it names DETACH DELETE."""
    violation = constraint_violation(exc, "node")
    # The fix first, the driver's text last. Postgres leads with "Key
    # (id, graph_id)=(1, default) is still referenced from table edges",
    # and two clauses of that before the answer is two clauses too many
    # -- but it names the constraint and the row, so it is kept.
    #
    # No __cause__ set here: the caller raises this `from exc`, which is
    # what chains the driver's error onto it. Assigning it as well was
    # dead code, and mutation testing is what noticed -- nothing could
    # observe the difference.
    return ConstraintViolation(
        f"cannot delete a node that still has edges: deleting it would leave edges "
        f"pointing at nothing. Pass detach=True (Cypher: DETACH DELETE) to delete those "
        f"edges with the node, or delete the edges first -- {violation}",
        constraint=violation.constraint, detail=violation.detail,
    )


class Mutator:
    """Implements Graph's delete and update methods.

    Every statement it builds is available without executing it -- the
    `*_statement` methods take no connection and touch no database, so
    the emitted SQL can be inspected (and asserted on in tests) with
    nothing running, exactly like build_query()."""

    def __init__(self, graph):
        self.g = graph

    # -- targeting ------------------------------------------------------

    @staticmethod
    def _blank(filt) -> bool:
        """Whether a filter constrains nothing. `{}` is not "no filter
        given" to resolve() -- it compiles to TRUE, which is precisely
        the accident this returns True for."""
        return filt is None or (isinstance(filt, (dict, list, tuple)) and not filt)

    def _guard(self, filters: list, call: str, what: str, all: bool) -> None:
        """The unfiltered-mutation refusal, in one place so every entry
        point -- Python, JSON and Cypher -- obeys the same rule.

        Both directions are an error. No filter without all=True is the
        accident this exists for; all=True *with* a filter is a caller
        who believes one of the two is being honoured, and only one of
        them is."""
        _flag(all, call, "all")
        unfiltered = all_(self._blank(f) for f in filters)
        if unfiltered and not all:
            raise ValueError(
                f"{call}() was given no filter, which would match every {what} "
                f"{self._scope_phrase()}. Filter it with where=..., pass all=True to say "
                f"you really mean every {what}, or call graph.clear() to empty the graph"
            )
        if all and not unfiltered:
            raise ValueError(
                f"{call}(all=True) also got a filter, and the two disagree about what to "
                f"touch. Drop whichever one you did not mean"
            )

    def _scope_phrase(self) -> str:
        """Where "every row" reaches. Naming a graph would be a lie on
        tables that have no discriminator column."""
        if self.g.graph_col is None:
            return "in these tables"
        return f"in graph {self.g.graph!r}"

    def _scope(self, table) -> list:
        """The graph predicate, as the list every statement starts from.

        Empty for `graph_col=None` tables rather than `literal(True)`:
        `WHERE true AND ...` is noise a reader has to look at twice, and
        it is the whole predicate when the filter is empty too."""
        return [] if self.g.graph_col is None else [self.g._scoped(table)]

    def _node_conditions(self, where, ids=None) -> list:
        """The node rows a mutation targets: the graph, then the filter.

        A blank filter is left out rather than resolved to TRUE -- it is
        only ever reached with all=True, and `WHERE graph_id = 'x' AND
        true` is a planner input as well as something a reader has to
        look at twice."""
        nt = self.g.nodes_tbl
        conditions = self._scope(nt)
        if not self._blank(where):
            conditions.append(resolve(nt.c.properties, where))
        if not self._blank(ids):
            # The one way to address a SPECIFIC row. `where` filters
            # properties and an id is not a property -- where={"id": 7}
            # matches nothing and says nothing, which is the trap this
            # closes for any caller holding ids from a traversal result.
            conditions.append(getattr(nt.c, self.g.node_id_col).in_(
                [_coerce_id(one) for one in ids]))
        return conditions

    def _edge_conditions(self, where, start, end, ids=None) -> list:
        """The edge rows a mutation targets: its own properties, plus
        optional filters on the nodes at either end.

        Endpoint filters are what makes `MATCH (a {name: 'Alice'})-[r:knows]->()
        DELETE r` translate exactly -- without them "Alice's knows edges"
        would have to be fetched and deleted by id, which is two round
        trips and a race between them."""
        et = self.g.edges_tbl
        conditions = self._scope(et)
        if not self._blank(where):
            conditions.append(resolve(et.c.properties, where))
        if not self._blank(ids):
            conditions.append(getattr(et.c, self.g.edge_id_col).in_(
                [_coerce_id(one) for one in ids]))
        for column, filt in ((self.g.edge_start_col, start), (self.g.edge_end_col, end)):
            if not self._blank(filt):
                conditions.append(getattr(et.c, column).in_(self._node_ids(filt)))
        return conditions

    def _node_ids(self, filt, ids=None):
        """The ids of this graph's nodes matching `filt`, as a subquery.
        Scoped like every other read: an unscoped one would let an
        endpoint filter match a node belonging to another graph."""
        nt = self.g.nodes_tbl
        return select(getattr(nt.c, self.g.node_id_col)).where(
            *self._node_conditions(filt, ids))

    # -- statements -----------------------------------------------------

    def delete_nodes_statement(self, where=None, all: bool = False, ids=None):
        self._guard([where, ids], "delete_nodes", "node", all)
        return delete(self.g.nodes_tbl).where(*self._node_conditions(where, ids))

    def detach_statement(self, where=None, all: bool = False, ids=None):
        """Every edge incident to the nodes `where` matches, in either
        direction. Run before the node delete and in the same
        transaction, so no caller of this library can insert an edge
        into the gap and fail the node delete with the error detach was
        supposed to prevent. It narrows that window rather than closing
        it: under READ COMMITTED a concurrent session can still commit
        an edge that the node delete then trips over."""
        self._guard([where, ids], "delete_nodes", "node", all)
        et = self.g.edges_tbl
        matched = self._node_ids(where, ids)
        return delete(et).where(*self._scope(et), or_(
            getattr(et.c, self.g.edge_start_col).in_(matched),
            getattr(et.c, self.g.edge_end_col).in_(matched),
        ))

    def delete_edges_statement(self, where=None, start=None, end=None, all: bool = False,
                               ids=None):
        self._guard([where, start, end, ids], "delete_edges", "edge", all)
        return delete(self.g.edges_tbl).where(
            *self._edge_conditions(where, start, end, ids))

    def update_nodes_statement(self, where=None, set=None, remove=None,
                               replace: bool = False, all: bool = False, ids=None):
        self._guard([where, ids], "update_nodes", "node", all)
        nt = self.g.nodes_tbl
        return update(nt).where(*self._node_conditions(where, ids)).values(
            properties=_new_properties(nt.c.properties, set, remove, replace, "update_nodes"))

    def update_edges_statement(self, where=None, start=None, end=None, set=None,
                               remove=None, replace: bool = False, all: bool = False,
                               ids=None):
        self._guard([where, start, end, ids], "update_edges", "edge", all)
        et = self.g.edges_tbl
        return update(et).where(*self._edge_conditions(where, start, end, ids)).values(
            properties=_new_properties(et.c.properties, set, remove, replace, "update_edges"))

    # -- execution ------------------------------------------------------

    def delete_nodes(self, where=None, detach: bool = False, all: bool = False,
                     connection=None, ids=None) -> MutationResult:
        started = time.perf_counter()
        _flag(detach, "delete_nodes", "detach")
        # The statements are built before the transaction opens: a
        # refusal (no filter, contradictory arguments) should raise
        # without having taken a connection out of the pool.
        detach_statement = self.detach_statement(where, all, ids) if detach else None
        statement = self.delete_nodes_statement(where, all, ids)
        with one_transaction(self.g, connection) as conn:
            edges = conn.execute(detach_statement).rowcount if detach else 0
            try:
                nodes = conn.execute(statement).rowcount
            except IntegrityError as exc:
                raise _detach_hint(exc) from exc
        return MutationResult(deleted_nodes=nodes, deleted_edges=edges,
                              elapsed_ms=(time.perf_counter() - started) * 1000)

    def repoint_edge(self, edge_id, start_id=None, end_id=None,
                     connection=None) -> MutationResult:
        """Move one edge to different endpoints, by id.

        Not expressible as an update: `set=` writes PROPERTIES, and an
        edge's endpoints are real columns -- which is the point of them.
        One UPDATE rather than a delete and an insert, so the edge keeps
        its id and its properties and there is no window where it does
        not exist.

        Nothing checks that the new endpoints are nodes of this graph,
        because the composite foreign key already does: an edge can only
        ever join two nodes of its own graph, and Postgres refuses the
        write rather than this code remembering to look."""
        started = time.perf_counter()
        statement = self.repoint_edge_statement(edge_id, start_id, end_id)
        with one_transaction(self.g, connection) as conn:
            try:
                moved = conn.execute(statement).rowcount
            except IntegrityError as exc:
                raise ValueError(
                    f"repoint_edge({edge_id!r}): no node with that id in graph "
                    f"{self.g.graph!r} -- an edge cannot leave its own graph, and the "
                    f"foreign key refused it ({exc.orig})") from exc
        return MutationResult(updated_edges=moved,
                              elapsed_ms=(time.perf_counter() - started) * 1000)

    def repoint_edge_statement(self, edge_id, start_id=None, end_id=None):
        """The UPDATE, with no connection -- so the graph discriminator
        can be asserted on with nothing running, the same way every
        other statement in this class is."""
        if start_id is None and end_id is None:
            raise ValueError(
                "repoint_edge() was given neither start_id nor end_id, so there is "
                "nothing to move -- name at least one endpoint")
        values = {}
        if start_id is not None:
            values[self.g.edge_start_col] = _coerce_id(start_id)
        if end_id is not None:
            values[self.g.edge_end_col] = _coerce_id(end_id)
        return (update(self.g.edges_tbl)
                .where(*self._edge_conditions(None, None, None, [edge_id]))
                .values(**values))

    def delete_edges(self, where=None, start=None, end=None, all: bool = False,
                     connection=None, ids=None) -> MutationResult:
        started = time.perf_counter()
        statement = self.delete_edges_statement(where, start, end, all, ids)
        with one_transaction(self.g, connection) as conn:
            deleted = conn.execute(statement).rowcount
        return MutationResult(deleted_edges=deleted,
                              elapsed_ms=(time.perf_counter() - started) * 1000)

    def update_nodes(self, where=None, set=None, remove=None, replace: bool = False,
                     all: bool = False, connection=None, ids=None) -> MutationResult:
        started = time.perf_counter()
        statement = self.update_nodes_statement(where, set, remove, replace, all, ids)
        with one_transaction(self.g, connection) as conn:
            try:
                updated = conn.execute(statement).rowcount
            except IntegrityError as exc:
                # An update is as capable of violating Required/Unique/
                # PropertyType as an insert, and the caller declared
                # those -- so it gets the same named ConstraintViolation
                # rather than a driver traceback.
                raise constraint_violation(exc, "node") from exc
        return MutationResult(updated_nodes=updated,
                              elapsed_ms=(time.perf_counter() - started) * 1000)

    def update_edges(self, where=None, start=None, end=None, set=None, remove=None,
                     replace: bool = False, all: bool = False,
                     connection=None, ids=None) -> MutationResult:
        started = time.perf_counter()
        statement = self.update_edges_statement(where, start, end, set, remove, replace, all, ids)
        with one_transaction(self.g, connection) as conn:
            try:
                updated = conn.execute(statement).rowcount
            except IntegrityError as exc:
                raise constraint_violation(exc, "edge") from exc
        return MutationResult(updated_edges=updated,
                              elapsed_ms=(time.perf_counter() - started) * 1000)

    def clear(self, connection=None) -> MutationResult:
        """Empty this graph: every edge, then every node, in one
        transaction. Other graphs in the same tables are untouched --
        this is `DELETE FROM ... WHERE graph_id = ...`, not TRUNCATE.

        Spelled out rather than delegating to delete_nodes(all=True,
        detach=True) because the subquery that one needs to find
        incident edges is exactly the work "all of them" makes
        unnecessary."""
        started = time.perf_counter()
        nt, et = self.g.nodes_tbl, self.g.edges_tbl
        with one_transaction(self.g, connection) as conn:
            edges = conn.execute(delete(et).where(*self._scope(et))).rowcount
            nodes = conn.execute(delete(nt).where(*self._scope(nt))).rowcount
        return MutationResult(deleted_nodes=nodes, deleted_edges=edges,
                              elapsed_ms=(time.perf_counter() - started) * 1000)

    def execute_operations(self, operations: list, connection=None) -> MutationResult:
        """Run a plan from spec_to_mutations() or cypher_to_mutations().

        One transaction for the whole plan, for the same reason a
        document's writes share one: a half-applied plan leaves the
        graph in a state neither the caller nor a retry can reason
        about."""
        started = time.perf_counter()
        total = MutationResult()
        methods = {"delete_nodes": self.delete_nodes, "delete_edges": self.delete_edges,
                   "update_nodes": self.update_nodes, "update_edges": self.update_edges}
        with one_transaction(self.g, connection) as conn:
            for operation in operations:
                kind = operation.get("op")
                if kind not in methods:
                    raise ValueError(
                        f"unknown operation {kind!r} -- a mutation plan is made of "
                        f"{', '.join(sorted(MUTATION_OPS))}"
                    )
                arguments = {k: v for k, v in operation.items() if k != "op"}
                result = methods[kind](connection=conn, **arguments)
                total.deleted_nodes += result.deleted_nodes
                total.deleted_edges += result.deleted_edges
                total.updated_nodes += result.updated_nodes
                total.updated_edges += result.updated_edges
        total.elapsed_ms = (time.perf_counter() - started) * 1000
        return total


def _new_properties(column, set, remove, replace: bool, call: str):
    """The properties expression an UPDATE assigns.

    Merge, replace and remove are three different intentions and the
    contradictory combinations raise instead of resolving to a silent
    winner -- an update that quietly does less than it was told is the
    kind of thing nobody notices until the data is wrong."""
    if set is not None and not isinstance(set, dict):
        raise TypeError(f"{call}(set=...) must be a dict of properties, "
                        f"got {type(set).__name__}")
    if remove is not None and isinstance(remove, str):
        raise TypeError(f"{call}(remove=...) takes a list of property names, not one "
                        f"string -- write remove=[{remove!r}]")
    remove = list(remove) if remove else []
    if bad := [k for k in remove if not isinstance(k, str)]:
        raise TypeError(f"{call}(remove=...) takes property names, got {bad!r}")

    # The replace checks come FIRST, and diagnose remove= before an
    # absent set=. Ordered the other way, replace=True with remove=[...]
    # fell through to "nothing to change -- ... or use remove=[...]",
    # which told the caller to do the thing they had just done.
    _flag(replace, call, "replace")
    if replace:
        if remove:
            raise ValueError(
                f"{call}(replace=True, remove=[...]) contradicts itself: replace already "
                f"decides the whole property bag, so removing a key from it means "
                f"leaving that key out of set="
            )
        if set is None:
            raise ValueError(
                f"{call}(replace=True) with no set= would erase every property of every "
                f"matched row. Pass the properties it should end up with -- set={{}} if "
                f"that really is an empty bag -- or use remove=[...] to drop keys"
            )
    elif not set and not remove:
        # `set is None` and `set={}` differ only here: an explicitly
        # empty bag means something with replace=True (Cypher's
        # `SET a = {}`, which clears every property) and nothing at all
        # without it.
        raise ValueError(
            f"{call}() was given nothing to change -- pass set={{...}} to write "
            f"properties, remove=[...] to drop them, or both"
        )
    overlap = sorted(set_(set or {}) & set_(remove))
    if overlap:
        raise ValueError(
            f"{call}() both sets and removes {overlap} -- pick one per property"
        )

    value = column
    if set is not None and (set or replace):
        try:
            # allow_nan=False, or json.dumps happily emits the non-JSON
            # tokens NaN and Infinity -- and the refusal below never
            # fires, leaving the caller a driver error raised after the
            # transaction opened, mid-plan.
            incoming = cast(literal(_json.dumps(set, allow_nan=False)), JSONB)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{call}(set=...) values must be JSON: {exc}") from exc
        value = incoming if replace else value.op("||")(incoming)
    if remove:
        # jsonb - text[] drops every listed key in one operation. The
        # cast is not optional: an untyped ARRAY[...] leaves Postgres
        # unable to tell `jsonb - text[]` from `jsonb - text`, and it
        # picks the wrong one for a single key.
        value = value.op("-")(cast(array(remove), ARRAY(Text)))
    return value


# ---------------------------------------------------------------------
# The JSON front end
# ---------------------------------------------------------------------

def spec_to_mutations(document: dict) -> list:
    """Convert a JSON mutation document into the operation list
    Mutator.execute_operations() runs -- exposed on its own so a plan
    can be reviewed, logged or shown to a caller before it changes
    anything.

        spec_to_mutations({"operations": [
            {"op": "delete_nodes", "where": {"type": "draft"}, "detach": True},
        ]})
    """
    if not isinstance(document, dict):
        raise TypeError(f"a mutation takes a dict with an 'operations' list, "
                        f"got {type(document).__name__}")
    unknown = document.keys() - {"operations"}
    if unknown:
        raise ValueError(f"unknown keys {sorted(unknown)} -- a mutation document has "
                         f"'operations', nothing else")
    operations = document.get("operations")
    if not operations:
        raise ValueError('a mutation document needs a non-empty "operations" list, e.g. '
                         '{"operations": [{"op": "delete_nodes", "where": {...}}]}')

    plan = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise TypeError(f"each operation must be an object, got {operation!r}")
        kind = operation.get("op")
        if kind not in _OP_KEYS:
            raise ValueError(f"unknown operation {kind!r} -- use one of "
                             f"{', '.join(sorted(_OP_KEYS))}")
        stray = operation.keys() - _OP_KEYS[kind] - {"op"}
        if stray:
            raise ValueError(f"{kind} does not take {sorted(stray)} -- it takes "
                             f"{sorted(_OP_KEYS[kind])}")
        translated = {"op": kind}
        for key, value in operation.items():
            if key == "op":
                continue
            # Filters arrive in the JSON grammar and become the same
            # objects the Python API passes, so both notations meet at
            # resolve() rather than each growing their own compiler.
            translated[key] = parse_filter(value) if key in _FILTER_KEYS else value
        plan.append(translated)
    return plan


#: The filter grammar, spelled out rather than referenced. A caller may
#: be handed mutate_graph and nothing else (the README recommends
#: exactly that), so "same grammar as traverse_graph" would be a
#: dangling pointer -- and a model that cannot express "older than 65"
#: reports zero rows changed and reaches for the one lever left, which
#: is the one that empties the graph.
_FILTER_GRAMMAR = (
    'An object is equality, ANDed across its keys: {"type": "person"}. Operators are '
    'objects too: {"and": [...]}, {"or": [...]}, {"not": {...}}, {"gt": [key, value]}, '
    '{"gte": [...]}, {"lt": [...]}, {"lte": [...]}, {"between": [key, lo, hi]}. Any other '
    'operator spelling is read as an equality test against a nested object and matches '
    'nothing.'
)

_ENDPOINT = (
    'Any number of nodes may match this, and every edge touching one of them is affected '
    '-- unlike ingest_graph, where start/end must identify exactly one node. '
)

#: One description per argument, shared by the operations that take it.
_ARGUMENT_SCHEMA = {
    "where": {"type": "object",
              "description": "Filter on the properties of the rows to change. "
                             + _FILTER_GRAMMAR},
    "ids": {"type": "array",
            "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            "description": "Target specific rows by their id column directly -- node ids "
                           "for delete_nodes/update_nodes, edge ids for "
                           "delete_edges/update_edges. `where` filters PROPERTIES, and an "
                           "id is not one: where={\"id\": 7} matches nothing. Combines "
                           "with `where` as AND, so both must match. An empty list is "
                           "what an empty variable looks like, same as an empty `where` -- "
                           "it raises unless all=True says every row is really meant."},
    "start": {"type": "object",
              "description": "Filter on the properties of the node the edge starts at. "
                             + _ENDPOINT + _FILTER_GRAMMAR},
    "end": {"type": "object",
            "description": "Filter on the properties of the node the edge ends at. "
                           + _ENDPOINT + _FILTER_GRAMMAR},
    "set": {"type": "object",
            "description": "Properties to write. Merged over the existing ones, leaving "
                           "properties not mentioned here alone."},
    "remove": {"type": "array", "items": {"type": "string"},
               "description": "Property names to drop. Removing a name that is not there "
                              "is not an error."},
    "replace": {"type": "boolean",
                "description": "If true, 'set' becomes the whole property bag and every "
                               "property not in it is dropped."},
    "detach": {"type": "boolean",
               "description": "Also delete the edges attached to the deleted nodes. "
                              "Without it, deleting a node that still has edges fails."},
    "all": {"type": "boolean",
            "description": "Apply to every row, with no filter at all. Must be said "
                           "explicitly: an operation with neither a filter nor this flag "
                           "raises, and so does one with both, since then one of the two "
                           "is being ignored. Deleting is not reversible."},
}


def _operation_schema(op: str) -> dict:
    """One branch of the document schema, built from the same _OP_KEYS
    the parser validates against -- so the schema cannot advertise an
    argument the parser rejects, or omit one it requires."""
    return {
        "type": "object",
        "properties": {"op": {"const": op}, **{key: _ARGUMENT_SCHEMA[key]
                                               for key in sorted(_OP_KEYS[op])}},
        "required": ["op"],
        # A constrained decoder follows the schema, not the prose. One
        # flat object listing all eight arguments made
        # {"op": "delete_nodes", "detach": true, "set": {...}} valid to
        # the model and rejected by spec_to_mutations().
        "additionalProperties": False,
    }


#: JSON Schema for the mutation document, the delete/update counterpart
#: of INGEST_TOOL_SCHEMA. Operations run in the order given, in one
#: transaction.
MUTATE_TOOL_SCHEMA: dict = {
    "name": "mutate_graph",
    "description": (
        "Change or delete nodes and edges that are already in a property graph. Each "
        "operation selects rows with the same filters a traversal uses and updates or "
        "deletes every row it matches. Operations run in order, in one transaction: if "
        "one fails, none of them happened. A filter is required -- omitting it raises "
        "rather than matching the whole graph, unless all=true says so explicitly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "minItems": 1,
                "description": "Operations to apply, in order.",
                "items": {"oneOf": [_operation_schema(op) for op in sorted(_OP_KEYS)]},
            },
        },
        "required": ["operations"],
    },
}
