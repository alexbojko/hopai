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
"""

from __future__ import annotations

import json as _json
import time
from dataclasses import dataclass

from sqlalchemy import ARRAY, Text, and_, cast, delete, literal, or_, select, update
from sqlalchemy.dialects.postgresql import JSONB, array
from sqlalchemy.exc import IntegrityError

from .constraints import ConstraintViolation
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
    "delete_nodes": {"where", "detach", "all"},
    "delete_edges": {"where", "start", "end", "all"},
    "update_nodes": {"where", "set", "remove", "replace", "all"},
    "update_edges": {"where", "start", "end", "set", "remove", "replace", "all"},
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


def _detach_hint(exc: IntegrityError) -> ConstraintViolation:
    """The foreign-key refusal, rewritten as the fix.

    Postgres says "still referenced from table edges", which is true and
    useless: the caller has to know that edges reference nodes and that
    the flag is spelled `detach`. Cypher's own error for this is the
    thing to imitate -- it names DETACH DELETE."""
    violation = constraint_violation(exc, "node")
    hint = ConstraintViolation(
        f"{violation} -- these nodes still have edges, and deleting them would leave "
        f"edges pointing at nothing. Pass detach=True (Cypher: DETACH DELETE) to delete "
        f"those edges with the nodes, or delete the edges first",
        constraint=violation.constraint, detail=violation.detail,
    )
    hint.__cause__ = exc
    return hint


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
        return filt is None or (isinstance(filt, dict) and not filt)

    def _guard(self, filters: list, call: str, what: str, all: bool) -> None:
        """The unfiltered-mutation refusal, in one place so every entry
        point -- Python, JSON and Cypher -- obeys the same rule.

        Both directions are an error. No filter without all=True is the
        accident this exists for; all=True *with* a filter is a caller
        who believes one of the two is being honoured, and only one of
        them is."""
        unfiltered = all_(self._blank(f) for f in filters)
        if unfiltered and not all:
            raise ValueError(
                f"{call}() was given no filter, which would match every {what} in graph "
                f"{self.g.graph!r}. Filter it with where=..., pass all=True to say you "
                f"really mean every {what}, or call graph.clear() to empty the graph"
            )
        if all and not unfiltered:
            raise ValueError(
                f"{call}(all=True) also got a filter, and the two disagree about what to "
                f"touch. Drop whichever one you did not mean"
            )

    def _node_predicate(self, where):
        nt = self.g.nodes_tbl
        conditions = [self.g._scoped(nt)]
        # A blank filter is left out rather than resolved to TRUE: it is
        # only ever reached with all=True, and `WHERE graph_id = 'x' AND
        # true` is a planner input as well as something a reader has to
        # look twice at.
        if not self._blank(where):
            conditions.append(resolve(nt.c.properties, where))
        return and_(*conditions)

    def _edge_predicate(self, where, start, end):
        """The edge rows a mutation targets: its own properties, plus
        optional filters on the nodes at either end.

        Endpoint filters are what makes `MATCH (a {name: 'Alice'})-[r:knows]->()
        DELETE r` translate exactly -- without them "Alice's knows edges"
        would have to be fetched and deleted by id, which is two round
        trips and a race between them."""
        et = self.g.edges_tbl
        conditions = [self.g._scoped(et)]
        if not self._blank(where):
            conditions.append(resolve(et.c.properties, where))
        for column, filt in ((self.g.edge_start_col, start), (self.g.edge_end_col, end)):
            if not self._blank(filt):
                conditions.append(getattr(et.c, column).in_(self._node_ids(filt)))
        return and_(*conditions)

    def _node_ids(self, filt):
        """The ids of this graph's nodes matching `filt`, as a subquery.
        Scoped like every other read: an unscoped one would let an
        endpoint filter match a node belonging to another graph."""
        nt = self.g.nodes_tbl
        return select(getattr(nt.c, self.g.node_id_col)).where(self._node_predicate(filt))

    # -- statements -----------------------------------------------------

    def delete_nodes_statement(self, where=None, all: bool = False):
        self._guard([where], "delete_nodes", "node", all)
        return delete(self.g.nodes_tbl).where(self._node_predicate(where))

    def detach_statement(self, where=None, all: bool = False):
        """Every edge incident to the nodes `where` matches, in either
        direction. Run before the node delete, in the same transaction --
        between two transactions an edge inserted in the gap would fail
        the node delete with the error detach was supposed to prevent."""
        self._guard([where], "delete_nodes", "node", all)
        et = self.g.edges_tbl
        matched = self._node_ids(where)
        return delete(et).where(and_(
            self.g._scoped(et),
            or_(getattr(et.c, self.g.edge_start_col).in_(matched),
                getattr(et.c, self.g.edge_end_col).in_(matched)),
        ))

    def delete_edges_statement(self, where=None, start=None, end=None, all: bool = False):
        self._guard([where, start, end], "delete_edges", "edge", all)
        return delete(self.g.edges_tbl).where(self._edge_predicate(where, start, end))

    def update_nodes_statement(self, where=None, set=None, remove=None,
                               replace: bool = False, all: bool = False):
        self._guard([where], "update_nodes", "node", all)
        nt = self.g.nodes_tbl
        return update(nt).where(self._node_predicate(where)).values(
            properties=_new_properties(nt.c.properties, set, remove, replace, "update_nodes"))

    def update_edges_statement(self, where=None, start=None, end=None, set=None,
                               remove=None, replace: bool = False, all: bool = False):
        self._guard([where, start, end], "update_edges", "edge", all)
        et = self.g.edges_tbl
        return update(et).where(self._edge_predicate(where, start, end)).values(
            properties=_new_properties(et.c.properties, set, remove, replace, "update_edges"))

    # -- execution ------------------------------------------------------

    def delete_nodes(self, where=None, detach: bool = False, all: bool = False,
                     connection=None) -> MutationResult:
        started = time.perf_counter()
        # The statements are built before the transaction opens: a
        # refusal (no filter, contradictory arguments) should raise
        # without having taken a connection out of the pool.
        detach_statement = self.detach_statement(where, all) if detach else None
        statement = self.delete_nodes_statement(where, all)
        with one_transaction(self.g, connection) as conn:
            edges = conn.execute(detach_statement).rowcount if detach else 0
            try:
                nodes = conn.execute(statement).rowcount
            except IntegrityError as exc:
                raise _detach_hint(exc) from exc
        return MutationResult(deleted_nodes=nodes, deleted_edges=edges,
                              elapsed_ms=(time.perf_counter() - started) * 1000)

    def delete_edges(self, where=None, start=None, end=None, all: bool = False,
                     connection=None) -> MutationResult:
        started = time.perf_counter()
        statement = self.delete_edges_statement(where, start, end, all)
        with one_transaction(self.g, connection) as conn:
            deleted = conn.execute(statement).rowcount
        return MutationResult(deleted_edges=deleted,
                              elapsed_ms=(time.perf_counter() - started) * 1000)

    def update_nodes(self, where=None, set=None, remove=None, replace: bool = False,
                     all: bool = False, connection=None) -> MutationResult:
        started = time.perf_counter()
        statement = self.update_nodes_statement(where, set, remove, replace, all)
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
                     connection=None) -> MutationResult:
        started = time.perf_counter()
        statement = self.update_edges_statement(where, start, end, set, remove, replace, all)
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
            edges = conn.execute(delete(et).where(self.g._scoped(et))).rowcount
            nodes = conn.execute(delete(nt).where(self.g._scoped(nt))).rowcount
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

    if not set and not remove:
        raise ValueError(
            f"{call}() was given nothing to change -- pass set={{...}} to write "
            f"properties, remove=[...] to drop them, or both"
        )
    if replace:
        if not set:
            raise ValueError(
                f"{call}(replace=True) with no set= would erase every property of every "
                f"matched row. Pass the properties it should end up with, or use "
                f"remove=[...] to drop specific keys"
            )
        if remove:
            raise ValueError(
                f"{call}(replace=True, remove=[...]) contradicts itself: replace already "
                f"decides the whole property bag, so removing a key from it means "
                f"leaving that key out of set="
            )
    overlap = sorted(set_(set or {}) & set_(remove))
    if overlap:
        raise ValueError(
            f"{call}() both sets and removes {overlap} -- pick one per property"
        )

    value = column
    if set:
        try:
            incoming = cast(literal(_json.dumps(set)), JSONB)
        except TypeError as exc:
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

def spec_to_mutations(spec: dict) -> list:
    """Convert a JSON mutation document into the operation list
    Mutator.execute_operations() runs -- exposed on its own so a plan
    can be reviewed, logged or shown to a caller before it changes
    anything.

        spec_to_mutations({"operations": [
            {"op": "delete_nodes", "where": {"type": "draft"}, "detach": True},
        ]})
    """
    if not isinstance(spec, dict):
        raise TypeError(f"a mutation takes a dict with an 'operations' list, "
                        f"got {type(spec).__name__}")
    unknown = spec.keys() - {"operations"}
    if unknown:
        raise ValueError(f"unknown keys {sorted(unknown)} -- a mutation document has "
                         f"'operations', nothing else")
    operations = spec.get("operations")
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


#: JSON Schema for the mutation document, the delete/update counterpart
#: of INGEST_TOOL_SCHEMA. Operations run in the order given, in one
#: transaction.
MUTATE_TOOL_SCHEMA: dict = {
    "name": "mutate_graph",
    "description": (
        "Change or delete nodes and edges that are already in a property graph. Each "
        "operation selects rows with the same filters a traversal uses and updates or "
        "deletes every row it matches. Operations run in order, in one transaction. "
        "A filter is required: omitting it raises rather than matching the whole graph, "
        "unless all=true says so explicitly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "minItems": 1,
                "description": "Operations to apply, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": sorted(_OP_KEYS),
                            "description": "Which rows to change: nodes or edges, "
                                           "updated or deleted.",
                        },
                        "where": {
                            "type": "object",
                            "description": "Filter on the properties of the rows to "
                                           "change. Same grammar as traverse_graph.",
                        },
                        "start": {
                            "type": "object",
                            "description": "Edges only: filter on the properties of the "
                                           "node the edge starts at.",
                        },
                        "end": {
                            "type": "object",
                            "description": "Edges only: filter on the properties of the "
                                           "node the edge ends at.",
                        },
                        "set": {
                            "type": "object",
                            "description": "Updates only: properties to write. Merged "
                                           "over the existing ones, leaving properties "
                                           "not mentioned here alone.",
                        },
                        "remove": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Updates only: property names to drop.",
                        },
                        "replace": {
                            "type": "boolean",
                            "description": "Updates only: if true, 'set' becomes the "
                                           "whole property bag and anything not in it "
                                           "is dropped.",
                        },
                        "detach": {
                            "type": "boolean",
                            "description": "delete_nodes only: also delete the edges "
                                           "attached to the deleted nodes. Without it, "
                                           "deleting a node that still has edges fails.",
                        },
                        "all": {
                            "type": "boolean",
                            "description": "Apply to every row, with no filter. Required "
                                           "to be explicit -- an operation with neither "
                                           "a filter nor this flag raises.",
                        },
                    },
                    "required": ["op"],
                },
            },
        },
        "required": ["operations"],
    },
}
