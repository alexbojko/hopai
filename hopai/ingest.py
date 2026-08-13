"""
hopai.ingest

Getting data into the graph, in the shapes callers already know.

    graph.create_schema()

    graph.add_nodes([
        {"id": 1, "type": "person", "name": "Alice"},
        {"id": 2, "type": "company", "name": "Acme"},
    ])
    graph.add_edges([
        {"start_id": 1, "end_id": 2, "kind": "works_at", "since": 2019},
    ])

Every writer accepts two spellings of a row, and the rule for which is
which is one line: **a row that has a `properties` key is in the nested
form; any other row is flat, and every key that is not an identity key is
a property.**

    {"id": 1, "type": "person"}                      flat
    {"id": 1, "properties": {"type": "person"}}      nested

The nested form is what a traversal returns, so a result can be fed
straight back into another graph without reshaping. The flat form is what
a person or a model writes by hand. Identity keys are `id` for nodes and
`id`, `start_id`, `end_id` for edges.

EDGES BY PROPERTY, not just by id. A model that just wrote some nodes
does not know their generated ids, and making it ask is how ingestion
turns into three round trips and a bug:

    graph.add_edges([
        {"start": {"email": "a@x.com"}, "end": {"email": "b@x.com"}, "kind": "knows"},
    ])

Each distinct reference is resolved in one batched lookup. A reference
matching no node, or more than one, raises rather than guessing -- put a
Unique() on whatever you reference and ambiguity becomes impossible.

MERGE, for ingestion that runs more than once:

    graph.merge_nodes([{"email": "a@x.com", "name": "Alice"}], on=["email"])

This is INSERT ... ON CONFLICT DO UPDATE, and it needs a unique index on
the `on` keys -- define one with Unique("email") first. On a match the
new properties are merged over the existing ones (`||`), leaving
properties you did not mention alone, which is what Cypher's
`ON MATCH SET` does. Pass replace=True to overwrite the whole bag.

ONE THING TO KNOW ABOUT MERGE AND CHECK CONSTRAINTS, because it surprises
everyone once: PostgreSQL evaluates CHECK constraints on the row being
inserted BEFORE it looks for a conflict. So a merge row must satisfy
every Check/Required/PropertyType constraint on its own, even when it is
destined to update an existing row that already satisfies them. With
Required("type") declared, `merge_nodes([{"email": ...}], on=["email"])`
is rejected -- include `type` in the row. ON CONFLICT resolves
uniqueness, not validity.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import bindparam, func, insert, or_, select, text
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.exc import IntegrityError, ProgrammingError

from .constraints import ConstraintViolation

#: Rows per statement. Large enough that the round trips disappear, small
#: enough that one bad batch stays diagnosable and memory stays flat.
BATCH_SIZE = 1000

_NODE_KEYS = frozenset({"id", "properties"})
_EDGE_KEYS = frozenset({"id", "start_id", "end_id", "start", "end", "properties"})


@dataclass
class IngestResult:
    """What a write did. Counts are rows written, not rows changed --
    an ON CONFLICT update that changes nothing still counts."""
    nodes: int = 0
    edges: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges, "elapsed_ms": self.elapsed_ms}

    def __repr__(self) -> str:
        return f"IngestResult(nodes={self.nodes}, edges={self.edges}, elapsed_ms={self.elapsed_ms:.1f})"


# ---------------------------------------------------------------------
# Row normalization
# ---------------------------------------------------------------------

def split_row(row: dict, identity_keys: frozenset, what: str) -> tuple:
    """Split one input row into (identity, properties).

    The nested form is detected by the presence of `properties`, and in
    that form nothing but identity keys may sit beside it -- silently
    dropping a stray key would lose data the caller believed it wrote."""
    if not isinstance(row, dict):
        raise TypeError(f"each {what} must be a dict, got {type(row).__name__}: {row!r}")

    if "properties" in row:
        stray = set(row) - identity_keys
        if stray:
            raise ValueError(
                f"{what} mixes both spellings: it has a 'properties' key, so "
                f"{sorted(stray)} would be silently dropped. Either move those inside "
                f"'properties', or drop the 'properties' key and write the row flat"
            )
        properties = row["properties"] or {}
        if not isinstance(properties, dict):
            raise TypeError(f"{what} 'properties' must be a dict, got {type(properties).__name__}")
    else:
        properties = {k: v for k, v in row.items() if k not in identity_keys}

    identity = {k: row[k] for k in identity_keys if k in row and k != "properties"}
    return identity, dict(properties)


def _chunks(rows: list, size: int = BATCH_SIZE):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def _translate(exc: IntegrityError, what: str) -> ConstraintViolation:
    """Turn the driver's IntegrityError into something that names the
    constraint the caller declared."""
    diag = getattr(exc.orig, "diag", None)
    name = getattr(diag, "constraint_name", None)
    detail = getattr(diag, "message_detail", None)
    message = f"{what} rejected by constraint {name!r}" if name else f"{what} rejected: {exc.orig}"
    if detail:
        message += f" -- {detail}"
    violation = ConstraintViolation(message, constraint=name, detail=detail)
    violation.__cause__ = exc
    return violation


# ---------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------

class Ingestor:
    """Implements Graph's write methods. Kept out of core.py so the
    traversal engine stays the only thing in there."""

    def __init__(self, graph):
        self.g = graph

    # -- helpers --------------------------------------------------------

    @contextmanager
    def _transaction(self, connection=None):
        """One transaction per call, and one shared by a whole document.

        Every write goes through here, batching included. A per-batch
        transaction would let a 2500-row write half-commit, and an agent
        retrying after that failure hits unique violations on the rows it
        believes never landed. Passing an existing connection joins the
        caller's transaction instead of opening a second one."""
        if connection is not None:
            yield connection
        else:
            with self.g.engine.begin() as opened:
                yield opened

    def _node_id_col(self):
        return getattr(self.g.nodes_tbl.c, self.g.node_id_col)

    def _sync_identity_sequence(self, connection, table, id_column) -> None:
        """After rows arrive with explicit ids, the identity sequence is
        still where it was, and the next generated id collides. Postgres
        will not do this for you; every hand-rolled loader that skips it
        works until the first row without an id."""
        sequence = connection.execute(
            select(func.pg_get_serial_sequence(_qualified(table), id_column.name))
        ).scalar()
        if sequence is None:
            return  # plain column, nothing generating values
        # The identifiers are interpolated because SQL has no parameter
        # position for them; both come from the Table object, never from
        # ingested data. The sequence name, which does come back from the
        # server, stays a bound parameter.
        maximum = f'SELECT COALESCE(MAX("{id_column.name}"), 0) FROM {_qualified(table)}'
        connection.execute(
            text(f"SELECT setval(:seq, GREATEST(({maximum}), 1))"), {"seq": sequence},
        )

    # -- nodes ----------------------------------------------------------

    def _node_payload(self, rows: list) -> tuple:
        id_column = self._node_id_col()
        payload, explicit_ids = [], False
        for row in rows:
            identity, properties = split_row(row, _NODE_KEYS, "node")
            record = {"properties": properties}
            if "id" in identity:
                record[id_column.name] = identity["id"]
                explicit_ids = True
            payload.append(record)
        if payload:
            _require_uniform(payload, id_column.name, "node")
        return payload, explicit_ids

    def add_nodes(self, rows: list, connection=None) -> int:
        table = self.g.nodes_tbl
        id_column = self._node_id_col()
        payload, explicit_ids = self._node_payload(rows)
        if not payload:
            return 0

        with self._transaction(connection) as conn:
            try:
                for chunk in _chunks(payload):
                    conn.execute(insert(table), chunk)
            except IntegrityError as exc:
                raise _translate(exc, "node") from exc
            if explicit_ids:
                self._sync_identity_sequence(conn, table, id_column)
        return len(payload)

    def merge_nodes(self, rows: list, on: list, replace: bool = False, connection=None) -> int:
        table = self.g.nodes_tbl
        id_column = self._node_id_col()
        payload, explicit_ids = self._node_payload(rows)
        if not payload:
            return 0

        with self._transaction(connection) as conn:
            self._merge_payload(conn, table, payload, on, replace, "node")
            if explicit_ids:
                self._sync_identity_sequence(conn, table, id_column)
        return len(payload)

    # -- edges ----------------------------------------------------------

    def add_edges(self, rows: list, connection=None) -> int:
        table = self.g.edges_tbl
        with self._transaction(connection) as conn:
            # Resolving references inside the write transaction, not
            # before it: a reference checked on another connection can be
            # deleted (breaking the foreign key) or duplicated (making the
            # id already chosen the wrong one) before the insert lands.
            payload = self._edge_payload(rows, conn)
            if not payload:
                return 0
            try:
                for chunk in _chunks(payload):
                    conn.execute(insert(table), chunk)
            except IntegrityError as exc:
                raise _translate(exc, "edge") from exc
        return len(payload)

    def merge_edges(self, rows: list, on: list, replace: bool = False, connection=None) -> int:
        with self._transaction(connection) as conn:
            payload = self._edge_payload(rows, conn)
            if not payload:
                return 0
            self._merge_payload(conn, self.g.edges_tbl, payload, on, replace, "edge")
        return len(payload)

    def _edge_payload(self, rows: list, connection) -> list:
        """Normalize edge rows, resolving any property references to node
        ids in one batched lookup rather than one query per edge."""
        start_col = self.g.edge_start_col
        end_col = self.g.edge_end_col
        edge_id_col = self.g.edge_id_col

        split = []
        references = []
        for row in rows:
            identity, properties = split_row(row, _EDGE_KEYS, "edge")
            for side in ("start", "end"):
                if side in row:
                    if isinstance(row[side], dict):
                        references.append(row[side])
                    else:
                        identity[f"{side}_id"] = row[side]
            split.append((row, identity, properties))

        resolved = self._resolve_references(references, connection)

        payload = []
        for row, identity, properties in split:
            record = {"properties": properties}
            for side, column in (("start", start_col), ("end", end_col)):
                if side in row and isinstance(row[side], dict):
                    record[column] = resolved[_reference_key(row[side])]
                elif f"{side}_id" in identity:
                    record[column] = identity[f"{side}_id"]
                else:
                    raise ValueError(
                        f"edge is missing {side}_id: give {side}_id, or {side}={{...}} "
                        f"to look the node up by its properties -- got {row!r}"
                    )
            if "id" in identity:
                record[edge_id_col] = identity["id"]
            payload.append(record)
        return payload

    def _resolve_references(self, references: list, connection) -> dict:
        """Map each distinct property reference to exactly one node id.

        One query for the whole batch: an OR of containment tests, which
        the GIN index on properties serves."""
        if not references:
            return {}
        distinct = {}
        for reference in references:
            if not reference:
                raise ValueError("an empty {} matches every node; give properties to match on")
            distinct.setdefault(_reference_key(reference), reference)

        table = self.g.nodes_tbl
        id_column = self._node_id_col()
        query = select(id_column, table.c.properties).where(
            or_(*(table.c.properties.op("@>")(
                bindparam(None, reference, type_=JSONB)) for reference in distinct.values()))
        )
        matches = connection.execute(query).all()

        resolved: dict = {}
        for key, reference in distinct.items():
            hits = [row[0] for row in matches
                    if all(row[1].get(k) == v for k, v in reference.items())]
            if not hits:
                raise ValueError(
                    f"no node matches {reference!r} -- create it before the edge that "
                    f"references it, or check the property values"
                )
            if len(hits) > 1:
                raise ValueError(
                    f"{len(hits)} nodes match {reference!r}, so the edge is ambiguous. "
                    f"Reference something unique, and add Unique(...) on it so this "
                    f"cannot happen again"
                )
            resolved[key] = hits[0]
        return resolved

    # -- merge ----------------------------------------------------------

    def _merge_payload(self, connection, table, payload, on, replace, what) -> int:
        if not payload:
            return 0
        if not on:
            raise ValueError(
                f"merge_{what}s(on=[...]) needs the keys that identify a row -- "
                f"without them there is nothing to conflict on. Define the matching "
                f"Unique({', '.join(repr(k) for k in ['key'])}) first"
            )
        from .constraints import _Target, key_sql

        # Rendered SQL, not expression objects: the conflict target has to
        # match the index expression exactly for Postgres to infer it, and
        # a bound parameter in that position is neither matchable nor
        # accepted. key_sql() is the same function CREATE INDEX uses.
        target = _Target(table, what)
        index_elements = [text(key_sql(target, key)) for key in on]

        for chunk in _chunks(payload):
            statement = pg_insert(table).values(chunk)
            merged = (statement.excluded.properties if replace
                      else table.c.properties.op("||")(statement.excluded.properties))
            statement = statement.on_conflict_do_update(
                index_elements=index_elements, set_={"properties": merged}
            )
            try:
                connection.execute(statement)
            except IntegrityError as exc:
                raise _translate(exc, what) from exc
            except ProgrammingError as exc:
                if "no unique or exclusion constraint" not in str(exc.orig):
                    raise
                keys = ", ".join(repr(k) for k in on)
                raise ConstraintViolation(
                    f"merge_{what}s(on=[{keys}]) needs a unique index over exactly "
                    f"those keys to detect a conflict, and there is none. Declare it "
                    f"first: graph.define_constraints({what}s=[Unique({keys})])"
                ) from exc
        return len(payload)

    # -- documents ------------------------------------------------------

    def ingest(self, document: dict, merge_nodes_on: Optional[list] = None,
               merge_edges_on: Optional[list] = None) -> IngestResult:
        if not isinstance(document, dict):
            raise TypeError(f"ingest() takes a dict with 'nodes' and/or 'edges', "
                            f"got {type(document).__name__}")
        unknown = set(document) - {"nodes", "edges"}
        if unknown:
            raise ValueError(f"unknown keys {sorted(unknown)} -- a document has 'nodes' "
                             f"and 'edges', nothing else")

        started = time.perf_counter()
        nodes = document.get("nodes") or []
        edges = document.get("edges") or []

        # One transaction for the whole document. Nodes first, always, so
        # an edge given as start={...} can resolve against a node created
        # by the same call -- and if the edges turn out to violate a
        # constraint, the nodes roll back with them. Committing half a
        # document would leave a retry hitting unique violations on rows
        # the caller was told had failed.
        with self._transaction() as connection:
            written_nodes = (self.merge_nodes(nodes, on=merge_nodes_on, connection=connection)
                             if merge_nodes_on
                             else self.add_nodes(nodes, connection=connection))
            written_edges = (self.merge_edges(edges, on=merge_edges_on, connection=connection)
                             if merge_edges_on
                             else self.add_edges(edges, connection=connection))
        return IngestResult(nodes=written_nodes, edges=written_edges,
                            elapsed_ms=(time.perf_counter() - started) * 1000)

    def add_networkx(self, nx_graph) -> IngestResult:
        nodes = [{"id": n, "properties": dict(data)} for n, data in nx_graph.nodes(data=True)]
        edges = [{"start_id": u, "end_id": v, "properties": dict(data)}
                 for u, v, data in nx_graph.edges(data=True)]
        return self.ingest({"nodes": nodes, "edges": edges})


def _reference_key(reference: dict) -> tuple:
    return tuple(sorted(reference.items()))


def _qualified(table) -> str:
    return f"{table.schema}.{table.name}" if table.schema else table.name


def _require_uniform(payload: list, id_name: str, what: str) -> None:
    """executemany binds one parameter set shape, so a batch where some
    rows carry an id and others do not would bind NULL for the missing
    ones and defeat the identity default."""
    with_id = sum(1 for record in payload if id_name in record)
    if with_id not in (0, len(payload)):
        raise ValueError(
            f"{with_id} of {len(payload)} {what}s have an explicit id. Either give every "
            f"row an id or none of them -- a mixed batch would insert NULL ids for the "
            f"rest instead of generating them. Split it into two calls"
        )


#: JSON Schema for the document form, to hand to a tool-calling model
#: alongside TRAVERSE_TOOL_SCHEMA. Deliberately describes only the flat
#: spelling: it is the one a model writes without being asked twice, and
#: the nested form is accepted anyway.
INGEST_TOOL_SCHEMA: dict = {
    "name": "ingest_graph",
    "description": (
        "Write nodes and edges into a property graph. Nodes carry an optional id and "
        "any other keys as properties. Edges connect two nodes, either by id "
        "(start_id/end_id) or by matching an existing node's properties (start/end). "
        "Nodes in the same call are written before edges, so an edge may reference a "
        "node created in the same request."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "description": "Nodes to create. Every key other than 'id' is a property.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "Optional; generated when omitted."},
                    },
                    "additionalProperties": True,
                },
            },
            "edges": {
                "type": "array",
                "description": "Edges to create. Every key other than the ones below is a property.",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_id": {"type": "integer", "description": "Id of the source node."},
                        "end_id": {"type": "integer", "description": "Id of the target node."},
                        "start": {
                            "type": "object",
                            "description": "Instead of start_id: properties identifying the "
                                           "source node. Must match exactly one node.",
                        },
                        "end": {
                            "type": "object",
                            "description": "Instead of end_id: properties identifying the "
                                           "target node. Must match exactly one node.",
                        },
                    },
                    "additionalProperties": True,
                },
            },
        },
    },
}
