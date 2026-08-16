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
`id`, `start_id`, `end_id` for edges -- plus, on a Graph built over a
custom table, every EXTRA COLUMN that table carries (Graph's
node_extra_cols/edge_extra_cols, discovered from the table itself; see
models.py's "EXTENDING THE MODEL"). `add_nodes([{"id": 1, "user_id": 7,
"type": "person"}])` writes `user_id` to its own real column the same
way it writes `id`, never into `properties` -- and merge_nodes()/
merge_edges() refresh an extra column's value on conflict, same as
`properties`. `update_nodes()`/`update_edges()` (mutate.py) do not:
`set=`/`remove=` stay `properties`-only.

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

import json as _json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import bindparam, cast, func, insert, literal, or_, select, text
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.exc import IntegrityError, ProgrammingError

from .constraints import ConstraintViolation
from .models import EDGE_IDENTITY_KEYS, NODE_IDENTITY_KEYS

#: Rows per statement. Large enough that the round trips disappear, small
#: enough that one bad batch stays diagnosable and memory stays flat.
BATCH_SIZE = 1000

_NODE_KEYS = NODE_IDENTITY_KEYS
_EDGE_KEYS = EDGE_IDENTITY_KEYS


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


@contextmanager
def one_transaction(graph, connection=None):
    """One transaction per call, and one shared by a whole document.

    Every write goes through here -- ingestion, batching, and the
    deletes and updates in mutate.py alike. A per-batch transaction
    would let a 2500-row write half-commit, and an agent retrying after
    that failure hits unique violations on the rows it believes never
    landed. Passing an existing connection joins the caller's
    transaction instead of opening a second one, which is what makes a
    multi-operation plan atomic."""
    if connection is not None:
        yield connection
    else:
        with graph.engine.begin() as opened:
            yield opened


def constraint_violation(exc: IntegrityError, what: str) -> ConstraintViolation:
    """Turn the driver's IntegrityError into something that names the
    constraint the caller declared."""
    diag = getattr(exc.orig, "diag", None)
    name = getattr(diag, "constraint_name", None)
    detail = getattr(diag, "message_detail", None)
    message = f"{what} rejected by constraint {name!r}" if name else f"{what} rejected: {exc.orig}"
    if detail:
        message += f" -- {detail}"
    # No __cause__ here: every caller raises this `from exc`, which is
    # what chains the driver's error onto it. Assigning it as well was
    # dead code -- mutation testing flagged it in both this function and
    # _detach_hint, and nothing can observe the difference.
    return ConstraintViolation(message, constraint=name, detail=detail)


# ---------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------

class Ingestor:
    """Implements Graph's write methods. Kept out of core.py so the
    traversal engine stays the only thing in there."""

    def __init__(self, graph):
        self.g = graph
        # Extra columns (see Graph.node_extra_cols/edge_extra_cols) are
        # identity keys too, the same way id/start_id/end_id are: a flat
        # row addresses one by name and split_row() must pull it out of
        # `properties` rather than leave it there.
        self._node_keys = _NODE_KEYS | frozenset(self.g.node_extra_cols)
        self._edge_keys = _EDGE_KEYS | frozenset(self.g.edge_extra_cols)

    # -- helpers --------------------------------------------------------

    def _transaction(self, connection=None):
        return one_transaction(self.g, connection)

    def _graph_stamp(self) -> dict:
        """The graph column for an inserted row, or nothing when the
        caller's tables have no such column."""
        if self.g.graph_col is None:
            return {}
        return {self.g.graph_col: self.g.graph}

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

    def _node_payload(self, rows: list, method: str = "add_nodes") -> tuple:
        id_column = self._node_id_col()
        vector_fields = _vector_fields(self.g, "nodes")
        payload, explicit_ids = [], False
        for row in rows:
            identity, properties = split_row(row, self._node_keys, "node")
            if vector_fields:
                _refuse_vector_property(properties, vector_fields, method, "nodes", identity.get("id"))
            record = {"properties": properties, **self._graph_stamp()}
            if "id" in identity:
                record[id_column.name] = identity["id"]
                explicit_ids = True
            for column in self.g.node_extra_cols:
                if column in identity:
                    record[column] = identity[column]
            payload.append(record)
        if payload:
            _require_uniform(payload, id_column.name, "node")
            _require_uniform_columns(payload, self.g.node_extra_cols, "node")
        return payload, explicit_ids

    def add_nodes(self, rows: list, connection=None, return_ids: bool = False):
        table = self.g.nodes_tbl
        id_column = self._node_id_col()
        payload, explicit_ids = self._node_payload(rows)
        if not payload:
            return [] if return_ids else 0

        ids: list = []
        with self._transaction(connection) as conn:
            try:
                for chunk in _chunks(payload):
                    if return_ids:
                        ids.extend(self._insert_returning(conn, table, id_column, chunk))
                    else:
                        conn.execute(insert(table), chunk)
            except IntegrityError as exc:
                raise constraint_violation(exc, "node") from exc
            if explicit_ids:
                self._sync_identity_sequence(conn, table, id_column)
        return ids if return_ids else len(payload)

    @staticmethod
    def _insert_returning(connection, table, id_column, chunk) -> list:
        """Insert and get the ids back, in the order the rows were given.

        sort_by_parameter_order is what makes row N of the result row N of
        the input -- without it PostgreSQL may return them in any order,
        and a caller wiring edges to those ids would connect the wrong
        nodes without ever seeing an error."""
        result = connection.execute(
            insert(table).returning(id_column, sort_by_parameter_order=True), chunk
        )
        return [row[0] for row in result]

    def merge_nodes(self, rows: list, on: list, replace: bool = False, connection=None) -> int:
        table = self.g.nodes_tbl
        id_column = self._node_id_col()
        payload, explicit_ids = self._node_payload(rows, method="merge_nodes")
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
                raise constraint_violation(exc, "edge") from exc
        return len(payload)

    def merge_edges(self, rows: list, on: list, replace: bool = False, connection=None) -> int:
        with self._transaction(connection) as conn:
            payload = self._edge_payload(rows, conn, method="merge_edges")
            if not payload:
                return 0
            self._merge_payload(conn, self.g.edges_tbl, payload, on, replace, "edge")
        return len(payload)

    def _edge_payload(self, rows: list, connection, method: str = "add_edges") -> list:
        """Normalize edge rows, resolving any property references to node
        ids in one batched lookup rather than one query per edge."""
        start_col = self.g.edge_start_col
        end_col = self.g.edge_end_col
        edge_id_col = self.g.edge_id_col
        vector_fields = _vector_fields(self.g, "edges")

        split = []
        references = []
        for row in rows:
            identity, properties = split_row(row, self._edge_keys, "edge")
            if vector_fields:
                _refuse_vector_property(properties, vector_fields, method, "edges", identity.get("id"))
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
            record = {"properties": properties, **self._graph_stamp()}
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
            for column in self.g.edge_extra_cols:
                if column in identity:
                    record[column] = identity[column]
            payload.append(record)
        if payload:
            _require_uniform_columns(payload, self.g.edge_extra_cols, "edge")
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
            # Scoped: without this an edge could resolve its endpoint to a
            # node in another graph, and the composite foreign key would
            # then reject the write with a message about nothing obvious.
            self.g._scoped(table),
            or_(*(table.c.properties.op("@>")(
                bindparam(None, reference, type_=JSONB)) for reference in distinct.values())),
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

    def _merge_payload(self, connection, table, payload, on, replace, what, returning=None,
                       update_properties=None):
        if not payload:
            return []
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
        target = _Target(table, what, self.g.graph, self.g.graph_col or "graph_id")
        if self.g.graph_col is None:
            target.graph = None
        # scope_index() is what Unique() used to build the index, so the
        # conflict target is spelled identically -- including the leading
        # graph column. Without it Postgres cannot infer the index and
        # every merge fails.
        index_elements = [text(key_sql(target, key)) for key in target.scope_index(tuple(on))]
        # Extra columns (see Graph.node_extra_cols/edge_extra_cols) have no
        # merge semantics of their own -- unlike `properties`, a plain
        # column cannot be `||`-ed -- so a re-import just overwrites one
        # with the incoming value, the same as it would overwrite `id`
        # were `id` mergeable at all. Only ones the payload actually
        # carries (uniform across it, by _require_uniform_columns) are
        # touched -- referencing an unlisted column in EXCLUDED is invalid
        # SQL, since EXCLUDED only has the columns this INSERT named.
        extra_cols = self.g.node_extra_cols if table is self.g.nodes_tbl else self.g.edge_extra_cols

        ids: list = []
        for chunk in _chunks(payload):
            statement = pg_insert(table).values(chunk)
            if update_properties is not None:
                # ON MATCH SET: a matched row gets these, not whatever the
                # incoming row happened to carry (which includes anything
                # ON CREATE SET added, and must not leak into an update).
                incoming = cast(literal(_json.dumps(update_properties)), JSONB)
            else:
                incoming = statement.excluded.properties
            merged = incoming if replace else table.c.properties.op("||")(incoming)
            set_ = {"properties": merged, **{
                column: statement.excluded[column] for column in extra_cols
                if chunk and column in chunk[0]
            }}
            statement = statement.on_conflict_do_update(
                index_elements=index_elements, set_=set_
            )
            if returning is not None:
                # DO UPDATE, unlike DO NOTHING, returns a row whether it
                # inserted or matched -- so this is the id either way.
                #
                # ONLY SAFE ONE ROW AT A TIME. There is no
                # sort_by_parameter_order for ON CONFLICT, and PostgreSQL
                # does not specify the RETURNING order when some rows
                # insert and others update -- so a batch would hand back
                # ids that pair with the wrong input rows, with matching
                # lengths and no error. _merge_with_sets passes one row;
                # keep it that way, or match the ids back by their keys.
                statement = statement.returning(returning)
            try:
                result = connection.execute(statement)
                if returning is not None:
                    ids.extend(row[0] for row in result)
            except IntegrityError as exc:
                raise constraint_violation(exc, what) from exc
            except ProgrammingError as exc:
                if "no unique or exclusion constraint" not in str(exc.orig):
                    raise
                keys = ", ".join(repr(k) for k in on)
                raise ConstraintViolation(
                    f"merge_{what}s(on=[{keys}]) needs a unique index over exactly "
                    f"those keys to detect a conflict, and there is none. Declare it "
                    f"first: graph.define_constraints({what}s=[Unique({keys})])"
                ) from exc
        return ids

    # -- documents ------------------------------------------------------

    def ingest(self, document: dict, merge_nodes_on: Optional[list] = None,
               merge_edges_on: Optional[list] = None, connection=None) -> IngestResult:
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
        with self._transaction(connection) as connection:
            written_nodes = (self.merge_nodes(nodes, on=merge_nodes_on, connection=connection)
                             if merge_nodes_on
                             else self.add_nodes(nodes, connection=connection))
            written_edges = (self.merge_edges(edges, on=merge_edges_on, connection=connection)
                             if merge_edges_on
                             else self.add_edges(edges, connection=connection))
        return IngestResult(nodes=written_nodes, edges=written_edges,
                            elapsed_ms=(time.perf_counter() - started) * 1000)

    def execute_operations(self, operations: list, connection=None) -> IngestResult:
        """Run a plan from hopai.cypher.cypher_to_operations().

        Everything lands in one transaction, and a table of variable ->
        node id is threaded through it: a created node's id comes back
        from the INSERT, a merged node's from the upsert's RETURNING, and
        a matched node's from a property lookup. Edges then refer to
        variables rather than to ids nobody knew when the query was
        written."""
        started = time.perf_counter()
        bound: dict = {}
        nodes = edges = 0

        with self._transaction(connection) as conn:
            for operation in operations:
                kind = operation["op"]

                if kind == "match":
                    resolved = self._resolve_references([operation["where"]], conn)
                    bound[operation["var"]] = resolved[_reference_key(operation["where"])]

                elif kind == "create_nodes":
                    ids = self.add_nodes(operation["rows"], connection=conn, return_ids=True)
                    nodes += len(ids)
                    _bind(bound, operation["vars"], ids)

                elif kind == "merge_nodes":
                    ids = self._merge_with_sets(conn, operation)
                    nodes += len(ids)
                    _bind(bound, operation["vars"], ids)

                elif kind in ("create_edges", "merge_edges"):
                    rows = [{"start_id": _lookup(bound, row["start_var"]),
                             "end_id": _lookup(bound, row["end_var"]),
                             "properties": row["properties"]}
                            for row in operation["rows"]]
                    if kind == "create_edges":
                        edges += self.add_edges(rows, connection=conn)
                    else:
                        from .constraints import Col
                        edges += self.merge_edges(
                            rows, on=[Col(self.g.edge_start_col), Col(self.g.edge_end_col),
                                      *operation["on"]],
                            connection=conn)

                else:  # pragma: no cover - the translator emits nothing else
                    raise ValueError(f"unknown operation {kind!r}")

        return IngestResult(nodes=nodes, edges=edges,
                            elapsed_ms=(time.perf_counter() - started) * 1000)

    def _merge_with_sets(self, connection, operation) -> list:
        """MERGE, honouring ON CREATE SET and ON MATCH SET.

        Cypher matches on every property in the pattern, so those are the
        conflict keys. ON CREATE SET adds properties to the inserted row
        only; ON MATCH SET replaces what an existing row gets, which is
        why the update cannot simply reuse the incoming row."""
        pattern = operation["rows"][0]
        on_create, on_match = operation["on_create"], operation["on_match"]
        payload = [{"properties": {**pattern, **on_create}}]
        update = None
        if on_create or on_match:
            update = {**pattern, **on_match}
        return self._merge_payload(
            connection, self.g.nodes_tbl, payload, operation["on"], False, "node",
            returning=self._node_id_col(), update_properties=update,
        )

    def add_networkx(self, nx_graph) -> IngestResult:
        """Load a networkx graph. The inverse of Subgraph.to_networkx(),
        extra columns included: an attribute matching an extra column's
        name is written there, not folded into `properties` -- the same
        split to_networkx() made when it produced that attribute."""
        nodes = [_networkx_row({"id": n}, data, self.g.node_extra_cols)
                 for n, data in nx_graph.nodes(data=True)]
        edges = [_networkx_row({"start_id": u, "end_id": v}, data, self.g.edge_extra_cols)
                 for u, v, data in nx_graph.edges(data=True)]
        return self.ingest({"nodes": nodes, "edges": edges})


def _networkx_row(identity: dict, data, extra_cols: tuple) -> dict:
    """One add_networkx() row: `identity` (id, or start_id/end_id) plus
    whichever attributes in `data` name an extra column, pulled out flat
    the way a hand-written row already addresses id/start_id/end_id;
    everything left over is `properties`."""
    properties = dict(data)
    row = dict(identity)
    for column in extra_cols:
        if column in properties:
            row[column] = properties.pop(column)
    row["properties"] = properties
    return row


def _vector_fields(graph, target: str) -> dict:
    """The declared vector fields for `target` ("nodes"/"edges"), or {}
    when define_vectors() was never called -- read fresh on every call
    rather than cached at Ingestor.__init__, since the Ingestor is
    cached on Graph and outlives a later define_vectors()."""
    vectors = graph.vectors
    return vectors.get(target, {}) if vectors else {}


def _refuse_vector_property(properties: dict, vector_fields: dict, method: str,
                            target: str, row_id) -> None:
    """A declared vector field's floats landing in `properties` instead
    of its vec_* column is silent corruption (#50): nothing errors,
    similarity just finds nothing for that row, and every read echoes
    the raw floats back in the JSONB bag. Refuse and name the rewrite
    rather than routing the value to set_vectors() ourselves -- that
    would fold a second write, with its own transaction and network
    call, into this one, and "embedding happens outside the
    transaction" stops being true the moment ingest.py does it too."""
    for name, value in properties.items():
        if name not in vector_fields or isinstance(value, str):
            continue
        id_repr = repr(row_id) if row_id is not None else "..."
        set_vectors_hint = (
            f"Pass the floats to set_vectors({target}=[{{\"id\": {id_repr}, "
            f"{name!r}: [...]}}]) instead"
        )
        text_hint = (
            ", or write the SOURCE TEXT here and let embed_stale() fill the vector."
            if vector_fields[name].embed is not None
            else f" -- {name!r} has no embed= configured, so it takes floats only."
        )
        raise ValueError(
            f"{method}(): {name!r} is a declared vector field, and {value!r} is not "
            f"text to embed -- writing it as a property would store the embedding in "
            f"JSONB, where similarity never reads it. {set_vectors_hint}{text_hint}"
        )


def _reference_key(reference: dict) -> tuple:
    return tuple(sorted(reference.items()))


def _bind(bound: dict, variables: list, ids: list) -> None:
    for variable, node_id in zip(variables, ids, strict=True):
        if variable is not None:
            bound[variable] = node_id


def _lookup(bound: dict, variable: str):
    if variable not in bound:
        raise ValueError(f"{variable!r} is not bound to a node")
    return bound[variable]


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


def _require_uniform_columns(payload: list, names: tuple, what: str) -> None:
    """The extra-column twin of _require_uniform, one name at a time: a
    batch where some rows carry a value for an extra column and others
    do not would bind NULL for the rest, silently discarding whatever
    that column's own default (or NOT NULL constraint) would otherwise
    have decided -- the same executemany trap `id` has, generalized to
    every column a custom table adds (see models.py)."""
    for name in names:
        with_value = sum(1 for record in payload if name in record)
        if with_value not in (0, len(payload)):
            raise ValueError(
                f"{with_value} of {len(payload)} {what}s have {name!r}. Either give every "
                f"row a value for {name!r} or none of them -- a mixed batch would bind NULL "
                f"for the rest instead of leaving the column to its default. Split it into "
                f"two calls"
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
