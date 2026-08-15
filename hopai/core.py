"""
hopai.core

The traversal engine. Builds one recursive CTE per Graph.traverse() call
and executes it in a single round trip.

DESIGN NOTES, kept close to the code because they were each earned by a
real bug during development, not decided upfront:

  - Every recursive walk carries a LOCAL path array (reset at each hop's
    boundary, not carried across the whole chain) for two reasons: cycle
    protection, and complete edge reconstruction when a hop spans more
    than one real edge. An earlier version tracked one GLOBAL path per
    destination node across the entire chain -- that silently dropped
    real fan-in whenever two different parents fed the same intermediate
    node, because only one path per node was kept. Local, per-hop path
    tracking plus a separate "which nodes did THIS hop reach" set fixes
    that.
  - "Continuing the walk" (which nodes were reached, for feeding the next
    hop) and "reporting the subgraph" (every real edge used by every
    qualifying walk) are two different dedup needs and use two different
    queries -- conflating them was the bug above.
  - Reported nodes are derived from the real edges found, not from the
    raw seed set. A seed node that never connects anywhere is a dead end
    and is correctly excluded this way, automatically, rather than needing
    a separate pruning pass.
  - `optional=True` is only accepted on the LAST hop. Supporting it
    mid-chain would mean every downstream hop tolerating a missing
    anchor node -- a materially bigger rewrite than a flag, and not
    something this library has committed to (yet).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal

from typing import Optional

from sqlalchemy import String, and_, cast, create_engine, distinct, func, literal, select, text
from sqlalchemy import union as sa_union
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .filters import resolve
from .hop import Hop, Start
from .models import DEFAULT_GRAPH, Edge, Node


def _qualify(table) -> str:
    if table.schema:
        return f'"{table.schema}"."{table.name}"'
    return f'"{table.name}"'


def _plain(value):
    """Aggregate results as plain Python. The driver returns NUMERIC as
    Decimal, which json.dumps refuses -- and an aggregation result is
    exactly the kind of thing that gets serialized straight into a tool
    response. Integral values come back as int (a count of 3, not 3.0)."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value


@dataclass
class Subgraph:
    """The result of a traversal: every node and edge that is part of at
    least one complete, filter-satisfying chain."""
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {"nodes": self.nodes, "edges": self.edges, "elapsed_ms": self.elapsed_ms}

    def to_networkx(self, multigraph: bool = False):
        """Build an in-memory graph from the result.

        multigraph=True uses nx.MultiDiGraph, which preserves parallel
        edges between the same two nodes. Plain DiGraph (the default)
        silently collapses them -- pass multigraph=True if your data can
        have more than one real edge between the same pair of nodes and
        that distinction matters to you.
        """
        import networkx as nx
        g = (nx.MultiDiGraph if multigraph else nx.DiGraph)()
        for n in self.nodes:
            g.add_node(n["id"], **n["properties"])
        for e in self.edges:
            g.add_edge(e["start_id"], e["end_id"], **e["properties"])
        return g

    def __repr__(self) -> str:
        return f"Subgraph(nodes={len(self.nodes)}, edges={len(self.edges)}, elapsed_ms={self.elapsed_ms:.1f})"


class Graph:
    """Entry point. Wraps a SQLAlchemy engine (or a DSN string) and the
    two tables traversal runs against.

        graph = Graph("postgresql+psycopg2://user:pass@host/db")
        result = graph.traverse(
            Start(where={"type": "person"}),
            Hop(where={"active": True}, via={"kind": "friend"}, hops=(1, 4)),
        )
    """

    def __init__(
        self,
        dsn_or_engine,
        graph: str = DEFAULT_GRAPH,
        node_table=None,
        edge_table=None,
        node_id_col: str = "id",
        edge_id_col: str = "id",
        edge_start_col: str = "start_id",
        edge_end_col: str = "end_id",
        graph_col: Optional[str] = "graph_id",
    ):
        self.engine: Engine = (
            dsn_or_engine if isinstance(dsn_or_engine, Engine) else create_engine(dsn_or_engine)
        )
        if not isinstance(graph, str) or not graph:
            raise ValueError(f"graph must be a non-empty string, got {graph!r}")
        self.graph = graph
        self.nodes_tbl = node_table if node_table is not None else Node.__table__
        self.edges_tbl = edge_table if edge_table is not None else Edge.__table__
        self.node_id_col = node_id_col
        self.edge_id_col = edge_id_col
        self.edge_start_col = edge_start_col
        self.edge_end_col = edge_end_col
        self.graph_col = graph_col
        self._schema = None
        self._vectors = None
        if graph_col is not None:
            for table in (self.nodes_tbl, self.edges_tbl):
                if graph_col not in table.c:
                    raise ValueError(
                        f"table {table.name!r} has no {graph_col!r} column, so graphs cannot "
                        f"be kept apart in it. Add the column (see create_schema), or pass "
                        f"graph_col=None to run a single unscoped graph against these tables"
                    )

    def __repr__(self) -> str:
        return f"Graph({self.engine.url!r}, graph={self.graph!r})"

    # -- graph scoping ---------------------------------------------------

    def in_graph(self, graph: str) -> Graph:
        """The same tables and engine, scoped to a different graph.

        `Graph` is a cheap handle, so this is how you move between graphs
        rather than building a second engine and pool for each.

        The new handle starts with NO schema and NO vector fields -- a
        different graph is allowed a different shape (different vector
        dimensions included), so neither travels implicitly; call
        define_schema()/define_vectors() on the new handle."""
        return Graph(self.engine, graph=graph, node_table=self.nodes_tbl,
                     edge_table=self.edges_tbl, node_id_col=self.node_id_col,
                     edge_id_col=self.edge_id_col, edge_start_col=self.edge_start_col,
                     edge_end_col=self.edge_end_col, graph_col=self.graph_col)

    def _scoped(self, table):
        """`graph_id = <this graph>` for one table, or an alias of it.

        Every read and every write goes through this. A query that
        forgets it does not fail -- it silently returns or writes another
        graph's rows, which is the one bug this design can produce.

        graph_col=None means the caller brought their own tables with no
        discriminator: there is only one graph, and the predicate is a
        no-op rather than an error."""
        if self.graph_col is None:
            return literal(True)
        return table.c[self.graph_col] == self.graph

    # -- query building -----------------------------------------------

    # NOTE: the whole walk -- SQLAlchemy's CTE self-reference included --
    # lives in the single _walk_matches loop below rather than being
    # split across methods that would each need the same alias objects
    # passed around. It became a (private) helper only because
    # build_aggregate_query needs the exact same seed/walk/match chain;
    # there is still exactly one place the walk is built.

    def _walk_matches(self, start: Start, hops: list):
        """The seed CTE plus, per hop, its (full recursive walk, match)
        CTE pair. Shared by build_query and build_aggregate_query, so
        the two can never disagree about what a traversal matches."""
        from sqlalchemy.dialects.postgresql import array
        from sqlalchemy.sql.expression import any_ as sa_any_
        from sqlalchemy import not_ as sa_not_

        nt, et = self.nodes_tbl, self.edges_tbl
        edge_start_col = getattr(et.c, self.edge_start_col)
        edge_end_col = getattr(et.c, self.edge_end_col)
        edge_id_col = getattr(et.c, self.edge_id_col)
        node_id_col = getattr(nt.c, self.node_id_col)

        seed_condition = and_(self._scoped(nt), resolve(nt.c.properties, start.where))
        if start.near is not None:
            # Similarity-seeded: the seed CTE becomes "the k most
            # similar nodes that also pass `where`", ranked inside the
            # same statement -- see hopai/vectors.py. Everything
            # downstream (walks, matches, dead-end pruning) is
            # unchanged, which is the point of doing it in the seed.
            from .vectors import ranked_ids, validate_nears
            from .vectors import validate_boosts
            nears = validate_nears(self, "nodes", start.near, start.keep, "Start", "keep")
            seed = ranked_ids(self, nt, node_id_col, nt, seed_condition, nears, start.keep,
                              validate_boosts(start.boost, "Start")).cte("seed")
        else:
            seed = (
                select(node_id_col.label("node_id"))
                .where(seed_condition)
                .cte("seed")
            )
        prev_match = seed
        pairs = []

        for i, hop in enumerate(hops):
            if hop.direction == "forward":
                join_col, move_col = edge_start_col, edge_end_col
            else:
                join_col, move_col = edge_end_col, edge_start_col

            def _edge_cols(alias, direction=hop.direction):
                """(join, move, id) for one edge alias, given direction."""
                if direction == "forward":
                    join, move = self.edge_start_col, self.edge_end_col
                else:
                    join, move = self.edge_end_col, self.edge_start_col
                return (getattr(alias.c, join), getattr(alias.c, move),
                        getattr(alias.c, self.edge_id_col))

            via_nears = None
            if hop.via_near is not None:
                # Similarity-ranked edges. Each anchor row joins to a
                # LATERAL yielding the edges worth following FROM IT,
                # and that lateral hands back exactly the (edge_id,
                # to_id) pair the plain join produced -- so depth, the
                # local path, the cycle guard and edge reconstruction
                # are all untouched below.
                from .vectors import edge_beam, validate_nears
                via_nears = validate_nears(self, "edges", hop.via_near, hop.via_keep,
                                           f"hop {i} ({hop.label or 'unlabeled'}) via_near",
                                           "via_keep")

            if via_nears is not None:
                base_alias = et.alias(f"via_base_{i}")
                base_join, base_move, base_id = _edge_cols(base_alias)
                base_beam = edge_beam(self, base_alias, base_join, prev_match.c.node_id,
                                      base_move, base_id, hop.via, via_nears, hop.via_keep,
                                      f"beam_{i}", correlate=(prev_match,))
                walk_base = select(
                    prev_match.c.node_id.label("from_id"),
                    base_beam.c.move_id.label("to_id"),
                    literal(1).label("depth"),
                    array([prev_match.c.node_id, base_beam.c.move_id]).label("local_path"),
                    array([base_beam.c.edge_id]).label("local_edges"),
                ).select_from(prev_match.join(base_beam, literal(True)))
            else:
                walk_base = (
                    select(
                        prev_match.c.node_id.label("from_id"),
                        move_col.label("to_id"),
                        literal(1).label("depth"),
                        array([prev_match.c.node_id, move_col]).label("local_path"),
                        array([edge_id_col]).label("local_edges"),
                    )
                    .select_from(et.join(prev_match, join_col == prev_match.c.node_id))
                    .where(and_(self._scoped(et), resolve(et.c.properties, hop.via)))
                )
            walk = walk_base.cte(f"walk_{i}", recursive=True)
            w = walk.alias()

            if via_nears is not None:
                rec_alias = et.alias(f"via_rec_{i}")
                rec_join, rec_move, rec_id = _edge_cols(rec_alias)
                # The cycle guard goes INSIDE the beam, not after it: a
                # top-via_keep beam that spent slots on edges leading
                # back into the path would follow fewer than via_keep usable
                # edges, and how many is invisible from the outside.
                rec_beam = edge_beam(
                    self, rec_alias, rec_join, w.c.to_id, rec_move, rec_id, hop.via,
                    via_nears, hop.via_keep, f"beam_rec_{i}",
                    extra=[sa_not_(rec_move == sa_any_(w.c.local_path))],
                    correlate=(w,),
                )
                recursive_term = (
                    select(
                        w.c.from_id,
                        rec_beam.c.move_id,
                        w.c.depth + 1,
                        w.c.local_path.op("||")(rec_beam.c.move_id),
                        w.c.local_edges.op("||")(rec_beam.c.edge_id),
                    )
                    .select_from(w.join(rec_beam, literal(True)))
                    .where(w.c.depth < hop.max_hops)
                )
            else:
                e = et.alias()
                e_join, e_move, e_id = _edge_cols(e)
                recursive_term = (
                    select(
                        w.c.from_id,
                        e_move,
                        w.c.depth + 1,
                        w.c.local_path.op("||")(e_move),
                        w.c.local_edges.op("||")(e_id),
                    )
                    .select_from(w.join(e, e_join == w.c.to_id))
                    .where(
                        and_(
                            w.c.depth < hop.max_hops,
                            self._scoped(e),
                            resolve(e.c.properties, hop.via),
                            sa_not_(e_move == sa_any_(w.c.local_path)),
                        )
                    )
                )
            full_walk = walk.union_all(recursive_term)

            if hop.near is not None:
                # Rank AFTER deduplication: many walks can reach one
                # node, and its similarity is one number -- computing
                # it per walk row would multiply the per-row subquery
                # by the path count for the same answer.
                from .vectors import ranked_ids, validate_nears
                nears = validate_nears(self, "nodes", hop.near, hop.keep,
                                       f"hop {i} ({hop.label or 'unlabeled'})", "keep")
                reached = (
                    select(distinct(full_walk.c.to_id).label("node_id"))
                    .where(full_walk.c.depth >= hop.min_hops)
                    .subquery(f"reached_{i}")
                )
                joined = reached.join(
                    nt, and_(node_id_col == reached.c.node_id, self._scoped(nt),
                             resolve(nt.c.properties, hop.where))
                )
                from .vectors import validate_boosts
                match_i = ranked_ids(
                    self, nt, reached.c.node_id, joined, None, nears, hop.keep,
                    validate_boosts(hop.boost, f"hop {i} ({hop.label or 'unlabeled'})"),
                ).cte(f"match_{i}")
            else:
                match_i = (
                    select(distinct(full_walk.c.to_id).label("node_id"))
                    .select_from(
                        full_walk.join(
                            nt, and_(node_id_col == full_walk.c.to_id, self._scoped(nt),
                                     resolve(nt.c.properties, hop.where))
                        )
                    )
                    .where(full_walk.c.depth >= hop.min_hops)
                    .cte(f"match_{i}")
                )
            pairs.append((full_walk, match_i))
            prev_match = match_i

        return seed, pairs

    def build_query(self, start: Start, hops: list):
        nt, et = self.nodes_tbl, self.edges_tbl
        edge_start_col = getattr(et.c, self.edge_start_col)
        edge_end_col = getattr(et.c, self.edge_end_col)
        edge_id_col = getattr(et.c, self.edge_id_col)
        node_id_col = getattr(nt.c, self.node_id_col)

        n = len(hops)
        for i, hop in enumerate(hops):
            if hop.optional and i != n - 1:
                raise ValueError(
                    f"hop {i} ({hop.label or 'unlabeled'}): optional=True is only supported on "
                    f"the LAST hop in a chain. If you need multiple optional extensions, run "
                    f"separate queries."
                )

        seed, pairs = self._walk_matches(start, hops)
        prev_match = seed
        hop_edge_ctes = []
        pre_optional_match = None

        for i, (full_walk, match_i) in enumerate(pairs):
            hop = hops[i]
            if hop.optional:
                pre_optional_match = prev_match

            hop_edges_i = (
                select(distinct(func.unnest(full_walk.c.local_edges)).label("edge_id"))
                .where(
                    and_(
                        full_walk.c.depth >= hop.min_hops,
                        full_walk.c.to_id.in_(select(match_i.c.node_id)),
                    )
                )
                .cte(f"hop_edges_{i}")
            )
            hop_edge_ctes.append(hop_edges_i)
            prev_match = match_i

        if not hop_edge_ctes:
            return select(
                literal("node").label("kind"), cast(node_id_col, String).label("id")
            ).select_from(seed.join(nt, and_(node_id_col == seed.c.node_id, self._scoped(nt))))

        # One union() over all the hops, NOT a fold of .union() calls:
        # Select.union() returns a CompoundSelect, which has no .union()
        # of its own, so folding raised AttributeError on the third hop
        # and every chain longer than two was unrunnable.
        all_edges_cte = sa_union(*(c.select() for c in hop_edge_ctes)).cte("all_edges")

        edge_rows = (
            select(
                edge_id_col.label("eid"),
                edge_start_col.label("a"),
                edge_end_col.label("b"),
            )
            .select_from(et.join(all_edges_cte, and_(all_edges_cte.c.edge_id == edge_id_col,
                                                     self._scoped(et))))
            .cte("edge_rows")
        )

        node_selects = [
            select(literal("node").label("kind"), cast(edge_rows.c.a, String).label("id")),
            select(literal("node").label("kind"), cast(edge_rows.c.b, String).label("id")),
        ]
        edge_selects = [select(literal("edge").label("kind"), cast(edge_rows.c.eid, String).label("id"))]

        if hops[-1].optional and pre_optional_match is not None:
            node_selects.append(
                select(literal("node").label("kind"), cast(pre_optional_match.c.node_id, String).label("id"))
            )

        return node_selects[0].union(*node_selects[1:], *edge_selects)

    def build_aggregate_query(self, start: Start, hops: list, aggregates: dict):
        """The single statement Graph.aggregate() runs: the same
        seed/walk/match chain as build_query, with the aggregates
        computed over the final match instead of the subgraph being
        reported. None of the edge-reconstruction CTEs (`hop_edges_*`,
        `all_edges`, `edge_rows`) are emitted -- an aggregation needs no
        edges, so it is strictly less work than the traversal it
        summarizes."""
        from .aggregates import resolve_aggregate

        if not isinstance(aggregates, dict) or not aggregates:
            raise ValueError(
                "aggregates must be a non-empty dict naming each result, e.g. "
                "{'friends': Count(), 'avg_age': Avg('age')}"
            )
        for i, hop in enumerate(hops):
            if hop.optional:
                raise ValueError(
                    f"hop {i} ({hop.label or 'unlabeled'}): optional=True has no effect on "
                    f"an aggregation -- aggregates run over the nodes the last hop matched, "
                    f"and a chain the hop did not extend contributes nothing either way. "
                    f"Drop the flag, so nobody reads it as changing the number."
                )

        nt = self.nodes_tbl
        node_id_col = getattr(nt.c, self.node_id_col)
        seed, pairs = self._walk_matches(start, hops)
        # The final match: what the last hop reached, or the seed set when
        # there are no hops. Aggregating any OTHER position would count
        # nodes with no continuation to the end of the chain -- the
        # question Cypher's mid-chain aggregates do NOT answer, which is
        # why cypher.py refuses them rather than landing here.
        final = pairs[-1][1] if pairs else seed
        columns = [
            resolve_aggregate(nt.c.properties, agg).label(name)
            for name, agg in aggregates.items()
        ]
        return select(*columns).select_from(
            final.join(nt, and_(node_id_col == final.c.node_id, self._scoped(nt)))
        )

    # -- schema ---------------------------------------------------------

    def create_schema(self) -> None:
        """Create the two tables and the indexes traversal depends on.

        Idempotent, so it is safe to call on every start-up -- which is
        the point: a project should not need a migration tool or a
        hand-copied DDL block to get a working graph.

        The baseline indexes are not optional decoration. Without the
        btree indexes on the edge endpoints every hop is a sequential
        scan, and without the GIN indexes every property filter is."""
        self.nodes_tbl.create(self.engine, checkfirst=True)
        self.edges_tbl.create(self.engine, checkfirst=True)

        nodes, edges = _qualify(self.nodes_tbl), _qualify(self.edges_tbl)
        g = self.graph_col
        # graph_id LEADS both endpoint indexes. Every hop filters on it,
        # so a trailing position would make the index useless the moment
        # a second graph exists -- and the cost of the discriminator is
        # only acceptable because it is indexed away.
        statements = [
            f'CREATE INDEX IF NOT EXISTS "ix_{self.edges_tbl.name}_graph_{self.edge_start_col}" '
            f'ON {edges} ("{g}", "{self.edge_start_col}")',
            f'CREATE INDEX IF NOT EXISTS "ix_{self.edges_tbl.name}_graph_{self.edge_end_col}" '
            f'ON {edges} ("{g}", "{self.edge_end_col}")',
            f'CREATE INDEX IF NOT EXISTS "ix_{self.nodes_tbl.name}_graph" ON {nodes} ("{g}")',
            f'CREATE INDEX IF NOT EXISTS "ix_{self.nodes_tbl.name}_properties" '
            f'ON {nodes} USING GIN (properties)',
            f'CREATE INDEX IF NOT EXISTS "ix_{self.edges_tbl.name}_properties" '
            f'ON {edges} USING GIN (properties)',
        ]
        with self.engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))

    def drop_schema(self) -> None:
        """Drop both tables and everything on them. Edges first, for the
        foreign key."""
        self.edges_tbl.drop(self.engine, checkfirst=True)
        self.nodes_tbl.drop(self.engine, checkfirst=True)

    # -- constraints ----------------------------------------------------

    def _targets(self, nodes: Optional[list], edges: Optional[list]) -> list:
        from .constraints import _Target
        pairs = []
        if nodes:
            pairs.append((_Target(self.nodes_tbl, "nodes", self.graph, self.graph_col), nodes))
        if edges:
            pairs.append((_Target(self.edges_tbl, "edges", self.graph, self.graph_col), edges))
        return pairs

    def constraint_ddl(self, nodes: Optional[list] = None, edges: Optional[list] = None) -> list:
        """The exact SQL define_constraints() would run, without running
        it. For review, for a migration file, or for showing an agent what
        it is about to change."""
        from .constraints import compile_constraint
        return [ddl for target, group in self._targets(nodes, edges)
                for _, _, ddl in (compile_constraint(c, target) for c in group)]

    def define_constraints(self, nodes: Optional[list] = None,
                           edges: Optional[list] = None) -> list:
        """Declare constraints on node and edge properties. Idempotent:
        an existing constraint of the same name is left alone, so this
        belongs next to create_schema() in your start-up path.

        Returns the names created or already present, in order."""
        from .constraints import compile_constraint, constraint_exists

        applied = []
        with self.engine.begin() as connection:
            for target, group in self._targets(nodes, edges):
                for constraint in group:
                    kind, name, ddl = compile_constraint(constraint, target)
                    if kind == "check" and constraint_exists(connection, target, name):
                        applied.append(name)
                        continue
                    connection.execute(text(ddl))
                    applied.append(name)
        return applied

    def drop_constraints(self, nodes: Optional[list] = None,
                         edges: Optional[list] = None) -> list:
        """Drop the constraints these declarations describe. Missing ones
        are ignored, so this is the exact inverse of define_constraints()."""
        from .constraints import compile_constraint

        dropped = []
        with self.engine.begin() as connection:
            for target, group in self._targets(nodes, edges):
                for constraint in group:
                    kind, name, _ = compile_constraint(constraint, target)
                    if kind == "index":
                        connection.execute(text(f'DROP INDEX IF EXISTS "{name}"'))
                    else:
                        connection.execute(text(
                            f'ALTER TABLE {target.qualified} DROP CONSTRAINT IF EXISTS "{name}"'))
                    dropped.append(name)
        return dropped

    # -- graph schema ---------------------------------------------------

    def define_schema(self, nodes: Optional[list] = None, edges: Optional[list] = None,
                      schema=None):
        """Declare the shape of this graph: node types, their
        properties, and which edge kinds connect which node types.
        Entries are NodeType/EdgeType primitives or plain
        dataclass/pydantic classes -- see hopai/schema.py for both
        notations and the annotation mapping. `schema=` adopts an
        already-built GraphSchema instead -- the second step of the
        infer -> review -> define -> enforce loop.

        In memory only: nothing touches the database until
        enforce_schema(). Calling this again replaces the schema.
        Returns the normalized GraphSchema."""
        from .schema import GraphSchema, build_schema
        if schema is not None:
            if nodes is not None or edges is not None:
                raise ValueError(
                    "pass either schema= or nodes=/edges= -- schema= adopts a finished "
                    "GraphSchema as-is, so there is nothing for nodes/edges to add to it"
                )
            if not isinstance(schema, GraphSchema):
                raise TypeError(
                    f"schema= takes a GraphSchema (e.g. from infer_schema()), "
                    f"got {type(schema).__name__}"
                )
            self._schema = schema
            return schema
        self._schema = build_schema(nodes, edges)
        return self._schema

    def infer_schema(self) -> tuple:
        """Derive the schema from the rows this graph already holds:
        node types from `properties->>'type'`, edge kinds from
        `properties->>'kind'` plus observed endpoint pairs, required
        and nullable from presence counts. Returns
        (GraphSchema, InferenceReport) and registers NOTHING -- an
        inferred schema is an observation; adopting it as the contract
        is `define_schema(schema=inferred)`, deliberately separate.
        Read the report first: untyped rows, 42-vs-"42" conflicts, and
        per-type row counts live there, not in the schema.

        Full sequential scans, meant for start-up or migration -- see
        hopai/schema.py's INFERENCE section for semantics and cost."""
        from .schema import infer_schema
        return infer_schema(self)

    @property
    def schema(self):
        """The declared schema as canonical dataclasses, or None when
        define_schema() has not been called on this handle -- that is
        the existence check."""
        return self._schema

    @property
    def schema_json(self) -> dict:
        """The schema in JSON Schema vocabulary, json.dumps-clean --
        made to be pasted into a system prompt or a tool result."""
        return self._defined_schema("schema_json").to_json()

    @property
    def schema_networkx(self):
        """The schema as an nx.MultiDiGraph meta-graph: node types as
        nodes, edge kinds as edges. Needs the networkx extra."""
        return self._defined_schema("schema_networkx").to_networkx()

    @property
    def schema_pydantic(self) -> dict:
        """Generated pydantic models, one per node type and edge kind.
        Needs pydantic v2 -- pip install hopai[pydantic]."""
        return self._defined_schema("schema_pydantic").to_pydantic()

    def _defined_schema(self, wanted: str):
        if self._schema is None:
            raise ValueError(
                f"{wanted} needs a schema and none is defined for this Graph -- "
                f"call define_schema(...) first"
            )
        return self._schema

    def _schema_targets(self) -> tuple:
        from .constraints import _Target
        return (_Target(self.nodes_tbl, "nodes", self.graph, self.graph_col),
                _Target(self.edges_tbl, "edges", self.graph, self.graph_col))

    def tool_schemas(self) -> list:
        """The three LLM tool definitions -- traverse, aggregate, ingest
        -- as deep copies, with THIS graph's declared schema summarized
        into each description so a function-calling model sees what
        exists instead of guessing labels and property names. With no
        schema defined, the static definitions come back unchanged (as
        copies): the schema stays optional.

        Only descriptions differ from the module constants; the
        `parameters` sections are identical, because what the parsers
        accept has not changed -- this is presentation, not grammar."""
        import copy

        from .ingest import INGEST_TOOL_SCHEMA
        from .json_api import AGGREGATE_TOOL_SCHEMA, TRAVERSE_TOOL_SCHEMA
        tools = [copy.deepcopy(tool) for tool in
                 (TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA, INGEST_TOOL_SCHEMA)]
        if self._schema is not None:
            from .schema import tool_summary
            summary = tool_summary(self._schema)
            for tool in tools:
                tool["description"] = f"{tool['description']} {summary}"
        return tools

    def schema_violations(self, sample: int = 5):
        """What enforce_schema() would reject, found by READING -- per
        violated rule, its would-be constraint name, a row count and up
        to `sample` offending ids. Falsy when clean.

        This is the step between defining and enforcing on a graph that
        grew before the schema did: ADD CONSTRAINT validates existing
        rows and fails opaquely on the first bad one, while this returns
        the whole work list. Read-only -- no DDL, nothing registered."""
        from .schema import find_violations
        self._defined_schema("schema_violations()")
        return find_violations(self, sample)

    def schema_ddl(self, endpoints: bool = False) -> list:
        """The exact SQL enforce_schema() would run, without running it.
        For review, for a migration file, or for showing an agent what
        it is about to change -- the same contract as constraint_ddl().
        endpoints=True appends the endpoint-trigger DDL."""
        from .schema import (
            compile_edge_constraints, compile_endpoint_ddl, compile_node_constraints,
        )
        schema = self._defined_schema("schema_ddl()")
        node_target, edge_target = self._schema_targets()
        ddl = [statement for _, statement in (compile_node_constraints(schema, node_target)
                                              + compile_edge_constraints(schema, edge_target))]
        if endpoints and schema.edge_types:
            ddl.extend(compile_endpoint_ddl(schema, self))
        return ddl

    def enforce_schema(self, endpoints: bool = False) -> list:
        """Compile the declared schema to Postgres CHECK constraints, so
        EVERY write path is validated by the server -- add_nodes, merge,
        Cypher CREATE/MERGE, even SQL from another service -- and a
        violation surfaces as ConstraintViolation.

        A separate, explicit step rather than a side effect of
        define_schema(), because ADD CONSTRAINT validates every existing
        row -- on pre-schema data that can fail, and it should fail in a
        call whose name says it enforces.

        endpoints=True additionally polices the declared (kind, source,
        target) triples with a CONSTRAINT TRIGGER -- the one rule a
        CHECK cannot express, because it must look at the endpoint
        nodes. It fires per edge write, which is why it is an opt-in
        with its cost stated here rather than the default. It validates
        edges as they are written; retyping a NODE under existing edges
        is not re-checked.

        Idempotent, and it reconciles: a schema-derived constraint or
        trigger that the current call no longer produces is dropped, so
        re-running after a schema change -- or without endpoints=True --
        converges instead of accreting. Only objects this mechanism
        named (ck_schema_*) are ever touched; define_constraints()
        declarations are not its to drop. Returns the names now in
        force, in order."""
        from .schema import (
            ENDPOINT_TRIGGER_EXISTS, SCHEMA_CHECKS, compile_edge_constraints,
            compile_endpoint_ddl, compile_node_constraints, endpoint_names,
            schema_constraint_prefixes,
        )
        schema = self._defined_schema("enforce_schema()")
        node_target, edge_target = self._schema_targets()
        groups = [(node_target, compile_node_constraints(schema, node_target)),
                  (edge_target, compile_edge_constraints(schema, edge_target))]
        applied = []
        with self.engine.begin() as connection:
            for target, pairs in groups:
                existing = {row[0] for row in connection.execute(
                    SCHEMA_CHECKS, {"table": target.qualified}).all()}
                current = {name for name, _ in pairs}
                prefixes = schema_constraint_prefixes(target)
                for stale in sorted(n for n in existing
                                    if n.startswith(prefixes) and n not in current):
                    connection.execute(text(
                        f'ALTER TABLE {target.qualified} DROP CONSTRAINT "{stale}"'))
                for name, ddl in pairs:
                    if name not in existing:
                        connection.execute(text(ddl))
                    applied.append(name)

            trigger_name, function_name = endpoint_names(self)
            wanted = endpoints and bool(schema.edge_types)
            present = connection.execute(ENDPOINT_TRIGGER_EXISTS, {
                "name": trigger_name, "table": edge_target.qualified}).first() is not None
            function_q = (f'"{self.edges_tbl.schema}"."{function_name}"'
                          if self.edges_tbl.schema else f'"{function_name}"')
            if wanted:
                for ddl in compile_endpoint_ddl(schema, self):
                    connection.execute(text(ddl))
                applied.append(trigger_name)
            elif present:
                connection.execute(text(
                    f'DROP TRIGGER IF EXISTS "{trigger_name}" ON {edge_target.qualified}'))
                connection.execute(text(f'DROP FUNCTION IF EXISTS {function_q}()'))
        return applied

    # -- vectors --------------------------------------------------------

    def define_vectors(self, nodes: Optional[list] = None, edges: Optional[list] = None) -> dict:
        """Declare named vector fields: Vector(name, dimensions)
        entries per target. The migration they imply is a separate,
        explicit migrate_vectors(), because ALTER TABLE should be a
        conscious moment. Calling this again replaces the declaration.

        Not purely in-memory, unlike define_schema(): the vec_*
        columns are attached to the SHARED SQLAlchemy Table metadata,
        so create_schema() on any handle for these tables emits them
        from now on. Nothing else touches the database until
        migrate_vectors().

            graph.define_vectors(nodes=[Vector("summary", 1536)])
            graph.migrate_vectors()

        Returns the normalized registry; hopai/vectors.py explains the
        storage model and every refusal."""
        from .vectors import attach_columns, build_registry
        self._vectors = build_registry(nodes, edges)
        attach_columns(self)
        return self._vectors

    @property
    def vectors(self) -> Optional[dict]:
        """The declared vector fields as {"nodes": {name: Vector},
        "edges": {...}}, or None when define_vectors() has not been
        called on this handle -- that is the existence check.

        A copy: handing out the live registry made editing the
        returned dict silently redeclare the graph."""
        if self._vectors is None:
            return None
        return {target: dict(fields) for target, fields in self._vectors.items()}

    def vector_ddl(self) -> list:
        """The exact SQL migrate_vectors() would run, without running
        it -- the same contract as constraint_ddl() and schema_ddl()."""
        from .vectors import vector_ddl
        return vector_ddl(self)

    def migrate_vectors(self) -> list:
        """Apply the declared vector fields: one nullable real[]
        column per field plus a per-graph dimension CHECK. Idempotent
        and drift-refusing -- see hopai/vectors.py. Belongs next to
        create_schema() in your start-up path once vectors are in use."""
        from .vectors import migrate_vectors
        return migrate_vectors(self)

    def drop_vectors(self, node_fields: Optional[list] = None,
                     edge_fields: Optional[list] = None) -> list:
        """Drop the named fields FOR THIS GRAPH: their dimension
        constraints go, their values in this graph's rows are set to
        NULL. The shared columns stay -- other graphs may use them."""
        from .vectors import drop_vectors
        return drop_vectors(self, node_fields, edge_fields)

    def set_vectors(self, nodes: Optional[list] = None, edges: Optional[list] = None) -> int:
        """Store vectors on existing rows, one transaction for the
        whole call. Each row is {"id": ..., <field>: <vector or None>}:

            graph.set_vectors(nodes=[{"id": 1, "summary": embedding}])

        This is the ONLY write path for vectors -- add_nodes/merge
        rows never carry them (hopai/vectors.py says why). Returns the
        number of rows updated; an id matching no row in this graph
        fails the whole call by name."""
        from .vectors import set_vectors
        return set_vectors(self, nodes, edges)

    def get_vectors(self, node_ids: Optional[list] = None, edge_ids: Optional[list] = None,
                    node_fields: Optional[list] = None,
                    edge_fields: Optional[list] = None) -> dict:
        """Read stored vectors back by id -- traversal and search
        results never include them (6KB of floats has no business in
        every subgraph). String or integer ids are both accepted.

            graph.get_vectors(node_ids=[1, 2])
            # -> {"nodes": {"1": {"summary": [...]}}, "edges": {}}

        `node_ids`/`edge_ids`, not `nodes`/`edges`: every OTHER
        *_vectors call takes field names or rows there, and passing a
        field name to this one used to surface as a raw driver error
        about bigint. The field filters are per target for the same
        reason -- node and edge field names are separate namespaces."""
        from .vectors import get_vectors
        return get_vectors(self, node_ids, edge_ids, node_fields, edge_fields)

    def stale_vectors(self, node_fields=None, edge_fields=None, limit=None) -> dict:
        """Which rows need (re-)embedding, per field: those with no
        vector, and those whose stored vector no longer matches the
        declared dimensions.

            for node_id in graph.stale_vectors()["nodes"]["summary"]["missing"]:
                graph.set_vectors(nodes=[{"id": node_id, "summary": embed(...)}])

        The second category is the window a dimension change opens --
        migrate_vectors() refuses to reinterpret stored vectors and
        set_vectors() refuses to write the wrong size, so this is what
        closes it without hand-writing the catalog query."""
        from .vectors import stale_vectors
        return stale_vectors(self, node_fields, edge_fields, limit)

    def pgvector_exit_ddl(self, index: Optional[str] = "hnsw") -> list:
        """The migration off this library's exact search and onto
        pgvector, as SQL to read before running -- generated without
        importing, requiring or checking for the extension.

        Outgrowing hopai's exact scan should be a documented door
        rather than a rewrite. Read hopai/vectors.py's docstring on
        this first: the conversion is one-way, it makes the search
        approximate, and the column is shared by every graph in the
        table."""
        from .vectors import pgvector_exit_ddl
        return pgvector_exit_ddl(self, index=index)

    def build_vector_search_query(self, *near, target: str = "nodes", k: int = 10,
                                  where=None, boost=None):
        """The single statement vector_search() runs, for inspection
        with no database -- the same contract as build_query()."""
        from .vectors import build_search_query
        return build_search_query(self, list(near), target=target, k=k, where=where,
                                  boost=boost)

    def build_vector_search_many_query(self, queries, target: str = "nodes", k: int = 10,
                                       where=None, boost=None):
        """The single statement vector_search_many() runs."""
        from .vectors import build_search_many_query
        return build_search_many_query(self, queries, target=target, k=k, where=where,
                                       boost=boost)

    def vector_search_many(self, queries, target: str = "nodes", k: int = 10, where=None,
                           boost=None) -> list:
        """Rank several queries in ONE round trip, returning one result
        list per query, in order:

            graph.vector_search_many([Near("summary", q1), Near("summary", q2)], k=5)
            # -> [[...5 hits for q1...], [...5 hits for q2...]]

        This is the shape retrieval actually has -- a question expanded
        into several sub-queries -- and it costs ONE round trip instead
        of N. It does not reduce the arithmetic: every query still
        scores every candidate (measured at 1.08x against a loop on a
        local database), so the win is latency, which is where the
        cost actually is when the database is not on localhost. Each
        entry may
        itself be a list of Near specs (a multivector query); every
        query must share the same field shape, which the call refuses
        rather than silently ranking them differently."""
        from .vectors import search_many
        return search_many(self, queries, target=target, k=k, where=where, boost=boost)

    def vector_search(self, *near, target: str = "nodes", k: int = 10, where=None,
                      boost=None) -> list:
        """Exact cosine similarity search over this graph's nodes or
        edges -- one statement, no extension, no approximation.

            graph.vector_search(Near("summary", embedding), k=10,
                                where={"type": "person"})
            # -> [{"id": "1", "similarity": 0.93, "properties": {...}}, ...]

        Several Near specs combine into one weighted score
        (multivector search); `where` is the same filter language as
        traversal and is applied BEFORE ranking, so a selective filter
        makes the search cheaper, never slower. Edge results also
        carry start_id/end_id. Rows are ordered most-similar first;
        ids are strings, like every other result. `boost` adds
        property terms to the score for hybrid retrieval; a boosted
        `similarity` is the combined score and can exceed 1, and
        reordering with `k` changes which rows come back. The cost
        model and every design refusal live in hopai/vectors.py."""
        from .vectors import search
        return search(self, list(near), target=target, k=k, where=where, boost=boost)

    # -- writing --------------------------------------------------------

    @property
    def _ingestor(self):
        from .ingest import Ingestor
        if not hasattr(self, "_ingestor_cache"):
            self._ingestor_cache = Ingestor(self)
        return self._ingestor_cache

    def add_nodes(self, rows: list) -> int:
        """Insert nodes. Each row is `{"id": ..., **properties}` or
        `{"id": ..., "properties": {...}}`; `id` may be omitted and
        generated. Returns the number written."""
        return self._ingestor.add_nodes(rows)

    def add_edges(self, rows: list) -> int:
        """Insert edges. Endpoints are given as `start_id`/`end_id`, or as
        `start`/`end` property dicts matching exactly one existing node
        each. Returns the number written."""
        return self._ingestor.add_edges(rows)

    def merge_nodes(self, rows: list, on: list, replace: bool = False) -> int:
        """Insert nodes, updating any that already match on `on`.
        Requires a unique index over those keys -- see Unique()."""
        return self._ingestor.merge_nodes(rows, on=on, replace=replace)

    def merge_edges(self, rows: list, on: list, replace: bool = False) -> int:
        """Insert edges, updating any that already match on `on`."""
        return self._ingestor.merge_edges(rows, on=on, replace=replace)

    def ingest(self, document: dict, merge_nodes_on: Optional[list] = None,
               merge_edges_on: Optional[list] = None):
        """Write a whole `{"nodes": [...], "edges": [...]}` document.

        Nodes are written before edges, so an edge in the same document
        may reference a node created by it. This is the call an agent
        makes -- see INGEST_TOOL_SCHEMA."""
        return self._ingestor.ingest(document, merge_nodes_on, merge_edges_on)

    def write_cypher(self, query: str, **options):
        """Run a Cypher CREATE/MERGE. Returns an IngestResult.

            graph.write_cypher('''
                CREATE (a:person {email: 'a@x.com'})-[:knows]->(b:person {email: 'b@x.com'})
            ''')

        The whole query is one transaction, and it compiles down to the
        same add_nodes/merge_nodes/add_edges the Python API calls --
        `graph.cypher_operations(query)` shows the plan without running
        it. Accepts the same node_label_key / edge_type_key options as
        the read side."""
        from .cypher import cypher_to_operations
        return self._ingestor.execute_operations(cypher_to_operations(query, **options))

    def cypher_operations(self, query: str, **options) -> list:
        """The ingestion plan a Cypher write compiles to, without running
        it -- for review, logging, or showing an agent what it is about
        to change."""
        from .cypher import cypher_to_operations
        return cypher_to_operations(query, **options)

    def cypher(self, query: str, **options):
        """Run any supported Cypher: reading, writing, or aggregating.

        Returns a Subgraph for a query that matches, an IngestResult for
        one that creates or merges, and a plain dict of numbers for one
        whose RETURN aggregates. Which one it is is visible in the
        query, and this is the entry point anyone arriving from a Neo4j
        driver reaches for first; traverse_cypher(), write_cypher() and
        aggregate_cypher() are the same thing when you would rather be
        explicit."""
        from .cypher import (
            _Parser, _ReturnClause, _tokenize, _WriteClause, aggregate_cypher,
            traverse_cypher,
        )
        clauses = _Parser(_tokenize(query)).parse()
        if any(isinstance(c, _WriteClause) for c in clauses):
            return self.write_cypher(query, **options)
        if any(isinstance(c, _ReturnClause) for c in clauses):
            return aggregate_cypher(self, query, **options)
        return traverse_cypher(self, query, **options)

    def add_networkx(self, nx_graph):
        """Load a networkx graph. Node keys become ids, and node/edge
        attribute dicts become properties -- the inverse of
        Subgraph.to_networkx()."""
        return self._ingestor.add_networkx(nx_graph)

    # -- execution ------------------------------------------------------

    def aggregate(self, start: Start, *hops: Hop, aggregates: dict) -> dict:
        """Aggregate over the nodes a traversal matches, without
        hydrating them. `start` and the hops mean exactly what they mean
        in traverse(); the aggregates run over the distinct nodes the
        LAST hop matched (the seed set when there are no hops), each one
        counted once however many paths reach it.

            graph.aggregate(
                Start(where={"type": "person"}),
                Hop(via={"kind": "friend"}, hops=(1, 4)),
                aggregates={"friends": Count(), "avg_age": Avg("age")},
            )
            # -> {"friends": 42, "avg_age": 31.5}

        Returns plain JSON-serializable values: count -> int, sum -> a
        number (0 when nothing matched), avg/min/max -> a number or None
        when nothing matched. One statement, one round trip."""
        query = self.build_aggregate_query(start, list(hops), aggregates)
        with Session(self.engine) as session:
            row = session.execute(query).one()
        return {name: _plain(value) for name, value in zip(aggregates, row, strict=True)}

    def traverse(self, start: Start, *hops: Hop) -> Subgraph:
        """Run one traversal. `start` picks the seed set; each `Hop`
        after it is one step of the walk. Returns a Subgraph with every
        node and edge on at least one complete matching chain."""
        hops = list(hops)
        query = self.build_query(start, hops)

        with Session(self.engine) as session:
            t0 = time.perf_counter()
            rows = session.execute(query).all()
            node_ids = [r.id for r in rows if r.kind == "node"]
            edge_ids = [r.id for r in rows if r.kind == "edge"]

            nt, et = self.nodes_tbl, self.edges_tbl
            node_id_col = getattr(nt.c, self.node_id_col)
            edge_id_col = getattr(et.c, self.edge_id_col)

            nodes = []
            if node_ids:
                q = select(cast(node_id_col, String).label("id"), nt.c.properties).where(
                    and_(self._scoped(nt), cast(node_id_col, String).in_(node_ids))
                )
                nodes = [{"id": r.id, "properties": r.properties} for r in session.execute(q).all()]

            edges = []
            if edge_ids:
                q = select(
                    cast(getattr(et.c, self.edge_start_col), String).label("start_id"),
                    cast(getattr(et.c, self.edge_end_col), String).label("end_id"),
                    et.c.properties,
                ).where(and_(self._scoped(et), cast(edge_id_col, String).in_(edge_ids)))
                edges = [
                    {"start_id": r.start_id, "end_id": r.end_id, "properties": r.properties}
                    for r in session.execute(q).all()
                ]

            elapsed_ms = (time.perf_counter() - t0) * 1000

        return Subgraph(nodes=nodes, edges=edges, elapsed_ms=elapsed_ms)
