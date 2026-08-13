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

from sqlalchemy import String, and_, cast, create_engine, distinct, func, literal, select
from sqlalchemy import union as sa_union
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .filters import resolve
from .hop import Hop, Start
from .models import Edge, Node


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
        node_table=None,
        edge_table=None,
        node_id_col: str = "id",
        edge_id_col: str = "id",
        edge_start_col: str = "start_id",
        edge_end_col: str = "end_id",
    ):
        self.engine: Engine = (
            dsn_or_engine if isinstance(dsn_or_engine, Engine) else create_engine(dsn_or_engine)
        )
        self.nodes_tbl = node_table if node_table is not None else Node.__table__
        self.edges_tbl = edge_table if edge_table is not None else Edge.__table__
        self.node_id_col = node_id_col
        self.edge_id_col = edge_id_col
        self.edge_start_col = edge_start_col
        self.edge_end_col = edge_end_col

    def __repr__(self) -> str:
        return f"Graph({self.engine.url!r})"

    # -- query building -----------------------------------------------

    # NOTE: the walk-building logic lives directly in build_query below,
    # not split into a helper -- SQLAlchemy's CTE self-reference (the
    # recursive term referring back to the CTE it's part of) reads more
    # clearly kept in one place than split across methods that each need
    # the same alias objects passed around.

    def build_query(self, start: Start, hops: list):
        from sqlalchemy.dialects.postgresql import array
        from sqlalchemy.sql.expression import any_ as sa_any_
        from sqlalchemy import not_ as sa_not_

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

        seed = (
            select(node_id_col.label("node_id"))
            .where(resolve(nt.c.properties, start.where))
            .cte("seed")
        )
        prev_match = seed
        hop_edge_ctes = []
        pre_optional_match = None

        for i, hop in enumerate(hops):
            if hop.optional:
                pre_optional_match = prev_match

            if hop.direction == "forward":
                join_col, move_col = edge_start_col, edge_end_col
            else:
                join_col, move_col = edge_end_col, edge_start_col

            walk_base = (
                select(
                    prev_match.c.node_id.label("from_id"),
                    move_col.label("to_id"),
                    literal(1).label("depth"),
                    array([prev_match.c.node_id, move_col]).label("local_path"),
                    array([edge_id_col]).label("local_edges"),
                )
                .select_from(et.join(prev_match, join_col == prev_match.c.node_id))
                .where(resolve(et.c.properties, hop.via))
            )
            walk = walk_base.cte(f"walk_{i}", recursive=True)
            w = walk.alias()
            e = et.alias()
            if hop.direction == "forward":
                e_join, e_move = getattr(e.c, self.edge_start_col), getattr(e.c, self.edge_end_col)
            else:
                e_join, e_move = getattr(e.c, self.edge_end_col), getattr(e.c, self.edge_start_col)

            recursive_term = (
                select(
                    w.c.from_id,
                    e_move,
                    w.c.depth + 1,
                    w.c.local_path.op("||")(e_move),
                    w.c.local_edges.op("||")(getattr(e.c, self.edge_id_col)),
                )
                .select_from(w.join(e, e_join == w.c.to_id))
                .where(
                    and_(
                        w.c.depth < hop.max_hops,
                        resolve(e.c.properties, hop.via),
                        sa_not_(e_move == sa_any_(w.c.local_path)),
                    )
                )
            )
            full_walk = walk.union_all(recursive_term)

            match_i = (
                select(distinct(full_walk.c.to_id).label("node_id"))
                .select_from(
                    full_walk.join(
                        nt, and_(node_id_col == full_walk.c.to_id, resolve(nt.c.properties, hop.where))
                    )
                )
                .where(full_walk.c.depth >= hop.min_hops)
                .cte(f"match_{i}")
            )

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
            return select(literal("node").label("kind"), cast(node_id_col, String).label("id")).select_from(
                seed.join(nt, node_id_col == seed.c.node_id)
            )

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
            .select_from(et.join(all_edges_cte, all_edges_cte.c.edge_id == edge_id_col))
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

    # -- execution ------------------------------------------------------

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
                    cast(node_id_col, String).in_(node_ids)
                )
                nodes = [{"id": r.id, "properties": r.properties} for r in session.execute(q).all()]

            edges = []
            if edge_ids:
                q = select(
                    cast(getattr(et.c, self.edge_start_col), String).label("start_id"),
                    cast(getattr(et.c, self.edge_end_col), String).label("end_id"),
                    et.c.properties,
                ).where(cast(edge_id_col, String).in_(edge_ids))
                edges = [
                    {"start_id": r.start_id, "end_id": r.end_id, "properties": r.properties}
                    for r in session.execute(q).all()
                ]

            elapsed_ms = (time.perf_counter() - t0) * 1000

        return Subgraph(nodes=nodes, edges=edges, elapsed_ms=elapsed_ms)
