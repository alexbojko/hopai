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
from dataclasses import replace as _respec
from decimal import Decimal

from typing import Optional, Union

from sqlalchemy import String, and_, cast, create_engine, distinct, func, literal, select, text
from sqlalchemy import union as sa_union
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex

from .filters import resolve
from .hop import Hop, Start
from .models import DEFAULT_GRAPH, EDGE_IDENTITY_KEYS, NODE_IDENTITY_KEYS, Edge, Node
from .vectors import VECTOR_COLUMN_PREFIX


def _extra_columns(table, reserved: set, graph_col: Optional[str]) -> tuple:
    """Every column on `table` this library does not already have a use
    for: not one of the identity/reserved names, and not a `vec_*`
    similarity field (define_vectors() may attach those to this very
    Table object, before or after this call -- they keep their own
    write path, set_vectors(), and must never be treated as a plain
    extra column). Column order, so output is deterministic."""
    names = set(reserved) | ({graph_col} if graph_col is not None else set())
    return tuple(c.name for c in table.columns
                 if c.name not in names and not c.name.startswith(VECTOR_COLUMN_PREFIX))


def _pinned(id_expr, pins: Optional[dict], index: int) -> tuple:
    """`(id IN (...),)` for a pinned step, or `()` for every other one.

    A tuple rather than a condition-or-None so the call sites splat it:
    `and_(a, b, *_pinned(...))` with no pins is `and_(a, b)`, the exact
    expression that was there before pinning existed, which is what
    keeps a rerank-less traversal's SQL byte-identical."""
    if not pins or index not in pins:
        return ()
    return (id_expr.in_(pins[index]),)


def _rerank_steps(start: Start, hops: list) -> list:
    """Every step carrying a rerank=, in CHAIN order: (index, spec).

    Index -1 is the Start and 0.. are the hops -- the same numbering
    vectors._near_sites() uses, so a step is named the same whichever
    layer is talking about it. Chain order is not cosmetic: hop N+1's
    candidates are whatever hop N's rerank left behind, so these are
    inherently serial and must be walked front to back."""
    steps = [(-1, start)] if start.rerank is not None else []
    steps.extend((i, hop) for i, hop in enumerate(hops) if hop.rerank is not None)
    return steps


def _step_owner(index: int, hops: list) -> str:
    """How a step is named in a refusal -- build_query()'s own wording
    for a hop, so one traversal never gets two spellings of hop 1."""
    return "Start" if index < 0 else f"hop {index} ({hops[index].label or 'unlabeled'})"


def _reads_paths(document_from: str, unknown: bool) -> bool:
    """Whether a document_from filter reads `.paths`, answering
    `unknown` when the subset cannot parse it at all.

    Path context costs a wider probe and a second hydration, so the
    common filter -- one that only reads the node's own properties --
    must not pay for it. `jqsafe.paths_read()` answers this by parsing,
    and OVER-reports rather than under-reports where it cannot be sure.

    The two callers want opposite answers for a filter the grammar
    rejects, which is why this takes the fallback rather than choosing
    one. HYDRATION asks with unknown=True: hydrating paths nobody reads
    is waste, missing paths somebody reads is a document built from
    `null`. The Start refusal asks with unknown=False, because such a
    filter already has a refusal of its own coming -- from
    build_documents(), with `document_from` as the owner and the
    offending construct named -- and answering "it reads .paths" there
    would blame something the caller never wrote."""
    from . import jqsafe
    try:
        read = jqsafe.paths_read(document_from)
    except Exception:
        return unknown
    return any(path == "paths" or path.startswith("paths.") for path in read)


def _rerank_documents(candidates: list, rerank) -> list:
    """(node id, candidate JSON) per DOCUMENT the provider will see.

    Default -- one document per distinct NODE, quoting every path that
    reached it. This follows the decision `_walk_matches()` already
    recorded for Near at a hop ("Rank AFTER deduplication..."): many
    walks reach one node, and paying per walk buys nothing when the
    answer is one number per node. The reranker also sees every route at
    once, which is more evidence than any single path carries.

    per_path=True -- one document per (node, path), because a node can
    genuinely read differently depending on the route that found it.
    Opt-in precisely because it costs |paths| documents instead of
    |nodes|, and the bill is per document."""
    if not rerank.per_path:
        return list(candidates)
    expanded = []
    for node_id, candidate in candidates:
        paths = candidate.get("paths")
        if not paths:
            expanded.append((node_id, candidate))
            continue
        expanded.extend((node_id, {**candidate, "paths": [path]}) for path in paths)
    return expanded


def _rerank_survivors(units: list, scores: list, keep: Optional[int]) -> list:
    """The ids that survive one reranked step, best first.

    A node's score is the MAX over its documents -- not the sum, not the
    mean -- so under per_path=True one strong route is enough to keep
    it. That is the same "any valid parent counts" semantics
    test_fan_in_both_parents_preserved exists to protect, and it is why
    a node survives or is dropped as a UNIT: every in-edge from a
    surviving parent is still reported, because the walk is re-run
    normally afterwards and derives its edges the way it always did.

    `keep=None` keeps them all, and the step is still narrowed: the
    probe fetched `rerank.candidates` of them, which is the input bound
    doing its job. Reordering alone changes nothing downstream, since a
    traversal is a subgraph rather than a ranking."""
    best: dict = {}
    for (node_id, _), score in zip(units, scores, strict=True):
        value = float(score)
        if node_id not in best or value > best[node_id]:
            best[node_id] = value
    ranked = sorted(best, key=lambda node_id: (-best[node_id], str(node_id)))
    return ranked if keep is None else ranked[:keep]


def rerank_plan(start: Start, hops: list) -> dict:
    """{step index: the text its Rerank scores against}, empty when
    nothing reranks -- and every pre-flight refusal, raised before a
    single statement runs.

    Called first by traverse()/aggregate() and, on the async path, by
    AsyncGraph BEFORE it embeds: `Near._with_vector()` hands back a spec
    carrying floats and no text, so the query has to be read while it is
    still there.

    The empty case returns before it imports anything, because every
    traversal in the library now runs through here and a chain with no
    rerank= must pay one attribute test for the privilege."""
    steps = _rerank_steps(start, hops)
    if not steps:
        return {}
    from .vectors import rerank_query_text

    plan = {}
    for index, spec in steps:
        owner = _step_owner(index, hops)
        if index < 0 and _reads_paths(spec.rerank.document_from, unknown=False):
            # A seed has no provenance -- nothing reached it -- so
            # `.paths` there would be `null` and every document would
            # quietly change shape. Refused at query time rather than
            # producing one, and named where the caller wrote it.
            raise ValueError(
                f"{owner}: document_from={spec.rerank.document_from!r} reads .paths, but a "
                f"seed has no provenance -- nothing reached it, so there is no path to "
                f"quote. `.paths` exists at a Hop and not at a Start. Move the rerank= to "
                f"the Hop that reaches these nodes, or build the seed's document from its "
                f"own properties"
            )
        plan[index] = rerank_query_text(spec.near, spec.rerank, spec.keep, owner,
                                        k_name="keep")
    return plan


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
    least one complete, filter-satisfying chain.

        nodes -> [{"id": "1", "properties": {...}}]
        edges -> [{"id": "7", "start_id": "1", "end_id": "2",
                   "properties": {...}}]

    Ids are strings everywhere, matching vector_search() and every
    *_vectors() call, so an edge found by traversal feeds straight into
    set_vectors(edges=[...]) -- which used to need a hand-written
    `SELECT id FROM edges`, since the edge id was the one identity this
    result dropped. It also tells parallel edges apart: two `friend`
    edges between one pair with identical properties were otherwise two
    identical dicts.

    Both lists therefore carry ids that already exist, so writing a
    result back with add_nodes/add_edges asks for those ids again and
    the primary key refuses. Drop them to copy a subgraph elsewhere:
    `add_edges([{k: v for k, v in e.items() if k != "id"} for e in
    result.edges])`. That has always been true of `nodes`; `edges` now
    says the same thing rather than being the one shape you could feed
    back by accident."""
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

        Extra columns (see Graph's node_extra_cols/edge_extra_cols) ride
        along as node/edge attributes beside the properties, the same
        flat namespace `id` already occupies as the node key -- so an
        extra column named the same as a property wins, exactly as `id`
        already would. add_networkx() is the inverse."""
        import networkx as nx
        g = (nx.MultiDiGraph if multigraph else nx.DiGraph)()
        for n in self.nodes:
            extra = {k: v for k, v in n.items() if k not in ("id", "properties")}
            g.add_node(n["id"], **{**n["properties"], **extra})
        for e in self.edges:
            extra = {k: v for k, v in e.items() if k not in ("start_id", "end_id", "properties")}
            g.add_edge(e["start_id"], e["end_id"], **{**e["properties"], **extra})
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

    A custom `node_table=`/`edge_table=` may carry columns beyond the
    ones above -- a foreign key to a `users` table, say. Every such
    column is discovered automatically (`node_extra_cols` /
    `edge_extra_cols`, computed once here) and from then on behaves
    like `id` or `start_id`: written by add_nodes()/add_edges()/
    merge_nodes()/merge_edges() when a row names it, and returned by
    traverse(). See models.py's "EXTENDING THE MODEL" note.
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
        self.nodes_tbl = node_table if node_table is not None else Node
        self.edges_tbl = edge_table if edge_table is not None else Edge
        self.node_id_col = node_id_col
        self.edge_id_col = edge_id_col
        self.edge_start_col = edge_start_col
        self.edge_end_col = edge_end_col
        self.graph_col = graph_col
        self._schema = None
        self._vectors = None
        # Set True only by in_graph() (see its docstring): lets
        # search()/search_many() populate this handle's registry from
        # the database, on first use, instead of leaving it blank for
        # the handle's whole life. A plain Graph() keeps this False, so
        # "you forgot define_vectors()" still refuses immediately and
        # by name rather than silently trying the catalog first.
        self._vectors_lazy = False
        if graph_col is not None:
            for table in (self.nodes_tbl, self.edges_tbl):
                if graph_col not in table.c:
                    raise ValueError(
                        f"table {table.name!r} has no {graph_col!r} column, so graphs cannot "
                        f"be kept apart in it. Add the column (see create_schema), or pass "
                        f"graph_col=None to run a single unscoped graph against these tables"
                    )
        self.node_extra_cols = _extra_columns(
            self.nodes_tbl, NODE_IDENTITY_KEYS | {self.node_id_col}, graph_col)
        self.edge_extra_cols = _extra_columns(
            self.edges_tbl,
            EDGE_IDENTITY_KEYS | {self.edge_id_col, self.edge_start_col, self.edge_end_col},
            graph_col)

    def __repr__(self) -> str:
        return f"Graph({self.engine.url!r}, graph={self.graph!r})"

    # -- graph scoping ---------------------------------------------------

    def in_graph(self, graph: str) -> Graph:
        """The same tables and engine, scoped to a different graph.

        `Graph` is a cheap handle, so this is how you move between graphs
        rather than building a second engine and pool for each.

        The new handle starts with NO schema and NO vector DECLARATION --
        a different graph is allowed a different shape (different vector
        dimensions included), so neither travels implicitly; call
        define_schema()/define_vectors() on the new handle to state one
        up front.

        Vectors are the one exception to "starts blank", and only
        LAZILY: the vec_* columns are physical storage SHARED by every
        graph in the table (only the dimension CHECK is per-graph), so
        a field this new handle's own graph already had migrated by
        some OTHER handle is not actually unknown, just not yet read
        back. Rather than eagerly querying the catalog here -- which
        would cost every in_graph() call a round trip even when nothing
        that follows ever touches a vector, and would make an offline
        Graph() (query building never connects; see the module docstring
        and Graph.build_query()) try to connect just from being handed a
        new graph name -- the returned handle only sets a flag
        (`_vectors_lazy`). vector_search()/vector_search_many() check it
        and call load_vectors() themselves, once, the first time they
        actually need a connection anyway. Anything else that needs
        vectors (set_vectors(), migrate_vectors(), ...) is unaffected by
        the flag and still refuses by name until you call
        define_vectors() or load_vectors() explicitly -- the lazy path
        exists for the READ side this issue was filed about, not as a
        blanket auto-declare."""
        handle = Graph(self.engine, graph=graph, node_table=self.nodes_tbl,
                       edge_table=self.edges_tbl, node_id_col=self.node_id_col,
                       edge_id_col=self.edge_id_col, edge_start_col=self.edge_start_col,
                       edge_end_col=self.edge_end_col, graph_col=self.graph_col)
        handle._vectors_lazy = True
        return handle

    def graphs(self) -> list:
        """Every graph that has rows in these tables, in name order.

        The counterpart to in_graph(): that moves to a graph you can
        name, this says which names there are. What it answers is "what
        is actually in this database", which is not the same question as
        "what did someone configure" -- a server or a UI that wants the
        second must keep its own list, because the DSN is the whole
        boundary here. Anyone who can run this query can already read
        every one of these graphs.

        Derived from `nodes`: an edge cannot exist without its endpoints
        (the composite foreign key sees to that), so a graph with rows
        has nodes. A graph that exists only as a saved schema, with no
        rows yet, is NOT here -- nothing has been written to it.

        With graph_col=None the tables carry no discriminator, so there
        is exactly one graph and this returns its name without querying.
        """
        if self.graph_col is None:
            return [self.graph]
        column = self.nodes_tbl.c[self.graph_col]
        with self.engine.connect() as connection:
            return [row[0] for row in connection.execute(
                select(column).distinct().order_by(column))]

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

    def _walk_matches(self, start: Start, hops: list, pins: Optional[dict] = None):
        """The seed CTE plus, per hop, its (full recursive walk, match)
        CTE pair. Shared by build_query and build_aggregate_query, so
        the two can never disagree about what a traversal matches.

        `pins` narrows a step to an explicit id list -- {-1: seed ids,
        i: hop i's ids} -- and is how a reranked traversal re-runs as an
        ORDINARY traversal with the survivors of each provider call
        already decided (see _rerank_pins()). It is private and stays
        private: `Hop(near=..., keep=N)` already narrows `match_i` to a
        subset through ranked_ids(), build_query() already derives that
        hop's reported edges from that subset, and pinning is the same
        operation with a different source for the list -- but a public
        field would invite a caller to hand-write ids that no step
        actually reached, which is a subgraph nothing in the graph
        supports.

        `pins=None` adds nothing anywhere: the SQL is byte-identical to
        what a traversal emitted before reranking existed."""
        from sqlalchemy.dialects.postgresql import array
        from sqlalchemy.sql.expression import any_ as sa_any_
        from sqlalchemy import not_ as sa_not_

        nt, et = self.nodes_tbl, self.edges_tbl
        edge_start_col = getattr(et.c, self.edge_start_col)
        edge_end_col = getattr(et.c, self.edge_end_col)
        edge_id_col = getattr(et.c, self.edge_id_col)
        node_id_col = getattr(nt.c, self.node_id_col)

        seed_condition = and_(self._scoped(nt), resolve(nt.c.properties, start.where),
                              *_pinned(node_id_col, pins, -1))
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
                pinned = _pinned(reached.c.node_id, pins, i)
                match_i = ranked_ids(
                    self, nt, reached.c.node_id, joined,
                    and_(*pinned) if pinned else None, nears, hop.keep,
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
                    .where(full_walk.c.depth >= hop.min_hops,
                           *_pinned(full_walk.c.to_id, pins, i))
                    .cte(f"match_{i}")
                )
            pairs.append((full_walk, match_i))
            prev_match = match_i

        return seed, pairs

    def build_query(self, start: Start, hops: list, pins: Optional[dict] = None):
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

        seed, pairs = self._walk_matches(start, hops, pins=pins)
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

        # The `is not None` half never decides anything, and the sentinel
        # above is therefore unobservable: `optional` is rejected anywhere
        # but the last hop, and `pairs` has one entry per hop, so
        # hops[-1].optional being true means the loop already assigned
        # this. Kept as a statement of that coupling, not as a guard.
        if hops[-1].optional and pre_optional_match is not None:
            node_selects.append(
                select(literal("node").label("kind"), cast(pre_optional_match.c.node_id, String).label("id"))
            )

        return node_selects[0].union(*node_selects[1:], *edge_selects)

    def build_aggregate_query(self, start: Start, hops: list, aggregates: dict,
                              pins: Optional[dict] = None):
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
        seed, pairs = self._walk_matches(start, hops, pins=pins)
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
        scan, and without the GIN indexes every property filter is.

        Each is a real sqlalchemy.Index attached to nodes_tbl/edges_tbl
        (see hopai.constraints), so a project's own Alembic
        --autogenerate sees them as declared schema rather than drift to
        propose dropping. suspended_declarations() is what keeps that
        attachment from backfiring here: Table.create() renders EVERY
        index/constraint currently attached, and if this process already
        called define_constraints()/enforce_schema()/migrate_vectors()
        on this table before create_schema() ever ran, those would
        otherwise get baked into the CREATE TABLE this issues instead of
        being applied on their own terms."""
        from .constraints import _attach_index, suspended_declarations

        with suspended_declarations(self.nodes_tbl, self.edges_tbl):
            self.nodes_tbl.create(self.engine, checkfirst=True)
            self.edges_tbl.create(self.engine, checkfirst=True)

        nt, et, g = self.nodes_tbl, self.edges_tbl, self.graph_col
        # graph_id LEADS both endpoint indexes. Every hop filters on it,
        # so a trailing position would make the index useless the moment
        # a second graph exists -- and the cost of the discriminator is
        # only acceptable because it is indexed away.
        indexes = [
            _attach_index(et, f"ix_{et.name}_graph_{self.edge_start_col}",
                         [et.c[g], et.c[self.edge_start_col]]),
            _attach_index(et, f"ix_{et.name}_graph_{self.edge_end_col}",
                         [et.c[g], et.c[self.edge_end_col]]),
            _attach_index(nt, f"ix_{nt.name}_graph", [nt.c[g]]),
            _attach_index(nt, f"ix_{nt.name}_properties", [nt.c.properties],
                         postgresql_using="gin"),
            _attach_index(et, f"ix_{et.name}_properties", [et.c.properties],
                         postgresql_using="gin"),
        ]
        with self.engine.begin() as connection:
            for idx in indexes:
                connection.execute(CreateIndex(idx, if_not_exists=True))

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
        are ignored, so this is the exact inverse of define_constraints().

        Also detaches the matching Index/CheckConstraint from the
        table's SQLAlchemy metadata (compile_constraint() attached it),
        so a dropped constraint does not linger as a phantom object a
        later constraint_ddl()/Alembic autogenerate would still see."""
        from .constraints import compile_constraint, detach_constraint

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
                    detach_constraint(target.table, kind, name)
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

        Refuses a property (either notation) named the same as a real
        column on this graph's table -- id/start_id/end_id/graph_id, a
        vec_* vector field, or an EXTRA COLUMN (models.py's "EXTENDING
        THE MODEL") -- naming exactly which, since that property could
        never be written or read the way its declaration implies. See
        schema.py's check_no_column_collisions().

        Returns the normalized GraphSchema."""
        from .schema import GraphSchema, build_schema, check_no_column_collisions
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
        else:
            schema = build_schema(nodes, edges)
        check_no_column_collisions(schema, self.nodes_tbl, self.edges_tbl)
        self._schema = schema
        return schema

    def infer_schema(self, sample_percent: Optional[float] = None,
                     vector_populated: bool = False) -> tuple:
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
        hopai/schema.py's INFERENCE section for semantics and cost.
        On tables too large for that, sample_percent=5 reads a
        TABLESAMPLE SYSTEM slice instead; counts become estimates and
        rare properties can be missed, and the report says so.

        The returned schema's `vectors` section mirrors THIS handle's
        OWN define_vectors() declaration -- dimensions, source, whether
        a field embeds -- for free; vector_populated=True additionally
        reads stale_vectors() for each declared field (one more query,
        on the same connection) and reports what fraction of the
        target's rows actually carry one, so plain infer_schema() never
        pays for a scan it did not ask for."""
        from .schema import infer_schema
        return infer_schema(self, sample_percent=sample_percent,
                            vector_populated=vector_populated)

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

    @property
    def schema_mermaid(self) -> str:
        """The schema as a Mermaid flowchart string -- paste it into a
        ```mermaid fence and any PR description or README renders the
        picture. No extra dependency, works offline."""
        return self._defined_schema("schema_mermaid").to_mermaid()

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

    def tool_schemas(self, rerank: bool = False) -> list:
        """The LLM tool definitions -- traverse, aggregate, ingest,
        mutate, and (when this graph has declared any) vector search --
        as deep copies, with THIS graph's declared schema summarized
        into each description so a function-calling model sees what
        exists instead of guessing labels and property names. With no
        schema defined, the static definitions come back unchanged (as
        copies): the schema stays optional.

        Every front end hopai has, including the one that deletes --
        handing a model three of four and leaving it to discover
        MUTATE_TOOL_SCHEMA would be worse than the decision being
        visible. Pick the subset you want to expose; this is the whole
        list.

        VECTOR_SEARCH_TOOL_SCHEMA joins the other four only WHEN
        self._vectors declares at least one field (issue #51) -- a
        graph with none has nothing to search by meaning, and offering
        the tool anyway would be a call that fails every time. Its
        static `field` parameter is a bare string; the copy appended
        here narrows it to an ENUM of this graph's own declared field
        names (vectors.field_names(), the same "what can be searched
        here" helper hopai.mcp's Served.vector_fields() calls), so a
        model is choosing among real fields rather than guessing one --
        the same treatment `search_field` already gets in hopai.mcp.

        Only descriptions differ from the module constants; the
        `parameters` sections are identical, because what the parsers
        accept has not changed -- this is presentation, not grammar."""
        import copy

        from .ingest import INGEST_TOOL_SCHEMA
        from .json_api import (
            AGGREGATE_TOOL_SCHEMA, TRAVERSE_TOOL_SCHEMA, VECTOR_SEARCH_TOOL_SCHEMA,
            without_rerank,
        )
        from .mutate import MUTATE_TOOL_SCHEMA
        # `rerank` comes OFF unless the caller says it has a reranker.
        # A Graph holds none -- the client is the operator's, and arrives
        # through hopai.mcp's serve(rerank=) or traverse_json(...,
        # rerank=RerankPolicy(...)) -- so against a bare Graph a spec
        # carrying `rerank` is refused BY NAME. Advertising a parameter
        # whose handler rejects every use of it is the defect CLAUDE.md
        # names, and it is the same judgement that keeps
        # VECTOR_SEARCH_TOOL_SCHEMA out of this list below for a graph
        # with no declared vectors: a call that cannot succeed should not
        # be offered.
        #
        # rerank=True is for a caller that HAS one and will fill in its
        # own published fields and ceiling -- hopai.mcp does exactly that,
        # and it builds on these schemas rather than beside them, so the
        # parameter has to survive long enough for it to configure.
        keep = (lambda tool: copy.deepcopy(tool)) if rerank else without_rerank
        tools = [keep(tool) for tool in (TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA)]
        tools += [copy.deepcopy(tool) for tool in (INGEST_TOOL_SCHEMA, MUTATE_TOOL_SCHEMA)]
        from .vectors import field_names
        names = field_names(self._vectors)
        if names:
            search = keep(VECTOR_SEARCH_TOOL_SCHEMA)
            search["parameters"]["$defs"]["near"]["properties"]["field"]["enum"] = names
            tools.append(search)
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
            compile_edge_constraints, compile_edge_uniques, compile_endpoint_ddl,
            compile_node_constraints, compile_node_uniques,
        )
        schema = self._defined_schema("schema_ddl()")
        node_target, edge_target = self._schema_targets()
        ddl = [statement for _, statement in (compile_node_constraints(schema, node_target)
                                              + compile_edge_constraints(schema, edge_target)
                                              + compile_node_uniques(schema, node_target)
                                              + compile_edge_uniques(schema, edge_target))]
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
        from .constraints import detach_constraint
        from .schema import (
            ENDPOINT_TRIGGER_EXISTS, SCHEMA_CHECKS, SCHEMA_UNIQUES, _graph_token,
            compile_edge_constraints, compile_edge_uniques, compile_endpoint_ddl,
            compile_node_constraints, compile_node_uniques, endpoint_names,
            schema_constraint_prefixes,
        )
        schema = self._defined_schema("enforce_schema()")
        node_target, edge_target = self._schema_targets()
        groups = [(node_target, compile_node_constraints(schema, node_target),
                   compile_node_uniques(schema, node_target)),
                  (edge_target, compile_edge_constraints(schema, edge_target),
                   compile_edge_uniques(schema, edge_target))]
        applied = []
        with self.engine.begin() as connection:
            for target, pairs, uniques in groups:
                existing = {row[0] for row in connection.execute(
                    SCHEMA_CHECKS, {"table": target.qualified}).all()}
                current = {name for name, _ in pairs}
                prefixes = schema_constraint_prefixes(target)
                for stale in sorted(n for n in existing
                                    if n.startswith(prefixes) and n not in current):
                    connection.execute(text(
                        f'ALTER TABLE {target.qualified} DROP CONSTRAINT "{stale}"'))
                    detach_constraint(target.table, "check", stale)
                for name, ddl in pairs:
                    if name not in existing:
                        connection.execute(text(ddl))
                    applied.append(name)

                # unique=True properties: partial unique indexes, same
                # reconcile discipline under their own uq_schema_ prefix
                index_prefix = f"uq_schema_{_graph_token(target.graph)}_"
                existing_uniques = {row[0] for row in connection.execute(
                    SCHEMA_UNIQUES, {"table": target.table.name}).all()}
                wanted_uniques = {name for name, _ in uniques}
                for stale in sorted(n for n in existing_uniques
                                    if n.startswith(index_prefix) and n not in wanted_uniques):
                    connection.execute(text(f'DROP INDEX IF EXISTS "{stale}"'))
                    detach_constraint(target.table, "index", stale)
                for name, ddl in uniques:
                    connection.execute(text(ddl))   # IF NOT EXISTS makes this idempotent
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

    def _schema_store(self) -> str:
        """The qualified name of the schema metadata table. It lives in
        the nodes table's Postgres schema so multi-tenant setups that
        namespace their graph tables namespace this one the same way."""
        if self.nodes_tbl.schema:
            return f'"{self.nodes_tbl.schema}"."hopai_schema"'
        return '"hopai_schema"'

    def save_schema(self) -> None:
        """Persist the declared schema so OTHER processes on this
        database can load_schema() instead of re-declaring -- the
        database becomes the single source of truth for the contract.

        Upserts this graph's row in hopai_schema(graph_id, document,
        saved_at), creating that table on first use. It is metadata,
        never on the query path: traversal and ingestion do not know it
        exists, and callers who never persist never create it. The
        document is schema_json verbatim -- human-readable in psql."""
        import json
        schema = self._defined_schema("save_schema()")
        store = self._schema_store()
        with self.engine.begin() as connection:
            # json, NOT jsonb: jsonb canonicalizes object key order, which
            # would hand load_schema() the properties alphabetized instead
            # of as declared -- breaking exact round-trip equality. This
            # table is never queried by containment, so jsonb buys nothing.
            connection.execute(text(
                f"CREATE TABLE IF NOT EXISTS {store} ("
                f"graph_id text PRIMARY KEY, "
                f"document json NOT NULL, "
                f"saved_at timestamptz NOT NULL DEFAULT now())"))
            connection.execute(text(
                f"INSERT INTO {store} (graph_id, document) "
                f"VALUES (:graph, CAST(:document AS json)) "
                f"ON CONFLICT (graph_id) DO UPDATE "
                f"SET document = EXCLUDED.document, saved_at = now()"),
                {"graph": self.graph, "document": json.dumps(schema.to_json())})

    def load_schema(self):
        """Read the schema save_schema() stored for this graph,
        ADOPT it on this handle (a saved schema was explicitly declared
        a contract -- unlike an inferred one, which is an observation)
        and return it. The stored document is data: it is rebuilt
        through the same validation define_schema() runs -- including
        the column-collision check -- so a corrupted row, or a schema
        that collides with THIS handle's table (a save/load pair can
        cross tables; its extra columns need not match), raises loudly
        instead of half-loading."""
        from .schema import check_no_column_collisions, schema_from_document
        row = None
        with self.engine.connect() as connection:
            exists = connection.execute(
                text("SELECT to_regclass(:name)"),
                {"name": self._schema_store()}).scalar()
            if exists is not None:
                row = connection.execute(
                    text(f"SELECT document FROM {self._schema_store()} "
                         f"WHERE graph_id = :graph"),
                    {"graph": self.graph}).first()
        if row is None:
            raise ValueError(
                f"no saved schema for graph {self.graph!r} -- save_schema() stores "
                f"the declared schema for other handles to load; on this handle, "
                f"declare it with define_schema(...)")
        schema = schema_from_document(row[0])
        check_no_column_collisions(schema, self.nodes_tbl, self.edges_tbl)
        self._schema = schema
        return schema

    # -- vectors --------------------------------------------------------

    def define_vectors(self, nodes: Optional[list] = None, edges: Optional[list] = None,
                       migrate: bool = False) -> Union[dict, list]:
        """Declare named vector fields: Vector(name, dimensions)
        entries per target. The migration they imply is a separate,
        explicit migrate_vectors(), because ALTER TABLE should be a
        conscious moment. Calling this again replaces the declaration.

        Not purely in-memory, unlike define_schema(): the vec_*
        columns are attached to the SHARED SQLAlchemy Table metadata,
        so create_schema() on any handle for these tables emits them
        from now on. Nothing else touches the database until
        migrate_vectors() runs -- which `migrate=True` does for you:

            graph.define_vectors(nodes=[Vector("summary", 1536)], migrate=True)

        is exactly

            graph.define_vectors(nodes=[Vector("summary", 1536)])
            graph.migrate_vectors()

        with `migrate_vectors()`'s own return value passed back, so the
        DDL it ran stays visible rather than happening invisibly. Reach
        for `migrate=True` in a start-up path that already carries
        schema-changing credentials, the same as create_schema().

        `migrate=False` (the default, unchanged) declares only and
        returns the normalized registry; hopai/vectors.py explains the
        storage model and every refusal. This is the shape to keep when
        migrations run separately from application code -- e.g. a
        process with read-only credentials that must not risk issuing
        DDL -- pairing this call with an explicit migrate_vectors()
        elsewhere."""
        from .vectors import attach_columns, build_registry
        self._vectors = build_registry(nodes, edges)
        attach_columns(self)
        if migrate:
            return self.migrate_vectors()
        return self._vectors

    @property
    def vectors(self) -> Optional[dict]:
        """The declared vector fields as {"nodes": {name: Vector},
        "edges": {...}}, or None when define_vectors() has not been
        called on this handle -- that is the existence check.

        NEVER CONNECTS, on any handle, including one from in_graph():
        that guarantee -- a Graph you only build queries with never
        touches the network -- outranks this property being perfectly
        live. A handle from in_graph() is marked to recover its vectors
        lazily from the database (see in_graph()'s docstring), but that
        recovery happens on first CONNECTION -- a search, or an
        explicit load_vectors() call -- never on reading this property.
        Until then, this answers None for such a handle even when the
        database can prove the graph has fields; call load_vectors()
        yourself first if you need the true answer without searching.

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

    def load_vectors(self, connection=None) -> dict:
        """Recover the vector declaration FROM THE DATABASE instead of
        from define_vectors() -- for a fresh process, a second Graph
        handle, or any other caller that forgot to redeclare a graph
        another handle already migrated:

            g2 = Graph(engine)     # a second handle; define_vectors() never ran here
            g2.load_vectors()      # -> {"nodes": {"summary": Vector("summary", 1536)}, ...}
            g2.vector_search(Near("summary", q))    # now works

        Every vec_* column already migrated FOR THIS GRAPH is found by
        its per-graph dimension CHECK (the same constraint
        migrate_vectors() creates, looked up by the exact name that
        handle would have used) -- a column present with no such
        constraint means some OTHER graph migrated it, and is skipped
        rather than guessed at, since the column itself is shared
        storage while the dimension is scoped per graph.

        RECOVERING THE SHAPE IS NOT RECOVERING THE POLICY: `embed=`
        (an application's own embedding client) and a non-default
        `source=` are never stored in SQL, so every recovered field
        comes back with embed=None and source=<field name>, Vector's
        defaults -- fine for vector_search() with a vector= or for
        reading with get_vectors(), but redeclare with
        define_vectors(..., embed=<your client>) before using
        Near(text=...) or set_vectors() with a string against it.

        Populates and returns the registry exactly like
        define_vectors() does (attach_columns() included), so the
        result is usable immediately -- this is also what in_graph()
        calls, lazily, the first time a returned handle needs vectors
        and was never given an explicit declaration (see its
        docstring)."""
        from .vectors import load_vectors
        return load_vectors(self, connection=connection)

    def drop_vectors(self, node_fields: Optional[list] = None,
                     edge_fields: Optional[list] = None) -> list:
        """Drop the named fields FOR THIS GRAPH: their dimension
        constraints go, their values in this graph's rows are set to
        NULL. The shared columns stay -- other graphs may use them."""
        from .vectors import drop_vectors
        return drop_vectors(self, node_fields, edge_fields)

    def set_vectors(self, nodes: Optional[list] = None, edges: Optional[list] = None) -> int:
        """Store vectors on existing rows, one transaction for the
        whole call. Each row is {"id": ..., <field>: <vector, text or
        None>}:

            graph.set_vectors(nodes=[{"id": 1, "summary": embedding}])
            graph.set_vectors(nodes=[{"id": 1, "summary": "a paper on Raft"}])

        A string is embedded with the field's declared embed= -- every
        string in the call batched into one provider request per field,
        before the transaction opens. A field with no embedder refuses
        text by name rather than guessing.

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

    def stale_vectors(self, node_fields=None, edge_fields=None, limit=None,
                      after=None) -> dict:
        """Which rows need (re-)embedding, per field: those with no
        vector, and those whose stored vector no longer matches the
        declared dimensions.

            for node_id in graph.stale_vectors()["nodes"]["summary"]["missing"]:
                graph.set_vectors(nodes=[{"id": node_id, "summary": embed(...)}])

        The second category is the window a dimension change opens --
        migrate_vectors() refuses to reinterpret stored vectors and
        set_vectors() refuses to write the wrong size, so this is what
        closes it without hand-writing the catalog query.

        embed_stale() runs that whole loop for fields that declare an
        embed=; this is the report for the ones you fill in yourself.

        To walk a large field, page with `limit` AND `after=<the
        largest id you saw>`. `limit` alone repeats itself: a row with
        nothing to embed stays stale forever and holds the window."""
        from .vectors import stale_vectors
        return stale_vectors(self, node_fields, edge_fields, limit, after)

    def embed_stale(self, node_fields=None, edge_fields=None, limit=None,
                    batch: int = 1000) -> dict:
        """Fill in every stale vector from its source property, for the
        fields that declare an embed=. The backfill loop stale_vectors()
        leaves you to write:

            graph.define_vectors(nodes=[Vector("summary", 1536,
                                               source="abstract",
                                               embed=openai.OpenAI())])
            graph.embed_stale()
            # -> {"nodes": {"summary": {"embedded": ["1"], "skipped": []}},
            #     "edges": {}}

        `skipped` is the rows whose source property is absent or blank
        -- reported rather than raised, since there is nothing to embed,
        but never silent.

        Walks each field in pages of `batch`, one embed call and one
        transaction each, so any size of backfill costs bounded memory
        and a re-run resumes rather than restarting. `limit` caps the
        rows one call takes on per field; the default is all of them."""
        from .vectors import embed_stale
        return embed_stale(self, node_fields, edge_fields, limit, batch)

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
                           boost=None, rerank=None) -> list:
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
        rather than silently ranking them differently.

        `rerank` is PER CALL, where `boost` already sits -- not per
        query: `candidates` decides how many rows the ONE statement
        fetches and this call takes ONE `k`, so per-query Reranks could
        disagree in a way one statement cannot serve. N queries produce
        N reranker calls (a score is the (query, document) relationship,
        so one call cannot serve two queries); AsyncGraph issues them
        concurrently."""
        from .vectors import search_many
        return search_many(self, queries, target=target, k=k, where=where, boost=boost,
                           rerank=rerank)

    def vector_search(self, *near, target: str = "nodes", k: int = 10, where=None,
                      boost=None, rerank=None) -> list:
        """Exact cosine similarity search over this graph's nodes or
        edges -- one statement, no extension, no approximation.

            graph.vector_search(Near("summary", embedding), k=10,
                                where={"type": "person"})
            # -> [{"id": "1", "similarity": 0.93, "properties": {...},
            #      "similarities": {"summary": 0.93}, "boosts": {}}, ...]

        Several Near specs combine into one weighted score
        (multivector search); `where` is the same filter language as
        traversal and is applied BEFORE ranking, so a selective filter
        makes the search cheaper, never slower. Edge results also
        carry start_id/end_id. Rows are ordered most-similar first;
        ids are strings, like every other result. `boost` adds
        property terms to the score for hybrid retrieval; a boosted
        `similarity` is the combined score and can exceed 1, and
        reordering with `k` changes which rows come back. Every hit
        also reports `similarities` (per Near field, keyed by name --
        `None` where that field's vector is missing, even if
        missing="zero" scored it 0 in the combined total) and `boosts`
        (per Boost, keyed by property), so a caller tuning weights can
        see what actually drove a result rather than only the sum. The
        cost model and every design refusal live in hopai/vectors.py.

        `rerank=Rerank(client, document_from='...', candidates=50)` adds
        the third retrieval stage: `candidates` rows are fetched instead
        of `k`, each becomes a document by evaluating the jq filter,
        ONE provider call scores them -- after the SQL round trip has
        closed, never inside it -- and the list comes back ordered by
        the new score and truncated to `k`. Purely additive: each hit
        gains `rerank_score` and `similarity` keeps its original value,
        so a caller can see what the reranker changed. It needs the
        query as TEXT (`Near("summary", text="...")`), because a
        reranker scores a query against a document by reading both."""
        from .vectors import search
        return search(self, list(near), target=target, k=k, where=where, boost=boost,
                      rerank=rerank)

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
        from .cypher import cypher_to_operations, resolve_strict
        return self._ingestor.execute_operations(
            cypher_to_operations(query, **resolve_strict(self, dict(options))))

    def cypher_operations(self, query: str, **options) -> list:
        """The ingestion plan a Cypher write compiles to, without running
        it -- for review, logging, or showing an agent what it is about
        to change."""
        from .cypher import cypher_to_operations, resolve_strict
        return cypher_to_operations(query, **resolve_strict(self, dict(options)))

    def cypher(self, query: str, **options):
        """Run any supported Cypher: reading, writing, deleting,
        updating, or aggregating.

        Returns a Subgraph for a query that matches, an IngestResult for
        one that creates or merges, a MutationResult for one that
        deletes or updates, and a plain dict of numbers for one whose
        RETURN aggregates. Which one it is is visible in the query, and
        this is the entry point anyone arriving from a Neo4j driver
        reaches for first; traverse_cypher(), write_cypher(),
        mutate_cypher() and aggregate_cypher() are the same thing when
        you would rather be explicit."""
        from .cypher import aggregate_cypher, classify_cypher, traverse_cypher
        kind = classify_cypher(query)
        if kind == "mutate":
            return self.mutate_cypher(query, **options)
        if kind == "write":
            return self.write_cypher(query, **options)
        if kind == "aggregate":
            return aggregate_cypher(self, query, **options)
        return traverse_cypher(self, query, **options)

    def add_networkx(self, nx_graph):
        """Load a networkx graph. Node keys become ids, and node/edge
        attribute dicts become properties -- the inverse of
        Subgraph.to_networkx()."""
        return self._ingestor.add_networkx(nx_graph)

    # -- deleting and updating ------------------------------------------

    @property
    def _mutator(self):
        from .mutate import Mutator
        if not hasattr(self, "_mutator_cache"):
            self._mutator_cache = Mutator(self)
        return self._mutator_cache

    def delete_nodes(self, where=None, detach: bool = False, all: bool = False):
        """Delete every node matching `where` -- the same filter language
        a traversal uses. Returns a MutationResult.

        A node that still has edges cannot be deleted: pass detach=True
        to delete its edges with it (Cypher's DETACH DELETE). A call with
        no filter raises rather than emptying the graph -- say it on
        purpose with all=True, or call clear()."""
        return self._mutator.delete_nodes(where, detach=detach, all=all)

    def delete_edges(self, where=None, start=None, end=None, all: bool = False):
        """Delete every edge matching `where`, optionally restricted to
        edges whose endpoints match `start`/`end`:

            graph.delete_edges(where={"kind": "knows"}, start={"name": "Alice"})

        `start`/`end` are filters, not references: any number of nodes
        may match one, and every edge touching any of them goes. That is
        the opposite of add_edges(), where `start`/`end` must identify
        exactly one node and ambiguity raises.

        Returns a MutationResult. Deleting an edge never affects the
        nodes it connected."""
        return self._mutator.delete_edges(where, start=start, end=end, all=all)

    def update_nodes(self, where=None, set=None, remove=None, replace: bool = False,
                     all: bool = False):
        """Update every node matching `where`. `set` is merged over the
        existing properties, leaving anything it does not mention alone;
        `remove` drops keys; `replace=True` makes `set` the whole
        property bag. Returns a MutationResult.

            graph.update_nodes(where={"type": "person"}, set={"active": False})
        """
        return self._mutator.update_nodes(where, set=set, remove=remove,
                                          replace=replace, all=all)

    def update_edges(self, where=None, start=None, end=None, set=None, remove=None,
                     replace: bool = False, all: bool = False):
        """Update every edge matching `where`, with the same set/remove/
        replace semantics as update_nodes() and the same endpoint
        filters as delete_edges() -- including that they are filters
        rather than references. Returns a MutationResult."""
        return self._mutator.update_edges(where, start=start, end=end, set=set,
                                          remove=remove, replace=replace, all=all)

    def clear(self):
        """Delete every node and edge in THIS graph, in one transaction.
        Other graphs in the same tables keep their rows -- this is a
        scoped DELETE, not a TRUNCATE. Returns a MutationResult."""
        return self._mutator.clear()

    def mutate(self, document: dict):
        """Run a whole `{"operations": [...]}` mutation document, in
        order, in one transaction -- the delete/update counterpart of
        ingest(), and the call an agent makes. See MUTATE_TOOL_SCHEMA.

            graph.mutate({"operations": [
                {"op": "update_nodes", "where": {"type": "draft"},
                 "set": {"status": "archived"}},
                {"op": "delete_edges", "where": {"kind": "draft_of"}},
            ]})
        """
        from .mutate import spec_to_mutations
        return self._mutator.execute_operations(spec_to_mutations(document))

    def mutate_cypher(self, query: str, **options):
        """Run a Cypher DELETE / DETACH DELETE / SET / REMOVE. Returns a
        MutationResult.

            graph.mutate_cypher("MATCH (a:person {email: 'a@x.com'}) DETACH DELETE a")
            graph.mutate_cypher("MATCH (a:person) WHERE a.age > 65 SET a.retired = true")

        The whole query is one transaction and compiles down to the same
        delete_nodes/update_nodes/... the Python API calls --
        `hopai.cypher_to_mutations(query)` shows the plan without running
        it. Accepts the same node_label_key / edge_type_key /
        strict_schema options as the read and write sides -- and a
        hallucinated label is worth refusing here most of all, since a
        delete that matched nothing reports the same success as one that
        had nothing to match."""
        from .cypher import cypher_to_mutations, resolve_strict
        return self._mutator.execute_operations(
            cypher_to_mutations(query, **resolve_strict(self, dict(options))))

    # -- execution ------------------------------------------------------

    def _aggregate_with_session(self, session, start: Start, hops: list, aggregates: dict,
                                pins: Optional[dict] = None) -> dict:
        """The aggregate work itself, given an OPEN session. aggregate()
        opens one and calls this directly; AsyncGraph.aggregate()
        (hopai/asyncio.py) reaches the SAME function through
        AsyncSession.run_sync() -- one implementation, two callers, so
        sync and async can never disagree about what an aggregate
        answers."""
        query = self.build_aggregate_query(start, hops, aggregates, pins=pins)
        row = session.execute(query).one()
        # strict= is unobservable here and that is on purpose:
        # build_aggregate_query() emits exactly one labeled column per
        # entry in `aggregates`, so the two lengths are equal by
        # construction. Kept as a claim about that, not as a guard.
        return {name: _plain(value) for name, value in zip(aggregates, row, strict=True)}

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
        when nothing matched. One statement, one round trip.

        With a `rerank=` anywhere in the chain the aggregates run over
        the reranked SURVIVORS -- still "the nodes the last hop
        matched", because pinning narrows exactly what `keep=` narrows,
        and it costs the probe round trips traverse() documents."""
        hops = list(hops)
        plan = rerank_plan(start, hops)
        pins = self._rerank_pins(start, hops, plan) if plan else None
        with Session(self.engine) as session:
            return self._aggregate_with_session(session, start, hops, aggregates, pins=pins)

    # -- step-wise reranking --------------------------------------------
    #
    # A traversal compiles to ONE recursive CTE, and a reranker is a
    # network call in the middle of the walk, so it cannot stay one
    # statement. The shape here is PROBE, RERANK, THEN RE-RUN THE
    # ORDINARY TRAVERSAL with each step's survivors pinned -- not
    # stitching partial results together.
    #
    # That choice is the whole safety argument. Because the final
    # execution is an ordinary traversal, fan-in, multi-hop edge
    # reconstruction and dead-end pruning are preserved for free: the
    # edges reported for a hop are still derived from `full_walk` against
    # `match_i`, and `match_i` is still what feeds the next hop. Pinning
    # only changes where that hop's id list came from -- `Hop(near=,
    # keep=N)` already narrows it to a subset through ranked_ids(), and
    # this narrows it to a subset the provider chose instead. Stitching
    # would have had to re-derive the edges by hand, which is exactly
    # what "reported nodes derive from the edges found" exists to stop.
    #
    # COST, stated honestly because rule 6 says a measurement rather than
    # "probably fine": 1 + (2 x reranked steps) round trips -- one probe
    # and one hydration per reranked step, then the traversal itself --
    # plus one provider call per reranked step. Those calls are SERIAL by
    # nature: hop N+1's candidates are whatever hop N's rerank left, so
    # unlike vector_search_many()'s N calls they cannot be issued
    # together. tests/test_vectors.py pins the count so a change that
    # doubles it is visible.

    def _rerank_probe(self, session, start: Start, hops: list, index: int,
                      pins: dict) -> list:
        """The candidates one reranked step offers, as
        (id, {"id", "properties"}) pairs -- plus "paths" at a hop, when
        the filter reads them.

        The step is run TRUNCATED: everything up to and including it,
        with its own `keep` widened to `rerank.candidates` (the input
        bound) and every earlier step's survivors already pinned. So the
        candidate set is exactly what that step would have reached, cut
        off at the number the caller agreed to pay for."""
        spec = start if index < 0 else hops[index]
        rerank = spec.rerank
        nt = self.nodes_tbl
        node_id_col = getattr(nt.c, self.node_id_col)

        # `rerank=None` on the copy, and not only for tidiness: the spec
        # re-validates on construction, and by the time the ASYNC path
        # gets here the Near carries floats with its text= already
        # resolved away -- which _validate_rerank() would (correctly, in
        # any other context) refuse as a raw-vector query. The probe has
        # no reranker to run anyway; it is what produces its input.
        if index < 0:
            probe_start = _respec(start, keep=rerank.candidates, rerank=None)
            probe_hops = []
        else:
            probe_start = start
            probe_hops = list(hops[:index]) + [
                _respec(hops[index], keep=rerank.candidates, rerank=None)]
        seed, pairs = self._walk_matches(probe_start, probe_hops, pins=pins)

        wants_paths = index >= 0 and _reads_paths(rerank.document_from, unknown=True)
        if index < 0:
            walks = [(row.node_id, None)
                     for row in session.execute(select(seed.c.node_id)).all()]
        else:
            full_walk, match_i = pairs[index]
            if wants_paths:
                # The same (walk, match) pair build_query() reads for
                # `hop_edges_i`, asked for local_path instead of
                # local_edges: one row per (candidate, route), which is
                # what makes a candidate here "a node PLUS how it was
                # reached" rather than a row.
                query = select(full_walk.c.to_id, full_walk.c.local_path).where(
                    and_(full_walk.c.depth >= probe_hops[index].min_hops,
                         full_walk.c.to_id.in_(select(match_i.c.node_id))))
                walks = [(row.to_id, list(row.local_path))
                         for row in session.execute(query).all()]
            else:
                walks = [(row.node_id, None)
                         for row in session.execute(select(match_i.c.node_id)).all()]

        routes: dict = {}
        for node_id, local_path in walks:
            paths = routes.setdefault(node_id, set())
            if local_path:
                # The walk's local_path is [anchor, ..., candidate]. The
                # candidate itself is not provenance -- it is already the
                # document's own `.properties` -- so a path is who
                # REACHED it, which is what makes `.paths[][-1]` the
                # immediate parent ("cited by") rather than the node
                # restating itself.
                paths.add(tuple(local_path[:-1]))

        wanted = set(routes)
        wanted.update(node_id for paths in routes.values() for path in paths for node_id in path)
        properties = self._rerank_properties(session, node_id_col, wanted)

        def hydrate(node_id):
            return {"id": str(node_id), "properties": properties.get(str(node_id), {})}

        candidates = []
        for node_id in sorted(routes, key=str):
            candidate = hydrate(node_id)
            if wants_paths:
                # Canonically ordered, so the same graph always produces
                # the same document (and, once max_paths trims it, the
                # same TRUNCATION) -- a reranker that scored the same
                # node differently between two identical runs would be
                # untestable and unreproducible.
                candidate["paths"] = [
                    [hydrate(step) for step in path]
                    for path in sorted(routes[node_id],
                                       key=lambda path: tuple(str(step) for step in path))
                ]
            candidates.append((node_id, candidate))
        return candidates

    def _rerank_properties(self, session, node_id_col, node_ids: set) -> dict:
        """Properties for every node a step's documents mention -- the
        candidates and, when paths are read, the nodes on them -- in ONE
        statement, keyed by the string id every result already uses."""
        if not node_ids:
            return {}
        query = select(cast(node_id_col, String).label("id"), self.nodes_tbl.c.properties).where(
            and_(self._scoped(self.nodes_tbl), node_id_col.in_(sorted(node_ids, key=str))))
        return {row.id: row.properties for row in session.execute(query).all()}

    def _rerank_pins(self, start: Start, hops: list, plan: dict) -> dict:
        """Every reranked step's survivors, walked front to back.

        Each step opens its own session and CLOSES it before the
        provider call: an HTTP round trip inside an open transaction
        holds a read snapshot for its whole duration, which is the same
        rule set_vectors() keeps for row locks.

        The documents are built with build_documents()' own defaults --
        the SAFE SUBSET, no `trusted=True`. The dangerous thing is the
        one you have to ask for, the polarity `allow_vectors=`, `all=`
        and `detach=` already keep: a filter can reach a Rerank from a
        tool call or over the wire as easily as from a caller's own
        Python, and the engine cannot tell which. Widening it is the
        job of whoever knows the filter's provenance, not of the code
        that runs every traversal."""
        pins: dict = {}
        for index, spec in _rerank_steps(start, hops):
            with Session(self.engine) as session:
                candidates = self._rerank_probe(session, start, hops, index, pins)
            units = _rerank_documents(candidates, spec.rerank)
            if not units:
                pins[index] = []
                continue
            documents = spec.rerank.build_documents([candidate for _, candidate in units])
            scores = spec.rerank.score(plan[index], documents)
            pins[index] = _rerank_survivors(units, scores, spec.keep)
        return pins

    def _traverse_with_session(self, session, start: Start, hops: list,
                               pins: Optional[dict] = None) -> Subgraph:
        """The traverse work itself, given an OPEN session -- see
        _aggregate_with_session() just above for why this split exists."""
        query = self.build_query(start, hops, pins=pins)
        t0 = time.perf_counter()
        rows = session.execute(query).all()
        node_ids = [r.id for r in rows if r.kind == "node"]
        edge_ids = [r.id for r in rows if r.kind == "edge"]

        nt, et = self.nodes_tbl, self.edges_tbl
        node_id_col = getattr(nt.c, self.node_id_col)
        edge_id_col = getattr(et.c, self.edge_id_col)

        # Extra columns (see node_extra_cols/edge_extra_cols) tag along
        # here, not in build_query() -- that statement only ever returns
        # tagged (kind, id) rows, and hydration is the one place a real
        # row is actually read. An empty tuple (the common case: no
        # custom table, or one with none) expands to nothing, so the
        # statement stays byte-identical to before this existed.
        node_extra = [getattr(nt.c, c) for c in self.node_extra_cols]
        edge_extra = [getattr(et.c, c) for c in self.edge_extra_cols]

        nodes = []
        if node_ids:
            q = select(cast(node_id_col, String).label("id"), nt.c.properties, *node_extra).where(
                and_(self._scoped(nt), cast(node_id_col, String).in_(node_ids))
            )
            nodes = [
                {"id": r.id, "properties": r.properties,
                 **{c: getattr(r, c) for c in self.node_extra_cols}}
                for r in session.execute(q).all()
            ]

        edges = []
        if edge_ids:
            q = select(
                cast(edge_id_col, String).label("id"),
                cast(getattr(et.c, self.edge_start_col), String).label("start_id"),
                cast(getattr(et.c, self.edge_end_col), String).label("end_id"),
                et.c.properties,
                *edge_extra,
            ).where(and_(self._scoped(et), cast(edge_id_col, String).in_(edge_ids)))
            edges = [
                {"id": r.id, "start_id": r.start_id, "end_id": r.end_id,
                 "properties": r.properties,
                 **{c: getattr(r, c) for c in self.edge_extra_cols}}
                for r in session.execute(q).all()
            ]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return Subgraph(nodes=nodes, edges=edges, elapsed_ms=elapsed_ms)

    def traverse(self, start: Start, *hops: Hop) -> Subgraph:
        """Run one traversal. `start` picks the seed set; each `Hop`
        after it is one step of the walk. Returns a Subgraph with every
        node and edge on at least one complete matching chain.

        ONE STATEMENT, unless a step carries `rerank=`. A reranker is a
        provider call in the middle of the walk, so a reranked traversal
        runs as: per reranked step, a probe for its candidates and a
        hydration for their properties, then the provider call with
        nothing open, and finally THIS ordinary traversal with each
        step's survivors pinned.

        COST, measured rather than assumed (see
        test_a_reranked_traversal_costs_one_plus_two_per_reranked_step):
        1 + (2 x reranked steps) round trips on top of what this
        traversal already costs -- the traversal itself is unchanged,
        its own hydration SELECTs included -- and one provider call per
        reranked step. Those calls are SERIAL by nature: hop N+1's
        candidates are whatever hop N's rerank left behind, so unlike
        vector_search_many()'s they cannot be issued together. Pruning
        therefore COMPOUNDS and is unrecoverable: prune wrong at hop 1
        and hop 2 never sees the right nodes, which is the argument for
        a generous `candidates` at early steps.

        The result shape does not change. A traversal returns a
        SUBGRAPH, not a ranking -- similarity scores have never survived
        into it and rerank scores get no exception. Use vector_search()
        when the scores themselves are the answer."""
        hops = list(hops)
        # Every refusal first, before a statement runs: a chain that will
        # be rejected must not spend a probe (or a provider call) on the
        # way to being rejected.
        plan = rerank_plan(start, hops)
        pins = self._rerank_pins(start, hops, plan) if plan else None
        with Session(self.engine) as session:
            return self._traverse_with_session(session, start, hops, pins=pins)
