"""
hopai.hop

The traversal spec. Deliberately split into two types instead of one:

  Start(where=...)                     -- the seed set. No direction, no
                                           hop count, no via -- those
                                           concepts don't apply to a
                                           starting point, so the type
                                           doesn't offer them.

  Hop(where=..., via=..., hops=...)    -- one step of the walk.

A single `Hop`-for-everything design (what earlier versions of this
library used) let you accidentally write nonsense like "the seed node
must be reached via a 'friend' edge" -- meaningless, but the type system
had no opinion. Splitting the types makes that combination impossible to
write, not just documented as wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

HopCount = Union[int, tuple]  # an int means "exactly N hops"; a tuple means (min, max)


def _normalize_hops(hops: HopCount) -> tuple:
    # The three messages below are asserted verbatim by
    # tests/test_query_shape.py, not just their exception types. That is
    # not fussiness: mutation testing replaced each of them with None and
    # the whole suite still passed, which meant nothing was stopping a
    # refactor from leaving a caller with a bare ValueError and no idea
    # which rule they broke. If you reword one, update its test.
    if isinstance(hops, int):
        if hops < 1:
            raise ValueError(f"hops must be >= 1, got {hops}")
        return (hops, hops)
    if isinstance(hops, tuple) and len(hops) == 2:
        lo, hi = hops
        if lo < 1 or hi < lo:
            raise ValueError(f"hops range must satisfy 1 <= min <= max, got {hops}")
        return (lo, hi)
    raise TypeError(f"hops must be an int or a (min, max) tuple -- got {hops!r}")


def _validate_near_k(owner: str, near, k, near_name: str = "near",
                     k_name: str = "keep") -> None:
    # Structural checks only: whether each Near names a real field of
    # the right dimensionality needs the Graph's registry, so that
    # validation lives in core/vectors at build time -- the same split
    # as optional=, which build_query validates.
    if k is not None:
        if near is None:
            raise ValueError(
                f"{owner}: {k_name}={k!r} without {near_name}= orders nothing -- {k_name} is "
                f"how many of the most similar to keep, so it needs a Near to rank by"
            )
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"{owner}: {k_name} must be a positive integer, got {k!r}")
    if isinstance(near, (list, tuple)) and not near:
        raise ValueError(f"{owner}: {near_name}=[] is empty -- pass a Near(...) or a list of "
                         f"them, or drop the argument")


def _validate_boost(owner: str, near, boost) -> None:
    """A boost adjusts the NODE ranking near= creates, so there has to
    be one. Saying "add near=" to someone who passed via_near= would be
    a lie that costs them a working query: via_near ranks EDGES, and
    following the advice would silently rank nodes instead."""
    if boost is not None and near is None:
        raise ValueError(
            f"{owner}: boost= adjusts the node ranking that near= creates -- an edge beam "
            f"(via_near=) has no boost term, and without near= nothing is ranked at all. "
            f"Add near=, or filter on the property with where= instead"
        )


def _validate_rerank(owner: str, near, keep, rerank, k_name: str = "keep",
                     via_near=None) -> None:
    """The ways a rerank= cannot mean anything, refused where the caller
    wrote it rather than three layers down at execution.

    Structural only, like _validate_near_k above: whether the field
    exists and whether the jq filter parses are checked where the Graph
    and the subset are in scope.

    `k_name` is not cosmetic -- it says WHICH SURFACE is asking. "keep"
    is a traversal step, which discards the reranker's order and can
    only be changed by truncation; "k" is vector_search(), which reports
    that order. See the comment above `discards_order` below.
    """
    if rerank is None:
        return

    # Lazy, like every other cross-module import in this file's callers:
    # hop.py is imported by core and vectors, and rerankers.py reaches
    # embeddings.py, so naming it at module scope would put a provider
    # seam in the import path of every traversal that never reranks.
    from .rerankers import Rerank

    # THE MISTAKE THE DOCS INVITE. "A plain callable is a first-class
    # client" is true of the thing you hand to Rerank(...), not of the
    # thing you hand to rerank=, and both are spelled the same at the
    # call site. Unchecked, a bare callable, a jq string or a list of
    # Reranks all construct and then die at execution with
    # `AttributeError: 'function' object has no attribute 'candidates'`
    # -- three layers from the line that wrote it, naming an internal
    # attribute rather than the fix. Same judgement and same shape as
    # validate_nears()' "near= takes Near(field, vector) specs".
    if not isinstance(rerank, Rerank):
        raise TypeError(
            f"{owner}: rerank= takes a Rerank(client, document_from='...') spec, got "
            f"{type(rerank).__name__} -- a bare client (or a bare filter string) is half "
            f"of one: hopai also needs the jq filter that turns each candidate into the "
            f"document the client reads, and neither half can be guessed from the other. "
            f"Write rerank=Rerank(client, document_from='.properties.title')"
        )

    # A reranker REORDERS a ranked candidate list; it cannot produce one.
    # Same judgement as boost= above and as a Near inside where=: the
    # thing that ranks and the thing that filters are different jobs.
    if near is None:
        # Told apart from the bare case for the reason _validate_boost's
        # docstring gives about its own message: "Add near=" to someone
        # who passed via_near= is a lie that costs them a working query.
        # via_near ranks EDGES, and following that advice would silently
        # rerank nodes -- a different answer, quietly.
        if via_near is not None:
            raise ValueError(
                f"{owner}: rerank= reranks the NODES this hop reaches, and via_near= ranks "
                f"the EDGES it walks -- an edge beam has no rerank stage, so there is "
                f"nothing here for a reranker to reorder. Add near= to rank the reached "
                f"nodes and rerank those, or drop rerank= and let via_near= pick the edges "
                f"on similarity alone"
            )
        raise ValueError(
            f"{owner}: rerank= reorders the candidates near= ranks -- on its own it has "
            f"nothing to reorder, because a reranker scores a list it is given rather than "
            f"choosing one. Add near=, or drop rerank="
        )

    # A reranker reads the query. A raw vector is not something a
    # cross-encoder can read, and no implementation makes it one -- so
    # this is what reranking IS, not a gap to close later. Refused rather
    # than papered over with a second way to supply the query, because
    # two sources for one query can disagree silently.
    for one in (near if isinstance(near, (list, tuple)) else [near]):
        # Something that is not a Near at all -- `near="database"` --
        # falls through to the check that owns it (validate_nears(), at
        # build time: "near= takes Near(field, vector) specs, got
        # 'database'"). Without this guard a string has no `.text`, so
        # the raw-vector refusal below fired instead and invented a spec
        # nobody wrote -- `Near('?', ...) was given a raw vector`, which
        # is false (a string was given) and names a rewrite,
        # `Near('?', text="...")`, that no caller can apply. A rerank=
        # must not change which diagnosis an unrelated mistake gets.
        if getattr(one, "field", None) is None:
            continue
        if getattr(one, "text", None) is None:
            field = getattr(one, "field", "?")
            raise ValueError(
                f"{owner}: rerank= needs the query as TEXT, but Near({field!r}, ...) was "
                f"given a raw vector -- a reranker scores a query against a document by "
                f"reading both, and there is nothing to read in a list of floats. Write "
                f"Near({field!r}, text=\"...\") and the field's own embed= turns it into "
                f"the vector, so the ranking and the reranking see the same query"
            )

    # WHICH SURFACE THIS IS decides the two refusals below, and the NAME
    # of the truncation bound is what tells them apart -- a traversal
    # step truncates by `keep` and throws the order away (a traversal
    # returns a subgraph, not a ranking), a vector_search truncates by
    # `k` and REPORTS the order, rerank_score and all. There is no second
    # flag because there is nothing for one to disagree with:
    # vectors.rerank_query_text() already forwards this exact name from
    # whichever surface called it, and it is the name every message here
    # quotes back.
    discards_order = k_name == "keep"
    candidates = rerank.candidates

    # A TRAVERSAL STEP RERANKS IN ORDER TO TRUNCATE, OR NOT AT ALL.
    # Since the order does not survive, the only mark a reranker can
    # leave on a subgraph is which nodes it drops -- and dropping is
    # what `keep` means. With no `keep` the step's bound became
    # `rerank.candidates` (core.py's _rerank_probe widens it, and
    # nothing truncates afterwards), so a reranker that changed NOTHING
    # still deleted every node past the candidate pool: 6 reached nodes
    # became 2 for `candidates=1`. `Hop(near=, keep=None)` is an
    # ordinary query -- min_similarity is the other bound -- so this is
    # reachable from Python and over MCP alike, and refusing is the only
    # answer that cannot be a silently different subgraph.
    if discards_order and keep is None:
        raise ValueError(
            f"{owner}: rerank= needs {k_name}= -- a traversal returns a SUBGRAPH, not a "
            f"ranking, so the order a reranker produces is discarded and truncating is "
            f"the only way it can change the result. {k_name}= IS that truncation, and "
            f"with none, rerank=Rerank(candidates={candidates}) quietly becomes the bound "
            f"instead: the step keeps {candidates} node(s) rather than every node it "
            f"reached, which is a different subgraph from the same query without rerank=. "
            f"Add {k_name}=N for how many of the reranked candidates continue -- N below "
            f"candidates={candidates}, raising candidates= as well if that leaves no room "
            f"-- or drop rerank="
        )

    # Reranking a pool no larger than what survives it cannot reorder
    # anything that matters. Clamping silently would hide that the two
    # numbers in the caller's own query disagree.
    if keep is not None and candidates < keep:
        raise ValueError(
            f"{owner}: rerank=Rerank(candidates={candidates}) reranks fewer candidates than "
            f"{k_name}={keep} keeps, so the reranking cannot change which rows survive. "
            f"Raise candidates above {k_name}, or lower {k_name}"
        )
    # AT EQUALITY A SEARCH AND A STEP PART COMPANY. vector_search reports
    # the new order and a `rerank_score` per hit, so reranking exactly
    # `k` candidates reorders `k` rows and the caller can see it happen.
    # A traversal step reports neither: the survivors are the same
    # top-`keep` nodes whichever score sorted them, so the call is a
    # guaranteed no-op that the provider still bills per document
    # (measured: identical subgraph, 3 documents spent). Refused for the
    # reason `candidates < keep` is -- the two numbers in the caller's
    # own query disagree about what the reranking is for.
    if discards_order and keep is not None and candidates == keep:
        raise ValueError(
            f"{owner}: rerank=Rerank(candidates={candidates}) reranks exactly as many "
            f"candidates as {k_name}={keep} keeps, so the reranking cannot change which "
            f"rows survive -- a traversal discards the order, so the same {keep} node(s) "
            f"continue the walk whichever score sorted them, and every document is billed "
            f"for a guaranteed no-op. Raise candidates above {k_name}={keep}, or drop "
            f"rerank= (vector_search() is where reranking exactly k rows still shows: it "
            f"reports the new order and a rerank_score)"
        )


@dataclass
class Start:
    """The seed set a traversal begins from.

    ids: seed from specific node ids directly, instead of (or as well
    as) a property filter. `where` filters PROPERTIES -- `where={"id":
    7}` is a containment test against the JSONB bag, matches nothing,
    and says nothing while doing it (mutate.py's TestAddressingRowsById
    has the same trap on the write side). `ids` is the one way to name
    a row you are already holding -- a UI with a node selected, a
    traversal result fed back in. Combines with `where` as AND: both
    are constraints on the same row, matching mutate.py's `ids=`.
    `None` means no id filter, same as `where=None`; `[]` is an
    explicit empty selection and matches nothing, the same as
    `where={"some_key": []}` does for a property. Always scoped to
    this graph -- see Graph._scoped() -- since a node id is a global
    primary key and an id from another graph would otherwise be a
    perfectly valid (wrong) match.

    near/keep: seed from vector similarity instead of (or as well as)
    a property filter -- Near(field, vector) specs to rank by, and
    `keep` for how many of the most similar nodes to keep. `where`
    still applies; similarity ranks what survives it.

    boost: Boost(property, weight) terms added to the ranked score,
    for hybrid retrieval. A boost reorders; it never changes which
    nodes qualify.

    rerank: a Rerank(...) that re-scores the seed candidates by READING
    them against the query, before any Hop walks. `near` picks
    `rerank.candidates` seeds cheaply, the reranker reorders them, and
    `keep` truncates -- so the walk starts from a better-chosen set
    and everything downstream is unaware it happened. Needs `near`
    with `text=`: a reranker reads the query, and a raw vector is not
    something it can read. It also needs `keep`, strictly below
    `candidates`: the order is discarded (see below), so truncating
    is the only mark a reranker can leave on a subgraph. See
    hopai/rerankers.py.

    A traversal returns a SUBGRAPH, not a ranking: the similarity
    scores and their order do not survive into the result -- and
    neither do rerank scores, for the same reason. Use
    vector_search() when you need the scores themselves. See
    hopai/vectors.py.
    """
    where: Optional[Any] = None
    ids: Optional[list] = None
    label: Optional[str] = None
    near: Optional[Any] = None
    keep: Optional[int] = None
    boost: Optional[Any] = None
    rerank: Optional[Any] = None

    def __post_init__(self):
        _validate_near_k("Start", self.near, self.keep)
        _validate_boost("Start", self.near, self.boost)
        _validate_rerank("Start", self.near, self.keep, self.rerank)


@dataclass
class Hop:
    """One step of a traversal, following edges from the previous step's
    matched nodes.

    where:      filter on the node reached by this hop (None = no filter)

    via:        filter on every edge traversed during this hop (None = any
                edge). A plain dict, e.g. {"kind": "friend"} -- or the
                STORED_IN shorthand, a bare non-empty string, e.g.
                via="friend", equivalent to via={"kind": "friend"} but
                compiled to a text-equality expression a declared edge
                type's btree index can serve (see Graph.define_edge_type())
                instead of the JSONB containment test the dict form uses.
                See filters.resolve_via().

    hops:       how many real edges this hop may span -- an int for an exact
                count, or a (min, max) tuple for a range. Default 1.

    direction:  "forward" (follow start_id -> end_id) or "backward"
                (follow end_id -> start_id, i.e. "what points to this")

    optional:   if True, nodes that reach this point in the chain are kept
                even if this hop finds nothing for them (Cypher's
                OPTIONAL MATCH). Only valid on the LAST hop in a chain --
                see Graph.traverse() for why.

    label:      a name for your own reference; not used to build SQL.

    near:       rank the nodes this hop reaches by vector similarity --
                Near(field, vector) specs against NODE vector fields
                (edges walk by `via`; rank those with via_near). With
                `keep`, only the most similar reached nodes continue
                the chain and are reported: a semantic beam.

    keep:       how many of the most similar reached nodes to keep.
                (The number of EDGES a hop spans is `hops`.)

    via_near:   rank the EDGES this hop walks, against edge vector
                fields -- the `via` of similarity. With via_keep, each
                node follows only its via_keep most similar edges (a
                beam per source node, not a global truncation);
                without it, a Near's min_similarity filters which
                edges are worth walking at all.

    via_keep:   how many of the most similar edges to follow FROM EACH
                node reached so far.

    boost:      Boost(property, weight) terms added to the node
                ranking `near` creates. Reorders; never changes which
                nodes qualify. Edge beams have no boost term.

    rerank:     a Rerank(...) that re-scores the nodes THIS hop reached,
                by reading them against the query, before the next hop
                walks. This is the step-wise beam: a candidate here is
                not a row but a node plus how it was reached, so
                `document_from` may also read `.paths` -- which exists
                at a hop and not at a Start, since a seed has no
                provenance. Needs `near` with `text=`, and `keep`
                strictly below `rerank.candidates`: the order is
                discarded, so truncating is the only mark a reranker
                can leave on a subgraph.

    Similarity scores do not survive into the result -- a traversal
    returns a subgraph, not a ranking -- and neither do rerank scores.
    Use vector_search() for scores.
    """
    where: Optional[Any] = None
    via: Optional[Any] = None
    hops: HopCount = 1
    direction: str = "forward"
    optional: bool = False
    label: Optional[str] = None
    near: Optional[Any] = None
    keep: Optional[int] = None
    via_near: Optional[Any] = None
    via_keep: Optional[int] = None
    boost: Optional[Any] = None
    rerank: Optional[Any] = None

    def __post_init__(self):
        self.min_hops, self.max_hops = _normalize_hops(self.hops)
        if self.direction not in ("forward", "backward"):
            raise ValueError(f"direction must be 'forward' or 'backward', got {self.direction!r}")
        _validate_near_k("Hop", self.near, self.keep)
        _validate_near_k("Hop", self.via_near, self.via_keep,
                         near_name="via_near", k_name="via_keep")
        _validate_boost("Hop", self.near, self.boost)
        _validate_rerank("Hop", self.near, self.keep, self.rerank,
                         via_near=self.via_near)
