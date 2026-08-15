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


@dataclass
class Start:
    """The seed set a traversal begins from.

    near/keep: seed from vector similarity instead of (or as well as)
    a property filter -- Near(field, vector) specs to rank by, and
    `keep` for how many of the most similar nodes to keep. `where`
    still applies; similarity ranks what survives it.

    boost: Boost(property, weight) terms added to the ranked score,
    for hybrid retrieval. A boost reorders; it never changes which
    nodes qualify.

    A traversal returns a SUBGRAPH, not a ranking: the similarity
    scores and their order do not survive into the result. Use
    vector_search() when you need the scores themselves. See
    hopai/vectors.py.
    """
    where: Optional[Any] = None
    label: Optional[str] = None
    near: Optional[Any] = None
    keep: Optional[int] = None
    boost: Optional[Any] = None

    def __post_init__(self):
        _validate_near_k("Start", self.near, self.keep)
        _validate_boost("Start", self.near, self.boost)


@dataclass
class Hop:
    """One step of a traversal, following edges from the previous step's
    matched nodes.

    where:      filter on the node reached by this hop (None = no filter)
    via:        filter on every edge traversed during this hop (None = any edge)
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

    Similarity scores do not survive into the result -- a traversal
    returns a subgraph, not a ranking. Use vector_search() for scores.
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

    def __post_init__(self):
        self.min_hops, self.max_hops = _normalize_hops(self.hops)
        if self.direction not in ("forward", "backward"):
            raise ValueError(f"direction must be 'forward' or 'backward', got {self.direction!r}")
        _validate_near_k("Hop", self.near, self.keep)
        _validate_near_k("Hop", self.via_near, self.via_keep,
                         near_name="via_near", k_name="via_keep")
        _validate_boost("Hop", self.near, self.boost)
