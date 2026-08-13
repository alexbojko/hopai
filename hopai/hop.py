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


@dataclass
class Start:
    """The seed set a traversal begins from."""
    where: Optional[Any] = None
    label: Optional[str] = None


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
    """
    where: Optional[Any] = None
    via: Optional[Any] = None
    hops: HopCount = 1
    direction: str = "forward"
    optional: bool = False
    label: Optional[str] = None

    def __post_init__(self):
        self.min_hops, self.max_hops = _normalize_hops(self.hops)
        if self.direction not in ("forward", "backward"):
            raise ValueError(f"direction must be 'forward' or 'backward', got {self.direction!r}")

# (probe) touched to exercise the CI mutation leg; reverted in the next commit
