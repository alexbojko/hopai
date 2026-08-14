"""
hopai.filters

The filter language used everywhere a `where=` or `via=` argument is
accepted. Two equivalent forms:

  Python form (import and use directly):
      {"type": "person"}                        equality
      {"type": "person", "active": True}        AND of keys, same dict
      {"type": ["person", "company"]}           OR of values, one key (IN-like)
      OR({"type": "person"}, {"type": "company"})
      AND(OR(...), {"active": True})
      NOT({"type": "person"})
      GT("age", 18) / GTE / LT / LTE
      BETWEEN("age", 18, 65)
      lambda col: col.op("~")("^A")             escape hatch: any callable
                                                 receiving the real column and
                                                 returning a real SQLAlchemy
                                                 boolean expression

  JSON form (for callers who can't or shouldn't write Python -- LLM tool
  calls, a future HTTP/MCP layer, config files):
      {"type": "person"}
      {"and": [{"type": "person"}, {"active": true}]}
      {"or": [{"type": "person"}, {"type": "company"}]}
      {"not": {"type": "person"}}
      {"gt": ["age", 18]}
      {"gte": ["age", 18]}
      {"lt": ["age", 65]}
      {"lte": ["age", 65]}
      {"between": ["age", 18, 65]}

  parse_filter() converts JSON form into the Python objects above; both
  forms compile through the exact same `resolve()` function, so there is
  only one code path to trust, not two parallel implementations.

WHY THIS SHAPE: a plain dict can only express AND-of-equalities (JSONB
containment can't express OR on its own), and a bare top-level list was
tried and rejected during development -- it read ambiguously as "AND
these" to a human, when it meant OR. OR()/AND()/NOT() are explicit on
purpose: what a filter means is visible from what you wrote, not
inferred from which Python type you happened to reach for.
"""

from __future__ import annotations

import json as _json
from typing import Any, Optional, Union

from sqlalchemy import Numeric, and_, cast, literal, not_, or_
from sqlalchemy.dialects.postgresql import JSONB


class OR:
    """OR across several filters: OR({"type": "person"}, {"type": "company"})
    Each argument can be a dict, another OR(...)/AND(...)/NOT(...), or a
    callable escape hatch."""

    def __init__(self, *filters: Any):
        self.filters = filters

    def __repr__(self) -> str:
        return f"OR({', '.join(repr(f) for f in self.filters)})"


class AND:
    """Explicit AND across several filters -- mainly for combining an
    OR(...) with an additional required condition:
        AND(OR({"type": "person"}, {"type": "company"}), {"active": True})
    A plain dict already means AND-of-its-keys on its own; use this class
    to AND whole sub-expressions together instead."""

    def __init__(self, *filters: Any):
        self.filters = filters

    def __repr__(self) -> str:
        return f"AND({', '.join(repr(f) for f in self.filters)})"


class NOT:
    """Negate a filter. NOT({"type": "person"}) matches everything that
    is NOT type=='person' -- including rows where 'type' is absent
    entirely, since JSONB containment evaluates to false (not unknown)
    for a missing key. This is a deliberate, documented difference from
    naive equality-based negation (`x <> 'person'`), which treats a
    missing key as SQL NULL and silently excludes it -- verified during
    development to be a real, easy-to-hit trap in Cypher's own `NOT`."""

    def __init__(self, filt: Any):
        self.filt = filt

    def __repr__(self) -> str:
        return f"NOT({self.filt!r})"


class GT:
    """key > value, numeric comparison on a property."""

    def __init__(self, key: str, value: Union[int, float]):
        self.key, self.value = key, value

    def __repr__(self) -> str:
        return f"GT({self.key!r}, {self.value!r})"


class GTE:
    def __init__(self, key: str, value: Union[int, float]):
        self.key, self.value = key, value

    def __repr__(self) -> str:
        return f"GTE({self.key!r}, {self.value!r})"


class LT:
    def __init__(self, key: str, value: Union[int, float]):
        self.key, self.value = key, value

    def __repr__(self) -> str:
        return f"LT({self.key!r}, {self.value!r})"


class LTE:
    def __init__(self, key: str, value: Union[int, float]):
        self.key, self.value = key, value

    def __repr__(self) -> str:
        return f"LTE({self.key!r}, {self.value!r})"


class BETWEEN:
    """low <= key <= high (inclusive both ends)."""

    def __init__(self, key: str, low: Union[int, float], high: Union[int, float]):
        self.key, self.low, self.high = key, low, high

    def __repr__(self) -> str:
        return f"BETWEEN({self.key!r}, {self.low!r}, {self.high!r})"


def _numeric_field(column, key: str):
    """Extract a JSONB property as text and cast to NUMERIC. A property
    that's missing or non-numeric compares as SQL NULL, which correctly
    excludes that row from a range match rather than raising."""
    return cast(column[key].astext, Numeric)


def resolve(column, filt: Any):
    """Compile a filter (Python form) into a real SQLAlchemy boolean
    expression bound to `column`. This is the single code path every
    filter -- from either the Python API or the JSON API -- passes
    through before it becomes SQL."""
    if filt is None:
        return literal(True)

    # Checked by marker, not isinstance: importing hopai.vectors here
    # would be a cycle, and the mistake deserves a better answer than
    # the generic rejection below.
    if getattr(filt, "_is_near", False):
        raise TypeError(
            "Near(...) ranks rows by similarity; it is not a boolean filter. Pass it as "
            "near= on Start/Hop or to vector_search(), not inside where=/via="
        )

    if callable(filt) and not isinstance(filt, (OR, AND, NOT)):
        return filt(column)

    if isinstance(filt, list):
        raise TypeError(
            "a bare list is ambiguous -- use OR(...) to mean 'any of these filters', "
            "e.g. OR({'type': 'person'}, {'type': 'company'}) instead of "
            "[{'type': 'person'}, {'type': 'company'}]"
        )

    if isinstance(filt, OR):
        return or_(*(resolve(column, f) for f in filt.filters))

    if isinstance(filt, AND):
        return and_(*(resolve(column, f) for f in filt.filters))

    if isinstance(filt, NOT):
        return not_(resolve(column, filt.filt))

    if isinstance(filt, GT):
        return _numeric_field(column, filt.key) > filt.value
    if isinstance(filt, GTE):
        return _numeric_field(column, filt.key) >= filt.value
    if isinstance(filt, LT):
        return _numeric_field(column, filt.key) < filt.value
    if isinstance(filt, LTE):
        return _numeric_field(column, filt.key) <= filt.value
    if isinstance(filt, BETWEEN):
        field = _numeric_field(column, filt.key)
        return and_(field >= filt.low, field <= filt.high)

    if isinstance(filt, dict):
        if not filt:
            return literal(True)
        conditions = []
        for key, value in filt.items():
            if isinstance(value, list):
                conditions.append(
                    or_(*(column.op("@>")(cast(literal(_json.dumps({key: v})), JSONB)) for v in value))
                )
            else:
                conditions.append(column.op("@>")(cast(literal(_json.dumps({key: value})), JSONB)))
        return and_(*conditions)

    raise TypeError(
        f"filter must be None, a dict, AND/OR/NOT/GT/GTE/LT/LTE/BETWEEN, or a callable "
        f"-- got {type(filt).__name__}"
    )


def parse_filter(spec: Optional[dict]):
    """Convert a JSON-form filter into the Python objects resolve()
    understands. This is the only function a JSON/tool-calling caller
    needs -- everything downstream is identical to the Python API.

    Accepts plain dicts (equality/AND, unchanged from Python form) plus
    the operator keys: and, or, not, gt, gte, lt, lte, between.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise TypeError(f"JSON filter must be an object -- got {type(spec).__name__}")

    operator_keys = {"and", "or", "not", "gt", "gte", "lt", "lte", "between"}
    present = operator_keys & spec.keys()

    if not present:
        # plain equality/AND-of-keys dict -- passed through unchanged
        return spec

    # `present` is a subset of spec's keys, so "more than one operator"
    # is already covered by "more than one key" -- one condition, not two
    # half-redundant ones a mutation can silently weaken.
    if len(spec) > 1:
        raise ValueError(
            f"an operator filter must have exactly one key from {sorted(operator_keys)} "
            f"and nothing else -- got {list(spec.keys())}"
        )

    op = next(iter(present))
    arg = spec[op]

    if op == "and":
        return AND(*(parse_filter(f) for f in arg))
    if op == "or":
        return OR(*(parse_filter(f) for f in arg))
    if op == "not":
        return NOT(parse_filter(arg))
    if op == "gt":
        return GT(arg[0], arg[1])
    if op == "gte":
        return GTE(arg[0], arg[1])
    if op == "lt":
        return LT(arg[0], arg[1])
    if op == "lte":
        return LTE(arg[0], arg[1])
    if op == "between":
        return BETWEEN(arg[0], arg[1], arg[2])

    raise AssertionError("unreachable")  # `present` already validated against operator_keys
