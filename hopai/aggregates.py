"""
hopai.aggregates

The aggregation DSL used by `Graph.aggregate()`: compute a number over
the nodes a traversal matches instead of hydrating them. Two equivalent
forms, exactly like filters:

  Python form (import and use directly):
      Count()                       how many nodes matched
      Count("priority")             how many matched nodes have the property
      Count("priority", distinct=True)   how many distinct values it takes
      Sum("priority") / Avg("priority")  over every matched node
      Sum("priority", distinct=True)     equal values collapse first
      Min("priority") / Max("priority")  (distinct could not change these)

  JSON form (for callers who can't or shouldn't write Python -- LLM tool
  calls, a future HTTP/MCP layer, config files):
      {"fn": "count"}
      {"fn": "count", "property": "priority", "distinct": true}
      {"fn": "avg", "property": "priority"}

  parse_aggregate() converts JSON form into the Python objects above,
  and cypher.py's RETURN translation produces these same objects too, so
  all three front ends compile through the one resolve_aggregate() --
  never add a second compilation path.

WHAT THE NUMBERS MEAN -- this is the part worth reading twice, because
"aggregate over a traversal" can mean three different things:

  1. Per matched node: every distinct node the traversal's LAST step
     matched counts once. This is hopai's native result set, and the
     default here.
  2. Per distinct value: equal property values collapse before the
     aggregate runs. That is `distinct=True`, and it is what Cypher's
     `avg(DISTINCT b.age)` means.
  3. Per path: a node reachable two ways counts twice. That is what
     Cypher's bare `avg(b.age)` means when hops are involved, and hopai
     CANNOT express it -- path multiplicity across hops is deliberately
     not tracked (see core.py on local paths). cypher.py refuses those
     spellings rather than approximating them.

EDGE CASES, all chosen to match what SQL and Cypher aggregates both do:

  - A missing property, a JSON null, and a non-numeric value are all
    ignored (NULL semantics) -- a stray string in one node must not
    error the whole query, and both Cypher and PG skip NULLs.
  - Over an empty match set: count -> 0, sum -> 0 (Cypher's choice, and
    Python's `sum([])`), avg/min/max -> None.
  - sum/avg/min/max are numeric. Lexicographic min/max over strings is a
    possible follow-up, not a silent fallback -- "z" > "10" answering a
    question about priorities would be the kind of quietly-wrong answer
    this library exists to refuse.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import Numeric, case, cast, distinct, func


class Count:
    """How many nodes the traversal matched; with `property`, how many of
    them carry it (distinct=True: how many distinct values it takes)."""

    def __init__(self, property: Optional[str] = None, distinct: bool = False):
        if property is not None and not isinstance(property, str):
            raise TypeError(f"property must be a string key, got {property!r}")
        if distinct and property is None:
            raise ValueError(
                "distinct without a property is redundant -- matched nodes are already "
                "distinct. Count() counts them; Count('key', distinct=True) counts values"
            )
        self.property, self.distinct = property, distinct

    def __repr__(self) -> str:
        if self.property is None:
            return "Count()"
        return f"Count({self.property!r}{', distinct=True' if self.distinct else ''})"


class _NumericAggregate:
    """Shared shape of Sum/Avg/Min/Max: one property key, numeric."""

    def __init__(self, property: str):
        if not isinstance(property, str):
            raise TypeError(f"property must be a string key, got {property!r}")
        self.property = property

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.property!r})"


class Sum(_NumericAggregate):
    """Sum of a numeric property over every matched node; distinct=True
    sums each value once however many nodes carry it. 0 when nothing
    matches."""

    def __init__(self, property: str, distinct: bool = False):
        super().__init__(property)
        self.distinct = distinct

    def __repr__(self) -> str:
        return f"Sum({self.property!r}{', distinct=True' if self.distinct else ''})"


class Avg(_NumericAggregate):
    """Average of a numeric property over every matched node;
    distinct=True averages the distinct values instead. None when
    nothing matches."""

    def __init__(self, property: str, distinct: bool = False):
        super().__init__(property)
        self.distinct = distinct

    def __repr__(self) -> str:
        return f"Avg({self.property!r}{', distinct=True' if self.distinct else ''})"


class Min(_NumericAggregate):
    """Smallest numeric value of the property among matched nodes. No
    distinct flag -- the minimum of the distinct values is the minimum."""


class Max(_NumericAggregate):
    """Largest numeric value of the property among matched nodes."""


def _numeric_value(column, key: str):
    """The property as NUMERIC when it holds a JSON number, else NULL.

    Guarded with jsonb_typeof rather than cast directly: an aggregate
    touches every matched row, so a single node carrying "high" where a
    number was expected would abort the whole query mid-scan. NULL
    instead means "ignored", which is what both Cypher and PG aggregates
    do with missing values -- and PropertyType('key', 'number') is the
    constraint that keeps such rows out in the first place."""
    return case((func.jsonb_typeof(column[key]) == "number", cast(column[key].astext, Numeric)))


def resolve_aggregate(column, agg: Any):
    """Compile one aggregate into a SQLAlchemy expression bound to
    `column` (the JSONB properties column of the matched nodes). This is
    the single code path every aggregate -- Python, JSON or Cypher --
    passes through before it becomes SQL."""
    if isinstance(agg, Count):
        if agg.property is None:
            return func.count()
        # ->> (astext), not ->: it is SQL NULL for a missing key AND for
        # an explicit JSON null, so both read as "property absent" --
        # the same judgement Cypher makes about null properties.
        value = column[agg.property].astext
        return func.count(distinct(value) if agg.distinct else value)

    if isinstance(agg, (Sum, Avg)):
        value = _numeric_value(column, agg.property)
        if agg.distinct:
            value = distinct(value)
        if isinstance(agg, Sum):
            # Empty-set sum is NULL in PG and 0 in Cypher and in Python's
            # sum([]); 0 is the answer nobody has to null-check.
            return func.coalesce(func.sum(value), 0)
        return func.avg(value)

    if isinstance(agg, (Min, Max)):
        fn = func.min if isinstance(agg, Min) else func.max
        return fn(_numeric_value(column, agg.property))

    raise TypeError(
        f"aggregate must be Count, Sum, Avg, Min or Max -- got {type(agg).__name__}"
    )


_AGGREGATE_CLASSES = {"count": Count, "sum": Sum, "avg": Avg, "min": Min, "max": Max}


def parse_aggregate(spec: dict):
    """Convert a JSON-form aggregate into the Python objects
    resolve_aggregate() understands. This is the only function a
    JSON/tool-calling caller needs -- everything downstream is identical
    to the Python API.

    Accepts {"fn": ..., "property": ..., "distinct": ...} where fn is
    one of count/sum/avg/min/max, property is required for everything
    but count, and distinct applies to count/sum/avg only.
    """
    if not isinstance(spec, dict):
        raise TypeError(f"JSON aggregate must be an object -- got {type(spec).__name__}")
    unknown = set(spec) - {"fn", "property", "distinct"}
    if unknown:
        raise ValueError(
            f"unknown aggregate keys {sorted(unknown)} -- an aggregate is "
            f'{{"fn": ..., "property": ..., "distinct": ...}}'
        )
    fn = spec.get("fn")
    if fn not in _AGGREGATE_CLASSES:
        raise ValueError(f"'fn' must be one of {sorted(_AGGREGATE_CLASSES)} -- got {fn!r}")
    prop = spec.get("property")
    distinct_flag = spec.get("distinct", False)
    if not isinstance(distinct_flag, bool):
        raise ValueError(f"'distinct' must be true or false -- got {distinct_flag!r}")

    if fn == "count":
        return Count(prop, distinct=distinct_flag)
    if prop is None:
        raise ValueError(f"'{fn}' aggregates a property -- add \"property\"")
    if fn in ("sum", "avg"):
        return _AGGREGATE_CLASSES[fn](prop, distinct=distinct_flag)
    if distinct_flag:
        raise ValueError(
            f"'distinct' does not apply to '{fn}' -- the {fn} of the distinct values "
            f"is the {fn}. Drop it"
        )
    return _AGGREGATE_CLASSES[fn](prop)
