"""
hopai.cypher

Cypher as an input syntax, for reading and for writing.

MATCH translates into the same `(Start, [Hop, ...])` pair
`hopai.json_api.spec_to_traversal()` produces. CREATE and MERGE translate
into the same add_nodes/merge_nodes/add_edges operations the Python API
calls. Neither has query logic of its own -- this module is a front end,
and there is exactly one traversal engine and one write path underneath.

    graph.cypher("MATCH (a:person)-[:friend*1..4]->(b {active: true}) RETURN b")
    graph.cypher("CREATE (a:person {email: 'a@x.com'})-[:friend]->(b:person {email: 'b@x.com'})")
    graph.cypher("MERGE (a:person {email: 'a@x.com'}) ON CREATE SET a.name = 'Alice'")

`graph.cypher()` returns a Subgraph for a query that reads and an
IngestResult for one that writes; traverse_cypher() and write_cypher()
are the same thing when you would rather be explicit, and
`graph.cypher_operations()` shows a write's plan without running it.

WRITES, and the three places they stop short of Cypher:

  MERGE on a whole path.  Cypher's `MERGE (a {..})-[:x]->(b {..})`
                          matches the ENTIRE pattern and creates all of
                          it when it does not match -- duplicating nodes
                          that already exist independently. That is a
                          famous footgun, so a relationship MERGE here
                          requires both endpoints to be bound already:
                          MATCH or MERGE the nodes, then the edge.
  MERGE needs an index.   The conflict keys are every property in the
                          pattern (which is what Cypher matches on), and
                          a unique index must cover exactly them.
                          Anything that should not take part in matching
                          belongs in ON CREATE SET. Cypher itself needs
                          no index and races instead; the error here
                          names the Unique() to declare.
  MATCH before a write.   Binds single nodes by their properties, one
                          lookup each. It does not traverse -- a write
                          driven by a multi-hop match is a different
                          feature.

SET on matched rows, DELETE and DETACH DELETE are not supported at all:
there is no update-by-query or delete API here yet, in Cypher or in
Python, and offering half of one through this door only would be a way
to lose data quietly.

THE RULE THIS MODULE IS BUILT ON: refuse, don't approximate. Cypher can
say things hopai cannot, and a few things it says in a shape hopai
supports but with *different semantics*. Every one of those raises
CypherError naming the rewrite, rather than returning a traversal that
quietly answers a different question. That is the same choice
`filters.resolve()` makes when it rejects a bare list.

WHAT DOES NOT TRANSLATE, and why:

  RETURN projects nothing.    hopai returns a subgraph -- every node and
                              edge on a matching chain -- so a plain
                              RETURN is parsed and ignored, and ORDER BY
                              / LIMIT still refuse. The exception is an
                              aggregating RETURN, which becomes a
                              Graph.aggregate() call -- see AGGREGATION
                              below for exactly which spellings, and why
                              the rest refuse.
  Cross-variable OR.          `WHERE a.x = 1 OR b.y = 2` spans two hops'
                              filters; a Hop filter binds one node.
  Unbounded `*`.              `max_hops` drives the recursion guard, so
                              `-[*]->` needs `*1..N` or max_var_length=.
  Undirected `-[]-`.          Hop.direction is forward or backward only.
  Disjoint patterns.          One linear chain, per the library's scope.
  Bare `<>` and `NOT x = y`.  THE SUBTLE ONE. Cypher evaluates
                              `a.type <> 'leaf'` to NULL when `type` is
                              missing, dropping that row; hopai's
                              containment-based NOT *includes* it. Same
                              spelling, different result set, so this
                              raises. The NULL-safe idiom
                              `a.type IS NULL OR a.type <> 'leaf'`
                              (recognized as a single unit, not by any
                              compositional rule) is what maps exactly
                              onto NOT({"type": "leaf"}).

AGGREGATION: `RETURN count(DISTINCT b)` and friends translate to
Graph.aggregate(), which aggregates over the distinct nodes the LAST
step of the chain matched. Cypher, however, aggregates over result ROWS
-- one per path -- and that difference decides everything here. Three
semantics exist:

  per matched node    each distinct node once. hopai's native result;
                      Cypher spells it `WITH DISTINCT b RETURN avg(b.age)`,
                      and that exact WITH form is accepted as a unit
                      (like the null-safe negation idiom).
  per distinct value  equal values collapse first. Cypher's
                      `avg(DISTINCT b.age)` -- accepted, exact.
  per path            a node reachable two ways counts twice. Cypher's
                      bare `avg(b.age)` when hops are involved. hopai
                      does not track path multiplicity across hops, so
                      these REFUSE rather than quietly answering the
                      per-node question.

Consequences, each a one-line rule:
  - `count(DISTINCT b)` is exact (the distinct nodes ARE the count).
  - `min`/`max` are exact bare or DISTINCT -- multiplicity cannot
    change an extremum.
  - With no hops there are no paths to multiply, so bare `count(a)` /
    `avg(a.age)` on a single-node pattern are exact too.
  - Only the LAST node of the chain can be aggregated: a mid-chain
    match may include nodes with no continuation to the chain's end,
    which Cypher would not count. Reverse the pattern instead.
  - Mixing aggregates with plain return items is grouping (GROUP BY),
    which hopai does not have yet; relationship-variable aggregates and
    collect()/stdev()/percentiles are not supported yet either.

LABELS: hopai has no label concept -- a node is its JSONB properties. So
`(a:person)` compiles to a property test, `{node_label_key: "person"}`,
defaulting to the `type` key this repo's own fixtures use. When a label
and an inline property collide on that key -- `(a:Node {type: 'leaf'})`,
the shape the Cypher in benchmarks/README.md is written in -- that is an
unsatisfiable AND, so it raises instead of silently matching nothing.
Pass node_label_key=None to ignore labels entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import not_ as _sa_not

from .core import Graph, Subgraph
from .filters import AND, GT, GTE, LT, LTE, NOT, OR
from .hop import Hop, Start


class CypherError(ValueError):
    """Raised for Cypher this translator will not translate: a syntax
    error, a construct hopai has no equivalent for, or -- the case worth
    knowing about -- one that would translate into different semantics
    than Cypher gives it. Subclasses ValueError so existing
    `except ValueError` handling still catches it."""


# ---------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------

# Longest match first: '<--' must beat '<-' must beat '<', and '..' must
# beat '.', or `*1..4` tokenizes as a decimal point.
_PUNCT = [
    "<--", "-->", "<-", "->", "--", "<>", "<=", ">=", "..",
    "(", ")", "[", "]", "{", "}", ":", ",", ".", "*", "|", "=", "<", ">", "-",
]
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"'}


@dataclass
class _Tok:
    kind: str  # 'name' | 'number' | 'string' | 'punct' | 'eof'
    value: Any
    pos: int


def _tokenize(text: str) -> list:
    toks: list = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        if c in "'\"":
            j, buf = i + 1, []
            while j < n and text[j] != c:
                if text[j] == "\\" and j + 1 < n:
                    buf.append(_ESCAPES.get(text[j + 1], text[j + 1]))
                    j += 2
                else:
                    buf.append(text[j])
                    j += 1
            if j >= n:
                raise CypherError(f"unterminated string literal at position {i}")
            toks.append(_Tok("string", "".join(buf), i))
            i = j + 1
            continue
        m = _NAME_RE.match(text, i)
        if m:
            toks.append(_Tok("name", m.group(), i))
            i = m.end()
            continue
        m = _NUM_RE.match(text, i)
        if m:
            raw = m.group()
            toks.append(_Tok("number", float(raw) if "." in raw else int(raw), i))
            i = m.end()
            continue
        for p in _PUNCT:
            if text.startswith(p, i):
                toks.append(_Tok("punct", p, i))
                i += len(p)
                break
        else:
            raise CypherError(f"unexpected character {c!r} at position {i}")
    toks.append(_Tok("eof", None, n))
    return toks


# ---------------------------------------------------------------------
# Parsed pattern / expression AST
# ---------------------------------------------------------------------

@dataclass
class _NodePat:
    var: Optional[str]
    labels: list
    props: dict


@dataclass
class _RelPat:
    var: Optional[str]
    types: list
    props: dict
    direction: str            # 'forward' | 'backward'
    lo: Optional[int]         # None = unbounded (no `*` at all is (1, 1))
    hi: Optional[int]


@dataclass
class _MatchClause:
    nodes: list
    rels: list
    where: Any
    optional: bool
    path_var: Optional[str]
    #: Patterns after the first comma. A traversal is one linear chain and
    #: refuses these; a write treats each as an independent lookup, which
    #: is what `MATCH (a {...}), (b {...}) CREATE (a)-[:x]->(b)` needs.
    extra_patterns: list = field(default_factory=list)


@dataclass
class _WriteClause:
    kind: str            # 'create' | 'merge'
    patterns: list       # [(nodes, rels), ...]
    on_create: dict = field(default_factory=dict)   # var -> {key: value}
    on_match: dict = field(default_factory=dict)


@dataclass
class _Or:
    terms: list


@dataclass
class _And:
    terms: list


@dataclass
class _Not:
    term: Any


@dataclass
class _Cmp:
    var: str
    key: str
    op: str
    value: Any


@dataclass
class _In:
    var: str
    key: str
    values: list


@dataclass
class _IsNull:
    var: str
    key: str
    negated: bool = False


@dataclass
class _AllRels:
    """all(r IN relationships(p) WHERE <expr>) -- an edge predicate over
    every relationship on a path, which is exactly what Hop.via means."""
    rel_var: str
    path_var: str
    expr: Any


@dataclass
class _AggItem:
    """One aggregate in a RETURN: `avg(DISTINCT b.age) AS mean`."""
    fn: str                  # 'count' | 'sum' | 'avg' | 'min' | 'max'
    var: Optional[str]       # None only for count(*)
    key: Optional[str]       # None for count(b) / count(*)
    distinct: bool
    alias: Optional[str]


@dataclass
class _ReturnClause:
    """An aggregating RETURN, possibly prefixed by `WITH DISTINCT var`.
    A RETURN with no aggregates never becomes a clause -- it is parsed
    and ignored, since the subgraph is the result either way."""
    items: list              # [_AggItem, ...], never empty
    distinct_var: Optional[str]


#: The aggregate functions Graph.aggregate() computes.
_TRANSLATED_AGGREGATES = {"count", "sum", "avg", "min", "max"}
#: Recognized as aggregation so they refuse helpfully instead of being
#: mistaken for a projection and ignored.
_AGGREGATES = _TRANSLATED_AGGREGATES | {"collect", "stdev", "percentiledisc",
                                        "percentilecont"}
_SUBGRAPH = ("hopai returns the whole subgraph of every node and edge on a matching "
             "chain, so there is nothing for this clause to act on")
_NO_DELETE = ("hopai has no delete API yet, in Cypher or in Python -- remove rows with SQL "
              "for now")

#: Complete sentences: each is raised as-is, because "X is not supported"
#: with a generic suffix reads wrong once the reasons differ this much.
_UNSUPPORTED_CLAUSES = {
    "WITH": f"WITH is not supported: {_SUBGRAPH}, and no intermediate projection to "
            f"carry. The one exception is `WITH DISTINCT <var> RETURN <aggregates>`, "
            f"the spelling of aggregation per matched node",
    "UNWIND": f"UNWIND is not supported: {_SUBGRAPH}",
    "SET": "a bare SET (updating rows a MATCH found) is not supported. Only "
           "`MERGE ... ON CREATE SET` and `ON MATCH SET` are",
    "DELETE": f"DELETE is not supported: {_NO_DELETE}",
    "DETACH": f"DETACH DELETE is not supported: {_NO_DELETE}",
    "REMOVE": f"REMOVE is not supported: {_NO_DELETE}",
    "CALL": "CALL is not supported",
    "FOREACH": "FOREACH is not supported",
    "UNION": f"UNION is not supported: {_SUBGRAPH}",
    "ORDER": f"ORDER BY is not supported: {_SUBGRAPH}, unordered",
    "SKIP": f"SKIP is not supported: {_SUBGRAPH}",
    "LIMIT": f"LIMIT is not supported: {_SUBGRAPH}",
}


# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------

class _Parser:
    def __init__(self, toks: list):
        self.toks = toks
        self.i = 0

    # -- token helpers --------------------------------------------------

    def _peek(self, ahead: int = 0) -> _Tok:
        j = min(self.i + ahead, len(self.toks) - 1)
        return self.toks[j]

    def _next(self) -> _Tok:
        t = self.toks[self.i]
        if t.kind != "eof":
            self.i += 1
        return t

    def _at_punct(self, *values) -> bool:
        t = self._peek()
        return t.kind == "punct" and t.value in values

    def _at_kw(self, *words) -> bool:
        t = self._peek()
        return t.kind == "name" and t.value.upper() in words

    def _at_name(self) -> bool:
        return self._peek().kind == "name"

    def _expect_punct(self, value: str) -> _Tok:
        if not self._at_punct(value):
            t = self._peek()
            raise CypherError(f"expected {value!r} at position {t.pos}, got {self._describe(t)}")
        return self._next()

    def _expect_kw(self, word: str) -> _Tok:
        if not self._at_kw(word):
            t = self._peek()
            raise CypherError(f"expected {word} at position {t.pos}, got {self._describe(t)}")
        return self._next()

    def _expect_name(self) -> str:
        t = self._peek()
        if t.kind != "name":
            raise CypherError(f"expected a name at position {t.pos}, got {self._describe(t)}")
        return self._next().value

    @staticmethod
    def _describe(t: _Tok) -> str:
        return "end of query" if t.kind == "eof" else repr(t.value)

    # -- top level ------------------------------------------------------

    def parse(self) -> list:
        clauses: list = []
        while self._peek().kind != "eof":
            if self._at_kw("MATCH"):
                self._next()
                clauses.append(self._parse_match(optional=False))
            elif self._at_kw("OPTIONAL"):
                self._next()
                self._expect_kw("MATCH")
                clauses.append(self._parse_match(optional=True))
            elif self._at_kw("CREATE"):
                self._next()
                clauses.append(self._parse_write("create"))
            elif self._at_kw("MERGE"):
                self._next()
                clauses.append(self._parse_write("merge"))
            elif self._at_kw("RETURN"):
                self._next()
                ret = self._parse_return(distinct_var=None)
                if ret is not None:
                    clauses.append(ret)
            elif self._at_kw("WITH"):
                clauses.append(self._parse_with_distinct_return())
            elif self._at_name():
                word = self._peek().value.upper()
                if word in _UNSUPPORTED_CLAUSES:
                    raise CypherError(_UNSUPPORTED_CLAUSES[word])
                raise CypherError(f"unexpected {self._peek().value!r} at position {self._peek().pos}")
            else:
                t = self._peek()
                raise CypherError(f"unexpected {self._describe(t)} at position {t.pos}")
        if not any(isinstance(c, (_MatchClause, _WriteClause)) for c in clauses):
            raise CypherError("query has no MATCH clause")
        return clauses

    def _parse_with_distinct_return(self) -> _ReturnClause:
        """`WITH DISTINCT <var> RETURN <aggregates>` -- Cypher's spelling
        of aggregation per matched node, recognized as one unit (the same
        move as the null-safe negation idiom: neither half translates
        alone). Any other WITH raises the general refusal."""
        self._expect_kw("WITH")
        if not (self._at_kw("DISTINCT") and self._peek(1).kind == "name"
                and self._peek(2).kind == "name" and self._peek(2).value.upper() == "RETURN"):
            raise CypherError(_UNSUPPORTED_CLAUSES["WITH"])
        self._next()                       # DISTINCT
        var = self._expect_name()
        self._expect_kw("RETURN")
        ret = self._parse_return(distinct_var=var)
        if ret is None:
            raise CypherError(
                f"WITH DISTINCT {var} is only supported before an aggregating RETURN -- "
                f"without an aggregate the result is the subgraph, already deduplicated, "
                f"so drop the WITH"
            )
        return ret

    def _parse_return(self, distinct_var: Optional[str]) -> Optional[_ReturnClause]:
        """A RETURN with aggregates parses into a _ReturnClause; one
        without is consumed and ignored (the result of a traversal is the
        full matching subgraph, so a plain projection has nothing to
        project) -- while still rejecting what following it would
        silently lose (ORDER BY, LIMIT, ...)."""
        if not self._sees_aggregate():
            if distinct_var is not None:
                return None   # _parse_with_distinct_return raises with the better message
            while self._peek().kind != "eof":
                t = self._peek()
                if t.kind == "name" and t.value.upper() in _UNSUPPORTED_CLAUSES:
                    raise CypherError(_UNSUPPORTED_CLAUSES[t.value.upper()])
                self._next()
            return None

        items = [self._parse_return_item()]
        while self._at_punct(","):
            self._next()
            items.append(self._parse_return_item())
        t = self._peek()
        if t.kind == "name" and t.value.upper() in _UNSUPPORTED_CLAUSES:
            raise CypherError(_UNSUPPORTED_CLAUSES[t.value.upper()])
        if t.kind != "eof":
            raise CypherError(f"unexpected {self._describe(t)} at position {t.pos}")
        return _ReturnClause(items=items, distinct_var=distinct_var)

    def _sees_aggregate(self) -> bool:
        """Whether the rest of the query contains an aggregate call --
        decides between strict item parsing and scan-and-ignore."""
        for j in range(self.i, len(self.toks) - 1):
            if (self.toks[j].kind == "name" and self.toks[j].value.lower() in _AGGREGATES
                    and self.toks[j + 1].kind == "punct" and self.toks[j + 1].value == "("):
                return True
        return False

    def _parse_return_item(self) -> _AggItem:
        t = self._peek()
        if t.kind != "name" or not (self._peek(1).kind == "punct" and self._peek(1).value == "("):
            raise CypherError(
                f"RETURN mixes an aggregate with a plain item at position {t.pos} -- that "
                f"is grouped aggregation (GROUP BY), which hopai does not support yet. "
                f"Either aggregate every item, or drop the aggregates and take the subgraph"
            )
        fn = self._next().value.lower()
        if fn not in _TRANSLATED_AGGREGATES:
            raise CypherError(
                f"{fn}(...) is not supported -- the aggregates hopai computes are "
                f"{', '.join(sorted(_TRANSLATED_AGGREGATES))}"
            )
        self._expect_punct("(")
        distinct = False
        if self._at_kw("DISTINCT"):
            self._next()
            distinct = True
        if self._at_punct("*"):
            self._next()
            var, key = None, None
        else:
            var = self._expect_name()
            key = None
            if self._at_punct("."):
                self._next()
                key = self._expect_name()
        self._expect_punct(")")
        alias = None
        if self._at_kw("AS"):
            self._next()
            alias = self._expect_name()
        return _AggItem(fn=fn, var=var, key=key, distinct=distinct, alias=alias)

    # -- MATCH ----------------------------------------------------------

    def _parse_match(self, optional: bool) -> _MatchClause:
        path_var = None
        if self._at_name() and self._peek(1).kind == "punct" and self._peek(1).value == "=":
            path_var = self._next().value
            self._next()  # '='
        patterns = self._parse_pattern_list()
        where = None
        if self._at_kw("WHERE"):
            self._next()
            where = self._parse_or()
        nodes, rels = patterns[0]
        return _MatchClause(nodes=nodes, rels=rels, where=where, optional=optional,
                            path_var=path_var, extra_patterns=patterns[1:])

    def _parse_pattern_list(self) -> list:
        patterns = [self._parse_pattern()]
        while self._at_punct(","):
            self._next()
            patterns.append(self._parse_pattern())
        return patterns

    def _parse_write(self, kind: str) -> _WriteClause:
        clause = _WriteClause(kind=kind, patterns=self._parse_pattern_list())
        if kind == "merge":
            while self._at_kw("ON"):
                self._next()
                if self._at_kw("CREATE"):
                    target = clause.on_create
                elif self._at_kw("MATCH"):
                    target = clause.on_match
                else:
                    tok = self._peek()
                    raise CypherError(f"expected CREATE or MATCH after ON at position {tok.pos}")
                self._next()
                self._expect_kw("SET")
                for var, assignments in self._parse_set_assignments().items():
                    target.setdefault(var, {}).update(assignments)
        return clause

    def _parse_set_assignments(self) -> dict:
        """`a.x = 1, a.y = 'two'` -- the only SET form there is here.
        `a += {...}` and `a = {...}` replace or merge whole maps, which
        would need semantics this does not have."""
        assignments: dict = {}
        while True:
            var, key = self._parse_property_ref()
            self._expect_punct("=")
            assignments.setdefault(var, {})[key] = self._parse_literal()
            if not self._at_punct(","):
                return assignments
            self._next()

    def _parse_pattern(self) -> tuple:
        nodes = [self._parse_node()]
        rels: list = []
        while self._at_punct("-", "<-", "--", "-->", "<--"):
            rels.append(self._parse_rel())
            nodes.append(self._parse_node())
        return nodes, rels

    def _parse_node(self) -> _NodePat:
        self._expect_punct("(")
        var = self._next().value if self._at_name() else None
        labels: list = []
        while self._at_punct(":"):
            self._next()
            labels.append(self._expect_name())
        props = self._parse_prop_map() if self._at_punct("{") else {}
        self._expect_punct(")")
        return _NodePat(var=var, labels=labels, props=props)

    def _parse_rel(self) -> _RelPat:
        t = self._next()  # one of - <- -- --> <--
        arrow = t.value

        if arrow == "--":
            raise self._undirected()
        if arrow == "-->":
            return _RelPat(None, [], {}, "forward", 1, 1)
        if arrow == "<--":
            return _RelPat(None, [], {}, "backward", 1, 1)

        leading_back = arrow == "<-"
        var = types = props = None
        lo = hi = 1
        if self._at_punct("["):
            var, types, props, lo, hi = self._parse_rel_detail()
        else:
            var, types, props = None, [], {}

        # trailing arrow decides direction for the `-[...]-` forms
        if self._at_punct("->"):
            self._next()
            if leading_back:
                raise CypherError("a relationship cannot point both ways (`<-[...]->`)")
            direction = "forward"
        elif self._at_punct("-"):
            self._next()
            if not leading_back:
                raise self._undirected()
            direction = "backward"
        else:
            tok = self._peek()
            raise CypherError(
                f"unterminated relationship pattern at position {tok.pos} -- expected "
                f"'->' or '-' after the relationship"
            )
        return _RelPat(var=var, types=types, props=props, direction=direction, lo=lo, hi=hi)

    @staticmethod
    def _undirected() -> CypherError:
        return CypherError(
            "undirected relationships (`-[...]-`) are not supported: Hop.direction is "
            "'forward' or 'backward', and matching both would need two traversals. "
            "Pick a direction, or run one traversal per direction and merge"
        )

    def _parse_rel_detail(self) -> tuple:
        self._expect_punct("[")
        var = None
        if self._at_name() and not self._at_kw("IN"):
            var = self._next().value
        types: list = []
        while self._at_punct(":") or (types and self._at_punct("|")):
            self._next()
            if self._at_punct(":"):  # the `|:TYPE` spelling
                self._next()
            types.append(self._expect_name())
        lo = hi = 1
        if self._at_punct("*"):
            self._next()
            lo = hi = None
            if self._peek().kind == "number":
                lo = hi = self._next().value
            if self._at_punct(".."):
                self._next()
                hi = self._next().value if self._peek().kind == "number" else None
        props = self._parse_prop_map() if self._at_punct("{") else {}
        self._expect_punct("]")
        return var, types, props, lo, hi

    def _parse_prop_map(self) -> dict:
        self._expect_punct("{")
        props: dict = {}
        while not self._at_punct("}"):
            key = self._expect_name()
            self._expect_punct(":")
            props[key] = self._parse_literal()
            if self._at_punct(","):
                self._next()
            elif not self._at_punct("}"):
                t = self._peek()
                raise CypherError(f"expected ',' or '}}' at position {t.pos}")
        self._expect_punct("}")
        return props

    def _parse_literal(self) -> Any:
        t = self._peek()
        if t.kind in ("string", "number"):
            return self._next().value
        if self._at_punct("-") and self._peek(1).kind == "number":
            self._next()
            return -self._next().value
        if t.kind == "name":
            word = t.value.upper()
            if word == "TRUE":
                self._next()
                return True
            if word == "FALSE":
                self._next()
                return False
            if word == "NULL":
                self._next()
                return None
        raise CypherError(
            f"expected a literal value at position {t.pos}, got {self._describe(t)} -- "
            f"only literals are supported, not expressions or references to other nodes"
        )

    # -- WHERE expressions ----------------------------------------------

    def _parse_or(self):
        terms = [self._parse_and()]
        while self._at_kw("OR"):
            self._next()
            terms.append(self._parse_and())
        return terms[0] if len(terms) == 1 else _Or(terms)

    def _parse_and(self):
        terms = [self._parse_not()]
        while self._at_kw("AND"):
            self._next()
            terms.append(self._parse_not())
        return terms[0] if len(terms) == 1 else _And(terms)

    def _parse_not(self):
        if self._at_kw("NOT"):
            self._next()
            return _Not(self._parse_not())
        return self._parse_predicate()

    def _parse_predicate(self):
        if self._at_punct("("):
            self._next()
            expr = self._parse_or()
            self._expect_punct(")")
            return expr
        if self._at_kw("ALL"):
            return self._parse_all()
        if self._at_kw("ANY", "NONE", "SINGLE"):
            word = self._peek().value
            raise CypherError(
                f"{word}(...) is not supported -- Hop.via filters every edge a hop "
                f"traverses, which is `all(...)`. There is no per-hop existential form"
            )
        if self._at_kw("EXISTS"):
            raise CypherError(
                "exists(...) is not supported -- for 'this property is set', write "
                "`x.key IS NOT NULL`"
            )

        var, key = self._parse_property_ref()

        if self._at_kw("IS"):
            self._next()
            negated = False
            if self._at_kw("NOT"):
                self._next()
                negated = True
            self._expect_kw("NULL")
            return _IsNull(var=var, key=key, negated=negated)

        if self._at_kw("IN"):
            self._next()
            return _In(var=var, key=key, values=self._parse_list_literal())

        if self._at_punct("=", "<>", "<", "<=", ">", ">="):
            op = self._next().value
            return _Cmp(var=var, key=key, op=op, value=self._parse_literal())

        t = self._peek()
        raise CypherError(
            f"expected a comparison after {var}.{key} at position {t.pos} -- a bare "
            f"property is not a predicate; write `{var}.{key} IS NOT NULL` if you meant "
            f"'this property is set'"
        )

    def _parse_property_ref(self) -> tuple:
        t = self._peek()
        if t.kind != "name":
            raise CypherError(f"expected a property reference at position {t.pos}, "
                              f"got {self._describe(t)}")
        var = self._next().value
        if not self._at_punct("."):
            raise CypherError(
                f"expected `{var}.<property>` at position {t.pos} -- filters compare a "
                f"property, not a whole node"
            )
        self._next()
        return var, self._expect_name()

    def _parse_list_literal(self) -> list:
        self._expect_punct("[")
        values: list = []
        while not self._at_punct("]"):
            values.append(self._parse_literal())
            if self._at_punct(","):
                self._next()
            elif not self._at_punct("]"):
                t = self._peek()
                raise CypherError(f"expected ',' or ']' at position {t.pos}")
        self._expect_punct("]")
        return values

    def _parse_all(self) -> _AllRels:
        """all(r IN relationships(p) WHERE <expr>)"""
        self._expect_kw("ALL")
        self._expect_punct("(")
        rel_var = self._expect_name()
        self._expect_kw("IN")
        if not self._at_kw("RELATIONSHIPS"):
            t = self._peek()
            raise CypherError(
                f"only `all(r IN relationships(p) WHERE ...)` is supported at position "
                f"{t.pos} -- that is the form that maps onto Hop.via"
            )
        self._next()
        self._expect_punct("(")
        path_var = self._expect_name()
        self._expect_punct(")")
        self._expect_kw("WHERE")
        expr = self._parse_or()
        self._expect_punct(")")
        return _AllRels(rel_var=rel_var, path_var=path_var, expr=expr)


# ---------------------------------------------------------------------
# Expression -> hopai filter
# ---------------------------------------------------------------------

_NEGATION_HELP = (
    "Cypher and hopai disagree here, so this raises rather than translating into "
    "something that answers a different question: Cypher evaluates `x.k <> v` to NULL "
    "when `k` is missing and drops that row, while hopai's containment-based NOT keeps "
    "it. Write the NULL-safe form -- `x.k IS NULL OR x.k <> v` -- which maps exactly "
    "onto NOT({'k': v})"
)


def _key_missing(key: str):
    """`x.key IS NULL` -- the property is absent. Containment cannot ask
    this (it tests for a value, and there is none), so it goes through
    the callable escape hatch as a JSONB key-existence test.

    One documented divergence: a key explicitly set to JSON `null` is
    absent to Cypher and present to `?`, so such a row is excluded here
    and included by Cypher."""
    return lambda column: _sa_not(column.has_key(key))


def _key_present(key: str):
    return lambda column: column.has_key(key)


def _referenced_vars(expr: Any) -> set:
    if isinstance(expr, (_Or, _And)):
        return set().union(*(_referenced_vars(t) for t in expr.terms))
    if isinstance(expr, _Not):
        return _referenced_vars(expr.term)
    if isinstance(expr, (_Cmp, _In, _IsNull)):
        return {expr.var}
    if isinstance(expr, _AllRels):
        return {expr.path_var}
    raise AssertionError(f"unhandled expression node {type(expr).__name__}")


def _split_conjuncts(expr: Any) -> list:
    """Top-level ANDs become independent predicates, so that each can be
    attached to whichever hop binds its variable. Nothing below the top
    level is split -- an OR is indivisible."""
    if isinstance(expr, _And):
        out: list = []
        for t in expr.terms:
            out.extend(_split_conjuncts(t))
        return out
    return [expr]


def _null_safe_negation(expr: _Or):
    """Recognize `x.k IS NULL OR x.k <> v` as a unit and return
    NOT({k: v}).

    This is a special case on purpose, not a compositional rule: neither
    half translates on its own (see _NEGATION_HELP), and it is only their
    conjunction that means what hopai's NOT means."""
    if len(expr.terms) != 2:
        return None
    for a, b in (expr.terms, expr.terms[::-1]):
        if (isinstance(a, _IsNull) and not a.negated and isinstance(b, _Cmp)
                and b.op == "<>" and a.var == b.var and a.key == b.key):
            return NOT({b.key: b.value})
    return None


def _to_filter(expr: Any):
    """Compile one predicate (already validated to reference a single
    variable) into a hopai filter object."""
    if isinstance(expr, _Or):
        null_safe = _null_safe_negation(expr)
        if null_safe is not None:
            return null_safe
        return OR(*(_to_filter(t) for t in expr.terms))

    if isinstance(expr, _And):
        return _combine([_to_filter(t) for t in expr.terms], "predicate")

    if isinstance(expr, _Not):
        raise CypherError(f"`NOT` on a comparison is not supported. {_NEGATION_HELP}")

    if isinstance(expr, _AllRels):
        raise CypherError(
            "all(...) constrains the edges of a hop, so it can only be ANDed with the "
            "rest of a WHERE clause -- it cannot appear inside an OR or a NOT"
        )

    if isinstance(expr, _In):
        return {expr.key: list(expr.values)}

    if isinstance(expr, _IsNull):
        return _key_present(expr.key) if expr.negated else _key_missing(expr.key)

    if isinstance(expr, _Cmp):
        if expr.op == "=":
            return {expr.key: expr.value}
        if expr.op == "<>":
            raise CypherError(f"`<>` is not supported on its own. {_NEGATION_HELP}")
        if not isinstance(expr.value, (int, float)) or isinstance(expr.value, bool):
            raise CypherError(
                f"`{expr.op}` compares numerically, so it needs a number -- got "
                f"{expr.value!r} for {expr.var}.{expr.key}"
            )
        return {">": GT, ">=": GTE, "<": LT, "<=": LTE}[expr.op](expr.key, expr.value)

    raise AssertionError(f"unhandled expression node {type(expr).__name__}")


def _combine(filters: list, what: str):
    """Fold several filters for one variable into one.

    Plain dicts merge into a single containment test rather than an AND
    of several -- `(a:person {active: true})` should compile to one `@>`,
    not two. Conflicting values for one key are unsatisfiable, so they
    raise instead of quietly matching nothing."""
    if not filters:
        return None
    merged: dict = {}
    others: list = []
    for f in filters:
        if isinstance(f, dict):
            for k, v in f.items():
                if k in merged and merged[k] != v:
                    raise CypherError(
                        f"{what} requires {k}={merged[k]!r} and {k}={v!r} at once, which "
                        f"nothing can match"
                    )
                merged[k] = v
        else:
            others.append(f)
    parts = ([merged] if merged else []) + others
    return parts[0] if len(parts) == 1 else AND(*parts)


# ---------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------

@dataclass
class _Options:
    node_label_key: Optional[str] = "type"
    edge_type_key: Optional[str] = "kind"
    max_var_length: Optional[int] = None


class _Translator:
    def __init__(self, opts: _Options):
        self.o = opts
        self.nodes: list = []          # _NodePat per chain position
        self.rels: list = []           # _RelPat per hop
        self.node_filters: dict = {}   # chain index -> [filter, ...]
        self.rel_filters: dict = {}    # hop index -> [filter, ...]
        self.node_index: dict = {}     # variable -> chain index
        self.rel_index: dict = {}      # variable -> hop index
        self.path_vars: dict = {}      # path variable -> [hop index, ...]
        self.optional_hop: Optional[int] = None

    # -- chain assembly -------------------------------------------------

    def _add_node(self, pat: _NodePat) -> int:
        idx = len(self.nodes)
        self.nodes.append(pat)
        self.node_filters[idx] = []
        if pat.var is not None:
            self.node_index[pat.var] = idx
        self._absorb_node(idx, pat)
        return idx

    def _absorb_node(self, idx: int, pat: _NodePat) -> None:
        """Fold a node pattern's labels and inline properties into that
        position's filters. Called again when a later MATCH re-references
        an already-bound variable."""
        if pat.labels:
            if self.o.node_label_key is None:
                pass  # labels explicitly ignored
            elif len(pat.labels) > 1:
                raise CypherError(
                    f"node {pat.var or '(anonymous)'} has multiple labels "
                    f"({':'.join(pat.labels)}) -- they would map onto the single property "
                    f"{self.o.node_label_key!r}, and a property has one value"
                )
            else:
                self.node_filters[idx].append({self.o.node_label_key: pat.labels[0]})
        if pat.props:
            self.node_filters[idx].append(dict(pat.props))

    def _add_rel(self, pat: _RelPat, path_var: Optional[str]) -> int:
        hop = len(self.rels)
        self.rels.append(pat)
        self.rel_filters[hop] = []
        if pat.var is not None:
            if pat.var in self.rel_index or pat.var in self.node_index:
                raise CypherError(f"variable {pat.var!r} is already bound")
            self.rel_index[pat.var] = hop
        if pat.types and self.o.edge_type_key is not None:
            value = pat.types[0] if len(pat.types) == 1 else list(pat.types)
            self.rel_filters[hop].append({self.o.edge_type_key: value})
        if pat.props:
            self.rel_filters[hop].append(dict(pat.props))
        if path_var is not None:
            self.path_vars.setdefault(path_var, []).append(hop)
        return hop

    def _build_chain(self, clauses: list) -> None:
        for ci, clause in enumerate(clauses):
            if clause.extra_patterns:
                raise CypherError(
                    "comma-separated patterns describe disjoint matches joined on shared "
                    "variables, which hopai does not support -- it traverses one linear "
                    "chain. (Before a CREATE or MERGE they are allowed, since there each "
                    "is just a lookup.)"
                )
            if clause.optional:
                if ci != len(clauses) - 1:
                    raise CypherError(
                        "OPTIONAL MATCH must be the last clause -- hopai supports "
                        "optional=True on the last hop only, because every hop after it "
                        "would have to tolerate a missing anchor node"
                    )
                if len(clause.rels) != 1:
                    raise CypherError(
                        f"OPTIONAL MATCH must extend the chain by exactly one "
                        f"relationship, got {len(clause.rels)}"
                    )

            nodes, rels = clause.nodes, clause.rels
            if not self.nodes:
                first_idx = self._add_node(nodes[0])
            else:
                first = nodes[0]
                if first.var is None or first.var not in self.node_index:
                    raise CypherError(
                        "every MATCH after the first must continue the chain from a "
                        "bound variable -- separate patterns joined on shared variables "
                        "are disjoint matching, which hopai does not support"
                    )
                first_idx = self.node_index[first.var]
                if first_idx != len(self.nodes) - 1:
                    raise CypherError(
                        f"this MATCH continues from {first.var!r}, which is not the last "
                        f"node of the chain so far -- hopai traverses one linear chain, "
                        f"so each MATCH must extend it from its end"
                    )
                self._absorb_node(first_idx, first)

            # strict=True: the parser always yields one more node than
            # relationship, so a mismatch here is a parser bug, not input.
            for rel, node in zip(rels, nodes[1:], strict=True):
                hop = self._add_rel(rel, clause.path_var)
                if node.var is not None and node.var in self.node_index:
                    raise CypherError(
                        f"variable {node.var!r} is re-used later in the pattern, which "
                        f"constrains the walk to return to that node -- hopai has no way "
                        f"to express that"
                    )
                self._add_node(node)
                if clause.optional:
                    self.optional_hop = hop

            if clause.path_var is not None and not rels:
                raise CypherError(f"path variable {clause.path_var!r} binds no relationships")

    # -- WHERE attachment ------------------------------------------------

    def _attach_where(self, clauses: list) -> None:
        for clause in clauses:
            if clause.where is None:
                continue
            for conj in _split_conjuncts(clause.where):
                if isinstance(conj, _AllRels):
                    self._attach_all_rels(conj)
                    continue
                variables = _referenced_vars(conj)
                if len(variables) > 1:
                    raise CypherError(
                        f"a predicate over several variables ({', '.join(sorted(variables))}) "
                        f"cannot be translated: each filter binds one node or one hop's "
                        f"edges, so only AND may span variables -- split it, or drop the "
                        f"cross-variable OR"
                    )
                var = next(iter(variables))
                self._attach_one(clause, var, _to_filter(conj))

    def _attach_all_rels(self, expr: _AllRels) -> None:
        if expr.path_var not in self.path_vars:
            raise CypherError(f"unknown path variable {expr.path_var!r}")
        variables = _referenced_vars(expr.expr)
        if variables != {expr.rel_var}:
            raise CypherError(
                f"all(...) must constrain only {expr.rel_var!r}, the relationship it "
                f"binds -- got {', '.join(sorted(variables))}"
            )
        filt = _to_filter(expr.expr)
        # relationships(p) is every edge on the path, so the predicate
        # applies to every hop that path spans -- which is precisely
        # Hop.via's meaning, hop by hop.
        for hop in self.path_vars[expr.path_var]:
            self.rel_filters[hop].append(filt)

    def _attach_one(self, clause: _MatchClause, var: str, filt: Any) -> None:
        if var in self.node_index:
            idx = self.node_index[var]
            if clause.optional and idx != len(self.nodes) - 1:
                raise CypherError(
                    f"an OPTIONAL MATCH's WHERE may only constrain what that hop reaches, "
                    f"not the already-matched {var!r} -- in Cypher such a predicate filters "
                    f"the optional extension rather than the row, which hopai cannot "
                    f"express. Move it to the earlier MATCH's WHERE if you meant to filter "
                    f"{var!r} itself"
                )
            self.node_filters[idx].append(filt)
        elif var in self.rel_index:
            self.rel_filters[self.rel_index[var]].append(filt)
        elif var in self.path_vars:
            raise CypherError(
                f"a path variable ({var!r}) can only be used as "
                f"`all({var}... IN relationships({var}) WHERE ...)`"
            )
        else:
            raise CypherError(f"unknown variable {var!r} in WHERE")

    # -- emit -------------------------------------------------------------

    def _hop_count(self, rel: _RelPat, hop: int):
        lo, hi = rel.lo, rel.hi
        for bound in (lo, hi):
            if bound is not None and not isinstance(bound, int):
                raise CypherError(f"hop {hop}: a variable-length bound must be a whole "
                                  f"number, got {bound!r}")
        if lo == 0 or hi == 0:
            raise CypherError(
                f"hop {hop}: a zero-length pattern (`*0..`) matches a node to itself, and "
                f"a Hop always traverses at least one edge -- use `*1..N`"
            )
        if hi is None:
            if self.o.max_var_length is None:
                raise CypherError(
                    f"hop {hop}: an unbounded `*` has no equivalent -- the recursive walk "
                    f"needs an upper bound to terminate. Write `*{lo or 1}..N`, or pass "
                    f"max_var_length=N to cap every unbounded pattern in the query"
                )
            hi = self.o.max_var_length
        lo = 1 if lo is None else lo
        if lo > hi:
            raise CypherError(f"hop {hop}: `*{lo}..{hi}` has a minimum above its maximum")
        return lo if lo == hi else (lo, hi)

    def emit(self) -> tuple:
        start = Start(
            where=_combine(self.node_filters[0], f"node {self.nodes[0].var or '(anonymous)'}"),
            label=self.nodes[0].var,
        )
        hops: list = []
        for hop, rel in enumerate(self.rels):
            node = self.nodes[hop + 1]
            name = node.var or "(anonymous)"
            hops.append(
                Hop(
                    where=_combine(self.node_filters[hop + 1], f"node {name}"),
                    via=_combine(self.rel_filters[hop], f"relationship at hop {hop}"),
                    hops=self._hop_count(rel, hop),
                    direction=rel.direction,
                    optional=(hop == self.optional_hop),
                    label=node.var,
                )
            )
        return start, hops

    # -- aggregating RETURN ----------------------------------------------

    def emit_aggregates(self, ret: _ReturnClause) -> dict:
        """The {name: aggregate} dict a _ReturnClause means, or a
        CypherError naming the rewrite when its Cypher meaning is not
        exactly what Graph.aggregate() computes -- see AGGREGATION in
        the module docstring for the rules being enforced here."""
        last_var = self.nodes[-1].var
        if self.optional_hop is not None:
            raise CypherError(
                "an OPTIONAL MATCH cannot feed an aggregation: the aggregate runs over "
                "the nodes the hop matched, and a chain the hop did not extend "
                "contributes nothing either way -- the number is identical without "
                "OPTIONAL, so drop it"
            )
        if ret.distinct_var is not None and ret.distinct_var != last_var:
            raise CypherError(
                f"WITH DISTINCT {ret.distinct_var} must name the last node of the chain "
                f"({last_var or 'which is anonymous -- give it a variable'}) -- that is "
                f"the only set hopai can aggregate"
            )
        # With no hops every node is its own (only) result row, and with
        # WITH DISTINCT the rows have been deduplicated to exactly the
        # matched nodes -- either way bare aggregates lose their per-path
        # multiplicity and become exact.
        per_node = ret.distinct_var is not None or not self.rels
        aggregates: dict = {}
        for item in ret.items:
            name = item.alias or (item.fn if item.key is None else f"{item.fn}_{item.key}")
            if name in aggregates:
                raise CypherError(
                    f"two aggregates both land on the result key {name!r} -- give one an "
                    f"alias: `{item.fn}(...) AS other_name`"
                )
            aggregates[name] = self._translate_aggregate(item, last_var, per_node)
        return aggregates

    def _translate_aggregate(self, item: _AggItem, last_var: Optional[str], per_node: bool):
        from .aggregates import Avg, Count, Max, Min, Sum

        def refuse_bare() -> CypherError:
            target = f"{item.var}.{item.key}" if item.key else (item.var or "*")
            return CypherError(
                f"bare {item.fn}({target}) aggregates one row per PATH -- a node two "
                f"paths reach counts twice -- and hopai does not track path multiplicity "
                f"across hops. Write {item.fn}(DISTINCT ...) to aggregate distinct "
                f"values, or `WITH DISTINCT {last_var or '<var>'} RETURN "
                f"{item.fn}(...)` to aggregate each matched node once"
            )

        if item.var is None:                    # count(*) et al.
            if item.fn != "count":
                raise CypherError(
                    f"{item.fn}(*) aggregates nothing in particular -- write "
                    f"{item.fn}(<var>.<property>)"
                )
            if not per_node:
                raise refuse_bare()
            return Count()

        if item.var in self.rel_index or item.var in self.path_vars:
            raise CypherError(
                f"aggregation over relationships ({item.fn}({item.var}...)) is not "
                f"supported yet -- aggregate the node at the end of the hop instead"
            )
        if item.var not in self.node_index:
            raise CypherError(f"unknown variable {item.var!r} in RETURN")
        if self.node_index[item.var] != len(self.nodes) - 1:
            raise CypherError(
                f"only the LAST node of the chain can be aggregated, and {item.var!r} is "
                f"not it: hopai does not track which mid-chain nodes lie on a complete "
                f"path, so {item.fn}({item.var}...) here would count nodes Cypher would "
                f"not. Reverse the pattern so {item.var} comes last, or run a separate "
                f"traversal"
            )

        if item.fn == "count":
            if item.key is None:
                # count(DISTINCT b) IS the distinct-node count -- exact
                # even with paths involved, unlike every other bare form.
                if item.distinct or per_node:
                    return Count()
                raise refuse_bare()
            if item.distinct:
                return Count(item.key, distinct=True)
            if per_node:
                return Count(item.key)
            raise refuse_bare()

        if item.key is None:
            raise CypherError(
                f"{item.fn}({item.var}) aggregates a whole node -- write "
                f"{item.fn}({item.var}.<property>)"
            )
        if item.fn in ("sum", "avg"):
            cls = Sum if item.fn == "sum" else Avg
            if item.distinct:
                return cls(item.key, distinct=True)
            if per_node:
                return cls(item.key)
            raise refuse_bare()
        # min/max: an extremum is immune to both duplication and
        # deduplication, so bare and DISTINCT spellings are both exact.
        return (Min if item.fn == "min" else Max)(item.key)


# ---------------------------------------------------------------------
# Writes: CREATE and MERGE
# ---------------------------------------------------------------------

class _WriteTranslator:
    """Turn CREATE/MERGE clauses into an ordered list of ingestion
    operations -- the same add_nodes/merge_nodes/add_edges the Python API
    calls, never a second write path.

    Variables are the whole problem. An edge needs its endpoints' ids,
    and those come from three places: a node this query just created
    (the insert returns them), a node this query merged (the upsert
    returns them), or a node a MATCH found (looked up by its properties).
    Every operation therefore binds its variables, and edges are emitted
    referring to variable names, resolved at execution."""

    def __init__(self, opts: _Options):
        self.o = opts
        self.operations: list = []
        self.bound: set = set()

    # -- shared with the read path --------------------------------------

    def _node_properties(self, pat: _NodePat) -> dict:
        properties = {}
        if pat.labels:
            if len(pat.labels) > 1:
                raise CypherError(
                    f"node {pat.var or '(anonymous)'} has multiple labels "
                    f"({':'.join(pat.labels)}) -- they would map onto the single property "
                    f"{self.o.node_label_key!r}, and a property has one value"
                )
            if self.o.node_label_key is not None:
                key = self.o.node_label_key
                if key in pat.props and pat.props[key] != pat.labels[0]:
                    raise CypherError(
                        f"node {pat.var or '(anonymous)'} sets {key}={pat.props[key]!r} and "
                        f"also carries the label :{pat.labels[0]}, which maps onto the same "
                        f"property. Drop one, or pass node_label_key=None"
                    )
                properties[key] = pat.labels[0]
        properties.update(pat.props)
        return properties

    def _rel_properties(self, pat: _RelPat) -> dict:
        properties = {}
        if pat.types:
            if len(pat.types) > 1:
                raise CypherError(
                    "a relationship being written has one type, not several -- "
                    f"`[:{'|'.join(pat.types)}]` is a pattern to match, not to create"
                )
            if self.o.edge_type_key is not None:
                properties[self.o.edge_type_key] = pat.types[0]
        properties.update(pat.props)
        return properties

    @staticmethod
    def _check_writable_rel(pat: _RelPat) -> None:
        if (pat.lo, pat.hi) != (1, 1):
            raise CypherError(
                "a variable-length relationship cannot be written -- `*` says how far to "
                "walk, and there is no such thing as creating half an edge"
            )

    # -- clauses ---------------------------------------------------------

    def _bind_matched(self, clause: _MatchClause) -> None:
        """A MATCH before a write exists only to name existing nodes, so
        each of its patterns must be a lone node with something to find it
        by. Traversal in a write query would be a different feature."""
        for nodes, rels in [(clause.nodes, clause.rels), *clause.extra_patterns]:
            if rels:
                raise CypherError(
                    "a MATCH that precedes CREATE or MERGE may only bind single nodes, "
                    "not traverse -- write `MATCH (a {...}), (b {...}) CREATE (a)-[:x]->(b)`"
                )
            pat = nodes[0]
            if pat.var is None:
                raise CypherError("a MATCH before a write must name what it binds, e.g. (a {...})")
            properties = self._node_properties(pat)
            if clause.where is not None:
                raise CypherError(
                    "WHERE is not supported before a write -- identify the node with a "
                    "property map, `MATCH (a {email: '...'})`"
                )
            if not properties:
                raise CypherError(
                    f"MATCH ({pat.var}) has nothing to find the node by; give it properties"
                )
            self.operations.append({"op": "match", "var": pat.var, "where": properties})
            self.bound.add(pat.var)

    def _create(self, clause: _WriteClause) -> None:
        rows, variables, edges = [], [], []
        for nodes, rels in clause.patterns:
            for pat in nodes:
                created = self._new_node(pat)
                if created is not None:
                    rows.append(created)
                    variables.append(pat.var)
            for index, rel in enumerate(rels):
                self._check_writable_rel(rel)
                left, right = nodes[index], nodes[index + 1]
                if rel.direction == "backward":
                    left, right = right, left      # `<-` reverses the edge
                edges.append({"start_var": self._require_var(left),
                              "end_var": self._require_var(right),
                              "properties": self._rel_properties(rel)})
        if rows:
            self.operations.append({"op": "create_nodes", "rows": rows, "vars": variables})
        if edges:
            self.operations.append({"op": "create_edges", "rows": edges})

    @staticmethod
    def _require_var(pat: _NodePat) -> str:
        if pat.var is None:
            raise CypherError(
                "a node an edge connects has to be named, so the edge can refer to it -- "
                "write `CREATE (a {...})-[:x]->(b {...})`, not `({...})-[:x]->({...})`"
            )
        return pat.var

    def _new_node(self, pat: _NodePat) -> Optional[dict]:
        """The node's properties, or None when this is a re-mention of an
        already-bound variable such as `MATCH (a {...}) CREATE (a)-[:x]->(b {...})`."""
        if pat.var is not None and pat.var in self.bound:
            if pat.labels or pat.props:
                raise CypherError(
                    f"{pat.var!r} is already bound, so `({pat.var} {{...}})` would be "
                    f"redefining it -- refer to it as `({pat.var})`"
                )
            return None
        if pat.var is not None:
            self.bound.add(pat.var)
        return self._node_properties(pat)

    def _merge(self, clause: _WriteClause) -> None:
        if len(clause.patterns) > 1:
            raise CypherError("MERGE takes one pattern at a time")
        nodes, rels = clause.patterns[0]

        if not rels:
            self._merge_node(nodes[0], clause)
            return

        # A relationship MERGE is only unambiguous when both ends already
        # exist. Cypher's own `MERGE (a {..})-[:x]->(b {..})` matches the
        # WHOLE path and creates the whole thing when it does not match --
        # duplicating nodes that already exist on their own. That is a
        # well-known footgun, and quietly reproducing it here would be
        # worse than refusing.
        if len(rels) > 1:
            raise CypherError("MERGE writes one relationship at a time")
        for pat in (nodes[0], nodes[1]):
            if pat.var is None or pat.var not in self.bound:
                raise CypherError(
                    "MERGE on a relationship needs both endpoints already bound, because "
                    "matching a whole path and creating it when absent would duplicate "
                    "nodes that already exist -- MATCH or MERGE the nodes first, then "
                    "`MERGE (a)-[:x]->(b)`"
                )
        rel = rels[0]
        self._check_writable_rel(rel)
        if clause.on_create or clause.on_match:
            raise CypherError("ON CREATE SET / ON MATCH SET are supported on node MERGE only")
        properties = self._rel_properties(rel)
        if not properties:
            raise CypherError(
                "MERGE on a relationship needs something to match it by -- give it a type, "
                "`MERGE (a)-[:knows]->(b)`"
            )
        left, right = nodes[0], nodes[1]
        if rel.direction == "backward":
            left, right = right, left
        self.operations.append({
            "op": "merge_edges",
            "rows": [{"start_var": left.var, "end_var": right.var, "properties": properties}],
            "on": sorted(properties),
        })

    def _merge_node(self, pat: _NodePat, clause: _WriteClause) -> None:
        if pat.var is not None and pat.var in self.bound:
            raise CypherError(f"{pat.var!r} is already bound; MERGE would redefine it")
        properties = self._node_properties(pat)
        if not properties:
            raise CypherError(
                "MERGE needs properties to match on -- `MERGE (a:person {email: '...'})`"
            )
        for label, group in (("ON CREATE SET", clause.on_create),
                             ("ON MATCH SET", clause.on_match)):
            unknown = set(group) - ({pat.var} if pat.var else set())
            if unknown:
                raise CypherError(
                    f"{label} refers to {sorted(unknown)}, which this MERGE does not bind")
        self.operations.append({
            "op": "merge_nodes",
            "rows": [properties],
            # Cypher matches on every property in the pattern, so those are
            # the conflict keys -- and a unique index must cover exactly
            # them. Properties that should not take part in matching belong
            # in ON CREATE SET.
            "on": sorted(properties),
            "on_create": clause.on_create.get(pat.var, {}) if pat.var else {},
            "on_match": clause.on_match.get(pat.var, {}) if pat.var else {},
            "vars": [pat.var],
        })
        if pat.var is not None:
            self.bound.add(pat.var)

    def translate(self, clauses: list) -> list:
        for clause in clauses:
            if isinstance(clause, _MatchClause):
                if clause.optional:
                    raise CypherError("OPTIONAL MATCH has no meaning before a write")
                self._bind_matched(clause)
            elif clause.kind == "create":
                self._create(clause)
            else:
                self._merge(clause)
        return self.operations


def cypher_to_operations(
    query: str,
    *,
    node_label_key: Optional[str] = "type",
    edge_type_key: Optional[str] = "kind",
    schema=None,
) -> list:
    """Translate a Cypher CREATE/MERGE into ordered ingestion operations.

    Returns plain dicts, so the plan can be inspected, logged or shown to
    a caller before it runs -- `graph.write_cypher()` is what executes
    them. Raises CypherError for a read-only query."""
    opts = _Options(node_label_key=node_label_key, edge_type_key=edge_type_key)
    clauses = _Parser(_tokenize(query)).parse()
    if not any(isinstance(c, _WriteClause) for c in clauses):
        raise CypherError(
            "this query only reads -- run it with graph.traverse_cypher() (or "
            "graph.cypher(), which picks for you)"
        )
    if any(isinstance(c, _ReturnClause) for c in clauses):
        raise CypherError(
            "RETURN with an aggregate after a write is not supported -- a write "
            "produces an IngestResult, not a number. Run the aggregation as its own "
            "MATCH query afterwards"
        )
    operations = _WriteTranslator(opts).translate(clauses)
    if schema is not None:
        from .schema import validate_operations
        validate_operations(schema, operations, node_label_key, edge_type_key)
    return operations


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def cypher_to_traversal(
    query: str,
    *,
    node_label_key: Optional[str] = "type",
    edge_type_key: Optional[str] = "kind",
    max_var_length: Optional[int] = None,
    schema=None,
) -> tuple:
    """Translate a Cypher MATCH into `(Start, [Hop, ...])` -- the same
    pair `spec_to_traversal()` returns, ready for `Graph.traverse()`.
    Exposed separately from traverse_cypher() so you can inspect or
    adjust the translation before running it.

    node_label_key:  property a node label compiles to (`(a:person)` ->
                     {"type": "person"}). None ignores labels entirely,
                     which is what you want for Cypher written against a
                     schema where the label carries no information --
                     `(a:Node {type: 'leaf'})`.
    edge_type_key:   property a relationship type compiles to
                     (`[:friend]` -> {"kind": "friend"}). None ignores
                     relationship types.
    max_var_length:  upper bound applied to unbounded patterns (`*`,
                     `*2..`). Left None, those raise rather than silently
                     acquiring a depth limit you did not choose.

    Raises CypherError for anything that does not translate -- including
    constructs that would translate into different semantics than Cypher
    gives them. See the module docstring for the full list.
    """
    opts = _Options(node_label_key=node_label_key, edge_type_key=edge_type_key,
                    max_var_length=max_var_length)
    translator, ret = _translate_read(query, opts)
    if ret is not None:
        raise CypherError(
            "this query aggregates -- run it with graph.aggregate_cypher() (or "
            "graph.cypher(), which picks for you)"
        )
    start, hops = translator.emit()
    if schema is not None:
        from .schema import validate_traversal
        validate_traversal(schema, start, hops, node_label_key, edge_type_key)
    return start, hops


def _translate_read(query: str, opts: _Options) -> tuple:
    """Parse and translate a read query's MATCH chain, returning the
    loaded translator and the aggregating _ReturnClause if there is one
    -- the shared front half of cypher_to_traversal and
    cypher_to_aggregation."""
    clauses = _Parser(_tokenize(query)).parse()
    if any(isinstance(c, _WriteClause) for c in clauses):
        raise CypherError(
            "this query writes -- run it with graph.write_cypher() (or graph.cypher(), "
            "which picks for you)"
        )
    matches = [c for c in clauses if isinstance(c, _MatchClause)]
    translator = _Translator(opts)
    translator._build_chain(matches)
    translator._attach_where(matches)
    return translator, next((c for c in clauses if isinstance(c, _ReturnClause)), None)


def cypher_to_aggregation(
    query: str,
    *,
    node_label_key: Optional[str] = "type",
    edge_type_key: Optional[str] = "kind",
    max_var_length: Optional[int] = None,
    schema=None,
) -> tuple:
    """Translate an aggregating Cypher MATCH into
    `(Start, [Hop, ...], {name: aggregate})` -- the exact arguments of
    `Graph.aggregate()`. Exposed separately from aggregate_cypher() so
    you can inspect or adjust the translation before running it.

    Takes the same keyword options as cypher_to_traversal(), refuses the
    same things, and additionally refuses aggregation spellings whose
    Cypher meaning differs from what hopai computes -- see AGGREGATION
    in the module docstring for the rules and each rewrite."""
    opts = _Options(node_label_key=node_label_key, edge_type_key=edge_type_key,
                    max_var_length=max_var_length)
    translator, ret = _translate_read(query, opts)
    if ret is None:
        raise CypherError(
            "this query has no aggregating RETURN -- run it with "
            "graph.traverse_cypher() (or graph.cypher(), which picks for you)"
        )
    aggregates = translator.emit_aggregates(ret)
    start, hops = translator.emit()
    if schema is not None:
        from .schema import validate_traversal
        validate_traversal(schema, start, hops, node_label_key, edge_type_key)
    return start, hops, aggregates


def resolve_strict(graph: Graph, options: dict) -> dict:
    """strict_schema=True on a graph-level call becomes schema=<the
    graph's declared schema> on the translator -- refusing, with the fix
    named, when there is nothing to be strict against."""
    if options.pop("strict_schema", False):
        if graph.schema is None:
            raise CypherError(
                "strict_schema=True needs a schema and none is defined for this Graph "
                "-- call define_schema(...) first"
            )
        options["schema"] = graph.schema
    return options


def traverse_cypher(graph: Graph, query: str, **options) -> Subgraph:
    """Run a traversal written in Cypher.

    Returns a Subgraph, not a dict -- unlike traverse_json(), whose
    JSON-in/JSON-out shape is the whole point, this one is called from
    Python and its caller usually wants `.nodes` / `.to_networkx()`. Call
    `.to_dict()` if you need it serializable.

    Accepts the same keyword options as cypher_to_traversal().
    """
    start, hops = cypher_to_traversal(query, **resolve_strict(graph, dict(options)))
    return graph.traverse(start, *hops)


def aggregate_cypher(graph: Graph, query: str, **options) -> dict:
    """Run an aggregation written in Cypher.

        aggregate_cypher(graph, '''
            MATCH (a:person)-[:friend*1..4]->(b)
            RETURN count(DISTINCT b), avg(DISTINCT b.age) AS mean_age
        ''')
        # -> {"count": 42, "mean_age": 31.5}

    Result keys are the `AS` aliases where given, else `fn` or
    `fn_property` (`count`, `avg_age`). Accepts the same keyword options
    as cypher_to_traversal().
    """
    start, hops, aggregates = cypher_to_aggregation(query, **resolve_strict(graph, dict(options)))
    return graph.aggregate(start, *hops, aggregates=aggregates)
