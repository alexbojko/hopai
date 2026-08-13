"""
hopai.cypher

Cypher as an input syntax. Translates a subset of Cypher's MATCH into the
same `(Start, [Hop, ...])` pair `hopai.json_api.spec_to_traversal()`
produces -- a third way to describe a traversal, compiling through the
identical `resolve()` path, with no query logic of its own.

    from hopai import traverse_cypher
    traverse_cypher(graph, '''
        MATCH (a:person)-[:friend*1..4]->(b {active: true})
        WHERE b.age > 18
        RETURN b
    ''')

THE RULE THIS MODULE IS BUILT ON: refuse, don't approximate. Cypher can
say things hopai cannot, and a few things it says in a shape hopai
supports but with *different semantics*. Every one of those raises
CypherError naming the rewrite, rather than returning a traversal that
quietly answers a different question. That is the same choice
`filters.resolve()` makes when it rejects a bare list.

WHAT DOES NOT TRANSLATE, and why:

  RETURN has no target.       hopai returns a subgraph -- every node and
                              edge on a matching chain. There is no
                              projection, aggregation, ORDER BY or LIMIT
                              to map onto, so RETURN is parsed and
                              ignored. Aggregates (`RETURN count(a)`)
                              raise rather than being silently dropped,
                              since the caller clearly wanted a number.
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
from dataclasses import dataclass
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


_AGGREGATES = {"count", "sum", "avg", "min", "max", "collect", "stdev", "percentiledisc",
               "percentilecont"}
_UNSUPPORTED_CLAUSES = {
    "WITH": "WITH (hopai runs one traversal, with no intermediate projection)",
    "UNWIND": "UNWIND",
    "CREATE": "writes (hopai is read-only)",
    "MERGE": "writes (hopai is read-only)",
    "SET": "writes (hopai is read-only)",
    "DELETE": "writes (hopai is read-only)",
    "REMOVE": "writes (hopai is read-only)",
    "DETACH": "writes (hopai is read-only)",
    "CALL": "CALL",
    "FOREACH": "FOREACH",
    "UNION": "UNION",
    "ORDER": "ORDER BY (hopai returns an unordered subgraph)",
    "SKIP": "SKIP",
    "LIMIT": "LIMIT (hopai returns the whole matching subgraph)",
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
            elif self._at_kw("RETURN"):
                self._next()
                self._parse_return()
            elif self._at_name():
                word = self._peek().value.upper()
                if word in _UNSUPPORTED_CLAUSES:
                    raise CypherError(
                        f"{_UNSUPPORTED_CLAUSES[word]} is not supported -- hopai returns "
                        f"a subgraph of every node and edge on a matching chain, so there "
                        f"is nothing for this clause to act on"
                    )
                raise CypherError(f"unexpected {self._peek().value!r} at position {self._peek().pos}")
            else:
                t = self._peek()
                raise CypherError(f"unexpected {self._describe(t)} at position {t.pos}")
        if not clauses:
            raise CypherError("query has no MATCH clause")
        return clauses

    def _parse_return(self) -> None:
        """RETURN is parsed only to reject what it would silently lose.
        The result of a traversal is always the full matching subgraph,
        so a projection has nothing to project."""
        while self._peek().kind != "eof":
            t = self._peek()
            if t.kind == "name":
                word = t.value.upper()
                if word in _UNSUPPORTED_CLAUSES:
                    raise CypherError(
                        f"{_UNSUPPORTED_CLAUSES[word]} is not supported -- hopai returns "
                        f"a subgraph of every node and edge on a matching chain"
                    )
                if t.value.lower() in _AGGREGATES and self._peek(1).kind == "punct" \
                        and self._peek(1).value == "(":
                    raise CypherError(
                        f"aggregation ({t.value}(...)) is not supported: a traversal returns "
                        f"the matching subgraph, not a scalar. Run the traversal and count "
                        f"the result -- len(result.nodes), or count the ids you care about"
                    )
            self._next()

    # -- MATCH ----------------------------------------------------------

    def _parse_match(self, optional: bool) -> _MatchClause:
        path_var = None
        if self._at_name() and self._peek(1).kind == "punct" and self._peek(1).value == "=":
            path_var = self._next().value
            self._next()  # '='
        nodes, rels = self._parse_pattern()
        if self._at_punct(","):
            raise CypherError(
                "comma-separated patterns describe disjoint matches joined on shared "
                "variables, which hopai does not support -- it traverses one linear chain"
            )
        where = None
        if self._at_kw("WHERE"):
            self._next()
            where = self._parse_or()
        return _MatchClause(nodes=nodes, rels=rels, where=where, optional=optional,
                            path_var=path_var)

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


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def cypher_to_traversal(
    query: str,
    *,
    node_label_key: Optional[str] = "type",
    edge_type_key: Optional[str] = "kind",
    max_var_length: Optional[int] = None,
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
    clauses = _Parser(_tokenize(query)).parse()
    translator = _Translator(opts)
    translator._build_chain(clauses)
    translator._attach_where(clauses)
    return translator.emit()


def traverse_cypher(graph: Graph, query: str, **options) -> Subgraph:
    """Run a traversal written in Cypher.

    Returns a Subgraph, not a dict -- unlike traverse_json(), whose
    JSON-in/JSON-out shape is the whole point, this one is called from
    Python and its caller usually wants `.nodes` / `.to_networkx()`. Call
    `.to_dict()` if you need it serializable.

    Accepts the same keyword options as cypher_to_traversal().
    """
    start, hops = cypher_to_traversal(query, **options)
    return graph.traverse(start, *hops)
