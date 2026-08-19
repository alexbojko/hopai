"""
hopai.json_api

A JSON-in / JSON-out interface, for callers that can't or shouldn't write
Python: an LLM tool call, an HTTP endpoint, a config file. It is what
hopai/mcp.py serves over MCP, and what an agent framework wires up
directly. Everything here is a thin translation layer over Graph.traverse()
and hopai.filters.parse_filter() -- there is no separate logic to
trust here, just a different way to describe the same traversal.

Spec shape:

    {
      "start": {"where": {"type": "person"}},
      "hops": [
        {"where": {"active": true}, "via": {"kind": "friend"},
         "hops": [1, 4], "direction": "forward"},
        {"where": {"type": "company"}, "hops": 3, "optional": true}
      ]
    }

`where` and `via` accept the same JSON filter grammar as
hopai.filters.parse_filter(): plain objects for equality/AND, plus
{"and": [...]}, {"or": [...]}, {"not": ...}, {"gt": [key, value]},
{"gte": [...]}, {"lt": [...]}, {"lte": [...]}, {"between": [key, lo, hi]}.
`via` additionally accepts a bare string -- the STORED_IN shorthand,
e.g. "via": "friend" in place of "via": {"kind": "friend"} -- compiled
by filters.resolve_via() to the SQL a declared edge type's index can
serve (Graph.define_edge_type()); `where` has no equivalent, since a
node has no single universal "kind"-like property this could name.

`hops` accepts either an integer (exact hop count) or a two-element
array [min, max].

`start` accepts `near` (a {"field", "text"} similarity spec or a list
of them -- hopai.vectors.parse_near), `keep`, `boost`
(hopai.vectors.parse_boost) and `rerank`; each hop accepts those plus
`via_near` and `via_keep`. They mirror Start/Hop exactly. Unknown keys
are REFUSED rather than ignored, because `top_k`/`limit`/`filter` are
the names a model reaches for and silently dropping one answers a
different question than the one asked.

`start` also accepts `ids`, a list of node ids to seed from directly --
ANDed with `where` when both are given. `where` filters PROPERTIES, and
an id is not one: `{"where": {"id": 7}}` is a JSONB containment test
that matches nothing. `ids` is the one way to seed from a specific row
you already hold, the read-side counterpart of mutate.py's `ids=`.

A `rerank` spec is {"document_from": <jq filter>, "candidates": N} and
NEVER a client: a reranker holds an API key and a socket, neither of
which travels in JSON. The reranker is the caller's, handed to these
functions as `rerank=RerankPolicy(Rerank(...), fields=[...])`, and a
spec may override only those two keys of it. `document_from` is the one
place a model writes CODE rather than data, so parse_rerank() is the
single site that holds it to hopai.jqsafe's total subset and to the
property paths the policy publishes -- the filter's output IS the
document, and the document is posted to a third party.

A near spec takes "text" OR "vector", and the difference is the whole
LLM story.

"text" is embedded by the field itself, with the client the
application declared -- so the query embedding comes from the same
model that wrote the stored ones, and a tool-calling model can fill it
in as truthfully as any filter.

"vector" is DELIBERATELY the one key
parsed and never advertised: asked for floats a model invents
plausible ones, and an invented embedding finds confidently wrong
neighbors. It exists for HTTP/config callers holding real vectors, and
traverse_json()/aggregate_json()/vector_search_json() refuse it unless
the caller says allow_vectors=True -- so the invariant is enforced and
not merely advertised. hopai/vectors.py has the full reasoning; a test
pins it.

An aggregation spec is the same thing plus an `aggregates` object (see
hopai.aggregates for what the forms mean), run with aggregate_json():

    {
      "start": {"where": {"type": "person"}},
      "hops": [{"via": {"kind": "friend"}, "hops": [1, 4]}],
      "aggregates": {"friends": {"fn": "count"},
                     "avg_age": {"fn": "avg", "property": "age"}}
    }
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Union

from . import jqsafe
from .aggregates import parse_aggregate
from .core import Graph, Subgraph
from .filters import parse_filter
from .hop import Hop, Start
from .rerankers import Rerank, RerankError
from .vectors import parse_boost, parse_near


#: Keys each spec object accepts. Unknown keys are refused rather than
#: ignored: `top_k`, `limit` and `filter` are exactly the names a model
#: has seen ten thousand times, and silently ignoring them answered a
#: different question than the one asked -- unfiltered, or with the
#: default k. parse_near/parse_aggregate are strict one level down;
#: this is the same rule at the top.
_START_KEYS = {"where", "ids", "label", "near", "keep", "boost", "rerank"}
_HOP_KEYS = {"where", "via", "hops", "direction", "optional", "label",
             "near", "keep", "via_near", "via_keep", "boost", "rerank"}
_SEARCH_KEYS = {"near", "target", "k", "where", "boost", "rerank"}
#: The keys whose values are near specs, and so the only ones that can
#: carry a literal embedding. Everything else in the similarity family
#: -- keep, via_keep, boost -- holds integers and property names a
#: model can supply as truthfully as any filter.
_NEAR_KEYS = ("near", "via_near")
#: The keys a spec's `rerank` object may carry. A CLIENT is not among
#: them and never will be: it holds an API key and an open socket,
#: neither of which travels in JSON. What a caller on the far side of a
#: tool call legitimately decides is WHICH TEXT each candidate is
#: reduced to (`document_from`) and how many candidates to spend on the
#: question (`candidates`) -- both retrieval choices a model is well
#: placed to make, unlike an embedding, which it would have to invent.
_RERANK_KEYS = {"document_from", "candidates"}


def _check_keys(spec: dict, allowed: set, what: str) -> None:
    if not isinstance(spec, dict):
        raise TypeError(f"{what} must be an object -- got {type(spec).__name__}")
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(
            f"unknown {what} keys {unknown} -- {what} accepts {sorted(allowed)}"
        )


def _near_of(spec: dict, key: str = "near"):
    return parse_near(spec[key]) if key in spec else None


def _parse_via(via_spec):
    """via= accepts everything parse_filter() does, PLUS a bare string --
    the STORED_IN shorthand (Hop.via="KIND"), which parse_filter's
    dict-only JSON form has no other use for. Kept here rather than
    widened into parse_filter() itself: `where=` reuses parse_filter too,
    and a bare string means nothing at that position -- it would still
    need refusing, just one layer later inside resolve() instead of here,
    for no benefit and a wider (so easier to misuse elsewhere) contract
    on the shared parser."""
    if isinstance(via_spec, str):
        return via_spec
    return parse_filter(via_spec)


def _boost_of(spec: dict):
    return parse_boost(spec["boost"]) if "boost" in spec else None


def _rerank_of(spec: dict, policy: Optional[RerankPolicy], owner: str):
    return parse_rerank(spec["rerank"], policy, owner) if "rerank" in spec else None


# ---------------------------------------------------------------------
# Reranking, from a spec that may have been written by a model
# ---------------------------------------------------------------------

def _published(fields, owner: str) -> tuple:
    """The operator's field allowlist, checked where they wrote it.

    A non-empty list is REQUIRED and there is no "everything" spelling,
    which is the whole point: the filter's output is the document and
    the document is posted to a third-party reranker, so somebody has to
    have decided what may leave the building. `["properties"]` is how
    you say "the whole properties bag" when that really is the
    decision -- said, rather than defaulted into."""
    if fields is None:
        raise ValueError(
            f"{owner}: name the property paths a document_from may read, e.g. "
            f'fields=["properties.title", "properties.summary"]. There is no way to '
            f"say `everything`: the filter's output is posted to a third-party "
            f"reranker, so `.properties.ssn` is one accepted filter away unless the "
            f'paths are named. Pass ["properties"] if the whole bag really is the '
            f"decision"
        )
    if isinstance(fields, str) or not isinstance(fields, (list, tuple)):
        raise TypeError(
            f"{owner}: fields= takes a list of dotted property paths, got "
            f'{type(fields).__name__} -- e.g. ["properties.title", "properties.tags"]'
        )
    if not fields:
        raise ValueError(
            f"{owner}: fields=[] publishes nothing, so every document_from would "
            f'refuse -- name the paths a filter may read, e.g. ["properties.title"]'
        )
    for field in fields:
        if not isinstance(field, str) or not field.strip():
            raise TypeError(
                f"{owner}: fields= takes dotted property paths as strings, got "
                f'{field!r} -- e.g. ["properties.title"]'
            )
    return tuple(fields)


@dataclass(frozen=True)
class RerankPolicy:
    """What a spec's `rerank` is allowed to become: the caller's own
    Rerank as a TEMPLATE, the property paths a spec-supplied
    `document_from` may read, and a ceiling on how many candidates one
    call may spend.

        RerankPolicy(
            Rerank(client, document_from='.properties.title', candidates=50),
            fields=["properties.title", "properties.summary"],
            max_candidates=100,
        )

    THE CLIENT IS NEVER SPEC-SUPPLIED, exactly as `embed=` is never
    spec-supplied: only `document_from` and `candidates` are overridden
    from a spec, and everything that holds a credential or decides a
    bill -- the client, the model, the batch size, the retry budget --
    stays as constructed here. A spec arriving over a tool call chooses
    what to rank on, never what to call.

    `fields` is the allowlist parse_rerank() hands to hopai.jqsafe. A
    path outside it refuses naming the published list, because
    `.properties.ssn` parses perfectly well in the safe subset and would
    ship straight to a vendor. Naming `paths` publishes the WHOLE nodes
    on every route that reached a candidate, their properties included
    -- it is the one entry that is not a single value.

    `max_candidates` bounds the bill: rerankers price per document, and
    a spec choosing candidates=200 across a three-hop traversal is 600
    documents for one query. A larger `candidates` REFUSES naming the
    ceiling rather than being clamped, for the same reason
    `candidates < keep` refuses -- quietly serving different numbers
    than the caller asked for hides that the numbers disagree. `None`
    disables it, and is for a caller whose specs are their own."""

    template: Any
    fields: Any = None
    max_candidates: Optional[int] = None

    def __post_init__(self):
        owner = "RerankPolicy"
        if not isinstance(self.template, Rerank):
            raise TypeError(
                f"{owner}: the template must be a Rerank, got "
                f"{type(self.template).__name__} -- e.g. RerankPolicy(Rerank(client, "
                f"document_from='.properties.title', candidates=50), "
                f'fields=["properties.title"])'
            )
        object.__setattr__(self, "fields", _published(self.fields, owner))
        if self.max_candidates is not None and (
                isinstance(self.max_candidates, bool)
                or not isinstance(self.max_candidates, int) or self.max_candidates < 1):
            raise ValueError(
                f"{owner}: max_candidates must be a positive integer, or None to serve "
                f"specs with no ceiling at all, got {self.max_candidates!r}"
            )
        if self.max_candidates is not None and self.template.candidates > self.max_candidates:
            # Refused here rather than per query: the template's own
            # `candidates` is what a spec that names none inherits, so
            # this configuration would have every default call refuse
            # for a reason that is the operator's, not the caller's.
            raise ValueError(
                f"{owner}: the template reranks {self.template.candidates} candidates, "
                f"over its own max_candidates={self.max_candidates} -- a spec naming no "
                f"`candidates` inherits the template's, so every such call would refuse. "
                f"Raise max_candidates, or lower the template's candidates="
            )
        # The operator's own filter, held to the list the operator
        # published -- at the line that wrote both, rather than on the
        # first query that happens to inherit it.
        jqsafe.validate(self.template.document_from, fields=self.fields,
                        owner="the reranker's own document_from")


def parse_rerank(spec: Any, policy: Optional[RerankPolicy], owner: str = "rerank"):
    """The JSON form of Rerank -- and THE ONE PLACE a spec-supplied jq
    filter is held to hopai.jqsafe's safe subset.

    This is to `document_from` what refuse_vectors() is to `vector`: the
    single enforcement site, public and named without an underscore so a
    front end calls it rather than repeating it. hopai/mcp.py must not
    import hopai.jqsafe at all -- every filter it serves arrives through
    a spec, every spec reaches a Rerank through here, and a second copy
    of the rule is a second place for it to rot.

    Both checks it makes are about what leaves the building rather than
    about what runs:

      - THE GRAMMAR. jq's `env` returns the process environment, and the
        filter's output IS the document that is POSTed to a reranking
        vendor, so `document_from='env.DATABASE_URL'` is one-line
        exfiltration of the DSN. hopai.jqsafe's subset does not parse
        `env`, `$ENV`, `input`, `def` or `range` at all -- see its
        module docstring for why an allowlist over parsed syntax is a
        decision procedure here rather than a blacklist arms race.
      - THE FIELDS. `.properties.ssn` parses perfectly in that subset.
        Only the policy's published paths may be read, and a filter
        reading anything else refuses naming them.

    The template is never replaced, only overridden: `document_from` and
    `candidates` come from the spec when it names them and from the
    template when it does not."""
    if policy is None:
        raise ValueError(
            f"{owner}: this spec asks for reranking and none is configured -- a reranker "
            f"client cannot travel in JSON, since it holds an API key and an open socket, "
            f"so the one to use has to be yours. Pass rerank=RerankPolicy(Rerank(client, "
            f"document_from='...'), fields=[...]) to this call, or drop \"rerank\" from "
            f"the spec"
        )
    if not isinstance(spec, dict):
        raise TypeError(
            f'"{owner}" must be an object -- got {type(spec).__name__}, e.g. '
            '{"document_from": ".properties.title", "candidates": 50}'
        )
    _check_keys(spec, _RERANK_KEYS, f'"{owner}"')
    template = policy.template

    document_from = spec.get("document_from", template.document_from)
    if not isinstance(document_from, str) or not document_from.strip():
        raise ValueError(
            f'{owner}: "document_from" is a jq filter as a non-empty string, got '
            f'{document_from!r} -- e.g. \'.properties.title + ": " + '
            f'(.properties.summary // "")\''
        )
    candidates = spec.get("candidates", template.candidates)
    if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates < 1:
        raise ValueError(
            f'{owner}: "candidates" is how many hits reach the reranker before keep/k '
            f"truncates, so it must be a positive integer, got {candidates!r}"
        )
    if policy.max_candidates is not None and candidates > policy.max_candidates:
        # Named, never clamped -- the same judgement `candidates < keep`
        # gets. A reranker is billed per document, so silently serving
        # 100 where 500 was asked for hides the disagreement in exactly
        # the place it costs money.
        raise ValueError(
            f'{owner}: "candidates" is {candidates}, over the {policy.max_candidates} '
            f"this reranker is configured to allow -- a reranking model is billed per "
            f"document, so the ceiling is not something a spec can raise. Ask for at "
            f"most {policy.max_candidates}, or narrow `where` so that many candidates "
            f"are worth more"
        )
    jqsafe.validate(document_from, fields=policy.fields, owner=f"{owner}.document_from")
    return _SpecRerank(
        template.client, document_from=document_from, candidates=candidates,
        per_path=template.per_path, max_paths=template.max_paths, model=template.model,
        batch_size=template.batch_size, retries=template.retries, backoff=template.backoff,
    )


def _document_failure(document_from: str, candidate: Any, failed: BaseException) -> ValueError:
    """The refusal that replaces a document-building failure whose own
    message quotes the row.

    MEASURED, not assumed: jq's runtime errors quote the offending value
    verbatim -- `ValueError: string ("SSN-123-45-6789") cannot be parsed
    as a number` -- and so do two of _evaluate()'s own, which repr the
    value the filter produced. `fields=` decides which properties a
    filter may READ, so this is not a privilege escalation; it is a row
    value taking a route out of the process that nobody chose, into a
    tool result and whatever logs the far side keeps.

    WHAT REPLACES IT STILL NAMES THE FIX, because an error a model
    cannot act on costs a retry: the filter, the candidate, the KIND of
    failure and the rewrite. The kind is read off the exception's shape
    rather than its text -- TypeError is "not a string", a ValueError
    with a `__cause__` is jq failing on the row, and one without is
    _evaluate()'s own count check -- so nothing here parses a message
    that another module is free to reword."""
    where = candidate.get("id") if isinstance(candidate, dict) else None
    if isinstance(failed, TypeError):
        reason = "it evaluated to something that is not a string"
        fix = ("Put the default BEFORE the conversion, e.g. "
               "'.properties.year // \"unknown\" | tostring'")
    elif failed.__cause__ is not None:
        reason = ("jq itself failed on that row -- most often a type mismatch, such as "
                  "adding text to a number or asking tonumber() of something that is not "
                  "one")
        fix = ("Guard every field the row may not have, e.g. "
               "'(.properties.title // \"\") + \": \" + (.properties.summary // \"\")'")
    else:
        reason = "it produced no document at all, or more than one"
        fix = ("Give it a fallback (e.g. '.properties.title // \"untitled\"'), or wrap "
               "several outputs into one (e.g. '[.properties.tags[]] | join(\", \")')")
    return ValueError(
        f"document_from={document_from!r} could not build a document for candidate "
        f"id={where!r} -- {reason}. The underlying error is not repeated here, because "
        f"it quotes that row's own values and this document is posted to a reranking "
        f"provider. {fix}"
    )


def _provider_failure(document_from: str) -> RerankError:
    """The refusal that replaces a spent RerankError.

    Same rule as above, one layer out: a provider SDK's exception repr
    can carry the configuration it was constructed with, API key
    included, and RerankError quotes it verbatim so a Python caller can
    see what happened. A Python caller is inside the trust boundary; the
    far side of a tool call is not.

    It stays a RerankError, so `except RerankError` still catches it and
    the "raised, never degraded" contract is unchanged -- only the text
    is."""
    return RerankError(
        f"rerank=(document_from={document_from!r}): the reranking provider call failed "
        f"and was not retried into success. The provider's own error is not repeated "
        f"here, because an SDK exception can carry the credentials it was configured "
        f"with. Nothing was ranked rather than falling back to the un-reranked order, "
        f"which would be a different answer with no signal. This is a server-side "
        f"failure, not something to fix in this call"
    )


class _SpecRerank(Rerank):
    """The Rerank a spec produced, whose failures may not quote the row.

    Identical to Rerank in every respect that decides an answer -- same
    client, same documents, same scores -- and different only in what it
    says when something goes wrong. Constructed by parse_rerank() alone,
    which is exactly the trust boundary: a Rerank written in Python
    keeps jq's own diagnostics, because the person reading them wrote
    the filter and already holds the rows.

    Documents are built ONE CANDIDATE AT A TIME so the failing candidate
    is known STRUCTURALLY rather than scraped out of a message. The cost
    is a Python call per candidate against a compiled, cached filter,
    which is noise beside the provider round trip that follows."""

    def build_documents(self, candidates: list, *, trusted: bool = False, fields=None) -> list:
        if isinstance(candidates, (dict, str, bytes)):
            # The base's own call-shape refusal, which names a type and
            # no data -- and which iterating here would turn into a
            # complaint about the string 'id'.
            return super().build_documents(candidates, trusted=trusted, fields=fields)
        documents = []
        for candidate in candidates:
            try:
                documents.extend(
                    super().build_documents([candidate], trusted=trusted, fields=fields))
            except jqsafe.UnsafeFilter:
                # About the FILTER, never about a row: it quotes the
                # program the caller wrote and names the construct that
                # is not allowed, which is the message they need.
                raise
            except (ValueError, TypeError) as failed:
                raise _document_failure(self.document_from, candidate, failed) from None
        return documents

    def score(self, query: str, documents: list) -> list:
        try:
            return super().score(query, documents)
        except RerankError:
            raise _provider_failure(self.document_from) from None

    async def ascore(self, query: str, documents: list) -> list:
        try:
            return await super().ascore(query, documents)
        except RerankError:
            raise _provider_failure(self.document_from) from None


def spec_to_traversal(spec: dict, rerank: Optional[RerankPolicy] = None) -> tuple:
    """Convert a JSON spec into (Start, [Hop, ...]). Exposed on its own
    in case you want to inspect or modify the parsed traversal before
    running it.

    `rerank` is the RerankPolicy a `rerank` key in the spec is measured
    against; without one, a spec asking to rerank refuses rather than
    reranking with a client it cannot have."""
    if "start" not in spec:
        raise ValueError("spec must have a 'start' key, e.g. {\"start\": {\"where\": {...}}}")

    _check_keys(spec, {"start", "hops"}, "traversal spec")
    start_spec = spec["start"]
    _check_keys(start_spec, _START_KEYS, '"start"')
    start = Start(
        where=parse_filter(start_spec.get("where")),
        ids=start_spec.get("ids"),
        label=start_spec.get("label"),
        near=_near_of(start_spec),
        keep=start_spec.get("keep"),
        boost=_boost_of(start_spec),
        rerank=_rerank_of(start_spec, rerank, "start.rerank"),
    )

    hops = []
    for index, h in enumerate(spec.get("hops", [])):
        _check_keys(h, _HOP_KEYS, '"hops" entry')
        hops.append(
            Hop(
                where=parse_filter(h.get("where")),
                via=_parse_via(h.get("via")),
                hops=tuple(h["hops"]) if isinstance(h.get("hops"), list) else h.get("hops", 1),
                direction=h.get("direction", "forward"),
                optional=h.get("optional", False),
                label=h.get("label"),
                near=_near_of(h),
                keep=h.get("keep"),
                via_near=_near_of(h, "via_near"),
                via_keep=h.get("via_keep"),
                boost=_boost_of(h),
                rerank=_rerank_of(h, rerank, f"hops[{index}].rerank"),
            )
        )
    return start, hops


def refuse_vectors(spec: dict, caller: str) -> None:
    """The one place the "a VECTOR never travels through the LLM"
    invariant is ENFORCED rather than advertised.

    Only the literal floats are refused. `text` is the model's way in
    and is meant to be used: the field embeds it with the client the
    application declared, so the embedding comes from the same model
    that wrote the stored ones. An invented `vector` cannot -- it finds
    confidently wrong neighbors and reports no error, which rule 4 says
    is the worst thing this library can produce.

    Omitting the key from the tool schemas is not a defence on its own:
    the schemas carry no additionalProperties:false, the JSON grammar
    documents `vector` for callers that hold real floats, and this
    function is what an agent integration actually calls.

    Public, and named without the underscore, because it is not only
    this module's business: hopai/mcp.py embeds the model's TEXT into
    a real vector itself and then passes allow_vectors=True, so it has
    to make this exact refusal first -- against the same keys, with the
    same message. A second copy of the rule is a second place for it to
    rot.

    `spec` itself is the third scope, not just start/hops: a
    vector_search_json() spec carries its near at the top level."""
    for scope in (spec.get("start") or {}, *(spec.get("hops") or []), spec):
        if not isinstance(scope, dict):
            continue
        for key in _NEAR_KEYS:
            for one in _as_list(scope.get(key) or []):
                if isinstance(one, dict) and "vector" in one:
                    raise ValueError(
                        f"{caller}: {key}={{...\"vector\": [...]}} cannot come from a tool "
                        f"call -- a model cannot supply a real embedding, and an invented "
                        f"one finds confidently wrong neighbors. Use "
                        # A placeholder rather than an example field name
                        # when the spec named none: inventing one reads
                        # as "you asked for summary", which they did not.
                        f'{{"field": {one.get("field", "<your field>")!r}, "text": "..."}} '
                        f'and let '
                        f"the field embed it, or pass allow_vectors=True if this spec came "
                        f"from your own code"
                    )


def traverse_json(graph: Graph, spec: dict, allow_vectors: bool = False,
                  rerank: Optional[RerankPolicy] = None) -> dict:
    """Run a traversal described entirely in JSON and return a
    JSON-serializable dict -- the single call an LLM tool integration or
    HTTP handler needs: no Python objects in, no Python objects out.

        result = traverse_json(graph, {
            "start": {"where": {"type": "person"}},
            "hops": [{"where": {"active": True}, "hops": [1, 4]}],
        })
        result["nodes"], result["edges"], result["elapsed_ms"]

    Similarity keys (`near`/`keep`/`via_near`/`via_keep`/`boost`) are
    REFUSED unless allow_vectors=True: this is the call an agent
    integration wires up, and a model cannot supply a real embedding.
    Application code holding real vectors opts in explicitly.

    `rerank` is a RerankPolicy -- YOUR reranker, plus the property
    paths a spec's `document_from` may read and the ceiling on how many
    candidates it may spend. Without one a spec's `rerank` refuses,
    since the client can only be yours.
    """
    if not allow_vectors:
        refuse_vectors(spec, "traverse_json()")
    start, hops = spec_to_traversal(spec, rerank)
    subgraph: Subgraph = graph.traverse(start, *hops)
    return subgraph.to_dict()


def spec_to_aggregation(spec: dict, rerank: Optional[RerankPolicy] = None) -> tuple:
    """Convert a JSON spec into (Start, [Hop, ...], {name: aggregate},
    group_by). The traversal half is exactly spec_to_traversal();
    `aggregates` maps result names to JSON-form aggregates for
    hopai.aggregates.parse_aggregate(). `group_by`, when the spec
    carries one, is a property name (a string) read off the SAME
    last-hop nodes the aggregates already run over -- see
    Graph.aggregate()'s own docstring for exactly what it means; `None`
    when the spec has no "group_by" key."""
    if not spec.get("aggregates"):
        raise ValueError(
            'spec must have a non-empty "aggregates" object, e.g. '
            '{"aggregates": {"n": {"fn": "count"}}}'
        )
    group_by = spec.get("group_by")
    if group_by is not None and not isinstance(group_by, str):
        raise ValueError(f'"group_by" must be a property name string -- got {group_by!r}')
    start, hops = spec_to_traversal(
        {k: v for k, v in spec.items() if k not in ("aggregates", "group_by")}, rerank)
    aggregates = {name: parse_aggregate(a) for name, a in spec["aggregates"].items()}
    return start, hops, aggregates, group_by


def aggregate_json(graph: Graph, spec: dict, allow_vectors: bool = False,
                   rerank: Optional[RerankPolicy] = None) -> Union[dict, list]:
    """Run an aggregation described entirely in JSON and return a
    JSON-serializable dict of the named results -- the aggregation
    counterpart of traverse_json(), and the call behind
    AGGREGATE_TOOL_SCHEMA.

        aggregate_json(graph, {
            "start": {"where": {"type": "person"}},
            "hops": [{"via": {"kind": "friend"}, "hops": [1, 4]}],
            "aggregates": {"friends": {"fn": "count"},
                           "avg_age": {"fn": "avg", "property": "age"}},
        })
        # -> {"friends": 42, "avg_age": 31.5}

    `"group_by": "<property>"` groups every aggregate by that property
    on the same nodes, and the result becomes a LIST of per-group dicts
    instead of one -- see Graph.aggregate()'s own docstring:

        aggregate_json(graph, {
            "start": {"where": {"type": "person"}},
            "aggregates": {"n": {"fn": "count"}},
            "group_by": "city",
        })
        # -> [{"city": "Berlin", "n": 2}, {"city": None, "n": 1}, ...]

    Refuses similarity keys unless allow_vectors=True, for the reason
    traverse_json() gives.
    """
    if not allow_vectors:
        refuse_vectors(spec, "aggregate_json()")
    start, hops, aggregates, group_by = spec_to_aggregation(spec, rerank)
    return graph.aggregate(start, *hops, aggregates=aggregates, group_by=group_by)


def vector_search_json(graph: Graph, spec: dict, allow_vectors: bool = False,
                       rerank: Optional[RerankPolicy] = None) -> dict:
    """Run a vector search described entirely in JSON -- the
    vector_search() counterpart of traverse_json(), and the call behind
    VECTOR_SEARCH_TOOL_SCHEMA.

        vector_search_json(graph, {
            "near": {"field": "summary", "text": "how do nodes agree?"},
            "k": 10,
            "target": "nodes",
            "where": {"type": "person"},
        })
        # -> {"results": [{"id": "1", "similarity": 0.93, "properties": {...},
        #                  "similarities": {"summary": 0.93}, "boosts": {}}, ...]}

    Same gate as traverse_json(): a literal "vector" is refused unless
    allow_vectors=True, while "text" is embedded by the field itself
    and is what a model is meant to send.

    `rerank` is a RerankPolicy, exactly as in traverse_json(); the
    spec's own `rerank` object sits beside `near` at the top level,
    because a flat search has one ranked list and one place to reorder
    it.
    """
    _check_keys(spec, _SEARCH_KEYS, "vector search spec")
    if not allow_vectors:
        refuse_vectors(spec, "vector_search_json()")
    if "near" not in spec:
        raise ValueError('spec must have a "near" key, e.g. '
                         '{"near": {"field": "summary", "text": "..."}}')
    results = graph.vector_search(
        *_as_list(parse_near(spec["near"])),
        target=spec.get("target", "nodes"),
        k=spec.get("k", 10),
        where=parse_filter(spec.get("where")),
        boost=_boost_of(spec),
        rerank=_rerank_of(spec, rerank, "rerank"),
    )
    return {"results": results}


def _as_list(near) -> list:
    return near if isinstance(near, list) else [near]


# A JSON Schema description of the spec format above, ready to hand
# directly to an LLM function-calling / tool definition. Kept here
# rather than hand-duplicated wherever hopai gets wired into an
# agent framework or MCP server later.
#
# EXACTLY ONE key is parsed and never advertised: a near spec's
# "vector". A model filling it invents floats, and invented embeddings
# find confidently wrong neighbors -- it exists for callers that hold
# real ones. "text" is its replacement here and is meant to be used:
# the field embeds it with the application's own client, so the query
# embedding comes from the model that wrote the stored ones.
# tests/test_vectors.py pins the omission.


def _near_schema(what: str) -> dict:
    """The near spec as JSON Schema -- everything parse_near() accepts
    except "vector". Built by a function because it appears four times
    and a hand-copied fifth would be the one that drifts."""
    return {
        "description": what,
        "anyOf": [{"$ref": "#/$defs/near"}, {"type": "array", "items": {"$ref": "#/$defs/near"}}],
    }


_NEAR_DEF: dict = {
    "type": "object",
    "properties": {
        "field": {
            "type": "string",
            "description": "Name of the vector field to compare against.",
        },
        "text": {
            "type": "string",
            "description": (
                "The text to search for. The application embeds this with the same "
                "model that produced the stored vectors -- write what you are looking "
                "for in words, and never numbers."
            ),
        },
        "weight": {
            "type": "number",
            "description": (
                "This field's coefficient when several near specs are combined into "
                "one score. Leave it out unless you are ranking on more than one field."
            ),
        },
        "min_similarity": {
            "type": "number",
            "description": (
                "Drop rows scoring below this cosine similarity on this field, "
                "between -1 and 1. A filter, applied before the row limit."
            ),
        },
        "missing": {
            "type": "string",
            "enum": ["exclude", "zero"],
            "description": (
                "What to do with rows that have no vector for this field: drop them "
                "(default), or score them 0 here and let other fields carry the row."
            ),
        },
    },
    "required": ["field", "text"],
}
#: A non-similarity term in a ranked score: hybrid retrieval. Only
#: meaningful alongside `near`, which is what Boost itself refuses
#: without -- said here so a model does not reach for it alone.
_BOOST_DEF: dict = {
    "type": "object",
    "description": (
        "Add a numeric property to the similarity score, so ranking is not by "
        "meaning alone. Only valid alongside `near`."
    ),
    "properties": {
        "property": {"type": "string", "description": "Numeric node property to add in."},
        "weight": {"type": "number", "description": "Its coefficient in the combined score."},
        "missing": {
            "type": "number",
            "description": "Value to use for rows lacking the property. Defaults to 0.",
        },
        "scale": {
            "type": "string",
            "enum": ["normalized", "raw"],
            "description": (
                "'normalized' (default) rescales the property into similarity's own "
                "[-1, 1] range over the candidate rows before weighting it, so the "
                "weight means what it says regardless of the property's own scale. "
                "'raw' multiplies the property as stored, unbounded -- yesterday's "
                "behavior, for a caller who already normalized it."
            ),
        },
    },
    "required": ["property", "weight"],
}

#: The two sentences a SERVER rewrites once it knows its own numbers --
#: hopai/mcp.py replaces each with the published field list and the real
#: ceiling, the same conditional-sentence idiom _with_search() uses for
#: `start.search`. Constants rather than prose so a reword here fails
#: loudly there instead of leaving two descriptions contradicting each
#: other. With no server in sight these can only say that a list and a
#: cap exist.
RERANK_FIELDS_SENTENCE = (
    "Only a small, always-terminating subset of jq is accepted, and only the properties "
    "the application publishes may be read -- anything else refuses and names them."
)
RERANK_CEILING_SENTENCE = (
    "The application may cap this, and a larger value refuses naming the cap rather than "
    "being quietly lowered."
)

#: The rerank spec as JSON Schema. `document_from` is the one place a
#: model writes CODE, so its description shows a whole candidate rather
#: than describing one: a model writes markedly better jq against a
#: concrete shape than against a paragraph about a shape. There is no
#: "client" key here and there never will be -- see _RERANK_KEYS.
_RERANK_DEF: dict = {
    "type": "object",
    "description": (
        "Rerank what `near` ranked, by having a reranking model READ each candidate "
        "against your query text -- the accurate, expensive stage after the cheap "
        "similarity one. It reorders and truncates a list that already exists; it never "
        "creates one, so it needs `near` (with `text`) beside it, and `keep`/`k` still "
        "decides how many survive. On a traversal step `keep` is REQUIRED with it: the "
        "result is a subgraph rather than a ranking, so the reranked order is discarded "
        "and truncating is the only mark the reranking can leave. NEITHER KEY BELOW IS "
        "REQUIRED -- whichever you leave out keeps the application's own setting, and "
        "`{}` asks for reranking exactly as it is configured."
    ),
    "properties": {
        "document_from": {
            "type": "string",
            "description": (
                "A jq filter that turns ONE candidate into the ONE string the reranking "
                "model reads. It is a RULE, evaluated once per candidate after the "
                "search runs -- not a document you write, because none of them exist "
                "yet. A candidate is exactly this shape: "
                '{"id": "7", "properties": {"title": "Raft", "abstract": "a consensus '
                'protocol", "tags": ["consensus", "replication"]}, "similarity": 0.81, '
                '"similarities": {"abstract": 0.81}, "boosts": {}} -- plus "paths", the '
                "routes that reached it, on a hop but never on `start`. So "
                "'.properties.title + \": \" + (.properties.abstract // \"\")' builds "
                "\"Raft: a consensus protocol\" from it. Guard anything a row may not "
                "have with `// \"\"`: a filter that yields nothing, several things, or "
                "something that is not text refuses rather than ranking against the "
                "wrong words. " + RERANK_FIELDS_SENTENCE + " Leave it out to use the "
                "application's own filter, which is the right answer unless you need to "
                "rank on different text."
            ),
        },
        "candidates": {
            "type": "integer",
            "description": (
                "How many hits the reranking model reads, before `keep`/`k` truncates -- "
                "the INPUT bound, where keep/k is the output bound. On a traversal step "
                "it has to be MORE than `keep`: a traversal returns a subgraph and throws "
                "the reranked order away, so truncating is the only thing reranking can "
                "change there, and candidates equal to (or below) `keep` refuses as a "
                "call billed for a guaranteed no-op. In a vector search, which reports "
                "the new order and a `rerank_score`, it need only be at least `k`. "
                + RERANK_CEILING_SENTENCE
            ),
        },
    },
}


TRAVERSE_TOOL_SCHEMA: dict = {
    "name": "traverse_graph",
    "description": (
        "Traverse a property graph stored in PostgreSQL. Follow edges from a "
        "starting set of nodes through one or more filtered hops, forward or "
        "backward, bounded or ranged depth, and return every node and edge "
        "on a complete matching path. `where`/`via` are EXACT property matches; "
        "to select nodes by MEANING instead, give `near` a field and the text to "
        "look for, with `keep` to say how many to keep."
    ),
    "parameters": {
        "type": "object",
        "$defs": {"near": _NEAR_DEF, "boost": _BOOST_DEF, "rerank": _RERANK_DEF},
        "properties": {
            "start": {
                "type": "object",
                "description": "The seed set of nodes to begin from.",
                "properties": {
                    "where": {"type": "object", "description": "Filter on starting nodes."},
                    "ids": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "description": (
                            "Seed from these specific node ids directly, instead of (or "
                            "as well as) `where`. `where` filters PROPERTIES, and an id "
                            "is not one -- {\"where\": {\"id\": 7}} matches nothing. "
                            "Combines with `where` as AND."
                        ),
                    },
                    "near": _near_schema(
                        "Rank the starting nodes by similarity to some text instead of "
                        "(or as well as) filtering them. Needs `keep`."),
                    "keep": {
                        "type": "integer",
                        "description": "How many of the highest-scoring starting nodes to keep.",
                    },
                    "boost": {"$ref": "#/$defs/boost"},
                    "rerank": {"$ref": "#/$defs/rerank"},
                },
                # A seed set has to come from SOMEWHERE. Before `near`
                # existed this was `required: ["where"]`; dropping it to
                # let a purely semantic seed through advertised `{}` as
                # a valid start, which is "every node in the graph" --
                # the unbounded result of #47, handed to a model as a
                # legal call. Either way in is fine; neither is not.
                "anyOf": [{"required": ["where"]}, {"required": ["near"]}],
            },
            "hops": {
                "type": "array",
                "description": "Ordered list of traversal steps, applied one after another.",
                "items": {
                    "type": "object",
                    "properties": {
                        "where": {"type": "object", "description": "Filter on the node reached by this hop."},
                        "via": {
                            "anyOf": [{"type": "object"}, {"type": "string"}],
                            "description": (
                                "Filter on edges traversed during this hop. Either an exact-"
                                "match object, e.g. {\"kind\": \"friend\"}, or a bare string "
                                "naming the edge's type directly, e.g. \"friend\" -- shorthand "
                                "for the same filter, on the declared-edge-type fast path."
                            ),
                        },
                        "hops": {
                            "description": "Exact hop count (integer) or [min, max] range.",
                            "anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}],
                        },
                        "direction": {"type": "string", "enum": ["forward", "backward"],
                                     "description": "Which way to walk this hop's edges: "
                                                    "forward follows start->end, backward "
                                                    "follows end->start. Default forward."},
                        "optional": {
                            "type": "boolean",
                            "description": "If true, keep prior matches even if this hop finds nothing. Only valid on the last hop.",
                        },
                        "near": _near_schema(
                            "Rank the nodes this hop reaches by similarity to some text. "
                            "Needs `keep`."),
                        "keep": {
                            "type": "integer",
                            "description": "How many of the highest-scoring reached nodes to keep.",
                        },
                        "via_near": _near_schema(
                            "Rank the EDGES this hop traverses by similarity to some text. "
                            "Needs `via_keep`."),
                        "via_keep": {
                            "type": "integer",
                            "description": "How many of the highest-scoring edges to follow.",
                        },
                        "boost": {"$ref": "#/$defs/boost"},
                        "rerank": {"$ref": "#/$defs/rerank"},
                    },
                },
            },
        },
        "required": ["start"],
    },
}



VECTOR_SEARCH_TOOL_SCHEMA: dict = {
    "name": "search_graph_by_meaning",
    "description": (
        "Find the nodes or edges whose stored text is closest in MEANING to what "
        "you describe -- semantic search, not an exact property match. Returns each "
        "hit with its similarity score. Use this when you do not know the exact "
        "property values to filter on, and traverse_graph when you do."
    ),
    "parameters": {
        "type": "object",
        "$defs": {"near": _NEAR_DEF, "boost": _BOOST_DEF, "rerank": _RERANK_DEF},
        "properties": {
            "near": _near_schema("The field to search and the text to search for."),
            "target": {
                "type": "string",
                "enum": ["nodes", "edges"],
                "description": "What to search. Defaults to nodes.",
            },
            "k": {
                "type": "integer",
                "description": "How many results to return, best first. Defaults to 10.",
            },
            "where": {
                "type": "object",
                "description": (
                    "Optional exact-match filter applied before ranking -- narrowing the "
                    "candidates is what makes this fast."
                ),
            },
            "boost": {"$ref": "#/$defs/boost"},
            "rerank": {"$ref": "#/$defs/rerank"},
        },
        "required": ["near"],
    },
}


# The aggregation counterpart. Spelled out rather than sharing objects
# with TRAVERSE_TOOL_SCHEMA because the two genuinely differ: hops here
# have no `optional` (it cannot change an aggregate, so aggregate()
# refuses it) -- a test keeps the shared parts in step instead.
AGGREGATE_TOOL_SCHEMA: dict = {
    "name": "aggregate_graph",
    "description": (
        "Aggregate over the nodes a graph traversal matches: count them, or "
        "sum/average/min/max a numeric property -- computed in the database in one "
        "round trip, without returning the nodes themselves. Takes the same "
        "start/hops traversal spec as traverse_graph. Aggregates run over the "
        "distinct nodes matched by the LAST hop (the starting set when there are "
        "no hops), each node counted once however many paths reach it. "
        "`where`/`via` are EXACT property matches; `near` selects by meaning. "
        "Add `group_by` to compute the aggregates once per distinct value of a "
        "property on those same nodes -- the result becomes a list of one object "
        "per group instead of a single object."
    ),
    "parameters": {
        "type": "object",
        "$defs": {"near": _NEAR_DEF, "boost": _BOOST_DEF, "rerank": _RERANK_DEF},
        "properties": {
            "start": {
                "type": "object",
                "description": "The seed set of nodes to begin from.",
                "properties": {
                    "where": {"type": "object", "description": "Filter on starting nodes."},
                    "ids": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "description": (
                            "Seed from these specific node ids directly, instead of (or "
                            "as well as) `where`. `where` filters PROPERTIES, and an id "
                            "is not one -- {\"where\": {\"id\": 7}} matches nothing. "
                            "Combines with `where` as AND."
                        ),
                    },
                    "near": _near_schema(
                        "Rank the starting nodes by similarity to some text instead of "
                        "(or as well as) filtering them. Needs `keep`."),
                    "keep": {
                        "type": "integer",
                        "description": "How many of the highest-scoring starting nodes to keep.",
                    },
                    "boost": {"$ref": "#/$defs/boost"},
                    "rerank": {"$ref": "#/$defs/rerank"},
                },
                # A seed set has to come from SOMEWHERE. Before `near`
                # existed this was `required: ["where"]`; dropping it to
                # let a purely semantic seed through advertised `{}` as
                # a valid start, which is "every node in the graph" --
                # the unbounded result of #47, handed to a model as a
                # legal call. Either way in is fine; neither is not.
                "anyOf": [{"required": ["where"]}, {"required": ["near"]}],
            },
            "hops": {
                "type": "array",
                "description": "Ordered list of traversal steps, applied one after another.",
                "items": {
                    "type": "object",
                    "properties": {
                        "where": {"type": "object", "description": "Filter on the node reached by this hop."},
                        "via": {
                            "anyOf": [{"type": "object"}, {"type": "string"}],
                            "description": (
                                "Filter on edges traversed during this hop. Either an exact-"
                                "match object, e.g. {\"kind\": \"friend\"}, or a bare string "
                                "naming the edge's type directly, e.g. \"friend\" -- shorthand "
                                "for the same filter, on the declared-edge-type fast path."
                            ),
                        },
                        "hops": {
                            "description": "Exact hop count (integer) or [min, max] range.",
                            "anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}],
                        },
                        "direction": {"type": "string", "enum": ["forward", "backward"],
                                     "description": "Which way to walk this hop's edges: "
                                                    "forward follows start->end, backward "
                                                    "follows end->start. Default forward."},
                        "near": _near_schema(
                            "Rank the nodes this hop reaches by similarity to some text. "
                            "Needs `keep`."),
                        "keep": {
                            "type": "integer",
                            "description": "How many of the highest-scoring reached nodes to keep.",
                        },
                        "via_near": _near_schema(
                            "Rank the EDGES this hop traverses by similarity to some text. "
                            "Needs `via_keep`."),
                        "via_keep": {
                            "type": "integer",
                            "description": "How many of the highest-scoring edges to follow.",
                        },
                        "boost": {"$ref": "#/$defs/boost"},
                        "rerank": {"$ref": "#/$defs/rerank"},
                    },
                },
            },
            "aggregates": {
                "type": "object",
                "description": (
                    "Aggregates to compute, keyed by the name each result should "
                    "come back under."
                ),
                "minProperties": 1,
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "fn": {"type": "string", "enum": ["count", "sum", "avg", "min", "max"],
                              "description": "Which aggregate to compute. count needs no "
                                             "`property`; sum/avg/min/max do."},
                        "property": {
                            "type": "string",
                            "description": (
                                "Node property to aggregate. Required for "
                                "sum/avg/min/max; count without it counts the "
                                "matched nodes themselves."
                            ),
                        },
                        "distinct": {
                            "type": "boolean",
                            "description": (
                                "If true, equal property values collapse before "
                                "aggregating. Applies to count/sum/avg only."
                            ),
                        },
                    },
                    "required": ["fn"],
                },
            },
            "group_by": {
                "type": "string",
                "description": (
                    "Property name to group by, read off the same nodes the "
                    "aggregates run over. When given, every aggregate is computed "
                    "once per distinct value of this property (a node missing the "
                    "property forms its own group), and the result is a list of "
                    "objects -- each carrying the group's value under this same "
                    "property name -- instead of a single object."
                ),
            },
        },
        "required": ["start", "aggregates"],
    },
}


def without_rerank(schema: dict) -> dict:
    """A copy of a tool schema with `rerank` taken off every step it
    appears on, and its `$def` with it.

    NO RERANKER, NO PARAMETER. The static schemas carry `rerank` because
    the parsers above read it, and a test derives one from the other so
    the two cannot drift. But a surface with no reranker CONFIGURED must
    not offer it: `traverse_json(graph, spec)` with no policy refuses a
    `rerank` key by name, so advertising it there would be a parameter
    whose handler rejects every use of it -- the defect CLAUDE.md names,
    and the same reasoning that already keeps VECTOR_SEARCH_TOOL_SCHEMA
    out of `tool_schemas()` for a graph that has declared no vectors.

    Lives here rather than in mcp.py because Graph.tool_schemas() needs
    it and core.py must not import the MCP front end -- that module asks
    for an optional SDK by name.

    mcp.py's _with_rerank() strips the parameter too and does NOT call
    this, which is worth knowing before "unifying" them: that one
    MUTATES the schema it was handed and returns it, because its caller
    built the copy already, while this one deep-copies because
    tool_schemas() hands it the module-level constants -- stripping
    those in place would edit TRAVERSE_TOOL_SCHEMA for the whole
    process. It also has no top-level branch, since the only schemas it
    is applied to are the traversal and aggregation ones; the top-level
    `rerank` below belongs to VECTOR_SEARCH_TOOL_SCHEMA, which only
    reaches this function. Two callers with genuinely different
    ownership of the dict, not one rule written twice."""
    copy = deepcopy(schema)
    properties = copy["parameters"]["properties"]
    steps = [properties[name] for name in ("start",) if name in properties]
    if "hops" in properties:
        steps.append(properties["hops"]["items"])
    if "rerank" in properties:          # vector_search: rerank is top level
        properties.pop("rerank")
    for step in steps:
        step.get("properties", {}).pop("rerank", None)
    copy["parameters"].get("$defs", {}).pop("rerank", None)
    return copy
