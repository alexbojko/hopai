"""
Reranking a candidate list, using the reranker client you already have.

Retrieval that works is three stages, not two: retrieve wide and cheap
(a vector distance, a `ts_rank`), then rerank a bounded top-N expensively
and accurately, then keep k. hopai has always had stage one. This module
is stage two, and like `embeddings.py` it is a seam rather than an
implementation: you construct and own the provider client -- keys,
timeouts, base_url, proxies -- and hopai calls one method on it.

    from hopai import Near
    from hopai.rerankers import Rerank
    import cohere

    graph.vector_search(
        Near("summary", text="how do nodes agree?"),
        rerank=Rerank(cohere.ClientV2(), model="rerank-v3.5",
                      document_from='.properties.title + ": "'
                                    ' + (.properties.summary // "")',
                      candidates=50),
        k=10,
    )

ONE METHOD IS THE WHOLE CONTRACT:

    score(query: str, documents: list[str]) -> list[float]

and it is deliberately NOT factored by modality. LanceDB -- the closest
comparable API, and worth reading before this one -- splits it into
`rerank_vector`, `rerank_fts` and `rerank_hybrid`, of which only the
last is mandatory. That is the wrong axis: a reranker reads (query,
document) pairs and cannot tell, or care, whether a candidate arrived
from a cosine, a lexical match, or a fusion of both. The split buys
nothing and adds a runtime trap -- implement `rerank_hybrid`, run a
vector-only search, fail in production. Splitting by modality here
would also make step-wise reranking impossible, since a traversal hop
is neither "vector" nor "fts": it is a node plus how it was reached.

FUSION IS NOT THE RERANKER'S JOB, for the same reason. LanceDB puts
`merge_results()` in the reranker base class and -- per its own docs --
that merge "ignores scores", which makes an RRF rank-arithmetic object
a sibling of a neural cross-encoder behind one interface. Two unrelated
things wearing one name is how a caller ends up with rank fusion where
they asked for a cross-encoder. A Rerank here never sees two candidate
lists; whatever produced the list produced it.

`document_from=` IS A RULE, NOT A DOCUMENT. Nothing about the documents
exists when the query is written, so the parameter holds a jq FILTER
that hopai evaluates once per candidate at execution time -- after the
SQL returns, before the provider call. That is the same ordering
`set_vectors()` keeps for embeddings and for the same reason: a
provider round trip must never happen with a transaction open. The
parameter is spelled `document_from` rather than `document` precisely
because the value is code that looks like data; `document='...'` reads
as though you are handing over the document itself, which is the one
thing you are not doing.

jq, and not a projection language invented here: its own operators
already cover nested fields, list flattening and defaults, and it is a
syntax a model has seen ten thousand times -- rule 2, verbatim. The
binding is an optional extra (`pip install 'hopai[rerankers]'`),
imported lazily, so nothing changes in the base install and a Rerank
that only ever calls score() on documents you built yourself never
needs it.

A filter is compiled ONCE per Rerank and reused across every candidate,
which is a measurement rather than a preference: compiling costs
milliseconds (1.6-2.4ms, jq 1.12) and evaluating costs tens of
microseconds per row, so compiling per candidate would make document
building two orders of magnitude more expensive and grow that cost with
the candidate count instead of paying it once.

SCORES ARE RE-PAIRED BY INDEX, NEVER BY ARRIVAL ORDER. Cohere and
Voyage answer with results SORTED BY RELEVANCE -- their whole job --
so the first result is the best document, not the first document. Zipping
that answer against the documents that were sent produces a plausible,
confidently wrong ranking with no error anywhere, which is the worst
thing this library can produce. Every such result carries `.index` back
into the request, and that is what is read here. A result set that does
not cover every document is refused rather than completed with guesses.

BATCHING IS SOUND HERE only because a relevance score is a property of
the (query, document) PAIR: every provider supported below scores each
pair independently, so splitting 2500 documents into three calls and
merging by position gives the same numbers as one call. A listwise
reranker -- one that ranked documents against each other -- could not
be batched this way, and none of the shapes below is one.

TRANSIENT FAILURES ARE RETRIED, terminal ones are not, and the
classification is `embeddings.py`'s, imported rather than restated:
one policy, one classifier, one place to fix. `retries=`/`backoff=`
carry the same defaults, and they multiply with your client's own --
pick one side rather than paying 3x3.

WHEN THE RETRIES ARE SPENT THE QUERY RAISES. Most retrieval stacks
degrade to the pre-rerank ordering here; this one refuses. A caller who
asked for reranking and quietly received fusion order has a DIFFERENT
ANSWER WITH NO SIGNAL, which is rule 4's case exactly and the hardest
kind of failure for an agent to notice. RerankError keeps the
provider's own exception as `__cause__`, so a caller who genuinely
wants to degrade can catch it and re-run without `rerank=` -- visibly,
in their own code.

NOTHING HERE MAY BLOCK THE EVENT LOOP, and unlike embeddings.py that is
not deferred: reranking is on the READ path, where hopai already has an
async surface (hopai/asyncio.py). So every call has an awaitable twin.
`ascore()` awaits an awaitable client directly -- `cohere.AsyncClientV2()`,
an async callable -- and runs a sync one in `asyncio.to_thread`, which
is legitimate for socket I/O because provider SDKs release the GIL
while blocked on it.

jq itself runs inline, and what makes that defensible is THE PRUNING
BELOW rather than a claim that jq is cheap. Measured (jq 1.12, one
core, `.properties.title` over 50 candidates), the unpruned cost was
~42us per row PLUS ~25us per KB of candidate payload -- because
`input_value()` marshals the WHOLE candidate into jq's value
representation, so a `properties` bag holding a document body was
converted in full to read one title out of it. That is proportional to
the ROW, not to the projection: a 100KB row cost 2.6ms and 50 of them
held the loop's thread for 121ms with the ticker scoring ZERO, the
signature of a `time.sleep` rather than of a slow function. So
`build_documents()` now hands jq only the paths the filter reads
(`jqsafe.paths_read()`), which removes the per-KB term entirely and
leaves the fixed ~10-20us per row -- 0.5-1ms for 50 candidates,
independent of how much text the rows carry.

A THREAD IS STILL NOT THE ANSWER, but not for the reason once written
here ("a C extension holding the GIL"), which was true at one
granularity and false at the other. THE BINDING RELEASES THE GIL
BETWEEN EVALUATIONS AND NOT WITHIN ONE, so both of these hold at the
same time and neither is a caveat on the other:

  - jqsafe.py is right that a runaway filter (`def f: f; f`) can be
    interrupted by NOTHING -- no watchdog thread, no signal, only
    SIGKILL. That is ONE evaluation, and inside one there is no
    boundary to hand the GIL back at. It is why that module refuses
    such filters structurally rather than by timeout, and nothing here
    softens it.
  - a BATCH of small evaluations -- which is precisely what building 50
    documents is -- hands it back 50 times. Measured against an idle
    loop of 778 ticks/s, building 50 unpruned 100KB documents inline
    scored 0 ticks/s (0%) and the same build through
    `asyncio.to_thread` scored 110 (14%). A thread is not nothing here;
    it is 14%.

14% is not the difference between blocking and not: a starved loop that
limps is still a starved loop, and it would cost a thread hop per step
plus a second place for the filter's refusals to surface from. Removing
the work is worth more than moving it, which is what pruning does; what
remains is bounded by MAX_DOCUMENTS below rather than left to grow.

WHAT THIS DELIBERATELY DOES NOT DO: fuse (above), cache (an
application's job, as with embeddings), or carry its own `top_n`.
LanceDB's `top_n` sits beside the query's own `.limit()` and the two
can silently disagree; here `candidates=` bounds the INPUT and k/keep
bounds the OUTPUT, they never overlap, and a `candidates` below the
surrounding k refuses instead of being clamped.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import random
import time
from typing import Any, Optional

# Deliberately shared with embeddings.py rather than restated: hopai has
# ONE retry policy and ONE classifier for provider failures, and a second
# copy is how the two drift into disagreeing about whether a 429 is worth
# another try. Private names, same package -- the sharing is the point.
# `_retryable` carries `_RETRY_NAMES`/`_TERMINAL_NAMES` and `_retry_after`
# carries `_MAX_RETRY_AFTER` with it, so those tables are shared too
# without being named again here (naming them would only be an unused
# import that a reader could mistake for a second copy).
from .embeddings import (
    _MAX_BACKOFF, _retry_after, _retryable,
)


def _is_async_call(fn) -> bool:
    """Whether calling `fn` hands back something to await.

    Answered by inspection rather than by calling: the alternative is
    spending a provider request to find out what kind of client this is.

    NOT imported from embeddings.py, unlike the retry policy above:
    that module answers the same question with a bare
    `inspect.iscoroutinefunction` at each of its binding sites, which
    reports False for a callable OBJECT whose `__call__` is the
    `async def`. Embedder can afford that -- an unrecognized shape
    simply takes its asyncio.to_thread() fallback -- but Rerank
    publishes the answer as `is_async` and refuses score() on it, so a
    wrong False here is a public attribute stating the opposite of the
    truth.

    No `fn is None` guard, unlike the embeddings.py sites this was
    lifted from: _bind() below never hands back a None target -- it
    tests each attribute before it binds to it -- so a guard here would
    be a branch nothing can reach."""
    if inspect.iscoroutinefunction(fn):
        return True
    call = getattr(fn, "__call__", None)          # noqa: B004 -- the OBJECT's own __call__
    return call is not None and inspect.iscoroutinefunction(call)


#: The second place hopai makes a network call, so -- as in
#: embeddings.py -- it gets its own logger, standard library, no handler
#: attached. DEBUG carries every provider call with its size, which is
#: what answers "how many documents did that query cost"; a spent call
#: logs at WARNING as well as raising, because a traversal reranking at
#: every hop may well be caught and retried a level up.
logger = logging.getLogger(__name__)

#: Providers whose rerank endpoint caps the documents per call. Both
#: document ~1000; chunking is ours to do, because a provider refusing
#: 2500 documents should not become the caller's problem and a silent
#: truncation would be worse -- it would drop candidates from the
#: ranking without saying so.
_BATCH_CAPS = {
    "cohere": 1000,
    "voyageai": 1000,
}

#: Anything we cannot identify -- a CrossEncoder, a callable, a local
#: service -- gets the same number rather than embeddings.py's smaller
#: default. A cap exists here to respect a REMOTE limit, not to pace a
#: local model (which batches internally anyway), and every documented
#: remote cap is this one.
_DEFAULT_BATCH = 1000

#: Which module names are recognized, by first dotted segment. Matched
#: by NAME and attribute shape, never isinstance -- an isinstance check
#: needs the import, and that would make `hopai[cohere]` a coupling
#: instead of a convenience. See TestNoProviderIsImported.
_PROVIDERS = frozenset({"cohere", "voyageai", "sentence_transformers"})

#: Same three attempts and same half-second doubling as Embedder, on
#: purpose: two different retry budgets for two provider calls in one
#: query is a thing a caller would have to look up.
_DEFAULT_RETRIES = 2
_DEFAULT_BACKOFF = 0.5

#: How many candidates reach the reranker when nothing says otherwise.
#: 50 is the number the retrieval literature and every provider's own
#: quickstart converge on -- wide enough that reranking has something to
#: fix, bounded enough to price.
_DEFAULT_CANDIDATES = 50

#: How many paths one document may quote about a node it reached. A cap
#: is required rather than optional: a high fan-in node can be reached
#: hundreds of ways, and a document quoting all of them blows the
#: provider's token limit -- as a hard error if you are lucky, as a
#: silent server-side truncation if you are not.
_DEFAULT_MAX_PATHS = 10

#: How many documents ONE reranker call may be built from, and the one
#: number in this module that exists because of a MULTIPLIER rather than
#: a preference.
#:
#: `candidates` bounds the NODES a step offers. Under `per_path=True`
#: the documents are (node, route) pairs, so the count is |routes|,
#: which the GRAPH decides and no option here bounds: a high fan-in node
#: contributes one document per way it was reached. `candidates=500`
#: over rows reached 200 ways is 100,000 documents -- measured at 4.64s
#: of document building, before the provider has been called at all,
#: and on the async path all of it on the event loop's own thread.
#:
#: 5000 is deliberately far above anything a real query asks for: the
#: defaults here (50 candidates, 10 paths) are 500, and 5000 is five
#: calls at the 1000-document batch cap both supported providers
#: document. It refuses rather than truncating, because dropping
#: documents drops the routes a node scored on and changes the ranking
#: with nothing to see -- rule 4, and the same judgement `max_paths`
#: makes VISIBLY for the per-document case.
MAX_DOCUMENTS = 5000

#: `document_from` has no honest default, and a sentinel is what lets
#: `Rerank(client, document='...')` be answered with the parameter's
#: real name instead of Python's "unexpected keyword argument", which
#: names the mistake and not the fix.
#:
#: It carries a `__repr__` because the sentinel is PUBLIC through
#: `help(Rerank)` and `inspect.signature`: a bare `object()` renders as
#: `document_from: str = <object object at 0x7f...>`, which tells a model
#: introspecting the signature the exact opposite of the truth -- that
#: the one required parameter is optional, and that its default is some
#: value it cannot name.
class _Required:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<required>"


_REQUIRED = _Required()

#: Said once, in the two places the same mistake arrives: `document=`
#: (or any near-miss keyword) and a second POSITIONAL argument. Every
#: sibling spec takes its required second argument positionally --
#: `Near("summary", q)`, `Boost("importance", 0.2)`,
#: `Embedder(client, "text-embedding-3-small")` -- so
#: `Rerank(client, '.properties.title')` is what a reader of those
#: writes, and Python's "takes 2 positional arguments but 3 were given"
#: names neither the parameter nor why it is keyword-only.
_DOCUMENT_FROM_IS_A_RULE = (
    "the value is a jq filter EVALUATED once per candidate at execution time, not the "
    "document itself. Nothing about the documents exists when you write the query. "
    "e.g. document_from='.properties.title + \": \" + (.properties.summary // \"\")'"
)


#: `_paths` before anything has asked for it. A distinct object rather
#: than None, because `_WHOLE_ROW` IS None -- "do not prune" and "not
#: worked out yet" are two different answers and conflating them would
#: recompute the parse on every candidate of every query.
_UNCOMPUTED = object()


class RerankError(RuntimeError):
    """A reranker call failed, or answered with something unusable.

    RAISED, NEVER DEGRADED. It would be easy to catch this internally
    and return the pre-rerank ordering -- most retrieval stacks do, and
    the query would keep working. It is also the exact failure this
    library is written against: a caller who asked for reranking and
    quietly got fusion order has a DIFFERENT ANSWER WITH NO SIGNAL, and
    nothing downstream -- least of all an agent reading the results --
    can tell that the expensive stage did not run.

    Raised only after the retries are spent, or immediately when the
    failure is terminal, so seeing one means waiting will not help.

    THE PROVIDER'S OWN EXCEPTION IS KEPT as `__cause__`, mirroring
    EmbeddingError, which is what makes degrading a decision a caller
    can take in their own code where it is visible:

        try:
            hits = graph.vector_search(near, rerank=rerank, k=10)
        except RerankError as failed:
            logger.warning("reranking unavailable: %s", failed.__cause__)
            hits = graph.vector_search(near, k=10)      # fusion order, on purpose"""


def _provider(client: Any) -> Optional[str]:
    """Which provider family a client belongs to, by module name.

    The same judgement `embeddings._provider()` makes, against this
    module's own table: `type(client).__module__` is 'cohere.client_v2'
    or 'voyageai.client' or similar, so the first dotted segment is the
    distribution."""
    # The `or ""` only keeps .split() off a None; no falsy __module__ can
    # produce a root that is a provider name, so the literal is
    # unobservable -- the same equivalent mutant embeddings._provider()
    # records.
    root = (type(client).__module__ or "").split(".")[0]
    return root if root in _PROVIDERS else None


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _number(value, owner: str, where: str) -> float:
    """One relevance score, coerced and checked.

    NaN and Infinity are refused rather than sorted: one NaN score turns
    a ranked list into arbitrary order with nothing to see in the
    output, which is the same reason _clean_vector() refuses them."""
    if isinstance(value, (bool, str)) or value is None:
        raise RerankError(
            f"{owner}: {where} came back as {type(value).__name__} where a relevance "
            f"score was expected"
        )
    try:
        score = float(value)                       # numpy scalars answer to this
    except (TypeError, ValueError) as exc:
        raise RerankError(
            f"{owner}: {where} came back as {type(value).__name__} where a relevance "
            f"score was expected"
        ) from exc
    if not math.isfinite(score):
        raise RerankError(
            f"{owner}: {where} came back as {score!r} -- a non-finite score cannot be "
            f"ranked, and sorting on it would return an arbitrary order"
        )
    return score


def _plain_scores(raw, expected: int, owner: str) -> list:
    """A provider that answers in the ORDER IT WAS ASKED: a cross-encoder's
    `predict`, a callable, a `.score` object.

    Length is checked here rather than at the caller, exactly as
    `_as_vectors()` checks it: a provider returning fewer scores than
    documents would otherwise pair scores with the wrong candidates, and
    a mis-paired score is a silently reordered result set."""
    if hasattr(raw, "tolist"):                     # numpy, torch
        raw = raw.tolist()
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise RerankError(
            f"{owner}: the reranker returned {type(raw).__name__} where a sequence of "
            f"{expected} score(s) was expected"
        )
    scores = [_number(value, owner, f"score {i}") for i, value in enumerate(raw)]
    if len(scores) != expected:
        raise RerankError(
            f"{owner}: sent {expected} document(s) and got {len(scores)} score(s) back -- "
            f"refusing rather than pairing scores with the wrong candidates"
        )
    return scores


def _get(item, name: str):
    """A field of one result object, however the SDK spells it.

    Attribute first (every native client returns objects), then mapping
    -- a raw JSON body, or a thin wrapper that never modelled the
    response, reaches here as dicts."""
    if hasattr(item, name):
        return getattr(item, name)
    if isinstance(item, dict):
        return item.get(name)
    return None


def _indexed_scores(raw, expected: int, owner: str) -> list:
    """A provider that answers with `.results`, SORTED BY RELEVANCE:
    Cohere, Voyage.

    This is the function that keeps this module honest. The results come
    back best-first and each one carries `.index` -- its position in the
    documents that were SENT -- so reading them in arrival order pairs
    every score with the wrong candidate and produces a ranking that is
    plausible, confidently wrong, and reported by nothing. Every score
    is therefore placed by its own index, and a set that does not cover
    every document refuses rather than inventing the rest."""
    results = _get(raw, "results")
    if results is None:
        results = raw                              # a thinner wrapper: the list itself
    scores: list = [None] * expected
    seen = 0
    for item in results:
        index = _get(item, "index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RerankError(
                f"{owner}: a result came back with index={index!r} -- results are sorted "
                f"by relevance, so without a usable index there is no way to tell which "
                f"document a score belongs to"
            )
        if not 0 <= index < expected:
            raise RerankError(
                f"{owner}: a result points at document {index}, but only {expected} "
                f"were sent"
            )
        if scores[index] is not None:
            raise RerankError(
                f"{owner}: two results both point at document {index} -- one of them "
                f"belongs to a document that would silently keep no score"
            )
        value = _get(item, "relevance_score")
        if value is None:
            value = _get(item, "score")
        scores[index] = _number(value, owner, f"the score for document {index}")
        seen += 1
    if seen != expected:
        missing = [i for i, score in enumerate(scores) if score is None]
        raise RerankError(
            f"{owner}: sent {expected} document(s) and got {seen} score(s) back, leaving "
            f"{len(missing)} unscored (document {missing[0]} first) -- a client configured "
            f"with top_n truncates the answer; drop it and let k/keep do the truncating, "
            f"which is the only place hopai can see it happen"
        )
    return scores


class Rerank:
    """A reranker client, plus the rule that builds each document.

        Rerank(cohere.ClientV2(), model="rerank-v3.5",
               document_from='.properties.title + ": " + (.properties.summary // "")',
               candidates=50)

        Rerank(voyageai.Client(), model="rerank-2", document_from='.properties.body')
        Rerank(CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
               document_from='.properties.title')
        Rerank(lambda query, documents: my_service.rank(query, documents),
               document_from='.properties.title')

    ONE METHOD IS THE WHOLE CONTRACT, and every client above is only a
    spelling of it:

        score(query: str, documents: list[str]) -> list[float]

    ONE FLOAT PER DOCUMENT, IN THE ORDER THE DOCUMENTS WERE GIVEN. A
    shorter list, a longer one, or an answer sorted by relevance without
    an index to put it back is refused rather than paired up -- a
    mis-paired score is a plausible, confidently wrong ranking that
    nothing reports. (Cohere's and Voyage's own answers ARE sorted by
    relevance, which is why their `.index` is read; see
    `_indexed_scores`.) Higher means more relevant; the scale is the
    reranker's own and is never rescaled here.

    A RERANKER REQUIRES A TEXT QUERY, and that is what reranking IS
    rather than a gap to close later: a cross-encoder scores a query
    against a document by READING BOTH, and a bare embedding is not
    something it can read. So `Near("summary", [0.1, ...])` combined
    with `rerank=` refuses, naming `Near("summary", text="...")` as the
    rewrite -- whose embedding then comes from the field's own `embed=`,
    so the dense stage and the rerank stage see the same query. (The
    refusal lives where the two specs meet, not here; this docstring is
    where it is explained.)

    document_from:  a jq filter, evaluated once per candidate at
                    execution time -- a RULE, not a document. It runs
                    against each candidate's JSON (the dict
                    vector_search() already returns) and MUST evaluate
                    to one non-empty string. One whole candidate, so
                    there is nothing left to guess about the shape:

                        {"id": 7,
                         "properties": {"title": "Raft",
                                        "summary": "a consensus protocol"},
                         "similarity": 0.81,
                         "similarities": {"summary": 0.81},
                         "boosts": {}}

                    -- plus "paths" at a traversal hop. So
                    '.properties.title' is "Raft" and
                    '.properties.title + ": " + .properties.summary' is
                    "Raft: a consensus protocol". Inline it at the call
                    site; `document_from=doc` hides the only part a
                    reader needs to see.

    candidates:     how many hits reach the reranker, before k/keep
                    truncates. An INPUT bound; k is the output bound,
                    and the two never overlap.

    per_path:       False (default) is one call per distinct NODE, whose
                    document may quote every path that reached it;
                    True is one call per (node, path), and the node's
                    score is the MAX over its paths. See below.

    max_paths:      how many paths one document may quote. A visible cap
                    rather than a silent truncation. Not consulted under
                    per_path=True, where a document carries exactly one
                    path by construction -- there the paths become
                    DOCUMENTS instead, which is what MAX_DOCUMENTS
                    bounds.

    model:          required for providers that take one, refused for
                    those that do not -- the judgement Embedder makes,
                    for the same reason: a silently chosen model is a
                    silently different ranking, and the caller cannot
                    see it happened.

    batch_size:     documents per provider call; defaults to the
                    provider's own cap.

    retries/backoff: `embeddings.py`'s policy, same defaults, same
                    full-jitter behaviour. Yours and hopai's MULTIPLY.

    FAN-IN, and why the default is the cheap one. `core.py` already
    settled the analogous question for Near at a hop -- rank AFTER
    deduplication, because a cosine is a property of the node's vector
    and does not change with the walk that arrived at it. Reranking
    breaks that premise only when the document READS the path, so the
    default follows the precedent: one call per distinct node, cost
    |nodes|, deterministic, and the reranker sees every route at once
    rather than one at a time. `per_path=True` is for when a node
    genuinely reads differently depending on the route that found it,
    and it is opt-in precisely because it costs |triples| documents
    instead of |nodes|.

    EITHER WAY FAN-IN IS PRESERVED: a node survives or is dropped as a
    unit, so every in-edge from a surviving parent is reported exactly
    as before. per_path=True takes the MAX and not the sum or the mean,
    so one strong route is enough to keep a node -- the same "any valid
    parent counts" semantics `test_fan_in_both_parents_preserved`
    exists to protect.

    AND IT IS THE ONE MULTIPLIER HERE, so it is bounded: `candidates`
    counts NODES, per_path=True bills (node, route) pairs, and how many
    routes reach a node is the graph's answer rather than an option's.
    `MAX_DOCUMENTS` (5000) is where that refuses -- naming the number,
    both knobs and the fix -- because a 100,000-document step measured
    4.64s of document building before a single provider call, and
    truncating instead would drop the routes a node scored on and
    change the ranking with nothing to see."""

    def __init__(self, client: Any, *misplaced, document_from: str = _REQUIRED,
                 candidates: int = _DEFAULT_CANDIDATES, per_path: bool = False,
                 max_paths: int = _DEFAULT_MAX_PATHS, model: Optional[str] = None,
                 batch_size: Optional[int] = None, retries: int = _DEFAULT_RETRIES,
                 backoff: float = _DEFAULT_BACKOFF, **unexpected):
        # The filter stays keyword-only -- positionally it would sit
        # where a model name reads naturally, and `Rerank(client,
        # "rerank-v3.5")` would compile a model name as a jq filter. But
        # every sibling spec DOES take its required second argument
        # positionally, so the mistake is the expected one and deserves
        # the same named sentence `document=` gets rather than Python's
        # "takes 2 positional arguments but 3 were given".
        if misplaced:
            raise TypeError(
                f"Rerank: document_from= is keyword-only -- pass it by name, "
                f"document_from={misplaced[0]!r}, and not positionally where a model "
                f"name reads naturally. It is keyword-only because {_DOCUMENT_FROM_IS_A_RULE}"
            )
        # `document=` is the mistake worth answering by name: it is what
        # the parameter would be called if the value were a document,
        # and Python's own error for a misspelled keyword names the
        # misspelling without ever naming the fix. `documents=`,
        # `text_from=`, `doc_from=` are the SAME mistake reaching for the
        # same parameter, so they are answered the same way rather than
        # falling through to a list of names.
        misnamed = sorted(name for name in unexpected
                          if "doc" in name.lower() or "text" in name.lower())
        if misnamed:
            raise TypeError(
                f"Rerank: the parameter is document_from=, not {misnamed[0]}= -- "
                f"{_DOCUMENT_FROM_IS_A_RULE}"
            )
        if unexpected:
            raise TypeError(
                f"Rerank: unexpected keyword argument(s) {sorted(unexpected)}")
        if document_from is _REQUIRED:
            raise TypeError(
                "Rerank: document_from= is required -- it is the jq filter that builds "
                "each candidate's document, e.g. document_from='.properties.title + "
                "\": \" + (.properties.summary // \"\")'. There is no default, because "
                "guessing which property holds the text would rank against the wrong "
                "words and report nothing"
            )
        # A callable is answered separately because the NAME invites it
        # -- `document_from=make_document` reads like a hook -- and
        # because `Boost(key=...)` genuinely accepts one, so the reader
        # is not guessing wildly. Naming the rewrite is what turns "not
        # a string" into something a caller can act on.
        if callable(document_from):
            raise ValueError(
                f"Rerank: document_from= is a jq filter STRING, not a function -- got "
                f"{getattr(document_from, '__name__', type(document_from).__name__)}. "
                f"(Boost(key=...) does take a callable; this does not, because the "
                f"filter text is what hopai quotes back in every error about a "
                f"document.) e.g. document_from='.properties.title'"
            )
        if not isinstance(document_from, str) or not document_from.strip():
            raise ValueError(
                f"Rerank: document_from= must be a non-empty jq filter string, got "
                f"{document_from!r}"
            )
        self.document_from = document_from
        if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates < 1:
            raise ValueError(
                f"Rerank: candidates must be a positive integer -- it is how many hits "
                f"reach the reranker before k/keep truncates, got {candidates!r}")
        self.candidates = candidates
        # Checked, not coerced, for the reason `all`/`detach`/`replace`
        # are: per_path="false" is TRUE in Python, and a JSON boolean
        # arriving as the string "false" is an ordinary tool-call
        # failure -- coercing would let it mean one provider call per
        # path, which is the expensive one.
        if not isinstance(per_path, bool):
            # The "'false' is truthy" sentence is the reason a STRING is
            # refused; against per_path=1 it explains a case that did not
            # happen, which reads as the library answering someone else's
            # question.
            why = ("a string is not coerced here, because 'false' is truthy in Python and "
                   "would silently select the per-path mode") if isinstance(per_path, str) \
                else ("nothing is coerced here, because a truthy value would silently "
                      "select the per-path mode")
            raise TypeError(
                f"Rerank: per_path must be True or False, got {per_path!r} -- {why}, "
                f"which costs one provider call per path")
        self.per_path = per_path
        if not isinstance(max_paths, int) or isinstance(max_paths, bool) or max_paths < 1:
            raise ValueError(
                f"Rerank: max_paths must be a positive integer -- it caps how many paths one "
                f"document may quote, got {max_paths!r}")
        self.max_paths = max_paths
        self.client = client
        self.model = model
        self.provider = _provider(client)
        if batch_size is not None and (
                not isinstance(batch_size, int) or isinstance(batch_size, bool)
                or batch_size < 1):
            raise ValueError(
                f"Rerank: batch_size must be a positive integer, got {batch_size!r}")
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError(
                f"Rerank: retries must be a non-negative integer (0 disables retrying), "
                f"got {retries!r}")
        if not isinstance(backoff, (int, float)) or isinstance(backoff, bool) \
                or not math.isfinite(backoff) or backoff <= 0:
            raise ValueError(
                f"Rerank: backoff must be a positive number of seconds, got {backoff!r}")
        self.retries = retries
        self.backoff = float(backoff)
        self.batch_size = batch_size or _BATCH_CAPS.get(self.provider or "", _DEFAULT_BATCH)
        # Bound eagerly, like Embedder: a client hopai cannot call should
        # be refused on the line that names it, not on the first query,
        # which may be a long way away and much later.
        self._invoke, self._shape, target = _bind(client, self.provider, model)
        #: Whether the bound provider call hands back something to await.
        #: Read in score() to refuse BEFORE the call rather than after --
        #: calling it would build a coroutine nobody awaits.
        #:
        #: `_is_async_call` above and not `iscoroutinefunction`, which
        #: reports False for a callable OBJECT whose `__call__` is the
        #: `async def` -- ascore() still works there (it awaits the
        #: awaitable the thread built), but it pays a thread it does not
        #: need and the public attribute states the opposite of the truth.
        self.is_async = _is_async_call(target)
        #: The compiled filter. Compiled HERE when the jq binding is
        #: present, for the reason _bind() refuses a bad client here:
        #: 'properties.title' -- no leading dot, the single most likely
        #: thing to write -- would otherwise construct fine and fail at
        #: query time, a long way from the line that named it. It stays
        #: lazy only when the extra is absent, so a Rerank that only ever
        #: calls score() on documents the caller built themselves is
        #: still constructible on a base install.
        self._program = None
        self._validated: set = set()
        #: The paths `document_from` reads, or `_WHOLE_ROW`. Computed on
        #: first use rather than here: `_projection_paths()` needs
        #: `hopai.jqsafe`, and __init__ must stay constructible for a
        #: caller who only ever calls score() on their own documents.
        self._paths = _UNCOMPUTED
        try:
            import jq
        except ImportError:
            pass
        else:
            self._compile(jq)

    def __repr__(self) -> str:
        """What this Rerank is, in the shape it was written.

        Calibrated against `Near.__repr__`: the arguments that decide the
        answer, none of the ones that do not. `__name__` before the type
        name because a plain callable IS the client here -- `type(...)`
        reports 'function' for every one of them, which identifies
        nothing. `document_from` is included because it is the field you
        stare at when a ranking looks wrong, and because this repr is the
        `owner` prefix of every runtime message this module raises: the
        filter that built the document is then quoted beside the
        complaint about it."""
        parts = [self.provider
                 or getattr(self.client, "__name__", type(self.client).__name__)]
        if self.model:
            parts.append(f"model={self.model!r}")
        parts.append(f"document_from={self.document_from!r}")
        parts.append(f"candidates={self.candidates}")
        if self.per_path:
            parts.append("per_path=True")
        return f"Rerank({', '.join(parts)})"

    # -----------------------------------------------------------------
    # The contract
    # -----------------------------------------------------------------

    def score(self, query: str, documents: list) -> list:
        """Relevance of each document to the query, in the order the
        documents were given.

        The whole interface. Which step of a traversal produced these
        candidates, whether they came from a cosine or a lexical match,
        how they were deduplicated -- none of it reaches here, and that
        is what lets hopai build documents differently at a seed step
        and at a mid-chain hop without any reranker knowing."""
        owner, documents = self._prepare(query, documents, "score")
        if self.is_async:
            raise TypeError(
                f"{owner}: this client's rerank call is async -- await ascore() instead, "
                f"or pass the provider's synchronous client")
        if not documents:
            return []
        scores: list = []
        for chunk in _chunks(documents, self.batch_size):
            logger.debug("%s: scoring %d document(s)", owner, len(chunk))
            raw = self._attempt(query, chunk, owner, len(scores))
            if inspect.isawaitable(raw):
                # A client whose method is not `async def` but hands back
                # an awaitable anyway. Closed rather than dropped, so the
                # refusal is not followed by a "coroutine was never
                # awaited" warning pointing at hopai's own frame.
                getattr(raw, "close", lambda: None)()
                raise TypeError(
                    f"{owner}: this client returned an awaitable -- await ascore() "
                    f"instead, or pass the provider's synchronous client")
            scores.extend(self._shape(raw, len(chunk), owner))
        return scores

    async def ascore(self, query: str, documents: list) -> list:
        """score(), without blocking the event loop.

        An awaitable client is awaited directly -- that is the shape
        `cohere.AsyncClientV2()` and an async callable already have, and
        awaiting it here is the "resolve before you open" move
        set_vectors() makes for embeddings. A synchronous client runs in
        `asyncio.to_thread`, which is legitimate for exactly this: the
        provider SDK is blocked on a socket and releases the GIL while
        it waits. (Document building stays inline, and that is now a
        measurement rather than a claim about the GIL: pruning took the
        per-KB term out of it, so a build is tens of microseconds a row
        and bounded by MAX_DOCUMENTS. jq's binding does release the GIL
        between evaluations, so a thread would recover ~15% of the loop
        across a batch -- which is not the difference between blocking
        and not. See the module docstring.)"""
        owner, documents = self._prepare(query, documents, "ascore")
        if not documents:
            return []
        scores: list = []
        for chunk in _chunks(documents, self.batch_size):
            logger.debug("%s: scoring %d document(s)", owner, len(chunk))
            raw = await self._aattempt(query, chunk, owner, len(scores))
            scores.extend(self._shape(raw, len(chunk), owner))
        return scores

    def _prepare(self, query: str, documents, method: str) -> tuple:
        """Validate the pair both entry points take, and name the call.

        A reranker reads the query as TEXT, so an empty one scores every
        document against nothing -- caught here, where the message can
        say so, rather than as a shrug from the provider."""
        owner = f"{self!r}.{method}"
        if not isinstance(query, str):
            raise TypeError(
                f"{owner}: the query must be text -- a reranker scores a query against a "
                f"document by reading both, and {type(query).__name__} is not something "
                f"it can read")
        if not query.strip():
            raise ValueError(
                f"{owner}: the query is empty or whitespace, so every document would be "
                f"scored against nothing")
        # The same guard `_plain_scores()` puts on the ANSWER, missing
        # here on the input: a str is iterable, so list("a document")
        # silently becomes ten single-character documents -- and the
        # complaint that comes back ("document 1 is empty or whitespace")
        # is about the space, which names nothing a caller can act on.
        if isinstance(documents, (str, bytes)):
            raise TypeError(
                f"{owner}: documents= is a single {type(documents).__name__} "
                f"({documents!r}), and iterating it would score one CHARACTER per "
                f"document -- a bare string is ONE document, so pass [document]")
        documents = list(documents)
        for index, document in enumerate(documents):
            if not isinstance(document, str):
                raise TypeError(
                    f"{owner}: document {index} is {type(document).__name__}, not a string "
                    f"-- build_documents() turns candidates into strings; score() takes "
                    f"the strings")
            if not document.strip():
                raise ValueError(
                    f"{owner}: document {index} is empty or whitespace, which scores that "
                    f"candidate against nothing and silently changes the ranking")
        return owner, documents

    # -----------------------------------------------------------------
    # Retry: embeddings.py's policy, once per transport
    # -----------------------------------------------------------------

    def _delay(self, failure: BaseException, attempt: int) -> float:
        """How long to wait before retry `attempt`.

        Full jitter rather than exact doubling, for the reason
        Embedder._attempt() states: several reranks failing at the same
        instant (a traversal reranking at every hop, N concurrent
        queries) would otherwise retry in lockstep and hit a
        rate-limited provider with the same synchronised burst that
        caused the 429. `Retry-After` wins when the provider sent one --
        the only number here that is not a guess."""
        window = min(self.backoff * (2 ** (attempt - 1)), _MAX_BACKOFF)
        delay = _retry_after(failure)
        return random.uniform(0, window) if delay is None else delay

    def _attempt(self, query: str, chunk: list, owner: str, done: int):
        """One provider call, retried while the failure looks transient.

        The async twin below is the same loop with `await
        asyncio.sleep`. They are written out twice on purpose: the sleep
        IS the difference, and a shared driver would either block the
        loop or make the sync path spin up an event loop to stand
        still."""
        attempts = self.retries + 1
        failure = None
        for attempt in range(attempts):
            if failure is not None:
                # Backing off BEFORE the retry rather than after the
                # failure keeps the loop bound the only thing deciding
                # how many calls happen -- see Embedder._attempt().
                delay = self._delay(failure, attempt)
                logger.warning("%s: %s: %s -- retrying in %.2fs (attempt %d of %d)",
                               owner, type(failure).__name__, failure, delay,
                               attempt + 1, attempts)
                time.sleep(delay)
            try:
                return self._invoke(query, chunk)
            except RerankError:
                raise
            except Exception as exc:                      # provider-side failure
                if not _retryable(exc):
                    self._give_up(owner, done, attempt + 1, exc)
                failure = exc
        self._give_up(owner, done, attempts, failure)

    async def _aattempt(self, query: str, chunk: list, owner: str, done: int):
        """_attempt(), awaiting instead of blocking."""
        attempts = self.retries + 1
        failure = None
        for attempt in range(attempts):
            if failure is not None:
                delay = self._delay(failure, attempt)
                logger.warning("%s: %s: %s -- retrying in %.2fs (attempt %d of %d)",
                               owner, type(failure).__name__, failure, delay,
                               attempt + 1, attempts)
                await asyncio.sleep(delay)
            try:
                if self.is_async:
                    return await self._invoke(query, chunk)
                raw = await asyncio.to_thread(self._invoke, query, chunk)
                # A sync method that hands back an awaitable anyway: the
                # thread only built it, so awaiting it here is what
                # actually runs it -- and it runs on this loop, not in
                # the thread.
                return await raw if inspect.isawaitable(raw) else raw
            except RerankError:
                raise
            except Exception as exc:
                if not _retryable(exc):
                    self._give_up(owner, done, attempt + 1, exc)
                failure = exc
        self._give_up(owner, done, attempts, failure)

    @staticmethod
    def _give_up(owner: str, done: int, attempts: int, exc: BaseException):
        """The one place a spent call is reported, so the log line and
        the exception can never disagree about how far it got.

        It RAISES. Returning the scores gathered so far, or None for the
        caller to interpret, is the silent degradation this module
        refuses -- see RerankError."""
        logger.warning("%s: reranker call failed after %d scored, %d attempt(s) (%s: %s)",
                       owner, done, attempts, type(exc).__name__, exc)
        raise RerankError(
            f"{owner}: the reranker call failed after {attempts} attempt(s) "
            f"({type(exc).__name__}: {exc}) -- refusing to fall back to the pre-rerank "
            f"order, which would be a different answer with no signal. Catch RerankError "
            f"and re-run without rerank= if that is what you want"
        ) from exc

    # -----------------------------------------------------------------
    # Documents: the rule, evaluated
    # -----------------------------------------------------------------

    def build_documents(self, candidates: list, *, trusted: bool = False,
                        fields=None) -> list:
        """Each candidate's document, by evaluating `document_from`
        against it.

        `candidates` is a LIST -- the dicts the read path already
        produces, {"id", "properties", "similarity", "similarities",
        "boosts"}, plus "paths" at a traversal hop. They are never
        mutated: a candidate whose paths exceed `max_paths` is COPIED
        with the cap applied, so the caller's own results still carry
        every path after the reranker has been given a bounded view.

        THE SAFE BEHAVIOUR IS THE DEFAULT, and `trusted=True` is the
        opt-in -- the polarity `allow_vectors=`, `all=` and `detach=`
        already keep, where the DANGEROUS thing is the one you have to
        ask for. The other way round (an `untrusted=True` you must
        remember) fails open: forget it on a filter that arrived from a
        model or over the wire and `env.HOME` runs, with the result
        POSTed to a third party as a document.

        By default the filter is held to `hopai.jqsafe`'s total subset
        -- a grammar in which `env`, `$ENV`, `input` and unbounded
        recursion do not parse at all -- optionally narrowed to the
        paths in `fields=`. `trusted=True` gets the full language, and
        is for a human writing jq in their own Python process: they
        already run arbitrary code there, so the subset would restrict
        nothing they cannot already do.

        A filter that errors, or evaluates to something other than one
        non-empty string, RAISES, naming the filter and the candidate.
        Falling back to an empty document would score that candidate
        against nothing and silently change the ranking -- exactly the
        answer nobody can see is wrong."""
        # The same class of mistake as a bare `str` reaching score(): one
        # candidate is a dict, a dict iterates its KEYS, and the filter
        # then fails on the string "id" with jq's own complaint --
        # "Cannot index string with string" -- which blames the filter
        # for the call shape.
        if isinstance(candidates, (dict, str, bytes)):
            raise TypeError(
                f"{self!r}.build_documents(): candidates is a LIST of candidates, and "
                f"this is one {type(candidates).__name__} -- iterating it yields its "
                f"keys, not candidates, so the filter would run against the string 'id'. "
                f"Pass [candidate]")
        candidates = list(candidates)
        self._bound(len(candidates))
        program = self._compiled(trusted, fields)
        paths = self._projection()
        documents = []
        for candidate in candidates:
            capped = self._capped(candidate)
            view = None if paths is _WHOLE_ROW or not isinstance(capped, dict) \
                else _pruned(capped, paths)
            documents.append(_evaluate(program, capped, self.document_from, view))
        return documents

    def _bound(self, documents: int) -> None:
        """Refuse a call that would build more documents than
        MAX_DOCUMENTS.

        HERE and not at construction, because the number is not knowable
        at construction: `candidates` bounds the NODES, and under
        `per_path=True` the documents are (node, route) pairs whose
        count the graph decides. This is the one place that sees the
        real total, on both drivers and both front ends.

        It refuses instead of truncating for the reason every other
        refusal in this module does: the documents that fell off the end
        are routes a node would have scored on, so the ranking would
        differ with nothing to show for it."""
        if documents <= MAX_DOCUMENTS:
            return
        per_path = (
            f" -- per_path=True builds one document per (node, route), so a step whose "
            f"nodes were reached many ways multiplies candidates={self.candidates} by a "
            f"number the GRAPH decides, not by max_paths={self.max_paths}"
        ) if self.per_path else (
            f" -- candidates={self.candidates} is the input bound, and it is above the "
            f"limit on its own"
        )
        raise ValueError(
            f"{self!r}.build_documents(): {documents} documents for one reranker call, "
            f"over the {MAX_DOCUMENTS} limit{per_path}. Lower candidates=, or drop "
            f"per_path=False's opt-in, or rerank a later step where fewer nodes survive. "
            f"Refusing rather than truncating, because the documents that fell off the "
            f"end are routes a node would have scored on")

    def _projection(self):
        """The paths `document_from` reads, computed once.

        Cached beside `_program` and for the same reason: it is one
        parse of a filter bounded by `jqsafe.MAX_LENGTH`, and paying it
        per candidate would put it back on the query path this exists to
        take work off."""
        if self._paths is _UNCOMPUTED:
            self._paths = _projection_paths(self.document_from)
        return self._paths

    def _capped(self, candidate):
        """The candidate a document is built from, with `paths` bounded.

        A copy, never a mutation: the caller's result rows keep every
        path they were reached by -- the cap is about what one provider
        call is allowed to quote, not about what the graph found."""
        paths = candidate.get("paths") if isinstance(candidate, dict) else None
        if not isinstance(paths, list) or len(paths) <= self.max_paths:
            return candidate
        capped = dict(candidate)
        capped["paths"] = paths[:self.max_paths]
        return capped

    def _compiled(self, trusted: bool, fields):
        """The compiled filter, built once and reused.

        Measured, not assumed: compiling costs 1.6-2.4ms against tens of
        microseconds to evaluate one row, so a compile per candidate
        would be almost the entire cost of building 50 documents and
        would scale with the candidate count instead of being paid
        once. Normally it is already compiled by __init__; this path
        only runs when the extra was absent at construction."""
        if not trusted:
            self._validate(fields)
        if self._program is None:
            self._compile(_jq())
        return self._program

    def _compile(self, jq):
        """Compile `document_from`, or refuse naming the filter.

        Called from __init__ when the binding is importable, so the
        refusal lands on the line that wrote the filter. jq's own message
        for the most likely mistake ('properties.title', no leading dot)
        is a parse error about a token, so the dot is spelled out here."""
        try:
            self._program = jq.compile(self.document_from)
        except Exception as exc:
            raise ValueError(
                f"Rerank: document_from={self.document_from!r} is not a valid jq "
                f"filter -- {exc}. jq paths start with a dot, e.g. "
                f"'.properties.title'"
            ) from exc
        return self._program

    def _validate(self, fields):
        """Hold a filter to the safe subset, once per (fields) it is
        asked about.

        Imported here rather than at module scope so this module stays
        importable -- and a trusted filter stays free -- whether or not
        the gate is reachable. It is a GATE, not an interpreter: what it
        accepts is still executed by libjq, because reimplementing jq's
        own semantics for `//`, `?` and null handling is how an edge
        case ends up subtly different from the language a caller tested
        against."""
        key = None if fields is None else tuple(sorted(fields))
        if key in self._validated:
            return
        from . import jqsafe
        jqsafe.validate(self.document_from, fields=fields, owner="document_from")
        self._validated.add(key)


#: What `inspect.signature(Rerank)` and `help(Rerank)` report.
#:
#: `__init__` really does take `*misplaced` and `**unexpected`, but both
#: are REFUSAL CHANNELS -- they exist only so a mistake can be answered
#: by name instead of by Python's generic message -- and neither is a
#: parameter anyone may pass. Left in the rendered signature they say the
#: opposite of the truth: `*misplaced` advertises positional arguments to
#: a reader whose first question is whether `document_from` is one, which
#: is the exact confusion the keyword-only rule exists to prevent.
#:
#: This matters more here than it would in most libraries: a model
#: discovers this API by introspecting it, so the signature IS the
#: documentation, and rule 2 says an LLM must get it right with no
#: custom instructions. Overriding it keeps the refusals AND an honest
#: contract; the alternative -- dropping the channels to clean up the
#: signature -- would trade a dozen named errors for a tidier repr.
Rerank.__signature__ = inspect.Signature([
    inspect.Parameter("client", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Any),
    inspect.Parameter("document_from", inspect.Parameter.KEYWORD_ONLY,
                      default=_REQUIRED, annotation=str),
    inspect.Parameter("candidates", inspect.Parameter.KEYWORD_ONLY,
                      default=_DEFAULT_CANDIDATES, annotation=int),
    inspect.Parameter("per_path", inspect.Parameter.KEYWORD_ONLY,
                      default=False, annotation=bool),
    inspect.Parameter("max_paths", inspect.Parameter.KEYWORD_ONLY,
                      default=_DEFAULT_MAX_PATHS, annotation=int),
    inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY,
                      default=None, annotation=Optional[str]),
    inspect.Parameter("batch_size", inspect.Parameter.KEYWORD_ONLY,
                      default=None, annotation=Optional[int]),
    inspect.Parameter("retries", inspect.Parameter.KEYWORD_ONLY,
                      default=_DEFAULT_RETRIES, annotation=int),
    inspect.Parameter("backoff", inspect.Parameter.KEYWORD_ONLY,
                      default=_DEFAULT_BACKOFF, annotation=float),
])


def _jq():
    """The jq binding, asked for by name when it is missing.

    Lazy and by name, the way mcp.py asks for the MCP SDK: reranking is
    an optional extra, and an ImportError reading "No module named 'jq'"
    tells a caller what is absent without telling them what to install."""
    try:
        import jq
    except ImportError as exc:
        raise ImportError(
            "document_from= is a jq filter, and the jq binding is an optional extra -- "
            "pip install 'hopai[rerankers]'"
        ) from exc
    return jq


#: What `_projection_paths()` answers when the filter's reads cannot be
#: enumerated -- a bare `.`, or anything `jqsafe` will not parse. Not
#: None, because None is a legitimate answer nowhere here and a caller
#: reading `paths is None` as "no paths" would prune to `{}`.
_WHOLE_ROW = None


def _projection_paths(document_from: str):
    """The paths a filter reads, or `_WHOLE_ROW` when it must be handed
    the candidate entire.

    THE PERFORMANCE FIX, and the reason it is safe. `input_value()`
    marshals whatever it is given into jq's own value representation, so
    the cost of building one document was proportional to the CANDIDATE
    -- ~25us per KB of it -- rather than to the projection: measured,
    `.properties.title` cost exactly as much as `.properties.body` on a
    row whose body was 100KB, because both handed jq the same 100KB.
    Fifty such rows held the event loop's thread for 121ms with an
    independent ticker scoring zero. `fields=` narrows what a filter may
    READ; it never narrowed what jq was HANDED.

    So the filter is asked what it reads, and only that is handed over.
    `jqsafe.paths_read()` is the same answer the `fields=` allowlist is
    enforced on, which is what makes this sound rather than an
    optimization on top of a guess: it OVER-reports by design, and an
    over-reported path only means a bigger view than necessary. It is
    also already differentially tested against libjq's own behaviour
    (`TestPathsReadAgreesWithWhatLibjqRuns` runs every accepted program
    against a row of per-field sentinels and asserts nothing reaches the
    output that was not reported) -- which is precisely the property
    pruning needs.

    CORRECT AND SLOW BEATS FAST AND WRONG, so every case the analysis
    cannot see through declines to prune:

      - a filter reading `.` gets the whole row;
      - a filter `jqsafe` cannot parse at all -- which is every
        `trusted=True` filter using the full language -- gets the whole
        row, since there is then no reported set to trust;
      - and `_pruned()` below stops descending the moment the value it
        would descend into is not an object."""
    from . import jqsafe
    try:
        read = jqsafe.paths_read(document_from)
    except Exception:
        # Deliberately bare: a filter outside the subset is not this
        # function's to refuse -- build_documents() has already run
        # _validate() on it when it matters, and a trusted one is
        # allowed to be unparseable here. The only answer this owes is
        # "I cannot see it", which is the whole row.
        return _WHOLE_ROW
    if "." in read:
        return _WHOLE_ROW
    # Shortest first, then drop anything beneath a path already kept:
    # `properties` and `properties.title` together are just `properties`,
    # and the prefix rule is the one _within() already uses.
    kept: list = []
    for path in sorted(read, key=lambda p: (p.count("."), p)):
        if not any(path == held or path.startswith(held + ".") for held in kept):
            kept.append(path)
    return tuple(kept)


def _pruned(value, paths):
    """`value`, reduced to what `paths` can reach.

    A copy, never a mutation, and never a DEFAULT: a key the row does
    not have stays absent rather than becoming null, because
    `.properties.title // "untitled"` and `Required` both read those two
    as different things.

    Three rules carry the correctness, and each is the conservative side
    of a case the analysis cannot see:

      - A non-object is returned WHOLE. `.properties.tags[0].name` is
        reported as `properties.tags.name` (an index is not a segment),
        so the walk arrives at a LIST with `name` still to go: it keeps
        the list entire. Same for a string, where jq's own "Cannot index
        string" is the answer that must survive.
      - A dotted path is AMBIGUOUS -- `."a.b"` and `.a.b` are both
        reported as `a.b` -- so both readings are kept when both exist.
        Over-keeping is free; picking one would build a document from
        `null`.
      - A key kept WHOLE wins over the same key kept partially, so
        `properties` and `properties.title` cannot fight over one slot.

    Keys come back in the ROW's order, not the paths' -- jq preserves
    object key order, and a reconstructed object could otherwise print
    differently under `tojson`. (It cannot actually reach `tojson`:
    consuming an object is a read, so its own path would have been
    reported and it would have been copied whole. Ordering it correctly
    costs one dict comprehension and removes the question.)"""
    if not isinstance(value, dict):
        return value
    kept: dict = {}
    deeper: dict = {}
    for path in paths:
        if path in value:                    # a property name containing dots
            kept[path] = value[path]
        head, dot, rest = path.partition(".")
        if dot and head in value:
            deeper.setdefault(head, []).append(rest)
    for head, rest in deeper.items():
        if head not in kept:                 # a whole copy already covers it
            kept[head] = _pruned(value[head], rest)
    return {key: kept[key] for key in value if key in kept}


def _evaluate(program, candidate, filter_text: str, view=None) -> str:
    """One candidate's document.

    `.all()` and not `.first()`: a filter emitting several outputs
    ('.properties.tags[]') has not said which one is the document, and
    taking the first silently picks one. A filter emitting none
    ('empty', or a path through a missing list) has produced no document
    at all. Both refuse, naming the candidate, because the alternative
    is a provider call about the wrong text or about nothing.

    `view` is what jq is HANDED -- the pruned candidate when
    `_projection_paths()` could see what the filter reads, the candidate
    itself otherwise. `candidate` is still what every message NAMES, so
    the id in a refusal is the row's own and not the view's.

    A FAILURE IS RE-RUN AGAINST THE WHOLE CANDIDATE before it is
    reported, and that is not belt-and-braces: jq QUOTES the offending
    value into its error text (`object ({"a":1,...}) and number (1)
    cannot be added`), so a message built from the view could name a
    smaller object than the caller has in front of them. Re-running
    costs the unpruned price exactly once, on a path that is about to
    raise anyway -- and it makes the error, message included,
    byte-identical to the one this function raised before pruning
    existed."""
    where = f"candidate id={candidate.get('id')!r}" if isinstance(candidate, dict) \
        else f"candidate {candidate!r}"
    try:
        outputs = program.input_value(candidate if view is None else view).all()
    except Exception as exc:
        if view is not None:
            return _evaluate(program, candidate, filter_text)
        raise ValueError(
            f"document_from={filter_text!r} failed on {where} -- {type(exc).__name__}: "
            f"{exc}. Refusing rather than scoring that candidate against an empty document"
        ) from exc
    # Two different mistakes, and one message could only fit one of
    # them. "Wrap it in [...] | join" is the fix for several outputs and
    # nonsense for none -- and none is the COMMON case: `select(...)`
    # that matched nothing, `empty`, `.properties.tags[]` over an empty
    # list, a path through a key this row does not have.
    if not outputs:
        raise ValueError(
            f"document_from={filter_text!r} produced 0 outputs for {where}, and a "
            f"document is exactly one string -- a filter selects nothing when the row "
            f"lacks the key, when a select() does not match, or when a list it "
            f"iterates is empty. Give it a fallback (e.g. "
            f"'.properties.title // \"untitled\"'), or drop that candidate before "
            f"reranking"
        )
    if len(outputs) != 1:
        raise ValueError(
            f"document_from={filter_text!r} produced {len(outputs)} outputs for {where}, "
            f"and a document is exactly one string -- wrap it "
            f"(e.g. '[.properties.tags[]] | join(\", \")') so the filter says which"
        )
    document = outputs[0]
    if not isinstance(document, str):
        # The default goes BEFORE tostring, and the order is the whole
        # advice. `| tostring` alone manufactures the literal text
        # "null" on any row where the property is missing, and the
        # reranker then scores the word "null" -- a confidently wrong
        # ranking with nothing to see, which is the failure this module
        # exists to refuse. `// ""` alone produces an empty document,
        # which the next check refuses -- so that half of the advice
        # walked the caller into a second error.
        raise TypeError(
            f"document_from={filter_text!r} evaluated to {type(document).__name__} "
            f"({document!r}) for {where}, not a string -- a reranker reads text. Put "
            f"the default BEFORE the conversion, e.g. '.properties.year // \"unknown\" "
            f"| tostring': `tostring` on a missing property yields the TEXT \"null\", "
            f"which would be scored as a word, and '// \"\"' on its own leaves an empty "
            f"document, which is refused too"
        )
    if not document.strip():
        raise ValueError(
            f"document_from={filter_text!r} evaluated to an empty document for {where} -- "
            f"scoring a candidate against nothing silently changes the ranking, so it "
            f"refuses instead. Filter the candidate out, or give the filter a default"
        )
    return document


def _bind(client: Any, provider: Optional[str], model: Optional[str]) -> tuple:
    """Pick how to call this client, once, at construction.

    Returns (invoke, shape, target): what to CALL, how to READ the
    answer, and the underlying provider callable -- which is the thing
    `inspect.iscoroutinefunction` has to see, since the closure around
    it is always a plain function.

    Invoking and shaping are two functions and not one because an async
    client returns an awaitable from `invoke`: the answer cannot be
    reshaped until it has been awaited, and a closure doing both would
    have to be written twice."""
    if isinstance(client, Rerank):
        raise TypeError(
            "Rerank(Rerank(...)) -- pass the provider client, not another Rerank")

    # 1. Native vendor clients, matched by module name + attribute shape.
    #    Both answer with results SORTED BY RELEVANCE, which is why they
    #    share _indexed_scores -- see its docstring.
    if provider == "cohere" and hasattr(client, "rerank"):
        if not model:
            raise ValueError(
                "Rerank: a Cohere client needs model= (e.g. 'rerank-v3.5') -- hopai will "
                "not pick one for you, because a different reranker is a different "
                "ranking")

        def invoke(query, documents):
            return client.rerank(model=model, query=query, documents=list(documents))
        return invoke, _indexed_scores, client.rerank

    if provider == "voyageai" and hasattr(client, "rerank"):
        if not model:
            raise ValueError("Rerank: a Voyage client needs model= (e.g. 'rerank-2')")

        def invoke(query, documents):
            return client.rerank(query, list(documents), model=model)
        return invoke, _indexed_scores, client.rerank

    # 2. sentence-transformers CrossEncoder: .predict over pairs, and no
    #    model name, because it IS the model. `tokenizer` alongside
    #    `predict` is what tells it apart from any object with a predict
    #    method -- the same two-attribute test Embedder uses for
    #    SentenceTransformer.
    if hasattr(client, "predict") and hasattr(client, "tokenizer"):
        if model:
            raise ValueError(
                "Rerank: a CrossEncoder already IS the model, so model= has nothing to "
                "select -- construct CrossEncoder(name) instead")

        def invoke(query, documents):
            return client.predict([[query, document] for document in documents])
        return invoke, _plain_scores, client.predict

    # 3. The duck-typed protocol: anything that already speaks the
    #    contract this module is built around.
    if hasattr(client, "score"):
        _refuse_model(model, "an object with .score(query, documents)")

        def invoke(query, documents):
            return client.score(query, list(documents))
        return invoke, _plain_scores, client.score

    # 4. A plain callable, the same escape hatch Embedder and Boost have.
    if callable(client):
        _refuse_model(model, "a plain callable")
        _check_arity(client)

        def invoke(query, documents):
            return client(query, list(documents))
        return invoke, _plain_scores, client

    raise TypeError(
        f"Rerank: {type(client).__name__} is not a reranker client hopai recognizes. "
        f"Accepted: a Cohere or Voyage client (.rerank), a sentence-transformers "
        f"CrossEncoder (.predict), anything with .score(query, documents), or a callable "
        f"taking (query, list[str]) -> list[float]"
    )


def _check_arity(client) -> None:
    """A callable that cannot be called as score(query, documents),
    refused where it was written.

    The callable is the shape with no SDK behind it to be wrong about,
    so `Rerank(lambda documents: ...)` is the likeliest first mistake --
    it is also the shape the notebooks use, since CI runs them with no
    network. Left to the query, it surfaced as
    `RerankError: the reranker call failed ... Catch RerankError and
    re-run without rerank=`, which reads as a provider outage and whose
    advice would make a typo permanent.

    `inspect.signature` inside a try, and SILENT when it cannot tell: a
    C-implemented callable or an SDK object with no introspectable
    signature has no arity to check, and refusing one we merely could
    not read would reject a working client. Only a signature that says
    NO is acted on."""
    try:
        signature = inspect.signature(client)
    except (TypeError, ValueError):
        return
    try:
        signature.bind("query", ["document"])
    except TypeError:
        raise TypeError(
            f"Rerank: {getattr(client, '__name__', type(client).__name__)}"
            f"{signature} cannot be called with (query, documents) -- a reranker is "
            f"score(query: str, documents: list[str]) -> list[float], one float per "
            f"document, in the order given. e.g. "
            f"lambda query, documents: [my_service.rank(query, d) for d in documents]"
        ) from None


def _refuse_model(model: Optional[str], shape: str) -> None:
    """model= where nothing reads it is a knob that looks like it works.

    The same refusal Embedder makes for a SentenceTransformer, for the
    same reason: silently ignoring it leaves a caller believing they
    selected a reranker they did not."""
    if model:
        raise ValueError(
            f"Rerank: model={model!r} has nothing to select -- {shape} chooses its own "
            f"model, so hopai would ignore this, and a ranking that ignored your model= "
            f"looks exactly like one that used it")
