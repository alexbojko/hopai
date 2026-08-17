"""
Turning text into vectors, using the embedding client you already have.

hopai stores and compares vectors; it does not produce them. This module
is the seam between the two: you construct and own the provider client
-- keys, timeouts, retries, base_url, proxies -- and hopai calls one
method on it.

    from hopai import Embedder, Vector
    import openai

    graph.define_vectors(nodes=[
        Vector("summary", 1536,
               embed=Embedder(openai.OpenAI(), model="text-embedding-3-small")),
    ])
    graph.set_vectors(nodes=[{"id": 1, "summary": "a paper about Raft"}])
    graph.vector_search(Near("summary", "how do nodes agree?"), k=10)

NO NEW DEPENDENCY, STILL. hopai imports no provider package, ever --
not even to recognize one. Clients are matched by module name and
attribute shape, never isinstance, because an isinstance check needs
the import and that would make the dependency real. `hopai[openai]` is
a convenience alias for "hopai plus the client you were installing
anyway", not a coupling, and a test asserts no provider module reaches
sys.modules after a full embed run.

WHAT IS ACCEPTED, in the order it is tried:

  1. An `Embedder` built here, wrapping a client plus its options.
  2. Anything with embed_documents/embed_query -- every LangChain
     `Embeddings`, and therefore its whole integration catalogue.
  3. Anything with get_text_embedding_batch/get_query_embedding --
     every LlamaIndex `BaseEmbedding`.
  4. A plain callable taking list[str] and returning list[list[float]],
     the same escape hatch the filter DSL and Boost both have.

The duck-typed protocols are where most of the reach comes from;
adapters exist only for native vendor clients, which speak neither.

DOCUMENTS AND QUERIES ARE NOT THE SAME CALL. Cohere, Voyage and Google
embed stored text and query text differently (`search_document` vs
`search_query`, `document` vs `query`, RETRIEVAL_DOCUMENT vs
RETRIEVAL_QUERY), and several sentence-transformers models want a
`query: ` / `passage: ` prefix. Getting it wrong raises nothing and
returns nothing odd -- the neighbours are just quietly worse, which is
the failure mode this library exists to refuse. So the protocol here is
two methods, never one, and every adapter declares which side it is on.
A provider with no asymmetry (OpenAI) simply answers both the same way.

EMBEDDING HAPPENS OUTSIDE THE TRANSACTION. set_vectors() is one
transaction, and holding a Postgres transaction open across an HTTP
round trip is a way to turn a slow provider into a database incident.
Every embed for a call is therefore resolved FIRST, in batches sized to
the provider's cap, and only then is the transaction opened. A failed
batch means nothing was written, which is the same all-or-nothing
promise the vector write path already made.

TRANSIENT FAILURES ARE RETRIED, terminal ones are not, and the
difference is the whole point. A 429 or a 503 is the provider saying
"later" and is retried with exponential backoff plus full jitter; a
401 or a 400 will fail identically forever, so retrying it only burns
the caller's rate limit to reach the same error more slowly. The
classification is duck-typed -- an HTTP status where the exception
carries one, the class name where it does not -- because naming
`openai.RateLimitError` would need the import this module refuses.
`Retry-After` wins over the computed backoff when the provider sent
one; it is the only number here that is not a guess.

Your client probably retries too, and the two policies MULTIPLY:
`Embedder(retries=0)` leaves it entirely to the client, and
`openai.OpenAI(max_retries=0)` leaves it entirely to hopai. Pick one
side rather than paying 3x3.

WHAT THIS DELIBERATELY DOES NOT DO: chunk long documents (an
application concern with a dozen strategies -- over-long input is
refused instead) or cache embeddings. EmbeddingError still keeps the
provider's own exception as `__cause__`, so a caller who wants to
classify differently than the heuristic above can.

IT RUNS ASYNCHRONOUSLY TOO, now that the Graph API and the SQLAlchemy
engine do (AsyncGraph, hopai/asyncio.py). The old note here said async
had to arrive at all three together or not at all, and that was right:
awaiting inside a sync set_vectors() would mean asyncio.run(), which
raises RuntimeError inside the running loop of the very application
that wanted it. So `aembed_query`/`aembed_queries`/`aembed_documents`
are the awaited half of the three methods above, and nothing here ever
starts a loop of its own -- an async client reaching a SYNC call is
refused by name instead.

AsyncGraph resolves every text through those before it enters
SQLAlchemy's greenlet bridge, because a blocking HTTP call inside
run_sync() runs on the event loop's OWN thread and stalls every other
task in the process (issue #74). An async client is recognized exactly
the way every other client is -- module name plus attribute shape,
never isinstance, still no provider import: openai.AsyncOpenAI(),
cohere.AsyncClientV2(), voyageai.AsyncClient(), google.genai's
client.aio, anything with aembed_documents/aembed_query (LangChain) or
aget_text_embedding_batch/aget_query_embedding (LlamaIndex), or a
plain `async def` callable.

A SYNC client on an async path is NOT refused: it runs in
asyncio.to_thread(). That is honest here, and only here, because a
provider SDK blocked on a socket has released the GIL -- so the loop
really does keep running. It is the compatibility path, not the
primary one: an async client costs no thread at all.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import random
import time
from typing import Any, Optional

#: The only network calls hopai makes, so they are the only ones worth a
#: log line. Standard library, no handler attached: an application
#: configures `hopai.embeddings` like any other logger, and one that
#: configures nothing sees nothing.
#:
#: DEBUG carries every provider call with its size, which is what
#: answers "why is this backfill slow" and "how many calls did that
#: cost". A failure logs at WARNING as well as raising, deliberately:
#: embed_stale() walks a field in pages, so a caller may well catch the
#: error and carry on, and a page that silently embedded nothing is
#: exactly what you want in the log afterwards.
logger = logging.getLogger(__name__)

#: Providers whose batch endpoint caps the number of inputs per call.
#: Chunking is ours to do: a provider that refuses 200 inputs should not
#: become the caller's problem, and a silent truncation would be worse.
#: Conservative where a provider documents a range -- the cost of an
#: extra round trip is a round trip; the cost of exceeding the cap is a
#: failed write.
_BATCH_CAPS = {
    "openai": 2048,
    "cohere": 96,
    "voyageai": 128,
    "google": 100,
}
#: Anything we cannot identify gets the smallest cap that is still one
#: round trip for ordinary use.
_DEFAULT_BATCH = 96

#: Retry defaults. Three attempts total, doubling from half a second and
#: capped, which covers the blip a provider recovers from within a few
#: seconds without turning an outage into a long hang. `Retry-After`
#: beyond this cap is refused rather than slept through -- a provider
#: asking for ten minutes is telling you to come back later, not to
#: hold a backfill open.
_DEFAULT_RETRIES = 2
_DEFAULT_BACKOFF = 0.5
_MAX_BACKOFF = 30.0
_MAX_RETRY_AFTER = 120.0

#: The two sides of the document/query asymmetry, per provider family.
#: Each entry is (documents, queries) for that provider's own spelling.
_INPUT_TYPES = {
    "cohere": ("search_document", "search_query"),
    "voyageai": ("document", "query"),
    "google": ("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"),
}


#: HTTP statuses worth trying again. 429 and the 5xx family are the
#: provider saying "later"; 408/409 are the request never landing.
#: Everything else -- 400, 401, 403, 404, 422 -- is a request that will
#: fail identically forever, and retrying it burns the caller's rate
#: limit to arrive at the same error more slowly.
_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

#: Exception class names that mean the same thing, for clients that
#: raise without a status. Matched as substrings of the CLASS NAME,
#: which is the only provider vocabulary available here -- importing
#: openai to name RateLimitError is the one thing this module cannot do,
#: and every provider spells these the same way anyway.
_RETRY_NAMES = ("ratelimit", "timeout", "connection", "unavailable",
                "overloaded", "internalserver", "serviceunavailable", "apierror")

#: Never retried however the class is spelled: these are the caller's
#: configuration being wrong, and no amount of waiting fixes them.
_TERMINAL_NAMES = ("authentication", "permission", "notfound", "badrequest",
                   "invalidrequest", "unprocessable")


def _status_of(exc: BaseException) -> Optional[int]:
    """The HTTP status a provider exception carries, if any.

    Duck-typed across the two shapes in the wild: `exc.status_code`
    (openai, anthropic) and `exc.response.status_code` (requests-based
    clients)."""
    for holder, attribute in ((exc, "status_code"), (getattr(exc, "response", None),
                                                     "status_code")):
        value = getattr(holder, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _retryable(exc: BaseException) -> bool:
    """Whether trying the same call again could plausibly succeed.

    Status first, because it is unambiguous; the class name only when
    there is none. A blanket retry would be worse than none: it turns a
    bad API key into five slow failures instead of one fast one, which
    is the opposite of handling an API exception correctly."""
    status = _status_of(exc)
    if status is not None:
        return status in _RETRY_STATUS
    name = type(exc).__name__.lower()
    if any(word in name for word in _TERMINAL_NAMES):
        return False
    return any(word in name for word in _RETRY_NAMES)


def _retry_after(exc: BaseException) -> Optional[float]:
    """The provider's own `Retry-After`, in seconds, when it sent one.

    Honoured over our own backoff because it is the only number here
    that is not a guess -- a 429 with Retry-After: 30 means backing off
    0.5s just spends another request to be told 30 again."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        value = float(headers.get("Retry-After") or headers.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        return None
    return value if 0 <= value <= _MAX_RETRY_AFTER else None


class EmbeddingError(RuntimeError):
    """A provider call failed, or answered with something unusable.

    Raised rather than returned so a half-embedded batch can never reach
    the write path: set_vectors() resolves every embed before it opens
    its transaction, so this always fires with nothing written.

    Raised only after the retries are spent, or immediately when the
    failure is terminal -- so seeing one means waiting will not help.

    THE PROVIDER'S OWN EXCEPTION IS KEPT as `__cause__`, for a caller
    who wants to classify it more precisely than `_retryable()` can
    without importing anything:

        try:
            graph.embed_stale()
        except EmbeddingError as failed:
            if isinstance(failed.__cause__, openai.RateLimitError):
                ...                       # back off and re-run; it resumes"""


def _provider(client: Any) -> Optional[str]:
    """Which provider family a client belongs to, by module name.

    By NAME and not isinstance, because isinstance needs the import --
    see the module docstring. `type(client).__module__` is
    'openai.resources...' or 'cohere.client_v2' or similar, so the first
    dotted segment is the distribution."""
    # The `or ""` only keeps .split() off a None; any placeholder would
    # do, since no falsy __module__ can produce a root that is a provider
    # key. Mutating the literal is therefore an equivalent mutant.
    module = type(client).__module__ or ""
    root = module.split(".")[0]
    return root if root in _BATCH_CAPS or root in _INPUT_TYPES else None


def _clean_texts(texts, owner: str) -> list:
    """Validate the strings before spending a provider call on them.

    Empty or whitespace-only text embeds to something with no direction,
    and Near() already refuses an all-zero query vector -- catching it
    here names the row instead of surfacing three layers down as a
    confusing cosine complaint."""
    cleaned = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"{owner}: item {index} is {type(text).__name__}, not a string -- "
                f"pass the text to embed, or a list of floats to store directly"
            )
        if not text.strip():
            raise ValueError(
                f"{owner}: item {index} is empty or whitespace, which embeds to a "
                f"vector with no direction -- every similarity against it would be "
                f"undefined. Skip the row, or give it real text"
            )
        cleaned.append(text)
    return cleaned


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


# ---------------------------------------------------------------------
# Reading a provider's answer
#
# One reader per provider, shared by the sync binding and the async one
# (_bind/_abind below). The two differ ONLY in awaiting the call; if
# each also carried its own copy of "where the vectors are in this
# answer", one could learn a new response shape and the other could
# not -- and a client answering correctly on one path and confusingly
# on the other is the kind of split this library exists to refuse.
# ---------------------------------------------------------------------

def _openai_answer(answer):
    return answer.data


def _cohere_answer(answer):
    embeddings = getattr(answer, "embeddings", answer)
    return getattr(embeddings, "float_", None) or getattr(embeddings, "float", embeddings)


def _voyage_answer(answer):
    return getattr(answer, "embeddings", answer)


def _google_answer(answer):
    return [e.values for e in answer.embeddings]


def _is_async_call(fn) -> bool:
    """Whether calling `fn` hands back something to await.

    Answered by inspection rather than by calling: the alternative is
    spending a provider request to find out what kind of client this
    is. Covers the callable OBJECT case too (a client whose __call__ is
    the `async def`), which `inspect.iscoroutinefunction` alone reports
    as False."""
    if fn is None:
        return False
    if inspect.iscoroutinefunction(fn):
        return True
    call = getattr(fn, "__call__", None)          # noqa: B004 -- the OBJECT's own __call__
    return call is not None and inspect.iscoroutinefunction(call)


async def _awaited(answer):
    """The provider's answer, whether it arrived directly or as
    something to await.

    Every async binding below goes through this, so a method that is
    not itself an `async def` but RETURNS an awaitable (a wrapper
    handing back a Task or a Future) works the same as one that is --
    the detection above cannot tell those apart without calling, and
    this makes it not need to."""
    return await answer if inspect.isawaitable(answer) else answer


def _as_vectors(raw, expected: int, owner: str) -> list:
    """Coerce a provider's answer into a list of float lists.

    Providers disagree about the container -- a list, a numpy array, an
    object with .embedding -- but agree about the contents. Length is
    checked here rather than at the write: a provider that silently
    returns fewer rows than it was given would otherwise pair vectors
    with the wrong ids, which is a silent wrong answer of the worst
    kind."""
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    vectors = []
    for item in raw:
        if hasattr(item, "embedding"):        # openai's per-row object
            item = item.embedding
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, (list, tuple)):
            raise EmbeddingError(
                f"{owner}: the provider returned {type(item).__name__} where a list of "
                f"numbers was expected"
            )
        vectors.append([float(value) for value in item])
    if len(vectors) != expected:
        raise EmbeddingError(
            f"{owner}: asked for {expected} embedding(s) and got {len(vectors)} -- "
            f"refusing rather than pairing vectors with the wrong rows"
        )
    return vectors


def _async_client_error(client: Any) -> EmbeddingError:
    """The one sentence said whenever an async client meets a sync call.

    EmbeddingError rather than TypeError because _attempt() lets ours
    through untouched: a client that cannot answer synchronously will
    not answer differently on the third try, and burying this under
    "the provider call failed after 3 attempt(s)" would hide the line
    that names the fix."""
    return EmbeddingError(
        f"{type(client).__name__} is an async embedding client and this is a synchronous "
        f"call -- hopai never starts an event loop of its own. Reach it through AsyncGraph "
        f"(hopai.asyncio), which awaits every embed, or await "
        f"Embedder.aembed_documents()/aembed_query() yourself. A plain Graph needs the "
        f"provider's SYNC client"
    )


def _sync_answer(answer, client: Any):
    """A provider's answer as the sync bindings need it: not an
    awaitable.

    Checked HERE, on the raw answer, rather than further down where the
    vectors are read out. Every reader above reaches into the answer's
    shape, so a coroutine reaching one is reported as "'coroutine'
    object has no attribute 'data'" -- true, useless, and three layers
    from the line that chose the client. _abind() recognizes every
    async client it can by inspection; this is what happens to the ones
    it cannot (a plain `def` handing back a Future, say)."""
    if not inspect.isawaitable(answer):
        return answer
    close = getattr(answer, "close", None)   # else Python warns "never awaited" at GC
    if close is not None:
        close()
    raise _async_client_error(client)


class Embedder:
    """A provider client plus the options hopai needs to call it.

        Embedder(openai.OpenAI(), model="text-embedding-3-small")
        Embedder(cohere.ClientV2(), model="embed-v4.0")
        Embedder(SentenceTransformer("all-MiniLM-L6-v2"))
        Embedder(lambda texts: my_service.embed(texts))

    An ASYNC client is accepted exactly the same way and is awaited
    rather than called -- what AsyncGraph wants, since it resolves
    every text off the event loop's thread:

        Embedder(openai.AsyncOpenAI(), model="text-embedding-3-small")
        Embedder(my_async_embed)              # an `async def` taking list[str]

    A sync client on an async path still works: it runs in a worker
    thread (see _arun()). `is_async` says which of the two you have.

    `model` is required for providers that take one and refused for
    those that do not, rather than defaulted: a silently chosen model is
    a silently different set of neighbours, and the caller cannot see it
    happened.

    `dimensions` is passed through where the provider supports
    truncation (OpenAI's text-embedding-3-*, Google's
    output_dimensionality). Elsewhere it is only checked, because
    padding or slicing a vector on the caller's behalf would change what
    the model said."""

    def __init__(self, client: Any, model: Optional[str] = None,
                 batch_size: Optional[int] = None, dimensions: Optional[int] = None,
                 retries: int = _DEFAULT_RETRIES, backoff: float = _DEFAULT_BACKOFF):
        self.client = client
        self.model = model
        self.provider = _provider(client)
        self.dimensions = dimensions
        if batch_size is not None and (
                not isinstance(batch_size, int) or isinstance(batch_size, bool)
                or batch_size < 1):
            raise ValueError(
                f"Embedder: batch_size must be a positive integer, got {batch_size!r}")
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError(
                f"Embedder: retries must be a non-negative integer (0 disables retrying), "
                f"got {retries!r}")
        if not isinstance(backoff, (int, float)) or isinstance(backoff, bool) \
                or not math.isfinite(backoff) or backoff <= 0:
            raise ValueError(
                f"Embedder: backoff must be a positive number of seconds, got {backoff!r}")
        self.retries = retries
        self.backoff = float(backoff)
        # `provider or ""` keeps .get() off a None key; as in _provider(),
        # the literal is unobservable -- no placeholder is a _BATCH_CAPS
        # key, so an unknown client takes _DEFAULT_BATCH either way.
        self.batch_size = batch_size or _BATCH_CAPS.get(self.provider or "", _DEFAULT_BATCH)
        # Both bindings are resolved here, for the same reason one was:
        # which half of the API a client can answer is a property of the
        # client, and finding out at the first embed puts the answer a
        # long way from the line that chose it.
        self._acall = _abind(client, self.provider, model)
        self._call = _bind(client, self.provider, model,
                           async_only=self._acall is not None)

    @property
    def is_async(self) -> bool:
        """Whether this client is awaited directly, or reached through a
        worker thread on the async path -- see _arun()."""
        return self._acall is not None

    def __repr__(self) -> str:
        parts = [self.provider or type(self.client).__name__]
        if self.model:
            parts.append(f"model={self.model!r}")
        return f"Embedder({', '.join(parts)})"

    def embed_documents(self, texts: list) -> list:
        """Embed text being STORED. See the module docstring on why this
        is not the same call as embed_query."""
        return self._run(texts, query=False)

    def embed_query(self, text: str) -> list:
        """Embed text being SEARCHED WITH."""
        return self._run([text], query=True)[0]

    def embed_queries(self, texts: list) -> list:
        """Several queries at once, on the query side of the asymmetry
        -- what vector_search_many() needs so N searches cost one
        provider round trip rather than N."""
        return self._run(texts, query=True)

    def _wait_before_retry(self, owner: str, failure: BaseException, attempt: int,
                           attempts: int) -> float:
        """How long to wait before `attempt`, logged as it is decided.

        The retry POLICY lives here, once, and _attempt()/_aattempt()
        differ only in how they spend the number -- time.sleep() versus
        awaiting it. A second copy of the doubling window, the cap, the
        full jitter and Retry-After winning would be free to drift from
        this one, and the numbers ARE the feature.

        Full jitter (`random.uniform(0, window)`) rather than the exact
        doubling: a backfill that fans out over several fields fails at
        the same instant and would otherwise retry in lockstep, hitting
        a rate-limited provider with the same synchronised burst that
        caused the 429.

        `Retry-After` wins over the computed window when the provider
        sent one -- it is the only number here that is not a guess."""
        window = min(self.backoff * (2 ** (attempt - 1)), _MAX_BACKOFF)
        delay = _retry_after(failure)
        if delay is None:
            delay = random.uniform(0, window)
        logger.warning("%s: %s: %s -- retrying in %.2fs (attempt %d of %d)",
                       owner, type(failure).__name__, failure, delay,
                       attempt + 1, attempts)
        return delay

    def _attempt(self, chunk: list, query: bool, owner: str, done: int):
        """One provider call, retried while the failure looks transient."""
        attempts = self.retries + 1
        failure = None
        for attempt in range(attempts):
            if failure is not None:
                # Back off BEFORE the retry rather than after the
                # failure, so the loop bound is the ONLY thing deciding
                # how many calls happen. With an "is this the last one"
                # test in the body instead, the two express the same
                # number twice and a wider bound is unreachable --
                # unobservable to any test, which is how a retry count
                # drifts from what the caller asked for.
                time.sleep(self._wait_before_retry(owner, failure, attempt, attempts))
            try:
                return self._call(chunk, query, self.dimensions)
            except EmbeddingError:
                raise
            except Exception as exc:                      # provider-side failure
                if not _retryable(exc):
                    self._give_up(owner, done, attempt + 1, exc)
                failure = exc
        self._give_up(owner, done, attempts, failure)

    async def _aattempt(self, chunk: list, query: bool, owner: str, done: int):
        """_attempt(), with the provider call and the backoff awaited.

        The same loop rather than a shared one because the difference IS
        the awaits, and an `async def` cannot be reached from the sync
        method without running a loop -- which this module never does.
        Everything that could drift silently (which failures are worth
        retrying, how long to wait, what a spent call reports) is a call
        into the shared helpers above."""
        attempts = self.retries + 1
        failure = None
        for attempt in range(attempts):
            if failure is not None:
                await asyncio.sleep(
                    self._wait_before_retry(owner, failure, attempt, attempts))
            try:
                return await self._acall(chunk, query, self.dimensions)
            except EmbeddingError:
                raise
            except Exception as exc:                      # provider-side failure
                if not _retryable(exc):
                    self._give_up(owner, done, attempt + 1, exc)
                failure = exc
        self._give_up(owner, done, attempts, failure)

    @staticmethod
    def _give_up(owner: str, done: int, attempts: int, exc: BaseException):
        """The one place a spent call is reported, so the log line and
        the exception can never disagree about how far it got."""
        logger.warning("%s: provider call failed after %d embedded, %d attempt(s) (%s: %s)",
                       owner, done, attempts, type(exc).__name__, exc)
        raise EmbeddingError(
            f"{owner}: the provider call failed after {attempts} attempt(s) "
            f"({type(exc).__name__}: {exc}) -- nothing was written"
        ) from exc

    def _owner(self, query: bool) -> str:
        # `query` is read only as `A if query else B` -- here and in every
        # binding _bind() returns -- so any falsy value is the document
        # side. Mutating query=False to another falsy value is equivalent.
        return f"{self!r}.{'embed_query' if query else 'embed_documents'}"

    def _check_dimensions(self, out: list, owner: str) -> None:
        if self.dimensions is None:
            return
        for index, vector in enumerate(out):
            if len(vector) != self.dimensions:
                raise EmbeddingError(
                    f"{owner}: item {index} came back with {len(vector)} dimensions, "
                    f"not the {self.dimensions} this field declares -- the model and "
                    f"the field disagree; re-declare the field or change the model"
                )

    def _run(self, texts, query: bool) -> list:
        owner = self._owner(query)
        cleaned = _clean_texts(list(texts), owner)
        if not cleaned:
            return []
        out = []
        for chunk in _chunks(cleaned, self.batch_size):
            logger.debug("%s: embedding %d text(s)", owner, len(chunk))
            raw = self._attempt(chunk, query, owner, len(out))
            out.extend(_as_vectors(raw, len(chunk), owner))
        self._check_dimensions(out, owner)
        return out

    # -- the awaited half -------------------------------------------

    async def aembed_documents(self, texts: list) -> list:
        """embed_documents(), awaited. See the module docstring on why
        the async path exists at all: AsyncGraph resolves every text
        through these BEFORE it enters the greenlet bridge, because a
        blocking provider call inside run_sync() holds the event loop's
        own thread."""
        return await self._arun(texts, query=False)

    async def aembed_query(self, text: str) -> list:
        """embed_query(), awaited."""
        return (await self._arun([text], query=True))[0]

    async def aembed_queries(self, texts: list) -> list:
        """embed_queries(), awaited."""
        return await self._arun(texts, query=True)

    async def _arun(self, texts, query: bool) -> list:
        if self._acall is None:
            # THE COMPATIBILITY PATH. A sync client cannot be awaited,
            # and calling it here would put an HTTP round trip on the
            # event loop's thread -- the whole bug (issue #74). A thread
            # is legitimate for THIS work and not in general: a provider
            # SDK blocked on a socket has released the GIL, so the loop
            # really does keep running. (A CPU-bound extension that
            # holds the GIL would starve the loop from a thread just the
            # same; that is the line to test anything else against.)
            # The cost is a thread per in-flight call, which is exactly
            # the ceiling hopai/asyncio.py chose the greenlet bridge to
            # avoid -- so this is the fallback, and an async client is
            # the answer.
            return await asyncio.to_thread(self._run, texts, query)
        owner = self._owner(query)
        cleaned = _clean_texts(list(texts), owner)
        if not cleaned:
            return []
        out = []
        for chunk in _chunks(cleaned, self.batch_size):
            logger.debug("%s: embedding %d text(s)", owner, len(chunk))
            raw = await self._aattempt(chunk, query, owner, len(out))
            out.extend(_as_vectors(raw, len(chunk), owner))
        self._check_dimensions(out, owner)
        return out


def _bind(client: Any, provider: Optional[str], model: Optional[str],
          async_only: bool = False):
    """Pick how to call this client, once, at construction.

    Resolved eagerly so a mistyped client is refused when the Embedder is
    built rather than on the first write, which might be a long way from
    the line that got it wrong.

    `async_only` is set when _abind() recognized a client this function
    does not (one exposing only aembed_documents/aembed_query, say).
    That is a working client used through the wrong half of the API,
    not a mistyped one, so it binds to a refusal that names the async
    call instead of failing construction -- which would put every
    async-only client out of AsyncGraph's reach as well."""
    # 1. Already one of ours.
    if isinstance(client, Embedder):
        raise TypeError(
            "Embedder(Embedder(...)) -- pass the provider client, not another Embedder")

    # 2. Native vendor clients, matched by module name + attribute shape.
    if provider == "openai" and hasattr(client, "embeddings"):
        if not model:
            raise ValueError(
                "Embedder: an OpenAI client needs model= (e.g. "
                "'text-embedding-3-small') -- hopai will not pick one for you, "
                "because a different model means different neighbours")

        def call(texts, query, dimensions):        # noqa: ARG001 -- no asymmetry
            extra = {"dimensions": dimensions} if dimensions else {}
            return _openai_answer(_sync_answer(
                client.embeddings.create(model=model, input=texts, **extra), client))
        return call

    if provider == "cohere" and hasattr(client, "embed"):
        if not model:
            raise ValueError(
                "Embedder: a Cohere client needs model= (e.g. 'embed-v4.0')")
        documents, queries = _INPUT_TYPES["cohere"]

        def call(texts, query, dimensions):        # noqa: ARG001
            return _cohere_answer(_sync_answer(client.embed(
                texts=texts, model=model, input_type=queries if query else documents,
                embedding_types=["float"]), client))
        return call

    if provider == "voyageai" and hasattr(client, "embed"):
        if not model:
            raise ValueError("Embedder: a Voyage client needs model= (e.g. 'voyage-3')")
        documents, queries = _INPUT_TYPES["voyageai"]

        def call(texts, query, dimensions):        # noqa: ARG001
            return _voyage_answer(_sync_answer(client.embed(
                texts, model=model, input_type=queries if query else documents), client))
        return call

    if provider == "google" and hasattr(client, "models"):
        if not model:
            raise ValueError(
                "Embedder: a Google GenAI client needs model= (e.g. "
                "'gemini-embedding-001')")
        documents, queries = _INPUT_TYPES["google"]

        def call(texts, query, dimensions):
            config = {"task_type": queries if query else documents}
            if dimensions:
                config["output_dimensionality"] = dimensions
            return _google_answer(_sync_answer(client.models.embed_content(
                model=model, contents=texts, config=config), client))
        return call

    # 3. sentence-transformers: .encode, no model name (it IS the model).
    if hasattr(client, "encode") and hasattr(client, "tokenize"):
        if model:
            raise ValueError(
                "Embedder: a SentenceTransformer already IS the model, so model= has "
                "nothing to select -- construct SentenceTransformer(name) instead")

        def call(texts, query, dimensions):        # noqa: ARG001
            return _sync_answer(client.encode(list(texts)), client)
        return call

    # 4. The duck-typed protocols, in the order they are most common.
    #    Their query methods take ONE text, so several queries are a
    #    loop here rather than a truncation: embed_queries() may hand
    #    down a batch, and `texts[0]` alone would drop the rest.
    if hasattr(client, "embed_documents") and hasattr(client, "embed_query"):
        def call(texts, query, dimensions):        # noqa: ARG001
            return _sync_answer(
                [client.embed_query(text) for text in texts] if query
                else client.embed_documents(list(texts)), client)
        return call

    if hasattr(client, "get_text_embedding_batch") and hasattr(client, "get_query_embedding"):
        def call(texts, query, dimensions):        # noqa: ARG001
            return _sync_answer(
                [client.get_query_embedding(text) for text in texts] if query
                else client.get_text_embedding_batch(list(texts)), client)
        return call

    # 5. A plain callable, which cannot express the asymmetry -- so it is
    #    the caller's job, and saying so here beats a silent difference.
    if callable(client):
        def call(texts, query, dimensions):        # noqa: ARG001
            return _sync_answer(client(list(texts)), client)
        return call

    if async_only:
        return _refuses_sync(client)

    raise TypeError(
        f"Embedder: {type(client).__name__} is not an embedding client hopai "
        f"recognizes. Accepted: an OpenAI/Cohere/Voyage/Google client, a "
        f"SentenceTransformer, anything with embed_documents+embed_query "
        f"(LangChain) or get_text_embedding_batch+get_query_embedding "
        f"(LlamaIndex), or a callable taking list[str] -> list[list[float]]"
    )


def _refuses_sync(client: Any):
    """The sync binding for a client that speaks ONLY async -- the same
    refusal _sync_answer() raises, said before the call rather than
    after it, because there is no sync method here to call at all."""
    def call(texts, query, dimensions):            # noqa: ARG001
        raise _async_client_error(client)
    return call


def _abind(client: Any, provider: Optional[str], model: Optional[str]):
    """Pick how to call this client WITHOUT holding the thread, or None
    when it has no async surface at all.

    Same matching rule as _bind(), in the same order -- module name plus
    attribute shape, never isinstance -- so recognizing an async client
    still imports no provider package. What differs is the second half
    of each test. A native vendor client is async when its own method
    is a coroutine function (openai.AsyncOpenAI, cohere.AsyncClientV2,
    voyageai.AsyncClient, google.genai's client.aio), which inspection
    answers without spending a request. The two duck-typed protocols
    declare it by NAME instead -- aembed_documents/aembed_query,
    aget_text_embedding_batch/aget_query_embedding -- because that `a`
    prefix IS the convention their whole catalogue is written to, and
    an object carrying both halves of the pair has said what they are.

    None means "sync only", and _arun() then runs the sync binding in a
    worker thread. See Embedder._arun() for why that is the
    compatibility path rather than the primary one."""
    if isinstance(client, Embedder):
        return None                                # _bind() refuses it by name

    if provider == "openai" and hasattr(client, "embeddings") \
            and _is_async_call(getattr(client.embeddings, "create", None)):
        async def call(texts, query, dimensions):  # noqa: ARG001 -- no asymmetry
            extra = {"dimensions": dimensions} if dimensions else {}
            return _openai_answer(await _awaited(
                client.embeddings.create(model=model, input=texts, **extra)))
        return call

    if provider == "cohere" and _is_async_call(getattr(client, "embed", None)):
        documents, queries = _INPUT_TYPES["cohere"]

        async def call(texts, query, dimensions):  # noqa: ARG001
            return _cohere_answer(await _awaited(client.embed(
                texts=texts, model=model, input_type=queries if query else documents,
                embedding_types=["float"])))
        return call

    if provider == "voyageai" and _is_async_call(getattr(client, "embed", None)):
        documents, queries = _INPUT_TYPES["voyageai"]

        async def call(texts, query, dimensions):  # noqa: ARG001
            return _voyage_answer(await _awaited(client.embed(
                texts, model=model, input_type=queries if query else documents)))
        return call

    # google-genai keeps its async surface on a SEPARATE object,
    # client.aio, rather than in a second client class -- so this is the
    # one provider where the sync and async bindings can both match the
    # same object, and they are meant to.
    if provider == "google" and _is_async_call(
            getattr(getattr(getattr(client, "aio", None), "models", None),
                    "embed_content", None)):
        documents, queries = _INPUT_TYPES["google"]

        async def call(texts, query, dimensions):
            config = {"task_type": queries if query else documents}
            if dimensions:
                config["output_dimensionality"] = dimensions
            return _google_answer(await _awaited(client.aio.models.embed_content(
                model=model, contents=texts, config=config)))
        return call

    if hasattr(client, "encode") and hasattr(client, "tokenize"):
        # A SentenceTransformer runs the model in THIS process; there is
        # no socket and so nothing to await. Returning None here rather
        # than falling through is load-bearing: it is an nn.Module, and
        # nn.Module is callable, so the plain-callable branch at the
        # bottom would otherwise get a look at it.
        return None

    if hasattr(client, "aembed_documents") and hasattr(client, "aembed_query"):
        async def call(texts, query, dimensions):  # noqa: ARG001
            if query:
                return [await _awaited(client.aembed_query(text)) for text in texts]
            return await _awaited(client.aembed_documents(list(texts)))
        return call

    if hasattr(client, "aget_text_embedding_batch") \
            and hasattr(client, "aget_query_embedding"):
        async def call(texts, query, dimensions):  # noqa: ARG001
            if query:
                return [await _awaited(client.aget_query_embedding(text))
                        for text in texts]
            return await _awaited(client.aget_text_embedding_batch(list(texts)))
        return call

    if _is_async_call(client):
        async def call(texts, query, dimensions):  # noqa: ARG001
            return await _awaited(client(list(texts)))
        return call

    return None


def as_embedder(embed: Any) -> Embedder:
    """Normalize whatever was passed to `embed=` into an Embedder."""
    return embed if isinstance(embed, Embedder) else Embedder(embed)


def norm(vector) -> float:
    """Euclidean norm, for callers checking a provider returns unit
    vectors. Cosine ignores magnitude, so this is diagnostic only."""
    return math.sqrt(sum(float(v) * float(v) for v in vector))
