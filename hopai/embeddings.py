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
    graph.vector_search(Near("summary", text="how do nodes agree?"), k=10)

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

WHAT THIS DELIBERATELY DOES NOT DO: chunk long documents (an
application concern with a dozen strategies -- over-long input is
refused instead), cache embeddings, or run anything asynchronously
(hopai is sync end to end, because SQLAlchemy here is).
"""

from __future__ import annotations

import math
from typing import Any, Optional

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

#: The two sides of the document/query asymmetry, per provider family.
#: Each entry is (documents, queries) for that provider's own spelling.
_INPUT_TYPES = {
    "cohere": ("search_document", "search_query"),
    "voyageai": ("document", "query"),
    "google": ("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"),
}


class EmbeddingError(RuntimeError):
    """A provider call failed, or answered with something unusable.

    Raised rather than returned so a half-embedded batch can never reach
    the write path: set_vectors() resolves every embed before it opens
    its transaction, so this always fires with nothing written."""


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


class Embedder:
    """A provider client plus the options hopai needs to call it.

        Embedder(openai.OpenAI(), model="text-embedding-3-small")
        Embedder(cohere.ClientV2(), model="embed-v4.0")
        Embedder(SentenceTransformer("all-MiniLM-L6-v2"))
        Embedder(lambda texts: my_service.embed(texts))

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
                 batch_size: Optional[int] = None, dimensions: Optional[int] = None):
        self.client = client
        self.model = model
        self.provider = _provider(client)
        self.dimensions = dimensions
        if batch_size is not None and (
                not isinstance(batch_size, int) or isinstance(batch_size, bool)
                or batch_size < 1):
            raise ValueError(
                f"Embedder: batch_size must be a positive integer, got {batch_size!r}")
        # `provider or ""` keeps .get() off a None key; as in _provider(),
        # the literal is unobservable -- no placeholder is a _BATCH_CAPS
        # key, so an unknown client takes _DEFAULT_BATCH either way.
        self.batch_size = batch_size or _BATCH_CAPS.get(self.provider or "", _DEFAULT_BATCH)
        self._call = _bind(client, self.provider, model)

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

    def _run(self, texts, query: bool) -> list:
        # `query` is read only as `A if query else B` -- here and in every
        # binding _bind() returns -- so any falsy value is the document
        # side. Mutating query=False to another falsy value is equivalent.
        owner = f"{self!r}.{'embed_query' if query else 'embed_documents'}"
        cleaned = _clean_texts(list(texts), owner)
        if not cleaned:
            return []
        out = []
        for chunk in _chunks(cleaned, self.batch_size):
            try:
                raw = self._call(chunk, query, self.dimensions)
            except EmbeddingError:
                raise
            except Exception as exc:                      # provider-side failure
                raise EmbeddingError(
                    f"{owner}: the provider call failed ({type(exc).__name__}: {exc}) -- "
                    f"nothing was written"
                ) from exc
            out.extend(_as_vectors(raw, len(chunk), owner))
        if self.dimensions is not None:
            for index, vector in enumerate(out):
                if len(vector) != self.dimensions:
                    raise EmbeddingError(
                        f"{owner}: item {index} came back with {len(vector)} dimensions, "
                        f"not the {self.dimensions} this field declares -- the model and "
                        f"the field disagree; re-declare the field or change the model"
                    )
        return out


def _bind(client: Any, provider: Optional[str], model: Optional[str]):
    """Pick how to call this client, once, at construction.

    Resolved eagerly so a mistyped client is refused when the Embedder is
    built rather than on the first write, which might be a long way from
    the line that got it wrong."""
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
            return client.embeddings.create(model=model, input=texts, **extra).data
        return call

    if provider == "cohere" and hasattr(client, "embed"):
        if not model:
            raise ValueError(
                "Embedder: a Cohere client needs model= (e.g. 'embed-v4.0')")
        documents, queries = _INPUT_TYPES["cohere"]

        def call(texts, query, dimensions):        # noqa: ARG001
            answer = client.embed(texts=texts, model=model,
                                  input_type=queries if query else documents,
                                  embedding_types=["float"])
            embeddings = getattr(answer, "embeddings", answer)
            return getattr(embeddings, "float_", None) or getattr(
                embeddings, "float", embeddings)
        return call

    if provider == "voyageai" and hasattr(client, "embed"):
        if not model:
            raise ValueError("Embedder: a Voyage client needs model= (e.g. 'voyage-3')")
        documents, queries = _INPUT_TYPES["voyageai"]

        def call(texts, query, dimensions):        # noqa: ARG001
            answer = client.embed(texts, model=model,
                                  input_type=queries if query else documents)
            return getattr(answer, "embeddings", answer)
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
            answer = client.models.embed_content(model=model, contents=texts,
                                                 config=config)
            return [e.values for e in answer.embeddings]
        return call

    # 3. sentence-transformers: .encode, no model name (it IS the model).
    if hasattr(client, "encode") and hasattr(client, "tokenize"):
        if model:
            raise ValueError(
                "Embedder: a SentenceTransformer already IS the model, so model= has "
                "nothing to select -- construct SentenceTransformer(name) instead")

        def call(texts, query, dimensions):        # noqa: ARG001
            return client.encode(list(texts))
        return call

    # 4. The duck-typed protocols, in the order they are most common.
    #    Their query methods take ONE text, so several queries are a
    #    loop here rather than a truncation: embed_queries() may hand
    #    down a batch, and `texts[0]` alone would drop the rest.
    if hasattr(client, "embed_documents") and hasattr(client, "embed_query"):
        def call(texts, query, dimensions):        # noqa: ARG001
            return [client.embed_query(text) for text in texts] if query \
                else client.embed_documents(list(texts))
        return call

    if hasattr(client, "get_text_embedding_batch") and hasattr(client, "get_query_embedding"):
        def call(texts, query, dimensions):        # noqa: ARG001
            return [client.get_query_embedding(text) for text in texts] if query \
                else client.get_text_embedding_batch(list(texts))
        return call

    # 5. A plain callable, which cannot express the asymmetry -- so it is
    #    the caller's job, and saying so here beats a silent difference.
    if callable(client):
        def call(texts, query, dimensions):        # noqa: ARG001
            return client(list(texts))
        return call

    raise TypeError(
        f"Embedder: {type(client).__name__} is not an embedding client hopai "
        f"recognizes. Accepted: an OpenAI/Cohere/Voyage/Google client, a "
        f"SentenceTransformer, anything with embed_documents+embed_query "
        f"(LangChain) or get_text_embedding_batch+get_query_embedding "
        f"(LlamaIndex), or a callable taking list[str] -> list[list[float]]"
    )


def as_embedder(embed: Any) -> Embedder:
    """Normalize whatever was passed to `embed=` into an Embedder."""
    return embed if isinstance(embed, Embedder) else Embedder(embed)


def norm(vector) -> float:
    """Euclidean norm, for callers checking a provider returns unit
    vectors. Cosine ignores magnitude, so this is diagnostic only."""
    return math.sqrt(sum(float(v) * float(v) for v in vector))
