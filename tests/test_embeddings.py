"""
The embedding seam: which clients are accepted, and what is refused.

Every provider client here is a FAKE whose __module__ is set to the
real distribution's name, because that is exactly how hopai recognizes
one -- by module name and attribute shape, never isinstance. Faking it
this way tests the real matching logic while keeping the promise the
module is built on: hopai imports no provider package, and
test_no_provider_package_is_ever_imported holds it to that.

The fakes are built by fixtures rather than at module scope on purpose:
mutmut runs the whole suite twice in one process, and shared mutable
module state is what broke its baseline twice already in this project.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading

import pytest

import hopai.embeddings as embeddings_module
from hopai.embeddings import Embedder, EmbeddingError


def run(coro):
    """No pytest-asyncio, same as tests/test_async.py -- a one-line
    asyncio.run() wrapper is the whole requirement."""
    return asyncio.run(coro)


def _named(module: str, cls):
    """A class that claims to live in `module`, which is what _provider()
    reads. Setting __module__ is not a trick around the design -- it is
    the design: a real openai.OpenAI reports 'openai.…' the same way."""
    cls.__module__ = module
    return cls


@pytest.fixture()
def calls() -> list:
    """Fresh per test, so nothing carries between them or between
    mutmut's two baseline runs."""
    return []


# ---------------------------------------------------------------------
# The duck-typed protocols: where most of the reach comes from
# ---------------------------------------------------------------------

class TestProtocols:
    def test_a_plain_callable_is_accepted(self):
        """The escape hatch, mirroring Boost's callable form."""
        embedder = Embedder(lambda texts: [[1.0, 0.0] for _ in texts])
        assert embedder.embed_documents(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
        assert embedder.embed_query("q") == [1.0, 0.0]

    def test_the_langchain_protocol_is_accepted(self, calls):
        """embed_documents/embed_query covers every LangChain Embeddings,
        and therefore its whole integration catalogue -- one protocol
        instead of one adapter per provider."""
        class LangChainish:
            def embed_documents(self, texts):
                calls.append(("documents", list(texts)))
                return [[0.1, 0.2] for _ in texts]

            def embed_query(self, text):
                calls.append(("query", text))
                return [0.9, 0.9]

        embedder = Embedder(LangChainish())
        assert embedder.embed_documents(["a"]) == [[0.1, 0.2]]
        assert embedder.embed_query("q") == [0.9, 0.9]
        assert calls == [("documents", ["a"]), ("query", "q")]

    def test_the_llamaindex_protocol_is_accepted(self, calls):
        class LlamaIndexish:
            def get_text_embedding_batch(self, texts):
                calls.append(("documents", list(texts)))
                return [[0.3, 0.4] for _ in texts]

            def get_query_embedding(self, text):
                calls.append(("query", text))
                return [0.5, 0.6]

        embedder = Embedder(LlamaIndexish())
        assert embedder.embed_documents(["a"]) == [[0.3, 0.4]]
        assert embedder.embed_query("q") == [0.5, 0.6]
        assert calls == [("documents", ["a"]), ("query", "q")]

    def test_an_unrecognized_client_names_what_is_accepted(self):
        """A refusal that lists the options, because "not recognized" on
        its own leaves the caller guessing which shape to reach for."""
        with pytest.raises(TypeError, match="is not an embedding client hopai recognizes"):
            Embedder(object())

    def test_an_embedder_is_not_a_client(self):
        # Anchored and case-sensitive: the message opens by echoing the
        # mistake back as code, `Embedder(Embedder(...))`, which is what
        # makes it recognizable at a glance. A looser match passes on a
        # message that has lost that opening.
        with pytest.raises(TypeError, match=r"^Embedder\(Embedder\(\.\.\.\)\) -- "):
            Embedder(Embedder(lambda texts: [[1.0]]))


# ---------------------------------------------------------------------
# The asymmetry: the one thing that fails silently if it is wrong
# ---------------------------------------------------------------------

class TestDocumentQueryAsymmetry:
    """Cohere, Voyage and Google embed stored text and query text
    differently. Getting it wrong raises nothing and returns nothing
    odd -- the neighbours are just quietly worse, forever. That is the
    exact failure mode this library refuses, so every provider with an
    asymmetry gets an assertion that the two sides DIFFER."""

    def test_cohere_sends_its_two_input_types(self, calls):
        class Cohereish:
            def embed(self, texts, model, input_type, embedding_types):
                calls.append(input_type)
                floats = [[1.0, 0.0]] * len(texts)
                return type("R", (), {
                    "embeddings": type("E", (), {"float_": floats})()})()

        embedder = Embedder(_named("cohere.client_v2", Cohereish)(), model="embed-v4.0")
        embedder.embed_documents(["a"])
        embedder.embed_query("q")
        assert calls == ["search_document", "search_query"]

    def test_a_batch_of_queries_stays_on_the_query_side(self, calls):
        """embed_queries() is the batched half of embed_query(), and
        the only thing making it a QUERY is one flag. Sending the
        document spelling raises nothing and quietly costs recall on
        every search vector_search_many() ranks -- which is the exact
        failure this module exists to prevent."""
        class Cohereish:
            def embed(self, texts, model, input_type, embedding_types):
                calls.append(input_type)
                return type("R", (), {"embeddings": type("E", (), {
                    "float_": [[1.0, 0.0]] * len(texts)})()})()

        embedder = Embedder(_named("cohere.client_v2", Cohereish)(), model="embed-v4.0")
        assert len(embedder.embed_queries(["q1", "q2"])) == 2
        assert calls == ["search_query"]

    def test_voyage_sends_its_two_input_types(self, calls):
        class Voyageish:
            def embed(self, texts, model, input_type):
                calls.append(input_type)
                return type("R", (), {"embeddings": [[1.0, 0.0]] * len(texts)})()

        embedder = Embedder(_named("voyageai.client", Voyageish)(), model="voyage-3")
        embedder.embed_documents(["a"])
        embedder.embed_query("q")
        assert calls == ["document", "query"]

    def test_google_sends_its_two_task_types(self, calls):
        class Models:
            def embed_content(self, model, contents, config):
                calls.append(config["task_type"])
                return type("R", (), {"embeddings": [
                    type("V", (), {"values": [1.0, 0.0]})() for _ in contents]})()

        class Googleish:
            models = Models()

        embedder = Embedder(_named("google.genai", Googleish)(),
                            model="gemini-embedding-001")
        embedder.embed_documents(["a"])
        embedder.embed_query("q")
        assert calls == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]

    def test_openai_has_no_asymmetry_and_invents_none(self, calls):
        """The other half of the rule: a provider WITHOUT an asymmetry
        must be called identically both ways. Inventing an input_type
        for OpenAI would be as wrong as omitting Cohere's."""
        class Embeddings:
            def create(self, model, input, **kwargs):
                calls.append((model, list(input), kwargs))
                return type("R", (), {"data": [
                    type("D", (), {"embedding": [1.0, 0.0]})() for _ in input]})()

        class OpenAIish:
            embeddings = Embeddings()

        embedder = Embedder(_named("openai.resources", OpenAIish)(),
                            model="text-embedding-3-small")
        embedder.embed_documents(["a"])
        embedder.embed_query("a")
        assert calls[0] == calls[1]


# ---------------------------------------------------------------------
# Native clients: model selection and dimension passthrough
# ---------------------------------------------------------------------

class TestNativeClients:
    @staticmethod
    def _openai(calls):
        class Embeddings:
            def create(self, model, input, **kwargs):
                calls.append(kwargs)
                return type("R", (), {"data": [
                    type("D", (), {"embedding": [1.0, 0.0]})() for _ in input]})()

        class OpenAIish:
            embeddings = Embeddings()
        return _named("openai.resources", OpenAIish)()

    @pytest.mark.parametrize("module,attribute", [
        ("openai.resources", "embeddings"),
        ("cohere.client_v2", "embed"),
        ("voyageai.client", "embed"),
    ])
    def test_a_provider_that_selects_a_model_demands_one(self, module, attribute):
        """Never defaulted. A silently chosen model is a silently
        different set of neighbours, and the caller cannot see it
        happened -- so hopai refuses rather than picking."""
        class Client:
            pass
        setattr(Client, attribute, (lambda *a, **k: None) if attribute == "embed"
                else type("E", (), {"create": lambda *a, **k: None})())
        with pytest.raises(ValueError, match="needs model="):
            Embedder(_named(module, Client)())

    def test_sentence_transformers_refuses_a_model_name(self):
        """It already IS the model, so model= selects nothing -- and a
        silently ignored argument is how a caller comes to believe they
        chose something."""
        class SentenceTransformerish:
            def tokenize(self, text): ...
            def encode(self, texts): return [[1.0, 0.0] for _ in texts]

        assert Embedder(SentenceTransformerish()).embed_query("q") == [1.0, 0.0]
        # Anchored: the refusal has to open by naming the class it is
        # talking about, and end by naming the rewrite.
        with pytest.raises(ValueError,
                           match=r"^Embedder: a SentenceTransformer already IS the "
                                 r"model.*SentenceTransformer\(name\) instead$"):
            Embedder(SentenceTransformerish(), model="all-MiniLM-L6-v2")

    def test_dimensions_are_passed_where_the_provider_truncates(self, calls):
        """OpenAI's text-embedding-3-* can return a shorter vector, so a
        declared field size is handed over rather than checked after the
        fact and rejected."""
        embedder = Embedder(self._openai(calls), model="text-embedding-3-small",
                            dimensions=2)
        embedder.embed_documents(["a"])
        assert calls[0] == {"dimensions": 2}

    def test_no_dimensions_means_no_argument(self, calls):
        embedder = Embedder(self._openai(calls), model="text-embedding-3-small")
        embedder.embed_documents(["a"])
        assert calls[0] == {}


# ---------------------------------------------------------------------
# Refusals that stop a bad write before it starts
# ---------------------------------------------------------------------

class TestRefusals:
    def test_empty_text_is_refused_before_the_provider_is_called(self, calls):
        """Whitespace embeds to a vector with no direction, and Near()
        already refuses an all-zero query. Catching it here names the
        offending item instead of surfacing three layers down as a
        confusing complaint about cosine."""
        embedder = Embedder(lambda texts: calls.append(texts) or [[1.0]])
        with pytest.raises(ValueError, match="item 1 is empty or whitespace"):
            embedder.embed_documents(["fine", "   "])
        assert calls == []          # the provider was never called

    def test_non_string_input_names_the_type_it_got(self):
        embedder = Embedder(lambda texts: [[1.0] for _ in texts])
        with pytest.raises(TypeError, match="item 1 is int, not a string"):
            embedder.embed_documents(["fine", 5])

    def test_a_short_answer_is_refused_rather_than_misaligned(self):
        """The worst failure this seam could have: fewer vectors back
        than texts sent would pair embeddings with the wrong ids, and
        every neighbour after that is confidently wrong."""
        embedder = Embedder(lambda texts: [[1.0, 0.0]])
        # The label is the diagnostic half: a miscount raised from inside
        # the answer coercion says nothing about WHICH call produced it
        # unless _run hands its owner down.
        with pytest.raises(
                EmbeddingError,
                match=r"^Embedder\(function\)\.embed_documents: asked for 2 "
                      r"embedding.* and got 1"):
            embedder.embed_documents(["a", "b"])
        with pytest.raises(EmbeddingError,
                           match=r"^Embedder\(function\)\.embed_query: asked for 1 "):
            Embedder(lambda texts: []).embed_query("a")

    def test_a_model_disagreeing_with_the_field_is_named(self):
        embedder = Embedder(lambda texts: [[1.0, 2.0, 3.0] for _ in texts], dimensions=2)
        with pytest.raises(EmbeddingError, match="came back with 3 dimensions"):
            embedder.embed_documents(["a"])

    def test_a_provider_failure_says_nothing_was_written(self, calls):
        """set_vectors() resolves every embed before opening its
        transaction, so this message can promise that safely -- and the
        promise is what tells a caller they may simply retry."""
        def boom(texts):
            raise RuntimeError("upstream 503")

        with pytest.raises(EmbeddingError, match="nothing was written"):
            Embedder(boom).embed_documents(["a"])

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "10", True])
    def test_batch_size_must_be_a_positive_integer(self, bad):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            Embedder(lambda texts: [[1.0]], batch_size=bad)


# ---------------------------------------------------------------------
# Batching: the provider's cap is ours to respect
# ---------------------------------------------------------------------

class TestBatching:
    def test_texts_are_chunked_to_the_batch_size(self, calls):
        embedder = Embedder(lambda texts: calls.append(len(texts)) or
                            [[1.0] for _ in texts], batch_size=2)
        assert len(embedder.embed_documents(["a", "b", "c", "d", "e"])) == 5
        assert calls == [2, 2, 1]

    def test_each_provider_gets_its_own_documented_cap(self):
        """Cohere's 96 is not OpenAI's 2048. Sending 200 inputs to a
        provider that caps at 96 is the caller's failed write, so the
        chunking has to know the difference."""
        class Cohereish:
            def embed(self, texts, model, input_type, embedding_types): ...

        class Embeddings:
            def create(self, model, input, **kwargs): ...

        class OpenAIish:
            embeddings = Embeddings()

        assert Embedder(_named("cohere.client_v2", Cohereish)(),
                        model="m").batch_size == 96
        assert Embedder(_named("openai.resources", OpenAIish)(),
                        model="m").batch_size == 2048
        # Anything unidentified gets the conservative default, not the
        # largest cap seen -- guessing high is a failed write.
        assert Embedder(lambda texts: [[1.0]]).batch_size == 96

    def test_an_explicit_batch_size_overrides_the_cap(self):
        class Embeddings:
            def create(self, model, input, **kwargs): ...

        class OpenAIish:
            embeddings = Embeddings()

        assert Embedder(_named("openai.resources", OpenAIish)(), model="m",
                        batch_size=8).batch_size == 8


# ---------------------------------------------------------------------
# The promise the whole module rests on
# ---------------------------------------------------------------------

class TestAnswerCoercion:
    """Providers disagree about the container and agree about the
    contents, so _as_vectors() has to accept several shapes -- and each
    shape is a branch nothing else exercises."""

    def test_a_numpy_like_container_is_accepted(self):
        """sentence-transformers returns an ndarray, not a list."""
        class Array:
            def tolist(self):
                return [[1.0, 2.0]]

        assert Embedder(lambda texts: Array()).embed_documents(["a"]) == [[1.0, 2.0]]

    def test_numpy_like_rows_are_accepted(self):
        class Row:
            def tolist(self):
                return [3.0, 4.0]

        assert Embedder(lambda texts: [Row()]).embed_documents(["a"]) == [[3.0, 4.0]]

    def test_a_row_object_carrying_embedding_is_unwrapped(self):
        """OpenAI answers with objects, not bare lists."""
        class Row:
            embedding = [5.0, 6.0]

        assert Embedder(lambda texts: [Row()]).embed_documents(["a"]) == [[5.0, 6.0]]

    def test_ints_become_floats(self):
        """real[] is float4; handing SQLAlchemy ints would work by luck
        and read as if the provider returned them."""
        vectors = Embedder(lambda texts: [[1, 2]]).embed_documents(["a"])
        assert vectors == [[1.0, 2.0]]
        assert all(isinstance(v, float) for v in vectors[0])

    def test_a_row_that_is_not_a_sequence_names_its_type(self):
        with pytest.raises(
                EmbeddingError,
                match=r"^Embedder\(function\)\.embed_documents: the provider returned "
                      r"str where a list of numbers"):
            Embedder(lambda texts: ["nope"]).embed_documents(["a"])


class TestDispatch:
    """_bind() picks how to call a client ONCE, at construction, so a
    mistyped client is refused at the line that got it wrong rather than
    at the first write. The order it tries things in is load-bearing."""

    def test_the_client_is_resolved_at_construction_not_at_first_use(self):
        with pytest.raises(TypeError):
            Embedder(object())          # not deferred to embed_documents()

    def test_a_native_client_beats_the_duck_typed_protocols(self, calls):
        """A real provider client may also happen to expose a method
        named like a protocol. The module name is the stronger signal --
        matching the protocol first would send OpenAI traffic through a
        path that never sets its model."""
        class Embeddings:
            def create(self, model, input, **kwargs):
                calls.append("native")
                return type("R", (), {"data": [
                    type("D", (), {"embedding": [1.0]})() for _ in input]})()

        class Hybrid:
            embeddings = Embeddings()

            def embed_documents(self, texts):
                calls.append("protocol")
                return [[9.9]]

            def embed_query(self, text):
                calls.append("protocol")
                return [9.9]

        Embedder(_named("openai.resources", Hybrid)(), model="m").embed_documents(["a"])
        assert calls == ["native"]

    def test_an_unknown_module_falls_through_to_the_protocols(self, calls):
        """Recognition is by module name, so a LangChain embedder from
        any distribution still lands on the protocol path."""
        class Anywhere:
            def embed_documents(self, texts):
                calls.append("documents")
                return [[1.0]]

            def embed_query(self, text):
                calls.append("query")
                return [1.0]

        client = _named("some.unrelated.package", Anywhere)()
        Embedder(client).embed_documents(["a"])
        assert calls == ["documents"]

    def test_cohere_answers_are_unwrapped_however_they_are_shaped(self):
        """The v2 client returns .embeddings.float_; older shapes return
        the list directly. Both are read rather than one being assumed."""
        def cohere_client(payload):
            class Cohereish:
                def embed(self, texts, model, input_type, embedding_types):
                    return payload
            return _named("cohere.client_v2", Cohereish)()

        floats = [[1.0, 0.0]]
        wrapped = type("R", (), {"embeddings": type("E", (), {"float_": floats})()})()
        # `.float` with no `.float_`. Without this third shape the second
        # getattr is never the one that answers -- for `bare` it falls
        # through to its default -- so the attribute name it looks up is
        # free to be anything at all.
        legacy = type("R", (), {"embeddings": type("E", (), {"float": floats})()})()
        bare = type("R", (), {"embeddings": floats})()
        for payload in (wrapped, legacy, bare):
            assert Embedder(cohere_client(payload),
                            model="m").embed_documents(["a"]) == floats


class TestEmbedderSurface:
    def test_repr_names_the_provider_and_model(self):
        class Embeddings:
            def create(self, model, input, **kwargs): ...

        class OpenAIish:
            embeddings = Embeddings()

        assert repr(Embedder(_named("openai.resources", OpenAIish)(), model="m")) \
            == "Embedder(openai, model='m')"
        # No provider and no model: the class name is all there is to say.
        assert repr(Embedder(lambda texts: [[1.0]])) == "Embedder(function)"

    def test_embedding_nothing_calls_nothing(self, calls):
        """An empty batch must not become a provider call -- every
        provider bills per request, and several reject an empty input."""
        embedder = Embedder(lambda texts: calls.append(texts) or [[1.0]])
        assert embedder.embed_documents([]) == []
        assert calls == []

    def test_embed_query_returns_one_vector_not_a_list_of_one(self):
        """The asymmetry in the RETURN shape, which is easy to get wrong
        because embed_query is implemented on top of the batch path."""
        assert Embedder(lambda texts: [[1.0, 2.0]]).embed_query("q") == [1.0, 2.0]

    def test_an_embedding_error_is_not_wrapped_twice(self):
        """A short answer already raises EmbeddingError with a precise
        message; catching and re-wrapping it as "the provider call
        failed" would bury the real diagnosis."""
        with pytest.raises(EmbeddingError, match="asked for 2 embedding"):
            Embedder(lambda texts: [[1.0]]).embed_documents(["a", "b"])


class TestDispatchGuardsNeedBothHalves:
    """Every provider guard is `module matches AND shape matches`, and
    each half carries its own weight.

    Drop the module half and any client with an `embeddings` attribute
    becomes an OpenAI client. Drop the shape half and a package merely
    named `cohere` does. Both send real traffic down a path that will
    call methods the client does not have -- and the failure surfaces
    from inside a provider adapter, naming nothing the caller can act
    on. The protocols are the same story: half of LangChain's pair is
    not LangChain."""

    def test_the_shape_alone_is_not_an_openai_client(self):
        class NotOpenAI:
            embeddings = object()       # right attribute, wrong provider

        with pytest.raises(TypeError, match="is not an embedding client"):
            Embedder(_named("some.other.sdk", NotOpenAI)())

    @pytest.mark.parametrize("module,attribute", [
        ("cohere.client_v2", "embed"),
        ("voyageai.client", "embed"),
        ("google.genai", "models"),
    ])
    def test_the_module_alone_is_not_that_client(self, module, attribute):
        class Bare:
            pass                        # right module, no callable surface

        assert not hasattr(Bare, attribute)
        with pytest.raises(TypeError, match="is not an embedding client"):
            Embedder(_named(module, Bare)())

    def test_half_of_the_langchain_pair_is_not_langchain(self):
        class HalfLangChain:
            def embed_documents(self, texts):
                return [[1.0]]
            # no embed_query: taking this path would crash on the first
            # search, a long way from the line that built the Embedder.

        with pytest.raises(TypeError, match="is not an embedding client"):
            Embedder(HalfLangChain())

    def test_half_of_the_llamaindex_pair_is_not_llamaindex(self):
        class HalfLlamaIndex:
            def get_query_embedding(self, text):
                return [1.0]

        with pytest.raises(TypeError, match="is not an embedding client"):
            Embedder(HalfLlamaIndex())

    def test_tokenize_alone_is_not_a_sentence_transformer(self):
        """`encode` is common; `tokenize` beside it is what makes it a
        SentenceTransformer rather than any object with an encode()."""
        class OnlyTokenize:
            def tokenize(self, text): ...

        with pytest.raises(TypeError, match="is not an embedding client"):
            Embedder(OnlyTokenize())


class TestTheModelReachesTheProvider:
    """model= is required, so it must also be USED. Nothing asserted
    that the name travelled, so every adapter could have sent None and
    silently embedded with the provider's default -- different
    neighbours, no error."""

    def test_openai_is_given_its_model(self, calls):
        class Embeddings:
            def create(self, model, input, **kwargs):
                calls.append(model)
                return type("R", (), {"data": [
                    type("D", (), {"embedding": [1.0]})() for _ in input]})()

        class OpenAIish:
            embeddings = Embeddings()

        Embedder(_named("openai.resources", OpenAIish)(),
                 model="text-embedding-3-small").embed_documents(["a"])
        assert calls == ["text-embedding-3-small"]

    def test_cohere_is_given_its_model_and_asks_for_floats(self, calls):
        """embedding_types=["float"] is not decoration: ask for the
        wrong representation and the vectors come back quantized."""
        class Cohereish:
            def embed(self, texts, model, input_type, embedding_types):
                calls.append((model, embedding_types))
                return type("R", (), {"embeddings": type("E", (), {
                    "float_": [[1.0]] * len(texts)})()})()

        Embedder(_named("cohere.client_v2", Cohereish)(),
                 model="embed-v4.0").embed_documents(["a"])
        assert calls == [("embed-v4.0", ["float"])]

    def test_voyage_is_given_its_model(self, calls):
        class Voyageish:
            def embed(self, texts, model, input_type):
                calls.append(model)
                return type("R", (), {"embeddings": [[1.0]] * len(texts)})()

        Embedder(_named("voyageai.client", Voyageish)(),
                 model="voyage-3").embed_documents(["a"])
        assert calls == ["voyage-3"]

    def test_google_is_given_its_model(self, calls):
        class Models:
            def embed_content(self, model, contents, config):
                calls.append(model)
                return type("R", (), {"embeddings": [
                    type("V", (), {"values": [1.0]})() for _ in contents]})()

        class Googleish:
            models = Models()

        Embedder(_named("google.genai", Googleish)(),
                 model="gemini-embedding-001").embed_documents(["a"])
        assert calls == ["gemini-embedding-001"]

    @pytest.mark.parametrize("module,builder", [
        ("cohere.client_v2", lambda payload: type(
            "C", (), {"embed": lambda self, texts, model, input_type,
                      embedding_types: payload})),
        ("voyageai.client", lambda payload: type(
            "V", (), {"embed": lambda self, texts, model, input_type: payload})),
    ])
    def test_an_answer_without_the_wrapper_is_still_read(self, module, builder):
        """`getattr(answer, "embeddings", answer)` -- the fallback is
        the branch that handles a client returning the list directly,
        which older and thinner wrappers do."""
        payload = [[1.0, 2.0]]
        client = _named(module, builder(payload))()
        assert Embedder(client, model="m").embed_documents(["a"]) == payload


@pytest.fixture()
def slept(monkeypatch) -> list:
    """Every backoff, without spending it. A retry test that really
    sleeps is a slow test that gets deleted."""
    waits = []
    monkeypatch.setattr(embeddings_module.time, "sleep", waits.append)
    return waits


def _raises(exc, times: int, then=None):
    """A client that fails `times` times and then succeeds."""
    state = {"calls": 0}

    def call(texts):
        state["calls"] += 1
        if state["calls"] <= times:
            raise exc
        return then if then is not None else [[1.0, 0.0] for _ in texts]
    call.state = state
    return call


class TestRetry:
    def test_a_transient_failure_is_retried_and_succeeds(self, slept):
        client = _raises(TimeoutError("upstream"), times=2)
        assert Embedder(client).embed_documents(["a"]) == [[1.0, 0.0]]
        assert client.state["calls"] == 3          # two failures, then the answer
        assert len(slept) == 2

    def test_a_terminal_failure_is_not_retried(self, slept):
        """The half that makes this correct rather than blanket: a bad
        key fails identically forever, so retrying it spends the
        caller's rate limit to arrive at the same error more slowly."""
        class AuthenticationError(Exception):
            pass

        client = _raises(AuthenticationError("bad key"), times=99)
        with pytest.raises(EmbeddingError, match="after 1 attempt"):
            Embedder(client).embed_documents(["a"])
        assert client.state["calls"] == 1
        assert slept == []

    @pytest.mark.parametrize("status,retried", [
        (429, True), (500, True), (503, True), (408, True),
        (400, False), (401, False), (403, False), (404, False), (422, False),
    ])
    def test_the_status_decides_when_there_is_one(self, slept, status, retried):
        """Status beats class name because it is unambiguous -- a
        provider that calls everything APIError still says 429."""
        class APIError(Exception):
            def __init__(self, code):
                super().__init__(f"http {code}")
                self.status_code = code

        client = _raises(APIError(status), times=1)
        run = lambda: Embedder(client, retries=1).embed_documents(["a"])  # noqa: E731
        if retried:
            assert run() == [[1.0, 0.0]]
        else:
            with pytest.raises(EmbeddingError):
                run()
        assert client.state["calls"] == (2 if retried else 1)

    def test_a_status_on_the_response_is_found_too(self, slept):
        """requests-based clients hang it off .response instead."""
        class Failure(Exception):
            response = type("R", (), {"status_code": 503})()

        client = _raises(Failure("later"), times=1)
        assert Embedder(client, retries=1).embed_documents(["a"]) == [[1.0, 0.0]]
        assert client.state["calls"] == 2

    def test_backoff_grows_and_stays_inside_its_window(self, slept):
        """Full jitter: each wait is somewhere in [0, 2^n * backoff),
        so the WINDOW doubles even though the samples are random. Two
        backfills failing at the same instant must not retry in
        lockstep and rebuild the burst that caused the 429."""
        client = _raises(TimeoutError("upstream"), times=3)
        Embedder(client, retries=3, backoff=1.0).embed_documents(["a"])
        assert len(slept) == 3
        for index, wait in enumerate(slept):
            assert 0 <= wait <= 1.0 * (2 ** index)

    def test_the_jitter_window_starts_at_zero(self, slept, monkeypatch):
        """FULL jitter, which means the low end is 0 and not the
        previous delay: `uniform(1, window)` still looks like jitter and
        still passes a bounds check, but it puts a floor under every
        wait and re-correlates the retries the jitter exists to spread.

        Asserted on the ARGUMENTS rather than on samples -- the one
        thing a random draw cannot be tested by is its own output."""
        windows = []
        monkeypatch.setattr(embeddings_module.random, "uniform",
                            lambda low, high: windows.append((low, high)) or high)
        client = _raises(TimeoutError("upstream"), times=3)
        Embedder(client, retries=3, backoff=0.5).embed_documents(["a"])
        assert windows == [(0, 0.5), (0, 1.0), (0, 2.0)]

    def test_backoff_is_capped(self, slept):
        client = _raises(TimeoutError("upstream"), times=4)
        Embedder(client, retries=4, backoff=100.0).embed_documents(["a"])
        assert all(wait <= 30.0 for wait in slept), slept

    def test_retry_after_wins_over_the_computed_backoff(self, slept):
        """The provider's own number is the only one here that is not a
        guess: backing off 0.5s against `Retry-After: 7` just spends
        another request to be told 7 again."""
        class RateLimit(Exception):
            response = type("R", (), {"headers": {"Retry-After": "7"}, "status_code": 429})()

        client = _raises(RateLimit("slow down"), times=1)
        Embedder(client, retries=1, backoff=0.5).embed_documents(["a"])
        assert slept == [7.0]

    def test_retry_after_is_read_case_insensitively(self, slept):
        """HTTP header names are case-insensitive and several clients
        hand back a plain dict, where they are not. Reading only the
        capitalised spelling silently discards the provider's own number
        and falls back to a guess."""
        class RateLimit(Exception):
            response = type("R", (), {"headers": {"retry-after": "5"}, "status_code": 429})()

        client = _raises(RateLimit("slow down"), times=1)
        Embedder(client, retries=1, backoff=0.5).embed_documents(["a"])
        assert slept == [5.0]

    @pytest.mark.parametrize("value,expected", [("0", 0.0), ("2.5", 2.5)])
    def test_a_zero_retry_after_is_honoured_not_discarded(self, slept, value, expected):
        """`Retry-After: 0` is a legal answer meaning "immediately". A
        lower bound above zero would throw it away and sleep a random
        window instead -- slower than the provider asked for."""
        class RateLimit(Exception):
            response = type("R", (), {"headers": {"Retry-After": value}, "status_code": 429})()

        client = _raises(RateLimit("now"), times=1)
        Embedder(client, retries=1, backoff=9.0).embed_documents(["a"])
        assert slept == [expected]

    @pytest.mark.parametrize("value,honoured", [("120", True), ("121", False)])
    def test_the_retry_after_cap_is_inclusive(self, slept, value, honoured):
        """120s is the documented ceiling, so a provider asking for
        exactly that is still obeyed -- an exclusive bound would throw
        away the one number here that is not a guess."""
        class RateLimit(Exception):
            response = type("R", (), {"headers": {"Retry-After": value},
                                      "status_code": 429})()

        client = _raises(RateLimit("later"), times=1)
        Embedder(client, retries=1, backoff=1.0).embed_documents(["a"])
        if honoured:
            assert slept == [120.0]
        else:
            assert slept[0] <= 1.0          # ours, not the provider's 121

    def test_a_negative_retry_after_falls_back_to_our_own_window(self, slept):
        class RateLimit(Exception):
            response = type("R", (), {"headers": {"Retry-After": "-5"}, "status_code": 429})()

        client = _raises(RateLimit("nonsense"), times=1)
        Embedder(client, retries=1, backoff=1.0).embed_documents(["a"])
        assert slept and 0 <= slept[0] <= 1.0

    def test_an_absurd_retry_after_is_not_slept_through(self, slept):
        """A provider asking for ten minutes is saying come back later,
        not hold a backfill open for ten minutes."""
        class RateLimit(Exception):
            response = type("R", (), {"headers": {"Retry-After": "600"}, "status_code": 429})()

        client = _raises(RateLimit("much later"), times=1)
        Embedder(client, retries=1, backoff=1.0).embed_documents(["a"])
        assert slept and slept[0] <= 1.0        # our own window, not 600

    def test_retries_are_exhausted_and_the_count_is_reported(self, slept):
        client = _raises(TimeoutError("upstream"), times=99)
        with pytest.raises(EmbeddingError, match=r"after 3 attempt\(s\)"):
            Embedder(client, retries=2).embed_documents(["a"])
        assert client.state["calls"] == 3

    def test_retries_zero_disables_it_entirely(self, slept):
        """The knob that stops hopai's policy multiplying with the
        client's own -- 3 attempts inside 3 is nine calls."""
        client = _raises(TimeoutError("upstream"), times=99)
        with pytest.raises(EmbeddingError, match="after 1 attempt"):
            Embedder(client, retries=0).embed_documents(["a"])
        assert client.state["calls"] == 1
        assert slept == []

    def test_every_retry_is_logged_with_its_wait(self, slept, caplog):
        client = _raises(TimeoutError("upstream"), times=1)
        with caplog.at_level(logging.WARNING, logger="hopai.embeddings"):
            Embedder(client, retries=1).embed_documents(["a"])
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        # Anchored on the owner: a message mangled at its edges still
        # contains every substring below.
        assert message.startswith("Embedder(function).embed_documents: ")
        assert "retrying in" in message
        assert "attempt 2 of 2" in message
        # Names WHAT failed, not just that something did: a retry line
        # without the exception type says nothing worth logging.
        assert "TimeoutError: upstream" in message

    def test_a_recovered_call_logs_no_failure(self, slept, caplog):
        """A retry that worked is not an error -- reporting one would
        train the reader to ignore the level that means 'rows were left
        unembedded'."""
        client = _raises(TimeoutError("upstream"), times=1)
        with caplog.at_level(logging.WARNING, logger="hopai.embeddings"):
            Embedder(client, retries=1).embed_documents(["a"])
        assert not any("failed after" in r.getMessage() for r in caplog.records)

    def test_our_own_refusals_are_never_retried(self, slept):
        """EmbeddingError from _as_vectors means the provider answered
        with something unusable, which asking again cannot fix."""
        client = _raises(TimeoutError("x"), times=0, then=[[1.0], [2.0]])
        with pytest.raises(EmbeddingError, match="asked for 1 embedding"):
            Embedder(client, retries=3).embed_documents(["a"])
        assert slept == []

    @pytest.mark.parametrize("bad", [-1, 1.5, "2", True])
    def test_a_meaningless_retry_count_is_refused(self, bad):
        with pytest.raises(ValueError, match="retries must be a non-negative integer"):
            Embedder(lambda texts: [[1.0]], retries=bad)

    @pytest.mark.parametrize("bad", [0, -1, float("inf"), "1", True])
    def test_a_meaningless_backoff_is_refused(self, bad):
        """backoff=0 would busy-loop against a rate limiter, which is
        the one thing a retry must never do."""
        with pytest.raises(ValueError, match="backoff must be a positive number"):
            Embedder(lambda texts: [[1.0]], backoff=bad)


class TestLogging:
    """The only network calls hopai makes are the only ones worth a log
    line, and an embedding failure is worth one even when it is caught:
    embed_stale() pages, so a caller can handle the error and carry on
    with rows silently unembedded."""

    def test_every_provider_call_is_logged_with_its_size(self, caplog):
        embedder = Embedder(lambda texts: [[1.0, 0.0] for _ in texts], batch_size=2)
        with caplog.at_level(logging.DEBUG, logger="hopai.embeddings"):
            embedder.embed_documents(["a", "b", "c"])
        sizes = [r.getMessage() for r in caplog.records]
        assert sizes == ["Embedder(function).embed_documents: embedding 2 text(s)",
                         "Embedder(function).embed_documents: embedding 1 text(s)"]

    def test_a_failure_logs_before_it_raises(self, caplog):
        def dies(texts):
            raise RuntimeError("provider is down")

        with caplog.at_level(logging.WARNING, logger="hopai.embeddings"), \
                pytest.raises(EmbeddingError):
            Embedder(dies).embed_documents(["a"])
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == "WARNING"
        # Names the call, how far it got, how many attempts it cost, and
        # the provider's own type -- a log line saying only "it failed"
        # is not worth writing. The counts are asserted because the log
        # builds them SEPARATELY from the exception message, so one can
        # drift while the other stays right (mutant _attempt_30).
        assert record.getMessage() == (
            "Embedder(function).embed_documents: provider call failed after 0 embedded, "
            "1 attempt(s) (RuntimeError: provider is down)")

    def test_the_failure_log_counts_the_attempts_it_really_made(self, slept, caplog):
        """Three attempts reported as three. Retryable this time, so the
        count is not trivially 1 -- and `0 embedded` is the second
        number, which says the batch got nowhere rather than partway."""
        client = _raises(TimeoutError("upstream"), times=99)
        with caplog.at_level(logging.WARNING, logger="hopai.embeddings"), \
                pytest.raises(EmbeddingError):
            Embedder(client, retries=2).embed_documents(["a"])
        failure = [r for r in caplog.records if "failed after" in r.getMessage()]
        assert len(failure) == 1
        # Whole message: the owner travels down a SECOND path here (the
        # loop running out, not a terminal error), so it can go missing
        # on one and not the other.
        assert failure[0].getMessage() == (
            "Embedder(function).embed_documents: provider call failed after 0 embedded, "
            "3 attempt(s) (TimeoutError: upstream)")

    def test_nothing_is_logged_above_debug_on_the_happy_path(self, caplog):
        """A library that chatters at INFO makes an application turn its
        logger off, which is how the WARNING above gets lost too."""
        with caplog.at_level(logging.INFO, logger="hopai.embeddings"):
            Embedder(lambda texts: [[1.0, 0.0]]).embed_documents(["a"])
        assert caplog.records == []


class TestErrorsNameTheCall:
    """The same rule the vector surface already carries: a refusal leads
    with the call that produced it. embed_documents and embed_query
    share every helper below them, so the method name is the only thing
    telling a caller which side went wrong."""

    def test_a_document_failure_names_embed_documents(self):
        with pytest.raises(ValueError, match=r"embed_documents: item 0 is empty"):
            Embedder(lambda texts: [[1.0]]).embed_documents(["  "])

    def test_a_query_failure_names_embed_query(self):
        with pytest.raises(ValueError, match=r"embed_query: item 0 is empty"):
            Embedder(lambda texts: [[1.0]]).embed_query("  ")

    def test_a_provider_failure_names_the_real_exception_type(self):
        """`type(exc).__name__` is the diagnosis -- a TimeoutError and a
        ValueError from the client mean different next steps."""
        def boom(texts):
            raise TimeoutError("upstream")

        # retries=0 keeps this about the message rather than the backoff;
        # the attempt count is asserted in TestRetry.
        with pytest.raises(EmbeddingError,
                           match=r"failed after 1 attempt\(s\) \(TimeoutError: upstream\)"):
            Embedder(boom, retries=0).embed_documents(["a"])

    def test_an_unrecognized_client_names_its_own_type(self):
        with pytest.raises(TypeError, match=r"^Embedder: dict is not an embedding"):
            Embedder({})

    @pytest.mark.parametrize("module,attribute,expected", [
        ("openai.resources", "embeddings", "an OpenAI client needs model="),
        ("cohere.client_v2", "embed", "a Cohere client needs model="),
        ("voyageai.client", "embed", "a Voyage client needs model="),
        ("google.genai", "models", "a Google GenAI client needs model="),
    ])
    def test_each_missing_model_refusal_names_its_provider(self, module, attribute,
                                                           expected):
        """Four providers share one rule; the sentence is what says
        WHICH client you built wrong."""
        class Client:
            pass
        setattr(Client, attribute, object())
        with pytest.raises(ValueError, match=rf"^Embedder: {expected}"):
            Embedder(_named(module, Client)())


class TestBatchSizeEdges:
    def test_one_is_a_legal_batch_size(self):
        """The boundary: a provider that accepts a single input per call
        is unusual but real, and `< 1` is what allows it. Off-by-one
        here would refuse a valid configuration at construction."""
        embedder = Embedder(lambda texts: [[1.0] for _ in texts], batch_size=1)
        assert embedder.batch_size == 1
        assert embedder.embed_documents(["a", "b"]) == [[1.0], [1.0]]


class TestAsyncDispatch:
    """_abind() is _bind()'s twin for clients that can be awaited, and
    it recognizes them the same way: module name plus attribute shape,
    never isinstance, still no provider import (issue #74). What differs
    is the second half of each test -- a native client is async when its
    own method is a coroutine function, and the duck-typed protocols say
    so by name.

    Every fake below is the ASYNC form of a fake TestNativeClients or
    TestProtocols already has, so the two halves cannot drift apart."""

    def test_a_plain_async_callable_is_awaited(self, calls):
        async def client(texts):
            calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

        embedder = Embedder(client)
        assert embedder.is_async
        assert run(embedder.aembed_documents(["a", "b"])) == [[1.0, 0.0], [1.0, 0.0]]
        assert run(embedder.aembed_query("q")) == [1.0, 0.0]
        assert calls == [["a", "b"], ["q"]]

    def test_a_callable_object_with_an_async_call_is_awaited(self):
        """`iscoroutinefunction(instance)` is False -- it is the OBJECT's
        own __call__ that is the `async def`. Without that second look,
        a client written as a class (which is how anything holding a
        session is written) falls back to a worker thread."""
        class Client:
            async def __call__(self, texts):
                return [[2.0] for _ in texts]

        embedder = Embedder(Client())
        assert embedder.is_async
        assert run(embedder.aembed_documents(["a"])) == [[2.0]]

    def test_openai_async_is_recognized_and_given_its_model(self, calls):
        class Embeddings:
            async def create(self, model, input, **kwargs):
                calls.append(model)
                return type("R", (), {"data": [
                    type("D", (), {"embedding": [1.0]})() for _ in input]})()

        class AsyncOpenAIish:
            embeddings = Embeddings()

        embedder = Embedder(_named("openai._client", AsyncOpenAIish)(),
                            model="text-embedding-3-small")
        assert embedder.is_async
        assert run(embedder.aembed_documents(["a"])) == [[1.0]]
        assert calls == ["text-embedding-3-small"]

    def test_cohere_async_still_sends_its_two_input_types(self, calls):
        """The asymmetry is the thing that fails silently, so the async
        adapter needs the same assertion the sync one has: getting it
        wrong raises nothing and just costs recall forever."""
        class AsyncCohereish:
            async def embed(self, texts, model, input_type, embedding_types):
                calls.append(input_type)
                return type("R", (), {"embeddings": type("E", (), {
                    "float_": [[1.0, 0.0]] * len(texts)})()})()

        embedder = Embedder(_named("cohere.client_v2", AsyncCohereish)(), model="embed-v4.0")
        run(embedder.aembed_documents(["a"]))
        run(embedder.aembed_query("q"))
        assert calls == ["search_document", "search_query"]

    def test_voyage_async_still_sends_its_two_input_types(self, calls):
        class AsyncVoyageish:
            async def embed(self, texts, model, input_type):
                calls.append(input_type)
                return type("R", (), {"embeddings": [[1.0]] * len(texts)})()

        embedder = Embedder(_named("voyageai.client", AsyncVoyageish)(), model="voyage-3")
        run(embedder.aembed_documents(["a"]))
        run(embedder.aembed_query("q"))
        assert calls == ["document", "query"]

    def test_google_is_awaited_through_its_aio_surface(self, calls):
        """google-genai keeps async on client.aio rather than in a
        second client class, so this is the one provider where both
        bindings match the same object -- and the async one has to look
        somewhere else entirely to find it."""
        class AsyncModels:
            async def embed_content(self, model, contents, config):
                calls.append(("async", config["task_type"],
                              config.get("output_dimensionality")))
                return type("R", (), {"embeddings": [
                    type("V", (), {"values": [1.0]})() for _ in contents]})()

        class Models:
            def embed_content(self, model, contents, config):
                calls.append(("sync", config["task_type"]))
                return type("R", (), {"embeddings": [
                    type("V", (), {"values": [9.0]})() for _ in contents]})()

        class Googleish:
            models = Models()
            aio = type("Aio", (), {"models": AsyncModels()})()

        embedder = Embedder(_named("google.genai", Googleish)(),
                            model="gemini-embedding-001", dimensions=1)
        assert embedder.is_async
        assert run(embedder.aembed_query("q")) == [1.0]
        assert embedder.embed_documents(["a"]) == [[9.0]]
        # Each half reached its own surface: the async one must not
        # quietly answer from the sync one, or the loop stalls again.
        # dimensions= travels with it -- Google truncates on request,
        # and an async adapter that dropped it would return the model's
        # full width and fail the field's own check instead.
        assert calls == [("async", "RETRIEVAL_QUERY", 1), ("sync", "RETRIEVAL_DOCUMENT")]

    def test_the_langchain_async_protocol_is_preferred(self, calls):
        """Every LangChain Embeddings carries both pairs; the base class
        implements the `a` half by running the sync one in an executor.
        Preferring aembed_* is what lets an integration that really is
        async be really async."""
        class LangChainish:
            def embed_documents(self, texts):
                calls.append("sync")
                return [[0.1] for _ in texts]

            def embed_query(self, text):
                calls.append("sync")
                return [0.1]

            async def aembed_documents(self, texts):
                calls.append(("async", "documents"))
                return [[0.2] for _ in texts]

            async def aembed_query(self, text):
                calls.append(("async", "query"))
                return [0.3]

        embedder = Embedder(LangChainish())
        assert run(embedder.aembed_documents(["a"])) == [[0.2]]
        assert run(embedder.aembed_query("q")) == [0.3]
        assert calls == [("async", "documents"), ("async", "query")]

    def test_the_llamaindex_async_protocol_is_accepted(self, calls):
        class LlamaIndexish:
            def get_text_embedding_batch(self, texts): ...
            def get_query_embedding(self, text): ...

            async def aget_text_embedding_batch(self, texts):
                calls.append(("async", "documents"))
                return [[0.4] for _ in texts]

            async def aget_query_embedding(self, text):
                calls.append(("async", "query"))
                return [0.5]

        embedder = Embedder(LlamaIndexish())
        assert run(embedder.aembed_documents(["a"])) == [[0.4]]
        assert run(embedder.aembed_query("q")) == [0.5]
        assert calls == [("async", "documents"), ("async", "query")]

    def test_a_method_returning_an_awaitable_is_awaited_too(self):
        """Detection reads `async def`, which is what every provider
        writes -- but a wrapper handing back a Future is spelled `def`
        and cannot be told apart without calling it. The protocols say
        what they are by NAME, so this one is recognized and the answer
        is awaited on its way out."""
        class Wrapper:
            def aembed_documents(self, texts):
                future = asyncio.get_running_loop().create_future()
                future.set_result([[7.0] for _ in texts])
                return future

            def aembed_query(self, text):
                future = asyncio.get_running_loop().create_future()
                future.set_result([7.0])
                return future

        embedder = Embedder(Wrapper())
        assert run(embedder.aembed_query("q")) == [7.0]

    def test_a_sentence_transformer_is_not_async(self):
        """It runs the model in this process -- no socket, nothing to
        await. It is also an nn.Module and therefore CALLABLE, so
        without an explicit stop the plain-callable branch would get a
        look at it and could match on an unrelated __call__."""
        class SentenceTransformerish:
            def encode(self, texts):
                return [[1.0] for _ in texts]

            def tokenize(self, text): ...

            async def __call__(self, texts):        # nn.Module.__call__
                raise AssertionError("encode() is the entry point, not __call__")

        assert not Embedder(SentenceTransformerish()).is_async

    def test_a_sync_client_of_the_same_shape_is_not_reported_as_async(self):
        """The half that keeps the fallback honest: `is_async` decides
        between awaiting the client and paying for a worker thread, so
        a SYNC client reported as async would be awaited and crash. The
        module name and the attribute are identical to the async fakes
        above -- only `async def` is not."""
        class Embeddings:
            def create(self, model, input, **kwargs):
                return type("R", (), {"data": [
                    type("D", (), {"embedding": [1.0]})() for _ in input]})()

        class OpenAIish:
            embeddings = Embeddings()

        class Cohereish:
            def embed(self, texts, model, input_type, embedding_types):
                return type("R", (), {"embeddings": [[1.0]] * len(texts)})()

        assert not Embedder(_named("openai.resources", OpenAIish)(), model="m").is_async
        assert not Embedder(_named("cohere.client_v2", Cohereish)(), model="m").is_async
        assert not Embedder(lambda texts: [[1.0]]).is_async


class TestASyncClientStillWorksOnTheAsyncPath:
    """The compatibility path (issue #74): asyncio.to_thread(), so a
    caller who passes the sync client they already had still gets a loop
    that breathes. Legitimate for THIS work and not in general -- a
    provider SDK blocked on a socket has released the GIL."""

    def test_a_sync_client_runs_in_a_worker_thread(self):
        seen = []
        embedder = Embedder(lambda texts: seen.append(threading.get_ident())
                            or [[1.0] for _ in texts])

        async def body():
            return await embedder.aembed_documents(["a"]), threading.get_ident()

        result, loop_thread = run(body())
        assert result == [[1.0]]
        assert seen and seen[0] != loop_thread

    def test_an_async_client_costs_no_thread(self):
        """The primary path, and the reason it is the primary one: a
        thread per in-flight call is exactly the ceiling hopai/asyncio.py
        chose the greenlet bridge to avoid."""
        seen = []

        async def client(texts):
            seen.append(threading.get_ident())
            return [[1.0] for _ in texts]

        async def body():
            await Embedder(client).aembed_documents(["a"])
            return threading.get_ident()

        assert seen == [run(body())]

    def test_the_thread_carries_the_same_refusals(self):
        """The fallback runs _run(), so everything that path checks is
        checked here too -- rather than a second, thinner validation
        nobody noticed was missing."""
        with pytest.raises(ValueError, match=r"embed_query: item 0 is empty"):
            run(Embedder(lambda texts: [[1.0]]).aembed_query("  "))
        with pytest.raises(EmbeddingError, match="asked for 2 embedding"):
            run(Embedder(lambda texts: [[1.0]]).aembed_documents(["a", "b"]))


class TestTheAsyncPathIsTheSameCall:
    """aembed_* is embed_* with the provider awaited -- not a second
    implementation with its own rules. Anything that could drift
    silently gets an assertion here."""

    def test_it_batches_to_the_same_cap(self, calls):
        async def client(texts):
            calls.append(len(texts))
            return [[1.0] for _ in texts]

        assert len(run(Embedder(client, batch_size=2).aembed_documents(
            ["a", "b", "c"]))) == 3
        assert calls == [2, 1]

    def test_embedding_nothing_calls_nothing(self, calls):
        async def client(texts):
            calls.append(texts)
            return [[1.0]]

        assert run(Embedder(client).aembed_documents([])) == []
        assert calls == []

    def test_aembed_query_returns_one_vector_not_a_list_of_one(self):
        async def client(texts):
            return [[1.0, 2.0]]

        assert run(Embedder(client).aembed_query("q")) == [1.0, 2.0]

    def test_aembed_queries_is_the_query_side_of_the_asymmetry(self, calls):
        class LangChainish:
            async def aembed_documents(self, texts):
                calls.append("documents")
                return [[0.1] for _ in texts]

            async def aembed_query(self, text):
                calls.append("query")
                return [0.2]

        assert run(Embedder(LangChainish()).aembed_queries(["a", "b"])) == [[0.2], [0.2]]
        assert calls == ["query", "query"]

    def test_the_declared_dimensions_are_checked(self):
        async def client(texts):
            return [[1.0, 2.0] for _ in texts]

        with pytest.raises(EmbeddingError, match="came back with 2 dimensions"):
            run(Embedder(client, dimensions=3).aembed_documents(["a"]))

    def test_every_provider_call_is_logged_with_its_size(self, caplog):
        async def client(texts):
            return [[1.0] for _ in texts]

        with caplog.at_level(logging.DEBUG, logger="hopai.embeddings"):
            run(Embedder(client, batch_size=2).aembed_documents(["a", "b", "c"]))
        assert [r.getMessage() for r in caplog.records] == [
            "Embedder(function).embed_documents: embedding 2 text(s)",
            "Embedder(function).embed_documents: embedding 1 text(s)"]


@pytest.fixture()
def awaited(monkeypatch) -> list:
    """Every async backoff, without spending it -- the async twin of
    the `slept` fixture."""
    waits = []

    async def record(delay):
        waits.append(delay)

    monkeypatch.setattr(embeddings_module.asyncio, "sleep", record)
    return waits


def _araises(exc, times: int, then=None):
    """The async twin of _raises(): fails `times` times, then answers."""
    state = {"calls": 0}

    async def call(texts):
        state["calls"] += 1
        if state["calls"] <= times:
            raise exc
        return then if then is not None else [[1.0, 0.0] for _ in texts]
    call.state = state
    return call


class TestAsyncRetry:
    """One retry POLICY, spent two ways. The numbers -- which failures
    are worth retrying, the doubling window, the cap, full jitter,
    Retry-After winning -- are shared with the sync path by construction
    (Embedder._wait_before_retry); these say the async loop really uses
    them, since a copy that drifted would be invisible to TestRetry."""

    def test_a_transient_failure_is_retried_and_succeeds(self, awaited):
        client = _araises(TimeoutError("upstream"), times=2)
        assert run(Embedder(client).aembed_documents(["a"])) == [[1.0, 0.0]]
        assert client.state["calls"] == 3
        assert len(awaited) == 2

    def test_a_terminal_failure_is_not_retried(self, awaited):
        class AuthenticationError(Exception):
            pass

        client = _araises(AuthenticationError("bad key"), times=99)
        with pytest.raises(EmbeddingError, match="after 1 attempt"):
            run(Embedder(client).aembed_documents(["a"]))
        assert client.state["calls"] == 1
        assert awaited == []

    def test_the_backoff_window_doubles_and_is_capped(self, awaited):
        client = _araises(TimeoutError("upstream"), times=4)
        run(Embedder(client, retries=4, backoff=100.0).aembed_documents(["a"]))
        assert len(awaited) == 4
        assert all(wait <= 30.0 for wait in awaited), awaited

    def test_retry_after_wins_over_the_computed_backoff(self, awaited):
        class RateLimit(Exception):
            response = type("R", (), {"headers": {"Retry-After": "7"},
                                      "status_code": 429})()

        client = _araises(RateLimit("slow down"), times=1)
        run(Embedder(client, retries=1, backoff=0.5).aembed_documents(["a"]))
        assert awaited == [7.0]

    def test_the_wait_is_awaited_not_slept(self, awaited, monkeypatch):
        """The point of the whole exercise: a retry that called
        time.sleep() would hold the event loop for the backoff as well
        as for the call."""
        def refuse(_delay):
            raise AssertionError("time.sleep() on the async path holds the loop")

        monkeypatch.setattr(embeddings_module.time, "sleep", refuse)
        client = _araises(TimeoutError("upstream"), times=1)
        run(Embedder(client, retries=1).aembed_documents(["a"]))
        assert len(awaited) == 1

    def test_every_retry_is_logged_with_its_wait(self, awaited, caplog):
        client = _araises(TimeoutError("upstream"), times=1)
        with caplog.at_level(logging.WARNING, logger="hopai.embeddings"):
            run(Embedder(client, retries=1).aembed_documents(["a"]))
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert message.startswith("Embedder(function).embed_documents: ")
        assert "retrying in" in message and "attempt 2 of 2" in message
        assert "TimeoutError: upstream" in message

    def test_a_spent_call_reports_how_far_it_got(self, awaited, caplog):
        client = _araises(TimeoutError("upstream"), times=99)
        with caplog.at_level(logging.WARNING, logger="hopai.embeddings"), \
                pytest.raises(EmbeddingError, match=r"after 3 attempt\(s\)"):
            run(Embedder(client, retries=2).aembed_documents(["a"]))
        assert any(r.getMessage() == (
            "Embedder(function).embed_documents: provider call failed after 0 embedded, "
            "3 attempt(s) (TimeoutError: upstream)") for r in caplog.records)

    def test_our_own_refusals_are_never_retried(self, awaited):
        client = _araises(TimeoutError("x"), times=0, then=[[1.0], [2.0]])
        with pytest.raises(EmbeddingError, match="asked for 1 embedding"):
            run(Embedder(client, retries=3).aembed_documents(["a"]))
        assert awaited == []

    def test_an_embedding_error_raised_inside_the_call_is_not_retried(self, awaited):
        """The refusal a binding itself raises -- an async-only client
        met by a sync call, say -- reaches the retry loop from INSIDE,
        where a blanket `except Exception` would swallow it and try
        twice more before reporting "the provider call failed"."""
        client = _araises(EmbeddingError("this client cannot answer"), times=99)
        with pytest.raises(EmbeddingError, match="^this client cannot answer$"):
            run(Embedder(client, retries=3).aembed_documents(["a"]))
        assert client.state["calls"] == 1
        assert awaited == []


class TestTheTwoHalvesMeetTheWrongWay:
    """An async client on a sync call, and a client that speaks only
    async. Both are a working client reached through the wrong half of
    the API, so both must say which half to use -- hopai never starts an
    event loop of its own to paper over it."""

    def test_an_async_client_on_a_sync_call_names_the_fix(self):
        async def client(texts):
            return [[1.0]]

        with pytest.raises(EmbeddingError, match="async embedding client") as failure:
            Embedder(client).embed_documents(["a"])
        assert "AsyncGraph" in str(failure.value)
        assert "aembed_documents" in str(failure.value)

    def test_a_native_async_client_is_refused_before_its_answer_is_read(self):
        """Caught on the RAW answer rather than where the vectors are
        read out of it: an adapter reaching into a coroutine reports
        "'coroutine' object has no attribute 'data'", which names
        nothing the caller can act on."""
        class Embeddings:
            async def create(self, model, input, **kwargs): ...

        class AsyncOpenAIish:
            embeddings = Embeddings()

        with pytest.raises(EmbeddingError, match="async embedding client"):
            Embedder(_named("openai._client", AsyncOpenAIish)(),
                     model="m").embed_documents(["a"])

    def test_an_async_only_client_is_accepted_not_refused_at_construction(self):
        """A client with only the `a` half of a protocol is a working
        async client, not a mistyped one. Refusing it at construction --
        which is what "not an embedding client hopai recognizes" would
        do -- would put it out of AsyncGraph's reach as well."""
        class AsyncOnly:
            async def aembed_documents(self, texts):
                return [[1.0] for _ in texts]

            async def aembed_query(self, text):
                return [1.0]

        embedder = Embedder(AsyncOnly())
        assert embedder.is_async
        assert run(embedder.aembed_documents(["a"])) == [[1.0]]
        with pytest.raises(EmbeddingError, match="async embedding client"):
            embedder.embed_documents(["a"])

    def test_an_unrecognized_client_is_still_refused_at_construction(self):
        """The async fallback must not widen _bind()'s refusal: an
        object with neither half of anything is still a mistake worth
        catching at the line that made it."""
        with pytest.raises(TypeError, match="is not an embedding client hopai recognizes"):
            Embedder(object())


class TestNoProviderIsImported:
    def test_no_provider_package_is_ever_imported(self):
        """hopai[openai] is a convenience alias for "hopai plus the
        client you were installing anyway", not a coupling. Recognizing
        a client by isinstance would need the import and would quietly
        make it real -- this is the same guard
        test_generates_without_importing_pgvector puts on the pgvector
        claim, and it runs AFTER every adapter above has been exercised."""
        leaked = {name for name in sys.modules
                  if name.split(".")[0] in {"openai", "cohere", "voyageai",
                                            "sentence_transformers", "langchain"}}
        assert leaked == set(), f"a provider package was imported: {sorted(leaked)}"
