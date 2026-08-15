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

import sys

import pytest

from hopai.embeddings import Embedder, EmbeddingError


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
        with pytest.raises(TypeError, match="pass the provider client"):
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
        with pytest.raises(ValueError, match="already IS the model"):
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
        with pytest.raises(EmbeddingError, match="asked for 2 embedding.* and got 1"):
            embedder.embed_documents(["a", "b"])

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
