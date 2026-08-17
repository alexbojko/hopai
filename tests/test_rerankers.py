"""
The reranking seam: which clients are accepted, how their answers are
read, and what is refused.

Every reranker client here is a FAKE whose __module__ is set to the real
distribution's name, because that is exactly how hopai recognizes one --
by module name and attribute shape, never isinstance. Faking it this way
tests the real matching logic while keeping the promise the module is
built on: hopai imports no provider package, and
test_no_provider_package_is_ever_imported holds it to that.

Two things here are worth more than the rest and are tested hardest:

  * Cohere and Voyage answer SORTED BY RELEVANCE. Reading those results
    in arrival order pairs every score with the wrong candidate and
    produces a confidently wrong ranking that nothing reports. The
    re-pairing by `.index` is what stops it.
  * A spent provider call RAISES. Falling back to the pre-rerank order
    is what most retrieval stacks do and is the exact "different answer
    with no signal" this library refuses.

No database, no network, and -- for everything but the jq-backed
document tests -- no optional extra either.

The fakes are built inside the tests rather than at module scope on
purpose: mutmut runs the whole suite twice in one process, and shared
mutable module state is what broke its baseline twice already in this
project.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
import threading
import time

import pytest

import hopai.rerankers as rerankers_module
from hopai import Hop, Near, Start
from hopai.rerankers import Rerank, RerankError


def needs_jq():
    """The jq binding is an optional extra (`hopai[rerankers]`), so a
    test that actually EVALUATES a filter skips without it -- while
    everything else here (the contract, the providers, the retries, the
    async path) runs on a base install.

    Constructing a Rerank must never require the extra, which is what
    TestTheJqExtra holds separately."""
    return pytest.importorskip("jq")


def run(coro):
    """No pytest-asyncio in this project -- test_async.py makes the same
    one-line wrapper for the same reason."""
    return asyncio.run(coro)


def _named(module: str, cls):
    """A class that claims to live in `module`, which is what _provider()
    reads. Setting __module__ is not a trick around the design -- it is
    the design: a real cohere.ClientV2 reports 'cohere.…' the same way."""
    cls.__module__ = module
    return cls


def _result(index: int, score: float):
    """One entry of a provider's `.results`, as an object with `.index`
    and `.relevance_score` -- Cohere's and Voyage's shared shape."""
    return type("Result", (), {"index": index, "relevance_score": score})()


def _answer(*pairs):
    """A provider answer: `.results`, in the order the PROVIDER chose."""
    return type("Answer", (), {"results": [_result(i, s) for i, s in pairs]})()


def _rerank(client, **options):
    """A Rerank over `client` with the one required argument filled in,
    so a test about batching does not have to restate the filter."""
    options.setdefault("document_from", ".properties.title")
    return Rerank(client, **options)


# ---------------------------------------------------------------------
# The contract: one method, whatever produced the candidates
# ---------------------------------------------------------------------

class TestTheContract:
    def test_a_plain_callable_is_accepted(self):
        """The escape hatch Embedder and Boost both have. Without it
        every local or in-house reranker would need an adapter here."""
        rerank = _rerank(lambda query, documents: [float(len(d)) for d in documents])
        assert rerank.score("q", ["aa", "bbb"]) == [2.0, 3.0]

    def test_the_query_reaches_the_client(self):
        """A reranker scores the (query, document) RELATIONSHIP. A
        binding that dropped the query would still return plausible
        numbers -- and rank by document length forever."""
        seen = []

        def client(query, documents):
            seen.append(query)
            return [1.0] * len(documents)

        _rerank(client).score("how do nodes agree?", ["a"])
        assert seen == ["how do nodes agree?"]

    def test_an_object_with_score_is_accepted(self):
        """The duck-typed protocol: anything already speaking the
        one-method contract needs no adapter at all."""
        class Reranker:
            def score(self, query, documents):
                return [0.5 for _ in documents]

        assert _rerank(Reranker()).score("q", ["a", "b"]) == [0.5, 0.5]

    def test_there_is_no_per_modality_method(self):
        """LanceDB splits this into rerank_vector/rerank_fts/rerank_hybrid.
        A reranker cannot tell where a candidate came from, so the split
        buys nothing and adds a runtime trap -- implement one, run the
        other kind of search, fail in production. One method is the
        design; a second entry point appearing here is the regression."""
        assert not [name for name in dir(Rerank)
                    if name.startswith("rerank") or name.endswith(("_vector", "_fts",
                                                                   "_hybrid"))]

    def test_the_class_docstring_states_the_contract(self):
        """`from hopai.rerankers import Rerank; help(Rerank)` is how a
        model learns this API, and the one line saying what a client must
        RETURN lived only in the module docstring -- which help(Rerank)
        never shows. Without it the reader knows a callable is accepted
        and not what it owes back."""
        doc = inspect.getdoc(Rerank)
        assert "score(query: str, documents: list[str]) -> list[float]" in doc
        assert "ONE FLOAT PER DOCUMENT, IN THE ORDER THE DOCUMENTS WERE GIVEN" in doc

    def test_the_document_from_entry_shows_a_real_candidate(self):
        """A model writes markedly better jq shown one concrete input
        than described one in prose: the filter runs against the WHOLE
        candidate dict, not just its properties, and that is a thing to
        see rather than to be told."""
        doc = inspect.getdoc(Rerank)
        for key in ('"id": 7', '"properties"', '"similarity": 0.81',
                    '"similarities"', '"boosts": {}'):
            assert key in doc, f"the example candidate lost {key}"

    def test_no_documents_costs_no_provider_call(self):
        """An empty candidate list is a real case (a filter matched
        nothing), and paying a round trip to rank nothing is a bill for
        no answer."""
        calls = []
        rerank = _rerank(lambda query, documents: calls.append(documents) or [])
        assert rerank.score("q", []) == []
        assert calls == []

    def test_an_unrecognized_client_names_what_is_accepted(self):
        """"Not recognized" on its own leaves the caller guessing which
        shape to reach for."""
        with pytest.raises(TypeError, match="is not a reranker client hopai recognizes"):
            _rerank(object())

    def test_a_rerank_is_not_a_client(self):
        # Anchored: the message opens by echoing the mistake back as
        # code, which is what makes it recognizable at a glance. It also
        # has to beat the .score branch below it -- a Rerank HAS a
        # .score, so without this check it would bind to itself.
        with pytest.raises(TypeError, match=r"^Rerank\(Rerank\(\.\.\.\)\) -- "):
            _rerank(_rerank(lambda query, documents: [1.0]))


# ---------------------------------------------------------------------
# Native clients
# ---------------------------------------------------------------------

class TestNativeClients:
    def test_cohere(self):
        calls = []

        class Cohereish:
            def rerank(self, *, model, query, documents):
                calls.append((model, query, list(documents)))
                return _answer((0, 0.9), (1, 0.1))

        client = _named("cohere.client_v2", Cohereish)()
        assert _rerank(client, model="rerank-v3.5").score("q", ["a", "b"]) == [0.9, 0.1]
        assert calls == [("rerank-v3.5", "q", ["a", "b"])]

    def test_voyage(self):
        calls = []

        class Voyageish:
            def rerank(self, query, documents, model):
                calls.append((model, query, list(documents)))
                return _answer((0, 0.4), (1, 0.6))

        client = _named("voyageai.client", Voyageish)()
        assert _rerank(client, model="rerank-2").score("q", ["a", "b"]) == [0.4, 0.6]
        assert calls == [("rerank-2", "q", ["a", "b"])]

    def test_a_cross_encoder_is_given_pairs(self):
        """CrossEncoder.predict takes [[query, document], ...] -- the
        pairing IS the model's input, so a binding that sent bare
        documents would score them against nothing."""
        seen = []

        class CrossEncoderish:
            tokenizer = object()

            def predict(self, pairs):
                seen.extend(pairs)
                return [0.2, 0.8]

        client = _named("sentence_transformers.cross_encoder", CrossEncoderish)()
        assert _rerank(client).score("q", ["a", "b"]) == [0.2, 0.8]
        assert seen == [["q", "a"], ["q", "b"]]

    def test_a_cross_encoder_answering_with_numpy_is_read(self):
        """sentence-transformers returns an ndarray, not a list. Refusing
        it over the container type would be pedantry -- _as_vectors()
        calls tolist() for the same reason."""
        class Numpyish:
            def tolist(self):
                return [0.1, 0.7]

        class CrossEncoderish:
            tokenizer = object()

            def predict(self, pairs):
                return Numpyish()

        client = _named("sentence_transformers.cross_encoder", CrossEncoderish)()
        assert _rerank(client).score("q", ["a", "b"]) == [0.1, 0.7]

    def test_predict_alone_is_not_a_cross_encoder(self):
        """`predict` is the most generic method name in machine learning.
        Both halves of the two-attribute test are load-bearing: without
        `tokenizer`, any estimator with a predict() would be handed
        (query, document) pairs it has never seen."""
        class Estimator:
            def predict(self, x):
                return [1.0]

        with pytest.raises(TypeError, match="is not a reranker client hopai recognizes"):
            _rerank(Estimator())

    def test_a_result_list_without_the_wrapper_is_still_read(self):
        """`_get(raw, "results") or raw` -- the fallback is the branch
        that handles a thinner wrapper returning the results directly."""
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return [_result(1, 0.3), _result(0, 0.7)]

        client = _named("cohere.client_v2", Cohereish)()
        assert _rerank(client, model="m").score("q", ["a", "b"]) == [0.7, 0.3]

    def test_results_as_plain_dicts_are_read(self):
        """A raw JSON body, or an SDK that never modelled the response,
        arrives as dicts. The index still has to be honoured."""
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return {"results": [{"index": 1, "relevance_score": 0.3},
                                    {"index": 0, "relevance_score": 0.7}]}

        client = _named("cohere.client_v2", Cohereish)()
        assert _rerank(client, model="m").score("q", ["a", "b"]) == [0.7, 0.3]

    def test_a_result_spelling_score_instead_of_relevance_score_is_read(self):
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return {"results": [{"index": 0, "score": 0.25}]}

        client = _named("cohere.client_v2", Cohereish)()
        assert _rerank(client, model="m").score("q", ["a"]) == [0.25]


# ---------------------------------------------------------------------
# The one that matters most
# ---------------------------------------------------------------------

class TestScoresArePairedByIndex:
    """Cohere and Voyage return results SORTED BY RELEVANCE -- that is
    their whole job. Zipping that answer against the documents that were
    sent gives every candidate somebody else's score: no exception, no
    warning, just a ranking that looks fine and is wrong. These tests
    are what stands between this module and that."""

    def test_reordered_results_are_repaired_by_index(self):
        """The provider answers best-first: document 2 scored highest,
        then document 0, then document 1. Read in arrival order the
        scores would come back [0.9, 0.7, 0.2] -- plausible, and wrong
        for every single row."""
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return _answer((2, 0.9), (0, 0.7), (1, 0.2))

        client = _named("cohere.client_v2", Cohereish)()
        scores = _rerank(client, model="m").score("q", ["a", "b", "c"])
        assert scores == [0.7, 0.2, 0.9]

    def test_reordered_voyage_results_are_repaired_too(self):
        """Same shape, same failure -- the pairing lives in one function
        so both providers cannot drift apart, and this is the test that
        would fail if one grew its own."""
        class Voyageish:
            def rerank(self, query, documents, model):
                return _answer((1, 0.95), (0, 0.05))

        client = _named("voyageai.client", Voyageish)()
        assert _rerank(client, model="m").score("q", ["a", "b"]) == [0.05, 0.95]

    def test_a_truncated_result_set_refuses(self):
        """A client configured with top_n answers about some documents
        and says nothing about the rest. There is no honest score to
        invent for a candidate the provider never read, and dropping it
        would silently shorten the ranking -- so it refuses, naming
        top_n as the thing to remove."""
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return _answer((2, 0.9), (0, 0.7))

        client = _named("cohere.client_v2", Cohereish)()
        with pytest.raises(RerankError, match="top_n"):
            _rerank(client, model="m").score("q", ["a", "b", "c"])

    def test_the_truncation_refusal_names_the_unscored_document(self):
        """WHICH document went unscored, and how many did. The list is
        built from the scores that are still None -- the ones nothing
        came back for -- and inverting that test to `is not None` names
        the documents the provider DID score instead: here it would
        report 2 unscored starting at document 0, when the only document
        without a score is 1. A caller reading that message goes looking
        at the wrong candidate, and the count that tells them how much of
        the answer is missing is wrong too. Matching only on "top_n"
        cannot see any of it."""
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return _answer((2, 0.9), (0, 0.7))

        client = _named("cohere.client_v2", Cohereish)()
        with pytest.raises(RerankError) as raised:
            _rerank(client, model="m").score("q", ["a", "b", "c"])
        assert "leaving 1 unscored (document 1 first)" in str(raised.value)

    def test_a_bad_indexed_score_names_the_rerank_and_the_document(self):
        """The two halves of `_number(value, owner, where)` at the one
        call site that knows WHICH document a score belongs to. Drop the
        owner and the message opens "None:", so nothing says which
        Rerank -- which model, which document_from -- produced it, and
        this repr is the only place the filter that built the document is
        quoted. Drop the `where` and it reads "None came back as str",
        naming no document at all: with one bad score among many, there
        is then nothing to look at. Neither loss is visible to a test
        that matches only "where a relevance score was expected"."""
        class Cohereish:
            def rerank(self, *, model, query, documents):
                # Best-first, so the unusable score arrives first and the
                # document it belongs to is 1 rather than the 0 a missing
                # `where` would coincidentally resemble.
                return {"results": [{"index": 1, "relevance_score": "high"},
                                    {"index": 0, "relevance_score": 0.5}]}

        client = _named("cohere.client_v2", Cohereish)()
        rerank = _rerank(client, model="m")
        with pytest.raises(RerankError) as raised:
            rerank.score("q", ["a", "b"])
        message = str(raised.value)
        assert message.startswith(f"{rerank!r}.score:")
        assert "the score for document 1 came back as str" in message

    @pytest.mark.parametrize("index, sent", [(7, 1), (1, 1), (2, 2), (-1, 2)])
    def test_an_index_out_of_range_refuses(self, index, sent):
        """Both ends, because only the absurd one is obvious. `index ==
        len(documents)` is the boundary the `<` exists for and a `<=`
        waves through -- one past the end reaches `scores[index]` as a
        bare IndexError, a crash where a RerankError names what the
        provider did. `-1` is the worse half: Python indexes it from the
        END, so it would quietly overwrite the LAST document's score and
        still satisfy the count check -- the confident mis-pairing this
        whole function exists to prevent, reported by nothing."""
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return _answer((index, 0.9))

        client = _named("cohere.client_v2", Cohereish)()
        with pytest.raises(RerankError, match=f"points at document {index}"):
            _rerank(client, model="m").score("q", ["a", "b"][:sent])

    def test_two_results_for_one_document_refuse(self):
        """Whichever won, some document would silently keep no score at
        all -- and the count check alone would pass."""
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return _answer((0, 0.9), (0, 0.1))

        client = _named("cohere.client_v2", Cohereish)()
        with pytest.raises(RerankError, match="both point at document 0"):
            _rerank(client, model="m").score("q", ["a", "b"])

    @pytest.mark.parametrize("index", [None, "0", 1.5, True])
    def test_a_result_without_a_usable_index_refuses(self, index):
        """Without an index there is nothing to pair against, and the
        arrival order is exactly the thing that must not be trusted.
        `True` is in the list on purpose: it is an int to Python and
        would quietly mean document 1."""
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return {"results": [{"index": index, "relevance_score": 0.5}]}

        client = _named("cohere.client_v2", Cohereish)()
        with pytest.raises(RerankError, match="no way to tell which document"):
            _rerank(client, model="m").score("q", ["a"])


class TestAnswerCoercion:
    def test_too_few_scores_refuse(self):
        """The same guard _as_vectors() puts on embeddings: a provider
        returning fewer scores than documents would pair scores with the
        wrong candidates from that point on."""
        rerank = _rerank(lambda query, documents: [1.0])
        with pytest.raises(RerankError, match=r"sent 3 document\(s\) and got 1 score"):
            rerank.score("q", ["a", "b", "c"])

    def test_too_many_scores_refuse(self):
        rerank = _rerank(lambda query, documents: [1.0, 2.0, 3.0])
        with pytest.raises(RerankError, match=r"sent 2 document\(s\) and got 3 score"):
            rerank.score("q", ["a", "b"])

    @pytest.mark.parametrize("value", [None, "0.5", True, object()])
    def test_a_non_numeric_score_refuses(self, value):
        """"0.5" would float() cleanly and hide a provider answering in
        a shape hopai has not actually been taught to read."""
        rerank = _rerank(lambda query, documents: [value])
        with pytest.raises(RerankError, match="where a relevance score was expected"):
            rerank.score("q", ["a"])

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_a_non_finite_score_refuses(self, value):
        """One NaN turns a sort into arbitrary order with nothing to see
        in the output -- the reason _clean_vector() refuses them too."""
        rerank = _rerank(lambda query, documents: [value])
        with pytest.raises(RerankError, match="cannot be ranked"):
            rerank.score("q", ["a"])

    def test_an_answer_that_is_not_a_sequence_refuses(self):
        rerank = _rerank(lambda query, documents: 0.5)
        with pytest.raises(RerankError, match="where a sequence of 1 score"):
            rerank.score("q", ["a"])

    def test_a_string_answer_is_not_a_sequence_of_scores(self):
        """A string is iterable, so without the explicit guard "0.5"
        would be read as three characters and refused for the wrong
        reason -- or, at one character, not refused at all."""
        rerank = _rerank(lambda query, documents: "x")
        with pytest.raises(RerankError, match="where a sequence of 1 score"):
            rerank.score("q", ["a"])

    def test_integer_scores_are_accepted_as_floats(self):
        """A cross-encoder trained on binary relevance answers 0/1."""
        assert _rerank(lambda query, documents: [0, 1]).score("q", ["a", "b"]) == [0.0, 1.0]


class TestTheQueryAndDocuments:
    @pytest.mark.parametrize("query", [None, 42, ["a"]])
    def test_a_non_text_query_refuses(self, query):
        """A reranker scores a query against a document by READING BOTH.
        This is the same refusal a raw-vector Near with rerank= gets, and
        it is a property of reranking rather than a hopai limitation."""
        with pytest.raises(TypeError, match="the query must be text"):
            _rerank(lambda q, d: [1.0]).score(query, ["a"])

    def test_an_empty_query_refuses(self):
        with pytest.raises(ValueError, match="scored against nothing"):
            _rerank(lambda q, d: [1.0]).score("   ", ["a"])

    def test_a_non_string_document_refuses(self):
        with pytest.raises(TypeError, match="document 1 is dict, not a string"):
            _rerank(lambda q, d: [1.0]).score("q", ["a", {"title": "b"}])

    def test_an_empty_document_refuses(self):
        """Never fall back to an empty document: scoring a candidate
        against nothing silently changes the ranking, which is precisely
        the answer nobody can see is wrong."""
        with pytest.raises(ValueError, match="document 0 is empty or whitespace"):
            _rerank(lambda q, d: [1.0]).score("q", ["  "])

    @pytest.mark.parametrize("documents", ["a document", b"a document"])
    def test_a_bare_string_is_one_document_not_a_list_of_characters(self, documents):
        """A str is iterable, so `list(documents)` turned "a document"
        into ten single-character documents and billed for all of them.
        The complaint that came back -- "document 1 is empty or
        whitespace" -- was about the SPACE, and named nothing a caller
        could act on. _plain_scores() already guards exactly this on the
        answer; the input had no guard at all."""
        with pytest.raises(TypeError, match=r"a bare string is ONE document"):
            _rerank(lambda q, d: [1.0]).score("q", documents)

    def test_a_bare_string_costs_no_provider_call(self):
        calls = []
        rerank = _rerank(lambda q, d: calls.append(d) or [1.0] * len(d))
        with pytest.raises(TypeError):
            rerank.score("q", "abc")
        assert calls == []

    def test_nothing_is_sent_when_a_document_is_invalid(self):
        """Validation happens before the call, like _clean_texts(): a
        batch about to be refused should never be paid for."""
        calls = []
        rerank = _rerank(lambda q, d: calls.append(d) or [1.0] * len(d))
        with pytest.raises(ValueError):
            rerank.score("q", ["ok", ""])
        assert calls == []


# ---------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------

class TestBatching:
    def test_a_large_call_is_split_at_the_provider_cap(self):
        """Cohere and Voyage cap a rerank call near 1000 documents.
        Chunking is ours to do -- a provider refusing 2500 should not
        become the caller's problem, and a silent truncation would drop
        candidates from the ranking without saying so."""
        sizes = []

        class Cohereish:
            def rerank(self, *, model, query, documents):
                sizes.append(len(documents))
                return _answer(*((i, float(i)) for i in range(len(documents))))

        client = _named("cohere.client_v2", Cohereish)()
        documents = [f"d{i}" for i in range(2500)]
        scores = _rerank(client, model="m").score("q", documents)
        assert sizes == [1000, 1000, 500]
        assert len(scores) == 2500

    def test_the_merged_order_is_the_documents_order(self):
        """Each batch is scored on its own, so the merge is where the
        positions could drift: batch 2's scores must land on batch 2's
        documents, not restart at zero. Each fake batch answers with its
        documents' own lengths, which are unique per position here."""
        class Cohereish:
            def rerank(self, *, model, query, documents):
                # Answered best-first, as the real one does, so the merge
                # and the index re-pairing are exercised together.
                pairs = sorted(((i, float(len(d))) for i, d in enumerate(documents)),
                               key=lambda pair: -pair[1])
                return _answer(*pairs)

        client = _named("cohere.client_v2", Cohereish)()
        documents = ["x" * (i + 1) for i in range(2500)]
        scores = _rerank(client, model="m").score("q", documents)
        assert scores == [float(len(d)) for d in documents]

    def test_batch_size_overrides_the_cap(self):
        sizes = []
        rerank = _rerank(lambda q, d: sizes.append(len(d)) or [1.0] * len(d),
                         batch_size=2)
        rerank.score("q", ["a", "b", "c", "d", "e"])
        assert sizes == [2, 2, 1]

    def test_an_unknown_client_still_batches(self):
        """A local cross-encoder or an in-house service has no published
        cap, so the default is the one every documented remote cap is --
        not embeddings.py's smaller one, which exists to pace an
        embedding endpoint."""
        assert _rerank(lambda q, d: [1.0] * len(d)).batch_size == 1000

    def test_a_failed_batch_logs_how_far_it_got(self, caplog):
        """"the reranker call failed" alone leaves a caller with 2500
        documents wondering whether it was the first call or the third
        -- and how many documents they were billed for before it died."""
        class Boom(Exception):
            pass

        state = {"calls": 0}

        def client(query, documents):
            state["calls"] += 1
            if state["calls"] == 2:
                raise Boom("nope")
            return [1.0] * len(documents)

        with caplog.at_level(logging.WARNING, logger="hopai.rerankers"), \
                pytest.raises(RerankError):
            _rerank(client, batch_size=2, retries=0).score("q", ["a", "b", "c"])
        assert any("after 2 scored" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------

class TestConstructorValidation:
    def test_document_from_is_required(self):
        with pytest.raises(TypeError, match="document_from= is required"):
            Rerank(lambda q, d: [1.0])

    def test_document_is_answered_with_document_from(self):
        """Python's own error for a misspelled keyword names the
        misspelling and not the fix. `document=` is the mistake worth
        answering by name, because it is what the parameter WOULD be
        called if the value were a document -- and it is not: it is a
        rule evaluated per candidate at execution time."""
        with pytest.raises(TypeError, match=r"the parameter is document_from=, not document="):
            Rerank(lambda q, d: [1.0], document=".properties.title")

    @pytest.mark.parametrize("keyword", ["documents", "document_for", "doc_from",
                                         "text_from", "document_text"])
    def test_every_near_miss_reaches_the_same_answer(self, keyword):
        """`document=` was answered by name and its four neighbours were
        not, so the same mistake got the useful message or a bare list of
        names depending on which word the caller reached for. Anything
        naming a document or its text is reaching for this parameter."""
        with pytest.raises(TypeError,
                           match=rf"the parameter is document_from=, not {keyword}="):
            Rerank(lambda q, d: [1.0], **{keyword: ".properties.title"})

    def test_another_unexpected_keyword_is_still_refused(self):
        with pytest.raises(TypeError, match=r"unexpected keyword argument\(s\) \['top_n'\]"):
            Rerank(lambda q, d: [1.0], document_from=".a", top_n=5)

    def test_document_from_is_keyword_only(self):
        """Positionally it would sit where a model name reads naturally,
        and `Rerank(client, "rerank-v3.5")` would compile a model name as
        a jq filter."""
        with pytest.raises(TypeError):
            Rerank(lambda q, d: [1.0], ".properties.title")

    def test_the_positional_form_gets_the_named_sentence_too(self):
        """Every sibling takes its required second argument positionally
        -- Near("summary", q), Boost("importance", 0.2),
        Embedder(client, "text-embedding-3-small") -- so this is what a
        reader of those writes. Python answers it with "takes 2
        positional arguments but 3 were given", which names neither the
        parameter nor why it is keyword-only."""
        with pytest.raises(TypeError, match=r"document_from= is keyword-only"):
            Rerank(lambda q, d: [1.0], ".properties.title")

    def test_a_callable_filter_names_the_rewrite(self):
        """`document_from=` READS like a hook, and Boost(key=...)
        genuinely accepts one -- so "must be a non-empty jq filter
        string" alone leaves the caller unsure whether hopai wants a
        different function or a different kind of value."""
        with pytest.raises(ValueError, match=r"document_from='\.properties\.title'"):
            Rerank(lambda q, d: [1.0], document_from=lambda candidate: candidate["id"])

    def test_the_required_sentinel_reads_as_required(self):
        """`help(Rerank)` and inspect.signature are how a model learns
        this API. A bare object() sentinel renders as
        `document_from: str = <object object at 0x7f...>`, which states
        the opposite of the truth about the one required parameter."""
        signature = inspect.signature(Rerank.__init__)
        assert repr(signature.parameters["document_from"].default) == "<required>"

    def test_a_wrong_arity_callable_refuses_at_construction(self):
        """Reported as a provider outage before the fix: "the reranker
        call failed after 1 attempt(s) (TypeError: <lambda>() takes 1
        positional argument but 2 were given) ... Catch RerankError and
        re-run without rerank=" -- advice that would make a typo
        permanent. The callable is the shape the notebooks use, so it is
        the most likely first mistake."""
        with pytest.raises(TypeError, match=r"cannot be called with \(query, documents\)"):
            _rerank(lambda documents: [1.0] * len(documents))

    def test_the_arity_refusal_names_the_contract(self):
        """"Wrong number of arguments" without the contract leaves the
        caller guessing which two, and in which order."""
        with pytest.raises(TypeError, match=r"score\(query: str, documents: list\[str\]\) "
                                            r"-> list\[float\]"):
            _rerank(lambda query, documents, model: [1.0])

    def test_a_callable_with_no_readable_signature_is_still_accepted(self):
        """Silent when it cannot tell, never refusing on a guess: a
        C-implemented callable or an SDK object has no arity to inspect,
        and rejecting one we merely could not read would turn a working
        client away."""
        class Opaque:
            def __call__(self, *args):
                return [1.0] * len(args[1])

            @property
            def __signature__(self):
                raise ValueError("no signature here")

        assert _rerank(Opaque()).score("q", ["a"]) == [1.0]

    def test_the_per_path_message_explains_the_case_that_happened(self):
        """The "'false' is truthy in Python" sentence is the reason a
        STRING is refused. Against per_path=1 it answers a question
        nobody asked, which reads as the library talking about someone
        else's mistake."""
        with pytest.raises(TypeError) as string_case:
            _rerank(lambda q, d: [1.0], per_path="false")
        assert "'false' is truthy in Python" in str(string_case.value)

        with pytest.raises(TypeError) as int_case:
            _rerank(lambda q, d: [1.0], per_path=1)
        assert "truthy in Python" not in str(int_case.value)
        assert "one provider call per path" in str(int_case.value)

    @pytest.mark.parametrize("value", ["", "   ", None, 42, [".a"]])
    def test_a_blank_or_non_string_filter_refuses(self, value):
        with pytest.raises(ValueError, match="must be a non-empty jq filter string"):
            Rerank(lambda q, d: [1.0], document_from=value)

    @pytest.mark.parametrize("value", [0, -1, 1.5, "50", True, None])
    def test_candidates_must_be_a_positive_integer(self, value):
        """True is in the list because `candidates=True` is 1 in Python:
        a bill for one document and a rerank that cannot reorder
        anything."""
        with pytest.raises(ValueError, match="candidates must be a positive integer"):
            _rerank(lambda q, d: [1.0], candidates=value)

    def test_candidates_defaults_to_fifty(self):
        assert _rerank(lambda q, d: [1.0]).candidates == 50

    @pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
    def test_per_path_is_checked_not_coerced(self, value):
        """The same rule `all`/`detach`/`replace` follow: "false" is
        TRUE in Python, and a JSON boolean arriving as a string is an
        ordinary tool-call failure. Coerced, it would silently select
        the mode that costs one provider call per path."""
        with pytest.raises(TypeError, match="per_path must be True or False"):
            _rerank(lambda q, d: [1.0], per_path=value)

    def test_per_path_defaults_to_the_cheap_mode(self):
        """One call per distinct node, following core.py's existing
        precedent for Near at a hop. The expensive mode is opt-in
        because it multiplies the bill by the path count."""
        assert _rerank(lambda q, d: [1.0]).per_path is False

    @pytest.mark.parametrize("value", [0, -1, 2.5, "10", True])
    def test_max_paths_must_be_a_positive_integer(self, value):
        with pytest.raises(ValueError, match="max_paths must be a positive integer"):
            _rerank(lambda q, d: [1.0], max_paths=value)

    @pytest.mark.parametrize("value", [-1, 1.5, "2", True])
    def test_retries_must_be_a_non_negative_integer(self, value):
        with pytest.raises(ValueError, match="retries must be a non-negative integer"):
            _rerank(lambda q, d: [1.0], retries=value)

    def test_retries_zero_is_allowed_and_disables_retrying(self):
        assert _rerank(lambda q, d: [1.0], retries=0).retries == 0

    @pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "0.5", True])
    def test_backoff_must_be_a_positive_number(self, value):
        with pytest.raises(ValueError, match="backoff must be a positive number"):
            _rerank(lambda q, d: [1.0], backoff=value)

    @pytest.mark.parametrize("value", [0, -1, 1.5, "10", True])
    def test_batch_size_must_be_a_positive_integer(self, value):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            _rerank(lambda q, d: [1.0], batch_size=value)

    def test_the_defaults_match_the_embedder(self):
        """Two different retry budgets for the two provider calls one
        query can make is a thing a caller would have to look up."""
        from hopai.embeddings import Embedder

        rerank = _rerank(lambda q, d: [1.0])
        embedder = Embedder(lambda texts: [[1.0]])
        assert (rerank.retries, rerank.backoff) == (embedder.retries, embedder.backoff)

    def test_repr_names_the_expensive_mode(self):
        """per_path=True multiplies the bill by the path count, so a
        repr that hid it would hide the thing worth noticing in a log."""
        assert repr(_rerank(lambda q, d: [1.0], per_path=True)) == (
            "Rerank(<lambda>, document_from='.properties.title', candidates=50, "
            "per_path=True)")

    def test_repr_names_the_provider_and_model(self):
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return _answer()

        client = _named("cohere.client_v2", Cohereish)()
        assert repr(_rerank(client, model="rerank-v3.5")) == (
            "Rerank(cohere, model='rerank-v3.5', document_from='.properties.title', "
            "candidates=50)")

    def test_repr_names_a_plain_callable_by_its_own_name(self):
        """Without the fix every callable client reprs as `function`,
        which identifies nothing -- and this repr is the `owner` prefix
        of every runtime message the module raises, so a log full of
        "Rerank(function, ...)" cannot say WHICH reranker failed."""
        def relevance(query, documents):
            return [1.0] * len(documents)

        assert repr(_rerank(relevance)).startswith("Rerank(relevance, ")

    def test_repr_carries_the_filter(self):
        """`document_from` is the argument you stare at when a ranking
        looks wrong, and it is the one this repr used to drop -- leaving
        every error message about a document silent about the rule that
        built it."""
        rerank = _rerank(lambda q, d: [1.0], document_from=".properties.body")
        assert "document_from='.properties.body'" in repr(rerank)


class TestTheModelIsRequiredWhereItSelects:
    """Embedder's judgement, for the same reason: a silently chosen
    model is a silently different ranking, and the caller cannot see it
    happened."""

    @pytest.mark.parametrize("module,method", [
        ("cohere.client_v2", "a Cohere client"),
        ("voyageai.client", "a Voyage client"),
    ])
    def test_a_hosted_client_needs_a_model(self, module, method):
        client = _named(module, type("C", (), {"rerank": lambda self, *a, **k: None}))()
        with pytest.raises(ValueError, match=f"{method} needs model="):
            _rerank(client)

    def test_a_cross_encoder_refuses_a_model(self):
        """It already IS the model, so model= has nothing to select --
        accepting it would leave a caller believing they picked one."""
        class CrossEncoderish:
            tokenizer = object()

            def predict(self, pairs):
                return []

        with pytest.raises(ValueError, match="already IS the model"):
            _rerank(_named("sentence_transformers.cross_encoder", CrossEncoderish)(),
                    model="rerank-v3.5")

    @pytest.mark.parametrize("client", [
        lambda query, documents: [1.0],
        type("S", (), {"score": lambda self, q, d: [1.0]})(),
    ])
    def test_a_client_that_chooses_its_own_model_refuses_one(self, client):
        """Ignoring it silently is the failure: a ranking that ignored
        your model= looks exactly like one that used it."""
        with pytest.raises(ValueError, match="has nothing to select"):
            _rerank(client, model="rerank-v3.5")


# ---------------------------------------------------------------------
# Failure policy -- embeddings.py's, reused rather than re-decided
# ---------------------------------------------------------------------

@pytest.fixture()
def slept(monkeypatch) -> list:
    """Every backoff, without spending it. A retry test that really
    sleeps is a slow test that gets deleted."""
    waits = []
    monkeypatch.setattr(rerankers_module.time, "sleep", waits.append)
    return waits


def _raises(exc, times: int):
    """A client that fails `times` times and then answers."""
    state = {"calls": 0}

    def client(query, documents):
        state["calls"] += 1
        if state["calls"] <= times:
            raise exc
        return [1.0 for _ in documents]
    client.state = state
    return client


class TestRetry:
    def test_a_transient_failure_is_retried_and_succeeds(self, slept):
        """A network call that gives up on one 429 is not finished
        work."""
        client = _raises(TimeoutError("upstream"), times=2)
        assert _rerank(client).score("q", ["a"]) == [1.0]
        assert client.state["calls"] == 3
        assert len(slept) == 2

    def test_a_terminal_failure_is_not_retried(self, slept):
        """A bad key fails identically forever, so retrying it spends the
        caller's rate limit to arrive at the same error more slowly."""
        class AuthenticationError(Exception):
            pass

        client = _raises(AuthenticationError("bad key"), times=99)
        with pytest.raises(RerankError, match="after 1 attempt"):
            _rerank(client).score("q", ["a"])
        assert client.state["calls"] == 1
        assert slept == []

    def test_the_classifier_is_the_embedders(self):
        """Not a second policy: hopai classifies a provider failure in
        ONE place, and a copy here is how the two drift into disagreeing
        about whether a 429 is worth another try."""
        from hopai import embeddings

        assert rerankers_module._retryable is embeddings._retryable
        assert rerankers_module._retry_after is embeddings._retry_after

    def test_retry_after_wins_over_the_computed_backoff(self, slept):
        """The provider's own number is the only one here that is not a
        guess: backing off 0.5s against `Retry-After: 7` just spends
        another request to be told 7 again."""
        class RateLimit(Exception):
            response = type("R", (), {"headers": {"Retry-After": "7"},
                                      "status_code": 429})()

        client = _raises(RateLimit("slow down"), times=1)
        _rerank(client).score("q", ["a"])
        assert slept == [7.0]

    def test_a_status_decides_when_there_is_one(self, slept):
        """Status beats class name because it is unambiguous -- a
        provider that calls everything APIError still says 429."""
        class APIError(Exception):
            status_code = 429

        client = _raises(APIError("busy"), times=1)
        assert _rerank(client, retries=1).score("q", ["a"]) == [1.0]
        assert client.state["calls"] == 2

    def test_retries_zero_leaves_it_to_the_client(self, slept):
        """The two policies MULTIPLY -- pick one side rather than paying
        3x3."""
        client = _raises(TimeoutError("upstream"), times=99)
        with pytest.raises(RerankError):
            _rerank(client, retries=0).score("q", ["a"])
        assert client.state["calls"] == 1

    def test_the_backoff_is_jittered_within_the_window(self, slept, monkeypatch):
        """Full jitter, not exact doubling: a traversal reranking at
        every hop fails at the same instant and would otherwise retry in
        lockstep, hitting a rate-limited provider with the same
        synchronised burst that caused the 429."""
        windows = []
        monkeypatch.setattr(rerankers_module.random, "uniform",
                            lambda low, high: windows.append((low, high)) or 0.0)
        _rerank(_raises(TimeoutError("x"), times=2), backoff=0.5).score("q", ["a"])
        assert windows == [(0, 0.5), (0, 1.0)]


class TestFailureRaisesAndNeverDegrades:
    """The refusal this module exists to make. Most retrieval stacks
    quietly return the pre-rerank order here, and the query keeps
    working -- with a different answer and no signal, which nothing
    downstream can notice."""

    def test_a_spent_call_raises_rather_than_returning_the_input_order(self):
        rerank = _rerank(_raises(TimeoutError("gone"), times=99), retries=1)
        with pytest.raises(RerankError) as raised:
            rerank.score("q", ["a", "b"])
        assert "refusing to fall back to the pre-rerank order" in str(raised.value)

    def test_the_providers_exception_is_kept_as_the_cause(self):
        """What makes degrading a decision a caller takes in their own
        code, where it is visible -- mirroring EmbeddingError."""
        boom = TimeoutError("gone")
        with pytest.raises(RerankError) as raised:
            _rerank(_raises(boom, times=99), retries=0).score("q", ["a"])
        assert raised.value.__cause__ is boom

    def test_no_score_list_is_returned_on_failure(self):
        """Belt and braces on the above: a partial list would be worse
        than an exception, because it would be USED."""
        rerank = _rerank(_raises(TimeoutError("gone"), times=99), retries=0, batch_size=1)
        with pytest.raises(RerankError):
            rerank.score("q", ["a", "b"])

    def test_rerank_error_is_a_runtime_error(self):
        """Callers catch RuntimeError around a query; a bare Exception
        subclass would slip through that."""
        assert issubclass(RerankError, RuntimeError)

    def test_a_rerank_error_from_the_client_is_not_wrapped_twice(self, slept):
        """A client that already speaks this module's vocabulary has
        said something specific; re-classifying it by class name would
        bury that message inside a generic "the reranker call failed"
        -- and "rerankerror" matches no retry name, so it would also be
        reported as terminal by accident rather than on purpose."""
        boom = RerankError("the local model is not loaded")

        def client(query, documents):
            raise boom

        with pytest.raises(RerankError) as raised:
            _rerank(client).score("q", ["a"])
        assert raised.value is boom
        assert slept == []

    def test_an_unusable_answer_is_not_retried(self, slept):
        """A provider that answered with the wrong number of scores has
        answered. Retrying would spend three calls to be told the same
        thing, and the RerankError already names it."""
        calls = {"n": 0}

        def client(query, documents):
            calls["n"] += 1
            return [1.0, 2.0]

        with pytest.raises(RerankError, match="refusing rather than pairing"):
            _rerank(client).score("q", ["a"])
        assert calls["n"] == 1


class TestLogging:
    def test_a_spent_call_logs_as_well_as_raising(self, caplog):
        """A traversal reranking at every hop may well be caught and
        retried a level up; a step that silently ranked nothing is
        exactly what you want in the log afterwards."""
        with caplog.at_level(logging.WARNING, logger="hopai.rerankers"), pytest.raises(RerankError):
            _rerank(_raises(TimeoutError("x"), times=99), retries=0).score("q", ["a"])
        assert any("reranker call failed" in record.getMessage() for record in caplog.records)

    def test_the_give_up_log_says_what_the_provider_actually_said(self, caplog):
        """"Fail loudly" is the whole point of _give_up, and a line
        reading "(TimeoutError: None)" is not loud -- the class name
        alone says a call timed out, not that the gateway answered 503,
        which is the half that tells you whether to retry or to look at
        your own config. Dropping the exception from the log's argument
        list leaves the class name in place, so a test matching only
        "reranker call failed" still passes while the one detail worth
        logging is gone."""
        boom = TimeoutError("upstream gateway said 503")
        with caplog.at_level(logging.WARNING, logger="hopai.rerankers"), \
                pytest.raises(RerankError) as raised:
            _rerank(_raises(boom, times=99), retries=0).score("q", ["a"])
        logged = [record.getMessage() for record in caplog.records
                  if "reranker call failed" in record.getMessage()]
        assert logged and "TimeoutError: upstream gateway said 503" in logged[0]
        # The log line and the exception are raised from the one place so
        # they can never disagree about the failure; both carry the text.
        assert "TimeoutError: upstream gateway said 503" in str(raised.value)

    def test_every_provider_call_is_visible_at_debug(self, caplog):
        """"How many documents did that query cost" is the question a
        per-document bill makes people ask."""
        with caplog.at_level(logging.DEBUG, logger="hopai.rerankers"):
            _rerank(lambda q, d: [1.0] * len(d), batch_size=1).score("q", ["a", "b"])
        assert sum("scoring" in record.getMessage() for record in caplog.records) == 2


# ---------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------

class TestAsync:
    def test_an_async_client_is_awaited(self):
        """`cohere.AsyncClientV2()` and an async callable are the shape
        an async application already holds. Wrapping one in a thread
        would hand back a coroutine nobody awaited."""
        seen = []

        async def client(query, documents):
            seen.append(query)
            return [0.5 for _ in documents]

        rerank = _rerank(client)
        assert rerank.is_async is True
        assert run(rerank.ascore("q", ["a", "b"])) == [0.5, 0.5]
        assert seen == ["q"]

    def test_a_callable_object_with_an_async_call_is_async(self):
        """`inspect.iscoroutinefunction(obj)` is False for an object
        whose `__call__` is the `async def`, so `is_async` -- a PUBLIC
        attribute -- reported the opposite of the truth, and ascore()
        paid a thread to build a coroutine it then awaited anyway.
        `rerankers._is_async_call()` exists for exactly this shape --
        the one question this module answers itself rather than sharing
        with embeddings.py, which can afford a bare
        `iscoroutinefunction` because an unrecognized shape only costs
        it a thread."""
        class AsyncCallable:
            async def __call__(self, query, documents):
                return [0.75 for _ in documents]

        rerank = _rerank(AsyncCallable())
        assert rerank.is_async is True
        assert run(rerank.ascore("q", ["a", "b"])) == [0.75, 0.75]

    def test_an_async_callable_object_refuses_sync_score(self):
        """`is_async` is what score() reads to refuse BEFORE the call.
        Reported as False, it called the client and got a coroutine back
        -- caught by the second guard, but only after the round trip was
        set up."""
        class AsyncCallable:
            async def __call__(self, query, documents):
                return [1.0]

        with pytest.raises(TypeError, match="this client's rerank call is async"):
            _rerank(AsyncCallable()).score("q", ["a"])

    def test_an_async_native_client_is_awaited_and_repaired_by_index(self):
        """The reshaping cannot happen until the answer is awaited --
        which is why invoking and reading the answer are two functions
        and not one."""
        class AsyncCohereish:
            async def rerank(self, *, model, query, documents):
                return _answer((1, 0.9), (0, 0.1))

        client = _named("cohere.client_v2", AsyncCohereish)()
        assert run(_rerank(client, model="m").ascore("q", ["a", "b"])) == [0.1, 0.9]

    def test_a_sync_client_runs_in_a_thread(self):
        """Legitimate for exactly this: a provider SDK blocked on a
        socket releases the GIL while it waits."""
        threads = []

        def client(query, documents):
            threads.append(threading.current_thread().name)
            return [1.0 for _ in documents]

        assert run(_rerank(client).ascore("q", ["a"])) == [1.0]
        assert threads and threads[0] != "MainThread"

    def test_the_event_loop_stays_responsive(self):
        """The hard requirement. A blocking client called inline would
        freeze every other task on the loop for the whole round trip --
        in a web application, that is every request, not just this one."""
        def slow(query, documents):
            time.sleep(0.3)
            return [1.0 for _ in documents]

        async def main():
            ticks = 0
            done = asyncio.Event()

            async def scoring():
                try:
                    return await _rerank(slow).ascore("q", ["a"])
                finally:
                    done.set()

            task = asyncio.ensure_future(scoring())
            while not done.is_set():
                await asyncio.sleep(0.001)
                ticks += 1
            return await task, ticks

        scores, ticks = run(main())
        assert scores == [1.0]
        # ~300 ticks if the loop ran freely, 0-1 if it was blocked. The
        # bound is loose on purpose: the assertion is "the loop kept
        # running", not a benchmark.
        assert ticks > 20, f"the loop only got {ticks} turns -- it was blocked"

    def test_a_client_returning_an_awaitable_is_awaited_too(self):
        """Not every async client is spelled `async def`: a wrapper
        returning a coroutine or a future is the other half of the
        shape, and running it in a thread only BUILDS the awaitable."""
        async def answer(documents):
            return [0.25 for _ in documents]

        class Wrapper:
            def score(self, query, documents):
                return answer(documents)

        rerank = _rerank(Wrapper())
        assert rerank.is_async is False
        assert run(rerank.ascore("q", ["a"])) == [0.25]

    def test_a_sync_score_against_an_async_client_names_ascore(self):
        """Silently running an event loop here is what breaks in the
        application that wanted async -- asyncio.run() raises when a
        loop is already running."""
        async def client(query, documents):
            return [1.0]

        with pytest.raises(TypeError, match="await ascore"):
            _rerank(client).score("q", ["a"])

    def test_a_sync_score_against_an_awaitable_answer_names_ascore(self):
        """The shape is_async cannot see at construction: caught after
        the call, and the coroutine is closed so the refusal is not
        followed by a "never awaited" warning."""
        async def answer():
            return [1.0]

        with pytest.raises(TypeError, match="await ascore"):
            _rerank(lambda query, documents: answer()).score("q", ["a"])

    def test_async_batches_the_same_way(self):
        sizes = []

        async def client(query, documents):
            sizes.append(len(documents))
            return [1.0 for _ in documents]

        run(_rerank(client, batch_size=2).ascore("q", ["a", "b", "c"]))
        assert sizes == [2, 1]

    def test_async_retries_without_blocking(self, monkeypatch):
        """The async loop is the sync one with `await asyncio.sleep` --
        the sleep IS the difference, which is why it is written twice."""
        naps = []

        async def fake_sleep(delay):
            naps.append(delay)

        monkeypatch.setattr(rerankers_module.asyncio, "sleep", fake_sleep)
        state = {"calls": 0}

        async def client(query, documents):
            state["calls"] += 1
            if state["calls"] == 1:
                raise TimeoutError("upstream")
            return [1.0 for _ in documents]

        assert run(_rerank(client).ascore("q", ["a"])) == [1.0]
        assert len(naps) == 1

    def test_async_failure_raises_rerank_error_too(self, monkeypatch):
        async def fake_sleep(delay):
            pass

        monkeypatch.setattr(rerankers_module.asyncio, "sleep", fake_sleep)
        boom = TimeoutError("gone")

        async def client(query, documents):
            raise boom

        with pytest.raises(RerankError) as raised:
            run(_rerank(client).ascore("q", ["a"]))
        assert raised.value.__cause__ is boom

    def test_async_does_not_retry_a_terminal_failure(self, monkeypatch):
        """The two loops share one classifier, and this is what would
        catch the async one drifting into retrying a bad key."""
        naps = []

        async def fake_sleep(delay):
            naps.append(delay)

        monkeypatch.setattr(rerankers_module.asyncio, "sleep", fake_sleep)
        state = {"calls": 0}

        class AuthenticationError(Exception):
            pass

        async def client(query, documents):
            state["calls"] += 1
            raise AuthenticationError("bad key")

        with pytest.raises(RerankError, match="after 1 attempt"):
            run(_rerank(client).ascore("q", ["a"]))
        assert state["calls"] == 1
        assert naps == []

    def test_async_does_not_wrap_a_rerank_error_twice(self):
        boom = RerankError("the local model is not loaded")

        async def client(query, documents):
            raise boom

        with pytest.raises(RerankError) as raised:
            run(_rerank(client).ascore("q", ["a"]))
        assert raised.value is boom

    def test_an_empty_call_costs_no_thread(self):
        assert run(_rerank(lambda q, d: [1.0]).ascore("q", [])) == []


# ---------------------------------------------------------------------
# Documents: the rule, evaluated
# ---------------------------------------------------------------------

class TestDocuments:
    """`document_from=` is a RULE, not a document: nothing about the
    documents exists when the query is written, so it is evaluated once
    per candidate at execution time -- after the SQL returns, before the
    provider call."""

    def _build(self, filter_text, **options):
        needs_jq()
        return _rerank(lambda q, d: [1.0] * len(d), document_from=filter_text, **options)

    def test_a_nested_field_is_read(self):
        rerank = self._build(".properties.author.name")
        assert rerank.build_documents(
            [{"id": "1", "properties": {"author": {"name": "Ada"}}}]) == ["Ada"]

    def test_fields_are_composed(self):
        """The canonical example, and the reason jq was chosen over a
        projection language invented here: concatenation, nesting and a
        default in one expression a model has already seen."""
        rerank = self._build('.properties.title + ": " + (.properties.summary // "")')
        candidates = [
            {"id": "1", "properties": {"title": "Raft", "summary": "consensus"}},
            {"id": "2", "properties": {"title": "Paxos"}},
        ]
        assert rerank.build_documents(candidates) == ["Raft: consensus", "Paxos: "]

    def test_a_list_is_joined(self):
        rerank = self._build('.properties.tags | join(", ")')
        assert rerank.build_documents(
            [{"id": "1", "properties": {"tags": ["db", "sql"]}}]) == ["db, sql"]

    def test_the_similarity_is_readable_too(self):
        """The filter runs against the whole candidate, not just its
        properties -- the same dict vector_search() already returns."""
        rerank = self._build('.properties.title + " (" + (.similarity|tostring) + ")"')
        assert rerank.build_documents(
            [{"id": "1", "properties": {"title": "Raft"}, "similarity": 0.5}]
        ) == ["Raft (0.5)"]

    def test_a_broken_filter_names_the_candidate(self):
        """A filter that errors mid-run has produced no document for
        THAT row, and "jq: error: Cannot index number" on its own leaves
        the caller grepping 50 candidates by hand."""
        rerank = self._build(".properties.title.name")
        with pytest.raises(ValueError, match=r"candidate id='7'"):
            rerank.build_documents([{"id": "7", "properties": {"title": "Raft"}}])

    def test_a_non_string_document_refuses(self):
        """A reranker reads TEXT. Coercing a number here would work
        until the day a filter selected an object and shipped its repr
        to a provider as the document."""
        rerank = self._build(".properties.year")
        with pytest.raises(TypeError, match="evaluated to int"):
            rerank.build_documents([{"id": "1", "properties": {"year": 2014}}])

    def test_the_type_advice_puts_the_default_before_tostring(self):
        """The old advice ("add tostring, or a default such as `// \"\"`")
        was two half-fixes that each break. Followed literally on a row
        where the property is MISSING, `| tostring` manufactures the
        string "null" and the reranker scores the WORD null -- a
        confidently wrong ranking with nothing to see. `// ""` alone
        produces an empty document, which the very next check refuses,
        so that half walked the caller into a second error."""
        rerank = self._build(".properties.year")
        with pytest.raises(TypeError) as failure:
            rerank.build_documents([{"id": "1", "properties": {"year": 2014}}])
        message = str(failure.value)
        assert '.properties.year // "unknown" | tostring' in message
        assert 'the TEXT "null"' in message

    def test_the_advised_rewrite_actually_works_on_a_missing_property(self):
        """The advice is only advice if following it produces a
        document: default first, then tostring, on the row that has no
        such property at all."""
        rerank = self._build('.properties.year // "unknown" | tostring')
        assert rerank.build_documents(
            [{"id": "1", "properties": {"year": 2014}}, {"id": "2", "properties": {}}]
        ) == ["2014", "unknown"]

    def test_one_candidate_is_answered_with_the_call_shape(self):
        """`build_documents(candidate)` iterates the dict's KEYS, so the
        filter ran against the string "id" and jq's own complaint --
        "Cannot index string with string" -- blamed the filter for a
        mistake in the call. Same class as a bare str reaching score()."""
        rerank = self._build(".properties.title")
        with pytest.raises(TypeError, match=r"candidates is a LIST of candidates"):
            rerank.build_documents({"id": "7", "properties": {"title": "Raft"}})

    def test_the_single_candidate_refusal_names_the_fix(self):
        rerank = self._build(".properties.title")
        with pytest.raises(TypeError, match=r"Pass \[candidate\]"):
            rerank.build_documents({"id": "7", "properties": {"title": "Raft"}})

    def test_an_empty_document_refuses(self):
        """Never fall back to an empty document -- it scores that
        candidate against nothing and silently changes the ranking."""
        rerank = self._build('.properties.summary // ""')
        with pytest.raises(ValueError, match="empty document"):
            rerank.build_documents([{"id": "3", "properties": {}}])

    def test_several_outputs_refuse(self):
        """`.properties.tags[]` emits one output per tag and has not
        said which is the document. Taking the first would silently pick
        one and rank against a fragment."""
        rerank = self._build(".properties.tags[]")
        with pytest.raises(ValueError, match="produced 2 outputs"):
            rerank.build_documents([{"id": "1", "properties": {"tags": ["a", "b"]}}])

    def test_no_output_refuses(self):
        rerank = self._build("empty")
        with pytest.raises(ValueError, match="produced 0 outputs"):
            rerank.build_documents([{"id": "1", "properties": {}}])

    def test_zero_outputs_and_several_outputs_get_different_advice(self):
        """One message served both branches and told a caller with NO
        document to "wrap it in [...] | join(", ") so the filter says
        which" -- nonsense for zero outputs, and zero is the commoner
        case: a select() that matched nothing, `empty`, a path through a
        key this row does not have."""
        with pytest.raises(ValueError) as none_at_all:
            self._build(".properties.tags[]").build_documents(
                [{"id": "1", "properties": {"tags": []}}])
        with pytest.raises(ValueError) as too_many:
            self._build(".properties.tags[]").build_documents(
                [{"id": "1", "properties": {"tags": ["a", "b"]}}])

        assert "join" not in str(none_at_all.value)
        assert '// "untitled"' in str(none_at_all.value)
        assert "drop that candidate before reranking" in str(none_at_all.value)
        assert "join" in str(too_many.value)

    def test_an_invalid_filter_refuses_at_construction(self):
        """`_bind()` refuses a client hopai cannot call on the line that
        names it; the filter gets the same treatment. Left lazy,
        `document_from='properties.title'` -- no leading dot, the single
        most likely thing to write -- constructed fine and failed at
        query time, a long way from the mistake."""
        needs_jq()
        with pytest.raises(ValueError, match="is not a valid jq filter"):
            _rerank(lambda q, d: [1.0], document_from=".properties.title +")

    def test_the_missing_dot_is_named(self):
        """jq's own message for it is a parse error about an undefined
        function (`properties/0 is not defined`), which does not mention
        the dot the caller left off."""
        needs_jq()
        with pytest.raises(ValueError, match=r"jq paths start with a dot, e\.g\. "
                                             r"'\.properties\.title'"):
            _rerank(lambda q, d: [1.0], document_from="properties.title")

    def test_the_filter_is_compiled_once(self, monkeypatch):
        """Measured, not assumed: compiling costs ~2.4ms against ~30us
        to evaluate a row, so a compile per candidate would be 98% of
        the cost of building 50 documents and would grow with the
        candidate count instead of being paid once. Patched BEFORE the
        Rerank exists, because the one compile now happens in
        __init__."""
        jq = needs_jq()
        compiled = []
        real = jq.compile
        monkeypatch.setattr(
            jq, "compile", lambda program: compiled.append(program) or real(program))
        rerank = self._build(".properties.title")
        candidates = [{"id": str(i), "properties": {"title": f"t{i}"}} for i in range(5)]
        assert rerank.build_documents(candidates) == [f"t{i}" for i in range(5)]
        rerank.build_documents(candidates)
        assert compiled == [".properties.title"]

    def test_the_candidates_own_dict_is_never_mutated(self):
        """The cap is about what one provider call may quote, not about
        what the graph found: the caller's result rows keep every path."""
        rerank = self._build(".properties.title", max_paths=1)
        candidate = {"id": "1", "properties": {"title": "Raft"},
                     "paths": [["a"], ["b"], ["c"]]}
        rerank.build_documents([candidate])
        assert candidate["paths"] == [["a"], ["b"], ["c"]]

    def test_max_paths_bounds_what_a_document_may_quote(self):
        """A high fan-in node can be reached hundreds of ways, and a
        document quoting all of them blows the provider's token limit --
        as an error if you are lucky, as a silent server-side truncation
        if you are not."""
        rerank = self._build(".paths | length | tostring", max_paths=2)
        candidate = {"id": "1", "paths": [["a"], ["b"], ["c"], ["d"]]}
        assert rerank.build_documents([candidate]) == ["2"]

    def test_paths_under_the_cap_are_untouched(self):
        rerank = self._build(".paths | length | tostring", max_paths=10)
        assert rerank.build_documents([{"id": "1", "paths": [["a"], ["b"]]}]) == ["2"]

    def test_documents_feed_score_unchanged(self):
        """The two halves of the module meet here: build, then score, in
        that order and outside any transaction."""
        needs_jq()
        seen = []
        rerank = _rerank(lambda q, d: seen.append(list(d)) or [1.0] * len(d),
                         document_from=".properties.title")
        documents = rerank.build_documents(
            [{"id": "1", "properties": {"title": "Raft"}}])
        rerank.score("consensus", documents)
        assert seen == [["Raft"]]


class TestTheJqExtra:
    def test_a_rerank_can_be_built_without_jq(self, monkeypatch):
        """Constructing one must not require the extra: a Rerank whose
        score() runs over documents the caller built themselves never
        touches jq, and compiling in __init__ would make that
        impossible."""
        monkeypatch.setitem(sys.modules, "jq", None)
        rerank = _rerank(lambda q, d: [1.0] * len(d))
        assert rerank.score("q", ["a"]) == [1.0]

    def test_without_the_binding_even_a_broken_filter_constructs(self, monkeypatch):
        """The eager compile is conditional on the binding being there.
        Making it unconditional would refuse a filter it could not read,
        and would break the promise above -- a score()-only Rerank on a
        base install."""
        monkeypatch.setitem(sys.modules, "jq", None)
        assert _rerank(lambda q, d: [1.0], document_from="properties.title")

    def test_a_missing_binding_names_the_extra(self, monkeypatch):
        """`No module named 'jq'` tells a caller what is absent without
        telling them what to install -- the same reason mcp.py asks for
        the SDK by name.

        The sentence is a CONTRACT, not prose: it is the only thing a
        base install ever says about `document_from=`, and it has to name
        the parameter, why the binding is wanted, and the extra to
        install -- as one uninterrupted run of text. So both halves are
        held, and held so they cannot drift apart: the message OPENS on
        the parameter, which a mangled or re-cased first literal cannot
        do, and "extra -- pip install" spans the seam between the two
        literals, which nothing can pad without breaking. Matching only
        the extra's name leaves the whole first half unheld."""
        # None in sys.modules is Python's own "this import fails" hook,
        # so the real import machinery still runs -- no fake __import__.
        monkeypatch.setitem(sys.modules, "jq", None)
        with pytest.raises(ImportError) as raised:
            _rerank(lambda q, d: [1.0]).build_documents([{"id": "1"}])
        message = str(raised.value)
        assert message.startswith("document_from= is a jq filter")
        assert "an optional extra -- pip install 'hopai[rerankers]'" in message


class TestUntrustedFilters:
    """A model MAY choose which fields the reranker reads -- that is a
    retrieval decision it is well placed to make, unlike an embedding,
    which it would have to invent. What it is not is an invitation to
    run arbitrary jq, and the gate for that lives in hopai.jqsafe, not
    here: this module only decides WHEN to ask.

    THE SAFE BEHAVIOUR IS THE DEFAULT and `trusted=True` is the opt-in,
    which is the polarity `allow_vectors=`, `all=` and `detach=` already
    keep: in hopai the DANGEROUS thing is the thing you ask for."""

    def test_the_default_is_the_safe_one(self, monkeypatch):
        """The whole point of the flip. With safety spelled
        `untrusted=True` it was opt-IN, so forgetting the flag on a
        filter that arrived over the wire ran `env.HOME` and POSTed the
        answer to a vendor. A forgotten flag must fail closed."""
        jqsafe = pytest.importorskip("hopai.jqsafe")
        needs_jq()
        calls = []
        monkeypatch.setattr(jqsafe, "validate",
                            lambda program, **kwargs: calls.append(program))
        _rerank(lambda q, d: [1.0], document_from=".properties.title").build_documents(
            [{"id": "1", "properties": {"title": "Raft"}}])
        assert calls == [".properties.title"]

    def test_a_forgotten_flag_still_refuses_an_unsafe_filter(self):
        """The same exfiltration test as below, with nothing passed at
        all -- which is what the mistake actually looks like."""
        jqsafe = pytest.importorskip("hopai.jqsafe")
        needs_jq()
        rerank = _rerank(lambda q, d: [1.0], document_from="env.DATABASE_URL")
        with pytest.raises(jqsafe.UnsafeFilter):
            rerank.build_documents([{"id": "1"}])

    def test_a_trusted_filter_is_not_validated(self, monkeypatch):
        """A human writing jq in their own Python process already runs
        arbitrary code there, so the safe subset would restrict nothing
        real -- and paying for a parse on every query would be a cost
        for nobody."""
        jqsafe = pytest.importorskip("hopai.jqsafe")
        needs_jq()
        calls = []
        monkeypatch.setattr(jqsafe, "validate",
                            lambda *args, **kwargs: calls.append(args))
        _rerank(lambda q, d: [1.0], document_from=".properties.title").build_documents(
            [{"id": "1", "properties": {"title": "Raft"}}], trusted=True)
        assert calls == []

    def test_trusted_true_really_gets_the_full_language(self):
        """`trusted=True` is the escape hatch, so it has to actually
        open: a filter the subset refuses must run when the caller says
        the filter is their own."""
        pytest.importorskip("hopai.jqsafe")
        needs_jq()
        rerank = _rerank(lambda q, d: [1.0], document_from='[range(3)] | tostring')
        assert rerank.build_documents([{"id": "1"}], trusted=True) == ["[0,1,2]"]

    def test_an_untrusted_filter_is_validated(self, monkeypatch):
        jqsafe = pytest.importorskip("hopai.jqsafe")
        needs_jq()
        calls = []
        monkeypatch.setattr(jqsafe, "validate",
                            lambda program, **kwargs: calls.append((program, kwargs)))
        rerank = _rerank(lambda q, d: [1.0], document_from=".properties.title")
        rerank.build_documents([{"id": "1", "properties": {"title": "Raft"}}],
                               fields=["properties.title"])
        assert calls == [(".properties.title",
                          {"fields": ["properties.title"], "owner": "document_from"})]

    def test_the_subset_is_parsed_once_per_field_list(self, monkeypatch):
        """The gate is a parse, and a traversal reranking at every hop
        would otherwise pay for it once per step for the same answer.
        Cached per `fields=`, not globally: a narrower allowlist has to
        be asked again."""
        jqsafe = pytest.importorskip("hopai.jqsafe")
        needs_jq()
        calls = []
        monkeypatch.setattr(jqsafe, "validate",
                            lambda program, **kwargs: calls.append(kwargs.get("fields")))
        rerank = _rerank(lambda q, d: [1.0], document_from=".properties.title")
        candidates = [{"id": "1", "properties": {"title": "Raft"}}]
        rerank.build_documents(candidates, fields=["properties.title"])
        rerank.build_documents(candidates, fields=["properties.title"])
        rerank.build_documents(candidates, fields=["properties.body"])
        assert calls == [["properties.title"], ["properties.body"]]

    def test_an_unsafe_filter_refuses_before_it_runs(self):
        """`env.DATABASE_URL` parses perfectly as jq, and the filter's
        result IS the document that gets POSTed to a third party -- so a
        model-supplied filter reading it is one-line exfiltration that
        looks, in the response, like a document that reranked oddly."""
        jqsafe = pytest.importorskip("hopai.jqsafe")
        needs_jq()
        rerank = _rerank(lambda q, d: [1.0], document_from="env.DATABASE_URL")
        with pytest.raises(jqsafe.UnsafeFilter):
            rerank.build_documents([{"id": "1"}], trusted=False)


# ---------------------------------------------------------------------
# The promise the whole module is built on
# ---------------------------------------------------------------------

class TestNoProviderIsImported:
    def test_no_provider_package_is_ever_imported(self):
        """hopai[cohere] is a convenience alias for "hopai plus the
        client you were installing anyway", not a coupling. Recognizing
        a client by isinstance would need the import and would quietly
        make it real -- the same guard test_embeddings.py puts on the
        embedding seam, and it runs AFTER every adapter above has been
        exercised."""
        leaked = {name for name in sys.modules
                  if name.split(".")[0] in {"cohere", "voyageai",
                                            "sentence_transformers", "openai"}}
        assert leaked == set(), f"a provider package was imported: {sorted(leaked)}"


class TestTheRenderedSignature:
    """What `inspect.signature(Rerank)` and `help(Rerank)` report.

    A model discovers this API by introspecting it, so the signature is
    documentation, not a detail. `__init__` really takes `*misplaced`
    and `**unexpected`, but both are refusal channels -- and rendered,
    `*misplaced` advertises positional arguments to a reader whose first
    question is whether `document_from` is one, which is the exact
    confusion keyword-only exists to prevent."""

    def test_the_refusal_channels_are_not_advertised(self):
        rendered = str(inspect.signature(Rerank))
        assert "misplaced" not in rendered and "unexpected" not in rendered

    def test_document_from_reads_as_required_and_keyword_only(self):
        """Two things a signature must not lie about: that the one
        required argument is required, and that it cannot be passed
        positionally."""
        parameters = inspect.signature(Rerank).parameters
        assert str(parameters["document_from"].default) == "<required>"
        assert parameters["document_from"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_every_advertised_parameter_is_really_accepted(self):
        """The override is hand-written, so it can drift from __init__.
        A parameter advertised but not accepted is worse than none: it is
        what a model would reach for first."""
        advertised = [name for name in inspect.signature(Rerank).parameters
                      if name != "client"]
        real = inspect.signature(Rerank.__init__).parameters
        assert [name for name in advertised if name not in real] == []

    @pytest.mark.parametrize("call, expected", [
        (lambda: Rerank(lambda query, documents: [1.0], ".a"),
         "document_from= is keyword-only"),
        (lambda: Rerank(lambda query, documents: [1.0], document=".a"),
         "the parameter is document_from=, not document="),
        (lambda: Rerank(lambda query, documents: [1.0], doc_from=".a"),
         "the parameter is document_from=, not doc_from="),
    ])
    def test_the_channels_still_answer_by_name(self, call, expected):
        """Hiding them from the signature must not disarm them -- the
        whole reason they exist is to beat Python's generic message."""
        with pytest.raises(TypeError, match=expected):
            call()


class TestRerankOnStartAndHop:
    """`rerank=` on a traversal spec, refused where the caller wrote it.

    These live here rather than beside the Near tests because they are
    the rerank feature's spec surface: hop.py holds the structural half
    of the rule (is there anything to rerank, is the query readable, do
    the two numbers agree) and rerankers.py holds the rest. A refusal
    that arrived at execution time instead would name a step the caller
    is no longer looking at."""

    @staticmethod
    def _rerank(**kwargs):
        return Rerank(lambda query, documents: [0.0] * len(documents),
                      document_from=".properties.title", **kwargs)

    def test_a_rerank_seeds_and_a_hop_beam_both_build(self):
        """The two places rerank= is allowed. If this stops constructing,
        every refusal below is testing a spec nobody can write."""
        Start(near=Near("bio", text="sql"), rerank=self._rerank(candidates=50), keep=10)
        Hop(via={"kind": "cites"}, near=Near("summary", text="raft"),
            rerank=self._rerank(candidates=50), keep=10)

    @pytest.mark.parametrize("spec", [Start, Hop])
    def test_rerank_without_near_is_refused(self, spec):
        """A reranker REORDERS a candidate list; it cannot produce one.
        Allowing it alone would send a provider call over a set nothing
        ranked -- billed, slow, and changing nothing."""
        with pytest.raises(ValueError, match="rerank= reorders the candidates near= ranks"):
            spec(rerank=self._rerank())

    @pytest.mark.parametrize("spec", [Start, Hop])
    def test_a_raw_vector_query_with_rerank_is_refused(self, spec):
        """A reranker scores a query against a document by READING both,
        and a list of floats is not something a cross-encoder can read.
        This is what reranking is, not a gap -- so it refuses rather than
        growing a second way to supply the query, which could disagree
        with the first silently."""
        with pytest.raises(ValueError, match="rerank= needs the query as TEXT"):
            spec(near=Near("bio", [0.1, 0.2]), rerank=self._rerank(), keep=5)

    def test_the_raw_vector_refusal_names_the_text_rewrite(self):
        """CLAUDE.md's rule: the message names the FIX. Without the
        rewrite spelled out, a caller has to guess that `text=` exists
        and that the field's own embed= will resolve it."""
        with pytest.raises(ValueError, match=r'Near\(.bio., text="\.\.\."\)'):
            Start(near=Near("bio", [0.1, 0.2]), rerank=self._rerank(), keep=5)

    def test_candidates_below_keep_is_refused(self):
        """Reranking a pool no bigger than what survives it cannot change
        which rows come back. Clamping silently would hide that the two
        numbers in the caller's own query disagree -- the same judgement
        `all=True` with a filter already makes."""
        with pytest.raises(ValueError, match="reranks fewer candidates than keep=10 keeps"):
            Start(near=Near("bio", text="sql"), rerank=self._rerank(candidates=5), keep=10)

    def test_an_edge_beam_is_told_it_has_no_rerank_stage(self):
        """`via_near=` ranks EDGES; rerank= reranks the NODES a hop
        reaches. Sending that caller the bare "Add near=" is the lie
        _validate_boost's docstring already refuses to tell -- they would
        follow it and silently rerank nodes, which is a different answer
        arrived at quietly."""
        with pytest.raises(ValueError, match="via_near= ranks the EDGES it walks"):
            Hop(via_near=Near("edgevec", text="cites"), via_keep=5, rerank=self._rerank())

    def test_the_edge_beam_refusal_is_not_the_bare_one(self):
        """The two refusals must stay textually distinguishable, the same
        way the two "not usable yet" vector refusals are: a caller who
        passed via_near= must not be handed advice written for someone
        who passed nothing."""
        with pytest.raises(ValueError) as edge_beam:
            Hop(via_near=Near("edgevec", text="cites"), via_keep=5, rerank=self._rerank())
        assert "reorders the candidates near= ranks" not in str(edge_beam.value)

    def test_candidates_equal_to_keep_is_refused_on_a_step(self):
        """The boundary is `<=` on a Start/Hop, not `<`. Reranking
        exactly as many as you keep DOES reorder them, but a traversal
        returns a subgraph and discards that order, so the same `keep`
        nodes continue the walk whichever score sorted them -- a
        provider call billed per document for a guaranteed no-op.
        vector_search() keeps the `<` boundary, because it reports the
        new order and a rerank_score."""
        with pytest.raises(ValueError,
                           match="reranks exactly as many candidates as keep=10 keeps"):
            Start(near=Near("bio", text="sql"), rerank=self._rerank(candidates=10), keep=10)

    def test_a_multivector_near_refuses_if_any_field_carries_a_vector(self):
        """One text= field does not make the query readable: the refusal
        has to see every Near, or a two-field query would rerank against
        whichever one happened to be first."""
        with pytest.raises(ValueError, match="rerank= needs the query as TEXT"):
            Start(near=[Near("bio", text="sql"), Near("summary", [0.1, 0.2])],
                  rerank=self._rerank(), keep=5)


#: Candidate rows the pruning proof below runs every accepted filter
#: against. Each one is a shape the analysis could get wrong, not a
#: sample of realistic data: a missing key must stay MISSING (jq's `//`
#: and `Required` both read absent and null as different things), an
#: intermediate that is not an object must be handed over whole so
#: libjq's own "Cannot index" survives, and a property name containing a
#: dot must not be confused with a nested path -- `paths_read()` reports
#: both as `a.b`.
PRUNING_ROWS = (
    {"id": "7",
     "properties": {"title": "Raft", "summary": "a consensus protocol", "type": "paper",
                    "name": "r", "n": 3, "tags": ["a", "b", "c"], "ssn": "123",
                    "odd key": {"title": "deep"}, "body": "x" * 200},
     "quoted key": "qk", "a": {"b": "ab"}, "c": "c",
     "similarity": 0.8, "similarities": {"summary": 0.8}, "boosts": {}},
    {"id": "8", "properties": {}},                       # every key missing
    {"id": "9"},                                         # properties itself missing
    {},                                                  # nothing at all
    {"id": "10", "properties": {"title": None, "tags": None, "n": None, "summary": None}},
    {"id": "11", "properties": "not an object"},         # jq: Cannot index string
    {"id": "12", "properties": ["a", "b"]},              # jq: Cannot index array
    {"id": "13", "properties": 42},
    {"id": "14", "properties": {"title": "T", "n": 0, "summary": "",
                                "tags": [{"name": "n1"}, {"name": "n2"}]}},
    {"id": "15", "properties": {"title": "", "tags": [], "summary": "s", "n": 1}},
    {"id": "16", "properties": {"title": "T"},
     "paths": [[{"id": "1", "properties": {"title": "P"}}]]},
    # The ambiguity: `."a.b"` and `.a.b` are the SAME reported path.
    {"id": "17", "properties": {"title": "T"}, "a.b": "a literal dotted key",
     "a": {"b": "nested"}},
    {"id": "18", "properties": {"title": "T", "n": 1.5, "flag": True,
                                "tags": [[1, 2], [3]], "odd key": {"title": "x", "z": 1}}},
)


#: Shapes ACCEPTED does not happen to contain, and each is a place the
#: pruner could get it wrong on its own: a filter reading a parent AND
#: one of its children (the two must not fight over one slot), and one
#: reading a property whose NAME contains a dot (indistinguishable, in
#: what `paths_read()` reports, from a nested path).
PRUNING_FILTERS = (
    '(.properties | tojson) + (.properties.title // "")',
    '.properties.title + (.properties | has("summary") | tostring)',
    '(.properties | length | tostring) + (.properties.tags | tostring)',
    '."a.b" // "none"',
    '.a.b // "none"',
    '(."a.b" // "") + (.a.b // "")',
    '.properties."odd key".title // "none"',
    '[.properties.title, .properties.summary, .id] | tostring',
)


def _both_ways(program, row):
    """(pruned answer, unpruned answer) for one filter and one row, as
    values a mismatch can be printed from.

    Raw `jq.compile()` rather than build_documents(): the fallback in
    `_evaluate()` re-runs a FAILING evaluation against the whole
    candidate, which is exactly right for the library and would make
    this check vacuous on every row where the filter errors. Here the
    two evaluations must agree with nothing catching them -- error text
    included, because jq quotes the offending value into it."""
    import json

    import jq
    from hopai.rerankers import _WHOLE_ROW, _projection_paths, _pruned

    paths = _projection_paths(program)
    if paths is _WHOLE_ROW:
        return None, None
    compiled = jq.compile(program)

    def answer(value):
        try:
            return "ok", json.dumps(compiled.input_value(value).all())
        except Exception as exc:                          # noqa: BLE001 -- compared, not handled
            return "err", f"{type(exc).__name__}: {exc}"

    return answer(_pruned(row, paths)), answer(row)


class TestPruningPreservesEveryDocument:
    """The correctness bar for the performance fix, and the reason it is
    a proof rather than an assumption.

    `build_documents()` hands jq only the paths `document_from` reads,
    because `input_value()` marshals whatever it is given and the cost
    was therefore proportional to the ROW rather than to the projection
    -- 121ms of held event loop for fifty 100KB candidates. That makes
    `jqsafe.paths_read()` load-bearing for CORRECTNESS as well as for
    the `fields=` allowlist: an under-reported path would silently build
    a document from `null`, which is the confidently-wrong answer this
    library exists to refuse.

    So every program in test_jqsafe's ACCEPTED corpus -- the same corpus
    the differential and termination tests run, so a construct added to
    the grammar is covered here automatically -- is evaluated against
    both the pruned candidate and the whole one, and the two answers
    must be identical. Not "close": the same JSON, or the same error
    with the same text."""

    def test_every_accepted_program_answers_the_same_pruned(self):
        needs_jq()
        from test_jqsafe import ACCEPTED

        compared = 0
        for program in ACCEPTED + PRUNING_FILTERS:
            for row in PRUNING_ROWS:
                pruned, whole = _both_ways(program, row)
                if pruned is None:
                    continue                              # reads `.`; not pruned at all
                compared += 1
                assert pruned == whole, (
                    f"{program!r} on {row!r}: pruned {pruned!r}, whole {whole!r}")
        # A floor, so the check cannot quietly become vacuous if
        # _projection_paths() ever starts declining to prune everything.
        assert compared > 500, f"only {compared} (program, row) pairs were pruned at all"

    def test_the_check_can_fail(self):
        """A proof that cannot fail proves nothing. Pruning to the wrong
        path is exactly the bug this guards, and it produces `null`
        rather than an error -- which is why the document, not the
        exception, is what is compared above."""
        import jq

        from hopai.rerankers import _pruned

        row = {"properties": {"title": "Raft"}}
        assert jq.compile(".properties.title").input_value(row).all() == ["Raft"]
        assert jq.compile(".properties.title").input_value(
            _pruned(row, ("properties.summary",))).all() == [None]

    def test_a_missing_key_stays_missing_rather_than_becoming_null(self):
        """The one difference a caller would never see in a document and
        would see in the RANKING: `.properties.title // "untitled"` takes
        its default on a missing key AND on a null one, but
        `.properties | has("title")` does not, and a pruner that filled
        in the paths it was asked for would answer the second wrongly on
        every row."""
        from hopai.rerankers import _pruned

        assert _pruned({"properties": {}}, ("properties.title",)) == {"properties": {}}
        assert _pruned({}, ("properties.title",)) == {}
        assert _pruned({"properties": {"title": None}}, ("properties.title",)) == \
            {"properties": {"title": None}}

    def test_a_value_that_cannot_be_descended_into_is_kept_whole(self):
        """`paths_read()` collapses indexing -- `.properties.tags[0].name`
        is reported as `properties.tags.name` -- so the walk arrives at a
        LIST with a segment still to go. Keeping the list entire is what
        makes that collapse safe; descending into it would drop every
        element."""
        from hopai.rerankers import _pruned

        row = {"properties": {"tags": [{"name": "a"}, {"name": "b"}], "ssn": "123"}}
        assert _pruned(row, ("properties.tags.name",)) == \
            {"properties": {"tags": [{"name": "a"}, {"name": "b"}]}}
        # A string, where libjq's own "Cannot index string" has to survive.
        assert _pruned({"properties": "text"}, ("properties.title",)) == \
            {"properties": "text"}

    def test_a_property_name_containing_a_dot_keeps_both_readings(self):
        """`."a.b"` and `.a.b` are reported identically, so the pruner
        cannot tell them apart -- and picking one would build a document
        from `null` on rows shaped the other way. It keeps both, because
        over-keeping only costs bytes."""
        from hopai.rerankers import _pruned

        row = {"a.b": "flat", "a": {"b": "nested", "ssn": "123"}, "other": "dropped"}
        assert _pruned(row, ("a.b",)) == {"a.b": "flat", "a": {"b": "nested"}}

    def test_a_whole_key_wins_over_a_partial_one(self):
        """`properties` and `properties.title` reported together must not
        fight over one slot -- the wider read is the one that has to
        survive."""
        from hopai.rerankers import _pruned, _projection_paths

        row = {"properties": {"title": "T", "summary": "S"}}
        assert _pruned(row, ("properties", "properties.title")) == row
        # ...and _projection_paths() drops the redundant one up front.
        assert _projection_paths(
            '.properties | select(.type == "x") | tojson') == ("properties",)

    def test_a_filter_reading_the_whole_row_is_not_pruned(self):
        """`.` is the case the analysis cannot narrow, and the rule is
        correct-and-slow over fast-and-wrong."""
        from hopai.rerankers import _WHOLE_ROW, _projection_paths

        assert _projection_paths(".") is _WHOLE_ROW
        assert _projection_paths("tojson") is _WHOLE_ROW
        assert _projection_paths(".properties.title") == ("properties.title",)

    def test_a_filter_the_subset_cannot_parse_is_not_pruned(self):
        """`trusted=True` gets the full jq language, which `jqsafe` does
        not parse -- so there is no reported set to trust and nothing may
        be taken away."""
        from hopai.rerankers import _WHOLE_ROW, _projection_paths

        assert _projection_paths("to_entries | map(.value) | join(\" \")") is _WHOLE_ROW
        assert _projection_paths("[paths] | tojson") is _WHOLE_ROW

    def test_a_trusted_full_language_filter_still_builds_its_document(self):
        """The end-to-end half of the above: an unprunable filter is not
        a refused one."""
        needs_jq()
        rerank = _rerank(lambda q, d: [1.0] * len(d),
                         document_from='[.properties | to_entries[] | .value] | join(" ")')
        assert rerank.build_documents(
            [{"id": "1", "properties": {"a": "x", "b": "y"}}], trusted=True) == ["x y"]

    def test_a_non_dict_candidate_is_handed_over_untouched(self):
        """The filter runs against whatever the caller passed, and a bare
        value has no paths to prune -- the refusal it earns must be the
        one it always earned."""
        needs_jq()
        rerank = _rerank(lambda q, d: [1.0] * len(d), document_from=".properties.title")
        with pytest.raises(ValueError, match="candidate 'a string'"):
            rerank.build_documents(["a string"])

    def test_an_error_names_the_whole_candidate_not_the_pruned_view(self):
        """jq quotes the offending VALUE into its error text, so a
        message built from the view would describe a row the caller does
        not have. `_evaluate()` re-runs a failure against the whole
        candidate before reporting it, which keeps the message
        byte-identical to the one raised before pruning existed."""
        needs_jq()
        rerank = _rerank(lambda q, d: [1.0] * len(d), document_from=".properties.title")
        with pytest.raises(ValueError) as failed:
            rerank.build_documents([{"id": "9", "properties": "not an object"}])
        assert "candidate id='9'" in str(failed.value)
        assert "Cannot index string with" in str(failed.value)


class TestPruningSurvivesGeneratedFilters:
    """The corpus above is hand-written, so it can only cover the
    constructs someone thought of. This composes filters and rows at
    random from the same grammar and asserts the same equality --
    bounded and seeded, so a failure is reproducible and the suite's
    runtime is not.

    The precedent is test_jqsafe's own fuzz run against libjq, which is
    where `1.a` was found: 37,000 generated programs, one divergence,
    and it was a real one."""

    FIELDS = ("title", "summary", "tags", "n", "type", "missing", "odd key")
    LEAVES = ("Raft", "", None, 3, 1.5, True, ["a", "b"], [], {"name": "x"}, {})

    def _row(self, rng):
        properties = {name: rng.choice(self.LEAVES)
                      for name in self.FIELDS if rng.random() < 0.7}
        row = {"id": str(rng.randrange(100)), "properties": properties}
        if rng.random() < 0.3:
            row["properties"] = rng.choice(("a string", 7, ["x"], None))
        if rng.random() < 0.3:
            row["paths"] = [[{"id": "1", "properties": {"title": "P"}}]]
        if rng.random() < 0.2:
            row["a.b"] = "flat"
            row["a"] = {"b": "nested"}
        return row

    def _filter(self, rng, depth=0):
        base = rng.choice((
            ".properties", ".properties.title", ".properties.tags", ".properties.n",
            '.properties."odd key"', ".properties.tags[]", ".properties.tags[0]",
            ".properties.tags[-1]", ".properties.tags[0:2]", ".properties.missing",
            ".properties.tags[0].name", ".paths", ".a.b", '.a."b"', ".id",
        ))
        if rng.random() < 0.4:
            base += "?"
        if depth < 2 and rng.random() < 0.6:
            base += rng.choice((
                " | tostring", " | length", " | type", " | tojson", " | not",
                " | strings", " | values", " | first", " | last", " | sort",
                " | reverse", " | unique", " | add", " | arrays", " | objects",
                ' | has("title")', ' | startswith("R")', ' | join(", ")',
                " | map(tostring)", ' | select(. != null)', " | ascii_downcase",
                ' | ltrimstr("R")', ' | split(" ")', " | numbers",
            ))
        if depth < 1 and rng.random() < 0.4:
            joiner = rng.choice((" + ", ", ", " // "))
            return f"({base}){joiner}({self._filter(rng, depth + 1)})"
        if depth < 1 and rng.random() < 0.2:
            return f'"text: \\({base})"'
        return base

    def test_generated_filters_answer_the_same_pruned(self):
        import random

        from hopai.jqsafe import UnsafeFilter, is_total

        needs_jq()
        rng = random.Random(20250817)                    # seeded: a failure reproduces
        rows = [self._row(rng) for _ in range(30)]
        checked = 0
        for _ in range(1500):
            program = self._filter(rng)
            try:
                if not is_total(program):
                    continue                             # outside the subset; not pruned
            except UnsafeFilter:                         # pragma: no cover -- is_total swallows
                continue
            row = rng.choice(rows)
            pruned, whole = _both_ways(program, row)
            if pruned is None:
                continue
            checked += 1
            assert pruned == whole, (
                f"{program!r} on {row!r}: pruned {pruned!r}, whole {whole!r}")
        assert checked > 500, f"only {checked} generated filters were actually pruned"


class TestDocumentBuildingDoesNotScaleWithTheRow:
    """The regression this whole change exists for.

    Before pruning, `.properties.title` on a 100KB row cost the same as
    reading the 100KB field itself: ~40us per row plus ~31us per KB of
    payload, because `input_value()` marshalled the whole candidate
    either way. That is the shape being pinned -- cost proportional to
    the PROJECTION, not to the row.

    MEASURED IN CPU TIME, not wall time, and that is what makes it safe
    to gate a merge on. `time.process_time()` counts this process's own
    cycles and nothing else on the box, so a shared CI runner with four
    Python versions building in parallel changes the number by noise
    rather than by a factor -- where wall time under 2x nproc of load
    moved it enough to fail a threshold that held comfortably on an idle
    machine. The ratio between two payload sizes is the primary
    assertion for the same reason: both sides are measured the same way
    in the same process, seconds apart."""

    @staticmethod
    def _rows(count, kilobytes):
        body = "x" * 1000
        return [{"id": str(i), "similarity": 0.5, "boosts": {},
                 "properties": dict({"title": f"node {i}", "summary": "s"},
                                    **{f"field_{f}": body for f in range(kilobytes)})}
                for i in range(count)]

    def _cost(self, rerank, rows):
        """CPU seconds per row, best of three. Best rather than mean
        because a GC pause inside one run is not the measurement."""
        rerank.build_documents(rows)                     # warm
        best = min(self._once(rerank, rows) for _ in range(3))
        return best / len(rows)

    @staticmethod
    def _once(rerank, rows):
        started = time.process_time()
        rerank.build_documents(rows)
        return time.process_time() - started

    def test_a_big_payload_costs_what_a_small_one_does(self):
        needs_jq()
        rerank = _rerank(lambda q, d: [1.0] * len(d), document_from=".properties.title")
        small = self._cost(rerank, self._rows(50, 1))
        large = self._cost(rerank, self._rows(50, 100))
        # Unpruned this ratio measured ~72x (40us against 2879us per
        # row). Ten is far above the run-to-run spread of a CPU-time
        # measurement and an order of magnitude below the regression.
        assert large < small * 10, (
            f"{large * 1e6:.0f}us/row at 100KB against {small * 1e6:.0f}us/row at 1KB -- "
            f"document building is proportional to the row again")

    def test_reading_the_big_field_costs_what_reading_the_title_does(self):
        """The control the audit used: `.properties.title` (60 characters
        out) and `.properties.field_0` (1KB out) on the SAME rows. Before
        pruning they cost the same because both marshalled 100KB; after
        it they cost the same because both marshal only what they name.
        Same equality, opposite reason -- so this pins the direction the
        one above cannot."""
        needs_jq()
        rows = self._rows(50, 100)
        title = self._cost(_rerank(lambda q, d: [], document_from=".properties.title"), rows)
        field = self._cost(_rerank(lambda q, d: [], document_from=".properties.field_0"), rows)
        assert field < title * 10 and title < field * 10, (
            f"{title * 1e6:.0f}us/row for a title against {field * 1e6:.0f}us/row for a "
            f"1KB field")
        # ...and both are the projection's price, not the row's: 50
        # candidates of 100KB each cost 144ms of CPU before this, every
        # millisecond of it the event loop's own thread on the async
        # path. 20ms is 7x below that and ~30x above what it now costs.
        assert title * len(rows) < 0.02, (
            f"{title * 1e6:.0f}us/row of CPU to read a title out of a 100KB candidate")


class TestTheDocumentCountIsBounded:
    """`candidates=` bounds the NODES a step offers; under
    `per_path=True` the documents are (node, route) pairs, and how many
    routes reach a node is the GRAPH's answer, not an option's. So the
    document count had no bound at all on the Python API --
    `candidates=500` over nodes reached 200 ways is 100,000 documents,
    measured at 4.64s of document building before the provider is even
    called, all of it on the event loop's own thread on the async path.

    MAX_DOCUMENTS refuses instead of truncating, for the reason every
    other refusal in this module does: the documents that fell off the
    end are routes a node would have scored on, so the ranking would
    differ with nothing to show for it."""

    def _rows(self, count):
        return [{"id": str(i), "properties": {"title": f"n{i}"}} for i in range(count)]

    def test_a_call_over_the_limit_refuses_naming_the_number(self):
        needs_jq()
        from hopai.rerankers import MAX_DOCUMENTS

        rerank = _rerank(lambda q, d: [1.0] * len(d), candidates=10)
        with pytest.raises(ValueError) as refused:
            rerank.build_documents(self._rows(MAX_DOCUMENTS + 1))
        message = str(refused.value)
        assert str(MAX_DOCUMENTS + 1) in message and str(MAX_DOCUMENTS) in message
        assert "Lower candidates=" in message

    def test_the_limit_itself_is_allowed(self):
        """The boundary is `>`, not `>=`: a call built from exactly the
        limit is a call the caller was told they could make."""
        needs_jq()
        from hopai.rerankers import MAX_DOCUMENTS

        rerank = _rerank(lambda q, d: [1.0] * len(d), candidates=10)
        assert len(rerank.build_documents(self._rows(MAX_DOCUMENTS))) == MAX_DOCUMENTS

    def test_the_per_path_message_names_what_actually_multiplied(self):
        """Under per_path=True the advice "lower candidates=" is only
        half of it -- the other factor is the fan-in, which max_paths
        does NOT bound. A caller told to lower max_paths would change
        nothing and conclude the limit is broken."""
        needs_jq()
        from hopai.rerankers import MAX_DOCUMENTS

        rerank = _rerank(lambda q, d: [1.0] * len(d), candidates=500, per_path=True,
                         max_paths=200)
        with pytest.raises(ValueError) as refused:
            rerank.build_documents(self._rows(MAX_DOCUMENTS + 1))
        message = str(refused.value)
        assert "per_path=True builds one document per (node, route)" in message
        assert "not by max_paths=200" in message

    def test_the_message_without_per_path_points_at_candidates(self):
        """The other half: with per_path=False the count cannot exceed
        `candidates`, so the knob that is too big is that one and the
        message says so rather than explaining fan-in to someone who
        never opted into it."""
        needs_jq()
        from hopai.rerankers import MAX_DOCUMENTS

        rerank = _rerank(lambda q, d: [1.0] * len(d), candidates=9000)
        with pytest.raises(ValueError) as refused:
            rerank.build_documents(self._rows(MAX_DOCUMENTS + 1))
        assert "candidates=9000 is the input bound" in str(refused.value)

    def test_it_refuses_before_it_builds_anything(self):
        """The point of the bound is the 4.64s, so it has to be spent
        before the loop rather than after it -- a filter that raises on
        every row must not be what reports the problem."""
        needs_jq()
        from hopai.rerankers import MAX_DOCUMENTS

        rerank = _rerank(lambda q, d: [1.0] * len(d), document_from=".nope.title")
        rows = [{"id": str(i)} for i in range(MAX_DOCUMENTS + 1)]
        with pytest.raises(ValueError, match="over the 5000 limit"):
            rerank.build_documents(rows)

    def test_a_generator_of_candidates_is_counted_not_consumed_twice(self):
        """build_documents() takes a LIST in every call site hopai owns,
        but it is public and an iterator is what a caller reaches for
        when the count is the problem. Materializing once is what lets
        the bound see the number at all."""
        needs_jq()
        rerank = _rerank(lambda q, d: [1.0] * len(d))
        assert rerank.build_documents(iter(self._rows(3))) == ["n0", "n1", "n2"]


# ---------------------------------------------------------------------
# A traversal step reranks IN ORDER TO TRUNCATE, or not at all
# ---------------------------------------------------------------------

class TestRerankNeedsSomethingToTruncate:
    """A traversal returns a SUBGRAPH, not a ranking: the reranked order
    is thrown away, so the only mark a reranker can leave on a Start or a
    Hop is which nodes it drops -- and dropping is what `keep` means.

    Every refusal here is raised on the line the caller wrote, which is
    what turns the reproduction below from a live query into a
    constructor call."""

    @staticmethod
    def _rerank(**options):
        options.setdefault("document_from", ".properties.title")
        return Rerank(lambda query, documents: [0.0] * len(documents), **options)

    def test_a_hop_reranking_with_no_keep_is_refused(self):
        """THE BUG THIS CLASS EXISTS FOR. `Hop(near=, keep=None)` is an
        ordinary query -- min_similarity is the other bound -- so a
        reranker could be attached to one, and then `rerank.candidates`
        silently became the OUTPUT bound: core.py's probe widens the
        step's `keep` to it and nothing truncates afterwards, so the
        candidate set IS the survivor set. Measured against a six-node
        fan-out, a reranker returning the same score for every document
        cut the result from 6 nodes to 2 for candidates=1, with nothing
        said. Refusing is the only answer that cannot be a quietly
        different subgraph."""
        near = Near("summary", text="database query planner", min_similarity=-1.0)
        with pytest.raises(ValueError, match="rerank= needs keep="):
            Hop(via={"kind": "cites"}, near=near, rerank=self._rerank(candidates=1))

    def test_a_start_reranking_with_no_keep_is_refused_too(self):
        """The seed step has the same shape and the same consequence:
        without `keep` the walk would start from `candidates` seeds
        rather than from every seed `near` qualified."""
        near = Near("summary", text="raft", min_similarity=-1.0)
        with pytest.raises(ValueError, match="rerank= needs keep="):
            Start(near=near, rerank=self._rerank(candidates=25))

    def test_the_no_keep_refusal_names_the_fix_and_the_reason(self):
        """CLAUDE.md's rule 3: the message names the FIX, and rule 4:
        it says why the semantics differ from what was expected. Both
        halves matter here, because "add keep=" is useless advice
        without "the order is discarded" beside it -- a caller who
        thinks a traversal returns a ranking has no reason to believe
        the two are related."""
        near = Near("summary", text="raft", min_similarity=-1.0)
        with pytest.raises(ValueError) as refused:
            Start(near=near, rerank=self._rerank(candidates=25))
        message = str(refused.value)
        assert "SUBGRAPH, not a ranking" in message
        assert "Add keep=N" in message and "candidates=25" in message

    def test_the_second_enforcement_site_catches_a_spec_that_dodged_the_first(self):
        """Construction is not the only gate: vectors._resolved_spec()
        and core.py's probe rebuild a step and re-attach its rerank
        AFTER __post_init__ has run, so rerank_plan() re-validates every
        reranked step before a statement runs. Without that second site
        a rebuilt spec could reach the probe with no `keep` at all --
        which is exactly the state the bug needed."""
        from hopai.core import rerank_plan

        start = Start(near=Near("summary", text="raft"), keep=5,
                      rerank=self._rerank(candidates=25))
        start.keep = None                       # what a rebuild could leave behind
        with pytest.raises(ValueError, match="rerank= needs keep="):
            rerank_plan(start, [])

    def test_candidates_equal_to_keep_buys_a_billed_no_op(self):
        """`candidates < keep` already refuses. At EQUALITY a traversal
        step is in the same position and was not told: the survivors are
        the top-`keep` either way, because the order the reranker
        produced does not survive -- so every document is billed for a
        result that cannot differ. vector_search() is where equality
        stays meaningful, and the message says so rather than leaving
        the caller to infer that the two surfaces differ."""
        with pytest.raises(ValueError) as refused:
            Hop(near=Near("summary", text="raft"), keep=4,
                rerank=self._rerank(candidates=4))
        message = str(refused.value)
        assert "reranks exactly as many candidates as keep=4 keeps" in message
        assert "Raise candidates above keep=4" in message
        assert "vector_search()" in message

    def test_a_search_may_still_rerank_exactly_k(self):
        """The other side of the boundary, and why it is not one rule.
        vector_search() REPORTS the new order and a `rerank_score` per
        hit, so reranking exactly `k` rows changes the answer a caller
        can see. Widening the step's refusal over it would put the
        commonest search shape out of reach."""
        from hopai.vectors import rerank_query_text

        query = rerank_query_text(Near("summary", text="raft"), self._rerank(candidates=10),
                                  10, "vector_search()")
        assert query == "raft"

    def test_a_bare_callable_where_a_rerank_belongs_is_refused(self):
        """The mistake the documentation invites: "a plain callable is a
        first-class client" is true of Rerank(client), not of rerank=.
        Unchecked it constructed fine and died at execution with
        `AttributeError: 'function' object has no attribute
        'candidates'` -- an internal attribute, three layers from the
        line that wrote it."""
        with pytest.raises(TypeError, match=r"rerank= takes a Rerank\(client"):
            Start(near=Near("summary", text="raft"), keep=2,
                  rerank=lambda query, documents: [0.0] * len(documents))

    @pytest.mark.parametrize("wrong", [".properties.title", [1], {"candidates": 5}])
    def test_every_other_shape_is_refused_the_same_way(self, wrong):
        """A filter string, a list of Reranks and a JSON-ish dict are
        the same mistake reaching for the same parameter, so they get
        the same sentence rather than three different AttributeErrors
        from three different attributes."""
        with pytest.raises(TypeError, match="rerank= takes a Rerank"):
            Hop(near=Near("summary", text="raft"), keep=2, rerank=wrong)

    def test_the_type_refusal_names_the_rewrite(self):
        """Naming `Rerank(client, document_from=...)` is what turns "not
        a Rerank" into something a caller can act on -- and saying that
        the filter is the missing half is what stops them from passing
        the client again."""
        with pytest.raises(TypeError) as refused:
            Start(near=Near("summary", text="raft"), keep=2, rerank=".properties.title")
        assert "document_from='.properties.title'" in str(refused.value)

    def test_a_search_refuses_a_bare_callable_before_it_queries(self):
        """vector_search()/vector_search_many() reach the same validator
        through rerank_query_text(), so the entry points that used to
        raise AttributeError deep inside candidate fetching now refuse
        by name."""
        from hopai.vectors import rerank_query_text, rerank_query_texts

        near = Near("summary", text="raft")
        with pytest.raises(TypeError, match=r"vector_search\(\): rerank= takes a Rerank"):
            rerank_query_text(near, lambda q, d: [], 10, "vector_search()")
        with pytest.raises(TypeError, match="rerank= takes a Rerank"):
            rerank_query_texts([near], [self._rerank()], 10, "vector_search_many()")

    def test_a_rerank_does_not_hijack_an_unrelated_near_mistake(self):
        """`near="database"` is a plain type error with its own message
        one line away. With a rerank= present the text check ran first
        against something that is not a Near at all, invented
        `Near('?', ...)`, claimed "was given a raw vector" (false -- a
        string was given) and named a rewrite nobody can apply. A
        rerank= must not change which diagnosis an unrelated mistake
        gets."""
        spec = Start(near="database", keep=2, rerank=self._rerank())
        from hopai import Graph
        graph = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline")
        with pytest.raises(TypeError,
                           match=r"near= takes Near\(field, vector\) specs, got 'database'"):
            graph.build_query(spec, [])

    def test_per_path_at_a_start_is_refused(self):
        """A seed has no provenance -- which is why `.paths` at a Start
        already refuses. `per_path=True` means one call per (node,
        path), so with no paths it IS the default under another name:
        the caller set a knob and hopai discarded it, which is the
        "a constraint the options discard is not no constraint" case."""
        from hopai.core import rerank_plan

        start = Start(near=Near("summary", text="raft"), keep=2,
                      rerank=self._rerank(candidates=25, per_path=True))
        with pytest.raises(ValueError, match="per_path=True"):
            rerank_plan(start, [])

    def test_per_path_stays_legal_at_a_hop(self):
        """The refusal is about a SEED, not about the mode. A hop is
        where a node genuinely reads differently depending on the route
        that found it, and breaking that would take the step-wise case
        with it."""
        from hopai.core import rerank_plan

        hop = Hop(via={"kind": "cites"}, near=Near("summary", text="raft"), keep=2,
                  rerank=self._rerank(candidates=25, per_path=True))
        assert rerank_plan(Start(where={"type": "doc"}), [hop]) == {0: "raft"}

    def test_the_policy_and_its_parser_are_importable_from_hopai(self):
        """parse_rerank()'s refusal names `RerankPolicy` as the thing to
        pass, and its own docstring calls itself public -- while
        `from hopai import RerankPolicy` raised ImportError. An error
        naming an unimportable symbol is the "you have to know that..."
        defect, and every sibling parser (parse_near, parse_boost,
        parse_filter, parse_aggregate) was already exported."""
        import hopai

        assert {"RerankPolicy", "parse_rerank"} <= set(hopai.__all__)
        assert hopai.RerankPolicy is not None and callable(hopai.parse_rerank)
        for name in hopai.__all__:
            assert getattr(hopai, name, None) is not None, name
