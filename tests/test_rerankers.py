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
import logging
import sys
import threading
import time

import pytest

import hopai.rerankers as rerankers_module
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

    def test_an_index_out_of_range_refuses(self):
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return _answer((7, 0.9))

        client = _named("cohere.client_v2", Cohereish)()
        with pytest.raises(RerankError, match="points at document 7"):
            _rerank(client, model="m").score("q", ["a"])

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

    def test_another_unexpected_keyword_is_still_refused(self):
        with pytest.raises(TypeError, match=r"unexpected keyword argument\(s\) \['top_n'\]"):
            Rerank(lambda q, d: [1.0], document_from=".a", top_n=5)

    def test_document_from_is_keyword_only(self):
        """Positionally it would sit where a model name reads naturally,
        and `Rerank(client, "rerank-v3.5")` would compile a model name as
        a jq filter."""
        with pytest.raises(TypeError):
            Rerank(lambda q, d: [1.0], ".properties.title")

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
            "Rerank(function, candidates=50, per_path=True)")

    def test_repr_names_the_provider_and_model(self):
        class Cohereish:
            def rerank(self, *, model, query, documents):
                return _answer()

        client = _named("cohere.client_v2", Cohereish)()
        assert repr(_rerank(client, model="rerank-v3.5")) == (
            "Rerank(cohere, model='rerank-v3.5', candidates=50)")


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

    def test_an_invalid_filter_refuses_when_it_is_compiled(self):
        rerank = self._build(".properties.title +")
        with pytest.raises(ValueError, match="is not a valid jq filter"):
            rerank.build_documents([{"id": "1", "properties": {}}])

    def test_the_filter_is_compiled_once(self, monkeypatch):
        """Measured, not assumed: compiling costs ~2.4ms against ~30us
        to evaluate a row, so a compile per candidate would be 98% of
        the cost of building 50 documents and would grow with the
        candidate count instead of being paid once."""
        jq = needs_jq()
        rerank = self._build(".properties.title")
        compiled = []
        real = jq.compile
        monkeypatch.setattr(
            jq, "compile", lambda program: compiled.append(program) or real(program))
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

    def test_a_missing_binding_names_the_extra(self, monkeypatch):
        """`No module named 'jq'` tells a caller what is absent without
        telling them what to install -- the same reason mcp.py asks for
        the SDK by name."""
        # None in sys.modules is Python's own "this import fails" hook,
        # so the real import machinery still runs -- no fake __import__.
        monkeypatch.setitem(sys.modules, "jq", None)
        with pytest.raises(ImportError, match=r"pip install 'hopai\[rerankers\]'"):
            _rerank(lambda q, d: [1.0]).build_documents([{"id": "1"}])


class TestUntrustedFilters:
    """A model MAY choose which fields the reranker reads -- that is a
    retrieval decision it is well placed to make, unlike an embedding,
    which it would have to invent. What it is not is an invitation to
    run arbitrary jq, and the gate for that lives in hopai.jqsafe, not
    here: this module only decides WHEN to ask."""

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
            [{"id": "1", "properties": {"title": "Raft"}}])
        assert calls == []

    def test_an_untrusted_filter_is_validated(self, monkeypatch):
        jqsafe = pytest.importorskip("hopai.jqsafe")
        needs_jq()
        calls = []
        monkeypatch.setattr(jqsafe, "validate",
                            lambda program, **kwargs: calls.append((program, kwargs)))
        rerank = _rerank(lambda q, d: [1.0], document_from=".properties.title")
        rerank.build_documents([{"id": "1", "properties": {"title": "Raft"}}],
                               untrusted=True, fields=["properties.title"])
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
        rerank.build_documents(candidates, untrusted=True, fields=["properties.title"])
        rerank.build_documents(candidates, untrusted=True, fields=["properties.title"])
        rerank.build_documents(candidates, untrusted=True, fields=["properties.body"])
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
            rerank.build_documents([{"id": "1"}], untrusted=True)


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
