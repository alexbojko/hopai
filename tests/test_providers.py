"""
hopai.providers -- building an Embedder from a provider name and the
environment.

No provider package is installed to run these, and that is deliberate
rather than a limitation: the module exists precisely so hopai can build
a client it never imports at module scope, and a suite that pip-installed
five SDKs to check it would be testing the SDKs. Every case here fakes
the import, which also lets the credential and model paths be exercised
for providers nobody has installed.

What is NOT faked is the refusal: each one is asserted on its exact
text, because the text is the feature. An operator reading
`--embed-provider azure-openai needs $AZURE_OPENAI_ENDPOINT` fixes it in
one step; one reading `KeyError` opens the source.
"""

from __future__ import annotations

import contextlib

import pytest

from hopai import providers
from hopai.providers import (
    ProviderError, describe, describe_rerank, embedder_from_env, provider_names,
    rerank_client_from_env, rerank_provider_names,
)


class FakeClient:
    """Stands in for a provider client.

    Two things make a client acceptable to hopai.embeddings, and a fake
    has to satisfy BOTH: its module root, which picks the family (see
    embeddings._provider()), and its attribute shape, which picks the
    call. So each fake below carries the attribute the real SDK exposes
    -- `embeddings` for OpenAI, `embed` for Cohere and Voyage, `models`
    for Google, `encode`+`tokenize` for SentenceTransformer. Faking only
    the module name gets a client hopai refuses, which is the check
    doing its job."""

    #: Set per family; the attributes an Embedder looks for.
    SHAPE: tuple = ()

    def __init__(self, **options):
        self.options = options
        for attribute in self.SHAPE:
            setattr(self, attribute, lambda *a, **k: None)


def fake_module(**attributes):
    """A stand-in for an imported SDK, holding whichever constructors the
    builder under test reaches for."""
    return type("FakeModule", (), attributes)


@pytest.fixture
def absent(monkeypatch):
    """Make a package look uninstalled, whether or not it is.

    _import()'s refusal is the one path the faked-import tests
    structurally cannot reach -- they replace the function that raises
    it -- so it has to run against the real __import__, which makes it
    the one path whose outcome depends on what happens to be in
    site-packages. That bit: `sentence-transformers` is a hopai extra
    and gets installed to try a cross-encoder, and in that environment
    the "not installed" test stopped asserting a refusal and started
    DOWNLOADING A MODEL from the Hugging Face hub -- a real network call
    (and a 401) inside a suite whose whole premise is that it makes
    none. `sys.modules[name] = None` is Python's own marker for "this
    import fails", so the refusal under test is reached identically on
    both kinds of machine.

    Not a blanket autouse: hiding a package everywhere would also hide
    it from the tests that WANT the real thing, and the point is to
    pin one refusal, not to pretend the environment is empty."""
    import sys

    def hide(*modules):
        for module in modules:
            monkeypatch.setitem(sys.modules, module, None)

    return hide


@pytest.fixture
def imported(monkeypatch):
    """Replace providers._import so no SDK has to be installed. Returns a
    dict recording what was asked for, so a test can assert the module
    name and extra a builder would have named."""
    asked: dict = {}

    def install(module):
        # kind= is the flag-naming context added with the reranking
        # registry (providers._Kind). Defaulted here rather than required
        # so the embedding builders, which do not pass it, still reach
        # this fake unchanged.
        def _import(name, extra, provider, kind=None):
            asked.update(module=name, extra=extra, provider=provider, kind=kind)
            return module
        monkeypatch.setattr(providers, "_import", _import)
        return asked

    return install


class FakeOpenAI(FakeClient):
    SHAPE = ("embeddings",)


# openai.AzureOpenAI lives in openai.lib.azure, so its module ROOT is
# `openai` -- which is why an Azure client is classified as the openai
# family and needs no special case anywhere in embeddings.py.
FakeOpenAI.__module__ = "openai.lib.azure"


class FakeCohere(FakeClient):
    SHAPE = ("embed",)


FakeCohere.__module__ = "cohere.client_v2"


class FakeVoyage(FakeClient):
    SHAPE = ("embed",)


FakeVoyage.__module__ = "voyageai.client"


class FakeGoogle(FakeClient):
    SHAPE = ("models",)


FakeGoogle.__module__ = "google.genai.client"


class FakeSentenceTransformer(FakeClient):
    SHAPE = ("encode", "tokenize")


FakeSentenceTransformer.__module__ = "sentence_transformers.SentenceTransformer"


class FakeCrossEncoder(FakeClient):
    """The RERANKING half of sentence-transformers, and deliberately not
    a subclass or an alias of the one above.

    A CrossEncoder's shape is `predict`+`tokenizer`; a
    SentenceTransformer's is `encode`+`tokenize`. rerankers._bind()
    reads the first pair and embeddings._bind() the second, so a fake
    that carried the wrong one would be accepted by the wrong half of
    hopai -- which is exactly the mix-up the two registries exist to
    prevent."""

    SHAPE = ("predict", "tokenizer")


FakeCrossEncoder.__module__ = "sentence_transformers.cross_encoder.CrossEncoder"


def openai_module(record: dict):
    def make(**options):
        record.update(options)
        return FakeOpenAI(**options)
    return fake_module(OpenAI=make, AzureOpenAI=make)


class TestTheNameItself:
    def test_an_unknown_provider_lists_the_accepted_ones(self):
        with pytest.raises(ProviderError, match="unknown embedding provider 'azure'"):
            embedder_from_env("azure", env={})

    def test_the_accepted_list_is_in_the_message(self):
        """A name that is nearly right is the common mistake -- `azure`
        for `azure-openai`, `voyageai` for `voyage` -- so the refusal
        carries the answer rather than only the verdict."""
        with pytest.raises(ProviderError) as caught:
            embedder_from_env("voyageai", env={})
        for name in provider_names():
            assert name in str(caught.value)

    def test_describe_names_every_variable_the_provider_reads(self):
        line = describe("azure-openai")
        assert "$AZURE_OPENAI_API_KEY" in line
        assert "$AZURE_OPENAI_ENDPOINT" in line
        assert "$AZURE_OPENAI_EMBEDDING_DEPLOYMENT" in line


class TestMissingPackage:
    @pytest.mark.parametrize("provider, extra", [
        ("openai", "openai"), ("azure-openai", "openai"), ("cohere", "cohere"),
        ("voyage", "voyageai"), ("google", "google"),
        ("sentence-transformers", "sentence-transformers"),
    ])
    def test_it_names_the_pip_extra_that_fixes_it(self, absent, provider, extra):
        """The real message an operator gets, from the real _import().
        azure-openai maps to the `openai` extra, which is the pairing
        worth pinning: the package is not named for the provider.

        `absent` rather than "none of these are installed": that was
        true of a bare dev environment and false of one where somebody
        pip-installed an extra, and there this test reached a live
        Hugging Face download instead of a refusal. See the fixture."""
        absent(providers.PROVIDERS[provider].package)
        with pytest.raises(ProviderError, match=rf'pip install "hopai\[{extra}\]"') as caught:
            embedder_from_env(provider, env={}, model="m")
        # And which provider asked for it. This path is the one
        # TestEveryRefusalNamesItsProvider structurally cannot cover --
        # it fakes _import, so the real refusal below it is never built,
        # and `_import(..., None)` survived a green suite because of it.
        assert f"--embed-provider {provider} " in str(caught.value)


class TestCredentials:
    def test_a_missing_key_is_named(self, imported):
        imported(openai_module({}))
        with pytest.raises(ProviderError, match=r"needs \$OPENAI_API_KEY"):
            embedder_from_env("openai", env={}, model="text-embedding-3-small")

    def test_an_empty_key_counts_as_missing(self, imported):
        """An exported-but-empty variable is the shape a broken .env or a
        missing secret mount actually takes, and `if not value` has to
        catch it -- `is None` would let "" through to the provider and
        fail as a 401 somewhere else."""
        imported(openai_module({}))
        with pytest.raises(ProviderError, match=r"needs \$OPENAI_API_KEY"):
            embedder_from_env("openai", env={"OPENAI_API_KEY": "   "}, model="m")

    def test_azure_names_the_endpoint_and_says_what_it_looks_like(self, imported):
        imported(openai_module({}))
        with pytest.raises(ProviderError, match="AZURE_OPENAI_ENDPOINT") as caught:
            embedder_from_env("azure-openai", env={"AZURE_OPENAI_API_KEY": "k"})
        assert "my-resource.openai.azure.com" in str(caught.value)

    @pytest.mark.parametrize("name", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
    def test_google_accepts_either_vendor_variable(self, imported, name):
        """The SDK reads $GEMINI_API_KEY; most deployments export
        $GOOGLE_API_KEY. Refusing the one the vendor's own client honours
        would be a trap, so both work."""
        record: dict = {}
        imported(fake_module(Client=lambda **o: record.update(o) or FakeGoogle(**o)))
        embedder_from_env("google", env={name: "k"}, model="text-embedding-004")
        assert record["api_key"] == "k"

    def test_google_refusal_names_both(self, imported):
        imported(fake_module(Client=FakeGoogle))
        with pytest.raises(ProviderError) as caught:
            embedder_from_env("google", env={}, model="m")
        assert "$GOOGLE_API_KEY" in str(caught.value)
        assert "$GEMINI_API_KEY" in str(caught.value)


class TestModelIsNeverGuessed:
    @pytest.mark.parametrize("provider, variable, env", [
        ("openai", "OPENAI_EMBEDDING_MODEL", {"OPENAI_API_KEY": "k"}),
        ("azure-openai", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
         {"AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "https://x.example"}),
    ])
    def test_a_missing_model_refuses_and_names_both_ways_to_set_it(
            self, imported, provider, variable, env):
        """No default, ever: the model that answers a query has to be the
        model that wrote the stored vectors, and hopai cannot know which
        that was. Picking one would return confidently wrong neighbours
        with nothing to see."""
        imported(openai_module({}))
        with pytest.raises(ProviderError) as caught:
            embedder_from_env(provider, env=env)
        message = str(caught.value)
        assert "--embed-model" in message
        assert f"${variable}" in message

    def test_the_flag_wins_over_the_variable(self, imported):
        record: dict = {}
        imported(openai_module(record))
        embedder = embedder_from_env(
            "openai", model="from-flag",
            env={"OPENAI_API_KEY": "k", "OPENAI_EMBEDDING_MODEL": "from-env"})
        assert embedder.model == "from-flag"

    def test_the_variable_is_used_when_the_flag_is_absent(self, imported):
        imported(openai_module({}))
        embedder = embedder_from_env(
            "openai", env={"OPENAI_API_KEY": "k", "OPENAI_EMBEDDING_MODEL": "from-env"})
        assert embedder.model == "from-env"


class TestAzureSpecifics:
    def _env(self, **extra):
        return {"AZURE_OPENAI_API_KEY": "secret",
                "AZURE_OPENAI_ENDPOINT": "https://my-resource.openai.azure.com",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "my-embed-deployment", **extra}

    def test_the_deployment_name_is_what_reaches_the_model_argument(self, imported):
        """Azure's `model` IS the deployment, not the model it serves --
        the two are often different strings, and passing the model name
        where Azure wants the deployment is the most common way this
        configuration fails."""
        imported(openai_module({}))
        embedder = embedder_from_env("azure-openai", env=self._env())
        assert embedder.model == "my-embed-deployment"

    def test_endpoint_and_key_reach_the_client(self, imported):
        record: dict = {}
        imported(openai_module(record))
        embedder_from_env("azure-openai", env=self._env())
        assert record["api_key"] == "secret"
        assert record["azure_endpoint"] == "https://my-resource.openai.azure.com"

    def test_the_api_version_has_a_default_because_it_is_not_a_model_choice(self, imported):
        record: dict = {}
        imported(openai_module(record))
        embedder_from_env("azure-openai", env=self._env())
        assert record["api_version"] == providers.DEFAULT_AZURE_API_VERSION

    def test_the_api_version_can_be_overridden(self, imported):
        record: dict = {}
        imported(openai_module(record))
        embedder_from_env("azure-openai",
                          env=self._env(AZURE_OPENAI_API_VERSION="2099-01-01"))
        assert record["api_version"] == "2099-01-01"

    def test_an_azure_client_is_the_openai_family_to_embeddings_py(self, imported):
        """The reason Azure needs no special case downstream: its client
        lives in openai.lib.azure, so the module ROOT embeddings.py reads
        is `openai`, and it gets openai's batch cap and call shape."""
        imported(openai_module({}))
        embedder = embedder_from_env("azure-openai", env=self._env())
        assert embedder.provider == "openai"


class TestTheImportEachBuilderAsksFor:
    """Which module a builder imports, and under which pip extra.

    The fixture above has always recorded this -- its docstring says so
    -- and nothing read it. That mattered more than it looks: every
    test here fakes _import, so the module NAME never reaches a real
    import, and `_import("XXopenaiXX", ...)` sailed through a complete
    mutation run. In production that name is the import, so a wrong one
    means the provider can never be built: `--embed-provider openai`
    would refuse with "needs the 'openai' package -- pip install
    hopai[openai]" on a machine where openai is installed, which sends
    the operator to reinstall a package they already have.

    The pairs are worth reading as documentation, since three of the
    six differ from the provider name: voyage imports voyageai, google
    imports google.genai, and azure-openai imports plain openai."""

    @pytest.mark.parametrize("provider, module, extra", [
        ("openai", "openai", "openai"),
        ("azure-openai", "openai", "openai"),
        ("cohere", "cohere", "cohere"),
        ("voyage", "voyageai", "voyageai"),
        ("google", "google.genai", "google"),
        ("sentence-transformers", "sentence_transformers", "sentence-transformers"),
    ])
    def test_it_imports_the_module_that_actually_holds_the_client(
            self, imported, provider, module, extra):
        asked = imported(fake_module(
            OpenAI=FakeOpenAI, AzureOpenAI=FakeOpenAI, ClientV2=FakeCohere,
            Client=FakeVoyage, SentenceTransformer=FakeSentenceTransformer))
        # google's Client fake is the wrong shape for this one call; the
        # import is recorded before any of that matters.
        with contextlib.suppress(ProviderError, TypeError):
            embedder_from_env(provider, env=TestEveryRefusalNamesItsProvider.FILLED,
                              model="m")
        assert (asked["module"], asked["extra"]) == (module, extra)

    def test_a_dotted_module_resolves_to_the_submodule_not_its_parent(self):
        """`fromlist` is why _import() returns google.genai rather than
        google, and google is the one dotted provider.

        Without it __import__("google.genai") hands back the `google`
        package, and the very next line -- genai.Client(...) -- is an
        AttributeError rather than a client. Every other test fakes
        _import, so the real one is only ever reached with a package
        that is absent; a dotted stdlib module exercises it without
        installing anything, which is the point of this file."""
        import os.path

        assert providers._import("os.path", "x", "y") is os.path


class TestPassThrough:
    def test_the_api_key_reaches_the_client_under_the_name_the_sdk_expects(
            self, imported):
        """`api_key` is a keyword the real SDK matches by name. The fake
        takes **options, so a renamed key is accepted here and would be
        a TypeError against the real client -- which is why the mutants
        that renamed it survived a full run of this file."""
        record: dict = {}
        imported(openai_module(record))
        embedder_from_env("openai", model="m", env={"OPENAI_API_KEY": "k"})
        assert record["api_key"] == "k"

    def test_openai_compatible_endpoints_are_reachable(self, imported):
        """A gateway or a self-hosted vLLM speaks the OpenAI API at a
        different address. The provider name says which SDK, not which
        vendor, so the base URL is read rather than ignored."""
        record: dict = {}
        imported(openai_module(record))
        embedder_from_env("openai", model="m",
                          env={"OPENAI_API_KEY": "k",
                               "OPENAI_BASE_URL": "http://vllm:8000/v1",
                               "OPENAI_ORG_ID": "org-1"})
        assert record["base_url"] == "http://vllm:8000/v1"
        assert record["organization"] == "org-1"

    def test_optional_variables_are_absent_rather_than_none(self, imported):
        """Passing base_url=None to the SDK is not the same as not passing
        it: the client's own default is the one that should apply."""
        record: dict = {}
        imported(openai_module(record))
        embedder_from_env("openai", model="m", env={"OPENAI_API_KEY": "k"})
        assert "base_url" not in record and "organization" not in record

    def test_embedder_options_reach_the_embedder(self, imported):
        imported(openai_module({}))
        embedder = embedder_from_env("openai", model="m", env={"OPENAI_API_KEY": "k"},
                                     retries=0, backoff=2.5, dimensions=256)
        assert (embedder.retries, embedder.backoff, embedder.dimensions) == (0, 2.5, 256)

    def test_sentence_transformers_passes_no_model_to_the_embedder(self, imported):
        """The name is baked into the loaded object, and Embedder REFUSES
        model= for this family rather than accepting a second copy of it
        that nothing reads."""
        imported(fake_module(
            SentenceTransformer=lambda name: FakeSentenceTransformer(name=name)))
        embedder = embedder_from_env("sentence-transformers",
                                     env={"SENTENCE_TRANSFORMERS_MODEL": "all-MiniLM-L6-v2"})
        assert embedder.model is None
        assert embedder.client.options["name"] == "all-MiniLM-L6-v2"


class TestTheEnvironmentIsNotLeaked:
    def test_an_explicit_env_is_used_instead_of_os_environ(self, imported, monkeypatch):
        """`env=` is what makes every test above hermetic, and what lets a
        caller build two providers with different credentials in one
        process. A real $OPENAI_API_KEY must not satisfy a call that
        passed its own mapping."""
        monkeypatch.setenv("OPENAI_API_KEY", "from-the-real-environment")
        imported(openai_module({}))
        with pytest.raises(ProviderError, match=r"needs \$OPENAI_API_KEY"):
            embedder_from_env("openai", env={}, model="m")


class TestTheOtherProviders:
    """cohere and voyage have no Azure-shaped subtlety, but they are real
    accepted names and had no coverage beyond "the package is missing" --
    which would have passed just as well if their builders reached for
    the wrong constructor or the wrong variable."""

    @pytest.mark.parametrize("provider, client, key_var, model_var, model", [
        ("cohere", FakeCohere, "COHERE_API_KEY", "COHERE_EMBEDDING_MODEL", "embed-v4.0"),
        ("voyage", FakeVoyage, "VOYAGE_API_KEY", "VOYAGE_EMBEDDING_MODEL", "voyage-3"),
    ])
    def test_key_and_model_reach_the_client(self, imported, provider, client,
                                            key_var, model_var, model):
        record: dict = {}
        constructor = "ClientV2" if provider == "cohere" else "Client"
        imported(fake_module(**{
            constructor: lambda **o: record.update(o) or client(**o)}))
        embedder = embedder_from_env(provider, env={key_var: "k", model_var: model})
        assert record["api_key"] == "k"
        assert embedder.model == model

    @pytest.mark.parametrize("provider, client, constructor", [
        ("cohere", FakeCohere, "ClientV2"), ("voyage", FakeVoyage, "Client"),
    ])
    def test_a_missing_key_is_named(self, imported, provider, client, constructor):
        imported(fake_module(**{constructor: client}))
        with pytest.raises(ProviderError, match=r"needs \$[A-Z_]+_API_KEY"):
            embedder_from_env(provider, env={}, model="m")


class TestEveryRefusalNamesItsProvider:
    """Which provider failed is the first thing an operator needs, and
    it was the one part of these sentences nothing checked.

    The tests above assert the VARIABLE (`needs $OPENAI_API_KEY`) and
    the extra (`pip install "hopai[cohere]"`), so the `provider`
    argument threaded through _need() and _model_for() could be
    replaced with None and every one of them still passed -- leaving
    `--embed-provider None needs $AZURE_OPENAI_API_KEY` on the console
    of someone who set HOPAI_EMBED_PROVIDER and cannot see the flag.
    Mutation testing surfaced it as a family (_need, _model_for, and
    every call site that passes the name), so this closes it as one:
    over PROVIDERS itself, which means a provider added tomorrow is
    covered without anyone remembering to add a row."""

    #: The client fake and constructor each builder reaches for, so the
    #: import can be faked far enough to reach the refusal underneath.
    BUILDERS = {
        "openai": (FakeOpenAI, "OpenAI"),
        "azure-openai": (FakeOpenAI, "AzureOpenAI"),
        "cohere": (FakeCohere, "ClientV2"),
        "voyage": (FakeVoyage, "Client"),
        "google": (FakeGoogle, "Client"),
        "sentence-transformers": (FakeSentenceTransformer, "SentenceTransformer"),
    }

    #: Everything any provider reads, so one variable can be emptied
    #: while the rest stay filled.
    FILLED = {"OPENAI_API_KEY": "k", "AZURE_OPENAI_API_KEY": "k",
              "AZURE_OPENAI_ENDPOINT": "https://r.openai.azure.com",
              "COHERE_API_KEY": "k", "VOYAGE_API_KEY": "k", "GOOGLE_API_KEY": "k"}

    @pytest.mark.parametrize("provider", sorted(provider_names()))
    def test_a_missing_credential_or_model_says_which_provider(self, imported, provider):
        client, constructor = self.BUILDERS[provider]
        imported(fake_module(**{constructor: client}))
        # An empty environment and no --embed-model, so whichever refusal
        # this provider reaches first -- a credential or the model -- is
        # the one under test. sentence-transformers needs no credential
        # and lands on the model; the rest land on a key.
        with pytest.raises(ProviderError) as caught:
            embedder_from_env(provider, env={})
        assert f"--embed-provider {provider} " in str(caught.value), (
            f"the refusal does not name {provider!r}: {caught.value}")

    @pytest.mark.parametrize("provider", sorted(provider_names()))
    def test_an_explicit_model_beats_the_environment(self, imported, provider):
        """`--embed-model` wins over the provider's own variable, and a
        builder that drops the argument is not a naming bug -- it is the
        wrong model answering queries.

        Dropping `model` from _model_for() left the environment as the
        only source: an operator passing --embed-model against a shell
        that already exports OPENAI_EMBEDDING_MODEL would silently get
        the exported one, and the model that answers a query would stop
        being the model that wrote the stored vectors. That is the
        failure the whole vector surface exists to prevent, and all
        2819 tests passed with it. The existing coverage supplied the
        model through the ENVIRONMENT, so this precedence -- the only
        reason the argument exists -- was never exercised.
        """
        client, constructor = self.BUILDERS[provider]
        seen: dict = {}

        def record(*args, **kwargs):
            seen["args"], seen["kwargs"] = args, kwargs
            return client(**kwargs)

        imported(fake_module(**{constructor: record}))
        spec = providers.PROVIDERS[provider]
        env = dict(self.FILLED)
        if spec.model_var:                       # a decoy the flag must beat
            env[spec.model_var] = "from-the-environment"
        embedder = embedder_from_env(provider, env=env, model="from-the-flag")

        # sentence-transformers bakes the name into the loaded object and
        # returns None as the model (Embedder refuses a second, ignorable
        # copy), so its choice is visible in the constructor instead.
        chosen = embedder.model or (seen["args"][0] if seen.get("args") else None)
        assert chosen == "from-the-flag", (
            f"{provider}: --embed-model was ignored in favour of {chosen!r}")

    @pytest.mark.parametrize("provider, missing", [
        (name, var) for name in sorted(provider_names())
        for var in providers.PROVIDERS[name].credentials
    ])
    def test_every_credential_it_reads_refuses_by_name(self, imported, provider, missing):
        """One row per (provider, variable), not per provider.

        azure-openai reads two, and _need() is called for them in order
        -- so an empty environment always stops at the first, and the
        SECOND call site's provider argument was never exercised.
        `_need(env, "AZURE_OPENAI_ENDPOINT", None)` survived a green
        suite for exactly that reason. Filling every variable except
        the one under test reaches each call site in turn."""
        client, constructor = self.BUILDERS[provider]
        imported(fake_module(**{constructor: client}))
        env = {key: value for key, value in self.FILLED.items() if key != missing}
        with pytest.raises(ProviderError) as caught:
            embedder_from_env(provider, env=env, model="m")
        assert f"--embed-provider {provider} needs ${missing}" in str(caught.value), (
            f"the refusal for {missing} does not name {provider!r}: {caught.value}")

    @pytest.mark.parametrize("provider", sorted(provider_names()))
    def test_a_missing_model_says_which_provider(self, imported, provider):
        """The model refusal specifically, reached by supplying whatever
        credentials that provider wants -- otherwise every row above
        would stop at the key and this sentence would stay unchecked."""
        client, constructor = self.BUILDERS[provider]
        imported(fake_module(**{constructor: client}))
        filled = {"OPENAI_API_KEY": "k", "AZURE_OPENAI_API_KEY": "k",
                  "AZURE_OPENAI_ENDPOINT": "https://r.openai.azure.com",
                  "COHERE_API_KEY": "k", "VOYAGE_API_KEY": "k", "GOOGLE_API_KEY": "k"}
        with pytest.raises(ProviderError) as caught:
            embedder_from_env(provider, env=filled)
        assert "needs a model" in str(caught.value)
        assert f"--embed-provider {provider} " in str(caught.value), (
            f"the model refusal does not name {provider!r}: {caught.value}")


# ---------------------------------------------------------------------
# Reranking: a second registry, a second pair of flags
# ---------------------------------------------------------------------

class TestTheRerankingRegistryIsSmaller:
    """Which names `--rerank-provider` accepts, and what happens to the
    ones it does not.

    Without this, adding a reranking builder for openai -- or copying
    PROVIDERS wholesale into RERANK_PROVIDERS -- would advertise a name
    that has no reranking endpoint to call, and the failure would be a
    404 from a vendor at the first query rather than a refusal at
    start-up."""

    def test_exactly_three_names_are_accepted(self):
        assert rerank_provider_names() == ["cohere", "sentence-transformers", "voyage"]

    @pytest.mark.parametrize("absent", ["openai", "azure-openai", "google"])
    def test_the_embedding_only_providers_are_absent(self, absent):
        """They ARE providers -- for the other role. Listing one here
        would build a client and then POST to an endpoint that does not
        exist."""
        assert absent in provider_names()
        assert absent not in rerank_provider_names()
        assert absent not in providers.RERANK_PROVIDERS

    def test_a_name_that_is_an_embed_provider_is_told_so(self):
        """The likeliest way to land here is a name that is spelled
        perfectly -- for --embed-provider. "unknown provider 'openai'"
        would send the operator to check a spelling that is correct."""
        with pytest.raises(ProviderError) as caught:
            rerank_client_from_env("openai", env={})
        message = str(caught.value)
        assert "'openai' has no reranking endpoint in hopai" in message
        assert "--rerank-provider accepts: cohere, sentence-transformers, voyage." in message
        assert "('openai' IS an --embed-provider; embedding and reranking are " \
               "different models and only some vendors offer both.)" in message

    def test_a_name_that_is_nothing_gets_no_cross_role_hint(self):
        """The other half of the sentence above: telling someone who
        typed `nope` that it IS an --embed-provider would be false, so
        the parenthetical is conditional and not decoration."""
        with pytest.raises(ProviderError) as caught:
            rerank_client_from_env("nope", env={})
        message = str(caught.value)
        assert "'nope' has no reranking endpoint in hopai" in message
        assert "--rerank-provider accepts: cohere, sentence-transformers, voyage." in message
        assert "IS an --embed-provider" not in message

    @pytest.mark.parametrize("provider, reads", [
        ("cohere", ["$COHERE_API_KEY", "$COHERE_RERANK_MODEL"]),
        ("voyage", ["$VOYAGE_API_KEY", "$VOYAGE_RERANK_MODEL"]),
        ("sentence-transformers", ["$SENTENCE_TRANSFORMERS_RERANK_MODEL"]),
    ])
    def test_describe_rerank_names_every_variable_that_role_reads(self, provider, reads):
        """`--rerank-provider-help` prints these lines, and an operator
        exports what they say. A line naming $COHERE_EMBEDDING_MODEL
        here would have them configure the wrong variable and read
        "needs a model" about one they just set."""
        line = describe_rerank(provider)
        assert line == f"{provider}: {', '.join(reads)}"
        # And the embedding registry's line for the same vendor is a
        # DIFFERENT sentence -- which is the whole reason for two dicts.
        if provider in providers.PROVIDERS:
            assert describe(provider) != line

    def test_sentence_transformers_needs_no_credential_in_either_role(self):
        """It runs locally, so its describe() line is a model variable
        and nothing else. A credential invented for symmetry would be a
        variable nobody can set."""
        assert providers.RERANK_PROVIDERS["sentence-transformers"].credentials == ()
        assert "$" + "SENTENCE_TRANSFORMERS_RERANK_MODEL" in \
            describe_rerank("sentence-transformers")


class TestARerankRefusalNamesTheRerankFlags:
    """Every reranking refusal names --rerank-provider/--rerank-model,
    and never the embedding pair.

    This is the entire reason `kind` is threaded through _need(),
    _import() and _model_for() rather than formatted in at the end. The
    two roles share those helpers, so a builder that forgot to pass
    RERANKING would print `--embed-provider cohere needs
    $COHERE_API_KEY` to an operator who is looking at --rerank-provider
    -- a correct sentence about a command they are not running, which
    sends them to export a variable for the flag they were not using.
    The default is EMBEDDING, so a forgotten argument is silent."""

    #: Every credential any rerank provider reads, so one can be emptied
    #: while the rest stay filled.
    FILLED = {"COHERE_API_KEY": "k", "VOYAGE_API_KEY": "k"}

    #: The client fake and constructor each rerank builder reaches for.
    BUILDERS = {
        "cohere": (FakeCohere, "ClientV2"),
        "voyage": (FakeVoyage, "Client"),
        "sentence-transformers": (FakeCrossEncoder, "CrossEncoder"),
    }

    @pytest.mark.parametrize("provider, extra, module", [
        ("cohere", "cohere", "cohere"), ("voyage", "voyageai", "voyageai"),
        ("sentence-transformers", "sentence-transformers", "sentence_transformers"),
    ])
    def test_a_missing_package_names_the_rerank_flag_and_the_pip_extra(
            self, absent, provider, extra, module):
        """The real message, from the real _import() -- the one path the
        faked-import tests below structurally cannot reach, since they
        replace the function that raises it.

        The MODULE NAME is asserted exactly, not just that the sentence
        has a shape: the name each builder passes to _import() is an
        import-time string literal, so a wrong one still raises
        ImportError and still produces a perfectly-formed refusal -- one
        naming a module the operator cannot install. `voyageai` is the
        case that makes this concrete: the provider is spelled `voyage`
        and the package is not."""
        absent(providers.RERANK_PROVIDERS[provider].package)
        with pytest.raises(ProviderError) as caught:
            rerank_client_from_env(provider, model="m", env={})
        message = str(caught.value)
        assert f"--rerank-provider {provider} needs the {module!r} package" in message
        assert f'pip install "hopai[{extra}]"' in message
        assert "--embed-provider" not in message

    @pytest.mark.parametrize("provider, missing", [
        ("cohere", "COHERE_API_KEY"), ("voyage", "VOYAGE_API_KEY"),
    ])
    def test_a_missing_credential_names_the_rerank_flag(self, imported, provider,
                                                        missing):
        client, constructor = self.BUILDERS[provider]
        imported(fake_module(**{constructor: client}))
        env = {key: value for key, value in self.FILLED.items() if key != missing}
        with pytest.raises(ProviderError) as caught:
            rerank_client_from_env(provider, model="m", env=env)
        message = str(caught.value)
        assert f"--rerank-provider {provider} needs ${missing} and it is unset or " \
               "empty. Export it before starting." in message
        assert "--embed-provider" not in message
        # And it ENDS there. `extra_help` is _need()'s one optional
        # argument, appended straight onto the sentence with no
        # separator; no reranking credential passes it, so anything after
        # the full stop is text that arrived from somewhere other than
        # this call -- which a containment assertion cannot see.
        assert message.endswith("Export it before starting.")

    @pytest.mark.parametrize("provider, variable, example", [
        ("cohere", "COHERE_RERANK_MODEL", "rerank-v3.5"),
        ("voyage", "VOYAGE_RERANK_MODEL", "rerank-2"),
        ("sentence-transformers", "SENTENCE_TRANSFORMERS_RERANK_MODEL",
         "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ])
    def test_a_missing_model_names_the_rerank_flag_and_its_own_variable(
            self, imported, provider, variable, example):
        client, constructor = self.BUILDERS[provider]
        imported(fake_module(**{constructor: client}))
        with pytest.raises(ProviderError) as caught:
            rerank_client_from_env(provider, env=dict(self.FILLED))
        message = str(caught.value)
        assert f"--rerank-provider {provider} needs a model and none was given." in message
        assert f"Pass --rerank-model {example} or export ${variable}={example}." in message
        assert "--embed-model" not in message and "--embed-provider" not in message

    def test_the_model_refusal_says_why_it_will_not_guess_for_a_reranker(self):
        """Not the embedding sentence. A reranker has no stored vectors
        to match, so "the model that wrote the stored vectors" would be
        a reason that does not apply -- and an operator who reads it
        goes looking for a vector column."""
        with pytest.raises(ProviderError) as caught:
            rerank_client_from_env("cohere", env=dict(self.FILLED))
        # The package is missing before the model is reached, so the
        # reason is asserted on the constant that supplies it.
        assert providers.RERANKING.why == (
            "a reranker's score is only meaningful against the model that produced it, "
            "and a silently chosen one is a silently different ranking")
        assert "stored vectors" in providers.EMBEDDING.why
        assert "stored vectors" not in providers.RERANKING.why
        assert isinstance(caught.value, ProviderError)

    @pytest.mark.parametrize("provider", ["cohere", "voyage", "sentence-transformers"])
    def test_the_kind_reaches_the_import_site_too(self, imported, provider):
        """_import() is the one helper whose `kind` is invisible in the
        tests above -- they fake it. Recording it is what pins that a
        builder passes RERANKING there as well as to _need()/_model_for(),
        since the argument defaults to EMBEDDING and a forgotten one
        raises nothing."""
        client, constructor = self.BUILDERS[provider]
        asked = imported(fake_module(**{constructor: client}))
        # The CrossEncoder fake takes its name as a keyword; the import
        # is recorded long before the constructor is reached.
        with contextlib.suppress(ProviderError, TypeError):
            rerank_client_from_env(provider, model="m", env=dict(self.FILLED))
        assert asked["kind"] is providers.RERANKING


class TestTheTwoRolesReadDifferentModelVariables:
    """$COHERE_RERANK_MODEL, never $COHERE_EMBEDDING_MODEL.

    Sharing one entry between the registries would mean the embedding
    model's name being POSTed to /rerank -- a 4xx at best, and at worst
    a vendor that accepts it and returns a nonsense ranking nothing
    reports. Setting ONLY the embedding variable and asserting the
    rerank build still refuses is what catches a builder that reached
    for PROVIDERS[...] instead of RERANK_PROVIDERS[...]; a test that set
    both would pass either way."""

    @pytest.mark.parametrize("provider, client, constructor, embedding_var, rerank_var", [
        ("cohere", FakeCohere, "ClientV2", "COHERE_EMBEDDING_MODEL",
         "COHERE_RERANK_MODEL"),
        ("voyage", FakeVoyage, "Client", "VOYAGE_EMBEDDING_MODEL",
         "VOYAGE_RERANK_MODEL"),
        ("sentence-transformers", FakeCrossEncoder, "CrossEncoder",
         "SENTENCE_TRANSFORMERS_MODEL", "SENTENCE_TRANSFORMERS_RERANK_MODEL"),
    ])
    def test_the_embedding_variable_alone_does_not_satisfy_a_reranker(
            self, imported, provider, client, constructor, embedding_var, rerank_var):
        imported(fake_module(**{constructor: client}))
        env = {"COHERE_API_KEY": "k", "VOYAGE_API_KEY": "k",
               embedding_var: "an-embedding-model"}
        with pytest.raises(ProviderError) as caught:
            rerank_client_from_env(provider, env=env)
        message = str(caught.value)
        assert "needs a model and none was given" in message
        assert f"${rerank_var}" in message
        assert embedding_var not in message

    @pytest.mark.parametrize("provider, client, constructor, rerank_var", [
        ("cohere", FakeCohere, "ClientV2", "COHERE_RERANK_MODEL"),
        ("voyage", FakeVoyage, "Client", "VOYAGE_RERANK_MODEL"),
    ])
    def test_the_rerank_variable_is_the_one_that_is_read(
            self, imported, provider, client, constructor, rerank_var):
        imported(fake_module(**{constructor: client}))
        _, model = rerank_client_from_env(
            provider, env={"COHERE_API_KEY": "k", "VOYAGE_API_KEY": "k",
                           rerank_var: "a-rerank-model",
                           "COHERE_EMBEDDING_MODEL": "an-embedding-model",
                           "VOYAGE_EMBEDDING_MODEL": "an-embedding-model"})
        assert model == "a-rerank-model"

    @pytest.mark.parametrize("provider, client, constructor", [
        ("cohere", FakeCohere, "ClientV2"), ("voyage", FakeVoyage, "Client"),
    ])
    def test_the_flag_still_beats_the_variable(self, imported, provider, client,
                                               constructor):
        imported(fake_module(**{constructor: client}))
        _, model = rerank_client_from_env(
            provider, model="from-the-flag",
            env={"COHERE_API_KEY": "k", "VOYAGE_API_KEY": "k",
                 "COHERE_RERANK_MODEL": "from-the-environment",
                 "VOYAGE_RERANK_MODEL": "from-the-environment"})
        assert model == "from-the-flag"

    @pytest.mark.parametrize("provider, client, constructor, key_var", [
        ("cohere", FakeCohere, "ClientV2", "COHERE_API_KEY"),
        ("voyage", FakeVoyage, "Client", "VOYAGE_API_KEY"),
    ])
    def test_the_api_key_reaches_the_client(self, imported, provider, client,
                                            constructor, key_var):
        record: dict = {}
        imported(fake_module(**{
            constructor: lambda **o: record.update(o) or client(**o)}))
        rerank_client_from_env(provider, model="m", env={key_var: "secret"})
        assert record["api_key"] == "secret"


class TestTheCrossEncoderIsNotTheBiEncoder:
    """`--rerank-provider sentence-transformers` loads CrossEncoder, and
    the embedding twin loads SentenceTransformer.

    One import, two constructors, and nothing downstream can tell them
    apart by the module they came from: a SentenceTransformer handed to
    Rerank has `encode`/`tokenize`, so `_bind()` falls past the
    cross-encoder branch and lands on "not a reranker client hopai
    recognizes" -- if it is lucky. What it must never do is score, since
    a bi-encoder cannot read a (query, document) pair at all."""

    def _module(self, record: dict):
        def cross_encoder(name):
            record["CrossEncoder"] = name
            return FakeCrossEncoder(name=name)

        def sentence_transformer(name):
            record["SentenceTransformer"] = name
            return FakeSentenceTransformer(name=name)

        return fake_module(CrossEncoder=cross_encoder,
                           SentenceTransformer=sentence_transformer)

    def test_it_constructs_the_cross_encoder_and_not_the_bi_encoder(self, imported):
        record: dict = {}
        imported(self._module(record))
        client, model = rerank_client_from_env(
            "sentence-transformers",
            env={"SENTENCE_TRANSFORMERS_RERANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"})
        assert record == {"CrossEncoder": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
        assert "SentenceTransformer" not in record
        assert isinstance(client, FakeCrossEncoder)
        assert model is None

    def test_the_flag_reaches_the_constructor(self, imported):
        record: dict = {}
        imported(self._module(record))
        rerank_client_from_env("sentence-transformers", model="from-the-flag",
                               env={"SENTENCE_TRANSFORMERS_RERANK_MODEL": "from-the-env"})
        assert record["CrossEncoder"] == "from-the-flag"

    def test_the_model_comes_back_as_none_because_rerank_refuses_one(self, imported):
        """The name is baked into the loaded object, and Rerank raises
        "a CrossEncoder already IS the model" for a model= it cannot
        use. Handing back the resolved name here would make the obvious
        call -- Rerank(client, model=model) -- refuse on every
        sentence-transformers server."""
        from hopai.rerankers import Rerank

        record: dict = {}
        imported(self._module(record))
        client, model = rerank_client_from_env("sentence-transformers", model="x")
        rerank = Rerank(client, model=model, document_from=".properties.title")
        assert rerank.model is None
        with pytest.raises(ValueError, match="already IS the model"):
            Rerank(client, model="x", document_from=".properties.title")

    def test_the_embedding_twin_still_builds_a_sentence_transformer(self, imported):
        """The regression this pairs with: teaching the shared
        sentence-transformers entry about CrossEncoder would silently
        turn --embed-provider into a reranker that cannot embed."""
        record: dict = {}
        imported(self._module(record))
        embedder_from_env("sentence-transformers",
                          env={"SENTENCE_TRANSFORMERS_MODEL": "all-MiniLM-L6-v2"})
        assert record == {"SentenceTransformer": "all-MiniLM-L6-v2"}
        assert "CrossEncoder" not in record


class TestTheEmbeddingRefusalsAreUnchanged:
    """The `kind` parameter defaults to EMBEDDING, and this is what
    proves the default is still what the embedding builders get.

    _need(), _import() and _model_for() grew a trailing argument to
    serve two roles. Every embedding builder calls them WITHOUT it, so a
    default flipped to RERANKING -- or a builder that started passing
    one -- would print `--rerank-provider openai needs $OPENAI_API_KEY`
    to someone running --embed-provider, and no existing assertion on
    "$OPENAI_API_KEY" would notice."""

    @pytest.mark.parametrize("provider", sorted(provider_names()))
    def test_no_embedding_refusal_ever_mentions_a_rerank_flag(self, imported, provider):
        client, constructor = TestEveryRefusalNamesItsProvider.BUILDERS[provider]
        imported(fake_module(**{constructor: client}))
        with pytest.raises(ProviderError) as caught:
            embedder_from_env(provider, env={})
        message = str(caught.value)
        assert "--rerank-provider" not in message and "--rerank-model" not in message
        assert f"--embed-provider {provider} " in message

    @pytest.mark.parametrize("provider", sorted(provider_names()))
    def test_a_missing_embedding_model_still_names_embed_model(self, imported, provider):
        client, constructor = TestEveryRefusalNamesItsProvider.BUILDERS[provider]
        imported(fake_module(**{constructor: client}))
        with pytest.raises(ProviderError) as caught:
            embedder_from_env(provider, env=dict(TestEveryRefusalNamesItsProvider.FILLED))
        message = str(caught.value)
        assert "--embed-model" in message and "--rerank-model" not in message

    @pytest.mark.parametrize("provider, extra", [
        ("openai", "openai"), ("azure-openai", "openai"), ("cohere", "cohere"),
        ("voyage", "voyageai"), ("google", "google"),
        ("sentence-transformers", "sentence-transformers"),
    ])
    def test_a_missing_embedding_package_still_names_embed_provider(
            self, absent, provider, extra):
        """The real _import(), whose `kind` default is the one under
        test -- the faked one above cannot see it."""
        absent(providers.PROVIDERS[provider].package)
        with pytest.raises(ProviderError) as caught:
            embedder_from_env(provider, env={}, model="m")
        message = str(caught.value)
        assert f"--embed-provider {provider} needs the " in message
        assert f'pip install "hopai[{extra}]"' in message
        assert "--rerank-provider" not in message

    def test_the_embedding_reason_is_still_about_the_stored_vectors(self):
        """The sentence an operator reads when they omit --embed-model.
        Swapping the two `why` strings would be invisible to every
        assertion about flags."""
        assert providers.EMBEDDING.why == (
            "the model that answers a query has to be the model that wrote the stored "
            "vectors, and only you know which that was")
        assert providers.EMBEDDING.provider_flag == "--embed-provider"
        assert providers.EMBEDDING.model_flag == "--embed-model"
        assert providers.RERANKING.provider_flag == "--rerank-provider"
        assert providers.RERANKING.model_flag == "--rerank-model"
