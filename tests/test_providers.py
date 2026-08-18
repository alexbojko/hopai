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

import pytest

from hopai import providers
from hopai.providers import ProviderError, describe, embedder_from_env, provider_names


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
def imported(monkeypatch):
    """Replace providers._import so no SDK has to be installed. Returns a
    dict recording what was asked for, so a test can assert the module
    name and extra a builder would have named."""
    asked: dict = {}

    def install(module):
        def _import(name, extra, provider):
            asked.update(module=name, extra=extra, provider=provider)
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
    def test_it_names_the_pip_extra_that_fixes_it(self, provider, extra):
        """None of these are installed, so this is the real message an
        operator gets. azure-openai maps to the `openai` extra, which is
        the pairing worth pinning: the package is not named for the
        provider."""
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


class TestPassThrough:
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
