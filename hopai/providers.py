"""
hopai.providers

Turn a PROVIDER NAME plus the environment into an Embedder, for callers
that configure a process rather than write Python: `hopai-mcp
--embed-provider azure-openai`, `hopai-api`, a container handed its
secrets by an orchestrator.

The Python API does not need this and is still the better answer where
it applies. `Vector("summary", 1536, embed=openai.OpenAI())` hands hopai
a client the application already built -- its timeout, its proxy, its
credential rotation, its retry policy. This module exists for the one
caller that cannot do that: a process started from a command line with
no application around it.

WHY THIS IS NOT IN embeddings.py. That module imports no provider
package, ever, and matches clients by module root and attribute shape
precisely so `hopai[openai]` stays a convenience alias rather than a
coupling. Building a client FROM A NAME cannot avoid the import -- it is
the entire job. So it lives here instead, and every import happens
INSIDE the builder, which keeps the property that actually mattered:
`import hopai` still pulls in no provider, and no provider is imported
unless an operator named it on a command line.

FAILING LOUDLY is the point of the module rather than a quality of it. A
server that starts without a working embedder and discovers it on the
first search has turned a configuration mistake into a user's failed
query, minutes or days later, somewhere else. So every failure -- an
unknown name, a package that is not installed, a variable that is not
set, a model that was never chosen -- raises HERE, at build time, names
the exact thing missing, and says how to supply it.

    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
    export AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
    hopai-mcp --dsn ... --vector nodes:summary:3072 --embed-provider azure-openai

WHICH MODEL IS NOT GUESSED. Every provider that takes a model name
requires one, from --embed-model or from its own environment variable,
and refuses without it. A default would be a silently chosen model, and
a silently chosen model is a silently different set of neighbours from
the ones already in the database -- the failure this library exists to
refuse rather than approximate. The examples in each error are examples,
not fallbacks.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from .embeddings import Embedder

#: Azure pins its REST contract in the query string rather than the SDK
#: version, so a default here is a compatibility choice and not a model
#: choice -- which is why this one HAS a default and the model does not.
DEFAULT_AZURE_API_VERSION = "2024-02-01"


class ProviderError(ValueError):
    """A provider could not be built from the environment.

    Its own class because the caller that catches it is a start-up path
    -- a CLI printing to stderr and exiting, a container failing its
    health check -- and those want to tell an operator's mistake apart
    from a bug in a traversal."""


@dataclass(frozen=True)
class _Provider:
    """One name an operator can pass, and what it needs to work."""
    package: str            # the import name
    extra: str              # pip install "hopai[<extra>]"
    credentials: tuple      # env vars that MUST be set
    model_var: Optional[str]  # where the model/deployment name comes from
    model_example: str
    build: Callable         # (env, model) -> (client, model_or_None)
    note: str = ""


def _need(env: dict, name: str, provider: str, extra_help: str = "") -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise ProviderError(
            f"--embed-provider {provider} needs ${name} and it is unset or empty. "
            f"Export it before starting.{extra_help}"
        )
    return value


def _import(module: str, extra: str, provider: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:
        raise ProviderError(
            f"--embed-provider {provider} needs the {module!r} package and it is not "
            f'installed -- pip install "hopai[{extra}]"'
        ) from exc


def _model_for(env: dict, spec: _Provider, provider: str, model: Optional[str]) -> str:
    """The model name, from the flag or the environment, or a refusal.

    Never a default. See the module docstring: a guessed model is a
    guessed set of neighbours, and nothing downstream can tell."""
    chosen = (model or env.get(spec.model_var or "") or "").strip()
    if not chosen:
        raise ProviderError(
            f"--embed-provider {provider} needs a model and none was given. Pass "
            f"--embed-model {spec.model_example} or export "
            f"${spec.model_var}={spec.model_example}. hopai will not pick one for "
            f"you: the model that answers a query has to be the model that wrote the "
            f"stored vectors, and only you know which that was.{spec.note}"
        )
    return chosen


# ---------------------------------------------------------------------
# The builders. Each imports its own package and nothing else.
# ---------------------------------------------------------------------

def _build_openai(env: dict, model: Optional[str]):
    openai = _import("openai", "openai", "openai")
    options: dict = {"api_key": _need(env, "OPENAI_API_KEY", "openai")}
    # Both are how an OpenAI-COMPATIBLE endpoint is reached (vLLM,
    # Together, a gateway), which is why they are read rather than
    # ignored: the provider name says which SDK, not which vendor.
    if env.get("OPENAI_BASE_URL"):
        options["base_url"] = env["OPENAI_BASE_URL"]
    if env.get("OPENAI_ORG_ID"):
        options["organization"] = env["OPENAI_ORG_ID"]
    return openai.OpenAI(**options), _model_for(env, PROVIDERS["openai"], "openai", model)


def _build_azure_openai(env: dict, model: Optional[str]):
    openai = _import("openai", "openai", "azure-openai")
    client = openai.AzureOpenAI(
        api_key=_need(env, "AZURE_OPENAI_API_KEY", "azure-openai"),
        azure_endpoint=_need(
            env, "AZURE_OPENAI_ENDPOINT", "azure-openai",
            " It is the resource URL, e.g. https://my-resource.openai.azure.com"),
        api_version=(env.get("AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_API_VERSION),
    )
    # On Azure the `model` argument is the DEPLOYMENT name, not the
    # model name -- they are often different strings, and passing the
    # model name where Azure wants the deployment is the single most
    # common way this configuration fails. The variable is named for
    # what Azure calls it so the mistake is harder to make.
    return client, _model_for(env, PROVIDERS["azure-openai"], "azure-openai", model)


def _build_cohere(env: dict, model: Optional[str]):
    cohere = _import("cohere", "cohere", "cohere")
    return (cohere.ClientV2(api_key=_need(env, "COHERE_API_KEY", "cohere")),
            _model_for(env, PROVIDERS["cohere"], "cohere", model))


def _build_voyage(env: dict, model: Optional[str]):
    voyageai = _import("voyageai", "voyageai", "voyage")
    return (voyageai.Client(api_key=_need(env, "VOYAGE_API_KEY", "voyage")),
            _model_for(env, PROVIDERS["voyage"], "voyage", model))


def _build_google(env: dict, model: Optional[str]):
    genai = _import("google.genai", "google", "google")
    # GEMINI_API_KEY is what the SDK itself reads, GOOGLE_API_KEY is what
    # most deployments already export. Accepting both is not indecision:
    # refusing the one the vendor's own client honours would be a trap.
    key = (env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise ProviderError(
            "--embed-provider google needs $GOOGLE_API_KEY (or $GEMINI_API_KEY) and "
            "neither is set. Export one before starting."
        )
    return genai.Client(api_key=key), _model_for(env, PROVIDERS["google"], "google", model)


def _build_sentence_transformers(env: dict, model: Optional[str]):
    module = _import("sentence_transformers", "sentence-transformers", "sentence-transformers")
    name = _model_for(env, PROVIDERS["sentence-transformers"], "sentence-transformers", model)
    # Returns None as the model: the name is baked into the loaded
    # object, and Embedder REFUSES model= for this family rather than
    # accepting a second, ignorable copy of it.
    return module.SentenceTransformer(name), None


#: Every name `--embed-provider` accepts. Adding one is an entry here
#: plus a builder above; nothing else in hopai learns a provider name.
PROVIDERS: dict = {
    "openai": _Provider(
        package="openai", extra="openai",
        credentials=("OPENAI_API_KEY",),
        model_var="OPENAI_EMBEDDING_MODEL",
        model_example="text-embedding-3-small",
        build=_build_openai,
        note=" Optional: $OPENAI_BASE_URL and $OPENAI_ORG_ID.",
    ),
    "azure-openai": _Provider(
        package="openai", extra="openai",
        credentials=("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"),
        model_var="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        model_example="text-embedding-3-large",
        build=_build_azure_openai,
        note=(" On Azure this is the DEPLOYMENT name you created in the portal, which "
              "is often not the same string as the model it serves. Optional: "
              f"$AZURE_OPENAI_API_VERSION (default {DEFAULT_AZURE_API_VERSION})."),
    ),
    "cohere": _Provider(
        package="cohere", extra="cohere",
        credentials=("COHERE_API_KEY",),
        model_var="COHERE_EMBEDDING_MODEL",
        model_example="embed-v4.0",
        build=_build_cohere,
    ),
    "voyage": _Provider(
        package="voyageai", extra="voyageai",
        credentials=("VOYAGE_API_KEY",),
        model_var="VOYAGE_EMBEDDING_MODEL",
        model_example="voyage-3",
        build=_build_voyage,
    ),
    "google": _Provider(
        package="google.genai", extra="google",
        credentials=("GOOGLE_API_KEY",),
        model_var="GOOGLE_EMBEDDING_MODEL",
        model_example="text-embedding-004",
        build=_build_google,
        note=" $GEMINI_API_KEY is accepted in place of $GOOGLE_API_KEY.",
    ),
    "sentence-transformers": _Provider(
        package="sentence_transformers", extra="sentence-transformers",
        credentials=(),
        model_var="SENTENCE_TRANSFORMERS_MODEL",
        model_example="all-MiniLM-L6-v2",
        build=_build_sentence_transformers,
        note=" Runs locally and needs no credentials.",
    ),
}


def provider_names() -> list:
    """Every accepted name, for `--help` and for error messages."""
    return sorted(PROVIDERS)


def describe(provider: str) -> str:
    """One line of "what this provider reads", for --help and for docs."""
    spec = PROVIDERS[provider]
    reads = list(spec.credentials) + ([spec.model_var] if spec.model_var else [])
    return f"{provider}: {', '.join('$' + name for name in reads)}"


def embedder_from_env(provider: str, model: Optional[str] = None,
                      dimensions: Optional[int] = None,
                      env: Optional[dict] = None, **options: Any) -> Embedder:
    """An Embedder for `provider`, built from environment variables.

        embedder_from_env("azure-openai")          # reads AZURE_OPENAI_*
        embedder_from_env("openai", model="text-embedding-3-large")

    Returns an Embedder rather than a bare callable, so everything the
    Python path gets -- batching to the provider's cap, the
    document/query asymmetry, retries with jitter, the hopai.embeddings
    logger -- applies to a CLI-configured server too. `**options` reach
    the Embedder (retries=, backoff=, batch_size=).

    Raises ProviderError, always naming what is missing and how to
    supply it. Nothing here falls back to a default that would let a
    misconfigured server start and fail later."""
    if provider not in PROVIDERS:
        raise ProviderError(
            f"unknown embedding provider {provider!r} -- accepted: "
            f"{', '.join(provider_names())}"
        )
    spec = PROVIDERS[provider]
    env = dict(os.environ if env is None else env)
    client, resolved = spec.build(env, model)
    return Embedder(client, model=resolved, dimensions=dimensions, **options)
