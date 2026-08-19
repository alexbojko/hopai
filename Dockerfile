# The image both hopai services run from -- `hopai-api` and `hopai-mcp`
# are the same package with different entry points, so one image serves
# both and docker-compose picks the command.
FROM python:3.12-slim AS base

# Which optional extras to install. `http` is the API, `mcp` the MCP
# server, `openai` covers BOTH --embed-provider openai and
# --embed-provider azure-openai (Azure is the same package). Override to
# add another provider:
#
#   docker compose build --build-arg HOPAI_EXTRAS=http,mcp,cohere
#
# sentence-transformers is deliberately not here by default: it pulls
# torch, which is several hundred megabytes for an image most people are
# running against a hosted provider.
ARG HOPAI_EXTRAS=http,mcp,openai

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# psycopg2-binary ships wheels, so no libpq-dev and no compiler are
# needed -- which is most of why this image is small and fast to build.
COPY pyproject.toml README.md ./
COPY hopai ./hopai
RUN pip install --no-cache-dir ".[${HOPAI_EXTRAS}]"

# Not root. The container reaches a database and answers a browser;
# neither needs privileges, and the default user is one less thing to
# have to think about later.
RUN useradd --create-home --uid 10001 hopai
USER hopai

# Overridden per service in docker-compose.yml.
CMD ["hopai-api"]
