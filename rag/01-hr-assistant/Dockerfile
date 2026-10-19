# syntax=docker/dockerfile:1
# Two images from one file:
#   docker build --target api    -t hr-api .      FastAPI (small: no Docling/torch)
#   docker build --target worker -t hr-worker .   ingest / cleanup / migrate jobs (Docling + models baked in)

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv
RUN useradd --create-home --uid 10001 app
WORKDIR /app

# ---------------------------------------------------------------- dependencies, then code (better caching)
FROM base AS api-build
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM base AS worker-build
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra ingest --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra ingest

# ---------------------------------------------------------------- api: what Cloud Run serves
FROM base AS api
COPY --from=api-build /app /app
ENV PATH="/app/.venv/bin:$PATH" PORT=8080
USER app
EXPOSE 8080
# Cloud Run sets $PORT. One process per container: Cloud Run scales by adding containers.
CMD ["sh", "-c", "exec uvicorn hr_assistant.api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]

# ---------------------------------------------------------------- worker: Cloud Run jobs
FROM base AS worker
COPY --from=worker-build /app /app
COPY alembic.ini ./
COPY migrations ./migrations
ENV PATH="/app/.venv/bin:$PATH" DOCLING_ARTIFACTS_PATH=/app/docling-models
# Bake the parsing models into the image: otherwise every job run downloads ~1 GB before working.
RUN docling-tools models download -o /app/docling-models && chown -R app /app/docling-models
USER app
ENTRYPOINT ["hr-assistant"]
CMD ["--help"]
