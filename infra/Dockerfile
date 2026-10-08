# syntax=docker/dockerfile:1
# Multi-stage: base -> dev (CPU/dev tools) and train (GPU training).
# NOTE: the CUDA base tag below must be verified on Docker Hub for your driver version.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04

FROM ${CUDA_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /workspace
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN uv python install 3.11

# Development image: tooling for tests, linting, notebooks.
FROM base AS dev
RUN uv sync --frozen --group dev --group train --group data --no-install-project
COPY . .
RUN uv sync --frozen --group dev --group train --group data
CMD ["bash"]

# Training image: no dev tools. Needs `--gpus all` (NVIDIA Container Toolkit) at run time.
FROM base AS train
RUN uv sync --frozen --no-dev --group train --group data --no-install-project
COPY . .
RUN uv sync --frozen --no-dev --group train --group data
CMD ["python", "scripts/hello_world.py"]
