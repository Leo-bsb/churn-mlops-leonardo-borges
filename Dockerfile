FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
ENV UV_LINK_MODE=copy
WORKDIR /app

# deps antes do código: mudar um .py não reinstala o mundo
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY src/ ./src/
RUN uv sync --frozen

# O dado NÃO é copiado para a imagem (Enc. 5): entra por volume (ou dvc pull).
CMD ["uv", "run", "python", "-m", "churn.model"]
