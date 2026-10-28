# Imagem da API de inferência (Enc. 12 / Bloco 4) — AUTOCONTIDA: API + frontend
# + artefato do modelo. Dados de treino (data/) e mlruns/ nunca entram (ver
# .dockerignore). O compose ainda pode montar ./artifacts por cima para usar um
# modelo recém-treinado sem rebuild, mas o `docker run` puro já serve a UI.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim
ENV UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    CHURN_MODEL_PATH=/app/artifacts/churn_model.joblib \
    PORT=8000

WORKDIR /app

# deps antes do código: mudar um .py não reinstala o mundo
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# código da API + artefato do modelo
COPY src/ ./src/
RUN uv sync --frozen --no-dev

COPY artifacts/churn_model.joblib /app/artifacts/churn_model.joblib
COPY artifacts/model_metrics.json /app/artifacts/model_metrics.json

# A1: o frontend entra na imagem. Se web/ não existir no build context, o COPY
# falha de forma clara ("Copy failed: web/: not found") — sem volume como
# substituto. O CI garante a presença de web/ antes do build.
COPY web/ ./web/

# A8: roda como usuário não-root, com chown completo de /app (venv + artefatos)
RUN adduser --system --group --home /app appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# A2: binários do venv direto (nada de `uv run`): o runtime nunca sincroniza
# dependências nem instala o grupo dev. Healthcheck respeita $PORT (Render).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD /app/.venv/bin/python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=4).status == 200 else 1)"

# A7: CMD respeita $PORT (injetado pela Render); default 8000 (compose/local).
# exec + sh -c apenas para expandir a variável mantendo o uvicorn como processo.
CMD ["/bin/sh", "-c", "exec /app/.venv/bin/uvicorn churn.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
