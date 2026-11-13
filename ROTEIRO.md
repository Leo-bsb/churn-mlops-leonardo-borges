# Roteiro de reprodução

Passo a passo para sair do `git clone` e chegar na API respondendo — com os
comandos na ordem em que fazem sentido. Para o *porquê* de cada decisão, veja o
[`README.md`](README.md).

Pré-requisitos: Git, [uv](https://docs.astral.sh/uv/) e Docker (opcional, para
as etapas 7 e 8). O Python é baixado pelo `uv` conforme o `uv.lock`.

---

## 1. Clonar e preparar o ambiente

```bash
git clone https://github.com/Leo-bsb/churn-mlops-leonardo-borges.git
cd churn-mlops-leonardo-borges
uv sync
```

`uv sync` cria o `.venv` e instala exatamente o que está no `uv.lock`
(produção + grupo `dev`: pytest, dvc, pyarrow, httpx).

## 2. Rodar a suíte de testes

```bash
uv run pytest -q          # esperado: 62 passed
```

Se os 62 passarem, o contrato de dados, as features, o treino, o MLflow e a API
estão funcionando neste ambiente.

## 3. Rodar o pipeline de treino

```bash
uv run dvc repro
```

O que acontece:

1. `data/churn.csv` passa pelo contrato (pandera);
2. features derivadas + split estratificado 75/25 (seed 42);
3. o `CategoricalEncoder` é ajustado **só no treino**;
4. o Random Forest (200 árvores, profundidade 8) é treinado e avaliado;
5. um run é registrado no MLflow (`sqlite:///mlflow.db`);
6. saem dois artefatos:
   - `artifacts/churn_model.pkl` — bundle `{"model", "encoder"}` (saída do DVC);
   - `artifacts/model_metrics.json` — cartão do modelo (versionado no Git).

Esperado no teste: `accuracy 0.7683`, `precision 0.6694`, `recall 0.5689`,
`f1 0.6151`, `roc_auc 0.8195`.

> `dvc repro` só reexecuta o stage se um `dep` (dado, `src/churn`,
> `pyproject.toml` ou `uv.lock`) mudou. Para forçar: `uv run dvc repro --force`.

## 4. Levar o modelo novo para a API

A API lê **apenas** `artifacts/churn_model.joblib` (comprimido), nunca o `.pkl`:

```bash
uv run python scripts/export_joblib.py
```

Depois disso:

- rodando local com `uvicorn`, basta reiniciar o processo;
- rodando em Docker, é preciso **rebuild** (`docker compose build api`).

## 5. Experimentos e promoção do champion

```bash
uv run python -m churn.experiments    # 9 runs (~30 s)
uv run python -m churn.registry       # melhor run -> gate >= 0.80 -> churn@champion
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir <http://localhost:5000>, entrar no experimento `churn`, ordenar por
`roc_auc` e conferir a aba *Models* (`churn` com o alias `champion`).

O melhor resultado do grid é `rf-n200-d8` (ROC AUC 0.8195) — o mesmo
`n_estimators`/`max_depth` que é o default de `config.py`.

> Faça commit **antes** de treinar: o run grava o hash do commit atual na tag
> `git_commit`.

## 6. API e frontend local

```bash
uv run uvicorn churn.api:app --port 8000
```

- UI: <http://localhost:8000/>
- Swagger: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

## 7. Docker

```bash
docker compose config                  # valida a configuração
docker compose up --build              # SÓ a API (autocontida: API + UI + modelo)

# treino em container (dado via volume; runs do MLflow persistidos em ./mlflow)
docker compose --profile treino up treino
```

Sem compose:

```bash
docker build -t churn-api:0.1 .
docker run -p 8000:8000 churn-api:0.1
```

## 8. Publicar o frontend no GitHub Pages

1. Push/merge em `main` (o workflow `pages.yml` roda quando `web/**` muda).
2. Em **Settings → Pages**, deixe a fonte em **GitHub Actions**.
3. Em **Settings → Secrets and variables → Actions → Variables**, crie
   `CHURN_API_BASE` com a URL pública do backend (ex.: `https://churn-api.onrender.com`).
4. Sem essa variável, o site sobe e mostra um aviso claro de backend não
   configurado.

## 9. Versionar dados (v1 → v2)

```bash
uv run python scripts/make_v2.py     # 7.043 -> 7.143 clientes
uv run dvc repro                     # retreina e atualiza dvc.lock
git add data/churn.csv dvc.lock
git commit -m "dados v2: +100 clientes"
```

Para voltar à versão anterior:

```bash
git checkout HEAD~1 -- data/churn.csv dvc.lock
uv run dvc repro
```

> Este repositório **não tem remote DVC**: o CSV (988 KB) está versionado no
> Git para o clone ser auto-suficiente. `dvc pull` / `dvc push` não são usados
> aqui. Para migrar ao fluxo canônico, veja o README §4.3.

## 10. Checklist de entrega

- [ ] `uv sync` sem erro
- [ ] `uv run pytest -q` → 62 passed
- [ ] `uv run dvc repro` → ROC AUC 0.8195
- [ ] `uv run python -m churn.registry` → `churn v1 @champion`
- [ ] `docker compose up --build` → UI e `/docs` respondendo
- [ ] CI verde no GitHub Actions (testes + build da imagem)
- [ ] `web/screenshot.png` atualizado, se a UI mudou
