# prj_churn

Projeto-fio-condutor do curso de MLOps: previsão de **churn** (classificação
binária tabular). Este repositório é o *depois* da refatoração do Encontro 2 —
o script monolítico `train_churn.py` reorganizado na estrutura modular.

## Estrutura

```
prj_churn/
├─ src/churn/
│  ├─ config.py      # Settings (Pydantic) — caminhos, hiperparâmetros
│  ├─ schema.py      # contrato de dados (pandera) — Enc. 4
│  ├─ data.py        # carga + validação + limpeza
│  ├─ lineage.py     # data_md5 + git_commit (linhagem) — Enc. 6
│  ├─ experiments.py # grid de 9 runs rastreados — Enc. 6
│  ├─ registry.py    # melhor run -> gate -> churn@champion — Enc. 7/8
│  ├─ features.py    # transformações puras + encoder sem vazamento
│  ├─ model.py       # treino (entrypoint)
│  └─ evaluate.py    # métricas
├─ tests/            # pytest (dados, features, schema)
├─ scripts/          # parquet_para_csv.py · make_v2.py (v2 do dado)
├─ data/             # dataset (versionado por DVC — Enc. 5)
├─ dvc.yaml          # pipeline reproduzível (dado + código -> modelo)
├─ Dockerfile        # imagem do treino (uv) — Enc. 3
├─ compose.yml       # docker compose up
└─ pyproject.toml / uv.lock   # deps travadas (uv)
```

## Como rodar (uv)

```bash
uv sync                          # cria o ambiente a partir do pyproject/uv.lock
uv run python -m churn.model     # treina e imprime as métricas
uv run pytest                    # roda os testes
```

Sem uv, para desenvolvimento rápido:

```bash
pip install -e ".[dev]"
python -m churn.model
pytest
```

## O que mudou em relação ao `train_churn.py`

| Script monolítico | Projeto modular |
|---|---|
| caminho hardcoded | `config.py` (Pydantic, sobrescrevível por env) |
| `fillna(2200)` mágico | `TotalCharges` em branco → `0.0` (cliente novo, `tenure=0`) |
| normalização por números mágicos | removida — Random Forest não precisa de escala |
| `LabelEncoder` no dataset inteiro (vazamento) | `CategoricalEncoder` com `fit` só no treino |
| sem `random_state` (acurácia instável) | semente fixa em split e modelo |
| só imprime `accuracy` | `evaluate` retorna accuracy, precision, recall, f1, roc_auc |
| tudo em nível de módulo | funções puras + orquestração em `train()` |

## Dados (DVC)

O `data/churn.csv` não está no Git: o Git guarda só o ponteiro `data/churn.csv.dvc`.

```bash
uv sync
dvc pull                 # baixa o dado do remote DVC
uv run dvc repro         # roda o pipeline (treino) só se algo mudou
```

Sem o remote, gere o CSV a partir do parquet: `uv run python scripts/parquet_para_csv.py churn.parquet`.

## Docker

```bash
docker build -t churn:0.1 .
docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts churn:0.1
# ou: docker compose up
```

O dado entra por volume (ou `dvc pull`), nunca copiado para a imagem.

## MLflow (Bloco 2 — Encontros 6, 7 e 8)

```bash
uv run python -m churn.model         # 1 treino rastreado (run "rf") + pickle p/ o DVC
uv run python -m churn.experiments   # grid 3x3 = 9 runs (~30 s, ~300 MB em mlruns/)
uv run python -m churn.registry      # melhor run -> gate roc_auc >= 0.80 -> churn@champion
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://localhost:5000
```

- O backend (`sqlite:///mlflow.db`) é fixado em `config.py`, não em variável de ambiente.
  Para outro destino: `CHURN_MLFLOW_TRACKING_URI=...`.
- Cada run guarda params, métricas, as tags `data_md5` e `git_commit`, o modelo (com
  signature + input_example, cloudpickle) e o `encoder`. **Faça commit antes de treinar**,
  senão o `git_commit` registrado não corresponde ao código que rodou.
- O gate (`min_roc_auc`) e os nomes do modelo/alias também estão em `config.py`.
- `mlflow.db` e `mlruns/` estão no `.gitignore` (não vão para o GitHub).

Carregar o modelo de produção:

```python
from churn.registry import load_champion
model = load_champion()      # mlflow.pyfunc, URI models:/churn@champion
model.predict(x_encoded)     # atenção: espera as features JÁ codificadas (ver limitações)
```

**Limitação conhecida:** o modelo registrado é só o `RandomForest`. As features derivadas
e o `CategoricalEncoder` não estão dentro dele (o encoder é salvo como artefato do run).
Quem consumir o `champion` precisa aplicar o mesmo pré-processamento — tema da API (Enc. 12).

## Próximos encontros

- **Bloco 3** — orquestração com Airflow.
- **Bloco 4** — API (FastAPI) com CI/CD.
