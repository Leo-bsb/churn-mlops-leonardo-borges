# Roteiro: do zip ao GitHub com DVC

Todos os comandos abaixo foram executados numa simulação (git remoto local +
remote DVC em pasta) e funcionaram. **Exceção:** o `docker build` não foi
testado (o ambiente da simulação não tem Docker) — rode você.

Pré-requisitos: Git, Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker.

---

## 0. Antes de tudo (no GitHub, pelo navegador)

1. Crie um repositório **vazio** (sem README, sem .gitignore, sem licença).
2. Copie a URL (`https://github.com/SEU_USUARIO/SEU_REPO.git`).

## 1. Preparar a pasta local

Abra esta pasta (`prj_churn`) no VS Code (`File > Open Folder`) e use o terminal
integrado (`Ctrl+'`).

```bash
uv sync                                    # cria .venv e instala tudo do uv.lock
uv run python scripts/parquet_para_csv.py CAMINHO/churn.parquet
#  -> deve imprimir: OK: igual à v1 do professor   (md5 d390bd07...)
uv run pytest -q                           # esperado: 11 passed
```

## 2. Conectar ao GitHub  (⚠ a ordem importa)

O `data/churn.csv` só pode ser adicionado ao Git **depois** do `dvc add`,
senão o CSV entra no histórico.

```bash
git init -b main
git config user.name  "Seu Nome"           # só se ainda não configurou
git config user.email "seu@email.com"
git remote add origin https://github.com/SEU_USUARIO/SEU_REPO.git

dvc init                                   # liga o DVC ao repo
dvc add data/churn.csv                     # cria data/churn.csv.dvc + data/.gitignore
```

## 3. Remote do DVC (onde os bytes moram)

Comece por um remote local (uma pasta **fora** do repositório):

```bash
dvc remote add -d local ../dvc_remote
```

## 4. Primeiro commit + push

```bash
git add -A
git status                                 # confira: NÃO deve aparecer data/churn.csv nem .venv
git commit -m "projeto churn: estrutura, schema pandera, Dockerfile, DVC (dados v1)"
git push -u origin main
dvc push                                   # envia o CSV (v1) para o remote
```

## 5. Rodar o pipeline (gera o .pkl)

```bash
uv run dvc repro                           # roda o stage `treino`, grava dvc.lock
git add -A
git commit -m "pipeline dvc: treino sobre dados v1"
git push
dvc push                                   # envia também o churn_model.pkl
```

> **Sobre "subir o .pkl por terminal":** o modelo sobe pelo `dvc push` (ele é
> `outs` do `dvc.yaml`), não pelo `git push`. No GitHub aparece só o
> `dvc.lock` com o hash. Se o professor pediu o `.pkl` no próprio GitHub,
> confirme com ele — vai contra o que o slide 6 do Encontro 5 ensina.

## 6. Criar a v2 do dado e alternar entre versões (checkpoint do Enc. 5)

```bash
uv run python scripts/make_v2.py           # +100 linhas válidas -> 7.144 linhas
dvc add data/churn.csv
uv run dvc repro                           # retreina com a v2 (atualiza dvc.lock)
git add -A
git commit -m "dados v2 (+100 linhas)"
git push
dvc push
```

Voltar para a v1 e depois para a v2:

```bash
git checkout HEAD~1 -- data/churn.csv.dvc dvc.lock
dvc checkout                               # churn.csv volta a ter 7.044 linhas
wc -l data/churn.csv

git checkout HEAD -- data/churn.csv.dvc dvc.lock
dvc checkout                               # de volta às 7.144
```

> ⚠ Use `HEAD~1` **só se** o commit "dados v2" for o último. Confira com
> `git log --oneline`.
>
> ⚠ Rode o treino sempre por `dvc repro`. Se rodar `python -m churn.model`
> direto, o `.pkl` fica diferente do `dvc.lock` e o `dvc checkout` recusa
> sobrescrever ("Can't remove unsaved files"). Se acontecer:
> `dvc checkout data/churn.csv.dvc` (só o dado) ou `dvc checkout --force`.

## 7. Provar que o contrato pega dado ruim (tarefa do Enc. 4)

Já existe em `tests/test_schema.py` (tenure = 999, `Contract` = "Vitalício",
ID duplicado, MonthlyCharges negativo). Para mostrar no terminal:

```bash
uv run pytest tests/test_schema.py -v
```

## 8. Docker (Enc. 3 — não testado por mim)

```bash
docker build -t churn:0.1 .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/artifacts:/app/artifacts" churn:0.1
# ou
docker compose up
```

Requer `data/churn.csv` presente (após `dvc pull` ou `dvc checkout`).
No Windows, prefira rodar pelo WSL2 (o slide de atrito do Enc. 3 avisa dos
caminhos do `-v`).

## 9. Reproduzir do zero (o que o professor pode testar)

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPO.git && cd SEU_REPO
uv sync
dvc pull        # precisa do remote acessível — ver aviso abaixo
uv run dvc repro
```

> ⚠ Um remote em **pasta local** só funciona na sua máquina. Quem clonar o
> repositório **não** consegue `dvc pull`. Pergunte ao professor se o
> remote local basta para a entrega (o slide 9 diz para começar por ele) ou se
> quer um remote compartilhado (Google Drive, S3, DagsHub).

---

# Parte 2 — Bloco 2 (MLflow: Encontros 6, 7 e 8)

Pré-requisito: a Parte 1 concluída (repo no GitHub, DVC funcionando, `data/churn.csv` presente).

## 10. Atualizar a codebase

Copie do zip novo para o seu repo, **sobrescrevendo** estes arquivos:

```
pyproject.toml   uv.lock   .gitignore   .dockerignore   README.md
src/churn/config.py   src/churn/model.py
src/churn/lineage.py  src/churn/experiments.py  src/churn/registry.py     (novos)
tests/test_lineage.py  tests/test_mlflow_pipeline.py                        (novos)
```

Depois:

```bash
uv sync                                  # instala mlflow (3.x) e pyyaml
uv run pytest -q                         # esperado: 18 passed (~25 s)
git checkout -b feat/mlflow              # opcional, mas é o hábito do curso
git add -A
git status                               # NÃO deve listar mlflow.db nem mlruns/
git commit -m "feat: tracking, registry e empacotamento com MLflow"
```

> Faça o commit **antes** de treinar: o run grava o hash do commit atual (`git_commit`).

## 11. Rodar o Bloco 2 (nesta ordem)

```bash
uv run python -m churn.model             # 1 run "rf" (roc_auc ≈ 0.805) — ou: uv run dvc repro
uv run python -m churn.experiments       # 9 runs; o melhor deve ser rf-n200-d8 = 0.8195
uv run python -m churn.registry          # -> "churn v1 @champion (run rf-n200-d8, roc_auc 0.8195)"
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abra http://localhost:5000, entre no experimento `churn`, ordene por `roc_auc` e
tire um print (comparação dos runs, aba *Models* com `churn` v1 e o alias `champion`).
Isso costuma ser a evidência que o professor quer ver.

Sobre `dvc repro`: cada `dvc repro` agora também cria um run no MLflow. O resultado do
`dvc.lock` não muda por causa do MLflow, só por causa de dado ou código.

## 12. O que mudou em relação aos slides (e por quê)

| Slide | Como está no código | Motivo |
|---|---|---|
| 33 `prepare(cfg)` | criei `prepare()` em `model.py`, devolvendo splits já codificados | a função não existe no projeto |
| 12 grid | cada run do grid também salva o modelo | o `register_model(runs:/.../model)` do slide 34 falha se o run não tiver modelo |
| 34 `search_runs(["churn"])` | `search_runs(experiment_names=[...])` | o 1º argumento posicional é **ids**, não nomes |
| 32 `export MLFLOW_TRACKING_URI` | URI fixada em `config.py` | evita runs "sumindo" entre terminais (ponto de atrito do slide 36) |
| 28 gate | `promote_best()` em `registry.py`, limiar em `config.py` | slide mostra só o `if`; aqui é função reutilizável e idempotente |
| 36 `mlartifacts/` | os artefatos vão para `mlruns/` no MLflow 3.16 | verificado na execução; ambos estão no `.gitignore` |
| 35 signature só no melhor | signature + input_example em **todos** os runs | simplifica e não custa nada |

## 13. Docker com MLflow (não testado)

O `docker compose up` continua treinando, mas o `mlflow.db`/`mlruns/` ficam **dentro do
container** e somem quando ele termina; `git_commit` sai `unknown` (não há git na imagem)
e `data_md5` só sai preenchido porque o `./data` é montado (traz o `churn.csv.dvc`).
Para guardar os runs do container, monte também `./mlruns` e um caminho para o sqlite —
peça e eu ajusto o `compose.yml`.
