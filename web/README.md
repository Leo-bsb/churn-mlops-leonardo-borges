# Frontend — Churn Predictor (estático)

Frontend **estático, responsivo e sem build** (HTML/CSS/JS puros, sem CDN, sem npm)
para testar o modelo de churn e visualizar o backend FastAPI.

## Arquivos

| Arquivo | Função |
|---|---|
| `index.html` | Formulário de predição (19 campos do contrato), painel do modelo e painel do backend |
| `styles.css` | Estilos responsivos e acessíveis (sem frameworks) |
| `app.js` | Consumo de `/predict`, `/health` e `/model-info`; estados de loading/erro |
| `config.js` | Define `window.CHURN_API_BASE` |

## Configurando a API (`CHURN_API_BASE`)

- `""` (vazio): o frontend chama a **mesma origem** — funciona quando o FastAPI
  serve esta pasta em `/`.
- URL completa (ex.: `https://churn-api.onrender.com`): quando o backend roda em
  outro host (Render, Railway, VM própria com CORS habilitado).

> GitHub Pages **não executa** FastAPI/Streamlit. O Pages serve apenas este site
> estático; o backend precisa estar publicado em outro lugar e ser apontado por
> `CHURN_API_BASE`.

## Rodando localmente (Docker / FastAPI)

Com o backend FastAPI no repositório:

```powershell
# opção 1: Docker (frontend + API na mesma origem)
docker compose up --build
# abrir http://127.0.0.1:8000/

# opção 2: uvicorn direto
uv run uvicorn churn.api:app --port 8000
# abrir http://127.0.0.1:8000/
```

Nesse cenário deixe `CHURN_API_BASE = ""` — o FastAPI serve o frontend em `/`.

> Evite abrir `web/index.html` com duplo clique (`file://`): nessa condição o
> navegador não alcança o backend e a página mostra o aviso "API não
> configurada". Sirva os arquivos pela origem do FastAPI.

## Comportamento do formulário

- Os campos numéricos (`tenure`, `MonthlyCharges`, `TotalCharges`) são validados
  em JS antes do `POST`: valores vazios ou fora da faixa geram mensagem visível
  e foco no primeiro campo inválido — nada é enviado como `0` silenciosamente.
  `tenure` precisa ser inteiro (0–72).
- Quando `PhoneService = No` (ou `InternetService = No`), os selects dependentes
  ficam desabilitados e recebem `No phone service` (ou `No internet service`):
  são os únicos valores que o contrato da API aceita nesses casos.
- Toda chamada de API tem timeout de 15 s; resposta não-JSON (ex.: HTML de um
  proxy) é tratada como erro, nunca como predição bem-sucedida.

## Publicando no GitHub Pages

O fluxo **normal** de publicação é: mergear o branch em `main` — o workflow
`.github/workflows/pages.yml` roda em todo push em `main` que altere `web/**`
ou o próprio workflow e publica a pasta `web/` no Pages.

Além disso, o workflow aceita execução manual (**`workflow_dispatch`**), o que
permite publicar um **branch de trabalho** sem merge: na aba *Actions → Deploy
web to GitHub Pages → Run workflow*, selecione o branch desejado. Útil para
validar o deploy antes de abrir/mergear o PR. Em ambos os casos a repository
variable `CHURN_API_BASE` é injetada em `config.js` pelo passo Node do workflow
(a injeção falha o job se o marcador `window.CHURN_API_BASE = "";` não existir
em `web/config.js`).

Passos para habilitar:

1. Faça push/merge desta pasta para o branch publicado (ou rode o workflow
   manualmente via `workflow_dispatch`, como descrito acima).
2. Em **Settings → Pages**, selecione **Source: GitHub Actions** (ou apenas
   garanta que o workflow rodou).
3. Em **Settings → Secrets and variables → Actions → Variables**, crie a
   repository variable **`CHURN_API_BASE`** com a URL do backend publicado
   (ex.: `https://churn-api.onrender.com`). O workflow injeta esse valor em
   `config.js` durante o deploy.
4. Sem essa variável, o site no Pages mostra um aviso claro de que o backend
   não está configurado.

## Deploy do backend

Publique o FastAPI (imagem do `Dockerfile`) em Render/Railway ou
similar, habilite CORS para a origem do Pages e use a URL resultante em
`CHURN_API_BASE`. Endpoints consumidos: `GET /health`, `GET /model-info` e
`POST /predict` (contrato visível na documentação interativa em `/docs`).

## Validação rápida de sintaxe

```powershell
node --check web/app.js
```
