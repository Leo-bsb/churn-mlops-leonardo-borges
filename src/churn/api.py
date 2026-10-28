"""API FastAPI de inferência do modelo de churn (Enc. 12 — Bloco 4).

Encapsula o pré-processamento que hoje é responsabilidade de quem consome o
``champion``: ``add_derived_features`` + o ``CategoricalEncoder`` salvo junto
com o modelo. Não reimplementa nenhuma codificação — importa de ``churn.features``.

Rodar localmente::

    uv run uvicorn churn.api:app --port 8000

Variáveis de ambiente (todas opcionais):

- ``CHURN_MODEL_PATH``       caminho do artefato (padrão ``artifacts/churn_model.joblib``)
- ``CHURN_DECISION_THRESHOLD`` corte de classificação (padrão ``0.5``; valor
  inválido não derruba o import: cai para ``0.5`` com aviso no log)
- ``CHURN_CORS_ORIGINS``     origens separadas por vírgula (padrão ``*``)
"""
from __future__ import annotations

import json
import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from churn import __version__ as MODEL_VERSION
from churn.features import add_derived_features

# ---------------------------------------------------------------------------
# Configuração via ambiente (nomes com prefixo CHURN_, iguais aos de Settings)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", "artifacts/churn_model.joblib"))
METRICS_PATH = Path(os.getenv("CHURN_METRICS_PATH", "artifacts/model_metrics.json"))
WEB_DIR = Path(os.getenv("CHURN_WEB_DIR", "web"))
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CHURN_CORS_ORIGINS", "*").split(",") if o.strip()
]

logger = logging.getLogger("churn.api")


def _decision_threshold_from_env() -> float:
    """A9: um ``CHURN_DECISION_THRESHOLD`` inválido nunca pode derrubar o import.

    Valor inválido (não numérico, NaN ou fora de [0, 1]) cai para o fallback
    seguro ``0.5``, com aviso registrado no log.
    """
    raw = os.getenv("CHURN_DECISION_THRESHOLD", "0.5")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "CHURN_DECISION_THRESHOLD=%r é inválido; usando fallback 0.5", raw
        )
        return 0.5
    if not 0.0 <= value <= 1.0:
        logger.warning(
            "CHURN_DECISION_THRESHOLD=%r está fora de [0, 1]; usando fallback 0.5",
            value,
        )
        return 0.5
    return value


DECISION_THRESHOLD = _decision_threshold_from_env()

# ---------------------------------------------------------------------------
# Estado do modelo (carregado no startup; nunca derruba o processo)
# ---------------------------------------------------------------------------


class ModelState:
    """Modelo + encoder em memória, com falha controlada se o artefato faltar."""

    def __init__(self) -> None:
        self.model = None
        self.encoder = None
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.encoder is not None

    def load(self) -> None:
        try:
            bundle = joblib.load(MODEL_PATH)
            self.model = bundle["model"]
            self.encoder = bundle["encoder"]
            self.load_error = None
        except Exception as exc:  # artefato ausente/corrompido: 503, não crash
            self.model = None
            self.encoder = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def encode(self, payload: "ChurnRequest") -> pd.DataFrame:
        """Payload -> DataFrame na ordem do treino -> derivadas -> encoder salvo.

        Reusa ``add_derived_features`` e ``encoder.transform`` (C5): a única
        codificação aplicada é a que foi aprendida no treino.
        """
        row = payload.model_dump()
        frame = add_derived_features(pd.DataFrame([row]))
        encoded = self.encoder.transform(frame)
        return encoded[list(self.model.feature_names_in_)]


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="Churn Inference API",
    description="API de inferência do modelo RandomForest de churn (projeto MLOps).",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _finite(value):
    """Troca inf/NaN por string legível (B1): o default do FastAPI ecoaria o
    valor inválido no ``detail`` do 422 e ``json.dumps`` de ``inf`` levanta
    ``ValueError`` -> 500. Nenhum ``inf``/``NaN`` sobrevive à sanitização."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return _finite(obj)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    """422 padrão do FastAPI, porém com os inputs inf/NaN neutralizados."""
    try:
        return await request_validation_exception_handler(request, exc)
    except (ValueError, TypeError):
        # o handler padrão não consegue serializar inf/NaN no eco do input
        return JSONResponse(
            status_code=422,
            content={"detail": _sanitize(exc.errors())},
        )

# ---------------------------------------------------------------------------
# Contratos de request/response (Pydantic — validação estrutural vira 422)
# ---------------------------------------------------------------------------

SimNao = Literal["Yes", "No"]
SemServicoTelefonico = Literal["No", "Yes", "No phone service"]
ServicoInternet = Literal["No", "Yes", "No internet service"]


class ChurnRequest(BaseModel):
    """Um cliente no formato do dataset Telco (todos os campos obrigatórios)."""

    model_config = {"extra": "forbid"}

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: SimNao
    Dependents: SimNao
    tenure: int = Field(ge=0, le=72)
    PhoneService: SimNao
    MultipleLines: SemServicoTelefonico
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: ServicoInternet
    OnlineBackup: ServicoInternet
    DeviceProtection: ServicoInternet
    TechSupport: ServicoInternet
    StreamingTV: ServicoInternet
    StreamingMovies: ServicoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: SimNao
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    # B1: ``allow_inf_nan=False`` rejeita ``Infinity``/``NaN`` no JSON (422) —
    # nenhum ``inf``/``NaN`` chega a ``predict_proba`` (evita 500).
    MonthlyCharges: float = Field(ge=0, le=200, allow_inf_nan=False)
    TotalCharges: float = Field(ge=0, allow_inf_nan=False)


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: Literal["baixo", "medio", "alto"]
    decision_threshold: float
    model_version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": state.loaded,
        "model_version": MODEL_VERSION,
        "timestamp": _utc_now(),
    }


@app.get("/model-info")
def model_info() -> dict:
    if not state.loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo não carregado ({state.load_error or 'artefato ausente'}).",
        )
    metrics: dict = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    names_attr = getattr(state.model, "feature_names_in_", None)
    imp_attr = getattr(state.model, "feature_importances_", None)
    names = [] if names_attr is None else list(names_attr)
    importances = sorted(
        (
            {"feature": name, "importance": round(float(imp), 6)}
            for name, imp in zip(names, [] if imp_attr is None else list(imp_attr))
        ),
        key=lambda d: d["importance"],
        reverse=True,
    )
    return {
        "model_type": type(state.model).__name__,
        "model_version": MODEL_VERSION,
        "n_features": int(state.model.n_features_in_),
        "n_estimators": int(getattr(state.model, "n_estimators", 0)),
        "classes": [int(c) for c in state.model.classes_],
        "decision_threshold": DECISION_THRESHOLD,
        "metrics": metrics.get("metrics", {}),
        "feature_importances": importances,
    }


def _risk_level(p: float) -> str:
    if p < 0.35:
        return "baixo"
    if p < 0.60:
        return "medio"
    return "alto"


@app.post("/predict", response_model=PredictResponse)
def predict(payload: ChurnRequest) -> PredictResponse:
    if not state.loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo indisponível para inferência. "
                f"Verifique o artefato em {MODEL_PATH} ({state.load_error or 'não carregado'}). "
                "O serviço continua no ar; nenhum processo foi derrubado."
            ),
        )
    encoded = state.encode(payload)
    # A4: arredonda UMA vez e deriva prediction e risk_level do mesmo valor
    # exibido em ``churn_probability`` (as faixas de risco ficam consistentes
    # com o número que o cliente vê, inclusive nas fronteiras 0.35 / 0.60).
    probability = round(float(state.model.predict_proba(encoded)[0][1]), 4)
    return PredictResponse(
        churn_probability=probability,
        churn_prediction=probability >= DECISION_THRESHOLD,
        risk_level=_risk_level(probability),
        decision_threshold=DECISION_THRESHOLD,
        model_version=MODEL_VERSION,
    )


# ---------------------------------------------------------------------------
# Frontend (web/ é criado por outra unidade; montar sem quebrar se não existir)
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index() -> Response:
    """Root: serve web/index.html se existir; senão, mensagem JSON simples."""
    index_html = WEB_DIR / "index.html"
    if index_html.is_file():
        return FileResponse(index_html)
    return JSONResponse(
        {
            "message": "Churn Inference API. Frontend ainda não construído.",
            "docs": "/docs",
            "health": "/health",
        },
        status_code=404,
    )


if WEB_DIR.is_dir():
    # Montado por último: StaticFiles só recebe o que as rotas acima não
    # pegaram (/, /health, /model-info, /predict, /docs têm precedência).
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
