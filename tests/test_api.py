"""Testes da API de inferência (FastAPI / TestClient).

Cobrem os critérios C1-C5 do contrato: /health, /model-info, POST /predict
(payload válido -> 200; payload inválido -> 422) e o comportamento degraded
(503) quando o artefato do modelo não existe — sem derrubar o processo.

Usam o artefato real ``artifacts/churn_model.joblib`` (somente leitura).
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import churn.api as api_module

# Payload de exemplo do contrato da API.
VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.1,
    "TotalCharges": 1069.2,
}


@pytest.fixture()
def client() -> TestClient:
    """TestClient como context manager: dispara o lifespan (carrega o modelo)."""
    with TestClient(api_module.app) as c:
        yield c


# ---------------------------------------------------------------------------
# C1 — /health
# ---------------------------------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "0.1.0"
    assert "timestamp" in body


# ---------------------------------------------------------------------------
# C2 — /model-info
# ---------------------------------------------------------------------------


def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "RandomForestClassifier"
    assert body["model_version"] == "0.1.0"
    assert body["n_features"] == 20
    assert body["n_estimators"] == 200
    assert body["classes"] == [0, 1]
    assert body["decision_threshold"] == pytest.approx(0.5)
    assert body["metrics"]["roc_auc"] == pytest.approx(0.8053, abs=5e-4)
    assert body["metrics"]["accuracy"] == pytest.approx(0.757, abs=5e-4)
    importances = body["feature_importances"]
    assert len(importances) == 20
    assert importances[0]["feature"] == "Contract"
    assert all(0.0 <= item["importance"] <= 1.0 for item in importances)


# ---------------------------------------------------------------------------
# C3 — POST /predict com payload válido
# ---------------------------------------------------------------------------


def test_predict_valid(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["churn_prediction"] == (body["churn_probability"] >= 0.5)
    assert body["risk_level"] in {"baixo", "medio", "alto"}
    assert body["decision_threshold"] == pytest.approx(0.5)
    assert body["model_version"] == "0.1.0"


def test_predict_risk_level_consistent(client: TestClient) -> None:
    """risk_level deve ser coerente com a probabilidade retornada."""
    body = client.post("/predict", json=VALID_PAYLOAD).json()
    p = body["churn_probability"]
    expected = "baixo" if p < 0.35 else "medio" if p < 0.60 else "alto"
    assert body["risk_level"] == expected


# ---------------------------------------------------------------------------
# A4 — fronteira: churn_prediction e risk_level derivados do valor ARREDONDADO
# exibido em churn_probability (round(..., 4) uma única vez)
# ---------------------------------------------------------------------------


class _StubModel:
    """Modelo fake com probabilidade fixa, para testar as fronteiras de 0.35/0.60."""

    def __init__(self, probability: float) -> None:
        self._probability = probability

    def predict_proba(self, frame):  # assinatura idêntica ao sklearn
        return [[1.0 - self._probability, self._probability]]


@pytest.mark.parametrize(
    "raw, expected_display",
    [
        (0.34994, 0.3499),  # arredonda para baixo: permanece "baixo"
        (0.34996, 0.3500),  # arredonda PARA a fronteira: passa a "medio"
        (0.35, 0.35),       # fronteira exata: medio
        (0.59996, 0.6),     # arredonda PARA a fronteira: passa a "alto"
        (0.59994, 0.5999),  # arredonda para baixo: permanece "medio"
        (0.60, 0.6),        # fronteira exata: alto
        (0.49996, 0.5),     # arredonda PARA o threshold: prediction True
        (0.49994, 0.4999),  # arredonda para baixo: prediction False
    ],
)
def test_prediction_and_risk_derive_from_rounded_probability(
    client: TestClient, monkeypatch, raw: float, expected_display: float
) -> None:
    """A4: a probabilidade é arredondada uma única vez e prediction/risk_level
    são derivados do MESMO valor exibido — nunca do valor cru do modelo."""
    monkeypatch.setattr(api_module.state, "encode", lambda payload: None)
    monkeypatch.setattr(api_module.state, "model", _StubModel(raw))
    body = client.post("/predict", json=VALID_PAYLOAD).json()
    p = body["churn_probability"]
    assert p == pytest.approx(expected_display)
    assert body["churn_prediction"] is (p >= 0.5)
    expected_risk = "baixo" if p < 0.35 else "medio" if p < 0.60 else "alto"
    assert body["risk_level"] == expected_risk


# ---------------------------------------------------------------------------
# A9 — CHURN_DECISION_THRESHOLD inválido não derruba o import (fallback 0.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", ["abc", "0,5", "", "1.5", "-0.1", "nan"])
def test_invalid_decision_threshold_falls_back(monkeypatch, bad_value: str) -> None:
    """Valor inválido fora de [0, 1] ou não numérico cai para 0.5, com aviso."""
    monkeypatch.setenv("CHURN_DECISION_THRESHOLD", bad_value)
    try:
        reloaded = importlib.reload(api_module)
        assert reloaded.DECISION_THRESHOLD == pytest.approx(0.5)
    finally:
        monkeypatch.delenv("CHURN_DECISION_THRESHOLD", raising=False)
        importlib.reload(api_module)


# ---------------------------------------------------------------------------
# C4 — payloads inválidos -> 422
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenure", 999),           # fora de 0..72
        ("Contract", "Vitalício"),  # valor não presente no treino
    ],
)
def test_predict_invalid_field_422(client: TestClient, field: str, value) -> None:
    payload = {**VALID_PAYLOAD, field: value}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field_422(client: TestClient) -> None:
    payload = dict(VALID_PAYLOAD)
    payload.pop("MonthlyCharges")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_extra_field_422(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID_PAYLOAD, "customerID": "X1"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# B1 — inf/NaN em MonthlyCharges/TotalCharges: 422, nunca chegam a predict_proba
# (JSON com Literais Infinity/NaN é aceito pelo parser; 1e400 vira inf no float)
# ---------------------------------------------------------------------------


def _raw_json(field: str, literal: str) -> bytes:
    """VALID_PAYLOAD como JSON cru com ``field`` substituído por um literal."""
    payload = {**VALID_PAYLOAD, field: 0.0}
    text = json.dumps(payload).replace(f'"{field}": 0.0', f'"{field}": {literal}')
    return text.encode("utf-8")


@pytest.mark.parametrize(
    "field, literal",
    [
        ("MonthlyCharges", "Infinity"),
        ("MonthlyCharges", "-Infinity"),
        ("MonthlyCharges", "NaN"),
        ("MonthlyCharges", "1e400"),  # transborda para inf no parse do float
        ("TotalCharges", "Infinity"),
        ("TotalCharges", "-Infinity"),
        ("TotalCharges", "NaN"),
        ("TotalCharges", "1e400"),
    ],
)
def test_predict_inf_nan_422(client: TestClient, field: str, literal: str) -> None:
    """B1: allow_inf_nan=False -> 422 na validação (não 500 em predict_proba)."""
    response = client.post(
        "/predict",
        content=_raw_json(field, literal),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_inf_nan_never_reaches_model(client: TestClient, monkeypatch) -> None:
    """B1: nenhum ``inf``/``NaN`` inválido chega a ``predict_proba`` (sem 500).

    ``predict_proba`` é monitorado; apenas payloads 200 a invocariam.
    """
    called: list = []

    def _guard(frame):
        called.append(frame)
        return [[0.0, 0.5]]

    monkeypatch.setattr(api_module.state.model, "predict_proba", _guard)
    for field in ("MonthlyCharges", "TotalCharges"):
        for literal in ("Infinity", "NaN", "1e400"):
            response = client.post(
                "/predict",
                content=_raw_json(field, literal),
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 422
    assert called == []  # predict_proba nunca foi invocado


# ---------------------------------------------------------------------------
# Documentação e raiz
# ---------------------------------------------------------------------------


def test_docs_and_openapi(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    for route in ("/health", "/model-info", "/predict"):
        assert route in paths


def test_root_never_breaks(client: TestClient) -> None:
    """``/`` serve web/index.html se existir; senão 404 com JSON — nunca explode."""
    response = client.get("/")
    if (api_module.WEB_DIR / "index.html").is_file():
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    else:
        assert response.status_code == 404
        assert response.json().get("message")


# ---------------------------------------------------------------------------
# Degradação graciosa: artefato ausente -> /health model_loaded=false, /predict 503
# ---------------------------------------------------------------------------


def test_model_missing_degrades_gracefully(monkeypatch, tmp_path: Path) -> None:
    """Com CHURN_MODEL_PATH apontando para arquivo inexistente, a API sobe
    e responde 503 em /predict sem derrubar o processo."""
    missing = tmp_path / "nao_existe.joblib"
    monkeypatch.setattr(api_module, "MODEL_PATH", missing)
    state_backup = (api_module.state.model, api_module.state.encoder)
    try:
        with TestClient(api_module.app) as c:  # lifespan recarrega -> falha controlada
            health = c.get("/health")
            assert health.status_code == 200
            assert health.json()["model_loaded"] is False

            predict = c.post("/predict", json=VALID_PAYLOAD)
            assert predict.status_code == 503
            assert "Modelo indisponível" in predict.json()["detail"]
    finally:
        api_module.state.model, api_module.state.encoder = state_backup
        api_module.state.load_error = None


# ---------------------------------------------------------------------------
# C5 — a API reutiliza add_derived_features e o encoder salvo (sem codificação própria)
# ---------------------------------------------------------------------------


def test_api_reuses_pipeline_functions() -> None:
    """Contrato de reuso: api.py importa add_derived_features de churn.features
    e usa encoder.transform — nunca recria o OrdinalEncoder."""
    source = Path(api_module.__file__).read_text(encoding="utf-8")
    assert "from churn.features import add_derived_features" in source
    assert "encoder.transform(" in source
    assert "OrdinalEncoder" not in source
    assert "CategoricalEncoder(" not in source


def test_encode_matches_training_preprocessing(client: TestClient) -> None:
    """A entrada codificada tem as 20 features do modelo, na ordem certa,
    incluindo a derivada gasto_por_mes = TotalCharges / (tenure + 1)."""
    encoded = api_module.state.encode(api_module.ChurnRequest(**VALID_PAYLOAD))
    model = api_module.state.model
    assert list(encoded.columns) == list(model.feature_names_in_)
    assert encoded["gasto_por_mes"].iloc[0] == pytest.approx(1069.2 / 13)
    # numéricas intactas e categóricas codificadas como números
    assert encoded["MonthlyCharges"].iloc[0] == pytest.approx(89.1)
    assert encoded["Contract"].iloc[0] != "Month-to-month"


def test_env_vars_are_honored_on_reload(monkeypatch) -> None:
    """CHURN_DECISION_THRESHOLD é lido no import: recarregar o módulo com o env
    setado deve mudar o threshold exposto pela API (usado no Render/compose)."""
    monkeypatch.setenv("CHURN_DECISION_THRESHOLD", "0.9")
    try:
        reloaded = importlib.reload(api_module)
        with TestClient(reloaded.app) as c:
            info = c.get("/model-info").json()
            assert info["decision_threshold"] == pytest.approx(0.9)
            body = c.post("/predict", json=VALID_PAYLOAD).json()
            assert body["decision_threshold"] == pytest.approx(0.9)
            assert body["churn_prediction"] == (body["churn_probability"] >= 0.9)
    finally:
        monkeypatch.delenv("CHURN_DECISION_THRESHOLD", raising=False)
        importlib.reload(api_module)
