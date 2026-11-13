"""Treino de ponta a ponta: artefatos, cartão do modelo e linhagem.

Usa uma mini-amostra replicada (rápido) e um mlflow.db temporário — nunca toca
no ``mlflow.db`` nem nos artefatos reais do repositório.
"""
from __future__ import annotations

import hashlib
import json
import pickle

import pandas as pd
import pytest

from churn.config import Settings
from churn.model import prepare, train


@pytest.fixture
def cfg(tmp_path, monkeypatch, raw_frame) -> Settings:
    monkeypatch.chdir(tmp_path)
    frame = pd.concat([raw_frame] * 8, ignore_index=True)
    frame["customerID"] = [f"C{i:03d}" for i in range(len(frame))]
    csv = tmp_path / "churn.csv"
    frame.to_csv(csv, index=False)
    return Settings(
        data_path=csv,
        artifact_path=tmp_path / "churn_model.pkl",
        metrics_path=tmp_path / "model_metrics.json",
        n_estimators=5,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )


def test_train_grava_pkl_e_cartao(cfg: Settings) -> None:
    """`python -m churn.model` precisa deixar os DOIS artefatos no disco."""
    train(cfg)

    assert cfg.artifact_path.is_file()
    with cfg.artifact_path.open("rb") as handle:
        bundle = pickle.load(handle)
    assert set(bundle) == {"model", "encoder"}

    assert cfg.metrics_path.is_file()
    card = json.loads(cfg.metrics_path.read_text(encoding="utf-8"))
    assert card["model_type"] == "RandomForestClassifier"
    assert card["n_estimators"] == 5
    assert card["max_depth"] == 8  # default que venceu o grid (regularização)
    assert card["n_features"] == 20
    assert card["classes"] == [0, 1]
    assert card["test_rows"] > 0
    assert 0.0 < card["positive_rate"] < 1.0


def test_cartao_tem_metricas_e_importancias_coerentes(cfg: Settings) -> None:
    train(cfg)
    card = json.loads(cfg.metrics_path.read_text(encoding="utf-8"))

    assert set(card["metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in card["metrics"].values())

    importancias = card["feature_importances"]
    assert len(importancias) == card["n_features"] == 20
    valores = [item["importance"] for item in importancias]
    assert valores == sorted(valores, reverse=True)  # já vem ordenado
    assert sum(valores) == pytest.approx(1.0, abs=1e-4)
    assert card["feature_names"][0] == "gender"  # ordem das colunas do treino


def test_cartao_registra_linhagem_do_dado(cfg: Settings) -> None:
    """Sem ponteiro DVC, o md5 vem do CSV: a linhagem nunca fica 'unknown'."""
    train(cfg)
    card = json.loads(cfg.metrics_path.read_text(encoding="utf-8"))
    esperado = hashlib.md5(cfg.data_path.read_bytes()).hexdigest()
    assert card["data_md5"] == esperado
    assert isinstance(card["git_commit"], str) and card["git_commit"]
    assert isinstance(card["trained_at"], str) and card["trained_at"].endswith("Z")


def test_prepare_usa_as_20_features_do_modelo(cfg: Settings) -> None:
    data = prepare(cfg)
    assert data.x_train.shape[1] == 20
    assert "gasto_por_mes" in data.x_train.columns
    assert list(data.x_train.columns) == list(data.x_test.columns)
