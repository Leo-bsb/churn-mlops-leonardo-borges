"""Integração do Bloco 2: tracking -> registry (com gate) -> load do champion.

Usa sqlite e artefatos em pasta temporária: não toca no seu mlflow.db real.
"""
import mlflow
import pandas as pd
import pytest

from churn.config import Settings
from churn.model import fit_and_log, prepare
from churn.registry import best_run, load_champion, promote_best


@pytest.fixture
def cfg(tmp_path, monkeypatch, raw_frame) -> Settings:
    monkeypatch.chdir(tmp_path)  # mlartifacts/ relativo cai aqui, não no repo
    # 8 cópias da mini-amostra, com customerID únicos (o schema exige unicidade)
    frame = pd.concat([raw_frame] * 8, ignore_index=True)
    frame["customerID"] = [f"C{i:03d}" for i in range(len(frame))]
    csv = tmp_path / "churn.csv"
    frame.to_csv(csv, index=False)
    return Settings(
        data_path=csv,
        artifact_path=tmp_path / "m.pkl",
        n_estimators=5,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )


def test_run_registra_params_metricas_tags_e_modelo(cfg):
    _, metrics, run_id = fit_and_log(cfg, prepare(cfg), run_name="teste")
    run = mlflow.get_run(run_id)
    assert run.data.params["n_estimators"] == "5"
    assert run.data.metrics["roc_auc"] == pytest.approx(metrics["roc_auc"])
    assert {"data_md5", "git_commit"} <= set(run.data.tags)
    # MLflow 3: o modelo é um "logged model" à parte — a prova é conseguir carregá-lo pelo run
    assert mlflow.pyfunc.load_model(f"runs:/{run_id}/model") is not None
    assert {a.path for a in mlflow.MlflowClient().list_artifacts(run_id)} == {"encoder"}


def test_promote_best_registra_champion_e_carrega_via_pyfunc(cfg):
    data = prepare(cfg)
    fit_and_log(cfg, data, run_name="teste")
    cfg_livre = cfg.model_copy(update={"min_roc_auc": 0.0})
    assert promote_best(cfg_livre) == 1
    assert promote_best(cfg_livre) == 1  # idempotente: mesmo run, sem versão nova
    previsoes = load_champion(cfg_livre).predict(data.x_test)
    assert len(previsoes) == len(data.x_test)


def test_gate_bloqueia_modelo_abaixo_do_limiar(cfg):
    fit_and_log(cfg, prepare(cfg), run_name="teste")
    with pytest.raises(ValueError, match="não promove"):
        promote_best(cfg.model_copy(update={"min_roc_auc": 1.01}))


def test_best_run_sem_experimentos_avisa(cfg):
    with pytest.raises(LookupError):
        best_run(cfg.model_copy(update={"experiment_name": "vazio"}))
