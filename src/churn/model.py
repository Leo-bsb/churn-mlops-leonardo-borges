"""Treino do modelo de churn — orquestra o pipeline de ponta a ponta.

Entrada: ``python -m churn.model`` (ou, no container, ``uv run python -m churn.model``).

Desde o Encontro 6 cada treino é um *run* do MLflow (params, métricas, tags de
linhagem e o modelo empacotado com signature). O pickle em ``artifacts/`` continua
sendo gerado porque o stage ``treino`` do ``dvc.yaml`` o declara como saída.
"""
from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_clean
from churn.evaluate import evaluate
from churn.features import (
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)
from churn.lineage import data_md5, git_commit


@dataclass(frozen=True)
class Prepared:
    """Dados prontos para treinar: splits JÁ codificados + o encoder ajustado no treino."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    encoder: CategoricalEncoder


def build_model(cfg: Settings) -> RandomForestClassifier:
    """Instancia o classificador com semente fixa (reprodutível)."""
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
    )


def save_artifact(model: RandomForestClassifier, encoder: CategoricalEncoder, path: Path) -> None:
    """Persiste modelo + encoder juntos (a inferência precisa do encoder ajustado)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": model, "encoder": encoder}, handle)


def prepare(cfg: Settings = settings) -> Prepared:
    """dados -> features -> split -> encode (o encoder aprende só no treino)."""
    frame = add_derived_features(load_clean(cfg.data_path, cfg.id_column, cfg.target))
    features, labels = split_features_target(frame, cfg.target, cfg.positive_label)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )
    encoder = CategoricalEncoder().fit(x_train)
    return Prepared(
        encoder.transform(x_train), encoder.transform(x_test), y_train, y_test, encoder
    )


def configure_tracking(cfg: Settings) -> None:
    """Fixa o backend NO CÓDIGO: sem isso, cada terminal grava num lugar e os runs 'somem'."""
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


def _log_encoder(encoder: CategoricalEncoder) -> None:
    """O encoder é parte do contrato de inferência: vai junto como artefato do run."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "encoder.pkl"
        with target.open("wb") as handle:
            pickle.dump(encoder, handle)
        mlflow.log_artifact(str(target), artifact_path="encoder")


def fit_and_log(
    cfg: Settings, data: Prepared, run_name: str
) -> tuple[RandomForestClassifier, dict[str, float], str]:
    """Treina UM modelo e registra tudo num run. Devolve (modelo, métricas, run_id)."""
    configure_tracking(cfg)
    with mlflow.start_run(run_name=run_name) as run:
        pointer = cfg.data_path.with_name(cfg.data_path.name + ".dvc")
        mlflow.set_tags({"data_md5": data_md5(pointer), "git_commit": git_commit()})
        mlflow.log_params(cfg.model_dump(mode="json"))  # Pydantic -> params

        model = build_model(cfg).fit(data.x_train, data.y_train)
        metrics = evaluate(model, data.x_test, data.y_test)
        mlflow.log_metrics(metrics)

        signature = infer_signature(data.x_train, model.predict(data.x_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=data.x_train.head(3),
            serialization_format="cloudpickle",
        )
        _log_encoder(data.encoder)
        return model, metrics, run.info.run_id


def train(cfg: Settings = settings) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Executa o pipeline: dados -> features -> treino rastreado -> avaliação -> pickle."""
    data = prepare(cfg)
    model, metrics, _ = fit_and_log(cfg, data, run_name="rf")
    save_artifact(model, data.encoder, cfg.artifact_path)
    return model, metrics


def main() -> None:
    _, metrics = train()
    print("Modelo treinado. Métricas no teste:")
    for name, value in metrics.items():
        print(f"  {name:>10}: {value:.4f}")


if __name__ == "__main__":
    main()
