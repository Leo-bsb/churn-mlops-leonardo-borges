"""Treino do modelo de churn — orquestra o pipeline de ponta a ponta.

Entrada: ``python -m churn.model`` (ou, no container, ``uv run python -m churn.model``).

Desde o Encontro 6 cada treino é um *run* do MLflow (params, métricas, tags de
linhagem e o modelo empacotado com signature). Além do run, o treino grava dois
artefatos locais:

- ``artifacts/churn_model.pkl``      — bundle ``{"model", "encoder"}`` (saída do DVC);
- ``artifacts/model_metrics.json``   — cartão do modelo lido pela API em ``/model-info``.

O ``.pkl`` é a saída oficial do stage ``treino`` do ``dvc.yaml``. O JSON é
versionado no Git (é pequeno e a imagem Docker da API precisa dele), por isso
não entra como ``outs`` do DVC: quem versiona é o Git, quem o gera é este módulo.
"""
from __future__ import annotations

import json
import pickle
import platform
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import sklearn
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

# Só o que influencia o modelo/split vira param no MLflow: caminhos e URI de
# tracking são ambiente, não hiperparâmetro (poluiriam a comparação de runs).
TRACKED_PARAMS = ("n_estimators", "max_depth", "random_state", "test_size")


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


def model_card(
    model: RandomForestClassifier,
    metrics: dict[str, float],
    *,
    n_test_rows: int,
    positive_rate: float,
    pointer: Path,
    data_path: Path,
) -> dict:
    """Monta o "cartão do modelo" (JSON serializável) consumido pela API.

    Separar o cartão do ``.pkl`` resolve um problema prático: a API mostra
    métricas e importâncias em ``/model-info`` sem precisar carregar o dataset
    nem reavaliar o modelo em produção.
    """
    names = list(getattr(model, "feature_names_in_", []))
    importances = sorted(
        (
            {"feature": name, "importance": round(float(weight), 6)}
            for name, weight in zip(names, getattr(model, "feature_importances_", []))
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return {
        "model_type": type(model).__name__,
        "n_estimators": int(getattr(model, "n_estimators", 0)),
        "max_depth": getattr(model, "max_depth", None),
        "n_features": int(model.n_features_in_),
        "feature_names": names,
        "classes": [int(c) for c in model.classes_],
        "test_rows": int(n_test_rows),
        "positive_rate": round(float(positive_rate), 4),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # Linhagem: qual dado e qual commit geraram este cartão.
        "data_md5": data_md5(pointer, data_path),
        "git_commit": git_commit(),
        "metrics": {name: float(value) for name, value in metrics.items()},
        "feature_importances": importances,
    }


def save_metrics(card: dict, path: Path) -> None:
    """Grava o cartão do modelo em JSON (UTF-8, indentado)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(card, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


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
        mlflow.set_tags(
            {"data_md5": data_md5(pointer, cfg.data_path), "git_commit": git_commit()}
        )
        mlflow.log_params({name: getattr(cfg, name) for name in TRACKED_PARAMS})

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
    """Pipeline completo: dados -> features -> treino rastreado -> avaliação -> artefatos."""
    data = prepare(cfg)
    model, metrics, _ = fit_and_log(cfg, data, run_name="rf")

    pointer = cfg.data_path.with_name(cfg.data_path.name + ".dvc")
    card = model_card(
        model,
        metrics,
        n_test_rows=len(data.y_test),
        positive_rate=float(data.y_test.mean()),
        pointer=pointer,
        data_path=cfg.data_path,
    )
    save_artifact(model, data.encoder, cfg.artifact_path)
    save_metrics(card, cfg.metrics_path)
    return model, metrics


def main() -> None:
    cfg = settings
    _, metrics = train()
    print("Modelo treinado. Métricas no teste:")
    for name, value in metrics.items():
        print(f"  {name:>10}: {value:.4f}")
    print(f"\n  artefato: {cfg.artifact_path}")
    print(f"  cartão:   {cfg.metrics_path}")


if __name__ == "__main__":
    main()
