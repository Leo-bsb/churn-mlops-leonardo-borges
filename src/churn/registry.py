"""Model Registry (Encontros 7 e 8): achar o melhor run, passar pelo gate e promover.

Uso: ``uv run python -m churn.registry``  -> registra o melhor run como ``churn@champion``.
"""
from __future__ import annotations

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from churn.config import Settings, settings
from churn.model import configure_tracking


def best_run(cfg: Settings = settings) -> pd.Series:
    """Melhor run do experimento por roc_auc (mesma resposta da UI, em código)."""
    configure_tracking(cfg)
    runs = mlflow.search_runs(
        experiment_names=[cfg.experiment_name],  # atenção: por NOME (o 1º arg posicional é ids)
        order_by=["metrics.roc_auc DESC"],
    )
    if runs.empty:
        raise LookupError(
            f"nenhum run em '{cfg.experiment_name}' — rode antes: uv run python -m churn.experiments"
        )
    return runs.iloc[0]


def promote_best(cfg: Settings = settings) -> int:
    """Registra o melhor run e move o alias champion — SE passar do gate. Devolve a versão."""
    best = best_run(cfg)
    roc_auc = float(best["metrics.roc_auc"])
    if roc_auc < cfg.min_roc_auc:  # o gate
        raise ValueError(
            f"roc_auc {roc_auc:.4f} abaixo do limiar {cfg.min_roc_auc} — não promove"
        )

    client = MlflowClient()
    try:  # idempotência: se o champion já é este run, não cria versão nova
        current = client.get_model_version_by_alias(
            cfg.registered_model_name, cfg.champion_alias
        )
        if current.run_id == best["run_id"]:
            return int(current.version)
    except MlflowException:
        pass  # ainda não existe champion

    version = mlflow.register_model(f"runs:/{best['run_id']}/model", cfg.registered_model_name)
    client.set_registered_model_alias(
        cfg.registered_model_name, cfg.champion_alias, version.version
    )
    return int(version.version)


def load_champion(cfg: Settings = settings):
    """Carrega o modelo de produção pelo contrato universal (pyfunc): só ``predict(df)``."""
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{cfg.registered_model_name}@{cfg.champion_alias}")


def main() -> None:
    cfg = settings
    best = best_run(cfg)
    version = promote_best(cfg)
    print(
        f"{cfg.registered_model_name} v{version} @{cfg.champion_alias} "
        f"(run {best['tags.mlflow.runName']}, roc_auc {best['metrics.roc_auc']:.4f})"
    )


if __name__ == "__main__":
    main()
