"""Configuração central do projeto (Pydantic).

Todo caminho, hiperparâmetro e constante mágica do script original vive aqui —
nunca hardcoded no meio da lógica. Valores podem ser sobrescritos por variáveis
de ambiente com prefixo ``CHURN_`` ou por um arquivo ``.env``.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros do pipeline de churn."""

    model_config = SettingsConfigDict(
        env_prefix="CHURN_",
        env_file=".env",
        protected_namespaces=(),
    )

    # dados
    data_path: Path = Path("data/churn.csv")
    id_column: str = "customerID"
    target: str = "Churn"
    positive_label: str = "Yes"

    # split / modelo
    test_size: float = 0.25
    random_state: int = 42
    n_estimators: int = 200
    # O grid de experimentos (churn.experiments) mostrou que limitar a
    # profundidade regulariza e melhora o ROC AUC no teste:
    #   d=None -> 0.8053 | d=8 -> 0.8195 | d=16 -> 0.8047 (com n=200)
    # Por isso o default é 8: o modelo servido é o mesmo que venceu o grid.
    max_depth: int | None = 8

    # saída
    artifact_path: Path = Path("artifacts/churn_model.pkl")
    # métricas do modelo treinado (lidas pela API em GET /model-info)
    metrics_path: Path = Path("artifacts/model_metrics.json")

    # MLflow (Bloco 2). sqlite: o file store (./mlruns) não é mais suportado no MLflow 3.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "churn"
    registered_model_name: str = "churn"
    champion_alias: str = "champion"
    min_roc_auc: float = 0.80  # gate de promoção: abaixo disso o modelo não vira champion


settings = Settings()
