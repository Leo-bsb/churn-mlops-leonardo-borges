"""Grid de experimentos rastreados (Encontro 6): 3 x 3 = 9 runs comparáveis.

Uso: ``uv run python -m churn.experiments``
Cada run também salva o modelo (necessário para registrar o vencedor depois);
espere algumas centenas de MB em ``mlartifacts/`` (ignorado pelo git).
"""
from __future__ import annotations

from churn.config import Settings, settings
from churn.model import fit_and_log, prepare

N_ESTIMATORS = (100, 200, 500)
MAX_DEPTHS = (None, 8, 16)


def run_grid(cfg: Settings = settings) -> list[tuple[str, dict[str, float]]]:
    data = prepare(cfg)  # o split e o encoder são iguais para todos: comparação justa
    results = []
    for n_estimators in N_ESTIMATORS:
        for max_depth in MAX_DEPTHS:
            run_cfg = cfg.model_copy(
                update={"n_estimators": n_estimators, "max_depth": max_depth}
            )
            name = f"rf-n{n_estimators}-d{max_depth}"
            _, metrics, _ = fit_and_log(run_cfg, data, run_name=name)
            results.append((name, metrics))
            print(f"{name:>14}  roc_auc={metrics['roc_auc']:.4f}  accuracy={metrics['accuracy']:.4f}")
    return results


if __name__ == "__main__":
    run_grid()
