"""Exporta o pickle do treino (``artifacts/churn_model.pkl``) como joblib
comprimido para a API (``artifacts/churn_model.joblib``).

O pipeline DVC (``churn.model``) grava o bundle ``{"model", "encoder"}`` em
``.pkl``; a API carrega o MESMO bundle em ``.joblib`` (embutido na imagem
Docker e versionado no Git). Este script é o caminho documentado para
atualizar o artefato da API a partir de um modelo recém-treinado::

    uv run python scripts/export_joblib.py
    uv run python scripts/export_joblib.py --source artifacts/churn_model.pkl \
        --dest artifacts/churn_model.joblib --compress 9

Depois de exportar, rebuild da imagem (ou monte um diretório que contenha o
``churn_model.joblib`` — ver README, seção "Artefatos do modelo").
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import joblib


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Converte o churn_model.pkl do treino no churn_model.joblib da API."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/churn_model.pkl"),
        help="pickle gerado pelo treino (padrão: artifacts/churn_model.pkl)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("artifacts/churn_model.joblib"),
        help="joblib comprimido de saída (padrão: artifacts/churn_model.joblib)",
    )
    parser.add_argument(
        "--compress",
        type=int,
        default=9,
        help="nível de compressão do joblib, 0-9 (padrão: 9)",
    )
    args = parser.parse_args(argv)

    source: Path = args.source
    dest: Path = args.dest

    if not source.is_file():
        print(
            f"ERRO: origem não encontrada: {source}\n"
            "Gere o pickle primeiro com o pipeline de treino "
            "(uv run dvc repro  /  uv run python -m churn.model).",
            file=sys.stderr,
        )
        return 1

    with source.open("rb") as handle:
        bundle = pickle.load(handle)

    if not isinstance(bundle, dict) or "model" not in bundle or "encoder" not in bundle:
        print(
            f"ERRO: {source} não é o bundle esperado "
            "(dict com as chaves 'model' e 'encoder').",
            file=sys.stderr,
        )
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, dest, compress=args.compress)

    # Sanidade: o destino precisa carregar de volta com as mesmas chaves.
    reloaded = joblib.load(dest)
    assert set(reloaded) >= {"model", "encoder"}

    size_in = source.stat().st_size
    size_out = dest.stat().st_size
    print(f"{source} -> {dest} (compress={args.compress})")
    print(f"  origem:  {size_in:,} bytes ({size_in / 1024 / 1024:.2f} MiB)")
    print(f"  destino: {size_out:,} bytes ({size_out / 1024 / 1024:.2f} MiB)")
    print(f"  joblib carregável: OK ({type(reloaded['model']).__name__} + "
          f"{type(reloaded['encoder']).__name__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
