"""Linhagem: de qual dado e de qual código veio cada run (Encontro 6).

Tudo é lido de forma tolerante: dentro do container não há ``.git`` (nem o
binário git), então devolvemos "unknown" em vez de quebrar o treino.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

UNKNOWN = "unknown"


def data_md5(pointer: Path = Path("data/churn.csv.dvc")) -> str:
    """md5 do dado versionado, lido do ponteiro DVC (não recalcula o CSV)."""
    try:
        content = yaml.safe_load(pointer.read_text())
        return str(content["outs"][0]["md5"])
    except (FileNotFoundError, KeyError, IndexError, TypeError):
        return UNKNOWN


def git_commit() -> str:
    """Hash curto do commit atual (ou "unknown" fora de um repositório git)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return UNKNOWN
    return result.stdout.strip() or UNKNOWN
