"""Linhagem: de qual dado e de qual código veio cada run (Encontro 6).

Tudo é lido de forma tolerante: dentro do container não há ``.git`` (nem o
binário git), então devolvemos "unknown" em vez de quebrar o treino.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

UNKNOWN = "unknown"
DEFAULT_POINTER = Path("data/churn.csv.dvc")
DEFAULT_DATA = Path("data/churn.csv")


def _md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """md5 de um arquivo, lido em blocos (não carrega o CSV inteiro na memória)."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def data_md5(
    pointer: Path = DEFAULT_POINTER, fallback: Path = DEFAULT_DATA
) -> str:
    """md5 do dado de treino, com dois caminhos possíveis.

    1. Se o ponteiro DVC existir, usa o md5 registrado nele — é o hash canônico
       do DVC e não exige recalcular o arquivo.
    2. Caso contrário, calcula o md5 do CSV em disco. O repositório mantém o
       CSV versionado no Git (ver README, "Dados e versionamento"), então a
       linhagem continua verdadeira mesmo sem um remote DVC configurado.
    3. Sem ponteiro e sem CSV, devolve ``unknown`` em vez de quebrar o treino.
    """
    try:
        content = yaml.safe_load(pointer.read_text(encoding="utf-8"))
        return str(content["outs"][0]["md5"])
    except (FileNotFoundError, KeyError, IndexError, TypeError, AttributeError, yaml.YAMLError):
        pass

    try:
        return _md5_of_file(fallback)
    except OSError:
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
