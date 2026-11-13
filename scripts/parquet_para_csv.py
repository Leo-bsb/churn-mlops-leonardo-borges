"""Regenera data/churn.csv a partir do churn.parquet do curso.

Uso:  uv run python scripts/parquet_para_csv.py caminho/para/churn.parquet

O CSV versionado neste repositório tem 7.043 clientes (21 colunas) e md5
``3a29edd8d69a0f5428433a8b0de2d0be``. Se o parquet for o mesmo dataset, o
arquivo gerado sai **byte a byte** igual ao versionado — o script confere isso
e avisa se houver divergência.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

MD5_VERSIONADO = "3a29edd8d69a0f5428433a8b0de2d0be"

origem = Path(sys.argv[1] if len(sys.argv) > 1 else "churn.parquet")
destino = Path("data/churn.csv")

if not origem.is_file():
    raise SystemExit(f"ERRO: parquet não encontrado: {origem}")

anterior = destino.read_bytes() if destino.is_file() else None
destino.parent.mkdir(parents=True, exist_ok=True)
pd.read_parquet(origem).to_csv(destino, index=False)

novo = destino.read_bytes()
md5 = hashlib.md5(novo).hexdigest()
linhas = novo.count(b"\n")
print(f"{destino} gerado — md5 {md5} ({linhas:,} linhas, {len(novo):,} bytes)")

if anterior is None:
    print("OK: arquivo criado (não havia versão anterior para comparar).")
elif hashlib.md5(anterior).hexdigest() == md5:
    print("OK: idêntico ao data/churn.csv versionado.")
elif md5 == MD5_VERSIONADO:
    print("OK: igual ao CSV canônico deste repositório.")
else:
    print(
        "ATENÇÃO: o conteúdo difere do data/churn.csv versionado "
        f"(esperado {MD5_VERSIONADO}, obtido {md5})."
    )
