"""Regenera data/churn.csv a partir do churn.parquet (mesmo conteúdo, byte a byte).

Uso:  uv run python scripts/parquet_para_csv.py caminho/para/churn.parquet
Resultado esperado: 7.044 linhas (com cabeçalho) e md5 d390bd07b5514a2256a2396993b8b0e3.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

origem = Path(sys.argv[1] if len(sys.argv) > 1 else "churn.parquet")
destino = Path("data/churn.csv")
destino.parent.mkdir(exist_ok=True)
pd.read_parquet(origem).to_csv(destino, index=False)
md5 = hashlib.md5(destino.read_bytes()).hexdigest()
print(f"{destino} gerado — md5 {md5}")
print("OK: igual à v1 do professor" if md5 == "d390bd07b5514a2256a2396993b8b0e3"
      else "ATENÇÃO: md5 diferente da v1 do professor (d390bd07...)")
