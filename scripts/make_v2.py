"""Cria a v2 do dado: +100 linhas VÁLIDAS (passam no ChurnSchema), sem tocar nas existentes.

Uso:  uv run python scripts/make_v2.py
As novas linhas são reamostras de clientes existentes com customerID inédito
(formato NNNN-XXXX) — semente fixa, então o resultado é reprodutível.
"""
import numpy as np
import pandas as pd

CSV = "data/churn.csv"
N_NOVAS = 100

df = pd.read_csv(CSV, dtype=str, keep_default_na=False)  # dtype=str: preserva o arquivo como está
if len(df) != 7043:
    raise SystemExit(f"Esperava a v1 (7.043 clientes), encontrei {len(df)}. Rode o script só uma vez.")

rng = np.random.default_rng(42)
novas = df.sample(n=N_NOVAS, random_state=42).reset_index(drop=True)
existentes = set(df["customerID"])
ids = []
while len(ids) < N_NOVAS:
    cand = f"{rng.integers(0, 10000):04d}-" + "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 4))
    if cand not in existentes:
        existentes.add(cand)
        ids.append(cand)
novas["customerID"] = ids

pd.concat([df, novas], ignore_index=True).to_csv(CSV, index=False)
print(f"{CSV}: {len(df)} -> {len(df) + N_NOVAS} clientes (v2)")
