"""Contrato de dados (pandera) — Encontro 4.

O schema é código: versiona, revisa em PR e roda no CI. Falhou o contrato,
o pipeline para ANTES do treino (falha barulhenta e cedo, não silenciosa).
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class ChurnSchema(pa.DataFrameModel):
    """Schema do Telco Customer Churn (após ``coerce_total_charges``)."""

    customerID: Series[str] = pa.Field(unique=True)
    gender: Series[str] = pa.Field(isin=["Male", "Female"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=["Yes", "No"])
    Dependents: Series[str] = pa.Field(isin=["Yes", "No"])
    tenure: Series[int] = pa.Field(ge=0, le=72)
    PhoneService: Series[str] = pa.Field(isin=["Yes", "No"])
    InternetService: Series[str] = pa.Field(isin=["DSL", "Fiber optic", "No"])
    Contract: Series[str] = pa.Field(isin=["Month-to-month", "One year", "Two year"])
    PaperlessBilling: Series[str] = pa.Field(isin=["Yes", "No"])
    MonthlyCharges: Series[float] = pa.Field(ge=0, le=200)
    TotalCharges: Series[float] = pa.Field(ge=0)
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config:
        coerce = True
        strict = False  # tolera as demais colunas (começa frouxo, aperta com o tempo)
