"""Testes do contrato de dados: o dado quebrado precisa ser pego pelo schema."""
import pandas as pd
import pandera.errors as pae
import pytest

from churn.data import coerce_total_charges
from churn.schema import ChurnSchema


@pytest.fixture
def clean_frame(raw_frame) -> pd.DataFrame:
    return coerce_total_charges(raw_frame)


def test_dado_valido_passa(clean_frame):
    assert len(ChurnSchema.validate(clean_frame, lazy=True)) == len(clean_frame)


def test_quebra_tenure_e_contract_reporta_todas_as_falhas(clean_frame):
    """Adultera de propósito (slide do Enc. 4): tenure=999 e Contract inédito."""
    ruim = clean_frame.copy()
    ruim.loc[0, "tenure"] = 999
    ruim.loc[1, "Contract"] = "Vitalício"
    with pytest.raises(pae.SchemaErrors) as exc:
        ChurnSchema.validate(ruim, lazy=True)
    falhas = exc.value.failure_cases
    assert {"tenure", "Contract"} <= set(falhas["column"])


def test_customerid_duplicado_falha(clean_frame):
    ruim = clean_frame.copy()
    ruim.loc[1, "customerID"] = ruim.loc[0, "customerID"]
    with pytest.raises(pae.SchemaErrors):
        ChurnSchema.validate(ruim, lazy=True)


def test_monthlycharges_negativo_falha(clean_frame):
    ruim = clean_frame.copy()
    ruim.loc[0, "MonthlyCharges"] = -5.0
    with pytest.raises(pae.SchemaErrors):
        ChurnSchema.validate(ruim, lazy=True)
