from pathlib import Path

import pandas as pd
import pytest

from churn.data import coerce_total_charges, drop_identifier, load_clean, validate


def test_coerce_total_charges_converte_branco_em_zero(raw_frame):
    out = coerce_total_charges(raw_frame)
    assert out["TotalCharges"].dtype.kind == "f"
    # a linha com tenure 0 tinha TotalCharges em branco -> 0.0
    assert out.loc[raw_frame["tenure"] == 0, "TotalCharges"].iloc[0] == 0.0


def test_coerce_e_pura(raw_frame):
    _ = coerce_total_charges(raw_frame)
    assert raw_frame["TotalCharges"].dtype == object  # original intacto


def test_validate_falha_sem_coluna(raw_frame):
    with pytest.raises(ValueError):
        validate(raw_frame.drop(columns=["Contract"]), target="Churn")


def test_drop_identifier(raw_frame):
    out = drop_identifier(raw_frame, "customerID")
    assert "customerID" not in out.columns


def test_dataset_real_atende_ao_contrato():
    """O CSV versionado precisa passar inteiro pelo contrato (não só a amostra).

    Se este teste falhar, dados e `ChurnSchema` divergiram — o treino real
    quebraria do mesmo jeito, só que mais tarde.
    """
    csv = Path(__file__).resolve().parents[1] / "data" / "churn.csv"
    if not csv.is_file():
        pytest.skip("data/churn.csv ausente (dataset opcional para esta suíte)")

    df = load_clean(csv, "customerID", "Churn")

    assert len(df) == 7043
    assert df.shape[1] == 20  # 21 colunas do CSV - customerID
    assert df.isna().sum().sum() == 0
    assert set(df["Churn"].unique()) == {"Yes", "No"}
    assert df["tenure"].between(0, 72).all()
    assert (df["TotalCharges"] >= 0).all()

