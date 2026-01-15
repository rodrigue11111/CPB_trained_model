"""Tests minimaux pour I/O et features."""

import numpy as np
import pandas as pd
import pytest

from src.config import REQUIRED_COLUMNS, TARGET_SLUMP, TARGET_UCS
from src.features import make_training_frames
from src.io_data import load_and_combine
from src.schema import standardize_required_columns


def _base_frame(rows: int) -> pd.DataFrame:
    """Cree un DataFrame minimal avec les colonnes requises."""
    data = {}
    for col in REQUIRED_COLUMNS:
        if col in ["Binder"]:
            data[col] = ["GUL"] * rows
        elif col in [TARGET_SLUMP, TARGET_UCS]:
            data[col] = np.linspace(10, 20, rows)
        else:
            data[col] = np.linspace(1, 2, rows)
    return pd.DataFrame(data)


def test_load_and_combine_adds_tailings(tmp_path):
    """Verifie que Tailings est ajoute lors du chargement."""
    l01_df = _base_frame(2)
    ww_df = _base_frame(3)

    l01_path = tmp_path / "l01.xlsx"
    ww_path = tmp_path / "ww.xlsx"
    l01_df.to_excel(l01_path, index=False)
    ww_df.to_excel(ww_path, index=False)

    combined = load_and_combine(l01_path, ww_path, None, None)

    assert "Tailings" in combined.columns
    assert set(combined["Tailings"]) == {"L01", "WW"}


def test_make_training_frames_different_sizes_when_ucs_missing():
    """Option A: UCS manquant => df_ucs plus petit."""
    df = _base_frame(4)
    df.loc[0, TARGET_UCS] = np.nan
    df.loc[1, TARGET_UCS] = np.nan

    df_slump, df_ucs = make_training_frames(df)

    assert len(df_slump) == 4
    assert len(df_ucs) == 2


def test_load_and_combine_checks_required_columns(tmp_path):
    """standardize_required_columns doit lever si une colonne manque."""
    l01_df = _base_frame(2)
    ww_df = _base_frame(2).drop(columns=["Ad %"])

    with pytest.raises(ValueError, match="Colonnes obligatoires manquantes"):
        standardize_required_columns(ww_df)
