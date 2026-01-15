"""Profil des donnees pour warnings et distributions de sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.schema import clean_dataframe, standardize_required_columns

DATASETS = {
    "WW": {
        "path": "data/WW-Optimisation.xlsx",
        "sheet": "Feuil3 (3)",
        "tailings": "WW",
    },
    "L01_OLD": {
        "path": "data/L01-Optimisation.xlsx",
        "sheet": "Feuil1 (2)",
        "tailings": "L01",
    },
    "L01_NEW": {
        "path": "data/L01-dataset.xlsx",
        "sheet": None,
        "tailings": "L01",
    },
}


@dataclass
class DataProfile:
    """Profil de donnees avec stats simples par colonne numerique."""

    df: pd.DataFrame
    numeric_stats: dict
    categorical_values: dict


def _first_sheet(path: str | Path) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if not wb.sheetnames:
            raise ValueError("Aucune feuille detectee.")
        return wb.sheetnames[0]
    finally:
        wb.close()


def _compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = pd.to_numeric(df[col], errors="coerce")
            stats[col] = {
                "min": float(series.min(skipna=True)) if series.notna().any() else np.nan,
                "max": float(series.max(skipna=True)) if series.notna().any() else np.nan,
                "mean": float(series.mean(skipna=True)) if series.notna().any() else np.nan,
                "std": float(series.std(skipna=True)) if series.notna().any() else np.nan,
            }
    return stats


@st.cache_data(show_spinner=False)
def load_profile(dataset_key: str) -> DataProfile | None:
    """Charge un profil de donnees pour un dataset.

    Notes:
        - Utilise le nettoyage minimal + normalisation des colonnes requises.
        - Ajoute/force la colonne Tailings pour coherer les predictions.
    """

    cfg = DATASETS.get(dataset_key)
    if not cfg:
        return None

    path = Path(cfg["path"])
    if not path.exists():
        return None

    sheet = cfg["sheet"]
    if sheet is None:
        sheet = _first_sheet(path)

    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        df["Tailings"] = cfg["tailings"]
        df = clean_dataframe(df)
        df = standardize_required_columns(df)
    except Exception:
        # Erreur silencieuse: l'UI affichera un message si le profil est None.
        return None

    numeric_stats = _compute_stats(df)
    categorical_values = {}
    for col in df.columns:
        if col in {"Binder", "Tailings"}:
            values = sorted({str(v) for v in df[col].dropna().unique()})
            categorical_values[col] = values

    return DataProfile(df=df, numeric_stats=numeric_stats, categorical_values=categorical_values)


def warn_out_of_profile(profile: DataProfile, inputs: dict) -> list[str]:
    """Retourne une liste d'avertissements si valeurs hors min/max ou z-score.

    Heuristique simple pour alerter l'utilisateur sans bloquer la prediction.
    """

    warnings = []
    for col, value in inputs.items():
        if col not in profile.numeric_stats:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        stats = profile.numeric_stats[col]
        if not np.isnan(stats.get("min", np.nan)) and val < stats["min"]:
            warnings.append(f"{col}: {val} < min observe ({stats['min']:.3f})")
        if not np.isnan(stats.get("max", np.nan)) and val > stats["max"]:
            warnings.append(f"{col}: {val} > max observe ({stats['max']:.3f})")
        std = stats.get("std", np.nan)
        mean = stats.get("mean", np.nan)
        if std and not np.isnan(std) and std > 0:
            z = abs((val - mean) / std)
            if z >= 3:
                warnings.append(f"{col}: z-score {z:.2f} (valeur atypique)")
    return warnings
