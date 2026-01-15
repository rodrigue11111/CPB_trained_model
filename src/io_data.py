"""Utilitaires de chargement et validation des donnees.

Ici on se contente de lire les fichiers et d'alerter sur les incoherences.
La normalisation fine est faite dans src.schema.
"""

from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

from pandas.api.types import is_numeric_dtype

from .config import TARGET_SLUMP, TARGET_UCS

LOGGER = logging.getLogger(__name__)


def _read_excel(path: Path, sheet_name: str | None) -> pd.DataFrame:
    """Lit un fichier Excel et renvoie un DataFrame pandas."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier Excel introuvable : {path}")

    if sheet_name is None:
        with pd.ExcelFile(path, engine="openpyxl") as xls:
            if not xls.sheet_names:
                raise ValueError(f"Aucune feuille trouvee dans {path}")
            sheet_name = xls.sheet_names[0]

    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except ValueError as exc:
        raise ValueError(
            f"Feuille '{sheet_name}' introuvable dans {path}"
        ) from exc

    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Valide les colonnes numeriques et quelques contraintes simples."""
    for col in df.columns:
        if col in {"Binder", "Tailings"}:
            if is_numeric_dtype(df[col]):
                LOGGER.warning(
                    "Colonne categorielle semble numerique : %s", col
                )
            continue
        if is_numeric_dtype(df[col]):
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        invalid = df[col].notna() & coerced.isna()
        if invalid.any():
            LOGGER.warning(
                "Colonne numerique avec valeurs non numeriques : %s (%d)",
                col,
                int(invalid.sum()),
            )

    if "E/C" in df.columns:
        # Exemple de contrainte simple: E/C doit etre > 0.
        ec_values = pd.to_numeric(df["E/C"], errors="coerce")
        invalid_ec = ec_values.notna() & (ec_values <= 0)
        if invalid_ec.any():
            raise ValueError("Valeurs E/C invalides (doivent etre > 0).")

    required_cols = [
        col
        for col in [TARGET_SLUMP, TARGET_UCS, "E/C", "Cw_f", "Ad %"]
        if col in df.columns
    ]
    if required_cols:
        required_frame = df[required_cols]
        nan_ratio = float(required_frame.isna().mean().mean())
        if nan_ratio > 0.2:
            LOGGER.warning(
                "Taux de valeurs manquantes eleve : %.1f%%", nan_ratio * 100
            )


def read_excel_file(
    path: str | Path,
    sheet_name: str | None,
) -> pd.DataFrame:
    """Facade simple autour de _read_excel."""
    return _read_excel(Path(path), sheet_name)


def load_and_combine(
    l01_path: str | Path,
    ww_path: str | Path,
    sheet_l01: str | None,
    sheet_ww: str | None,
) -> pd.DataFrame:
    """Charge L01 + WW, ajoute Tailings, et concatene."""
    l01_df = _read_excel(Path(l01_path), sheet_l01)
    l01_df["Tailings"] = "L01"

    ww_df = _read_excel(Path(ww_path), sheet_ww)
    ww_df["Tailings"] = "WW"

    df = pd.concat([l01_df, ww_df], ignore_index=True, sort=False)
    validate_dataframe(df)

    return df


def load_dataset(
    l01_path: str | Path,
    ww_path: str | Path,
    sheet_l01: str | None,
    sheet_ww: str | None,
) -> pd.DataFrame:
    """Alias historique pour load_and_combine."""
    return load_and_combine(l01_path, ww_path, sheet_l01, sheet_ww)
