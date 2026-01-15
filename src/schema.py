"""Nettoyage minimal et normalisation du schema.

Ce module fait volontairement peu de modifications :
- on uniformise seulement les colonnes requises (targets + core process)
- on garde toutes les autres colonnes telles quelles
L'objectif est de supporter des schemas OLD et NEW sans forcer un alignement.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

REQUIRED_TARGETS = ["UCS28d (kPa)", "Slump (mm)"]
REQUIRED_CORE = ["Binder", "E/C", "Cw_f", "Ad %"]
REQUIRED_COLUMNS = REQUIRED_CORE + REQUIRED_TARGETS

_CATEGORICAL_COLUMNS = {"Binder", "Tailings"}


def _normalize_key(name: str) -> str:
    """Normalise un nom de colonne pour la comparaison (espace + minuscules)."""
    return "".join(str(name).split()).lower()


def _normalize_alnum(name: str) -> str:
    """Normalise un nom de colonne en retirant les symboles non alphanumeriques."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _strip_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les entetes (strip + conversion en str)."""
    df = df.copy()
    df.columns = [
        "" if col is None else str(col).strip() for col in df.columns
    ]
    return df


def _drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes totalement vides ou uniquement des chaines vides."""
    df = df.dropna(axis=1, how="all").copy()
    to_drop = []
    for col in df.columns:
        series = df[col]
        if series.isna().all():
            to_drop.append(col)
            continue
        if series.dtype == object:
            stripped = series.astype(str).str.strip()
            empty_mask = series.isna() | (stripped == "")
            if bool(empty_mask.all()):
                to_drop.append(col)
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def _should_convert(series: pd.Series, converted: pd.Series) -> bool:
    """Decide si une colonne doit etre convertie en numerique."""
    non_null = series.notna().sum()
    if non_null == 0:
        return True
    ratio = converted.notna().sum() / float(non_null)
    return ratio >= 0.5


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute muscovite_ratio si muscovite_added et muscovite_total existent."""
    df = df.copy()
    added_col = None
    total_col = None

    for col in df.columns:
        normalized = _normalize_alnum(col)
        if normalized == "muscoviteadded":
            added_col = col
        elif normalized == "muscovitetotal":
            total_col = col

    if added_col and total_col:
        # On garde les colonnes originales et on ajoute une colonne ratio.
        added = pd.to_numeric(df[added_col], errors="coerce")
        total = pd.to_numeric(df[total_col], errors="coerce")
        ratio = added / total
        ratio = ratio.where(total > 0)
        df["muscovite_ratio"] = ratio

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les colonnes et tente une conversion numerique prudente.

    Hypotheses:
        - Binder et Tailings sont categorielles.
        - Les autres colonnes peuvent etre numeriques ou mixtes.
    """
    df = _strip_headers(df)
    df = _drop_empty_columns(df)

    for col in df.columns:
        if col in _CATEGORICAL_COLUMNS:
            # Binder/Tailings doivent rester des chaines.
            df[col] = df[col].astype(str)
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if _should_convert(series, converted):
            df[col] = converted

    df = add_engineered_features(df)
    return df


def standardize_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme uniquement les colonnes requises vers les noms canoniques.

    IMPORTANT:
        - On ne renomme pas les colonnes non requises pour respecter le schema NEW.
        - Les colonnes requises sont: Binder, E/C, Cw_f, Ad %, Slump (mm),
          UCS28d (kPa).
    """
    df = df.copy()
    lookup: dict[str, list[str]] = {}
    for col in df.columns:
        lookup.setdefault(_normalize_key(col), []).append(col)

    def _find_column(names: Iterable[str]) -> str | None:
        for name in names:
            key = _normalize_key(name)
            if key in lookup:
                return lookup[key][0]
        return None

    aliases = {
        "UCS28d (kPa)": ["UCS28d (kPa)", "UCS28d(kPa)", "UCS28d\n(kPa)"],
        "Slump (mm)": ["Slump (mm)", "Slump(mm)", "Slump\n(mm)"],
        "Binder": ["Binder", "binder"],
        "E/C": ["E/C", "E/C "],
        "Cw_f": ["Cw_f", "Cw_f "],
        "Ad %": ["Ad %", "Ad%", "Ad  %"],
    }

    rename_map = {}
    for canonical, names in aliases.items():
        if canonical in df.columns:
            continue
        match = _find_column(names)
        if match:
            rename_map[match] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        available = ", ".join(df.columns)
        missing_str = ", ".join(missing)
        raise ValueError(
            "Colonnes obligatoires manquantes : "
            f"{missing_str} | Disponibles : {available}"
        )

    return df
