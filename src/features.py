"""Pretraitement des features et utilitaires de pipeline.

Le but est de rester dynamique : on detecte les colonnes disponibles
et on construit les features en fonction du schema reel du dataset.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import TARGET_SLUMP, TARGET_UCS


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Force la conversion numerique pour une liste de colonnes."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def infer_feature_columns(
    df: pd.DataFrame,
    target_cols: Iterable[str] | None = None,
    drop_constant: bool = True,
) -> tuple[list[str], list[str]]:
    """Retourne (categorical_cols, numeric_cols) en fonction du df fourni.

    Args:
        df: DataFrame nettoye (colonnes numeriques deja coerces si possible).
        target_cols: colonnes cibles a exclure des features.
        drop_constant: supprime les colonnes numeriques constantes.

    Notes:
        - La selection est dynamique pour supporter OLD vs NEW sans forcer
          une liste de features fixe.
        - Tailings est utilise seulement si plusieurs valeurs existent.
    """
    if target_cols is None:
        target_cols = {TARGET_SLUMP, TARGET_UCS}
    else:
        target_cols = set(target_cols)

    categorical_cols: list[str] = []
    if "Binder" in df.columns:
        categorical_cols.append("Binder")
    if "Tailings" in df.columns:
        unique_tailings = df["Tailings"].nunique(dropna=True)
        if unique_tailings > 1:
            # Tailings devient feature seulement si au moins 2 valeurs.
            categorical_cols.append("Tailings")

    numeric_cols: list[str] = []
    for col in df.columns:
        if col in categorical_cols or col in target_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    filtered_numeric: list[str] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        if drop_constant and series.nunique() <= 1:
            # On supprime les colonnes constantes qui n'apportent pas d'info.
            continue
        filtered_numeric.append(col)

    return categorical_cols, filtered_numeric


def get_feature_columns(
    df: pd.DataFrame,
    target_cols: Iterable[str] | None = None,
    drop_constant: bool = True,
) -> list[str]:
    """Retourne la liste complete des features (cat + num)."""
    categorical_cols, numeric_cols = infer_feature_columns(
        df, target_cols, drop_constant=drop_constant
    )
    return categorical_cols + numeric_cols


def make_training_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Option A: separe les lignes valides par cible, sans drop global.

    Returns:
        df_slump: lignes avec Slump (mm) connu.
        df_ucs: lignes avec UCS28d (kPa) connu.

    IMPORTANT:
        Ne pas faire de dropna global, pour eviter la fuite de cible.
    """
    df_slump = df[df[TARGET_SLUMP].notna()].copy()
    df_ucs = df[df[TARGET_UCS].notna()].copy()
    return df_slump, df_ucs


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Construit un preprocesseur standard (imputation + OHE).

    Args:
        numeric_cols: features numeriques.
        categorical_cols: features categorielles.

    Returns:
        ColumnTransformer pret a etre integre dans un Pipeline sklearn.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Separe X/y a partir de listes de colonnes explicites."""
    feature_cols = list(categorical_cols) + list(numeric_cols)
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y
