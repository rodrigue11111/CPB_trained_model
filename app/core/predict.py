"""Prediction UCS/Slump a partir d'un modele charge."""

from __future__ import annotations

import re

import pandas as pd


def _normalize_key(name: str) -> str:
    """Normalise un nom de colonne pour comparer des variantes."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_key(values: dict, target: str) -> str | None:
    """Retourne la cle reelle d'un dictionnaire selon un nom normalise."""
    target_key = _normalize_key(target)
    for key in values:
        if _normalize_key(key) == target_key:
            return key
    return None


def _compute_muscovite_ratio(values: dict) -> dict:
    """Calcule muscovite_ratio si possible, sans ecraser une valeur existante.

    Cette version accepte des noms de colonnes avec symboles (ex: "(%)").
    """
    if "muscovite_ratio" in values:
        return values

    added_key = _find_key(values, "muscovite_added")
    total_key = _find_key(values, "muscovite_total")
    if not added_key or not total_key:
        return values

    try:
        total = float(values.get(total_key))
        added = float(values.get(added_key))
    except (TypeError, ValueError):
        return values

    if total > 0:
        values["muscovite_ratio"] = added / total
    return values


def build_input_frame(inputs: dict, required_cols: list[str]) -> pd.DataFrame:
    """Construit un DataFrame a 1 ligne avec les colonnes requises.

    Args:
        inputs: dictionnaire des valeurs saisies.
        required_cols: liste des colonnes attendues par le pipeline.

    Returns:
        DataFrame avec exactement les colonnes requises (ordre conserve).
    """

    values = dict(inputs)
    values = _compute_muscovite_ratio(values)

    data = {}
    for col in required_cols:
        data[col] = [values.get(col)]
    return pd.DataFrame(data)


def validate_inputs(inputs: dict, required_cols: list[str]) -> list[str]:
    """Retourne la liste des colonnes manquantes dans les inputs."""
    missing = [col for col in required_cols if col not in inputs]
    return missing


def predict_targets(slump_model, ucs_model, df_input: pd.DataFrame) -> tuple[float, float]:
    """Predit Slump et UCS a partir des pipelines sklearn."""
    slump_pred = float(slump_model.predict(df_input)[0])
    ucs_pred = float(ucs_model.predict(df_input)[0])
    return slump_pred, ucs_pred
