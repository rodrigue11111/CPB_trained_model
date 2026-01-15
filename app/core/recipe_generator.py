"""Generation de recettes candidates a partir des modeles entraines."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.optimize import sample_candidates
from src.features import infer_feature_columns


def _score_candidates(df: pd.DataFrame, mode: str, targets: dict) -> pd.Series:
    """Calcule un score simple pour classer les candidats.

    Mode "min": favorise UCS eleve et penalise E/C et Ad %.
    Mode "target": penalise l'ecart aux cibles.
    """

    if mode == "target":
        ucs_target = targets.get("ucs_target")
        slump_target = targets.get("slump_target")
        score = df["UCS_pred"].copy()
        if ucs_target is not None:
            score -= (df["UCS_pred"] - float(ucs_target)).abs() * 0.05
        if slump_target is not None:
            score -= (df["Slump_pred"] - float(slump_target)).abs() * 0.1
        return score

    score = df["UCS_pred"].copy()
    if "E/C" in df.columns:
        score -= df["E/C"] * 0.02
    if "Ad %" in df.columns:
        score -= df["Ad %"] * 0.02
    return score


def generate_recipes(
    df_ref: pd.DataFrame,
    tailings: str,
    binders: list[str],
    n_samples: int,
    search_mode: str,
    slump_min: float | None,
    ucs_min: float | None,
    slump_target: float | None,
    ucs_target: float | None,
    tol_slump: float | None,
    tol_ucs: float | None,
    top_k: int,
    slump_model,
    ucs_model,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Genere des recettes candidates selon des contraintes ou cibles.

    Returns:
        - DataFrame complet, trie avec score (le filtrage PASS est fait cote UI)
        - stats (pass_rate_pct, total, passed)
    """

    # Mode min = contraintes minimales, mode target = cibles avec tolérance.
    mode = "target" if (slump_target is not None or ucs_target is not None) else "min"
    _, numeric_cols = infer_feature_columns(df_ref, drop_constant=False)

    all_candidates = []
    total = 0
    passed = 0

    for binder in binders:
        candidates = sample_candidates(
            df_ref,
            tailings,
            binder,
            n_samples,
            search_mode=search_mode,
            numeric_cols=numeric_cols,
            seed=seed,
        )
        total += len(candidates)

        slump_pred = slump_model.predict(candidates)
        ucs_pred = ucs_model.predict(candidates)
        candidates = candidates.copy()
        candidates["Slump_pred"] = slump_pred
        candidates["UCS_pred"] = ucs_pred

        # Filtrage par contraintes min ou par tolérance autour d'une cible.
        if mode == "min":
            ok = True
            if slump_min is not None:
                ok &= candidates["Slump_pred"] >= float(slump_min)
            if ucs_min is not None:
                ok &= candidates["UCS_pred"] >= float(ucs_min)
            mask = ok
        else:
            ok = True
            if slump_target is not None and tol_slump is not None:
                ok &= (candidates["Slump_pred"] - float(slump_target)).abs() <= float(tol_slump)
            if ucs_target is not None and tol_ucs is not None:
                ok &= (candidates["UCS_pred"] - float(ucs_target)).abs() <= float(tol_ucs)
            mask = ok

        # IMPORTANT: colonne "pass" utilisée pour filtrer côté UI.
        candidates["pass"] = mask
        passed += int(mask.sum())
        all_candidates.append(candidates)

    if not all_candidates:
        return pd.DataFrame(), {"total": 0, "passed": 0, "pass_rate_pct": 0.0}

    df_all = pd.concat(all_candidates, ignore_index=True)
    df_all["score"] = _score_candidates(
        df_all,
        mode=mode,
        targets={"ucs_target": ucs_target, "slump_target": slump_target},
    )

    # Tri par score puis par regles classiques si disponibles.
    sort_cols = ["score", "UCS_pred"]
    ascending = [False, False]
    if "E/C" in df_all.columns:
        sort_cols.append("E/C")
        ascending.append(True)
    if "Ad %" in df_all.columns:
        sort_cols.append("Ad %")
        ascending.append(True)

    df_all = df_all.sort_values(sort_cols, ascending=ascending)

    pass_rate = (passed / total * 100.0) if total else 0.0
    stats = {"total": total, "passed": passed, "pass_rate_pct": pass_rate}
    return df_all, stats


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convertit un DataFrame en bytes Excel pour download Streamlit."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Top_Recipes")
    return buffer.getvalue()
