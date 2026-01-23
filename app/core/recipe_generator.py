"""Generation de recettes candidates a partir des modeles entraines."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from src.features import infer_feature_columns

MUSCOVITE_RATIO_COL = "muscovite_ratio"
MUSCOVITE_ADDED_COLS = ("muscovite_added (%)", "muscovite_added")
MUSCOVITE_TOTAL_COLS = ("muscovite_total (%)", "muscovite_total")


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


def _select_series(
    df_ref: pd.DataFrame,
    col: str,
    tailings: str,
    binder: str,
) -> pd.Series:
    """Recupere une serie pour un couple (tailings, binder) avec fallback."""
    if col not in df_ref.columns:
        return pd.Series(dtype=float)

    subset = df_ref
    if "Tailings" in df_ref.columns:
        subset = subset[subset["Tailings"] == tailings]

    if "Binder" in df_ref.columns:
        subgroup = subset[subset["Binder"] == binder]
        # Si le sous-groupe est trop petit, on retombe sur le tailings.
        if len(subgroup) >= 10:
            subset = subgroup

    return pd.to_numeric(subset[col], errors="coerce").dropna()


def _get_min_max(
    df_ref: pd.DataFrame,
    col: str,
    tailings: str,
    binder: str,
    numeric_stats: dict | None = None,
) -> tuple[float, float]:
    """Recupere les bornes min/max observees pour une colonne."""
    series = _select_series(df_ref, col, tailings, binder)
    if not series.empty:
        return float(series.min()), float(series.max())

    stats = numeric_stats.get(col, {}) if numeric_stats else {}
    return float(stats.get("min", np.nan)), float(stats.get("max", np.nan))


def _sample_uniform(
    low: float,
    high: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if np.isnan(low) or np.isnan(high):
        raise ValueError(f"Borne manquante pour un tirage uniforme: {low}, {high}.")
    if low == high:
        return np.full(n, low)
    return rng.uniform(low, high, size=n)


def _sample_bootstrap(
    values: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if values.size == 0:
        raise ValueError("Aucune valeur disponible pour bootstrap.")
    if values.size == 1:
        return np.full(n, float(values[0]))
    return rng.choice(values, size=n, replace=True)


def _get_bootstrap_values(
    df_ref: pd.DataFrame,
    col: str,
    tailings: str,
    binder: str,
    sample_values: dict | None = None,
) -> np.ndarray:
    """Recupere des valeurs observees pour un bootstrap."""
    if sample_values and sample_values.get(col):
        return np.asarray(sample_values.get(col, []), dtype=float)
    series = _select_series(df_ref, col, tailings, binder)
    return series.to_numpy(dtype=float, copy=False)


def _resolve_muscovite_cols(columns: list[str]) -> tuple[str | None, str | None]:
    """Trouve les colonnes muscovite_added / muscovite_total si presentes."""
    added_col = next((c for c in MUSCOVITE_ADDED_COLS if c in columns), None)
    total_col = next((c for c in MUSCOVITE_TOTAL_COLS if c in columns), None)
    return added_col, total_col


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
    constraints: dict[str, dict[str, Any]] | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    numeric_stats: dict | None = None,
    sample_values: dict | None = None,
    allow_extrapolation: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Genere des recettes candidates selon des contraintes ou cibles.

    Returns:
        - DataFrame complet, trie avec score (le filtrage PASS est fait cote UI)
        - stats (pass_rate_pct, total, passed)
    """

    # Mode min = contraintes minimales, mode target = cibles avec tolerance.
    mode = "target" if (slump_target is not None or ucs_target is not None) else "min"
    if numeric_features is None or categorical_features is None:
        categorical_features, numeric_features = infer_feature_columns(
            df_ref, drop_constant=False
        )

    constraints = constraints or {}
    numeric_stats = numeric_stats or {}
    rng = np.random.default_rng(seed)

    missing_numeric = [col for col in numeric_features if col not in df_ref.columns]
    # muscovite_ratio peut etre derive, on le retire du check strict.
    missing_numeric = [
        col for col in missing_numeric if col != MUSCOVITE_RATIO_COL
    ]
    if missing_numeric:
        raise ValueError(
            "Colonnes numeriques manquantes pour la generation: "
            + ", ".join(missing_numeric)
        )

    unsupported_cat = [
        col
        for col in categorical_features
        if col not in {"Binder", "Tailings"}
    ]
    if unsupported_cat:
        raise ValueError(
            "Colonnes categorielles non gerees: " + ", ".join(unsupported_cat)
        )

    all_candidates = []
    total = 0
    passed = 0
    validation_report: dict[str, Any] = {
        "ood_features": [],
        "allow_extrapolation": allow_extrapolation,
    }

    for binder in binders:
        data: dict[str, np.ndarray] = {}
        for col in numeric_features:
            if col == MUSCOVITE_RATIO_COL and col not in constraints:
                # Calcule plus bas a partir de muscovite_added/total.
                continue

            constraint = constraints.get(col)
            if constraint:
                if constraint.get("mode") == "fixed":
                    value = float(constraint.get("value"))
                    train_min, train_max = _get_min_max(
                        df_ref, col, tailings, binder, numeric_stats
                    )
                    if not allow_extrapolation and (value < train_min or value > train_max):
                        raise ValueError(
                            f"Valeur hors domaine pour {col}: {value} "
                            f"(min={train_min:.3f}, max={train_max:.3f})."
                        )
                    if value < train_min or value > train_max:
                        validation_report["ood_features"].append(
                            {
                                "feature": col,
                                "value": value,
                                "min": train_min,
                                "max": train_max,
                            }
                        )
                    data[col] = np.full(n_samples, value)
                elif constraint.get("mode") == "range":
                    min_val = float(constraint.get("min"))
                    max_val = float(constraint.get("max"))
                    train_min, train_max = _get_min_max(
                        df_ref, col, tailings, binder, numeric_stats
                    )
                    if min_val > max_val:
                        raise ValueError(f"Plage invalide pour {col}: {min_val} > {max_val}.")
                    if not allow_extrapolation:
                        if min_val < train_min or max_val > train_max:
                            raise ValueError(
                                f"Plage hors domaine pour {col}: "
                                f"[{min_val:.3f}, {max_val:.3f}] "
                                f"(min={train_min:.3f}, max={train_max:.3f})."
                            )
                    else:
                        if min_val < train_min or max_val > train_max:
                            validation_report["ood_features"].append(
                                {
                                    "feature": col,
                                    "min": train_min,
                                    "max": train_max,
                                    "min_value": min_val,
                                    "max_value": max_val,
                                }
                            )
                    if search_mode == "bootstrap":
                        values = _get_bootstrap_values(
                            df_ref,
                            col,
                            tailings,
                            binder,
                            sample_values=sample_values,
                        )
                        if values.size:
                            values = values[(values >= min_val) & (values <= max_val)]
                        if values.size:
                            data[col] = _sample_bootstrap(values, n_samples, rng)
                        else:
                            data[col] = _sample_uniform(min_val, max_val, n_samples, rng)
                    else:
                        data[col] = _sample_uniform(min_val, max_val, n_samples, rng)
                else:
                    raise ValueError(f"Mode de contrainte inconnu pour {col}.")
                continue

            # Tirage standard si aucune contrainte utilisateur.
            if search_mode == "bootstrap":
                values = _get_bootstrap_values(
                    df_ref,
                    col,
                    tailings,
                    binder,
                    sample_values=sample_values,
                )
                if values.size:
                    data[col] = _sample_bootstrap(values, n_samples, rng)
                else:
                    low, high = _get_min_max(
                        df_ref, col, tailings, binder, numeric_stats
                    )
                    data[col] = _sample_uniform(low, high, n_samples, rng)
            else:
                low, high = _get_min_max(df_ref, col, tailings, binder, numeric_stats)
                data[col] = _sample_uniform(low, high, n_samples, rng)

        # Colonnes categorielles requises par le modele.
        if "Tailings" in categorical_features:
            data["Tailings"] = np.array([tailings] * n_samples, dtype=object)
        if "Binder" in categorical_features:
            data["Binder"] = np.array([binder] * n_samples, dtype=object)

        candidates_model = pd.DataFrame(data)

        # Ajout automatique de muscovite_ratio si requis et non contraint.
        if MUSCOVITE_RATIO_COL in numeric_features and MUSCOVITE_RATIO_COL not in constraints:
            if MUSCOVITE_RATIO_COL not in candidates_model.columns:
                added_col, total_col = _resolve_muscovite_cols(
                    candidates_model.columns.tolist()
                )
                if not added_col or not total_col:
                    raise ValueError(
                        "Impossible de calculer muscovite_ratio sans muscovite_added/total."
                    )
                total_vals = candidates_model[total_col].astype(float)
                added_vals = candidates_model[added_col].astype(float)
                ratio = np.where(total_vals > 0, added_vals / total_vals, 0.0)
                candidates_model[MUSCOVITE_RATIO_COL] = ratio

        # Verification finale: toutes les features necessaires doivent etre presentes.
        missing_features = [
            col for col in numeric_features + categorical_features if col not in candidates_model.columns
        ]
        if missing_features:
            raise ValueError(
                "Features manquantes apres generation: " + ", ".join(missing_features)
            )

        total += len(candidates_model)

        slump_pred = slump_model.predict(candidates_model)
        ucs_pred = ucs_model.predict(candidates_model)
        candidates = candidates_model.copy()
        candidates["Slump_pred"] = slump_pred
        candidates["UCS_pred"] = ucs_pred

        # Filtrage par contraintes min ou par tolerance autour d'une cible.
        if mode == "min":
            ok = pd.Series(True, index=candidates.index)
            if slump_min is not None:
                ok &= candidates["Slump_pred"] >= float(slump_min)
            if ucs_min is not None:
                ok &= candidates["UCS_pred"] >= float(ucs_min)
            mask = ok
        else:
            ok = pd.Series(True, index=candidates.index)
            if slump_target is not None and tol_slump is not None:
                ok &= (candidates["Slump_pred"] - float(slump_target)).abs() <= float(tol_slump)
            if ucs_target is not None and tol_ucs is not None:
                ok &= (candidates["UCS_pred"] - float(ucs_target)).abs() <= float(tol_ucs)
            mask = ok

        # IMPORTANT: colonne "pass" utilisee pour filtrer cote UI.
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
    stats = {
        "total": total,
        "passed": passed,
        "pass_rate_pct": pass_rate,
        "validation": validation_report,
    }
    return df_all, stats


def select_top_k_pass(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Retourne uniquement les Top K avec pass == True."""
    if df.empty:
        return df
    if "pass" not in df.columns:
        return df.head(top_k)
    df_pass = df[df["pass"] == True]
    return df_pass.head(top_k)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convertit un DataFrame en bytes Excel pour download Streamlit."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Top_Recipes")
    return buffer.getvalue()
