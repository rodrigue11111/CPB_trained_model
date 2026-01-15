"""Optimisation des recettes par Monte-Carlo.

Le principe : generer des recettes candidates, predire Slump/UCS,
filtrer par seuils, puis trier les meilleures recettes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_BINDERS, OUTPUT_TAILINGS, TARGET_SLUMP, TARGET_UCS
from .features import coerce_numeric, infer_feature_columns


def sample_candidates(
    df_ref: pd.DataFrame,
    tailings: str,
    binder: str,
    n: int,
    search_mode: str = "uniform",
    numeric_cols: list[str] | None = None,
    seed: int = 42,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Genere un DataFrame de candidats pour un couple (tailings, binder)."""
    if n <= 0:
        raise ValueError("Le nombre d'echantillons doit etre > 0.")

    if numeric_cols is None:
        _, numeric_cols = infer_feature_columns(
            df_ref, target_cols=[TARGET_SLUMP, TARGET_UCS], drop_constant=False
        )
    df_ref = coerce_numeric(df_ref, numeric_cols)
    subgroup = df_ref[
        (df_ref["Tailings"] == tailings) & (df_ref["Binder"] == binder)
    ]
    if len(subgroup) < 10:
        # Trop peu de points: on elargit aux bornes du Tailings.
        subgroup = df_ref[df_ref["Tailings"] == tailings]

    if subgroup.empty:
        raise ValueError(
            f"Aucune donnee disponible pour Tailings={tailings}."
        )

    if rng is None:
        rng = np.random.default_rng(seed)
    data = {}
    mode = search_mode.lower()
    if mode not in {"uniform", "bootstrap"}:
        raise ValueError("search_mode doit etre 'uniform' ou 'bootstrap'.")

    for col in numeric_cols:
        series = subgroup[col].dropna()
        if series.empty:
            # Fallback: bornes par tailings ou global si necessaire.
            tail_subset = df_ref[df_ref["Tailings"] == tailings][col].dropna()
            series = tail_subset if not tail_subset.empty else df_ref[col].dropna()
        if series.empty:
            raise ValueError(
                f"Aucune donnee disponible pour inferer des bornes : {col}."
            )
        if mode == "bootstrap":
            # Tirage parmi les valeurs observees.
            if len(series) == 1:
                data[col] = np.full(n, float(series.iloc[0]))
            else:
                data[col] = rng.choice(series.to_numpy(), size=n, replace=True)
        else:
            # Tirage uniforme entre min et max observes.
            low, high = float(series.min()), float(series.max())
            if low == high:
                data[col] = np.full(n, low)
            else:
                data[col] = rng.uniform(low, high, size=n)

    data["Tailings"] = [tailings] * n
    data["Binder"] = [binder] * n
    return pd.DataFrame(data)


def optimize_recipes(
    pipe_slump,
    pipe_ucs,
    df_ref: pd.DataFrame,
    slump_min: float,
    ucs_min: float,
    n_samples: int,
    search_mode: str = "uniform",
    top_n: int = 25,
    export_all: bool = False,
    export_dir: str | Path = "outputs",
    fit_mode: str = "combined",
    models_by_tailings: dict | None = None,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Optimise les recettes et retourne (resultats, stats).

    Args:
        pipe_slump/pipe_ucs: pipelines sklearn si fit_mode=combined.
        df_ref: donnees de reference (apres nettoyage/standardisation).
        slump_min, ucs_min: seuils de filtrage.
        n_samples: nombre de candidats par groupe.
        search_mode: uniform ou bootstrap.
        top_n: nombre de recettes conservees par groupe.
        export_all: export brut des candidats avant filtrage.
        fit_mode: combined ou hybrid_by_tailings.
        models_by_tailings: dictionnaire de pipelines WW/L01 si hybride.

    Returns:
        results: dict { "L01_GUL": df, ... }
        summary: stats globales (pass_rate_pct, totals, etc.).
    """
    results: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict] = {}
    total_candidates = 0
    total_passed = 0
    rng = np.random.default_rng(seed)
    _, global_numeric_cols = infer_feature_columns(
        df_ref, target_cols=[TARGET_SLUMP, TARGET_UCS], drop_constant=False
    )

    for tailings in OUTPUT_TAILINGS:
        for binder in OUTPUT_BINDERS:
            key = f"{tailings}_{binder}"
            tail_df = df_ref[df_ref["Tailings"] == tailings]
            if fit_mode == "combined":
                # Mode global: meme espace de recherche pour tout le monde.
                numeric_cols = global_numeric_cols
            else:
                # Mode hybride: on se base sur les colonnes du tailings courant.
                _, numeric_cols = infer_feature_columns(
                    tail_df,
                    target_cols=[TARGET_SLUMP, TARGET_UCS],
                    drop_constant=False,
                )
            candidates = sample_candidates(
                df_ref,
                tailings,
                binder,
                n_samples,
                search_mode=search_mode,
                numeric_cols=numeric_cols,
                rng=rng,
            )

            if fit_mode in {"by_tailings", "hybrid_by_tailings"}:
                if not models_by_tailings or tailings not in models_by_tailings:
                    raise ValueError(
                        f"Modele manquant pour Tailings={tailings}."
                    )
                slump_model = models_by_tailings[tailings]["slump_pipe"]
                ucs_model = models_by_tailings[tailings]["ucs_pipe"]
            else:
                slump_model = pipe_slump
                ucs_model = pipe_ucs

            # Prediction des cibles pour chaque candidat.
            slump_pred = slump_model.predict(candidates)
            ucs_pred = ucs_model.predict(candidates)

            candidates = candidates.copy()
            candidates["Slump_pred"] = slump_pred
            candidates["UCS_pred"] = ucs_pred

            if export_all:
                # Export brut de tous les candidats, avant filtrage.
                export_dir = Path(export_dir)
                export_dir.mkdir(parents=True, exist_ok=True)
                export_path = (
                    export_dir / f"All_Candidates_{tailings}_{binder}.csv"
                )
                candidates.to_csv(export_path, index=False)

            filtered = candidates[
                (candidates["Slump_pred"] >= slump_min)
                & (candidates["UCS_pred"] >= ucs_min)
            ]

            pass_count = len(filtered)
            # Classement: UCS d'abord, puis minimiser E/C et Ad %.
            filtered = filtered.sort_values(
                by=["UCS_pred", "E/C", "Ad %"],
                ascending=[False, True, True],
            )

            filtered = filtered.head(top_n)
            base_cols = [
                col
                for col in ["Tailings", "Binder"]
                if col in candidates.columns
            ]
            ordered_cols = base_cols + numeric_cols + [
                "Slump_pred",
                "UCS_pred",
            ]
            results[key] = filtered[ordered_cols]
            stats[key] = {
                "total": int(len(candidates)),
                "passed": int(pass_count),
            }
            total_candidates += len(candidates)
            total_passed += pass_count

    pass_rate_pct = 0.0
    if total_candidates > 0:
        pass_rate_pct = (total_passed / total_candidates) * 100

    summary = {
        "total_candidates": int(total_candidates),
        "total_passed": int(total_passed),
        "pass_rate_pct": float(pass_rate_pct),
        "per_group": stats,
    }

    return results, summary


def export_top_recipes(
    recipes_by_group: dict[str, pd.DataFrame], out_path: str | Path
) -> None:
    """Ecrit l'Excel final avec 4 onglets fixes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name in ["L01_GUL", "L01_Slag", "WW_GUL", "WW_Slag"]:
            df = recipes_by_group.get(sheet_name, pd.DataFrame())
            df.to_excel(writer, sheet_name=sheet_name, index=False)
