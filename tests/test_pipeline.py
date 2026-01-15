"""Tests de pipeline, optimisation et cross-validation."""

import numpy as np
import pandas as pd

from src.config import NUMERIC_BASE_COLS, TARGET_SLUMP, TARGET_UCS
from src.optimize import export_top_recipes, optimize_recipes
from src.features import (
    infer_feature_columns,
    make_training_frames,
    split_features_target,
)
from src.train import build_pipeline, cross_validate_report, fit_final_models


def _sample_df() -> pd.DataFrame:
    """Cree un petit dataset artificiel pour tests rapides."""
    rng = np.random.default_rng(0)
    rows = []
    for tailings in ["L01", "WW"]:
        for binder in ["GUL", "Slag"]:
            for _ in range(3):
                row = {
                    "Binder": binder,
                    "Tailings": tailings,
                    TARGET_SLUMP: float(rng.uniform(50, 150)),
                    TARGET_UCS: float(rng.uniform(500, 1500)),
                }
                for col in NUMERIC_BASE_COLS:
                    row[col] = float(rng.uniform(0.1, 10.0))
                rows.append(row)
    return pd.DataFrame(rows)


def test_cross_validate_report_returns_metrics():
    """Le rapport CV doit contenir overall + groupes."""
    df = _sample_df()
    cat_cols, num_cols = infer_feature_columns(
        df, target_cols=[TARGET_SLUMP, TARGET_UCS]
    )
    X, y = split_features_target(df, TARGET_SLUMP, cat_cols, num_cols)
    pipe = build_pipeline("gbr", numeric_cols=num_cols, categorical_cols=cat_cols)
    report = cross_validate_report(pipe, X, y, groups_df=df)

    assert "overall" in report
    assert "Tailings" in report
    assert "Binder" in report


def test_optimize_recipes_creates_output(tmp_path):
    """Optimisation: doit produire un fichier Excel valide."""
    df = _sample_df()
    df_slump, df_ucs = make_training_frames(df)
    slump_model, ucs_model = fit_final_models(df_slump, df_ucs)
    out_path = tmp_path / "recipes.xlsx"

    results, stats = optimize_recipes(
        slump_model,
        ucs_model,
        df,
        slump_min=60,
        ucs_min=800,
        n_samples=50,
    )
    export_top_recipes(results, out_path)

    assert out_path.exists()
    assert "L01_GUL" in results
    assert "WW_Slag" in results
    assert "pass_rate_pct" in stats
