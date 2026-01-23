"""Tests basiques pour le generateur de recettes Streamlit."""

import numpy as np
import pandas as pd

from app.core.recipe_generator import generate_recipes, select_top_k_pass


class DummyModel:
    """Modele factice avec une API predict simple."""

    def predict(self, X):
        return np.zeros(len(X))


def _sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(20):
        rows.append(
            {
                "Tailings": "L01",
                "Binder": "GUL",
                "E/C": float(rng.uniform(0.2, 1.0)),
                "Ad %": float(rng.uniform(0.0, 5.0)),
                "muscovite_added (%)": float(rng.uniform(0.0, 10.0)),
                "muscovite_total (%)": float(rng.uniform(1.0, 20.0)),
            }
        )
    return pd.DataFrame(rows)


def test_constraints_fixed_and_range():
    df_ref = _sample_df()
    constraints = {
        "E/C": {"mode": "fixed", "value": 0.5},
        "Ad %": {"mode": "range", "min": 1.0, "max": 2.0},
    }
    df, _ = generate_recipes(
        df_ref,
        tailings="L01",
        binders=["GUL"],
        n_samples=10,
        search_mode="uniform",
        slump_min=None,
        ucs_min=None,
        slump_target=None,
        ucs_target=None,
        tol_slump=None,
        tol_ucs=None,
        top_k=10,
        slump_model=DummyModel(),
        ucs_model=DummyModel(),
        constraints=constraints,
        numeric_features=["E/C", "Ad %"],
        categorical_features=["Binder", "Tailings"],
    )

    assert (df["E/C"] == 0.5).all()
    assert (df["Ad %"] >= 1.0).all() and (df["Ad %"] <= 2.0).all()


def test_muscovite_ratio_created():
    df_ref = _sample_df()
    df, _ = generate_recipes(
        df_ref,
        tailings="L01",
        binders=["GUL"],
        n_samples=5,
        search_mode="uniform",
        slump_min=None,
        ucs_min=None,
        slump_target=None,
        ucs_target=None,
        tol_slump=None,
        tol_ucs=None,
        top_k=5,
        slump_model=DummyModel(),
        ucs_model=DummyModel(),
        numeric_features=["muscovite_added (%)", "muscovite_total (%)", "muscovite_ratio"],
        categorical_features=["Binder", "Tailings"],
    )

    ratio = df["muscovite_ratio"].to_numpy()
    added = df["muscovite_added (%)"].to_numpy()
    total = df["muscovite_total (%)"].to_numpy()
    expected = np.where(total > 0, added / total, 0.0)
    assert np.allclose(ratio, expected)


def test_select_top_k_pass_only():
    df = pd.DataFrame({"pass": [True, False, True], "val": [1, 2, 3]})
    top = select_top_k_pass(df, top_k=2)
    assert not top.empty
    assert top["pass"].all()


def test_range_out_of_domain_raises_when_extrapolation_off():
    df_ref = _sample_df()
    constraints = {
        "E/C": {"mode": "range", "min": -10.0, "max": 10.0},
    }
    try:
        generate_recipes(
            df_ref,
            tailings="L01",
            binders=["GUL"],
            n_samples=5,
            search_mode="uniform",
            slump_min=None,
            ucs_min=None,
            slump_target=None,
            ucs_target=None,
            tol_slump=None,
            tol_ucs=None,
            top_k=5,
            slump_model=DummyModel(),
            ucs_model=DummyModel(),
            constraints=constraints,
            numeric_features=["E/C"],
            categorical_features=["Binder", "Tailings"],
            allow_extrapolation=False,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_fixed_out_of_domain_raises_when_extrapolation_off():
    df_ref = _sample_df()
    constraints = {
        "E/C": {"mode": "fixed", "value": 999.0},
    }
    try:
        generate_recipes(
            df_ref,
            tailings="L01",
            binders=["GUL"],
            n_samples=5,
            search_mode="uniform",
            slump_min=None,
            ucs_min=None,
            slump_target=None,
            ucs_target=None,
            tol_slump=None,
            tol_ucs=None,
            top_k=5,
            slump_model=DummyModel(),
            ucs_model=DummyModel(),
            constraints=constraints,
            numeric_features=["E/C"],
            categorical_features=["Binder", "Tailings"],
            allow_extrapolation=False,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised
