# -*- coding: utf-8 -*-
"""Page Debug Modèles Streamlit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core import model_registry
from app.core.data_profiles import load_profile
from app.core.predict import build_input_frame, predict_targets
from app.ui.components import load_css


def main() -> None:
    st.title("Debug modèles")
    load_css("app/ui/styles.css")

    with st.sidebar:
        st.header("Configuration")
        dataset_label = st.selectbox(
            "Famille de r\u00e9sidus",
            ["WW", "L01 OLD", "L01 NEW"],
            index=0,
        )
        if dataset_label == "L01 NEW":
            model_version = "FINAL_new_best"
        else:
            model_version = "FINAL_old_best"

    bundle = model_registry.load_bundle(model_version)
    if not bundle:
        st.error("Modèles introuvables.")
        return

    tailings = "WW" if dataset_label == "WW" else "L01"
    slump_model, ucs_model = model_registry.resolve_tailings_model(bundle, tailings)

    st.subheader("Metadata")
    st.json(bundle.metadata or {})

    st.subheader("Metrics")
    st.json(bundle.metrics or {})

    st.subheader("Best params")
    st.json(bundle.best_params or {})

    st.subheader("Features utilisées")
    st.json(model_registry.get_features_for_tailings(bundle, tailings))

    # Sanity check: prediction avec valeurs moyennes du profil.
    profile_key = "WW" if dataset_label == "WW" else "L01_OLD" if dataset_label == "L01 OLD" else "L01_NEW"
    profile = load_profile(profile_key)
    if profile and slump_model and ucs_model:
        features = model_registry.get_features_for_tailings(bundle, tailings)
        required_cols = features["categorical"] + features["numeric"]
        inputs = {}
        for col in features["categorical"]:
            options = profile.categorical_values.get(col, ["GUL", "Slag"])
            inputs[col] = options[0] if options else "GUL"
        for col in features["numeric"]:
            inputs[col] = profile.numeric_stats.get(col, {}).get("mean", 0.0)

        df_input = build_input_frame(inputs, required_cols)
        slump_pred, ucs_pred = predict_targets(slump_model, ucs_model, df_input)
        st.subheader("Sanity check prediction")
        st.write(f"Slump: {slump_pred:.2f} mm")
        st.write(f"UCS: {ucs_pred:.2f} kPa")


if __name__ == "__main__":
    main()
