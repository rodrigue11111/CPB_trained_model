# -*- coding: utf-8 -*-
"""Page Générateur de recettes Streamlit."""

from __future__ import annotations

import time
import pandas as pd
import streamlit as st

from app.core import model_registry
from app.core.data_profiles import load_profile
from app.core.recipe_generator import generate_recipes, to_excel_bytes
from app.ui.components import load_css, section_header


def main() -> None:
    st.title("Générateur de recettes")
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

        st.caption("Modèle utilisé")
        st.code(model_version)

    bundle = model_registry.load_bundle(model_version)
    if not bundle:
        st.error(
            "Modèles introuvables. Placez-les dans outputs/final_models ou "
            "définissez MODEL_DOWNLOAD_URL dans secrets."
        )
        return

    tailings = "WW" if dataset_label == "WW" else "L01"
    slump_model, ucs_model = model_registry.resolve_tailings_model(bundle, tailings)
    if not slump_model or not ucs_model:
        st.error("Modèles incomplets pour ce tailings.")
        return

    profile_key = "WW" if dataset_label == "WW" else "L01_OLD" if dataset_label == "L01 OLD" else "L01_NEW"
    profile = load_profile(profile_key)
    if not profile:
        st.error("Impossible de charger le dataset pour le sampling.")
        return

    cat_values = model_registry.get_categorical_values_for_tailings(bundle, tailings)
    binders = profile.categorical_values.get("Binder") or cat_values.get("Binder") or ["GUL", "Slag"]

    st.subheader("Objectif et contraintes")
    mode = st.radio(
        "Type d'objectif",
        ["Contraintes min", "Cible + tolérance"],
        horizontal=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.number_input("N candidats", value=15000, step=1000)
    with col2:
        top_k = st.number_input("Top K", value=50, step=10)
    with col3:
        if not profile.bootstrap_ready:
            st.caption("Mode bootstrap indisponible (donnees brutes non disponibles).")
            search_options = ["uniform"]
            default_index = 0
        else:
            search_options = ["uniform", "bootstrap"]
            default_index = 1
        search_mode = st.selectbox("Search mode", search_options, index=default_index)

    binder_choice = st.selectbox("Binder", ["Tous"] + binders)
    selected_binders = binders if binder_choice == "Tous" else [binder_choice]

    slump_min = ucs_min = None
    slump_target = ucs_target = None
    tol_slump = tol_ucs = None

    if mode == "Contraintes min":
        c1, c2 = st.columns(2)
        with c1:
            slump_min = st.number_input("Slump >= (mm)", value=70.0)
        with c2:
            ucs_min = st.number_input("UCS >= (kPa)", value=900.0)
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            slump_target = st.number_input("Slump cible (mm)", value=100.0)
        with c2:
            tol_slump = st.number_input("Tol Slump (+/-)", value=10.0)
        with c3:
            ucs_target = st.number_input("UCS cible (kPa)", value=1200.0)
        with c4:
            tol_ucs = st.number_input("Tol UCS (+/-)", value=100.0)

    run = st.button("Générer", type="primary")

    if not run:
        return

    start = time.time()
    progress = st.progress(0)
    with st.spinner("Génération des recettes..."):
        progress.progress(10)
        ranked_df, stats = generate_recipes(
            profile.df,
            tailings=tailings,
            binders=selected_binders,
            n_samples=int(n_samples),
            search_mode=search_mode,
            slump_min=slump_min,
            ucs_min=ucs_min,
            slump_target=slump_target,
            ucs_target=ucs_target,
            tol_slump=tol_slump,
            tol_ucs=tol_ucs,
            top_k=int(top_k),
            slump_model=slump_model,
            ucs_model=ucs_model,
        )
        progress.progress(100)

    elapsed = time.time() - start
    pass_rate = stats.get("pass_rate_pct", 0.0)

    section_header("Résultats")
    st.write(f"Pass rate: {pass_rate:.2f} %")
    st.write(f"Temps d'exécution: {elapsed:.2f} s")

    if not ranked_df.empty and "pass" in ranked_df.columns:
        df_pass = ranked_df[ranked_df["pass"] == True]
    else:
        df_pass = ranked_df
    df_top = df_pass.head(int(top_k)) if not df_pass.empty else pd.DataFrame()

    if df_top.empty:
        st.warning(
            "Aucune recette ne satisfait les contraintes. "
            "Augmentez N candidats ou relâchez les seuils."
        )
        with st.expander("Voir les meilleurs candidats même si non-pass"):
            st.dataframe(ranked_df.head(int(top_k)))
    else:
        st.dataframe(df_top)
        excel_bytes = to_excel_bytes(df_top)
        st.download_button(
            "Télécharger Excel",
            data=excel_bytes,
            file_name="Top_Recipes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
