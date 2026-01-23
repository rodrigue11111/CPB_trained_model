# -*- coding: utf-8 -*-
"""Page Generateur de recettes Streamlit."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.core import model_registry
from app.core.data_profiles import (
    format_bounds,
    get_feature_profile,
    load_profile,
    ood_level,
)
from app.core.recipe_generator import generate_recipes, select_top_k_pass, to_excel_bytes
from app.ui.components import badge, load_css, section_header

LOT_FEATURES = [
    "P20 (\u00b5m)",
    "P80 (\u00b5m)",
    "Gs",
    "phyllosilicates (%)",
    "muscovite_total (%)",
    "muscovite_added (%)",
]

RECIPE_FEATURES = [
    "E/C",
    "Cw_f",
    "Ad %",
]


def _ensure_state(key: str, default: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = default


def _default_range(profile, df_ref: pd.DataFrame, col: str) -> tuple[float, float, float]:
    stats = profile.numeric_stats.get(col, {}) if profile else {}
    min_val = stats.get("min")
    max_val = stats.get("max")
    mean_val = stats.get("mean")

    if min_val is None or max_val is None or np.isnan(min_val) or np.isnan(max_val):
        if col in df_ref.columns:
            series = pd.to_numeric(df_ref[col], errors="coerce").dropna()
            if not series.empty:
                min_val = float(series.min())
                max_val = float(series.max())
                mean_val = float(series.mean())

    min_val = float(min_val) if min_val is not None and not np.isnan(min_val) else 0.0
    max_val = float(max_val) if max_val is not None and not np.isnan(max_val) else min_val
    mean_val = float(mean_val) if mean_val is not None and not np.isnan(mean_val) else min_val
    return min_val, max_val, mean_val


def _apply_preset(options: list[str], preset: list[str], key: str) -> None:
    selected = set(st.session_state.get(key, []))
    selected.update([col for col in preset if col in options])
    st.session_state[key] = sorted(selected)


def _badge_for_level(level: str) -> str:
    if level == "ok":
        return badge("OK", "ok")
    if level == "warn":
        return badge("Attention", "warn")
    if level == "out":
        return badge("Hors domaine", "fail")
    return badge("Inconnu", "warn")


def main() -> None:
    st.title("G\u00e9n\u00e9rateur de recettes")
    load_css("app/ui/styles.css")

    with st.sidebar:
        st.header("Configuration")
        dataset_label = st.selectbox(
            "Famille de r\u00e9sidus",
            ["WW", "L01 OLD", "L01 NEW"],
            index=2,
        )
        if dataset_label == "L01 NEW":
            model_version = "FINAL_new_best"
        else:
            model_version = "FINAL_old_best"

        st.caption("Mod\u00e8le utilis\u00e9")
        st.code(model_version)

    bundle = model_registry.load_bundle(model_version)
    if not bundle:
        st.error(
            "Mod\u00e8les introuvables. Placez-les dans outputs/final_models ou "
            "d\u00e9finissez MODEL_DOWNLOAD_URL dans secrets."
        )
        return

    tailings = "WW" if dataset_label == "WW" else "L01"
    slump_model, ucs_model = model_registry.resolve_tailings_model(bundle, tailings)
    if not slump_model or not ucs_model:
        st.error("Mod\u00e8les incomplets pour ce r\u00e9sidu.")
        return

    features = model_registry.get_features_for_tailings(bundle, tailings)
    model_numeric = features.get("numeric", [])
    categorical_features = features.get("categorical", [])
    if not model_numeric:
        st.error("Features introuvables dans metadata.json.")
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
        ["Contraintes min", "Cible + tol\u00e9rance"],
        horizontal=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.number_input("N candidats", value=15000, step=1000)
    with col2:
        top_k = st.number_input("Top K", value=50, step=10)
    with col3:
        if not profile.bootstrap_ready:
            st.caption("Mode bootstrap indisponible (donn\u00e9es brutes non disponibles).")
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

    advanced = st.toggle(
        "Mode avanc\u00e9",
        value=False,
        help="Contraindre certaines variables num\u00e9riques du mod\u00e8le.",
    )

    allow_extrapolation = st.toggle(
        "Autoriser l'extrapolation (hors min/max)",
        value=False,
        help="A activer uniquement si vous assumez la prediction hors domaine.",
    )

    constraints: dict[str, dict[str, Any]] = {}
    blocked_fields: list[str] = []
    clamp_messages: list[str] = []

    if advanced:
        _ensure_state("constraint_features", [])
        with st.expander("Contraintes sur les variables num\u00e9riques", expanded=False):
            st.caption(
                "Choisissez les variables \u00e0 fixer ou borner. Les autres resteront "
                "tir\u00e9es automatiquement selon le mode de recherche."
            )

            show_all = st.toggle(
                "Afficher toutes les variables du dataset",
                value=False,
            )
            all_numeric = sorted(set(model_numeric) | set(profile.numeric_stats.keys()))
            selectable = all_numeric if show_all else model_numeric

            p1, p2 = st.columns(2)
            with p1:
                if st.button("Preset LOT (mat\u00e9riau)"):
                    _apply_preset(selectable, LOT_FEATURES, "constraint_features")
            with p2:
                if st.button("Preset RECETTE (proc\u00e9d\u00e9)"):
                    _apply_preset(selectable, RECIPE_FEATURES, "constraint_features")

            selected_features = st.multiselect(
                "Variables \u00e0 contraindre",
                options=selectable,
                key="constraint_features",
                format_func=lambda c: (
                    f"{c} (n'influence pas la pr\u00e9diction)" if c not in model_numeric else c
                ),
            )

            for col in selected_features:
                min_val, max_val, mean_val = _default_range(profile, profile.df, col)
                profile_stats = get_feature_profile(profile, col)
                bounds_text = format_bounds(profile_stats)

                st.markdown(f"**{col}**")
                mode_choice = st.radio(
                    f"Mode {col}",
                    ["Fixe", "Plage"],
                    horizontal=True,
                    key=f"constraint_mode_{col}",
                )
                if mode_choice == "Fixe":
                    value = st.number_input(
                        f"Valeur {col}",
                        value=float(mean_val),
                        key=f"constraint_value_{col}",
                    )
                    constraints[col] = {"mode": "fixed", "value": value}

                    status = ood_level(value, profile_stats)
                    st.markdown(
                        f"{_badge_for_level(status['level'])} " +
                        (f"Training: {bounds_text}" if bounds_text else "Profil insuffisant"),
                        unsafe_allow_html=True,
                    )
                    if status["level"] == "out" and not allow_extrapolation:
                        blocked_fields.append(col)
                        st.error(
                            "Valeur hors domaine d'entra\u00eenement : le mod\u00e8le extrapole."
                        )
                else:
                    cmin, cmax = st.columns(2)
                    with cmin:
                        min_in = st.number_input(
                            f"Min {col}",
                            value=float(min_val),
                            key=f"constraint_min_{col}",
                        )
                    with cmax:
                        max_in = st.number_input(
                            f"Max {col}",
                            value=float(max_val),
                            key=f"constraint_max_{col}",
                        )

                    min_adj, max_adj = min_in, max_in
                    min_profile = profile_stats.get("min")
                    max_profile = profile_stats.get("max")
                    if not allow_extrapolation and min_profile is not None and max_profile is not None:
                        min_adj = max(min_in, float(min_profile))
                        max_adj = min(max_in, float(max_profile))
                        if min_adj != min_in or max_adj != max_in:
                            clamp_messages.append(
                                f"{col}: plage ajust\u00e9e a [{min_adj:.3f}, {max_adj:.3f}]"
                            )

                    constraints[col] = {"mode": "range", "min": min_adj, "max": max_adj}

                    status_min = ood_level(min_in, profile_stats)
                    status_max = ood_level(max_in, profile_stats)
                    levels = {status_min["level"], status_max["level"]}
                    if "out" in levels:
                        level = "out"
                    elif "warn" in levels:
                        level = "warn"
                    elif "unknown" in levels:
                        level = "unknown"
                    else:
                        level = "ok"

                    st.markdown(
                        f"{_badge_for_level(level)} " +
                        (f"Training: {bounds_text}" if bounds_text else "Profil insuffisant"),
                        unsafe_allow_html=True,
                    )
                    if level == "out" and not allow_extrapolation:
                        st.warning(
                            "Plage hors domaine : ajust\u00e9e automatiquement au domaine d'entra\u00eenement."
                        )

            if clamp_messages:
                st.warning("Plages ajust\u00e9es : " + " | ".join(clamp_messages))

            if "muscovite_ratio" in model_numeric and "muscovite_ratio" not in constraints:
                st.caption("muscovite_ratio est calcul\u00e9 automatiquement si possible.")

    if blocked_fields and not allow_extrapolation:
        st.warning(
            "Certaines valeurs fixes sont hors domaine. Corrigez-les ou activez l'extrapolation."
        )

    run = st.button("G\u00e9n\u00e9rer", type="primary")

    if not run:
        return

    if blocked_fields and not allow_extrapolation:
        st.error("G\u00e9n\u00e9ration bloqu\u00e9e : valeurs hors domaine d'entra\u00eenement.")
        return

    start = time.time()
    progress = st.progress(0)
    with st.spinner("G\u00e9n\u00e9ration des recettes..."):
        progress.progress(10)
        try:
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
                constraints=constraints,
                numeric_features=model_numeric,
                categorical_features=categorical_features,
                numeric_stats=profile.numeric_stats,
                sample_values=profile.sample_values,
                allow_extrapolation=allow_extrapolation,
            )
        except Exception as exc:
            st.error(f"Erreur pendant la g\u00e9n\u00e9ration: {exc}")
            return
        finally:
            progress.progress(100)

    elapsed = time.time() - start
    pass_rate = stats.get("pass_rate_pct", 0.0)

    section_header("R\u00e9sultats")
    st.write(f"Pass rate: {pass_rate:.2f} %")
    st.write(f"Temps d'ex\u00e9cution: {elapsed:.2f} s")

    validation = stats.get("validation", {})
    if validation:
        with st.expander("Qualit\u00e9 des entr\u00e9es", expanded=False):
            if validation.get("ood_features"):
                st.warning("Variables hors domaine d'entra\u00eenement :")
                for item in validation["ood_features"]:
                    st.write(f"- {item.get('feature')}")
            if validation.get("clamped_ranges"):
                st.info("Plages ajust\u00e9es au domaine d'entra\u00eenement :")
                for item in validation["clamped_ranges"]:
                    st.write(
                        f"- {item.get('feature')}: [{item.get('min_before'):.3f}, {item.get('max_before'):.3f}] -> "
                        f"[{item.get('min_after'):.3f}, {item.get('max_after'):.3f}]"
                    )
            if not validation.get("ood_features") and not validation.get("clamped_ranges"):
                st.write("Aucune alerte OOD.")

    show_non_pass = st.toggle("Montrer aussi les non-pass", value=False)

    df_top = select_top_k_pass(ranked_df, int(top_k))
    if df_top.empty:
        st.warning(
            "0 recette valide avec ces contraintes, "
            "\u00e9largis les plages ou baisse les seuils."
        )
        if show_non_pass and not ranked_df.empty:
            with st.expander("Voir les meilleurs candidats m\u00eame si non-pass"):
                st.dataframe(ranked_df.head(int(top_k)))
        return

    st.dataframe(df_top)
    excel_bytes = to_excel_bytes(df_top)
    st.download_button(
        "T\u00e9l\u00e9charger Excel",
        data=excel_bytes,
        file_name="Top_Recipes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if show_non_pass:
        df_non_pass = ranked_df[ranked_df.get("pass") == False]
        if not df_non_pass.empty:
            with st.expander("Voir les meilleurs non-pass"):
                st.dataframe(df_non_pass.head(int(top_k)))


if __name__ == "__main__":
    main()
