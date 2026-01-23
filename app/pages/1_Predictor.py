# -*- coding: utf-8 -*-
"""Page Predictor Streamlit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core import model_registry
from app.core.data_profiles import (
    format_bounds,
    get_feature_profile,
    load_profile,
    ood_level,
    warn_out_of_profile,
)
from app.core.predict import build_input_frame, predict_targets, validate_inputs
from app.ui.components import badge, load_css, metric_card, section_header, warning_list


def _unit_help(col: str) -> str:
    units = {
        "UCS28d (kPa)": "kPa",
        "Slump (mm)": "mm",
        "E/C": "rapport eau/ciment",
        "Cw_f": "eau libre",
        "Ad %": "%",
    }
    if col in units:
        return units[col]
    lowered = col.lower()
    if any(ch.isdigit() for ch in lowered) and ("p" in lowered or "d" in lowered):
        return "um (granulometrie)"
    return ""


def _classify_column(col: str) -> str:
    name = col.lower()
    if "muscovite" in name:
        return "Param\u00e8tres additionnels"
    if "p" in name and any(ch.isdigit() for ch in name):
        return "Granulom\u00e9trie"
    if "d" in name and any(ch.isdigit() for ch in name):
        return "Granulom\u00e9trie"
    return "Ingr\u00e9dients / Dosages"


def _badge_for_level(level: str) -> str:
    if level == "ok":
        return badge("OK", "ok")
    if level == "warn":
        return badge("Attention", "warn")
    if level == "out":
        return badge("Hors distribution", "fail")
    return badge("Inconnu", "warn")


def _safe_selectbox(label: str, options: list[str], key: str, default: str | None = None) -> str:
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state.pop(key)
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, key=key)


def main() -> None:
    st.title("Pr\u00e9dicteur CPB")
    load_css("app/ui/styles.css")

    with st.sidebar:
        st.header("Configuration")
        dataset_label = _safe_selectbox(
            "Famille de r\u00e9sidus",
            ["WW", "L01 OLD", "L01 NEW"],
            key="pred_dataset",
            default="L01 NEW",
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
            "definissez MODEL_DOWNLOAD_URL dans secrets."
        )
        return

    tailings = "WW" if dataset_label == "WW" else "L01"
    slump_model, ucs_model = model_registry.resolve_tailings_model(bundle, tailings)
    if not slump_model or not ucs_model:
        st.error("Mod\u00e8les incomplets pour ce r\u00e9sidu.")
        return

    features = model_registry.get_features_for_tailings(bundle, tailings)
    cat_values = model_registry.get_categorical_values_for_tailings(bundle, tailings)
    if not features["categorical"] and not features["numeric"]:
        st.error("Features non detectees dans metadata.json.")
        return

    profile_key = (
        "WW" if dataset_label == "WW" else "L01_OLD" if dataset_label == "L01 OLD" else "L01_NEW"
    )
    profile = load_profile(profile_key)

    # Si l'utilisateur change de dataset, on purge l'ancienne prediction.
    if st.session_state.get("last_dataset") != dataset_label:
        st.session_state.pop("last_prediction", None)
        st.session_state["last_dataset"] = dataset_label

    st.subheader("Saisir une recette")

    inputs: dict = {}
    with st.form("predictor_form"):
        cat_cols = features["categorical"]
        num_cols = features["numeric"]
        has_ratio = "muscovite_ratio" in num_cols
        num_cols_input = [col for col in num_cols if col != "muscovite_ratio"]

        if cat_cols:
            section_header("Param\u00e8tres cat\u00e9goriels")
            for col in cat_cols:
                if col == "Tailings":
                    inputs[col] = _safe_selectbox(col, [tailings], key=f"pred_{col}", default=tailings)
                    continue
                if col == "Binder" and profile:
                    options = profile.categorical_values.get("Binder", [])
                elif col in cat_values:
                    options = cat_values[col]
                else:
                    options = ["GUL", "Slag"]
                inputs[col] = _safe_selectbox(col, options, key=f"pred_{col}")

        sections = {
            "Ingr\u00e9dients / Dosages": [],
            "Granulom\u00e9trie": [],
            "Param\u00e8tres additionnels": [],
        }
        for col in num_cols_input:
            sections[_classify_column(col)].append(col)

        for section, cols in sections.items():
            if not cols:
                continue
            section_header(section)
            for col in cols:
                default_val = 0.0
                if profile and col in profile.numeric_stats:
                    default_val = profile.numeric_stats[col].get("mean", 0.0) or 0.0
                key = f"pred_{col}"
                value = st.number_input(
                    col,
                    value=float(st.session_state.get(key, default_val)),
                    help=_unit_help(col),
                    key=key,
                )
                inputs[col] = value

                if profile:
                    stats = get_feature_profile(profile, col)
                    bounds_text = format_bounds(stats)
                    status = ood_level(value, stats)
                    if status["level"] != "ok":
                        st.markdown(
                            f"{_badge_for_level(status['level'])} "
                            + (f"Training: {bounds_text}" if bounds_text else "Profil insuffisant"),
                            unsafe_allow_html=True,
                        )
                        if status["level"] == "out":
                            st.caption(
                                "Valeur hors distribution : la prediction peut etre moins fiable."
                            )

        if has_ratio:
            section_header("Param\u00e8tres additionnels")
            st.text_input(
                "muscovite_ratio (calcule automatiquement)",
                value="sera calcule apres prediction",
                disabled=True,
                key="muscovite_ratio_placeholder",
            )

        submitted = st.form_submit_button("Pr\u00e9dire")

    if submitted:
        required_cols = features["categorical"] + features["numeric"]
        required_cols_input = [
            col for col in required_cols if col != "muscovite_ratio"
        ]
        missing = validate_inputs(inputs, required_cols_input)
        if missing:
            st.error(f"Champs manquants: {', '.join(missing)}")
            return

        df_input = build_input_frame(inputs, required_cols)
        if "muscovite_ratio" in required_cols:
            inputs["muscovite_ratio"] = df_input["muscovite_ratio"].iloc[0]
        slump_pred, ucs_pred = predict_targets(slump_model, ucs_model, df_input)

        st.session_state["last_prediction"] = {
            "inputs": inputs,
            "slump_pred": slump_pred,
            "ucs_pred": ucs_pred,
            "required_cols": required_cols,
            "tailings": tailings,
        }

    prediction = st.session_state.get("last_prediction")
    if not prediction:
        return

    slump_pred = prediction["slump_pred"]
    ucs_pred = prediction["ucs_pred"]
    used_inputs = prediction["inputs"]

    rmse = model_registry.get_rmse_for_tailings(bundle, tailings)
    ucs_rmse = rmse.get("ucs")
    slump_rmse = rmse.get("slump")

    col1, col2 = st.columns(2)
    with col1:
        status = "OK" if ucs_pred >= 900 else "Alerte"
        metric_card("UCS pr\u00e9dit", f"{ucs_pred:.1f}", "kPa", status)
        if ucs_rmse is not None:
            st.caption(
                f"UCS pr\u00e9dite = {ucs_pred:.1f} +/- {ucs_rmse:.0f} kPa (RMSE approx.)"
            )
    with col2:
        status = "OK" if slump_pred >= 70 else "Alerte"
        metric_card("Slump pr\u00e9dit", f"{slump_pred:.1f}", "mm", status)
        if slump_rmse is not None:
            st.caption(
                f"Slump pr\u00e9dit = {slump_pred:.1f} +/- {slump_rmse:.0f} mm (RMSE approx.)"
            )

    st.subheader("R\u00e9capitulatif des inputs")
    st.dataframe(pd.DataFrame([used_inputs]))

    if profile:
        warnings = warn_out_of_profile(profile, used_inputs)
        warning_list(warnings)

    if "muscovite_ratio" in features["numeric"]:
        ratio_value = used_inputs.get("muscovite_ratio")
        display_ratio = "N/A" if pd.isna(ratio_value) else f"{ratio_value:.4f}"
        st.text_input(
            "muscovite_ratio (calcul\u00e9 automatiquement)",
            value=display_ratio,
            disabled=True,
            key="muscovite_ratio_value",
        )
        st.info(f"muscovite_ratio calcul\u00e9 automatiquement: {display_ratio}")

    st.subheader("Comparer \u00e0 un test labo (optionnel)")
    col_a, col_b = st.columns(2)
    with col_a:
        ucs_labo = st.number_input("UCS labo (kPa)", value=float(st.session_state.get("pred_ucs_labo", 0.0)), key="pred_ucs_labo")
    with col_b:
        slump_labo = st.number_input("Slump labo (mm)", value=float(st.session_state.get("pred_slump_labo", 0.0)), key="pred_slump_labo")

    if ucs_labo > 0:
        err = ucs_pred - ucs_labo
        st.write(
            f"Erreur UCS: {err:.1f} kPa | "
            f"Abs: {abs(err):.1f} kPa | "
            f"%: {abs(err) / ucs_labo * 100:.1f}"
        )
    if slump_labo > 0:
        err = slump_pred - slump_labo
        st.write(
            f"Erreur Slump: {err:.1f} mm | "
            f"Abs: {abs(err):.1f} mm | "
            f"%: {abs(err) / slump_labo * 100:.1f}"
        )


if __name__ == "__main__":
    main()
