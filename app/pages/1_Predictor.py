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
        return "Parametres additionnels"
    if "p" in name and any(ch.isdigit() for ch in name):
        return "Granulometrie"
    if "d" in name and any(ch.isdigit() for ch in name):
        return "Granulometrie"
    return "Ingredients / Dosages"


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


def _parse_float(value: object) -> float | None:
    """Convertit une saisie texte en float (None si invalide)."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def main() -> None:
    st.title("Predicteur CPB")
    load_css("app/ui/styles.css")

    with st.sidebar:
        st.header("Configuration")
        dataset_label = _safe_selectbox(
            "Famille de residus",
            ["WW", "L01 OLD", "L01 NEW"],
            key="pred_dataset",
            default="L01 NEW",
        )
        if dataset_label == "L01 NEW":
            model_version = "FINAL_new_best"
        else:
            model_version = "FINAL_old_best"

        st.caption("Modele utilise")
        st.code(model_version)

    bundle = model_registry.load_bundle(model_version)
    if not bundle:
        st.error(
            "Modeles introuvables. Placez-les dans outputs/final_models ou "
            "definissez MODEL_DOWNLOAD_URL dans secrets."
        )
        return

    tailings = "WW" if dataset_label == "WW" else "L01"
    slump_model, ucs_model = model_registry.resolve_tailings_model(bundle, tailings)
    if not slump_model or not ucs_model:
        st.error("Modeles incomplets pour ce residu.")
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
    num_cols_input: list[str] = []

    cat_cols = features["categorical"]
    num_cols = features["numeric"]
    has_ratio = "muscovite_ratio" in num_cols
    num_cols_input = [col for col in num_cols if col != "muscovite_ratio"]

    if cat_cols:
        section_header("Parametres categoriels")
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
        "Ingredients / Dosages": [],
        "Granulometrie": [],
        "Parametres additionnels": [],
    }
    for col in num_cols_input:
        sections[_classify_column(col)].append(col)

    invalid_fields: list[str] = []
    raw_values: dict[str, object] = {}
    for section, cols in sections.items():
        if not cols:
            continue
        section_header(section)
        for col in cols:
            default_val = 0.0
            if profile and col in profile.numeric_stats:
                default_val = profile.numeric_stats[col].get("mean", 0.0) or 0.0
            key = f"pred_{col}"
            default_text = st.session_state.get(key, f"{default_val:.4f}")
            raw = st.text_input(
                col,
                value=str(default_text),
                help=_unit_help(col),
                key=key,
            )
            raw_values[col] = raw
            parsed = _parse_float(raw)
            if parsed is None:
                invalid_fields.append(col)
            else:
                inputs[col] = parsed

    if has_ratio:
        section_header("Parametres additionnels")
        st.text_input(
            "muscovite_ratio (calcule automatiquement)",
            value="sera calcule apres prediction",
            disabled=True,
            key="muscovite_ratio_placeholder",
        )

    submitted = st.button("Predire")

    if profile and num_cols_input:
        warnings_to_show = []
        for col in num_cols_input:
            value = raw_values.get(col)
            parsed = _parse_float(value)
            if parsed is None:
                continue
            stats = get_feature_profile(profile, col)
            status = ood_level(parsed, stats)
            if status["level"] != "ok":
                bounds_text = format_bounds(stats)
                warnings_to_show.append(
                    (col, status["level"], bounds_text)
                )

        if warnings_to_show:
            section_header("Avertissement distribution")
            for col, level, bounds_text in warnings_to_show:
                st.markdown(
                    f"{_badge_for_level(level)} {col} "
                    + (f"| Training: {bounds_text}" if bounds_text else "| Profil insuffisant"),
                    unsafe_allow_html=True,
                )
                if level == "out":
                    st.caption("Valeur hors distribution : la prediction peut etre moins fiable.")

    if submitted:
        required_cols = features["categorical"] + features["numeric"]
        required_cols_input = [
            col for col in required_cols if col != "muscovite_ratio"
        ]
        missing = validate_inputs(inputs, required_cols_input)
        if invalid_fields:
            st.error(
                "Valeurs invalides : "
                + ", ".join(invalid_fields)
                + ". Utiliser un nombre (ex: 5.9)."
            )
            return
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
        metric_card("UCS predit", f"{ucs_pred:.1f}", "kPa", status)
        if ucs_rmse is not None:
            st.caption(
                f"UCS predite = {ucs_pred:.1f} +/- {ucs_rmse:.0f} kPa (RMSE approx.)"
            )
    with col2:
        status = "OK" if slump_pred >= 70 else "Alerte"
        metric_card("Slump predit", f"{slump_pred:.1f}", "mm", status)
        if slump_rmse is not None:
            st.caption(
                f"Slump predit = {slump_pred:.1f} +/- {slump_rmse:.0f} mm (RMSE approx.)"
            )

    st.subheader("Recapitulatif des inputs")
    st.dataframe(pd.DataFrame([used_inputs]))

    if profile:
        warnings = warn_out_of_profile(profile, used_inputs)
        warning_list(warnings)

    if "muscovite_ratio" in features["numeric"]:
        ratio_value = used_inputs.get("muscovite_ratio")
        display_ratio = "N/A" if pd.isna(ratio_value) else f"{ratio_value:.4f}"
        st.text_input(
            "muscovite_ratio (calcule automatiquement)",
            value=display_ratio,
            disabled=True,
            key="muscovite_ratio_value",
        )
        st.info(f"muscovite_ratio calcule automatiquement: {display_ratio}")

    st.subheader("Comparer a un test labo (optionnel)")
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
