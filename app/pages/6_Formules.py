# -*- coding: utf-8 -*-
"""Page Streamlit Formules UCS (L01 NEW)."""

from __future__ import annotations

from pathlib import Path
import io

import pandas as pd
import streamlit as st

from app.core import formulas_registry
from app.ui.components import load_css, section_header, metric_card, badge

DEFAULT_BASE_DIRS = ["outputs/formulas", "artifacts/formulas"]
DEFAULT_COMMAND = (
    "python scripts/build_formulas_l01_new.py "
    "--dataset-xlsx data/L01-dataset.xlsx "
    "--models-dir outputs/final_models/FINAL_new_best "
    "--out-dir outputs/formulas/L01_new --seed 42"
)


def _latex_label(text: str) -> str:
    label = text.replace("%", "\\%")
    label = label.replace(" ", "\\,")
    return label


def _equation_latex(equation: dict) -> str:
    terms = equation.get("terms", [])
    if not terms:
        return ""
    parts = [f"{equation.get('intercept', 0.0):.3f}"]
    for term in terms:
        name = term.get("term_name", "")
        coef = float(term.get("coefficient", 0.0))
        if term.get("term_type") == "categorical":
            label = name.replace("Binder_", "Binder=")
        else:
            label = name
        parts.append(f"{coef:+.3f}\\times {_latex_label(label)}")
    return "UCS = " + " ".join(parts)


def _render_equation_block(equation: dict) -> None:
    latex = _equation_latex(equation)
    if latex:
        st.latex(latex)
    text = equation.get("equation_text") or ""
    if text:
        st.markdown(text)


def _build_inputs(equation: dict) -> dict:
    numeric_features = equation.get("numeric_features", [])
    ordered = [
        "Gs",
        "E/C",
        "P20 (\u00b5m)",
        "P80 (\u00b5m)",
        "Cw_f",
        "Ad %",
        "muscovite_added (%)",
        "muscovite_total (%)",
        "phyllosilicates (%)",
        "muscovite_ratio",
    ]
    ordered = [f for f in ordered if f in numeric_features or f == "muscovite_ratio"]
    remaining = [f for f in numeric_features if f not in ordered]
    features = ordered + remaining

    inputs = {}
    for feat in features:
        if feat == "muscovite_ratio":
            continue
        key = f"classic_{feat}"
        inputs[feat] = st.number_input(feat, value=0.0, step=0.1, key=key)

    added = inputs.get("muscovite_added (%)")
    total = inputs.get("muscovite_total (%)")
    ratio = 0.0
    if total and total > 0:
        ratio = added / total if added is not None else 0.0
    if "muscovite_ratio" in numeric_features:
        st.text_input(
            "muscovite_ratio (calcul\u00e9 automatiquement)",
            value=f"{ratio:.4f}",
            disabled=True,
        )
        inputs["muscovite_ratio"] = ratio
    return inputs


def _warnings_for_inputs(inputs: dict) -> None:
    warnings = formulas_registry.check_out_of_distribution(inputs, profile_key="L01_NEW")
    for msg in warnings:
        st.warning(msg)


def _csv_download_button(path: Path, label: str) -> None:
    if not path.exists():
        st.caption(f"{label} indisponible.")
        return
    data = path.read_bytes()
    st.download_button(label, data=data, file_name=path.name)


def main() -> None:
    load_css("app/ui/styles.css")
    st.title("Formules UCS \u2013 L01 NEW")
    st.caption("Equations interpr\u00e9tables \u00e0 partir de L01 NEW (sans recalcul ML).")

    with st.expander("Source des formules"):
        mode = st.radio(
            "Mode de chargement",
            options=["Auto", "Chemin local", "URL zip"],
            horizontal=True,
            key="formula_source_mode",
        )
        source_path = None
        if mode == "Auto":
            dirs = formulas_registry.find_formulas_dirs(DEFAULT_BASE_DIRS)
            if not dirs:
                st.info("Formules non disponibles. G\u00e9n\u00e9rez-les en local.")
                st.code(DEFAULT_COMMAND, language="bash")
            else:
                choice = st.selectbox("Dossier detecte", list(dirs.keys()))
                source_path = dirs.get(choice)
        elif mode == "Chemin local":
            path_text = st.text_input("Chemin du dossier de formules", value="")
            if path_text:
                source_path = Path(path_text)
        else:
            url = st.text_input("URL du zip (formules)", value="")
            if url and st.button("Charger le zip"):
                source_path = formulas_registry.load_from_url(url)
                st.session_state["formula_url_path"] = str(source_path) if source_path else ""
            elif st.session_state.get("formula_url_path"):
                source_path = Path(st.session_state["formula_url_path"])

    if not source_path or not Path(source_path).exists():
        st.warning("Formules non disponibles \u2013 ex\u00e9cute build_formulas_l01_new.py en local.")
        return

    formulas_dir = Path(source_path)
    equation = formulas_registry.load_classic_equation(formulas_dir)
    if not equation:
        st.error("Equation classique introuvable dans ce dossier.")
        return

    tabs = st.tabs(
        [
            "Equation classique",
            "Spline + ElasticNet (r\u00e9sum\u00e9)",
            "R\u00e8gles (arbre)",
            "T\u00e9l\u00e9chargements",
        ]
    )

    # ---- Tab 1: equation classique
    with tabs[0]:
        section_header("Equation classique (UCS)")
        metrics = equation.get("metrics", {}) or {}
        r2_label = "R\u00b2"
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"{badge(r2_label, 'ok')} {metrics.get('r2', 'n/a')}")
        with col_b:
            st.markdown(f"{badge('RMSE', 'warn')} {metrics.get('rmse', 'n/a')} kPa")

        st.markdown("Equation (LaTeX) :")
        _render_equation_block(equation)

        eq_text = equation.get("equation_text") or _equation_latex(equation)
        if st.button("Copier l'equation"):
            st.info("Selectionne le texte ci-dessous puis Ctrl+C.")
        st.text_area("Equation copiable", value=eq_text, height=150)

        section_header("Calculateur UCS")
        binder_values = equation.get("binder_values", ["GUL", "20G80S"])
        binder = st.selectbox("Binder", options=binder_values)

        inputs = _build_inputs(equation)
        inputs["Binder"] = binder

        _warnings_for_inputs(inputs)

        if st.button("Calculer"):
            ucs, contrib_df = formulas_registry.compute_ucs_classic(equation, inputs)
            rmse = metrics.get("rmse")
            if rmse:
                st.success(f"UCS \u2248 {ucs:.2f} kPa \u00b1 {rmse:.2f} kPa")
            else:
                st.success(f"UCS \u2248 {ucs:.2f} kPa")

            st.dataframe(contrib_df, use_container_width=True)

            csv_buf = io.StringIO()
            contrib_df.to_csv(csv_buf, index=False)
            st.download_button(
                "Telecharger contributions CSV",
                data=csv_buf.getvalue(),
                file_name="contributions_ucs.csv",
            )

    # ---- Tab 2: spline summary
    with tabs[1]:
        section_header("Spline + ElasticNet (r\u00e9sum\u00e9)")
        global_md = formulas_registry.load_text_file_safe(formulas_dir / "global_formula.md")
        global_metrics = formulas_registry.load_metrics_safe(
            formulas_dir / "global_spline_enet_metrics.json"
        )
        if global_metrics:
            st.markdown(f"R\u00b2 = {global_metrics.get('r2', 'n/a')}, RMSE = {global_metrics.get('rmse', 'n/a')} kPa")
        if global_md:
            st.markdown(global_md)
        st.info("Formule spline utilisable via le .joblib (pas \u00e0 la main).")

    # ---- Tab 3: rules
    with tabs[2]:
        section_header("R\u00e8gles (arbre)")
        overall = formulas_registry.load_text_file_safe(formulas_dir / "rules_tree_overall.txt")
        if overall:
            st.text(overall)
        binder_rules = sorted(formulas_dir.glob("rules_tree_by_binder_*.txt"))
        if binder_rules:
            sub_tabs = st.tabs([p.stem.replace("rules_tree_by_binder_", "") for p in binder_rules])
            for tab, path in zip(sub_tabs, binder_rules):
                with tab:
                    st.text(path.read_text(encoding="utf-8"))

    # ---- Tab 4: downloads
    with tabs[3]:
        section_header("Telechargements")
        _csv_download_button(formulas_dir / "classic_linear_coefficients.csv", "Coefficients (CSV)")
        _csv_download_button(formulas_dir / "classic_linear_metrics.json", "Metriques (JSON)")
        _csv_download_button(formulas_dir / "classic_linear_equation.md", "Equation (MD)")
        _csv_download_button(formulas_dir / "global_formula.md", "Resume spline (MD)")
        _csv_download_button(formulas_dir / "rules_tree_overall.txt", "Regles arbre (TXT)")
