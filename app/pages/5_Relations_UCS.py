# -*- coding: utf-8 -*-
"""Page Streamlit pour visualiser les sorties d'interpretabilite UCS."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.core import interpretability_registry as registry
from app.ui.components import section_header

DEFAULT_BASE_DIR = "outputs/interpretability"
DEFAULT_COMMAND = (
    "python scripts/explain_ucs_relationships.py "
    "--models-dir outputs/final_models/FINAL_new_best "
    "--dataset-xlsx data/L01-dataset.xlsx "
    "--out-dir outputs/interpretability/FINAL_new_best"
)


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _plot_importance(df: pd.DataFrame):
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    data = df.sort_values("importance_mean", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(data["feature"], data["importance_mean"])
    ax.set_xlabel("Importance moyenne")
    ax.set_title("Importance (permutation)")
    fig.tight_layout()
    return fig


def _plot_pdp(df: pd.DataFrame):
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["grid_value"], df["pred_mean"], marker="o")
    ax.set_xlabel("Valeur")
    ax.set_ylabel("UCS predit (kPa)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_binder(df: pd.DataFrame):
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["Binder"], df["pred_mean"])
    ax.set_xlabel("Binder")
    ax.set_ylabel("UCS predit (kPa)")
    fig.tight_layout()
    return fig


def _plot_heatmap(df: pd.DataFrame):
    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    cols = [c.lower() for c in df.columns]
    if {"x", "y", "pred_mean"}.issubset(cols):
        df = df.copy()
        df.columns = cols
        pivot = df.pivot(index="y", columns="x", values="pred_mean")
        x_vals = pivot.columns.to_numpy()
        y_vals = pivot.index.to_numpy()
        z = pivot.to_numpy()
    elif df.shape[1] >= 3:
        df = df.copy()
        df.columns = ["x", "y", "pred_mean"] + list(df.columns[3:])
        pivot = df.pivot(index="y", columns="x", values="pred_mean")
        x_vals = pivot.columns.to_numpy()
        y_vals = pivot.index.to_numpy()
        z = pivot.to_numpy()
    else:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Interaction (UCS predit)")
    fig.tight_layout()
    return fig


def _show_generation_help() -> None:
    st.info("Aucune analyse trouv\u00e9e. G\u00e9n\u00e9rez-la d'abord.")
    st.code(DEFAULT_COMMAND, language="bash")


def main() -> None:
    st.title("Relations (UCS)")
    st.caption("Interpr\u00e9tabilit\u00e9 du mod\u00e8le (importance, PDP, effet du liant, interactions).")

    base_dir = Path(DEFAULT_BASE_DIR)
    runs = registry.list_interpretability_runs(base_dir)
    run_names = [p.name for p in runs]

    col_a, col_b = st.columns([2, 3])
    selection = None
    with col_a:
        if run_names:
            selection = st.selectbox(
                "Choisir un dossier d'analyse",
                options=run_names,
                index=0,
                key="interpretability_run",
            )
        else:
            st.caption("Aucun dossier detecte dans outputs/interpretability.")
    with col_b:
        custom_path = st.text_input(
            "Chemin personnalis\u00e9 (optionnel)",
            value="",
            placeholder="outputs/interpretability/FINAL_new_best",
            key="interpretability_custom",
        )

    if custom_path:
        run_dir = Path(custom_path)
    elif selection:
        run_dir = base_dir / selection
    else:
        _show_generation_help()
        return

    if not run_dir.exists():
        st.error(f"Dossier introuvable: {run_dir}")
        _show_generation_help()
        return

    st.caption(f"Dossier: {run_dir}")

    files = registry.detect_files(run_dir)
    pdp_files = registry.parse_pdp_files(run_dir)
    interaction_files = registry.parse_interaction_files(run_dir)

    tabs = st.tabs(
        [
            "Importance",
            "PDP (variables)",
            "Effet du liant",
            "Interactions",
        ]
    )

    with tabs[0]:
        section_header("Importance globale")
        if not files["importance_csv"]:
            st.warning("Fichier d'importance introuvable.")
        else:
            df = pd.read_csv(files["importance_csv"])
            if "importance_mean" in df.columns:
                df = df.sort_values("importance_mean", ascending=False)
            st.dataframe(df, use_container_width=True)

            if files["importance_png"] and files["importance_png"].exists():
                st.image(str(files["importance_png"]))
            else:
                fig = _plot_importance(df) if "importance_mean" in df.columns else None
                if fig is None:
                    st.warning("matplotlib requis pour afficher le graphique.")
                else:
                    st.pyplot(fig, clear_figure=True)

        st.info(
            "Interpr\u00e9tation: l'importance mesure l'impact moyen d'une variable "
            "sur la pr\u00e9diction, pas une causalit\u00e9 directe."
        )

    with tabs[1]:
        section_header("PDP (tendance moyenne)")
        if not pdp_files:
            st.warning("Aucun fichier PDP trouv\u00e9.")
        else:
            labels = {token: meta.get("label", token) for token, meta in pdp_files.items()}
            token = st.selectbox(
                "Variable",
                options=list(labels.keys()),
                format_func=lambda key: labels[key],
                key="pdp_variable",
            )
            meta = pdp_files.get(token, {})
            if meta.get("png") and Path(meta["png"]).exists():
                st.image(str(meta["png"]))
            elif meta.get("csv"):
                df = pd.read_csv(meta["csv"])
                if {"grid_value", "pred_mean"}.issubset(df.columns):
                    fig = _plot_pdp(df)
                    if fig is None:
                        st.warning("matplotlib requis pour afficher le graphique.")
                    else:
                        st.pyplot(fig, clear_figure=True)
                st.dataframe(df, use_container_width=True)
                if "grid_value" in df.columns:
                    st.caption(
                        f"Plage observee: {df['grid_value'].min():.2f} \u2192 {df['grid_value'].max():.2f}"
                    )
            else:
                st.warning("Fichier PDP introuvable pour cette variable.")

            st.caption("Tendance moyenne apprise par le mod\u00e8le, pas causalit\u00e9.")

    with tabs[2]:
        section_header("Effet du liant (Binder)")
        if not files["binder_csv"]:
            st.warning("Fichier binder_effect.csv introuvable.")
        else:
            df = pd.read_csv(files["binder_csv"])
            st.dataframe(df, use_container_width=True)
            if files["binder_png"] and Path(files["binder_png"]).exists():
                st.image(str(files["binder_png"]))
            else:
                fig = _plot_binder(df) if {"Binder", "pred_mean"}.issubset(df.columns) else None
                if fig is None:
                    st.warning("matplotlib requis pour afficher le graphique.")
                else:
                    st.pyplot(fig, clear_figure=True)
            st.caption(
                "Diff\u00e9rence moyenne pr\u00e9dite entre les types de liant."
            )

    with tabs[3]:
        section_header("Interactions (PDP 2D)")
        if not interaction_files:
            st.warning("Aucun fichier d'interaction trouv\u00e9.")
        else:
            labels = {token: meta.get("label", token) for token, meta in interaction_files.items()}
            token = st.selectbox(
                "Couple de variables",
                options=list(labels.keys()),
                format_func=lambda key: labels[key],
                key="interaction_pair",
            )
            meta = interaction_files.get(token, {})
            if meta.get("png") and Path(meta["png"]).exists():
                st.image(str(meta["png"]))
            elif meta.get("csv"):
                df = pd.read_csv(meta["csv"])
                fig = _plot_heatmap(df)
                if fig is None:
                    st.warning("Impossible de tracer la heatmap (format CSV non reconnu).")
                else:
                    st.pyplot(fig, clear_figure=True)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("Fichier d'interaction introuvable pour ce couple.")
            st.caption(
                "Attention: extrapolation possible si la grille depasse les plages observees."
            )

    st.divider()
    with st.expander("Rapport d'analyse"):
        if files["report"] and Path(files["report"]).exists():
            st.markdown(Path(files["report"]).read_text(encoding="utf-8"))
        else:
            st.warning("rapport.md introuvable.")
            _show_generation_help()
