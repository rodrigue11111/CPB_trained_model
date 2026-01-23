# -*- coding: utf-8 -*-
"""Page Streamlit Formules UCS (L01 NEW)."""

from __future__ import annotations

from pathlib import Path
import io

import pandas as pd
import streamlit as st

from app.core import data_profiles, formulas_registry
from app.ui.components import load_css, section_header

DEFAULT_BASE_DIRS = ["outputs/formulas", "artifacts/formulas"]
DEFAULT_COMMAND = (
    "python scripts/build_formulas_l01_new.py "
    "--dataset-xlsx data/L01-dataset.xlsx "
    "--models-dir outputs/final_models/FINAL_new_best "
    "--out-dir outputs/formulas/L01_new --seed 42"
)

# Coefficients affichés pour L01 NEW (arrondis à 3 décimales à l'affichage).
# IMPORTANT: ces valeurs sont uniquement pour l'UI (le calculateur reste inchangé).
DISPLAY_INTERCEPT = -8096.2626
DISPLAY_BONUS_20G80S = 230.7892
DISPLAY_BONUS_GUL = -230.7892
DISPLAY_TERMS = [
    {"label": "Gs", "latex": "Gs", "excel": "Gs", "coef": 419.0878},
    {"label": "(E/C)", "latex": "(E/C)", "excel": "EC", "coef": -115.9450},
    {"label": "P80", "latex": "P80", "excel": "P80", "coef": 88.4569},
    {"label": "Ad", "latex": "Ad", "excel": "Ad", "coef": -41.4015},
    {
        "label": "muscovite_ratio",
        "latex": "muscovite\\_ratio",
        "excel": "muscovite_ratio",
        "coef": 36.4561,
    },
    {"label": "Cw_f", "latex": "Cw\\_f", "excel": "Cw_f", "coef": 33.8206},
    {"label": "phyllosilicates", "latex": "phyllosilicates", "excel": "phyllosilicates", "coef": -18.4575},
    {"label": "P20", "latex": "P20", "excel": "P20", "coef": -13.1245},
    {
        "label": "muscovite_added",
        "latex": "muscovite\\_added",
        "excel": "muscovite_added",
        "coef": 12.0953,
    },
    {
        "label": "muscovite_total",
        "latex": "muscovite\\_total",
        "excel": "muscovite_total",
        "coef": 10.0835,
    },
]


def _latex_label(text: str) -> str:
    """Rend un libellé LaTeX sûr (sans caractères spéciaux non échappés)."""
    if not text:
        return "\\mathrm{}"
    # Normalisation simple : on remplace les caractères problématiques.
    cleaned = []
    for ch in text:
        if ch.isalnum():
            # Evite les caractères non-ASCII (µ, etc.) qui cassent KaTeX.
            cleaned.append(ch if ch.isascii() else "_")
        elif ch in {" ", "-", "/", "%", "(", ")", "[", "]"}:
            cleaned.append("_")
        elif ch == "=":
            cleaned.append("=")
        elif ch in {".", "_"}:
            cleaned.append("_")
        else:
            cleaned.append("_")
    label = "".join(cleaned)
    while "__" in label:
        label = label.replace("__", "_")
    label = label.strip("_")
    # Echappe les underscores pour éviter les sous-indices accidentels.
    label = label.replace("_", "\\_")
    return f"\\mathrm{{{label}}}"


def _equation_latex(equation: dict, max_terms: int = 12) -> str:
    """Construit un aperçu LaTeX lisible (avec sauts de ligne)."""
    terms = equation.get("terms", [])
    if not terms:
        return ""
    intercept = float(equation.get("intercept", 0.0))
    lines = []
    line = f"\\mathrm{{UCS}} = {intercept:.3f}"
    for idx, term in enumerate(terms[:max_terms], start=1):
        name = term.get("term_name", "")
        coef = float(term.get("coefficient", 0.0))
        if term.get("term_type") == "categorical":
            label = name.replace("Binder_", "Binder=")
        else:
            label = name
        line += f" {coef:+.3f}\\times {_latex_label(label)}"
        if idx % 4 == 0:
            lines.append(line)
            line = ""
    if line:
        lines.append(line)
    if len(terms) > max_terms:
        lines.append("+ \\dots")
    return "\\begin{aligned}" + " \\\\ ".join(lines) + "\\end{aligned}"


def _render_equation_block(equation: dict) -> None:
    latex = _equation_latex(equation)
    if latex:
        st.latex(latex)
    text = equation.get("equation_text") or ""
    if text:
        with st.expander("Voir le détail complet de l’équation"):
            st.markdown(text)


def _pick_default_value(stats: dict) -> float:
    """Choisit une valeur par défaut lisible pour éviter les zéros absurdes."""
    mean = stats.get("mean")
    if mean is not None and not pd.isna(mean):
        return float(mean)
    p05 = stats.get("p05")
    p95 = stats.get("p95")
    if p05 is not None and p95 is not None and not pd.isna(p05) and not pd.isna(p95):
        return float((p05 + p95) / 2)
    min_val = stats.get("min")
    max_val = stats.get("max")
    if min_val is not None and max_val is not None and not pd.isna(min_val) and not pd.isna(max_val):
        return float((min_val + max_val) / 2)
    return 0.0


def _numeric_step(value: float) -> float:
    """Ajuste le pas en fonction de l'ordre de grandeur."""
    abs_val = abs(value)
    if abs_val >= 100:
        return 1.0
    if abs_val >= 10:
        return 0.1
    return 0.01


def _format_metric(value: object, digits: int = 3) -> str:
    """Formate une métrique en texte court."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{val:.{digits}f}"


def _render_classic_equation_display() -> None:
    """Affiche l'équation classique en 3 formats lisibles (texte, LaTeX, Excel)."""
    st.markdown("### Équation lisible (texte)")
    lines = [f"UCS (kPa) =\n  {DISPLAY_INTERCEPT:.3f}"]
    for term in DISPLAY_TERMS:
        coef = term["coef"]
        sign = "+" if coef >= 0 else "-"
        lines.append(f"  {sign} {abs(coef):.3f} × {term['label']}")
    lines.append("  + bonus_binder")
    text_block = "\n".join(lines)
    st.code(text_block, language="text")
    st.markdown(
        "où :\n"
        f"- bonus_binder = +{DISPLAY_BONUS_20G80S:.3f} si Binder = 20G80S\n"
        f"- bonus_binder = {DISPLAY_BONUS_GUL:.3f} si Binder = GUL\n"
        "- bonus_binder = 0 sinon\n"
        "\n(P20, P80 en µm ; Ad en %)"
    )

    st.markdown("### Équation LaTeX")
    # Construction multi-lignes pour éviter une ligne trop longue.
    line1 = (
        f"\\mathrm{{UCS}} = {DISPLAY_INTERCEPT:.3f}"
        f" {DISPLAY_TERMS[0]['coef']:+.3f}\\, {DISPLAY_TERMS[0]['latex']}"
        f" {DISPLAY_TERMS[1]['coef']:+.3f}\\, {DISPLAY_TERMS[1]['latex']}"
        f" {DISPLAY_TERMS[2]['coef']:+.3f}\\, {DISPLAY_TERMS[2]['latex']}"
    )
    line2 = (
        f"\\quad {DISPLAY_TERMS[3]['coef']:+.3f}\\, {DISPLAY_TERMS[3]['latex']}"
        f" {DISPLAY_TERMS[4]['coef']:+.3f}\\, {DISPLAY_TERMS[4]['latex']}"
        f" {DISPLAY_TERMS[5]['coef']:+.3f}\\, {DISPLAY_TERMS[5]['latex']}"
    )
    line3 = (
        f"\\quad {DISPLAY_TERMS[6]['coef']:+.3f}\\, {DISPLAY_TERMS[6]['latex']}"
        f" {DISPLAY_TERMS[7]['coef']:+.3f}\\, {DISPLAY_TERMS[7]['latex']}"
    )
    line4 = (
        f"\\quad {DISPLAY_TERMS[8]['coef']:+.3f}\\, {DISPLAY_TERMS[8]['latex']}"
        f" {DISPLAY_TERMS[9]['coef']:+.3f}\\, {DISPLAY_TERMS[9]['latex']}"
        " + bonus_{{binder}}"
    )
    st.latex(
        "\\begin{aligned}"
        + " \\\\ ".join([line1, line2, line3, line4])
        + "\\end{aligned}"
    )
    st.latex(
        "\\begin{aligned}"
        "bonus_{binder} ="
        "\\begin{cases}"
        f"+{DISPLAY_BONUS_20G80S:.3f} & \\text{{si Binder = 20G80S}} \\\\"
        f"{DISPLAY_BONUS_GUL:.3f} & \\text{{si Binder = GUL}} \\\\"
        "0 & \\text{sinon}"
        "\\end{cases}"
        "\\end{aligned}"
    )

    st.markdown("### Formule Excel (copier-coller)")
    excel_lines = [
        f"={DISPLAY_INTERCEPT:.3f}",
    ]
    for term in DISPLAY_TERMS:
        coef = term["coef"]
        sign = "+" if coef >= 0 else "-"
        excel_lines.append(f" {sign}{abs(coef):.3f}*{term['excel']}")
    excel_lines.append(
        f" +IF(Binder=\"20G80S\",{DISPLAY_BONUS_20G80S:.3f},"
        f"IF(Binder=\"GUL\",{DISPLAY_BONUS_GUL:.3f},0))"
    )
    st.code("\n".join(excel_lines), language="text")
    st.caption("Adapter les noms des cellules/colonnes si besoin.")

    st.info(
        "Comment lire la formule ?\n"
        "- Coefficients + : UCS augmente quand la variable augmente (toutes choses égales).\n"
        "- Coefficients − : UCS diminue quand la variable augmente.\n"
        "- Valable dans la plage des données L01 NEW."
    )


def _build_inputs(equation: dict, defaults: dict[str, float]) -> dict:
    numeric_features = equation.get("numeric_features", [])
    features_set = set(numeric_features)
    groups = {
        "Proc\u00e9d\u00e9 (recette)": ["E/C", "Cw_f", "Ad %"],
        "Granulom\u00e9trie": ["P20 (\u00b5m)", "P80 (\u00b5m)"],
        "Min\u00e9raux": [
            "phyllosilicates (%)",
            "muscovite_added (%)",
            "muscovite_total (%)",
            "muscovite_ratio",
        ],
        "Physique": ["Gs"],
    }
    for group in list(groups.keys()):
        groups[group] = [f for f in groups[group] if f in features_set]
        if not groups[group]:
            groups.pop(group)

    used = set()
    for feats in groups.values():
        used.update(feats)
    remaining = [f for f in numeric_features if f not in used]

    inputs = {}

    def _render_group(title: str, feats: list[str]) -> None:
        if not feats:
            return
        st.markdown(f"**{title}**")
        for idx in range(0, len(feats), 2):
            cols = st.columns(2)
            for col, feat in zip(cols, feats[idx: idx + 2]):
                if feat == "muscovite_ratio":
                    continue
                key = f"classic_{feat}"
                default_val = float(defaults.get(feat, 0.0))
                step = _numeric_step(default_val)
                with col:
                    inputs[feat] = st.number_input(
                        feat,
                        value=default_val,
                        step=step,
                        format="%.4f",
                        key=key,
                    )

    for group_name, feats in groups.items():
        _render_group(group_name, feats)

    if remaining:
        _render_group("Autres variables", remaining)

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
        st.caption(
            "Le ratio est recalcul\u00e9 \u00e0 partir de muscovite_added et muscovite_total."
        )
        inputs["muscovite_ratio"] = ratio
    return inputs


def _warnings_for_inputs(inputs: dict) -> list[str]:
    return formulas_registry.check_out_of_distribution(inputs, profile_key="L01_NEW")


def _csv_download_button(path: Path, label: str) -> None:
    if not path.exists():
        st.caption(f"{label} indisponible.")
        return
    data = path.read_bytes()
    st.download_button(label, data=data, file_name=path.name)


def _get_default_values(equation: dict) -> dict[str, float]:
    defaults: dict[str, float] = {}
    profile = data_profiles.load_profile("L01_NEW")
    if profile is None:
        return defaults
    for feat in equation.get("numeric_features", []):
        stats = data_profiles.get_feature_profile(profile, feat)
        if stats:
            defaults[feat] = _pick_default_value(stats)
    return defaults


def main() -> None:
    load_css("app/ui/styles.css")
    st.title("Formules UCS \u2013 L01 NEW")
    st.caption(
        "Équations interprétables issues de L01 NEW (lecture seule, pas de recalcul ML)."
    )

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
                choice = st.selectbox("Dossier détecté", list(dirs.keys()))
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
        st.error("Équation classique introuvable dans ce dossier.")
        return

    tabs = st.tabs(
        [
            "Équation classique",
            "Spline + ElasticNet (r\u00e9sum\u00e9)",
            "R\u00e8gles (arbre)",
            "T\u00e9l\u00e9chargements",
        ]
    )

    defaults = _get_default_values(equation)

    # ---- Tab 1: équation classique
    with tabs[0]:
        section_header("Équation classique (UCS)")
        metrics = equation.get("metrics", {}) or {}
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("R\u00b2", _format_metric(metrics.get("r2")))
        with col_b:
            st.metric("RMSE (kPa)", _format_metric(metrics.get("rmse"), digits=1))

        _render_classic_equation_display()

        section_header("Calculateur UCS")
        binder_values = equation.get("binder_values", ["GUL", "20G80S"])
        with st.form("classic_calculator"):
            binder = st.selectbox("Binder", options=binder_values)
            inputs = _build_inputs(equation, defaults)
            inputs["Binder"] = binder
            submitted = st.form_submit_button("Calculer")

        if submitted:
            warnings = _warnings_for_inputs(inputs)
            if warnings:
                if len(warnings) == 1 and "Plage inconnue" in warnings[0]:
                    st.info(warnings[0])
                else:
                    st.warning("Valeurs hors distribution :")
                    for msg in warnings:
                        st.write(f"- {msg}")

            ucs, contrib_df = formulas_registry.compute_ucs_classic(equation, inputs)
            rmse = metrics.get("rmse")
            if rmse:
                st.success(f"UCS \u2248 {ucs:.2f} kPa \u00b1 {rmse:.2f} kPa")
            else:
                st.success(f"UCS \u2248 {ucs:.2f} kPa")

            with st.expander("Voir le détail des contributions"):
                st.dataframe(contrib_df, use_container_width=True)

            csv_buf = io.StringIO()
            contrib_df.to_csv(csv_buf, index=False)
            st.download_button(
                "Télécharger contributions CSV",
                data=csv_buf.getvalue(),
                file_name="contributions_ucs.csv",
            )

    # ---- Tab 2: spline summary
    with tabs[1]:
        section_header("Spline + ElasticNet (résumé)")
        global_md = formulas_registry.load_text_file_safe(formulas_dir / "global_formula.md")
        global_metrics = formulas_registry.load_metrics_safe(
            formulas_dir / "global_spline_enet_metrics.json"
        )
        if global_metrics:
            st.markdown(
                f"R\u00b2 = {_format_metric(global_metrics.get('r2'))}, "
                f"RMSE = {_format_metric(global_metrics.get('rmse'), digits=1)} kPa"
            )
        if global_md:
            st.markdown(global_md)
        st.info("Formule spline utilisable via le .joblib (pas à la main).")

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
        section_header("Téléchargements")
        _csv_download_button(formulas_dir / "classic_linear_coefficients.csv", "Coefficients (CSV)")
        _csv_download_button(formulas_dir / "classic_linear_metrics.json", "Métriques (JSON)")
        _csv_download_button(formulas_dir / "classic_linear_equation.md", "Équation (MD)")
        _csv_download_button(formulas_dir / "global_formula.md", "Résumé spline (MD)")
        _csv_download_button(formulas_dir / "rules_tree_overall.txt", "Règles arbre (TXT)")
