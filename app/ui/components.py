"""Composants UI simples pour Streamlit.

Ici on centralise le rendu HTML pour garder un style coherent.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def load_css(css_path: str | Path) -> None:
    """Charge un fichier CSS local dans la page Streamlit."""
    path = Path(css_path)
    if not path.exists():
        return
    css = path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def section_header(title: str, help_text: str | None = None) -> None:
    """Affiche un titre de section avec option d'aide."""
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if help_text:
        st.markdown(
            f"<div class='helper-text'>{help_text}</div>",
            unsafe_allow_html=True,
        )


def badge(text: str, kind: str = "ok") -> str:
    """Retourne un badge HTML (ok, warn, fail)."""
    css = "badge-ok" if kind == "ok" else "badge-warn" if kind == "warn" else "badge-fail"
    return f"<span class='badge {css}'>{text}</span>"


def metric_card(title: str, value: str, unit: str = "", status: str | None = None) -> None:
    """Affiche une carte metrique (UCS/Slump)."""
    status_html = ""
    if status:
        status_html = badge(status, "ok" if status == "OK" else "warn")
    html = (
        "<div class='app-card'>"
        f"<div class='card-title'>{title}</div>"
        f"<div class='card-value'>{value} {unit}</div>"
        f"{status_html}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def warning_list(items: list[str]) -> None:
    """Affiche une liste de warnings sous forme de texte."""
    if not items:
        return
    st.warning("Valeurs hors distribution observee:")
    for item in items:
        st.write(f"- {item}")
