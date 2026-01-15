# -*- coding: utf-8 -*-
"""Point d'entree Streamlit Cloud.

Navigation simple entre les pages definies dans app/pages.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import streamlit as st

from app.ui.components import load_css

st.set_page_config(page_title="CPB Predictor", layout="wide")
load_css("app/ui/styles.css")

PAGES = {
    "Prédicteur": Path("app/pages/1_Predictor.py"),
    "Générateur de recettes": Path("app/pages/2_Generateur_Recettes.py"),
    "Docs & Méthode": Path("app/pages/3_Docs_&_Methode.py"),
    "Debug modèles": Path("app/pages/4_Debug_Modeles.py"),
}


def _load_page(path: Path):
    """Charge un module Python depuis un chemin de fichier."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with st.sidebar:
    st.title("CPB Predictor")
    selection = st.radio("Navigation", list(PAGES.keys()))
    st.caption("Choisissez une page pour commencer.")

page_path = PAGES[selection]
module = _load_page(page_path)
if not module or not hasattr(module, "main"):
    st.error("Page introuvable ou invalide.")
else:
    module.main()
