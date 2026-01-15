"""Chargement robuste des modeles et metadata pour Streamlit.

Ce module isole la logique de detection des artefacts, pour faciliter
le deploiement sur Streamlit Cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import zipfile
import urllib.request

import joblib
import streamlit as st

MODEL_BASE_DIR = Path("outputs/final_models")


@dataclass
class ModelBundle:
    """Conteneur des artefacts d'un modele final.

    Attributes:
        base_dir: dossier contenant les joblib/metadata.
        metadata: contenu de metadata.json si present.
        metrics: contenu de metrics.json si present.
        best_params: contenu de best_params.json si present.
        models: dictionnaire des pipelines joblib charges.
    """

    base_dir: Path
    metadata: dict
    metrics: dict
    best_params: dict
    models: dict


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _download_models_if_needed() -> None:
    """Telecharge un zip de modeles si MODEL_DOWNLOAD_URL est defini.

    Option B: l'URL est fournie via .streamlit/secrets.toml.
    Si l'URL est absente, on ne fait rien.
    """
    # IMPORTANT: st.secrets leve une exception si aucun secrets.toml n'existe.
    # On capture l'erreur pour garder un comportement "optionnel".
    try:
        url = st.secrets.get("MODEL_DOWNLOAD_URL", "")
    except Exception:
        url = ""
    if not url:
        return

    if (MODEL_BASE_DIR / "FINAL_old_best").exists() or (MODEL_BASE_DIR / "FINAL_new_best").exists():
        return

    MODEL_BASE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODEL_BASE_DIR / "models_bundle.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(MODEL_BASE_DIR)
    finally:
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)


def _load_joblib(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def detect_model_dirs() -> dict:
    """Retourne les dossiers de modeles connus s'ils existent."""
    return {
        "FINAL_old_best": MODEL_BASE_DIR / "FINAL_old_best",
        "FINAL_new_best": MODEL_BASE_DIR / "FINAL_new_best",
    }


@st.cache_resource(show_spinner=False)
def load_bundle(version_key: str) -> ModelBundle | None:
    """Charge un bundle de modeles en memoire.

    Args:
        version_key: "FINAL_old_best" ou "FINAL_new_best".

    Returns:
        ModelBundle ou None si introuvable.
    """

    _download_models_if_needed()
    model_dirs = detect_model_dirs()
    base_dir = model_dirs.get(version_key)
    if not base_dir or not base_dir.exists():
        return None

    metadata = _read_json(base_dir / "metadata.json")
    metrics = _read_json(base_dir / "metrics.json")
    best_params = _read_json(base_dir / "best_params.json")

    models = {
        "ww_ucs": _load_joblib(base_dir / "ww_ucs.joblib"),
        "ww_slump": _load_joblib(base_dir / "ww_slump.joblib"),
        "l01_ucs": _load_joblib(base_dir / "l01_ucs.joblib"),
        "l01_slump": _load_joblib(base_dir / "l01_slump.joblib"),
        "ucs": _load_joblib(base_dir / "ucs.joblib"),
        "slump": _load_joblib(base_dir / "slump.joblib"),
    }

    return ModelBundle(
        base_dir=base_dir,
        metadata=metadata,
        metrics=metrics,
        best_params=best_params,
        models=models,
    )


def resolve_tailings_model(bundle: ModelBundle, tailings: str) -> tuple:
    """Retourne les pipelines (slump, ucs) pour un tailings donne.

    Args:
        bundle: ModelBundle charge.
        tailings: "WW" ou "L01".

    Returns:
        (slump_pipe, ucs_pipe)
    """

    tail = tailings.upper().strip()
    if tail == "WW" and bundle.models.get("ww_slump"):
        return bundle.models.get("ww_slump"), bundle.models.get("ww_ucs")
    if tail == "L01" and bundle.models.get("l01_slump"):
        return bundle.models.get("l01_slump"), bundle.models.get("l01_ucs")
    return bundle.models.get("slump"), bundle.models.get("ucs")


def get_features_for_tailings(bundle: ModelBundle, tailings: str) -> dict:
    """Retourne les features (categorical/numeric) depuis metadata.json.

    IMPORTANT:
        Si les features ne sont pas presentes, on renvoie des listes vides
        pour forcer un message d'erreur cote UI.
    """

    features = bundle.metadata.get("features", {})
    tail = tailings.upper().strip()
    if tail in features:
        section = features.get(tail, {})
    else:
        section = features

    cat_cols = set()
    num_cols = set()
    for key in ["slump", "ucs"]:
        part = section.get(key, {}) if isinstance(section, dict) else {}
        for col in part.get("categorical", []) or []:
            cat_cols.add(col)
        for col in part.get("numeric", []) or []:
            num_cols.add(col)

    return {
        "categorical": sorted(cat_cols),
        "numeric": sorted(num_cols),
    }


def get_rmse_for_tailings(bundle: ModelBundle, tailings: str) -> dict:
    """Retourne les RMSE (UCS/Slump) pour le tailings selectionne.

    IMPORTANT:
        - En mode hybride, on privilegie les metriques par Tailings.
        - Sinon, on retombe sur les metriques overall.
    """

    metrics = bundle.metrics or {}
    tail = tailings.upper().strip()

    ucs_rmse = (
        metrics.get("UCS", {})
        .get("Tailings", {})
        .get(tail, {})
        .get("rmse")
    )
    slump_rmse = (
        metrics.get("Slump", {})
        .get("Tailings", {})
        .get(tail, {})
        .get("rmse")
    )

    if ucs_rmse is None:
        ucs_rmse = metrics.get("UCS", {}).get("overall", {}).get("rmse")
    if slump_rmse is None:
        slump_rmse = metrics.get("Slump", {}).get("overall", {}).get("rmse")

    return {"ucs": ucs_rmse, "slump": slump_rmse}
