# -*- coding: utf-8 -*-
"""Chargement et calculs autour des formules UCS (L01 NEW)."""

from __future__ import annotations

import json
import hashlib
import tempfile
import zipfile
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from app.core import data_profiles


def find_formulas_dirs(base_dirs: list[str]) -> dict[str, Path]:
    """Detecte les dossiers de formules disponibles."""
    results: dict[str, Path] = {}
    for base in base_dirs:
        base_path = Path(base)
        if not base_path.exists():
            continue
        # Cas: le dossier courant est deja un dossier de formules.
        if (base_path / "classic_linear_coefficients.csv").exists():
            results[base_path.name] = base_path
            continue
        for child in base_path.iterdir():
            if not child.is_dir():
                continue
            if (child / "classic_linear_coefficients.csv").exists():
                results[child.name] = child
    return results


def _download_zip(url: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "formulas_bundle.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return target_dir


def load_from_url(url: str) -> Path | None:
    """Telecharge un zip et retourne le dossier extrait."""
    if not url:
        return None
    key = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    target_dir = Path(tempfile.gettempdir()) / f"cpb_formulas_{key}"
    if target_dir.exists():
        return target_dir
    return _download_zip(url, target_dir)


def load_text_file_safe(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def load_metrics_safe(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_classic_equation(formulas_dir: Path) -> dict[str, Any] | None:
    coef_path = formulas_dir / "classic_linear_coefficients.csv"
    metrics_path = formulas_dir / "classic_linear_metrics.json"
    equation_path = formulas_dir / "classic_linear_equation.md"

    if not coef_path.exists():
        return None

    coef_df = pd.read_csv(coef_path)
    if "term_type" not in coef_df.columns:
        def _infer_type(name: str) -> str:
            if str(name).lower() == "intercept":
                return "intercept"
            if str(name).startswith("Binder_"):
                return "categorical"
            return "numeric"
        coef_df["term_type"] = coef_df["term_name"].apply(_infer_type)

    intercept_row = coef_df[coef_df["term_type"] == "intercept"]
    intercept = float(intercept_row["coefficient"].iloc[0]) if not intercept_row.empty else 0.0
    terms_df = coef_df[coef_df["term_type"] != "intercept"].copy()

    numeric_terms = terms_df[terms_df["term_type"] == "numeric"]["term_name"].tolist()
    binder_terms = terms_df[terms_df["term_type"] == "categorical"]["term_name"].tolist()
    binder_values = []
    for term in binder_terms:
        if term.startswith("Binder_"):
            binder_values.append(term.replace("Binder_", ""))
    binder_values = sorted(set(binder_values)) or ["GUL", "20G80S"]

    metrics = load_metrics_safe(metrics_path)
    best_model_key = metrics.get("best_model")
    best_metrics = metrics.get("metrics", {}).get(best_model_key, {}) if best_model_key else {}

    return {
        "intercept": intercept,
        "terms": terms_df.to_dict(orient="records"),
        "numeric_features": numeric_terms,
        "binder_values": binder_values,
        "metrics": best_metrics,
        "raw_metrics": metrics,
        "equation_text": load_text_file_safe(equation_path),
        "coefficients_path": coef_path,
        "metrics_path": metrics_path,
        "equation_path": equation_path,
    }


def compute_ucs_classic(equation: dict, inputs: dict) -> tuple[float, pd.DataFrame]:
    """Calcule UCS et contributions terme a terme."""
    intercept = float(equation.get("intercept", 0.0))
    binder_value = str(inputs.get("Binder", ""))

    rows = [
        {
            "terme": "intercept",
            "valeur": "",
            "coefficient": intercept,
            "contribution": intercept,
        }
    ]
    total = intercept

    for term in equation.get("terms", []):
        name = term.get("term_name")
        coef = float(term.get("coefficient", 0.0))
        term_type = term.get("term_type")
        if term_type == "categorical":
            if name.startswith("Binder_"):
                label = name.replace("Binder_", "")
                contrib = coef if label == binder_value else 0.0
                rows.append(
                    {
                        "terme": f"Binder={label}",
                        "valeur": 1 if label == binder_value else 0,
                        "coefficient": coef,
                        "contribution": contrib,
                    }
                )
                total += contrib
            continue

        value = inputs.get(name)
        try:
            val = float(value)
        except (TypeError, ValueError):
            val = 0.0
        contrib = coef * val
        rows.append(
            {
                "terme": name,
                "valeur": val,
                "coefficient": coef,
                "contribution": contrib,
            }
        )
        total += contrib

    df = pd.DataFrame(rows)
    return total, df


def check_out_of_distribution(inputs: dict, profile_key: str = "L01_NEW") -> list[str]:
    """Retourne des warnings si valeurs hors distributions connues."""
    profile = data_profiles.load_profile(profile_key)
    if profile is None:
        return ["Plage inconnue sur le cloud (profil indisponible)."]

    def _normalize(name: str) -> str:
        # Nettoyage simple pour corriger les artefacts d'encodage (ex: Âµ).
        return str(name).replace("Â", "").strip()

    normalized_map = {}
    for col in profile.numeric_stats.keys():
        normalized_map[_normalize(col)] = col

    warnings = []
    for name, value in inputs.items():
        if name in {"Binder", "Tailings"}:
            continue
        lookup_name = normalized_map.get(_normalize(name), name)
        feature_profile = data_profiles.get_feature_profile(profile, lookup_name)
        if not feature_profile:
            continue
        result = data_profiles.ood_level(value, feature_profile)
        bounds_text = data_profiles.format_bounds(feature_profile)
        if result["level"] == "out":
            msg = f"{name}: hors domaine d'entra\u00eenement."
            if bounds_text:
                msg += f" ({bounds_text})"
            warnings.append(msg)
        elif result["level"] == "warn":
            msg = f"{name}: proche des bornes d'entra\u00eenement."
            if bounds_text:
                msg += f" ({bounds_text})"
            warnings.append(msg)
    return warnings
