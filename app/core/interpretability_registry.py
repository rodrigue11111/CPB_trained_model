# -*- coding: utf-8 -*-
"""Acces robuste aux sorties d'interpretabilite (CSV/PNG/rapport)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_BASE_DIR = Path("outputs/interpretability")


def list_interpretability_runs(base_dir: str | Path = DEFAULT_BASE_DIR) -> list[Path]:
    """Liste les dossiers d'analyses disponibles."""
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)


def detect_files(run_dir: str | Path) -> dict[str, Any]:
    """Detecte les fichiers connus dans un dossier d'analyse."""
    run_path = Path(run_dir)
    files: dict[str, Any] = {
        "report": None,
        "importance_csv": None,
        "importance_png": None,
        "binder_csv": None,
        "binder_png": None,
        "pdp_csvs": [],
        "pdp_pngs": [],
        "interaction_csvs": [],
        "interaction_pngs": [],
    }

    if not run_path.exists():
        return files

    for path in run_path.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        stem = path.stem.lower()

        if name == "rapport.md":
            files["report"] = path
            continue

        if name == "importance_permutation.csv":
            files["importance_csv"] = path
            continue
        if name == "importance_permutation.png":
            files["importance_png"] = path
            continue
        if name == "importance_model.csv" and files["importance_csv"] is None:
            files["importance_csv"] = path
            continue
        if name == "importance_model.png" and files["importance_png"] is None:
            files["importance_png"] = path
            continue

        if name == "binder_effect.csv":
            files["binder_csv"] = path
            continue
        if name == "binder_effect.png":
            files["binder_png"] = path
            continue

        if name.endswith(".csv") and stem.startswith("pdp_") and not stem.startswith("pdp2d_"):
            files["pdp_csvs"].append(path)
            continue
        if name.endswith(".png") and stem.startswith("pdp_") and not stem.startswith("pdp2d_"):
            files["pdp_pngs"].append(path)
            continue

        if name.endswith(".csv") and (stem.startswith("interaction_") or stem.startswith("pdp2d_")):
            files["interaction_csvs"].append(path)
            continue
        if name.endswith(".png") and (stem.startswith("interaction_") or stem.startswith("pdp2d_")):
            files["interaction_pngs"].append(path)
            continue

    files["pdp_csvs"] = sorted(files["pdp_csvs"], key=lambda p: p.name)
    files["pdp_pngs"] = sorted(files["pdp_pngs"], key=lambda p: p.name)
    files["interaction_csvs"] = sorted(files["interaction_csvs"], key=lambda p: p.name)
    files["interaction_pngs"] = sorted(files["interaction_pngs"], key=lambda p: p.name)
    return files


def _humanize_token(token: str) -> str:
    label = token.replace("_", " ")
    label = label.replace("pct", "%")
    label = label.replace(" um", " \u00b5m")
    return label


def _token_from_pdp_stem(stem: str) -> str:
    return stem[len("pdp_"):]


def _token_from_interaction_stem(stem: str) -> str:
    if stem.startswith("interaction_"):
        base = stem[len("interaction_"):]
    else:
        base = stem[len("pdp2d_"):]
    if "__" in base:
        left, right = base.split("__", 1)
        return f"{left}__{right}"
    if "_x_" in base:
        left, right = base.split("_x_", 1)
        return f"{left}__{right}"
    return base


def parse_pdp_files(run_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Retourne un mapping {token: {label, csv?, png?}}."""
    files = detect_files(run_dir)
    output: dict[str, dict[str, Path]] = {}
    for path in files["pdp_csvs"]:
        token = _token_from_pdp_stem(path.stem)
        output.setdefault(token, {"label": _humanize_token(token)})["csv"] = path
    for path in files["pdp_pngs"]:
        token = _token_from_pdp_stem(path.stem)
        output.setdefault(token, {"label": _humanize_token(token)})["png"] = path
    return output


def parse_interaction_files(run_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Retourne un mapping {token: {label, csv?, png?}}."""
    files = detect_files(run_dir)
    output: dict[str, dict[str, Path]] = {}
    for path in files["interaction_csvs"]:
        token = _token_from_interaction_stem(path.stem)
        label = _humanize_token(token.replace("__", " vs "))
        output.setdefault(token, {"label": label})["csv"] = path
    for path in files["interaction_pngs"]:
        token = _token_from_interaction_stem(path.stem)
        label = _humanize_token(token.replace("__", " vs "))
        output.setdefault(token, {"label": label})["png"] = path
    return output
