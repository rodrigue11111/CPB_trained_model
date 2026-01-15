"""Comparaison des runs et synthese des resultats.

Ce script est un comparateur simple (legacy) base sur metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Compare les runs et genere un resume."
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Dossier de sortie principal.",
    )
    return parser.parse_args(argv)


def _safe_get(data: dict, *keys, default=np.nan):
    """Acces securise a un champ avec valeur par defaut."""
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _score_row(row: pd.Series) -> float:
    """Score simple pour classer les runs."""
    return (
        row["ucs_r2_L01"]
        - (row["ucs_rmse_L01"] / 1000.0)
        + 0.15 * row["ucs_r2_WW"]
        - 0.05 * (row["slump_rmse_global"] / 10.0)
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entree principal du comparateur."""
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"

    rows = []
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            status_path = run_dir / "status.json"
            metrics_path = run_dir / "metrics.json"
            if not status_path.exists() or not metrics_path.exists():
                continue

            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "ok":
                continue

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            meta_models = metrics.get("meta", {}).get("models", {})

            def _model_name(field: str) -> str:
                # Mode combine ou hybride: on formate un nom lisible.
                if field in meta_models:
                    return meta_models.get(field, {}).get("name", "")
                ww = (
                    meta_models.get("WW", {})
                    .get(field, {})
                    .get("name", "")
                )
                l01 = (
                    meta_models.get("L01", {})
                    .get(field, {})
                    .get("name", "")
                )
                if ww and l01:
                    return f"WW:{ww}|L01:{l01}"
                return ww or l01

            row = {
                "run_id": run_dir.name,
                "model_slump": _model_name("slump"),
                "model_ucs": _model_name("ucs"),
                "slump_r2_global": _safe_get(
                    metrics, "Slump", "overall", "r2"
                ),
                "slump_rmse_global": _safe_get(
                    metrics, "Slump", "overall", "rmse"
                ),
                "ucs_r2_global": _safe_get(
                    metrics, "UCS", "overall", "r2"
                ),
                "ucs_rmse_global": _safe_get(
                    metrics, "UCS", "overall", "rmse"
                ),
                "ucs_r2_L01": _safe_get(
                    metrics, "UCS", "Tailings", "L01", "r2"
                ),
                "ucs_rmse_L01": _safe_get(
                    metrics, "UCS", "Tailings", "L01", "rmse"
                ),
                "ucs_r2_WW": _safe_get(
                    metrics, "UCS", "Tailings", "WW", "r2"
                ),
                "pass_rate": _safe_get(
                    metrics, "optimisation", "pass_rate_pct"
                ),
                "slump_min": _safe_get(
                    metrics, "optimisation", "slump_min"
                ),
                "ucs_min": _safe_get(metrics, "optimisation", "ucs_min"),
                "n_samples": _safe_get(
                    metrics, "optimisation", "n_samples"
                ),
                "search_mode": _safe_get(
                    metrics, "optimisation", "search_mode"
                ),
                "seed": _safe_get(metrics, "meta", "seed"),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("Aucun run valide trouve.")
        return 1

    df["score"] = df.apply(_score_row, axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_summary.csv"
    xlsx_path = out_dir / "sweep_summary.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    def _print_top(title: str, sort_col: str) -> None:
        print(title)
        top = df.sort_values(by=sort_col, ascending=False).head(5)
        for _, row in top.iterrows():
            value = row.get(sort_col, np.nan)
            print(f"  {row['run_id']}: {value}")

    _print_top("Top 5 par score", "score")
    _print_top("Top 5 par UCS R2 L01", "ucs_r2_L01")
    print("Top 5 par UCS RMSE L01 (plus bas)")
    top_rmse = df.sort_values(by="ucs_rmse_L01", ascending=True).head(5)
    for _, row in top_rmse.iterrows():
        value = row.get("ucs_rmse_L01", np.nan)
        print(f"  {row['run_id']}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
