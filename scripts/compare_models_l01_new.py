# -*- coding: utf-8 -*-
"""Comparatif exhaustif des runs L01 NEW (sans recalcul).

Ce script lit les artefacts déjà produits par un sweep (CSV/JSON),
agrège les résultats, puis exporte un rapport prêt à partager.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:  # matplotlib optionnel
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "run": "run_id",
        "run_key": "run_id",
        "dataset": "dataset_id",
        "searchmode": "search_mode",
        "search_mode": "search_mode",
    }
    df = df.rename(columns=mapping)
    return df


def _infer_dataset_id(run_id: str, path_hint: str = "") -> str:
    run_id = (run_id or "").lower()
    hint = path_hint.lower()
    if "new" in run_id or "new" in hint:
        return "new"
    if "old" in run_id or "old" in hint:
        return "old"
    return ""


def _compute_run_score(row: pd.Series) -> float | None:
    if pd.notna(row.get("run_score")):
        return float(row["run_score"])
    if pd.notna(row.get("final_score")):
        return float(row["final_score"])
    try:
        ucs_r2_l01 = float(row.get("ucs_r2_l01"))
        ucs_rmse_l01 = float(row.get("ucs_rmse_l01"))
        ucs_r2_ww = float(row.get("ucs_r2_ww"))
        slump_rmse = float(row.get("slump_rmse_overall"))
    except Exception:
        return None
    return (ucs_r2_l01) - (ucs_rmse_l01 / 1000.0) + 0.20 * (ucs_r2_ww) - 0.05 * (slump_rmse / 10.0)


def _make_config_short(row: pd.Series) -> str:
    parts = [
        f"WW:{row.get('ww_ucs_model', '')}",
        f"L01:{row.get('l01_ucs_model', '')}",
        f"tr:{row.get('ww_ucs_transform', '')}/{row.get('l01_ucs_transform', '')}",
        f"out:{row.get('ww_ucs_outliers', '')}/{row.get('l01_ucs_outliers', '')}",
        f"tune:{row.get('ww_tune', '')}/{row.get('l01_tune', '')}",
        f"seed:{row.get('seed', '')}",
        f"mode:{row.get('search_mode', '')}",
    ]
    return "|".join(parts)


def _load_from_sweep_runs(sweep_dir: Path) -> pd.DataFrame:
    files = list(sweep_dir.rglob("sweep_runs.csv"))
    # Si un dossier "new" existe, on privilégie ce fichier pour éviter les doublons.
    new_files = [f for f in files if "new" in [p.lower() for p in f.parts]]
    if new_files:
        files = new_files
    frames = []
    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception:
            continue
        df = _normalize_columns(df)
        df["source_file"] = str(file)
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _extract_from_metrics(metrics_path: Path) -> dict[str, Any]:
    metrics = _read_json(metrics_path)
    meta = metrics.get("meta", {})
    run_id = meta.get("run_id") or metrics_path.parent.name
    dataset_id = _infer_dataset_id(run_id, str(metrics_path))
    optimisation = metrics.get("optimisation", {})
    models = meta.get("models", {})

    def _get_model_info(scope: str, target: str) -> dict[str, Any]:
        return models.get(scope, {}).get(target, {})

    ww_ucs = _get_model_info("WW", "ucs")
    l01_ucs = _get_model_info("L01", "ucs")
    ww_slump = _get_model_info("WW", "slump")
    l01_slump = _get_model_info("L01", "slump")

    return {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "seed": meta.get("seed"),
        "search_mode": optimisation.get("search_mode"),
        "fit_mode": meta.get("fit_mode"),
        "ww_ucs_model": ww_ucs.get("name"),
        "ww_ucs_transform": ww_ucs.get("transform"),
        "ww_ucs_outliers": ww_ucs.get("outliers"),
        "ww_tune": ww_ucs.get("tune"),
        "l01_ucs_model": l01_ucs.get("name"),
        "l01_ucs_transform": l01_ucs.get("transform"),
        "l01_ucs_outliers": l01_ucs.get("outliers"),
        "l01_tune": l01_ucs.get("tune"),
        "ww_slump_model": ww_slump.get("name"),
        "ww_slump_tune": ww_slump.get("tune"),
        "l01_slump_model": l01_slump.get("name"),
        "l01_slump_tune": l01_slump.get("tune"),
        "n_samples": optimisation.get("n_samples"),
        "slump_min": optimisation.get("slump_min"),
        "ucs_min": optimisation.get("ucs_min"),
        "pass_rate_pct": optimisation.get("pass_rate_pct"),
        "ucs_r2_l01": metrics.get("UCS", {}).get("Tailings", {}).get("L01", {}).get("r2"),
        "ucs_rmse_l01": metrics.get("UCS", {}).get("Tailings", {}).get("L01", {}).get("rmse"),
        "ucs_r2_ww": metrics.get("UCS", {}).get("Tailings", {}).get("WW", {}).get("r2"),
        "ucs_rmse_ww": metrics.get("UCS", {}).get("Tailings", {}).get("WW", {}).get("rmse"),
        "slump_r2_overall": metrics.get("Slump", {}).get("overall", {}).get("r2"),
        "slump_rmse_overall": metrics.get("Slump", {}).get("overall", {}).get("rmse"),
    }


def _load_from_metrics(sweep_dir: Path) -> pd.DataFrame:
    files = list(sweep_dir.rglob("metrics.json"))
    rows = []
    for file in files:
        row = _extract_from_metrics(file)
        if row.get("run_id"):
            row["source_file"] = str(file)
            rows.append(row)
    return pd.DataFrame(rows)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _write_markdown(out_path: Path, df: pd.DataFrame) -> None:
    total_runs = len(df)
    valid_runs = df["run_score"].notna().sum()
    best_final = df.sort_values("run_score", ascending=False).head(1)
    best_r2 = df.sort_values("ucs_r2_l01", ascending=False).head(1)
    best_rmse = df.sort_values("ucs_rmse_l01", ascending=True).head(1)

    def _row_to_text(row: pd.Series) -> str:
        return (
            f"- run_id: {row.get('run_id')}\n"
            f"  - L01 model: {row.get('l01_ucs_model')}\n"
            f"  - WW model: {row.get('ww_ucs_model')}\n"
            f"  - R² L01: {row.get('ucs_r2_l01')}\n"
            f"  - RMSE L01: {row.get('ucs_rmse_l01')}\n"
            f"  - score: {row.get('run_score')}\n"
        )

    top10 = df.sort_values("run_score", ascending=False).head(10)
    top10_table = top10[
        ["run_id", "l01_ucs_model", "ww_ucs_model", "ucs_r2_l01", "ucs_rmse_l01", "run_score"]
    ]

    model_counts = (
        top10["l01_ucs_model"].value_counts().rename_axis("l01_model").reset_index(name="count")
    )

    content = [
        "# Comparatif des runs sweep L01 NEW",
        "",
        f"- Nombre total de runs: {total_runs}",
        f"- Runs valides (score calculable): {valid_runs}",
        "",
        "## Meilleur run (final_score)",
        _row_to_text(best_final.iloc[0]) if not best_final.empty else "Aucun run disponible.",
        "## Meilleur run (R² L01)",
        _row_to_text(best_r2.iloc[0]) if not best_r2.empty else "Aucun run disponible.",
        "## Meilleur run (RMSE L01)",
        _row_to_text(best_rmse.iloc[0]) if not best_rmse.empty else "Aucun run disponible.",
        "",
        "## Top 10 (final_score)",
        _df_to_markdown(top10_table),
        "",
        "## Lecture rapide (modèles L01 les plus fréquents dans le top 10)",
        _df_to_markdown(model_counts) if not model_counts.empty else "Pas de données.",
        "",
        "Commande:",
        "```bash",
        "python scripts/compare_models_l01_new.py --sweep-dir outputs/sweep_new_3h --out-dir outputs/reports/L01_new",
        "```",
    ]
    out_path.write_text("\n".join(content), encoding="utf-8")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Construit un tableau Markdown sans dépendance externe."""
    if df.empty:
        return "Aucune donnée."
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            val = row.get(col)
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparatif runs L01 NEW (sweep).")
    parser.add_argument("--sweep-dir", required=True, help="Dossier du sweep (ex: outputs/sweep_new_3h)")
    parser.add_argument("--out-dir", required=True, help="Dossier de sortie (ex: outputs/reports/L01_new)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_from_sweep_runs(sweep_dir)
    if df.empty:
        df = _load_from_metrics(sweep_dir)

    if df.empty:
        raise SystemExit("Aucun run détecté dans le dossier de sweep.")

    df = _normalize_columns(df)
    df["dataset_id"] = df.get("dataset_id")
    df["dataset_id"] = df["dataset_id"].fillna(df["run_id"].apply(_infer_dataset_id))
    df = df[df["dataset_id"].str.lower() == "new"].copy()
    if "run_id" in df.columns:
        df = df.drop_duplicates(subset=["run_id"])

    # S'assurer que les colonnes principales existent.
    base_cols = [
        "dataset_id",
        "run_id",
        "seed",
        "search_mode",
        "fit_mode",
        "ww_ucs_model",
        "ww_ucs_transform",
        "ww_ucs_outliers",
        "ww_tune",
        "l01_ucs_model",
        "l01_ucs_transform",
        "l01_ucs_outliers",
        "l01_tune",
        "ww_slump_model",
        "ww_slump_tune",
        "l01_slump_model",
        "l01_slump_tune",
        "n_samples",
        "slump_min",
        "ucs_min",
        "ucs_r2_l01",
        "ucs_rmse_l01",
        "ucs_r2_ww",
        "ucs_rmse_ww",
        "slump_r2_overall",
        "slump_rmse_overall",
        "pass_rate_pct",
        "run_score",
    ]
    df = _ensure_columns(df, base_cols)
    # Harmonise les colonnes "mean_*" si besoin pour les exports.
    if "mean_ucs_r2_l01" not in df.columns:
        df["mean_ucs_r2_l01"] = df["ucs_r2_l01"]
    if "mean_ucs_rmse_l01" not in df.columns:
        df["mean_ucs_rmse_l01"] = df["ucs_rmse_l01"]
    if "mean_ucs_r2_ww" not in df.columns:
        df["mean_ucs_r2_ww"] = df["ucs_r2_ww"]
    if "mean_ucs_rmse_ww" not in df.columns:
        df["mean_ucs_rmse_ww"] = df["ucs_rmse_ww"]

    df["run_score"] = df.apply(lambda r: _compute_run_score(r), axis=1)
    df["config_short"] = df.apply(_make_config_short, axis=1)

    # Ranks
    df["rank_final_score"] = df["run_score"].rank(ascending=False, method="dense")
    df["rank_ucs_r2_l01"] = df["ucs_r2_l01"].rank(ascending=False, method="dense")
    df["rank_ucs_rmse_l01"] = df["ucs_rmse_l01"].rank(ascending=True, method="dense")

    # Exports CSV
    runs_csv = out_dir / "runs_L01_new.csv"
    df.to_csv(runs_csv, index=False)

    # Excel avec plusieurs onglets
    xlsx_path = out_dir / "comparatif_L01_new.xlsx"
    top_final = df.sort_values("run_score", ascending=False).head(20)
    top_r2 = df.sort_values("ucs_r2_l01", ascending=False).head(20)
    top_rmse = df.sort_values("ucs_rmse_l01", ascending=True).head(20)

    summary_l01 = df.groupby("l01_ucs_model")[["ucs_r2_l01", "ucs_rmse_l01", "run_score"]].agg(
        ["mean", "median"]
    )
    summary_l01.columns = [f"{col[0]}_{col[1]}" for col in summary_l01.columns]
    summary_l01 = summary_l01.reset_index()

    summary_ww = df.groupby("ww_ucs_model")[["ucs_r2_ww", "ucs_rmse_ww", "run_score"]].agg(
        ["mean", "median"]
    )
    summary_ww.columns = [f"{col[0]}_{col[1]}" for col in summary_ww.columns]
    summary_ww = summary_ww.reset_index()

    with pd.ExcelWriter(xlsx_path) as writer:
        df.to_excel(writer, index=False, sheet_name="ALL_RUNS")
        top_final.to_excel(writer, index=False, sheet_name="TOP_FINAL_SCORE")
        top_r2.to_excel(writer, index=False, sheet_name="TOP_L01_R2")
        top_rmse.to_excel(writer, index=False, sheet_name="LOWEST_L01_RMSE")
        summary_l01.to_excel(writer, index=False, sheet_name="SUMMARY_BY_L01_MODEL")
        summary_ww.to_excel(writer, index=False, sheet_name="SUMMARY_BY_WW_MODEL")

    # Markdown résumé
    md_path = out_dir / "comparatif_L01_new.md"
    _write_markdown(md_path, df)

    # Figures optionnelles
    if plt is not None:
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        scatter_path = fig_dir / "scatter_rmse_vs_r2_l01.png"
        plt.figure(figsize=(6, 4))
        plt.scatter(df["ucs_rmse_l01"], df["ucs_r2_l01"], alpha=0.7)
        plt.xlabel("RMSE L01 (kPa)")
        plt.ylabel("R² L01")
        plt.title("RMSE vs R² (L01 NEW)")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=160)
        plt.close()

        bar_path = fig_dir / "bar_mean_r2_by_l01_model.png"
        model_means = df.groupby("l01_ucs_model")["ucs_r2_l01"].mean().sort_values(ascending=False)
        plt.figure(figsize=(6, 4))
        model_means.plot(kind="bar")
        plt.ylabel("R² moyen (L01)")
        plt.title("R² moyen par modèle L01")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=160)
        plt.close()

    print(f"Rapport L01 NEW exporté dans: {out_dir}")


if __name__ == "__main__":
    main()
