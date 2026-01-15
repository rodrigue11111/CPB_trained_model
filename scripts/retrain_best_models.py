"""Relance un entrainement final a partir des meilleurs sweeps.

But: reproduire exactement les meilleurs runs sans re-tuning aleatoire,
en reutilisant best_params.json si disponible.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import cli


@dataclass(frozen=True)
class DatasetSpec:
    """Specification d'un scenario dataset (paths + feuilles)."""
    dataset_id: str
    l01_path: str
    l01_sheet: str | None
    ww_path: str
    ww_sheet: str | None


DEFAULT_OLD = DatasetSpec(
    dataset_id="old",
    l01_path="data/L01-Optimisation.xlsx",
    l01_sheet="Feuil1 (2)",
    ww_path="data/WW-Optimisation.xlsx",
    ww_sheet="Feuil3 (3)",
)

DEFAULT_NEW = DatasetSpec(
    dataset_id="new",
    l01_path="data/L01-dataset.xlsx",
    l01_sheet=None,
    ww_path="data/WW-Optimisation.xlsx",
    ww_sheet="Feuil3 (3)",
)

CONFIG_FIELDS = [
    "dataset_id",
    "search_mode",
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
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI du script de retrain."""
    parser = argparse.ArgumentParser(
        description="Relance des entrainements finaux a partir des sweeps."
    )
    parser.add_argument(
        "--sweep-old",
        default="",
        help="Dossier de sweep pour L01_old (optionnel).",
    )
    parser.add_argument(
        "--sweep-new",
        default="",
        help="Dossier de sweep pour L01_new (optionnel).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aleatoire pour l'entrainement final.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50_000,
        help="Nombre de candidats pour l'optimisation finale.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/final_models",
        help="Dossier de sortie des modeles finaux.",
    )
    return parser.parse_args(argv)


def _read_csv_header(path: Path) -> list[str]:
    """Lit uniquement la premiere ligne d'un CSV."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _find_csv_with_column(root: Path, column: str) -> Path | None:
    """Cherche recursivement un CSV contenant une colonne donnee."""
    for path in root.rglob("*.csv"):
        try:
            header = _read_csv_header(path)
        except Exception:
            continue
        if column in header:
            return path
    return None


def _find_best_summary_csv(sweep_dir: Path) -> Path | None:
    """Trouve un CSV de resume contenant final_score."""
    for name in ("sweep_runs.csv", "sweep_results.csv"):
        for path in sweep_dir.rglob(name):
            header = _read_csv_header(path)
            if "final_score" in header:
                return path
    return _find_csv_with_column(sweep_dir, "final_score")


def _safe_float(value) -> float:
    """Conversion robuste en float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _row_matches_config(row: pd.Series, config: dict) -> bool:
    """Verifie si une ligne de CSV correspond a une config."""
    for key in CONFIG_FIELDS:
        if key not in row or key not in config:
            continue
        if str(row[key]) != str(config[key]):
            return False
    return True


def _build_run_id(config: dict, seed: int) -> str:
    """Construit un run_id stable (meme format que sweep.py)."""
    return (
        f"{config['dataset_id']}_seed{seed}_{config['search_mode']}"
        f"_WW{config['ww_ucs_model']}-{config['ww_ucs_transform']}"
        f"-{config['ww_ucs_outliers']}-t{config['ww_tune']}"
        f"_L01{config['l01_ucs_model']}-{config['l01_ucs_transform']}"
        f"-{config['l01_ucs_outliers']}-t{config['l01_tune']}"
    )


def _config_from_row(row: dict, dataset_id: str) -> dict:
    """Normalise une ligne de CSV en config utilisable."""
    config = {"dataset_id": dataset_id}
    for key in CONFIG_FIELDS:
        if key not in row:
            continue
        value = row[key]
        if pd.isna(value) or value == "":
            continue
        if key.endswith("_tune"):
            normalized = str(value).strip().lower()
            config[key] = "true" if normalized in {"true", "1", "yes"} else "false"
        elif key in {
            "search_mode",
            "ww_ucs_model",
            "ww_ucs_transform",
            "ww_ucs_outliers",
            "l01_ucs_model",
            "l01_ucs_transform",
            "l01_ucs_outliers",
            "ww_slump_model",
            "l01_slump_model",
        }:
            config[key] = str(value).strip().lower()
        else:
            config[key] = str(value)
    return config


def _locate_best_config(
    sweep_dir: Path, dataset_id: str
) -> tuple[dict, Path | None]:
    """Recupere la meilleure config (max final_score) depuis un sweep."""
    # On cherche un CSV de synthese avec final_score pour identifier le meilleur.
    summary_path = _find_best_summary_csv(sweep_dir)
    if not summary_path:
        return {}, None

    df = pd.read_csv(summary_path)
    if "final_score" not in df.columns:
        return {}, summary_path

    if "dataset_id" in df.columns:
        df = df[df["dataset_id"].astype(str).str.lower() == dataset_id.lower()]

    df["final_score"] = df["final_score"].apply(_safe_float)
    df = df[df["final_score"].notna()]
    if df.empty:
        return {}, summary_path

    best_row = df.sort_values("final_score", ascending=False).iloc[0]
    return _config_from_row(best_row.to_dict(), dataset_id), summary_path


def _locate_run_dir(
    sweep_dir: Path,
    dataset_id: str,
    config: dict,
    seed: int,
) -> Path | None:
    """Localise le dossier de run associe a une config."""
    runs_dir = sweep_dir / dataset_id / "runs"
    if runs_dir.exists():
        run_id = config.get("run_id")
        if run_id:
            candidate = runs_dir / run_id
            if candidate.exists():
                return candidate

    runs_csv = sweep_dir / dataset_id / "sweep_runs.csv"
    if runs_csv.exists():
        df_runs = pd.read_csv(runs_csv)
        if "dataset_id" in df_runs.columns:
            df_runs = df_runs[
                df_runs["dataset_id"].astype(str).str.lower()
                == dataset_id.lower()
            ]
        for key in CONFIG_FIELDS:
            if key in df_runs.columns and key in config:
                df_runs = df_runs[
                    df_runs[key].astype(str) == str(config[key])
                ]
        if not df_runs.empty and "run_id" in df_runs.columns:
            df_runs["run_score"] = df_runs["run_score"].apply(_safe_float)
            best = df_runs.sort_values(
                "run_score", ascending=False
            ).iloc[0]
            candidate = runs_dir / str(best["run_id"])
            if candidate.exists():
                return candidate

    if runs_dir.exists():
        expected = _build_run_id(config, seed)
        candidate = runs_dir / expected
        if candidate.exists():
            return candidate
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            if config.get("search_mode", "") not in name:
                continue
            ww_key = (
                f"WW{config.get('ww_ucs_model')}-"
                f"{config.get('ww_ucs_transform')}-"
                f"{config.get('ww_ucs_outliers')}-t{config.get('ww_tune')}"
            )
            l01_key = (
                f"L01{config.get('l01_ucs_model')}-"
                f"{config.get('l01_ucs_transform')}-"
                f"{config.get('l01_ucs_outliers')}-t{config.get('l01_tune')}"
            )
            if ww_key in name and l01_key in name:
                return run_dir

    return None


def _load_best_params(path: Path) -> dict:
    """Charge best_params.json et normalise sa structure."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    if "WW" in data or "L01" in data:
        output = {"WW": {}, "L01": {}}
        for tail in ("WW", "L01"):
            section = data.get(tail, {})
            if isinstance(section, dict):
                output[tail] = {
                    "ucs": section.get("ucs", {}) or {},
                    "slump": section.get("slump", {}) or {},
                }
        return output
    if "ucs" in data or "slump" in data:
        return {
            "WW": {
                "ucs": data.get("ucs", {}) or {},
                "slump": data.get("slump", {}) or {},
            },
            "L01": {
                "ucs": data.get("ucs", {}) or {},
                "slump": data.get("slump", {}) or {},
            },
        }
    return {
        "WW": {"ucs": data, "slump": {}},
        "L01": {"ucs": data, "slump": {}},
    }


def _write_fixed_params(
    base_dir: Path, label: str, params: dict
) -> Path | None:
    """Ecrit un JSON de params fixes pour un modele."""
    if not params:
        return None
    # Les params fixes garantissent la reproductibilite exacte du sweep.
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{label}_fixed_params.json"
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return path


def _fallback_config(dataset_id: str) -> dict:
    """Config par defaut si aucun sweep exploitable."""
    return {
        "dataset_id": dataset_id,
        "search_mode": "bootstrap",
        "ww_ucs_model": "gbr",
        "ww_ucs_transform": "none",
        "ww_ucs_outliers": "none",
        "ww_tune": "false",
        "l01_ucs_model": "et",
        "l01_ucs_transform": "none",
        "l01_ucs_outliers": "none",
        "l01_tune": "true",
        "ww_slump_model": "gbr",
        "ww_slump_tune": "false",
        "l01_slump_model": "gbr",
        "l01_slump_tune": "true",
    }


def _build_cli_args(
    spec: DatasetSpec,
    config: dict,
    seed: int,
    n_samples: int,
    out_dir: Path,
    run_id: str,
    fixed_params: dict | None,
) -> list[str]:
    """Construit les arguments pour appeler src.cli.main() en Python."""
    # hybrid_by_tailings entraine des pipelines distincts pour WW et L01.
    args: list[str] = [
        "--fit-mode",
        "hybrid_by_tailings",
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--out-dir",
        str(out_dir),
        "--l01",
        spec.l01_path,
        "--ww",
        spec.ww_path,
        "--slump-min",
        "70",
        "--ucs-min",
        "900",
        "--n-samples",
        str(n_samples),
        "--search-mode",
        config.get("search_mode", "bootstrap"),
        "--ww-model",
        config.get("ww_ucs_model", "gbr"),
        "--ww-ucs-transform",
        config.get("ww_ucs_transform", "none"),
        "--ww-ucs-outliers",
        config.get("ww_ucs_outliers", "none"),
        "--ww-tune",
        config.get("ww_tune", "false"),
        "--l01-model",
        config.get("l01_ucs_model", "et"),
        "--l01-ucs-transform",
        config.get("l01_ucs_transform", "none"),
        "--l01-ucs-outliers",
        config.get("l01_ucs_outliers", "none"),
        "--l01-tune",
        config.get("l01_tune", "true"),
        "--ww-slump-model",
        config.get("ww_slump_model", "gbr"),
        "--ww-slump-tune",
        config.get("ww_slump_tune", "false"),
        "--l01-slump-model",
        config.get("l01_slump_model", "gbr"),
        "--l01-slump-tune",
        config.get("l01_slump_tune", "true"),
        "--save-models",
        "true",
        "--models-dir",
        str(out_dir / run_id),
        "--out",
        str(out_dir / f"Top_Recipes_FINAL_{config['dataset_id']}.xlsx"),
    ]

    if spec.l01_sheet:
        args.extend(["--sheet-l01", spec.l01_sheet])
    if spec.ww_sheet:
        args.extend(["--sheet-ww", spec.ww_sheet])

    if fixed_params:
        # On ecrit les params fixes sur disque pour reutiliser la CLI.
        fixed_dir = out_dir / run_id / "fixed_params"
        ww_ucs = _write_fixed_params(
            fixed_dir, "ww_ucs", fixed_params.get("WW", {}).get("ucs", {})
        )
        l01_ucs = _write_fixed_params(
            fixed_dir, "l01_ucs", fixed_params.get("L01", {}).get("ucs", {})
        )
        ww_slump = _write_fixed_params(
            fixed_dir, "ww_slump", fixed_params.get("WW", {}).get("slump", {})
        )
        l01_slump = _write_fixed_params(
            fixed_dir,
            "l01_slump",
            fixed_params.get("L01", {}).get("slump", {}),
        )

        if ww_ucs:
            args.extend(["--ww-ucs-fixed-params", str(ww_ucs)])
        if l01_ucs:
            args.extend(["--l01-ucs-fixed-params", str(l01_ucs)])
        if ww_slump:
            args.extend(["--ww-slump-fixed-params", str(ww_slump)])
        if l01_slump:
            args.extend(["--l01-slump-fixed-params", str(l01_slump)])

    return args


def _print_metrics_summary(run_dir: Path, label: str) -> None:
    """Affiche un resume rapide des metriques UCS."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ucs_l01 = metrics.get("UCS", {}).get("Tailings", {}).get("L01", {})
    ucs_ww = metrics.get("UCS", {}).get("Tailings", {}).get("WW", {})
    print(f"[{label}] UCS L01 R2={ucs_l01.get('r2')} RMSE={ucs_l01.get('rmse')}")
    print(f"[{label}] UCS WW  R2={ucs_ww.get('r2')} RMSE={ucs_ww.get('rmse')}")


def _retrain_for_dataset(
    sweep_dir: Path | None,
    spec: DatasetSpec,
    seed: int,
    n_samples: int,
    out_dir: Path,
    run_id: str,
) -> dict:
    """Relance un entrainement final pour un dataset."""
    config = {}
    best_params = None
    if sweep_dir and sweep_dir.exists():
        config, summary_path = _locate_best_config(sweep_dir, spec.dataset_id)
        if config:
            run_dir = _locate_run_dir(
                sweep_dir, spec.dataset_id, config, seed
            )
            if run_dir:
                best_params_path = run_dir / "best_params.json"
                if best_params_path.exists():
                    best_params = _load_best_params(best_params_path)
        else:
            print(
                f"[{spec.dataset_id}] Aucun resume exploitable, fallback."
            )
    if not config:
        config = _fallback_config(spec.dataset_id)

    args = _build_cli_args(
        spec, config, seed, n_samples, out_dir, run_id, best_params
    )
    code = cli.main(args)
    if code != 0:
        raise RuntimeError(
            f"Entrainement final echoue pour {spec.dataset_id}."
        )

    run_dir = out_dir / "runs" / run_id
    recipes_src = run_dir / "Top_Recipes.xlsx"
    recipes_dst = out_dir / f"Top_Recipes_FINAL_{spec.dataset_id}.xlsx"
    if recipes_src.exists():
        recipes_dst.write_bytes(recipes_src.read_bytes())
    _print_metrics_summary(run_dir, spec.dataset_id)
    return {"run_id": run_id, "config": config, "run_dir": run_dir}


def main(argv: list[str] | None = None) -> int:
    """Point d'entree principal pour relancer les modeles finaux."""
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if args.sweep_new or not args.sweep_old:
        sweep_new = Path(args.sweep_new) if args.sweep_new else None
        results.append(
            _retrain_for_dataset(
                sweep_new,
                DEFAULT_NEW,
                args.seed,
                args.n_samples,
                out_dir,
                "FINAL_new_best",
            )
        )
    if args.sweep_old:
        sweep_old = Path(args.sweep_old)
        results.append(
            _retrain_for_dataset(
                sweep_old,
                DEFAULT_OLD,
                args.seed,
                args.n_samples,
                out_dir,
                "FINAL_old_best",
            )
        )

    print("Entrainement final termine.")
    for result in results:
        print(
            f"- {result['run_id']} -> {result['run_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
