"""Tests pour verifier les metriques en mode hybride."""

import json

import numpy as np
import pandas as pd

from src import cli
from src.config import REQUIRED_COLUMNS, TARGET_SLUMP, TARGET_UCS


def _make_df(rows: int, binder_values: list[str]) -> pd.DataFrame:
    """Construit un DataFrame jouet avec targets valides."""
    data = {col: [] for col in REQUIRED_COLUMNS}
    for i in range(rows):
        data["Binder"].append(binder_values[i % len(binder_values)])
        for col in REQUIRED_COLUMNS:
            if col in {"Binder", TARGET_SLUMP, TARGET_UCS}:
                continue
            data[col].append(float(1 + i))
        data[TARGET_SLUMP].append(float(80 + i))
        data[TARGET_UCS].append(float(1000 + i * 10))
    return pd.DataFrame(data)


def test_hybrid_metrics_contains_l01_transform(tmp_path):
    """Verifie que la transformation UCS L01 est bien enregistree."""
    l01_df = _make_df(6, ["GUL", "Slag"])
    ww_df = _make_df(6, ["GUL", "Slag"])

    l01_path = tmp_path / "l01.xlsx"
    ww_path = tmp_path / "ww.xlsx"
    l01_df.to_excel(l01_path, index=False)
    ww_df.to_excel(ww_path, index=False)

    out_dir = tmp_path / "outputs"
    run_id = "hybrid_test"

    args = [
        "--l01",
        str(l01_path),
        "--ww",
        str(ww_path),
        "--fit-mode",
        "hybrid_by_tailings",
        "--l01-ucs-transform",
        "log",
        "--ww-ucs-transform",
        "none",
        "--out-dir",
        str(out_dir),
        "--run-id",
        run_id,
        "--n-samples",
        "10",
    ]

    code = cli.main(args)
    assert code == 0

    metrics_path = out_dir / "runs" / run_id / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert (
        metrics["meta"]["models"]["L01"]["ucs"]["transform"] == "log"
    )
