"""Configuration et valeurs par defaut du projet.

Ce fichier centralise les constantes utilisees dans plusieurs modules.
"""

from __future__ import annotations

from dataclasses import dataclass


# Colonnes minimales requises apres standardisation.
REQUIRED_COLUMNS = [
    "Binder",
    "E/C",
    "Cw_f",
    "Ad %",
    "Slump (mm)",
    "UCS28d (kPa)",
]

# Colonnes categorielles standard.
CATEGORICAL_COLS = ["Binder", "Tailings"]
# Colonnes numeriques de base (utiles pour tests).
NUMERIC_BASE_COLS = ["E/C", "Cw_f", "Ad %"]

TARGET_SLUMP = "Slump (mm)"
TARGET_UCS = "UCS28d (kPa)"

OUTPUT_BINDERS = ["GUL", "Slag"]
OUTPUT_TAILINGS = ["L01", "WW"]


@dataclass(frozen=True)
class Defaults:
    """Valeurs par defaut utilisees par la CLI."""
    sheet_l01: str | None = None
    sheet_ww: str | None = None
    slump_min: float = 70.0
    ucs_min: float = 900.0
    n_samples: int = 50_000
    random_state: int = 42
    n_splits: int = 5


DEFAULTS = Defaults()
