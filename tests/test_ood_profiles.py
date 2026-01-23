"""Tests pour la detection OOD des profils."""

import numpy as np

from app.core.data_profiles import ood_level


def test_ood_level_with_complete_profile():
    profile = {"min": 0.0, "max": 10.0, "p05": 2.0, "p95": 8.0}
    assert ood_level(5.0, profile)["level"] == "ok"
    assert ood_level(1.0, profile)["level"] == "warn"
    assert ood_level(11.0, profile)["level"] == "out"


def test_ood_level_with_fallback_std():
    profile = {"min": 0.0, "max": 10.0, "mean": 5.0, "std": 2.0}
    result = ood_level(9.0, profile)
    assert result["level"] in {"ok", "warn"}


def test_ood_level_unknown_when_missing_profile():
    profile = {"min": np.nan, "max": np.nan}
    result = ood_level(5.0, profile)
    assert result["level"] == "unknown"
