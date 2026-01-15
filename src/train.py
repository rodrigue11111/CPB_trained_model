"""Entrainement et evaluation des modeles.

Contient :
- creation des pipelines sklearn (preprocess + modele)
- filtrage outliers UCS
- cross-validation et metriques
- entrainement hybride par tailings
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from .config import TARGET_SLUMP, TARGET_UCS
from .features import coerce_numeric, infer_feature_columns, split_features_target


def _make_onehot(sparse: bool) -> OneHotEncoder:
    """Compat scikit-learn: sparse_output pour versions recentes."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)


def build_pipeline(
    model_name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int = 42,
) -> Pipeline:
    """Construit un pipeline sklearn complet (preprocess + modele).

    Args:
        model_name: nom court du modele (gbr, rf, et, hgb, svr, enet).
        numeric_cols: colonnes numeriques a utiliser.
        categorical_cols: colonnes categorielles a encoder.
        random_state: graine pour les modeles aleatoires.

    Returns:
        Pipeline sklearn: preprocess -> model.
    """
    model_key = (model_name or "gbr").lower()

    if model_key in {"gbr", "gradientboosting", "gradient_boosting"}:
        model = GradientBoostingRegressor(random_state=random_state)
    elif model_key in {"rf", "randomforest", "random_forest"}:
        model = RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1
        )
    elif model_key in {"et", "extratrees", "extra_trees"}:
        model = ExtraTreesRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        )
    elif model_key in {"hgb", "histgradientboosting", "hist_gradient_boosting"}:
        model = HistGradientBoostingRegressor(random_state=random_state)
    elif model_key in {"svr"}:
        model = SVR(kernel="rbf")
    elif model_key in {"enet", "elasticnet"}:
        model = ElasticNet(random_state=random_state)
    else:
        raise ValueError(
            "Modele inconnu. Utiliser 'gbr', 'rf', 'et', 'hgb', 'svr' ou 'enet'."
        )

    # SVR/ENet ont besoin d'un scaling numerique; les arbres non.
    scale_numeric = model_key in {"svr", "enet", "elasticnet"}
    ohe_sparse = not scale_numeric

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot(ohe_sparse)),
        ]
    )
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numeric_pipeline, numeric_cols),
        ]
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def build_estimator(
    model_name: str,
    target_transform: str = "none",
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    random_state: int = 42,
) -> Pipeline | TransformedTargetRegressor:
    """Construit un estimateur avec transform optionnel sur la cible UCS.

    Args:
        model_name: type de modele a utiliser.
        target_transform: "none" ou "log" (log1p/expm1).
        numeric_cols: features numeriques.
        categorical_cols: features categorielles.
        random_state: graine pour la reproductibilite.
    """
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []
    pipe = build_pipeline(
        model_name,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        random_state=random_state,
    )
    transform = (target_transform or "none").lower()
    if transform == "none":
        return pipe
    if transform == "log":
        # log1p/expm1 pour stabiliser la variance de UCS.
        return TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    raise ValueError("target_transform_ucs doit etre 'none' ou 'log'.")


def filter_ucs_outliers(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Filtre les outliers UCS selon une methode simple.

    Args:
        df: DataFrame avec la colonne UCS28d (kPa).
        method: none | iqr | zscore.

    Returns:
        DataFrame filtre (peut etre vide si tout est exclu).
    """
    method = (method or "none").lower()
    if method == "none":
        return df

    series = pd.to_numeric(df[TARGET_UCS], errors="coerce")
    valid_mask = series.notna()
    series = series[valid_mask]
    if series.empty:
        return df.iloc[0:0].copy()

    if method == "iqr":
        # Methode inter-quartile.
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        keep = (series >= lower) & (series <= upper)
        return df.loc[series.index[keep]].copy()

    if method == "zscore":
        # Methode z-score classique.
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return df.copy()
        zscores = (series - mean) / std
        keep = zscores.abs() <= 3.0
        return df.loc[series.index[keep]].copy()

    raise ValueError("outliers_ucs doit etre 'none', 'iqr' ou 'zscore'.")


def _param_prefix(estimator) -> str:
    """Prefix des hyperparametres selon que la cible est transformee ou non.

    IMPORTANT:
        - Pipeline direct: prefix = "model__"
        - TransformedTargetRegressor: prefix = "regressor__model__"
        Cela evite de passer des params au mauvais objet.
    """
    if isinstance(estimator, TransformedTargetRegressor):
        return "regressor__model__"
    return "model__"


def _param_distributions(model_name: str, prefix: str) -> dict:
    """Grilles de tuning (RandomizedSearchCV) par type de modele."""
    key = (model_name or "gbr").lower()
    if key in {"gbr", "gradientboosting", "gradient_boosting"}:
        return {
            f"{prefix}n_estimators": [200, 400, 800],
            f"{prefix}learning_rate": [0.02, 0.05, 0.1],
            f"{prefix}max_depth": [2, 3, 4],
            f"{prefix}subsample": [0.6, 0.8, 1.0],
        }
    if key in {"rf", "randomforest", "random_forest"}:
        return {
            f"{prefix}n_estimators": [300, 500, 800, 1200],
            f"{prefix}max_depth": [6, 10, 14, 18, None],
            f"{prefix}min_samples_leaf": [1, 2, 4, 8, 12],
            f"{prefix}max_features": ["sqrt", 0.5, 0.7, 1.0],
        }
    if key in {"et", "extratrees", "extra_trees"}:
        return {
            f"{prefix}n_estimators": [300, 500, 800, 1200],
            f"{prefix}max_depth": [6, 10, 14, 18, None],
            f"{prefix}min_samples_leaf": [1, 2, 4, 8, 12],
            f"{prefix}max_features": ["sqrt", 0.5, 0.7, 1.0],
        }
    if key in {"hgb", "histgradientboosting", "hist_gradient_boosting"}:
        return {
            f"{prefix}learning_rate": [0.02, 0.05, 0.1],
            f"{prefix}max_depth": [2, 3, 4, None],
            f"{prefix}max_leaf_nodes": [15, 31, 63, 127],
            f"{prefix}min_samples_leaf": [5, 10, 20, 40],
            f"{prefix}l2_regularization": [0.0, 1e-4, 1e-3, 1e-2],
        }
    if key in {"svr"}:
        return {
            f"{prefix}C": loguniform(1, 1000),
            f"{prefix}gamma": loguniform(1e-4, 1e-1),
            f"{prefix}epsilon": [0.01, 0.05, 0.1, 0.2],
        }
    if key in {"enet", "elasticnet"}:
        return {
            f"{prefix}alpha": loguniform(1e-4, 1.0),
            f"{prefix}l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    return {}


def _neg_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Score negatif RMSE pour l'optimiseur sklearn."""
    return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


def tune_estimator(
    estimator,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 12,
    cv: int | KFold = 3,
    random_state: int = 42,
) -> tuple[object, dict]:
    """Tuning simple via RandomizedSearchCV.

    Notes:
        - On optimise le RMSE (score negatif).
        - Si aucun param_distributions, on renvoie tel quel.
    """
    prefix = _param_prefix(estimator)
    param_dist = _param_distributions(model_name, prefix)
    if not param_dist:
        return estimator, {}

    if isinstance(cv, int):
        cv_splitter = KFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
    else:
        cv_splitter = cv

    scorer = make_scorer(_neg_rmse, greater_is_better=True)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring=scorer,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def get_model_meta(estimator, model_name: str) -> dict:
    """Extrait le nom et les parametres internes du modele."""
    if isinstance(estimator, TransformedTargetRegressor):
        inner = (
            estimator.regressor_
            if hasattr(estimator, "regressor_")
            else estimator.regressor
        )
    else:
        inner = estimator

    if isinstance(inner, Pipeline):
        model = inner.named_steps.get("model", inner)
    else:
        model = inner

    params = model.get_params() if hasattr(model, "get_params") else {}
    return {"name": model_name, "params": params}


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calcule RMSE et R2."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "r2": float(r2)}


def _metrics_by_group(
    y_true: pd.Series, y_pred: np.ndarray, group: pd.Series
) -> dict:
    """Calcule des metriques par groupe (Tailings ou Binder)."""
    metrics = {}
    for value in sorted(group.dropna().unique()):
        mask = group == value
        if mask.sum() < 2:
            metrics[str(value)] = {"rmse": float("nan"), "r2": float("nan")}
            continue
        metrics[str(value)] = _compute_metrics(y_true[mask], y_pred[mask])
    return metrics


def cross_validate_report(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups_df: pd.DataFrame | None = None,
    random_state: int = 42,
) -> dict:
    """Genere un rapport CV global + par groupe.

    Returns:
        dict avec "overall" + "Tailings" + "Binder" si disponibles.
    """
    if len(y) < 5:
        raise ValueError(
            f"Pas assez d'echantillons ({len(y)}) pour une CV a 5 plis."
        )

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    preds = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)

    report = {"overall": _compute_metrics(y, preds)}
    if groups_df is not None:
        if "Tailings" in groups_df.columns:
            report["Tailings"] = _metrics_by_group(
                y, preds, groups_df["Tailings"]
            )
        if "Binder" in groups_df.columns:
            report["Binder"] = _metrics_by_group(
                y, preds, groups_df["Binder"]
            )
    return report


def cross_validate_report_hybrid(
    models_by_tailings: dict,
    df: pd.DataFrame,
    target_col: str,
    outliers_by_tailings: dict | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """CV hybride: on applique chaque modele a son tailings.

    Notes:
        - On fait une CV 5 plis par tailings, puis on concatene les predictions.
        - Les metriques par groupe restent comparables entre L01 et WW.
    """
    y_all: list[float] = []
    pred_all: list[float] = []
    tailings_all: list[str] = []
    binder_all: list[str] = []

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for tailings in ["L01", "WW"]:
        if tailings not in models_by_tailings:
            continue
        subset = df[df["Tailings"] == tailings].copy()
        if subset.empty:
            continue

        if target_col == TARGET_UCS and outliers_by_tailings:
            # Filtrage outliers UCS specifique au tailings.
            method = outliers_by_tailings.get(tailings, "none")
            subset = filter_ucs_outliers(subset, method)
            if subset.empty:
                continue

        X, y, _, _ = _prepare_xy(subset, target_col)
        mask = y.notna()
        X = X.loc[mask].copy()
        y = y.loc[mask].copy()

        if len(y) < n_splits:
            raise ValueError(
                f"Pas assez d'echantillons pour {tailings} (n={len(y)})."
            )

        estimator_key = "ucs_pipe" if target_col == TARGET_UCS else "slump_pipe"
        estimator = models_by_tailings[tailings][estimator_key]
        preds = cross_val_predict(estimator, X, y, cv=cv, n_jobs=-1)

        y_all.extend(y.to_list())
        pred_all.extend(preds.tolist())
        tailings_all.extend([tailings] * len(y))
        binder_all.extend(subset.loc[X.index, "Binder"].astype(str).to_list())

    if not y_all:
        raise ValueError("Aucune donnee disponible pour la CV hybride.")

    y_true = pd.Series(y_all)
    y_pred = np.array(pred_all)
    tailings_series = pd.Series(tailings_all)
    binder_series = pd.Series(binder_all)

    report = {"overall": _compute_metrics(y_true, y_pred)}
    report["Tailings"] = _metrics_by_group(y_true, y_pred, tailings_series)
    report["Binder"] = _metrics_by_group(y_true, y_pred, binder_series)
    return report


def _print_report(label: str, report: dict) -> None:
    """Affiche un rapport CV lisible en console."""
    overall = report.get("overall", {})
    rmse = overall.get("rmse", float("nan"))
    r2 = overall.get("r2", float("nan"))
    print(f"[{label}] CV RMSE={rmse:.3f} R2={r2:.3f}")

    for group_name in ("Tailings", "Binder"):
        group_metrics = report.get(group_name, {})
        for value, metrics in group_metrics.items():
            rmse_v = metrics.get("rmse", float("nan"))
            r2_v = metrics.get("r2", float("nan"))
            print(
                f"  {group_name} {value}: RMSE={rmse_v:.3f} R2={r2_v:.3f}"
            )


def _prepare_xy(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepare X/y avec selection dynamique des features."""
    categorical_cols, numeric_cols = infer_feature_columns(
        df, target_cols=[TARGET_SLUMP, TARGET_UCS]
    )
    df = coerce_numeric(df, list(numeric_cols) + [target_col])
    X, y = split_features_target(df, target_col, categorical_cols, numeric_cols)
    return X, y, categorical_cols, numeric_cols


def fit_final_models(
    df_slump: pd.DataFrame,
    df_ucs: pd.DataFrame,
    model_slump: str = "gbr",
    model_ucs: str = "gbr",
    target_transform_ucs: str = "none",
) -> tuple[Pipeline, Pipeline]:
    """Entraine des modeles finaux (mode combined historique).

    Note:
        Cette fonction reste utile pour des tests rapides en mode global.
    """
    if df_slump.empty:
        raise ValueError(
            "Aucune ligne avec Slump (mm) disponible pour l'entrainement."
        )
    if df_ucs.empty:
        raise ValueError(
            "Aucune ligne avec UCS28d (kPa) disponible pour l'entrainement."
        )

    X_slump, y_slump, cat_slump, num_slump = _prepare_xy(
        df_slump, TARGET_SLUMP
    )
    slump_pipe = build_estimator(
        model_slump, numeric_cols=num_slump, categorical_cols=cat_slump
    )
    report_slump = cross_validate_report(
        slump_pipe, X_slump, y_slump, groups_df=df_slump
    )
    _print_report("Slump", report_slump)
    slump_pipe.fit(X_slump, y_slump)

    X_ucs, y_ucs, cat_ucs, num_ucs = _prepare_xy(df_ucs, TARGET_UCS)
    ucs_pipe = build_estimator(
        model_ucs,
        target_transform=target_transform_ucs,
        numeric_cols=num_ucs,
        categorical_cols=cat_ucs,
    )
    report_ucs = cross_validate_report(
        ucs_pipe, X_ucs, y_ucs, groups_df=df_ucs
    )
    _print_report("UCS", report_ucs)
    ucs_pipe.fit(X_ucs, y_ucs)

    return slump_pipe, ucs_pipe


def fit_models_hybrid(
    df_slump: pd.DataFrame,
    df_ucs: pd.DataFrame,
    config: dict,
    random_state: int = 42,
) -> dict:
    """Entraine des modeles WW/L01 differents selon la config.

    Args:
        df_slump: donnees Slump (Option A).
        df_ucs: donnees UCS (Option A).
        config: dictionnaire par tailings avec model/transform/outliers/tune.
        random_state: graine pour les modeles aleatoires.
    """
    models: dict = {}
    for tailings in ["L01", "WW"]:
        if tailings not in config:
            continue
        cfg = config[tailings]

        slump_df = df_slump[df_slump["Tailings"] == tailings].copy()
        ucs_df = df_ucs[df_ucs["Tailings"] == tailings].copy()

        if slump_df.empty:
            raise ValueError(
                f"Aucune ligne Slump pour Tailings={tailings}."
            )
        if ucs_df.empty:
            raise ValueError(
                f"Aucune ligne UCS pour Tailings={tailings}."
            )

        ucs_outliers = cfg["ucs"].get("outliers", "none")
        ucs_df = filter_ucs_outliers(ucs_df, ucs_outliers)
        if ucs_df.empty:
            raise ValueError(
                f"Aucune ligne UCS apres filtre outliers pour {tailings}."
            )

        X_slump, y_slump, cat_slump, num_slump = _prepare_xy(
            slump_df, TARGET_SLUMP
        )
        X_ucs, y_ucs, cat_ucs, num_ucs = _prepare_xy(ucs_df, TARGET_UCS)

        slump_mask = y_slump.notna()
        X_slump = X_slump.loc[slump_mask].copy()
        y_slump = y_slump.loc[slump_mask].copy()

        ucs_mask = y_ucs.notna()
        X_ucs = X_ucs.loc[ucs_mask].copy()
        y_ucs = y_ucs.loc[ucs_mask].copy()

        if cfg["ucs"].get("transform", "none") == "log":
            if (y_ucs <= -1).any():
                raise ValueError(
                    f"UCS contient des valeurs <= -1 pour {tailings}."
                )

        slump_estimator = build_estimator(
            cfg["slump"]["model"],
            numeric_cols=num_slump,
            categorical_cols=cat_slump,
            random_state=random_state,
        )
        ucs_estimator = build_estimator(
            cfg["ucs"]["model"],
            target_transform=cfg["ucs"].get("transform", "none"),
            numeric_cols=num_ucs,
            categorical_cols=cat_ucs,
            random_state=random_state,
        )

        slump_fixed = cfg["slump"].get("fixed_params")
        ucs_fixed = cfg["ucs"].get("fixed_params")
        if slump_fixed:
            # Fixed params -> pas de tuning pour ce modele.
            try:
                slump_estimator.set_params(**slump_fixed)
            except ValueError as exc:
                raise ValueError(
                    f"Params fixes invalides Slump pour {tailings}: {exc}"
                ) from exc
        if ucs_fixed:
            try:
                ucs_estimator.set_params(**ucs_fixed)
            except ValueError as exc:
                raise ValueError(
                    f"Params fixes invalides UCS pour {tailings}: {exc}"
                ) from exc

        best_params = {"slump": {}, "ucs": {}}
        tune_slump = cfg["slump"].get("tune") and not slump_fixed
        tune_ucs = cfg["ucs"].get("tune") and not ucs_fixed

        if slump_fixed:
            best_params["slump"] = slump_fixed
        elif tune_slump:
            # RandomizedSearchCV si tune actif et pas de fixed params.
            slump_estimator, best_params["slump"] = tune_estimator(
                slump_estimator,
                cfg["slump"]["model"],
                X_slump,
                y_slump,
                random_state=random_state,
            )
        if ucs_fixed:
            best_params["ucs"] = ucs_fixed
        elif tune_ucs:
            ucs_estimator, best_params["ucs"] = tune_estimator(
                ucs_estimator,
                cfg["ucs"]["model"],
                X_ucs,
                y_ucs,
                random_state=random_state,
            )

        slump_estimator.fit(X_slump, y_slump)
        ucs_estimator.fit(X_ucs, y_ucs)

        models[tailings] = {
            "slump_pipe": slump_estimator,
            "ucs_pipe": ucs_estimator,
            "best_params": best_params,
            "features": {
                "slump": {"categorical": cat_slump, "numeric": num_slump},
                "ucs": {"categorical": cat_ucs, "numeric": num_ucs},
            },
        }

    return models
