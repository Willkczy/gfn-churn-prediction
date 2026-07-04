"""XGBoost training, prediction, and artifact persistence.

`train_xgboost` returns the fitted `XGBClassifier` and a dict with the val
metrics computed at the early-stopping best iteration. `save_artifacts`
writes the model (JSON), predictions (parquet), and metrics (JSON) under
`models/<name>_*` so they can be reloaded by `evaluate.py` or the ensemble
notebook without re-running training.

Hyperparameters live in `src.models.config.XGB_PARAMS` — change them there,
not here, so the rest of the pipeline picks them up automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import MODELS_DIR, SEED, XGB_PARAMS
from .data_loaders import class_imbalance_ratio
from .evaluate import compute_metrics, find_best_threshold


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
) -> tuple[xgb.XGBClassifier, dict]:
    params = dict(params or XGB_PARAMS)
    params.setdefault("random_state", SEED)
    params["scale_pos_weight"] = class_imbalance_ratio(y_train)

    clf = xgb.XGBClassifier(**params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_proba = clf.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val.to_numpy(), val_proba)

    fit_info = {
        "best_iteration": int(clf.best_iteration),
        "n_estimators_used": int(clf.best_iteration + 1),
        "scale_pos_weight": float(params["scale_pos_weight"]),
        "val_metrics": val_metrics,
    }
    return clf, fit_info


def predict_proba(clf: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    return clf.predict_proba(X)[:, 1]


def save_artifacts(
    clf: xgb.XGBClassifier,
    name: str,
    val_predictions: tuple[pd.Series, np.ndarray],  # (y_true, y_proba)
    test_predictions: tuple[pd.Series, np.ndarray],
    fit_info: dict,
    test_threshold: float | None = None,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """Save model, predictions, and metrics under `<models_dir>/<name>_*`.

    The deployment threshold is selected on val (maximises F1), then frozen
    when computing test metrics. This avoids tuning a threshold on test.
    Returns a dict {val: {...}, test: {...}, paths: {...}} for the caller.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{name}.json"
    clf.save_model(model_path)

    y_val_true, y_val_proba = val_predictions
    y_test_true, y_test_proba = test_predictions

    # Threshold chosen on val, applied to test
    if test_threshold is None:
        test_threshold, _ = find_best_threshold(y_val_true.to_numpy(), y_val_proba)

    val_metrics = compute_metrics(y_val_true.to_numpy(), y_val_proba, threshold=test_threshold)
    test_metrics = compute_metrics(y_test_true.to_numpy(), y_test_proba, threshold=test_threshold)

    metrics_path = models_dir / f"{name}_metrics.json"
    payload = {
        "model": "xgboost",
        "name": name,
        "fit_info": fit_info,
        "threshold": float(test_threshold),
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    pred_path = models_dir / f"{name}_predictions.parquet"
    preds_df = pd.DataFrame(
        {
            "user_id": list(y_val_true.index) + list(y_test_true.index),
            "split": ["val"] * len(y_val_true) + ["test"] * len(y_test_true),
            "y_true": np.concatenate([y_val_true.to_numpy(), y_test_true.to_numpy()]),
            "y_proba": np.concatenate([y_val_proba, y_test_proba]),
        }
    )
    preds_df.to_parquet(pred_path, index=False)

    return {
        "val": val_metrics,
        "test": test_metrics,
        "paths": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "predictions": str(pred_path),
        },
    }
