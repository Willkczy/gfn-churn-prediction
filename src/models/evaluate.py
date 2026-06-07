"""Shared evaluation utilities for all Phase 4 models.

`compute_metrics` returns a dict that is JSON-serializable for persistence
under models/<name>_metrics.json. `find_best_threshold` is used by ensemble
glue code (and per-model evaluation) to pick a deployment threshold on
val before reporting test metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1"
) -> tuple[float, float]:
    """Sweep thresholds in [0.05, 0.95] step 0.01 and pick the one that
    maximises the chosen metric. Returns (threshold, metric_value).
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_t, best_v = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            v = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        if v > best_v:
            best_v, best_t = v, t
    return float(best_t), float(best_v)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Compute ROC-AUC, PR-AUC, plus F1/precision/recall + confusion matrix
    at the given threshold (or the threshold that maximises F1 if None).
    """
    if threshold is None:
        threshold, _ = find_best_threshold(y_true, y_proba, metric="f1")

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def format_metrics(metrics: dict) -> str:
    """Pretty one-line summary for logs / notebook output."""
    return (
        f"AUC={metrics['roc_auc']:.4f}  "
        f"PR-AUC={metrics['pr_auc']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  "
        f"@thr={metrics['threshold']:.2f}"
    )
