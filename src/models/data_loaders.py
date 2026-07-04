"""Load Phase 3 split datasets into model-ready arrays.

XGBoost loader returns a pandas DataFrame so column names stay attached
(useful for SHAP). LSTM loader returns NumPy arrays plus a fitted
StandardScaler — fit on the train split, applied to val/test via the
same scaler. The scaler must be persisted alongside the LSTM model so
inference on new data uses the same normalization.

Feature scales span six orders of magnitude (peak_hour_ratio ∈ [0, 1] vs
frame_drop_rate ∈ [0, 25000]); LSTM convergence is much faster after
StandardScaler. XGBoost does not need scaling — trees are scale-invariant.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED_DIR

Split = str  # one of "train" / "val" / "test"


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def load_xgb_split(
    split: Split, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(processed_dir / f"xgboost_{split}.parquet")
    y = df["churn"]
    X = df.drop(columns=["churn"])
    return X, y


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def _apply_scaler_3d(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Scale a (N, T, F) tensor by reshaping to (N*T, F)."""
    n, t, f = X.shape
    return scaler.transform(X.reshape(n * t, f)).reshape(n, t, f)


def load_lstm_split(
    split: Split,
    scaler: StandardScaler | None = None,
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    """Return (X, y, user_ids, feature_names, scaler).

    For the train split, pass `scaler=None`; the function will fit a fresh
    StandardScaler and return it. For val/test, pass the scaler returned
    from the train load — `_apply_scaler_3d` then transforms in-place.
    """
    data = np.load(processed_dir / f"lstm_{split}.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    user_ids = data["user_ids"]
    feature_names = [str(c) for c in data["feature_names"]]

    if scaler is None:
        scaler = StandardScaler()
        n, t, f = X.shape
        scaler.fit(X.reshape(n * t, f))

    X_scaled = _apply_scaler_3d(X, scaler).astype(np.float32)
    return X_scaled, y, user_ids, feature_names, scaler


def load_lstm_all(
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, dict]:
    """Convenience: load all three splits in one call.

    Fits the scaler on train, then applies it to val and test. Returns a
    dict keyed by split name, plus a top-level "scaler" entry.
    """
    X_tr, y_tr, ids_tr, feats, scaler = load_lstm_split("train", scaler=None, processed_dir=processed_dir)
    X_va, y_va, ids_va, _, _ = load_lstm_split("val", scaler=scaler, processed_dir=processed_dir)
    X_te, y_te, ids_te, _, _ = load_lstm_split("test", scaler=scaler, processed_dir=processed_dir)
    return {
        "train": {"X": X_tr, "y": y_tr, "user_ids": ids_tr},
        "val":   {"X": X_va, "y": y_va, "user_ids": ids_va},
        "test":  {"X": X_te, "y": y_te, "user_ids": ids_te},
        "feature_names": feats,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Class imbalance helpers
# ---------------------------------------------------------------------------

def class_imbalance_ratio(y: np.ndarray | pd.Series) -> float:
    """Return `neg / pos` — the multiplier for XGBoost `scale_pos_weight`
    or PyTorch `BCEWithLogitsLoss(pos_weight=...)`.

    With churn rate ~12.2% this lands around 7.2.
    """
    y_arr = np.asarray(y)
    pos = int((y_arr == 1).sum())
    neg = int((y_arr == 0).sum())
    if pos == 0:
        raise ValueError("No positive samples in y")
    return neg / pos
