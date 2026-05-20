"""Stratified train/val/test split by churn label.

Same `user_id` partition is meant to be applied to both XGBoost and LSTM
formats (callers index into both formats using the returned id arrays),
so a user appearing in `train_ids` ends up in both train datasets.
"""

import numpy as np
import pandas as pd

from .config import SEED, TRAIN_RATIO, VAL_RATIO


def stratified_split(
    labels: pd.DataFrame,
    user_ids: np.ndarray,
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> dict[str, np.ndarray]:
    """Return {"train": ids, "val": ids, "test": ids}.

    `user_ids` defines the universe and the order — the labels frame may
    contain extras. Stratifies by churn so each split preserves the
    population churn rate.
    """
    rng = np.random.default_rng(seed)

    labels_df = labels.set_index("user_id").loc[user_ids].reset_index()
    pos = labels_df.loc[labels_df["churn"] == 1, "user_id"].to_numpy()
    neg = labels_df.loc[labels_df["churn"] == 0, "user_id"].to_numpy()

    def _slice(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        perm = rng.permutation(len(ids))
        n_train = int(len(ids) * train_ratio)
        n_val = int(len(ids) * val_ratio)
        return (
            ids[perm[:n_train]],
            ids[perm[n_train : n_train + n_val]],
            ids[perm[n_train + n_val :]],
        )

    pos_tr, pos_va, pos_te = _slice(pos)
    neg_tr, neg_va, neg_te = _slice(neg)

    splits = {
        "train": np.concatenate([pos_tr, neg_tr]),
        "val": np.concatenate([pos_va, neg_va]),
        "test": np.concatenate([pos_te, neg_te]),
    }
    for ids in splits.values():
        rng.shuffle(ids)
    return splits
