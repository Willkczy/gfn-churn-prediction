"""End-to-end Phase 3 pipeline.

Reads:
  - data/processed/weekly_features.parquet (Phase 2 output)
  - data/raw/users.parquet
  - data/raw/session_logs.parquet

Writes:
  - data/processed/xgboost_{train,val,test}.parquet
  - data/processed/lstm_{train,val,test}.npz

Run:
  python -m src.dataset.pipeline
"""

import os

import numpy as np
import pandas as pd

from .aggregation import aggregate_xgboost, reshape_lstm
from .labels import compute_churn_labels
from .split import stratified_split

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"


def run(
    raw_dir: str = DEFAULT_RAW_DIR,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
) -> None:
    weekly_features = pd.read_parquet(f"{processed_dir}/weekly_features.parquet")
    users = pd.read_parquet(f"{raw_dir}/users.parquet", columns=["user_id"])
    sessions = pd.read_parquet(
        f"{raw_dir}/session_logs.parquet", columns=["user_id", "start_time"]
    )

    labels = compute_churn_labels(users, sessions)
    print(
        f"Churn rate: {labels['churn'].mean():.1%} "
        f"({int(labels['churn'].sum()):,} / {len(labels):,})"
    )

    xgb_df = aggregate_xgboost(weekly_features, labels)
    X_lstm, y_lstm, user_ids, feature_names = reshape_lstm(weekly_features, labels)
    print(f"XGBoost: {xgb_df.shape}   LSTM X: {X_lstm.shape}")

    splits = stratified_split(labels, user_ids)
    for name, ids in splits.items():
        cn = int(labels.set_index("user_id").loc[ids, "churn"].sum())
        print(f"  {name:<5} {len(ids):>6,} users  churn={cn:>5,} ({cn/len(ids):.1%})")

    os.makedirs(processed_dir, exist_ok=True)
    uid_to_row = {uid: i for i, uid in enumerate(user_ids)}

    for split_name, split_ids in splits.items():
        split_id_set = set(split_ids)
        xgb_split = xgb_df.loc[xgb_df.index.isin(split_id_set)]
        xgb_split.to_parquet(f"{processed_dir}/xgboost_{split_name}.parquet")

        row_idx = np.array([uid_to_row[uid] for uid in split_ids])
        np.savez(
            f"{processed_dir}/lstm_{split_name}.npz",
            X=X_lstm[row_idx],
            y=y_lstm[row_idx],
            user_ids=np.array(split_ids),
            feature_names=np.array(feature_names),
        )

    print(f"Saved 6 split files to {processed_dir}/")


if __name__ == "__main__":
    run()
