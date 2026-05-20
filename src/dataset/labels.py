"""Churn label generation.

`churn = 1` iff a user has zero sessions in the prediction window
(calendar weeks 9-10). Returned DataFrame is keyed by `user_id` and has
a single `churn` column of {0, 1}.
"""

import pandas as pd

from .config import PRED_END, PRED_START


def compute_churn_labels(users: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    pred_sessions = sessions[
        (sessions["start_time"].dt.date >= PRED_START)
        & (sessions["start_time"].dt.date < PRED_END)
    ]
    active_in_pred = set(pred_sessions["user_id"].unique())

    labels = users[["user_id"]].copy()
    labels["churn"] = (~labels["user_id"].isin(active_in_pred)).astype(int)
    return labels
