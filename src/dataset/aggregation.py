"""XGBoost flat aggregation + LSTM 3D tensor reshape.

XGBoost: one row per user. Aggregation by feature type — see
`configs/feature_spec.md` for the column-by-column rationale.

LSTM: (users, 4 weeks, num_features) tensor. NaN-fill strategy is
also documented in `configs/feature_spec.md`.
"""

import numpy as np
import pandas as pd

# --- Feature column classification ---
# These lists mirror Section 2 of phase3_dataset_construction.ipynb.

MEAN_COLS = [
    "weekly_session_count", "avg_session_duration_min", "total_playtime_min",
    "peak_hour_ratio", "weekend_ratio", "session_regularity",
    "weekly_session_count_norm", "total_playtime_min_norm",
    "avg_latency", "avg_fps", "frame_drop_rate", "disconnect_rate",
    "avg_bitrate", "avg_jitter", "packet_loss_avg", "crash_exit_ratio",
    "unique_games_played", "top_game_concentration", "genre_entropy",
    "daily_playtime_std", "daily_playtime_cv", "session_duration_std",
]

LAST_COLS = [
    "session_count_wow_change", "playtime_wow_change",
    "session_count_vs_baseline", "playtime_vs_baseline",
    "longest_inactive_days", "new_game_trial_rate",
]

EXTRA_LAST_COLS = [
    "weekly_session_count", "total_playtime_min", "crash_exit_ratio",
]

STATIC_COLS = [
    "current_tier_numeric", "days_since_signup",
    "tier_changes_count", "has_downgraded",
    "total_spend_last_4w", "payment_count_last_4w",
    "failed_payment_count", "refund_count",
    "days_since_last_payment", "payment_frequency_change",
]


def _compute_slope(group: pd.DataFrame) -> float:
    """Linear slope of total_playtime_min across the 4 obs weeks."""
    x = group["week_num"].values.astype(float)
    y = group["total_playtime_min"].values.astype(float)
    n = len(x)
    return (n * np.dot(x, y) - x.sum() * y.sum()) / (
        n * np.dot(x, x) - x.sum() ** 2
    )


def aggregate_xgboost(
    weekly_features: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    wf = weekly_features

    agg_mean = wf.groupby("user_id")[MEAN_COLS].mean()

    week4 = wf[wf["week_num"] == 4].set_index("user_id")
    agg_last = week4[LAST_COLS].rename(columns={c: f"{c}_last" for c in LAST_COLS})
    agg_extra_last = week4[EXTRA_LAST_COLS].rename(
        columns={c: f"{c}_last" for c in EXTRA_LAST_COLS}
    )

    agg_static = wf.drop_duplicates("user_id").set_index("user_id")[STATIC_COLS]

    agg_slope = wf.groupby("user_id").apply(_compute_slope, include_groups=False)
    agg_slope.name = "playtime_slope"

    return (
        agg_mean.join(agg_last)
        .join(agg_extra_last)
        .join(agg_static)
        .join(agg_slope)
        .join(labels.set_index("user_id"))
    )


# LSTM NaN fill strategy (see configs/feature_spec.md)
_LSTM_FILL_ZERO = [
    "session_count_wow_change", "playtime_wow_change", "new_game_trial_rate",
]
_LSTM_FILL_ONE = [
    "session_count_vs_baseline", "playtime_vs_baseline",
]
_LSTM_FILL_NEG_ONE = ["days_since_last_payment"]
_LSTM_FILL_ZERO_TAIL = ["payment_frequency_change"]


def reshape_lstm(
    weekly_features: pd.DataFrame, labels: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, user_ids, feature_names)."""
    lstm_feature_cols = [
        c for c in weekly_features.columns if c not in ["user_id", "week_num"]
    ]

    lstm_df = weekly_features.copy()
    lstm_df[_LSTM_FILL_ZERO] = lstm_df[_LSTM_FILL_ZERO].fillna(0.0)
    lstm_df[_LSTM_FILL_ONE] = lstm_df[_LSTM_FILL_ONE].fillna(1.0)
    lstm_df[_LSTM_FILL_NEG_ONE] = lstm_df[_LSTM_FILL_NEG_ONE].fillna(-1)
    lstm_df[_LSTM_FILL_ZERO_TAIL] = lstm_df[_LSTM_FILL_ZERO_TAIL].fillna(0.0)
    # Any remaining NaN (session_duration_std, daily_playtime_cv from low-session weeks) -> 0
    lstm_df[lstm_feature_cols] = lstm_df[lstm_feature_cols].fillna(0.0)

    lstm_df = lstm_df.sort_values(["user_id", "week_num"])
    user_ids = lstm_df["user_id"].unique()
    n_users = len(user_ids)
    n_features = len(lstm_feature_cols)

    X = lstm_df[lstm_feature_cols].values.reshape(n_users, 4, n_features)
    y = labels.set_index("user_id").loc[user_ids, "churn"].values

    return X, y, user_ids, lstm_feature_cols
