import numpy as np
import pandas as pd
import pytest

from src.dataset.aggregation import (
    EXTRA_LAST_COLS,
    LAST_COLS,
    MEAN_COLS,
    STATIC_COLS,
    aggregate_xgboost,
    reshape_lstm,
)

# Pure-mean columns: in MEAN_COLS but not also re-extracted as an EXTRA_LAST_COLS last value.
_PURE_MEAN_COLS = [c for c in MEAN_COLS if c not in EXTRA_LAST_COLS]

_STATIC_VALUES = {
    "u1": [1, 100, 2, 0, 50.0, 4, 0, 0, 5, 0.1],
    "u2": [2, 200, 1, 1, 75.0, 2, 1, 1, 10, -0.2],
}

_TOTAL_PLAYTIME = {
    "u1": [100.0, 200.0, 300.0, 400.0],  # linear, slope = +100/week
    "u2": [400.0, 300.0, 200.0, 100.0],  # linear, slope = -100/week
}


def _build_weekly_features(multipliers: dict[str, float]) -> pd.DataFrame:
    """2 users x 4 weeks. `multipliers` scales the per-user pure-mean/last values
    so u1 and u2 produce distinguishable aggregates.
    """
    rows = []
    for user_id, mult in multipliers.items():
        for week_num in (1, 2, 3, 4):
            row = {"user_id": user_id, "week_num": week_num}
            for col in _PURE_MEAN_COLS:
                row[col] = week_num * mult
            row["weekly_session_count"] = week_num * mult
            row["total_playtime_min"] = _TOTAL_PLAYTIME[user_id][week_num - 1]
            row["crash_exit_ratio"] = 0.1 * week_num * mult
            for col in LAST_COLS:
                row[col] = week_num * 10 * mult
            for col, val in zip(STATIC_COLS, _STATIC_VALUES[user_id]):
                row[col] = val
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def weekly_features():
    return _build_weekly_features({"u1": 1, "u2": 2})


@pytest.fixture
def labels():
    return pd.DataFrame({"user_id": ["u1", "u2"], "churn": [1, 0]})


def test_mean_columns_average_correctly(weekly_features, labels):
    result = aggregate_xgboost(weekly_features, labels)

    assert result.loc["u1", "weekly_session_count"] == pytest.approx(2.5)
    assert result.loc["u2", "weekly_session_count"] == pytest.approx(5.0)
    assert result.loc["u1", "crash_exit_ratio"] == pytest.approx(0.25)
    assert result.loc["u2", "crash_exit_ratio"] == pytest.approx(0.5)
    assert result.loc["u1", "total_playtime_min"] == pytest.approx(250.0)
    assert result.loc["u2", "total_playtime_min"] == pytest.approx(250.0)


def test_last_columns_come_from_week_4(weekly_features, labels):
    result = aggregate_xgboost(weekly_features, labels)

    # LAST_COLS: week_num * 10 * mult, week 4 -> 40 * mult
    assert result.loc["u1", "session_count_wow_change_last"] == pytest.approx(40)
    assert result.loc["u2", "session_count_wow_change_last"] == pytest.approx(80)
    # EXTRA_LAST_COLS drawn from the same source columns as MEAN_COLS
    assert result.loc["u1", "weekly_session_count_last"] == pytest.approx(4)
    assert result.loc["u2", "weekly_session_count_last"] == pytest.approx(8)
    assert result.loc["u1", "total_playtime_min_last"] == pytest.approx(400.0)
    assert result.loc["u2", "total_playtime_min_last"] == pytest.approx(100.0)


def test_playtime_slope_matches_hand_computed_value(weekly_features, labels):
    result = aggregate_xgboost(weekly_features, labels)

    assert result.loc["u1", "playtime_slope"] == pytest.approx(100.0)
    assert result.loc["u2", "playtime_slope"] == pytest.approx(-100.0)


def test_output_index_is_user_id(weekly_features, labels):
    result = aggregate_xgboost(weekly_features, labels)

    assert result.index.name == "user_id"
    assert set(result.index) == {"u1", "u2"}


def test_churn_column_joins_correctly(weekly_features, labels):
    result = aggregate_xgboost(weekly_features, labels)

    assert result.loc["u1", "churn"] == 1
    assert result.loc["u2", "churn"] == 0


def test_reshape_lstm_shape_and_alignment(weekly_features, labels):
    X, y, user_ids, feature_names = reshape_lstm(weekly_features, labels)

    n_features = len(weekly_features.columns) - 2  # minus user_id, week_num
    assert X.shape == (2, 4, n_features)
    assert list(user_ids) == ["u1", "u2"]
    assert list(y) == [1, 0]
    assert len(feature_names) == n_features


def test_reshape_lstm_nan_fill_strategy(labels):
    wf = _build_weekly_features({"u1": 1, "u2": 2})
    wf.loc[(wf["user_id"] == "u1") & (wf["week_num"] == 1), "session_count_wow_change"] = np.nan
    wf.loc[(wf["user_id"] == "u1") & (wf["week_num"] == 1), "session_count_vs_baseline"] = np.nan
    wf.loc[(wf["user_id"] == "u1") & (wf["week_num"] == 1), "days_since_last_payment"] = np.nan
    wf.loc[(wf["user_id"] == "u1") & (wf["week_num"] == 1), "payment_frequency_change"] = np.nan
    wf.loc[(wf["user_id"] == "u1") & (wf["week_num"] == 1), "avg_latency"] = np.nan

    X, y, user_ids, feature_names = reshape_lstm(wf, labels)

    u1_week1 = X[list(user_ids).index("u1"), 0, :]
    col_idx = {name: i for i, name in enumerate(feature_names)}
    assert u1_week1[col_idx["session_count_wow_change"]] == pytest.approx(0.0)
    assert u1_week1[col_idx["session_count_vs_baseline"]] == pytest.approx(1.0)
    assert u1_week1[col_idx["days_since_last_payment"]] == pytest.approx(-1.0)
    assert u1_week1[col_idx["payment_frequency_change"]] == pytest.approx(0.0)
    assert u1_week1[col_idx["avg_latency"]] == pytest.approx(0.0)  # catch-all fill


def test_reshape_lstm_raises_on_incomplete_user_weeks(labels):
    wf = _build_weekly_features({"u1": 1, "u2": 2})
    wf = wf.drop(wf[(wf["user_id"] == "u1") & (wf["week_num"] == 1)].index)

    with pytest.raises(ValueError):
        reshape_lstm(wf, labels)
