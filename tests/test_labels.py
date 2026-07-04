import pandas as pd

from src.dataset.labels import compute_churn_labels


def _users(user_ids):
    return pd.DataFrame({"user_id": user_ids})


def _sessions(rows):
    """rows: list of (user_id, start_time_str)."""
    return pd.DataFrame(
        {
            "user_id": [r[0] for r in rows],
            "start_time": pd.to_datetime([r[1] for r in rows]),
        }
    )


def test_sessions_only_before_prediction_window_churns():
    users = _users(["u1"])
    sessions = _sessions([("u1", "2024-02-20 10:00")])  # week 8, before pred window

    labels = compute_churn_labels(users, sessions)

    assert labels.set_index("user_id").loc["u1", "churn"] == 1


def test_session_inside_prediction_window_does_not_churn():
    users = _users(["u1"])
    sessions = _sessions([("u1", "2024-03-01 10:00")])  # inside weeks 9-10

    labels = compute_churn_labels(users, sessions)

    assert labels.set_index("user_id").loc["u1", "churn"] == 0


def test_boundary_pred_start_is_inclusive():
    users = _users(["u1"])
    sessions = _sessions([("u1", "2024-02-26 00:00")])  # exactly PRED_START

    labels = compute_churn_labels(users, sessions)

    assert labels.set_index("user_id").loc["u1", "churn"] == 0


def test_boundary_pred_end_is_exclusive():
    users = _users(["u1"])
    sessions = _sessions([("u1", "2024-03-11 00:00")])  # exactly PRED_END

    labels = compute_churn_labels(users, sessions)

    assert labels.set_index("user_id").loc["u1", "churn"] == 1


def test_user_with_zero_sessions_anywhere_churns():
    users = _users(["u1"])
    sessions = _sessions([])

    labels = compute_churn_labels(users, sessions)

    assert labels.set_index("user_id").loc["u1", "churn"] == 1
