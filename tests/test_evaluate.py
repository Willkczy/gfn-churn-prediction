import json

import numpy as np
import pytest

from src.models.evaluate import compute_metrics, find_best_threshold


def test_compute_metrics_hand_checked_confusion_matrix():
    # threshold 0.5: preds = [0, 1, 0, 1, 1] vs true = [0, 1, 1, 0, 1]
    # tn=1 (idx0), fn=1 (idx2), fp=1 (idx3), tp=2 (idx1,4)
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.6, 0.8])

    metrics = compute_metrics(y_true, y_proba, threshold=0.5)

    cm = metrics["confusion_matrix"]
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (1, 1, 1, 2)
    precision = 2 / (2 + 1)
    recall = 2 / (2 + 1)
    expected_f1 = 2 * precision * recall / (precision + recall)
    assert metrics["f1"] == pytest.approx(expected_f1)
    assert metrics["precision"] == pytest.approx(precision)
    assert metrics["recall"] == pytest.approx(recall)
    assert metrics["threshold"] == pytest.approx(0.5)
    assert metrics["n_samples"] == 5
    assert metrics["n_positive"] == 3


def test_find_best_threshold_known_optimum():
    # Perfect separation at 0.5: score < 0.5 -> negative, score >= 0.5 -> positive.
    # F1 = 1.0 for any threshold in (0.3, 0.7]; the sweep should land in that band.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold, best_f1 = find_best_threshold(y_true, y_proba, metric="f1")

    assert 0.05 <= threshold <= 0.95
    assert best_f1 == pytest.approx(1.0)
    assert 0.3 < threshold <= 0.7


def test_metrics_dict_round_trips_through_json():
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.6, 0.8])

    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    round_tripped = json.loads(json.dumps(metrics))

    assert round_tripped == metrics
