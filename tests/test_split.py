import numpy as np
import pandas as pd

from src.dataset.split import stratified_split


def _population(n_users=1000, churn_rate=0.15, seed=0):
    rng = np.random.default_rng(seed)
    user_ids = np.array([f"u{i}" for i in range(n_users)])
    churn = (rng.random(n_users) < churn_rate).astype(int)
    labels = pd.DataFrame({"user_id": user_ids, "churn": churn})
    return labels, user_ids


def test_splits_are_disjoint_and_cover_input():
    labels, user_ids = _population()

    splits = stratified_split(labels, user_ids, seed=42)

    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(user_ids)


def test_split_sizes_approximate_70_15_15():
    labels, user_ids = _population(n_users=1000)

    splits = stratified_split(labels, user_ids, seed=42, train_ratio=0.70, val_ratio=0.15)

    n = len(user_ids)
    assert abs(len(splits["train"]) - 0.70 * n) <= 2
    assert abs(len(splits["val"]) - 0.15 * n) <= 2
    assert abs(len(splits["test"]) - 0.15 * n) <= 2


def test_churn_rate_per_split_close_to_population():
    labels, user_ids = _population(n_users=2000, churn_rate=0.15)
    population_rate = labels["churn"].mean()

    splits = stratified_split(labels, user_ids, seed=42)

    labels_idx = labels.set_index("user_id")
    for ids in splits.values():
        split_rate = labels_idx.loc[ids, "churn"].mean()
        assert abs(split_rate - population_rate) <= 0.01


def test_same_seed_gives_identical_splits():
    labels, user_ids = _population()

    splits_a = stratified_split(labels, user_ids, seed=42)
    splits_b = stratified_split(labels, user_ids, seed=42)

    for key in splits_a:
        assert list(splits_a[key]) == list(splits_b[key])


def test_different_seed_gives_different_splits():
    labels, user_ids = _population()

    splits_a = stratified_split(labels, user_ids, seed=42)
    splits_b = stratified_split(labels, user_ids, seed=7)

    assert list(splits_a["train"]) != list(splits_b["train"])
