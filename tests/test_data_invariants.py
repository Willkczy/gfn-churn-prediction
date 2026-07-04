"""Invariant checks against the real generated dataset under data/processed/.

Skipped automatically when that directory (or its expected files) is absent,
e.g. in CI where data is never generated.
"""

import pandas as pd
import pytest

from src.models.config import PROCESSED_DIR

_SPLITS = ("train", "val", "test")
_XGB_FILES = {s: PROCESSED_DIR / f"xgboost_{s}.parquet" for s in _SPLITS}

_DATA_PRESENT = all(p.exists() for p in _XGB_FILES.values())

pytestmark = pytest.mark.data

skip_if_no_data = pytest.mark.skipif(
    not _DATA_PRESENT, reason="data/processed/*.parquet not present"
)


@skip_if_no_data
def test_churn_rate_within_expected_band():
    rates = []
    for path in _XGB_FILES.values():
        df = pd.read_parquet(path, columns=["churn"])
        rates.append(df["churn"].mean())

    for rate in rates:
        assert 0.10 <= rate <= 0.20, f"churn rate {rate:.1%} outside expected 10-20% band"


@skip_if_no_data
def test_train_val_test_user_ids_disjoint():
    id_sets = {}
    for split, path in _XGB_FILES.items():
        df = pd.read_parquet(path, columns=[])
        id_sets[split] = set(df.index)

    assert id_sets["train"].isdisjoint(id_sets["val"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["val"].isdisjoint(id_sets["test"])


@skip_if_no_data
def test_no_persona_column_in_xgboost_files():
    for path in _XGB_FILES.values():
        df = pd.read_parquet(path)
        assert "persona" not in df.columns, f"{path.name} leaks the persona column"
