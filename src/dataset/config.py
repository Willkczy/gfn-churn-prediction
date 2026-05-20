"""Constants for Phase 3 dataset construction.

`PRED_*` are the prediction-window boundaries used to derive the churn label.
`TRAIN_RATIO` / `VAL_RATIO` define a 70/15/15 stratified split. `SEED` makes
the split deterministic. See `configs/temporal_conventions.md` for the
canonical week timeline.
"""

import datetime

PRED_START = datetime.date(2024, 2, 26)  # calendar week 9 start
PRED_END = datetime.date(2024, 3, 11)    # calendar week 11 start (exclusive)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO = 0.15

SEED = 42
