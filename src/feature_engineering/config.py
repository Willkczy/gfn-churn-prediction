"""Temporal constants for Phase 2 feature engineering.

Week numbering follows the 1-indexed calendar-week convention defined in
configs/temporal_conventions.md. The constants below name the boundaries
between the four 12-week roles (baseline / observation / prediction / buffer).
"""

import datetime

DATA_START = datetime.date(2024, 1, 1)     # calendar week 1 start
BASELINE_END = datetime.date(2024, 1, 29)  # calendar week 5 start (baseline = weeks 1-4)
OBS_START = datetime.date(2024, 1, 29)     # observation window start (week 5)
OBS_END = datetime.date(2024, 2, 26)       # observation window end / pred start (week 9)
BASELINE_WEEKS = 4
OBS_WEEKS = 4  # number of weeks inside the observation window
