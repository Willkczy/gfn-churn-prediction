"""Phase 2 feature engineering pipeline.

Entry point: `src.feature_engineering.pipeline.run()` or
`python -m src.feature_engineering.pipeline`. Per-section computations are
in the sibling modules (session_patterns, engagement_decay, ...).
"""

from .pipeline import build_weekly_features, run

__all__ = ["build_weekly_features", "run"]
