"""Phase 3 dataset construction.

Entry point: `src.dataset.pipeline.run()` or
`python -m src.dataset.pipeline`. Component modules: labels, aggregation,
split.
"""

from .pipeline import run

__all__ = ["run"]
