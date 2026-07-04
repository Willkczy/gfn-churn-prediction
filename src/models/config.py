"""Phase 4 model hyperparameters and paths.

All seeds default to 42 (matches the Phase 3 split seed). Hyperparameters
were chosen by the rationale documented at the top of each model module —
update the docstring there if you tune them.
"""

from pathlib import Path

SEED = 42

# Resolve repo root from this file's location so paths work regardless
# of the caller's cwd (notebook in notebooks/, script in repo root, etc.).
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
XGB_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 30,
    "tree_method": "hist",
    "random_state": SEED,
    # scale_pos_weight is computed from the training set at fit time
}

# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
LSTM_PARAMS: dict = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "head_hidden": 32,
}

LSTM_TRAIN_PARAMS: dict = {
    "batch_size": 256,
    "max_epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 3,
    "early_stopping_patience": 5,  # epochs on val PR-AUC plateau
    "grad_clip_norm": 1.0,
}
