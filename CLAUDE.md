# CLAUDE.md

## Project
GFN Churn Prediction — end-to-end ML pipeline predicting user churn on a cloud gaming platform (modeled after NVIDIA GeForce NOW). Uses synthetic data, PySpark feature engineering, XGBoost + LSTM dual-model, full MLOps on AWS.

## Current Status
- **Phases 1–3** complete and merged to `develop`: data generation (v0.1.0-phase1), PySpark feature engineering (PR #1), dataset construction (PR #2)
- **Phase 4** (model training): in progress on `feature/phase4-model-training` — XGBoost v1 trained (test AUC 0.969); SHAP, LSTM, ensemble pending
- Execution follows `IMPLEMENTATION_PLAN.md` (milestones M1–M11 with user checkpoints); background in `DEVELOPMENT_PLAN.md`
- Known issue: churn rate is 12.2% but the design predicts 15–18% — investigate (IMPLEMENTATION_PLAN M1) before further model work

## Temporal Design
- 12-week data window: 2024-01-01 ~ 2024-03-25 (week numbers everywhere are **1-indexed calendar weeks** — see [configs/temporal_conventions.md](configs/temporal_conventions.md) for the canonical timeline)
- Week 1–4: historical baseline (for trend comparison)
- Week 5–8: observation window (features extracted here; about_to_churn decay factor 1.00 → 0.25)
- Week 9–10: prediction window (zero sessions → churn = 1; about_to_churn decay factor 0.00)
- Week 11–12: buffer for retraining simulation

## Key Files
- `PROJECT_PLAN.md` — phases, progress checklist, architecture decisions
- `IMPLEMENTATION_PLAN.md` — milestone-by-milestone execution plan (follow this when implementing)
- `configs/data_design.md` — **how data was generated**: personas, decay mechanics, causal relationships, noise. Read before modeling.
- `configs/feature_spec.md` — feature definitions and output-format spec
- `notebooks/phase4_xgboost.ipynb` — active development notebook (EDA + XGBoost training)
- `notebooks/phase2_feature_engineering_pandas.ipynb` — pandas equivalent of the Spark pipeline, for learning/reference
- `src/data_generation/` — Phase 1 generators (50K users, 3.55M sessions)
- `src/feature_engineering/` — Phase 2 PySpark pipeline (runs in the Spark dev container)
- `src/dataset/` — Phase 3 labels, split, XGBoost/LSTM formatting
- `src/models/` — Phase 4 config, loaders, evaluator, XGBoost; LSTM/ensemble to come
- `data/raw/`, `data/processed/`, `models/` — generated artifacts (gitignored, regenerable)

## Conventions
- **Commits**: conventional commits — `feat/fix/docs/chore/data(scope): message`
- **Branches**: `feature/phase{N}-{description}`, merge to `develop`, then `develop` → `main`
- **Workflow**: notebook-first exploration → extract to `src/` modules
- **Package manager**: uv
- **Feature engineering**: PySpark (Docker Spark cluster in infrastructure/docker/)

## Data Schema (5 tables)
- `users` (50K) — user_id, signup_date, subscription_tier, device_type, region, referral_source, age_group, gender, persona
- `session_logs` (3.55M) — session_id, user_id, game_id, start_time, end_time, stream_resolution, avg_latency_ms, avg_fps, total_frame_drops, disconnect_count, input_lag_ms, avg_bitrate_mbps, avg_jitter_ms, packet_loss_rate, exit_type
- `game_catalog` (200) — game_id, game_title, genre, publisher, popularity_tier, supported_max_resolution
- `subscription_events` (38.7K) — event_id, user_id, event_date, event_type, from_tier, to_tier
- `payments` (238.9K) — payment_id, user_id, payment_date, amount_usd, payment_type, payment_method, status

## Important Notes
- `persona` column in users must be EXCLUDED from model features (data leakage)
- Free tier sessions capped at 60 min in generated data
- Churn decay for about_to_churn users starts at calendar week 6 (1-indexed), reaches zero at week 9 — see `configs/temporal_conventions.md`
