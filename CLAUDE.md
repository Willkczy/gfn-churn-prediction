# CLAUDE.md

## Project
GFN Churn Prediction — end-to-end ML pipeline predicting user churn on a cloud gaming platform (modeled after NVIDIA GeForce NOW). Uses synthetic data, PySpark feature engineering, XGBoost + LSTM dual-model, full MLOps on AWS.

## Current Status
- **Phase 1** (data generation): Complete, merged, tagged v0.1.0-phase1
- **Phase 2** (feature engineering): In progress on `feature/phase2-feature-engineering`
- Working in `notebooks/phase2_feature_engineering.ipynb` first, then extracting to `src/feature_engineering/`

## Temporal Design
- 12-week data window: 2024-01-01 ~ 2024-03-25
- Week 1–4: historical baseline (for trend comparison)
- Week 5–8: observation window (features extracted here)
- Week 9–10: prediction window (zero sessions → churn = 1)
- Week 11–12: buffer for retraining simulation

## Key Files
- `PROJECT_PLAN.md` — phases, progress checklist, architecture decisions
- `configs/data_design.md` — **how data was generated**: personas, decay mechanics, causal relationships, noise. Read before modeling.
- `configs/feature_spec.md` — feature definitions with checkboxes (Phase 2 working reference)
- `notebooks/phase2_feature_engineering.ipynb` — active development notebook (PySpark)
- `notebooks/phase2_feature_engineering_pandas.ipynb` — pandas equivalent for learning/reference
- `src/data_generation/` — completed Phase 1 generators (50K users, 3.55M sessions)
- `data/raw/` — generated parquet files (users, session_logs, game_catalog, subscription_events, payments)

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
- Churn decay for about_to_churn users starts at gen week 6 (0-indexed)
