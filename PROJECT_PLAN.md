# GFN Churn Prediction — Project Plan

> **Purpose**: Track project phases, progress, and key decisions. For data schemas, see `CLAUDE.md`. For feature definitions, see `configs/feature_spec.md`. For generation logic, see `src/data_generation/`.

---

## 1. Project Overview

**Goal**: Build an end-to-end ML pipeline using synthetic data that mimics GFN (cloud gaming) user behavior, predicting which users will churn. Pipeline spans from data generation to full MLOps on AWS.

**Key Targets**:
- Dual-model: XGBoost (interpretability) + LSTM (sequential patterns)
- Ensemble AUC ≥ 0.89
- 50+ engineered features from 3.55M+ session logs
- SHAP-based feature importance for retention insights
- Full MLOps: CI/CD, monitoring, drift detection, auto-retraining

**Temporal Design**:
- 12-week window: 2024-01-01 ~ 2024-03-25
- Week 1–4: historical baseline | Week 5–8: observation (features) | Week 9–10: prediction (churn label) | Week 11–12: buffer

---

## 2. Project Phases & Timeline

| Phase | Description | Est. Duration |
|---|---|---|
| Phase 1 | Synthetic data generation | 2–3 days |
| Phase 2 | Feature engineering (PySpark) | 3–4 days |
| Phase 3 | Dataset construction (labeling, split, formatting) | 1–2 days |
| Phase 4 | Model training (XGBoost + LSTM + Ensemble) | 4–5 days |
| Phase 5 | Experiment tracking (MLflow) | 1 day |
| Phase 6 | AWS deployment (SageMaker, S3, ECR) | 3–4 days |
| Phase 7 | MLOps (CI/CD, monitoring, retraining) | 4–5 days |

**Total**: ~3–4 weeks

---

## 3. Current Progress

- [x] Project planning & schema design
- [x] Phase 1: Synthetic data generation ✅ (v0.1.0-phase1)
  - [x] Environment setup (uv, git repo, folder structure, branching)
  - [x] `generate_game_catalog.py` — 200 games, 8 genres, popularity power law
  - [x] `generate_users.py` — 50K users (small), 4 personas, persona-based tier assignment, Taiwan counties as regions
  - [x] `generate_session_logs.py` — 3.55M sessions (98.5 MB) with:
    - Persona-based behavior profiles (session frequency, duration, quality)
    - Churn decay logic (weeks 8-11 progressive decline)
    - Dynamic causal: poor quality → session cut short + forced exit type
    - Dynamic causal: bad experience → 35% chance to skip next session
    - Hot event weeks: 2-3 random weeks with 1.3-1.8x activity boost
    - Noise injection: AFK outliers (2%), latency spikes (3%), late night sessions (1%)
    - Free tier 60-min session cap
    - Game selection weighted by popularity tier (S=10x, A=5x, B=2x, C=1x)
  - [x] `generate_subscription_events.py` — 38.7K events, persona-based temporal patterns (cancel/downgrade late-weighted for churn users)
  - [x] `generate_payments.py` — 238.9K payments, 3 types (subscription/day_pass/in_app), churn users: 10% failed + 8% refund rate
  - [x] `run_all.py` — unified entry point, supports SCALE and SEED env vars
  - [ ] Data validation & EDA notebook (optional, can do in Phase 2)
- [ ] Phase 2: Feature engineering (PySpark)
  - [x] Dev environment setup: Docker Compose Spark cluster integrated into project (.devcontainer + infrastructure/docker)
  - [x] Session pattern features (weekly_session_count, avg_session_duration, total_playtime, session_regularity, peak_hour_ratio, weekend_ratio)
  - [x] Engagement decay features (session_count_wow_change, playtime_wow_change, longest_inactive_days, baseline ratios, rolling slope)
  - [x] Streaming quality features (avg_latency, avg_fps, frame_drop_rate, disconnect_rate, avg_bitrate, avg_jitter, packet_loss_avg, crash_exit_ratio)
  - [ ] Game diversity features (unique_games_played, genre_entropy, new_game_trial_rate, top_game_concentration)
  - [ ] Playtime volatility features (daily_playtime_std, daily_playtime_cv, session_duration_std)
  - [ ] Payment & subscription features (aggregated over 4-week)
  - [ ] Combine all features → XGBoost format (flat) + LSTM format (sequential)
  - [ ] Extract finalized code into src/feature_engineering/*.py modules
- [ ] Phase 3: Dataset construction
- [ ] Phase 4: Model training
- [ ] Phase 5: Experiment tracking
- [ ] Phase 6: AWS deployment
- [ ] Phase 7: MLOps

**Generated data summary (small scale)**:
| Table | Rows | File Size |
|---|---|---|
| game_catalog | 200 | 0.0 MB |
| users | 50,000 | 0.5 MB |
| session_logs | 3,557,265 | 98.5 MB |
| subscription_events | 38,763 | 0.5 MB |
| payments | 238,948 | 2.3 MB |

**Current branch**: `feature/phase2-feature-engineering` (from `develop`)
**Next Step**: Phase 2 — Implement remaining feature sections (Game Diversity, Playtime Volatility, Subscription & Payment).

---

## 4. Key Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2025-03-12 | 5 tables: users, session_logs, game_catalog, subscription_events, payments | Payments added because payment behavior was an important feature in the original project |
| 2025-03-12 | Added streaming quality fields (bitrate, jitter, packet_loss) to session_logs | GFN is a streaming service; network quality directly impacts user experience and churn |
| 2025-03-12 | Added demographic fields (age_group, gender) to users | Enables demographic-based analysis and feature engineering |
| 2025-03-12 | AWS as target cloud platform | User preference for practicing AWS services |
| 2025-03-12 | Full MLOps target (CI/CD, monitoring, drift detection) | Maximize learning and interview readiness |
| 2025-03-12 | Git Flow simplified branching with conventional commits | Balance between structure and practicality for solo project |
| 2025-03-12 | Two-tier data volume: small (50K users/3M sessions) for dev, full (200K/30M+) for Spark | Fast iteration during development, realistic scale for Spark practice |
| 2025-03-12 | Regions use Taiwan counties instead of global regions | Matches original project context (Taiwan Mobile collaboration) |
| 2025-03-12 | Added dynamic causal relationships in session generation | Poor quality → early exit, bad experience → skip next session; makes data more realistic and gives LSTM meaningful sequential patterns |
| 2025-03-12 | Added hot event weeks mechanism | Simulates new game launches causing activity spikes; adds non-stationarity to time series |
| 2025-03-12 | Added noise injection layer (AFK, latency spikes, late night) | Prevents overly clean synthetic data; real data has unexplainable outliers |
| 2025-03-12 | Python environment managed with uv | User preference |
| 2025-03-13 | Integrated Docker Spark cluster into project repo (.devcontainer + infrastructure/docker) | Self-contained dev environment: clone → open in Dev Container → ready. Spark 3.5.6 + Java 17 + 2 workers. PostgreSQL/pgAdmin included for potential Phase 5-6 use |
| 2025-03-13 | Feature engineering developed in notebook first, then extracted to .py modules | Notebook for interactive exploration and validation; .py modules for reproducible pipeline |
| 2025-03-13 | Feature design decisions: wow_change week1=null, rolling_slope at aggregation layer, frame_drop_rate simplified to AVG(drops), quality_downgrade_count deferred, new_game_trial_rate week1=null, 0-session days included in daily stats, payment features aggregated over 4 weeks only | Balances completeness with implementation complexity; defers uncertain features |
| 2025-03-19 | Shifted observation window to week 5–8, prediction to week 9–10 | Original week 1–4 obs window preceded the churn decay onset (week 8), making churn labels ineffective. New placement captures early decay signals in obs and meaningful churn in prediction window. Week 1–4 retained as historical baseline for trend features. |
| 2025-03-19 | Split PROJECT_PLAN into lean plan + CLAUDE.md + configs/feature_spec.md | Original 596-line plan was too large. Data schemas/generation logic now live in code. Feature spec is a working checklist in configs/. CLAUDE.md provides AI-agent context. |

---

*Last updated: 2025-03-22*