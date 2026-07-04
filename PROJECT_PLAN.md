# GFN Churn Prediction — Project Plan

> **Purpose**: Track project phases, progress, and key decisions. For data schemas, see `CLAUDE.md`. For feature definitions, see `configs/feature_spec.md`. For generation logic, see `src/data_generation/`.

---

## 1. Project Overview

**Goal**: Build an end-to-end ML pipeline using synthetic data that mimics GFN (cloud gaming) user behavior, predicting which users will churn. Pipeline spans from data generation to full MLOps on AWS.

**Key Targets**:
- Dual-model: XGBoost (interpretability) + LSTM (sequential patterns)
- Pre-registered ensemble success criterion: at precision ≥ 0.75 on test (threshold frozen on val), ensemble recall ≥ XGBoost recall + 2pp. (Replaced the original "AUC ≥ 0.89" target on 2026-07-05 — XGBoost alone hit 0.969 test AUC, so raw AUC no longer discriminates; see decision log.)
- 42 engineered features (flat) / 38 weekly features (sequential) from 3.55M+ session logs
- SHAP-based feature importance for retention insights
- Full MLOps: CI/CD, monitoring, drift detection, auto-retraining

**Temporal Design** (week numbers are 1-indexed calendar weeks — see [configs/temporal_conventions.md](configs/temporal_conventions.md)):
- 12-week window: 2024-01-01 ~ 2024-03-25
- Week 1–4: historical baseline | Week 5–8: observation (features) | Week 9–10: prediction (churn label) | Week 11–12: buffer
- About_to_churn decay: factor 1.00 → 0.75 → 0.50 → 0.25 across weeks 5–8, hits 0.00 at week 9

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
    - Churn decay logic (about_to_churn: weeks 6–9 progressive decline; calendar-week-indexed — see configs/temporal_conventions.md)
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
- [x] Phase 2: Feature engineering (PySpark) ✅ (PR #1 → develop)
  - [x] Dev environment setup: Docker Compose Spark cluster integrated into project (.devcontainer + infrastructure/docker)
  - [x] Session pattern features (weekly_session_count, avg_session_duration, total_playtime, session_regularity, peak_hour_ratio, weekend_ratio)
  - [x] Engagement decay features (session_count_wow_change, playtime_wow_change, longest_inactive_days, baseline ratios, rolling slope)
  - [x] Streaming quality features (avg_latency, avg_fps, frame_drop_rate, disconnect_rate, avg_bitrate, avg_jitter, packet_loss_avg, crash_exit_ratio)
  - [x] Game diversity features (unique_games_played, genre_entropy, new_game_trial_rate, top_game_concentration)
  - [x] Playtime volatility features (daily_playtime_std, daily_playtime_cv, session_duration_std)
  - [x] Payment & subscription features (aggregated over 4-week)
  - [x] Population normalization (weekly_session_count_norm, total_playtime_min_norm) — removes hot-week effects
  - [x] Combine all features → XGBoost format (flat) + LSTM format (sequential)
  - [x] Extract finalized code into src/feature_engineering/*.py modules
- [x] Phase 3: Dataset construction
  - [x] Churn labels (zero sessions in pred window → churn=1)
  - [x] XGBoost flat aggregation (50K × 43 cols)
  - [x] LSTM 3D tensor (50K × 4 × 38)
  - [x] Stratified train/val/test split (70/15/15, seed=42)
  - [x] Phase 3 validation notebook (14 checks across structure / label / feature)
  - [x] Extract notebook to src/dataset/*.py modules (output byte-identical to notebook)
- [ ] Phase 4: Model training (in progress on `feature/phase4-model-training`; sequencing per IMPLEMENTATION_PLAN.md)
  - [x] Model config, data loaders, shared evaluator (src/models/)
  - [x] XGBoost EDA + training pipeline (notebook, section 1–2)
  - [x] XGBoost v1 baseline trained — test AUC 0.969, PR-AUC 0.709 (models/xgboost_v1_*)
  - [x] Investigate churn-rate gap (M1) — root cause: unguarded session-count noise (`generate_session_logs.py:185`) lets ~25% of about_to_churn users escape the label via ghost sessions; fix folded into M4
  - [ ] Tests + CI safety net before data regen (M2)
  - [ ] Harden data generation v2 + regen + retrain XGBoost v2 (M4; target AUC band 0.85–0.93)
  - [ ] SHAP feature importance on v2 (M5)
  - [ ] LSTM model (M6)
  - [ ] Ensemble — success: recall@precision≥0.75 beats XGBoost by ≥2pp (M7)
- [ ] Phase 5: Experiment tracking (MLflow — pulled forward to before LSTM work, M3)
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

**Current branch**: `feature/phase4-model-training` (from `develop`)
**Next Step**: IMPLEMENTATION_PLAN.md M1 — investigate the churn-rate gap (12.2% actual vs 15–18% designed), read-only, findings to user before any code changes.

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
| 2026-05-20 | Fixed off-by-one in `generate_session_logs.py` decay logic; introduced explicit `calendar_week = week + 1`; created `configs/temporal_conventions.md` as the canonical timeline doc | Loop counter `week` was 0-indexed but the formula was written assuming 1-indexed, so decay-zero landed on calendar week 10 instead of the intended week 9. Result: Phase 3 churn rate was 3.5% instead of the expected ~15%. Fix: branch on `calendar_week`. All docs now consistently use 1-indexed calendar weeks. Requires Phase 1 data regen + Phase 2 rerun. |
| 2026-06-07 | Anchored `.gitignore` `models/` rule to `/models/` | Unanchored rule also matched `src/models/`, silently ignoring Phase 4 source (config, data_loaders, evaluate, xgboost_model). Commit 2b79a29 had landed empty as a result. Source now tracked. |
| 2026-07-05 | Full repo audit → `DEVELOPMENT_PLAN.md` (roadmap) + `IMPLEMENTATION_PLAN.md` (11 milestones, 5 user checkpoints) | Evidence-based planning before continuing Phase 4. Key findings: zero tests despite two past silent-failure bugs; churn rate 12.2% vs designed 15–18%; XGBoost v1 at 0.969 test AUC saturates the old target. |
| 2026-07-05 | Harden synthetic data (v2): randomized decay onset, partial churners, persona overlap, softer quality shift — one regen cycle combined with the churn-gap fix | XGBoost v1 alone hit 0.969 test AUC, leaving no headroom for the LSTM/ensemble story. Target band for retrained baseline: AUC 0.85–0.93, churn rate 13–18%. |
| 2026-07-05 | Replaced "ensemble AUC ≥ 0.89" with pre-registered criterion: recall at precision ≥ 0.75 (val-frozen threshold) must beat XGBoost by ≥ 2pp | Raw AUC no longer discriminates between models on this data; recall-at-precision reflects the retention use case (how many churners caught at acceptable alert quality). Criterion fixed before LSTM training to avoid post-hoc metric shopping. |
| 2026-07-05 | MLflow (Phase 5) pulled forward to before LSTM work; local file store | Metrics were unversioned (`/models/` gitignored). Tracking must exist before the v1→v2 regen comparison and LSTM sweeps. |
| 2026-07-05 | Docs/structure cleanup: README synced to actual state (roadmap ticks, stack status column, real tree), CLAUDE.md status refreshed, pyproject description filled, deleted orphan pre-split intermediates (`lstm_features.npz`, `xgboost_features.parquet`) | README advertised unbuilt components (tests/, workflows, terraform) and unchecked completed phases; orphans were referenced by no code. |
| 2026-07-05 | M1 complete — churn-gap root cause is the unguarded count jitter at `generate_session_logs.py:185` (`max(0, int(rng.normal(weekly_count, 1)))`): fully-decayed weeks have ~15.9% odds of a ghost session, so 24.75% of about_to_churn users escape the churn label. Fix folded into M4, designed jointly with the partial-churner mechanic (CP1 decision). | Fixing the bug alone would raise churn to ~15.9% but make the data easier (escapees are the current hard cases). One regen cycle: bug guard + deliberate hardening land together. Findings independently verified against raw parquet. |

---

*Last updated: 2026-07-05*