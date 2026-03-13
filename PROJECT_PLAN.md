# GFN Churn Prediction — Full Project Plan

> **Purpose**: This document captures the complete project plan, data schemas, tech stack, and development workflow. It is designed so that any developer or AI agent can pick up the project at any stage and continue seamlessly.

---

## 1. Project Overview

### Background
This project recreates a real-world churn prediction system originally built for **NVIDIA GeForce NOW (GFN)** — a cloud gaming streaming service. The original project was a research collaboration between Taiwan Mobile and NTU (Jan–Jun 2024).

### Objective
Build an end-to-end ML pipeline using **synthetic data** that mimics GFN user behavior, predicting which users will churn (stop using the service). The pipeline spans from data generation to full MLOps deployment on AWS.

### Key Results Target
- Dual-model framework: **XGBoost** (interpretability) + **LSTM** (sequential patterns)
- Ensemble AUC target: **≥ 0.89**
- 50+ engineered behavioral features from 3M+ session logs
- SHAP-based feature importance for product retention insights

---

## 2. Data Schema (5 Tables)

### 2.1 `users` — User Profile (~50K rows)

| Column | Type | Description | Example Values |
|---|---|---|---|
| user_id | string | Unique identifier | `U00001` |
| signup_date | date | Registration date | `2023-06-15` |
| subscription_tier | string | Current plan | free / priority / ultimate |
| device_type | string | Primary device | PC, Mac, Chromebook, SHIELD, Mobile |
| region | string | Geographic region | NA, EU, APAC, LATAM |
| referral_source | string | Acquisition channel | organic, ad, friend_referral, bundled |
| age_group | string | Age bracket | 18-24, 25-34, 35-44, 45+ |
| gender | string | Gender | M, F, other, undisclosed |

**Notes**:
- `subscription_tier` reflects the user's tier at the time of snapshot; historical changes are tracked in `subscription_events`.
- Distribution target: ~60% free, ~30% priority, ~10% ultimate.
- Churn rate target: ~15-20% overall, higher for free tier.

### 2.2 `session_logs` — Streaming Session Records (~3M+ rows)

| Column | Type | Description | Example Values |
|---|---|---|---|
| session_id | string | Unique identifier | `S00000001` |
| user_id | string | FK → users | `U00001` |
| game_id | string | FK → game_catalog | `G0042` |
| start_time | timestamp | Session start | `2024-01-15 20:30:00` |
| end_time | timestamp | Session end | `2024-01-15 22:15:00` |
| stream_resolution | string | Streaming resolution | 720p, 1080p, 1440p, 4K |
| avg_latency_ms | float | Average network latency | 15.0 – 200.0 |
| avg_fps | float | Average frame rate | 24.0 – 120.0 |
| total_frame_drops | int | Total dropped frames | 0 – 500 |
| disconnect_count | int | Number of disconnections | 0 – 5 |
| input_lag_ms | float | Average input delay | 10.0 – 150.0 |
| avg_bitrate_mbps | float | Average stream bitrate | 5.0 – 50.0 |
| avg_jitter_ms | float | Network jitter | 1.0 – 80.0 |
| packet_loss_rate | float | Packet loss ratio (0–1) | 0.0 – 0.15 |
| exit_type | string | How session ended | normal, crash, disconnect, timeout |

**Notes**:
- Session duration derived from `end_time - start_time`. Typical range: 10 min – 6 hours.
- Churn-prone users should show degrading patterns: increasing latency, more disconnects, shorter sessions, more crash/disconnect exits in final weeks.
- Free tier users have session queue wait times and 1-hour limits; priority/ultimate do not.

### 2.3 `game_catalog` — Game Information (~200 rows)

| Column | Type | Description | Example Values |
|---|---|---|---|
| game_id | string | Unique identifier | `G0042` |
| game_title | string | Game name | `Synthetic RPG Alpha` |
| genre | string | Game genre | FPS, RPG, Racing, Strategy, Sports, Casual, Simulation, Battle_Royale |
| publisher | string | Publisher name | `Publisher_A` |
| popularity_tier | string | Platform popularity rank | S, A, B, C |
| supported_max_resolution | string | Max supported stream quality | 1080p, 1440p, 4K |

**Notes**:
- Use synthetic game names (not real titles) to avoid IP issues.
- ~8 genres, ~20 publishers, popularity follows power law (few S-tier, many C-tier).

### 2.4 `subscription_events` — Subscription Changes

| Column | Type | Description | Example Values |
|---|---|---|---|
| event_id | string | Unique identifier | `SE00001` |
| user_id | string | FK → users | `U00001` |
| event_date | date | Event date | `2024-02-10` |
| event_type | string | Type of change | upgrade, downgrade, cancel, renew |
| from_tier | string (nullable) | Previous tier | free, priority, ultimate, NULL |
| to_tier | string (nullable) | New tier | free, priority, ultimate, NULL |

**Notes**:
- `cancel` events: `to_tier = NULL`.
- `renew` events: `from_tier = to_tier`.
- Downgrade events are strong churn signals.

### 2.5 `payments` — Payment Records

| Column | Type | Description | Example Values |
|---|---|---|---|
| payment_id | string | Unique identifier | `PAY00001` |
| user_id | string | FK → users | `U00001` |
| payment_date | date | Transaction date | `2024-01-01` |
| amount_usd | float | Payment amount | 0.0 – 19.99 |
| payment_type | string | What was paid for | subscription, day_pass, in_app |
| payment_method | string | Payment method | credit_card, paypal, gift_card |
| status | string | Transaction result | success, failed, refunded |

**Notes**:
- Free tier users may still have `day_pass` purchases.
- Typical pricing: priority ~$9.99/mo, ultimate ~$19.99/mo, day_pass ~$3.99.
- Refund and failed payment patterns are strong churn indicators.
- Payment frequency decline over time is a key feature.

---

## 3. Synthetic Data Generation Strategy

### 3.1 Data Volume Strategy (Two-Tier)

Development uses a small dataset for fast iteration; the full-scale dataset is for Spark practice and production simulation.

| Table | Small (dev) | Full (Spark) |
|---|---|---|
| users | 50K | 200K |
| session_logs | ~3M | 30M–50M |
| game_catalog | 200 | 200 |
| subscription_events | ~100K | ~500K |
| payments | ~150K | ~800K |

- **Small dataset**: Used during Phase 1–4 development. Pandas-friendly, fast feedback loop.
- **Full dataset**: Generated once logic is validated. session_logs targets 5–10 GB parquet, large enough to justify PySpark for feature engineering.
- Data generation scripts should accept a `scale` parameter (`small` / `full`) to switch between the two.

### 3.2 User Personas

To produce realistic churn patterns, users are generated under 4 behavioral personas:

| Persona | % of Users | Behavior Pattern |
|---|---|---|
| **Hardcore** | 20% | High frequency, long sessions, stable quality, ultimate/priority tier, low churn |
| **Regular** | 35% | Moderate frequency, mix of games, priority tier, moderate churn |
| **Casual** | 30% | Low frequency, short sessions, free tier dominant, higher churn |
| **About-to-churn** | 15% | Starts like Regular/Casual, then shows decay: declining session frequency, rising latency tolerance, shorter durations, more disconnects in final 1-2 weeks |

### 3.3 Temporal Design
- **Data window**: 12 weeks total
- **Observation period**: Week 1–4 (features extracted from here)
- **Prediction window**: Week 5–6 (if zero sessions → churn = 1)
- **Buffer / future data**: Week 7–12 (for model retraining simulation)

### 3.4 Churn Signal Patterns (for about-to-churn users)
- Session count drops 40-70% week-over-week in final 2 weeks
- Session duration shortens by 30-50%
- `exit_type` shifts from mostly `normal` to more `crash`/`disconnect`
- `avg_latency_ms` and `input_lag_ms` increase (user may switch to worse network or care less)
- Game diversity narrows (plays fewer unique games)
- Payment failures or refund requests appear
- Possible downgrade event in `subscription_events`

### 3.5 Per-Table Generation Logic

#### 3.5.1 `game_catalog`
- 200 games with synthetic names (format: `{Genre}_Game_{number}`)
- **Genre**: 8 types (FPS, RPG, Racing, Strategy, Sports, Casual, Simulation, Battle_Royale), uniformly distributed
- **Publisher**: 20 anonymous publishers (Publisher_A ~ Publisher_T), uniformly distributed
- **Popularity tier**: Power law — S: 5%, A: 15%, B: 35%, C: 45%
- **Max resolution**: 1080p: 40%, 1440p: 35%, 4K: 25%
- Seed-based reproducibility via `np.random.default_rng(seed)`

#### 3.5.2 `users`
- Scale: small=50K, full=200K users
- **Persona assignment**: hardcore 20%, regular 35%, casual 30%, about_to_churn 15%
- **Subscription tier is persona-dependent** (not random):

| Persona | free | priority | ultimate |
|---|---|---|---|
| hardcore | 5% | 35% | 60% |
| regular | 20% | 55% | 25% |
| casual | 70% | 25% | 5% |
| about_to_churn | 50% | 35% | 15% |

- **Signup date**: Uniformly distributed across 2023-01-01 ~ 2023-12-31
- **Device**: PC 45%, Mobile 20%, Mac 15%, Chromebook 10%, SHIELD 10%
- **Region**: Taiwan counties, weighted by approximate population (New Taipei 17%, Kaohsiung 12%, Taichung 12%, Taipei 11%, Taoyuan 10%, etc.)
- **Referral**: organic 40%, ad 25%, friend_referral 20%, bundled 15%
- **Age**: 18-24: 30%, 25-34: 35%, 35-44: 20%, 45+: 15%
- **Gender**: M 55%, F 30%, other 5%, undisclosed 10%
- `persona` column is kept for downstream generation but must be excluded from model features (data leakage)

#### 3.5.3 `session_logs`

**Base behavior profiles (per persona)**:

| Persona | Sessions/week (μ,σ) | Duration min (μ,σ) | Peak hour ratio | Weekend ratio |
|---|---|---|---|---|
| hardcore | (14, 3) | (150, 40) | 0.5 | 0.4 |
| regular | (6, 2) | (90, 30) | 0.6 | 0.5 |
| casual | (2, 1) | (40, 15) | 0.7 | 0.6 |
| about_to_churn | (5, 2) | (80, 25) | 0.6 | 0.5 |

**Streaming quality profiles (per tier)**:

| Tier | Latency (μ,σ) | FPS (μ,σ) | Bitrate (μ,σ) | Jitter (μ,σ) | Packet loss (μ,σ) |
|---|---|---|---|---|---|
| free | (50, 20) | (45, 10) | (10, 3) | (30, 15) | (0.05, 0.03) |
| priority | (30, 10) | (55, 8) | (25, 5) | (15, 8) | (0.02, 0.01) |
| ultimate | (18, 5) | (58, 5) | (40, 8) | (8, 4) | (0.008, 0.005) |

**Resolution distribution by tier**:

| Tier | 720p | 1080p | 1440p | 4K |
|---|---|---|---|---|
| free | 50% | 40% | 8% | 2% |
| priority | 10% | 45% | 35% | 10% |
| ultimate | 2% | 15% | 40% | 43% |

**Exit type weights**:
- Normal: [0.85, 0.05, 0.07, 0.03] (normal, crash, disconnect, timeout)
- Degraded (churn users weeks 8+): [0.50, 0.18, 0.22, 0.10]

**Churn decay mechanism** (about_to_churn, weeks 8–11):
- `decay_factor` = 1.0 − 0.25 × (week − 7) → 0.75, 0.50, 0.25, 0.0
- `duration_multiplier` = 0.7 − 0.15 × (week − 8), min 0.3
- Uses degraded exit weights

**Dynamic causal: quality → duration**:
- Latency > 100ms → probability `min(0.6, (latency-100)/200)` to quit early, cutting duration to 20-70%
- Each disconnect has 30% chance to terminate session (duration × 0.3-0.8)
- Forced exit type = "disconnect" when triggered

**Dynamic causal: bad experience → skip next session**:
- Tracked via `had_bad_experience` flag (set when exit_type is crash/disconnect or latency > 120ms)
- Next session has 35% chance to be skipped entirely

**Hot event weeks**:
- 2-3 randomly selected weeks across the 12-week window (global, shared by all users)
- Activity multiplier: 1.3x–1.8x on weekly session count

**Noise injection** (applied post-generation):
- 2% AFK: session extended to 3-12 hours, fps drops to 1-10, exit_type = timeout
- 3% latency spike: latency 150-400ms, jitter 50-150ms, packet_loss 0.05-0.2
- 1% late night: session moved to 0:00-5:00

**Other rules**:
- Free tier: session capped at 60 minutes
- Game selection: weighted by popularity (S=10x, A=5x, B=2x, C=1x)
- Peak hours: 19:00-23:59; off-peak: 08:00-18:59

#### 3.5.4 `subscription_events`

**Event probabilities by persona** (probability of at least one event in 12-week window):

| Persona | upgrade | downgrade | cancel | renew |
|---|---|---|---|---|
| hardcore | 5% | 2% | 1% | 80% |
| regular | 15% | 10% | 5% | 60% |
| casual | 10% | 15% | 10% | 40% |
| about_to_churn | 3% | 35% | 40% | 15% |

**Temporal patterns**:
- Renew / upgrade: can happen any time in the window
- Downgrade: biased toward second half (day range: obs_days//2 ~ obs_days)
- Cancel: biased toward last third (day range: obs_days×2/3 ~ obs_days)

**Logic**:
- Events are generated in order: renew → upgrade → downgrade → cancel
- `current_tier` updates after each event, so subsequent events reference the correct tier
- Upgrade from ultimate or downgrade from free returns None (event skipped)
- Cancel sets `to_tier = None`

#### 3.5.5 `payments`

**Pricing**:
- Subscription: free=$0, priority=$9.99/mo, ultimate=$19.99/mo
- Day pass: $3.99
- In-app: $0.99 (30%), $1.99 (30%), $2.99 (20%), $4.99 (15%), $9.99 (5%)

**Payment behavior by persona**:

| Persona | Day pass/mo (μ,σ) | In-app/mo (μ,σ) | Failed prob | Refund prob |
|---|---|---|---|---|
| hardcore | (0.2, 0.1) | (1.5, 0.5) | 2% | 1% |
| regular | (0.5, 0.3) | (0.8, 0.4) | 3% | 2% |
| casual | (1.0, 0.5) | (0.3, 0.2) | 5% | 3% |
| about_to_churn | (0.3, 0.2) | (0.2, 0.2) | 10% | 8% |

**Logic**:
- Subscription payments: monthly auto-pay on day 1-5, only for paid tiers
- about_to_churn: 50% chance to skip 3rd month subscription payment
- Day pass and in-app: randomly distributed across the 84-day window
- Payment method fixed per user: credit_card 60%, paypal 25%, gift_card 15%
- Status determined by `failed_prob` + `refund_prob` per persona

---

## 4. Feature Engineering Plan (50+ Features)

Features are computed at **weekly granularity** for the 4-week observation window.

### 4.1 Session Patterns
- `weekly_session_count` — sessions per week
- `avg_session_duration_min` — average session length
- `total_playtime_min` — total play time per week
- `session_regularity` — std of inter-session time gaps
- `peak_hour_ratio` — % of sessions during peak hours (7pm-12am)
- `weekend_ratio` — % of sessions on weekends

### 4.2 Engagement Decay
- `session_count_wow_change` — week-over-week session count change rate
- `playtime_wow_change` — week-over-week playtime change rate
- `rolling_avg_playtime_slope` — slope of 4-week rolling average playtime
- `longest_inactive_days` — max consecutive days with no session

### 4.3 Streaming Quality
- `avg_latency` — mean latency across sessions
- `avg_fps` — mean frame rate
- `frame_drop_rate` — total_frame_drops / total_frames
- `disconnect_rate` — disconnects per session
- `avg_bitrate` — mean bitrate
- `avg_jitter` — mean jitter
- `packet_loss_avg` — mean packet loss rate
- `crash_exit_ratio` — % of sessions ending in crash/disconnect/timeout
- `quality_downgrade_count` — times resolution dropped below user's usual

### 4.4 Game Diversity
- `unique_games_played` — distinct games per week
- `genre_entropy` — Shannon entropy of genre distribution
- `new_game_trial_rate` — % of sessions on games not played before
- `top_game_concentration` — % of playtime on most-played game

### 4.5 Playtime Volatility
- `daily_playtime_std` — std of daily playtime
- `daily_playtime_cv` — coefficient of variation
- `session_duration_std` — std of individual session lengths

### 4.6 Subscription & Payment
- `current_tier_numeric` — free=0, priority=1, ultimate=2
- `days_since_signup` — account age
- `tier_changes_count` — total subscription changes
- `has_downgraded` — binary: ever downgraded
- `total_spend_last_4w` — total payment amount in observation window
- `payment_count_last_4w` — number of payments
- `failed_payment_count` — number of failed transactions
- `refund_count` — number of refunds
- `days_since_last_payment` — recency of last successful payment
- `payment_frequency_change` — payment count change vs prior period

### 4.7 Aggregated Features (for XGBoost)
All above features averaged/aggregated across the 4-week window into a single row per user.

### 4.8 Sequential Features (for LSTM)
The same features kept as a (user, 4, num_features) 3D tensor — one vector per week, 4 weeks.

---

## 5. Model Architecture

### 5.1 Model A — XGBoost
- **Input**: Aggregated feature vector per user
- **Purpose**: Interpretable baseline, SHAP analysis
- **Tuning**: Optuna with 5-fold stratified CV
- **Key hyperparams**: max_depth, learning_rate, n_estimators, scale_pos_weight, subsample, colsample_bytree
- **Target AUC**: ≥ 0.85

### 5.2 Model B — LSTM
- **Input**: (batch, 4, num_features) sequential tensor
- **Architecture**: 2-layer LSTM (hidden_dim=64) → Dropout(0.3) → Dense(32) → Sigmoid
- **Training**: PyTorch, Adam optimizer, BCEWithLogitsLoss, early stopping on val AUC
- **Target AUC**: ≥ 0.87

### 5.3 Ensemble
- **Method**: Weighted average or stacking (LogisticRegression on both models' predicted probabilities)
- **Target AUC**: ≥ 0.89

---

## 6. Tech Stack

| Layer | Technology |
|---|---|
| Data Generation | Python, Faker, NumPy, Pandas |
| Storage Format | Parquet (local), S3 (AWS) |
| Feature Engineering | PySpark (local mode or EMR) |
| ML Training | XGBoost, PyTorch, scikit-learn, Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Model Serving | AWS SageMaker Endpoint or Lambda + API Gateway |
| Batch Inference | SageMaker Batch Transform or Step Functions |
| CI/CD | GitHub Actions |
| Infrastructure | Terraform, AWS (S3, ECR, SageMaker, CloudWatch) |
| Monitoring | Evidently AI (data drift), CloudWatch (infra) |
| Orchestration | SageMaker Pipelines |

---

## 7. Project Phases & Timeline

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

## 8. GitHub Workflow

### 8.1 Repository Structure

```
gfn-churn-prediction/
├── README.md
├── PROJECT_PLAN.md              # This document
├── .gitignore
├── pyproject.toml               # Project dependencies (or requirements.txt)
├── Dockerfile
├── Makefile                     # Common commands
├── data/
│   ├── raw/                     # Generated synthetic data (parquet)
│   ├── processed/               # Feature-engineered data
│   └── splits/                  # Train/val/test sets
├── src/
│   ├── data_generation/         # Phase 1: synthetic data scripts
│   ├── feature_engineering/     # Phase 2: PySpark feature pipelines
│   ├── dataset/                 # Phase 3: labeling, splitting, formatting
│   ├── models/                  # Phase 4: XGBoost, LSTM, ensemble
│   ├── evaluation/              # Metrics, SHAP analysis
│   └── serving/                 # Phase 6: inference scripts
├── notebooks/                   # EDA and experimentation notebooks
├── configs/                     # Hyperparameters, feature lists, etc.
├── tests/                       # Unit and integration tests
├── infrastructure/
│   ├── terraform/               # AWS infrastructure as code
│   └── docker/                  # Dockerfiles for different stages
├── .github/
│   └── workflows/               # GitHub Actions CI/CD pipelines
└── mlflow/                      # MLflow tracking configs
```

### 8.2 Branching Strategy (Git Flow Simplified)

```
main          ← production-ready code only, protected branch
  └── develop ← integration branch, all features merge here
        ├── feature/phase1-data-generation
        ├── feature/phase2-feature-engineering
        ├── feature/phase3-dataset-construction
        ├── feature/phase4-xgboost
        ├── feature/phase4-lstm
        ├── feature/phase4-ensemble
        ├── feature/phase5-mlflow
        ├── feature/phase6-sagemaker-deploy
        ├── feature/phase7-cicd
        └── feature/phase7-monitoring
```

**Rules**:
- `main`: Only merged from `develop` via PR, requires passing CI.
- `develop`: Integration branch. Feature branches merge here via PR.
- `feature/*`: One branch per logical unit of work. Named as `feature/<phase>-<description>`.
- Never commit directly to `main` or `develop`.

### 8.3 Commit Convention (Conventional Commits)

Format: `<type>(<scope>): <short description>`

| Type | When to Use | Example |
|---|---|---|
| `feat` | New feature or capability | `feat(data-gen): add user persona generation logic` |
| `fix` | Bug fix | `fix(features): correct week-over-week calculation` |
| `refactor` | Code restructure, no behavior change | `refactor(models): extract training loop into function` |
| `docs` | Documentation only | `docs: update PROJECT_PLAN with payment schema` |
| `test` | Add or update tests | `test(data-gen): add unit tests for session generator` |
| `chore` | Build, CI, config changes | `chore: add PySpark to requirements` |
| `data` | Data schema or generation changes | `data: add payments table schema` |

### 8.4 PR Workflow
1. Create feature branch from `develop`
2. Work in small, focused commits
3. Push and open PR to `develop`
4. PR description should reference the Phase and what it accomplishes
5. Squash-merge to keep `develop` history clean
6. After phase completion, merge `develop` → `main` with a version tag (e.g., `v0.1.0-phase1`)

### 8.5 Version Tags
- `v0.1.0` — Phase 1 complete (data generation)
- `v0.2.0` — Phase 2 complete (feature engineering)
- `v0.3.0` — Phase 3 complete (dataset construction)
- `v0.4.0` — Phase 4 complete (models trained)
- `v0.5.0` — Phase 5 complete (experiment tracking)
- `v0.6.0` — Phase 6 complete (deployed)
- `v1.0.0` — Phase 7 complete (full MLOps)

---

## 9. Current Progress

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

**Current branch**: `develop` (Phase 1 merged)
**Next Step**: Phase 2 — Feature engineering with PySpark.

---

## 10. Key Decisions Log

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

---

*Last updated: 2025-03-12*