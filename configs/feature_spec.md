# Feature Engineering Spec

> Features computed at **weekly granularity** over the 4-week observation window (week 5–8).
> Baseline period: week 1–4. Prediction window: week 9–10 (zero sessions → churn = 1).

## 1. Session Patterns
- [x] `weekly_session_count` — sessions per week
- [x] `avg_session_duration_min` — average session length
- [x] `total_playtime_min` — total play time per week
- [x] `session_regularity` — std of inter-session time gaps
- [x] `peak_hour_ratio` — % of sessions during peak hours (7pm-12am)
- [x] `weekend_ratio` — % of sessions on weekends

## 1.5 Population Normalization
- [x] `weekly_session_count_norm` — session count / population median for that week
- [x] `total_playtime_min_norm` — playtime / population median for that week

> Removes hot-week effects (1.3-1.8x global activity spikes). Medians computed from active users only (before zero-fill). Trend features (wow_change, vs_baseline) are computed on normalized values.

## 2. Engagement Decay
- [x] `session_count_wow_change` — week-over-week session count change rate (week 1 = null)
- [x] `playtime_wow_change` — week-over-week playtime change rate (week 1 = null)
- [x] `rolling_avg_playtime_slope` — slope of 4-week rolling average playtime (computed at aggregation)
- [x] `longest_inactive_days` — max consecutive days with no session (within obs window)
- [x] `session_count_vs_baseline` — obs window avg session count / baseline (week 1–4) avg
- [x] `playtime_vs_baseline` — obs window avg playtime / baseline avg playtime

**Note**: Week 1–4 is a historical baseline for trend features — detecting decline relative to the user's own prior activity, not just cross-user comparison.

## 3. Streaming Quality
- [x] `avg_latency` — mean latency across sessions
- [x] `avg_fps` — mean frame rate
- [x] `frame_drop_rate` — AVG(total_frame_drops) per session (simplified)
- [x] `disconnect_rate` — disconnects per session
- [x] `avg_bitrate` — mean bitrate
- [x] `avg_jitter` — mean jitter
- [x] `packet_loss_avg` — mean packet loss rate
- [x] `crash_exit_ratio` — % of sessions ending in crash/disconnect/timeout
- [ ] `quality_downgrade_count` — deferred (needs user-level "usual resolution" definition)

## 4. Game Diversity
- [x] `unique_games_played` — distinct games per week
- [x] `genre_entropy` — Shannon entropy of genre distribution
- [x] `new_game_trial_rate` — % of sessions on games not played before (week 1 = null)
- [x] `top_game_concentration` — % of playtime on most-played game

## 5. Playtime Volatility
- [x] `daily_playtime_std` — std of daily playtime (0-session days included)
- [x] `daily_playtime_cv` — coefficient of variation
- [x] `session_duration_std` — std of individual session lengths

## 6. Subscription & Payment
- [x] `current_tier_numeric` — free=0, priority=1, ultimate=2
- [x] `days_since_signup` — account age
- [x] `tier_changes_count` — total subscription changes
- [x] `has_downgraded` — binary: ever downgraded
- [x] `total_spend_last_4w` — total payment amount in observation window
- [x] `payment_count_last_4w` — number of payments
- [x] `failed_payment_count` — number of failed transactions
- [x] `refund_count` — number of refunds
- [x] `days_since_last_payment` — recency of last successful payment
- [x] `payment_frequency_change` — payment count change vs prior period

## 7. Output Formats

### XGBoost (aggregated)
Single row per user. Aggregation strategy by feature type:
- **Mean** (22 cols): behavioral features averaged across 4 weeks
- **Last-week** (9 cols): recency-sensitive features from week 4 only (wow_change, vs_baseline, longest_inactive, new_game_trial + extra last for session_count, playtime, crash_exit)
- **Static** (10 cols): user-level features (tier, spend, payment counts) — same all 4 weeks
- **Slope** (1 col): `playtime_slope` — linear slope of total_playtime across weeks 1-4

NaN handling: left as-is (XGBoost handles NaN natively for `days_since_last_payment`, `payment_frequency_change`, baseline ratios).

### LSTM (sequential)
`(users, 4, num_features)` 3D tensor — one vector per week, 4 weeks.

NaN fill strategy: `wow_change`/`new_game_trial_rate` → 0, `vs_baseline` → 1.0, `days_since_last_payment` → -1, remaining → 0.

### Output files
- `data/processed/xgboost_features.parquet` — flat, 50K rows
- `data/processed/lstm_features.npz` — arrays: `X`, `y`, `user_ids`, `feature_names`

## Design Decisions
- `wow_change` features: week 1 = null (no prior week to compare)
- `rolling_avg_playtime_slope`: computed at the aggregation layer, not per-week
- `frame_drop_rate`: simplified to AVG(drops) since total_frames isn't in the schema
- `quality_downgrade_count`: deferred — requires defining "usual resolution" per user
- `new_game_trial_rate`: week 1 = null (no history to compare against)
- 0-session days included in daily stats (important for churn signal)
- Payment features aggregated over obs window (4 weeks) only
