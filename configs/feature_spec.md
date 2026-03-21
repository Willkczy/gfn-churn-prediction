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

## 2. Engagement Decay
- [ ] `session_count_wow_change` — week-over-week session count change rate (week 1 = null)
- [ ] `playtime_wow_change` — week-over-week playtime change rate (week 1 = null)
- [ ] `rolling_avg_playtime_slope` — slope of 4-week rolling average playtime (computed at aggregation)
- [ ] `longest_inactive_days` — max consecutive days with no session (within obs window)
- [ ] `session_count_vs_baseline` — obs window avg session count / baseline (week 1–4) avg
- [ ] `playtime_vs_baseline` — obs window avg playtime / baseline avg playtime

**Note**: Week 1–4 is a historical baseline for trend features — detecting decline relative to the user's own prior activity, not just cross-user comparison.

## 3. Streaming Quality
- [ ] `avg_latency` — mean latency across sessions
- [ ] `avg_fps` — mean frame rate
- [ ] `frame_drop_rate` — AVG(total_frame_drops) per session (simplified)
- [ ] `disconnect_rate` — disconnects per session
- [ ] `avg_bitrate` — mean bitrate
- [ ] `avg_jitter` — mean jitter
- [ ] `packet_loss_avg` — mean packet loss rate
- [ ] `crash_exit_ratio` — % of sessions ending in crash/disconnect/timeout
- [ ] `quality_downgrade_count` — deferred (needs user-level "usual resolution" definition)

## 4. Game Diversity
- [ ] `unique_games_played` — distinct games per week
- [ ] `genre_entropy` — Shannon entropy of genre distribution
- [ ] `new_game_trial_rate` — % of sessions on games not played before (week 1 = null)
- [ ] `top_game_concentration` — % of playtime on most-played game

## 5. Playtime Volatility
- [ ] `daily_playtime_std` — std of daily playtime (0-session days included)
- [ ] `daily_playtime_cv` — coefficient of variation
- [ ] `session_duration_std` — std of individual session lengths

## 6. Subscription & Payment
- [ ] `current_tier_numeric` — free=0, priority=1, ultimate=2
- [ ] `days_since_signup` — account age
- [ ] `tier_changes_count` — total subscription changes
- [ ] `has_downgraded` — binary: ever downgraded
- [ ] `total_spend_last_4w` — total payment amount in observation window
- [ ] `payment_count_last_4w` — number of payments
- [ ] `failed_payment_count` — number of failed transactions
- [ ] `refund_count` — number of refunds
- [ ] `days_since_last_payment` — recency of last successful payment
- [ ] `payment_frequency_change` — payment count change vs prior period

## 7. Output Formats

### XGBoost (aggregated)
All above features averaged/aggregated across the 4-week window → single row per user.

### LSTM (sequential)
Same features kept as `(user, 4, num_features)` 3D tensor — one vector per week, 4 weeks.

## Design Decisions
- `wow_change` features: week 1 = null (no prior week to compare)
- `rolling_avg_playtime_slope`: computed at the aggregation layer, not per-week
- `frame_drop_rate`: simplified to AVG(drops) since total_frames isn't in the schema
- `quality_downgrade_count`: deferred — requires defining "usual resolution" per user
- `new_game_trial_rate`: week 1 = null (no history to compare against)
- 0-session days included in daily stats (important for churn signal)
- Payment features aggregated over obs window (4 weeks) only
