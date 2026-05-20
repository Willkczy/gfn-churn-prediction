# Data Generation Design

> How the synthetic data was generated and why. Read this before feature engineering or modeling — understanding the embedded patterns is essential for interpreting model behavior.

## User Personas (4 types)

| Persona | % of Users | Behavior | Churn Risk |
|---|---|---|---|
| **Hardcore** | 20% | High frequency (14/wk), long sessions (150 min), ultimate/priority tier | Low |
| **Regular** | 35% | Moderate frequency (6/wk), 90 min sessions, priority tier | Moderate |
| **Casual** | 30% | Low frequency (2/wk), short sessions (40 min), mostly free tier | Higher |
| **About-to-churn** | 15% | Starts like Regular, then decays: declining frequency, shorter sessions, worse quality | Very High |

**Subscription tier is persona-dependent** (not random):

| Persona | free | priority | ultimate |
|---|---|---|---|
| hardcore | 5% | 35% | 60% |
| regular | 20% | 55% | 25% |
| casual | 70% | 25% | 5% |
| about_to_churn | 50% | 35% | 15% |

> `persona` column exists in users table but must be **excluded from model features** — it's data leakage.

## Churn Decay Mechanism (about_to_churn users, weeks 6–12)

This is the core pattern the model should learn to detect. Week numbers below
are **1-indexed calendar weeks** (see [temporal_conventions.md](temporal_conventions.md)).

- `decay_factor` = max(0, 1.0 − 0.25 × (`calendar_week` − 5))
  → week 6=0.75, 7=0.50, 8=0.25, 9–12=0.00
- `duration_multiplier` = max(0.3, 0.7 − 0.15 × (`calendar_week` − 6))
- Exit type shifts from [85% normal, 5% crash, 7% disconnect, 3% timeout] to **[50% normal, 18% crash, 22% disconnect, 10% timeout]**
- Session count and duration progressively decline
- Game diversity narrows

**Implication for modeling**: The observation window (week 5–8) captures the **full transition** from normal to deep decay. Week 5 looks normal; weeks 6–8 show progressively stronger decline (factor 0.75 → 0.25). Weeks 9–10 (prediction window) hit zero, so virtually every about_to_churn user yields `churn = 1`. The model must learn the week 5→8 decline pattern in the obs window.

## Dynamic Causal Relationships

These create realistic sequential dependencies — important for LSTM:

### Poor quality → early exit
- Latency > 100ms → probability `min(0.6, (latency-100)/200)` to quit early (duration × 20-70%)
- Each disconnect has 30% chance to terminate session (duration × 30-80%)
- Exit type forced to "disconnect" when triggered

### Bad experience → skip next session
- Triggered when exit_type is crash/disconnect or latency > 120ms
- Next session has **35% chance to be skipped entirely**
- Creates realistic gaps in session sequences

**Implication for modeling**: Session-level features (latency, exit type) causally affect engagement features (session count, inactive days). LSTM should capture this sequential dependency.

## Streaming Quality Profiles (per tier)

| Tier | Latency (μ,σ) | FPS (μ,σ) | Bitrate (μ,σ) | Jitter (μ,σ) | Packet loss (μ,σ) |
|---|---|---|---|---|---|
| free | (50, 20) | (45, 10) | (10, 3) | (30, 15) | (0.05, 0.03) |
| priority | (30, 10) | (55, 8) | (25, 5) | (15, 8) | (0.02, 0.01) |
| ultimate | (18, 5) | (58, 5) | (40, 8) | (8, 4) | (0.008, 0.005) |

**Implication for modeling**: Quality features are correlated with tier. The model should learn that quality degradation *within* a tier is a churn signal, not just absolute quality levels.

## Session Behavior Profiles

| Persona | Sessions/week (μ,σ) | Duration min (μ,σ) | Peak hour ratio | Weekend ratio |
|---|---|---|---|---|
| hardcore | (14, 3) | (150, 40) | 0.5 | 0.4 |
| regular | (6, 2) | (90, 30) | 0.6 | 0.5 |
| casual | (2, 1) | (40, 15) | 0.7 | 0.6 |
| about_to_churn | (5, 2) | (80, 25) | 0.6 | 0.5 |

## Subscription & Payment Patterns

**Event probabilities by persona** (in 12-week window):

| Persona | upgrade | downgrade | cancel | renew |
|---|---|---|---|---|
| hardcore | 5% | 2% | 1% | 80% |
| regular | 15% | 10% | 5% | 60% |
| casual | 10% | 15% | 10% | 40% |
| about_to_churn | 3% | 35% | 40% | 15% |

**Payment failure/refund rates**: hardcore 2%/1%, regular 3%/2%, casual 5%/3%, about_to_churn **10%/8%**

**Temporal bias**: Downgrades biased toward 2nd half of window. Cancels biased toward last third. About_to_churn users have 50% chance to skip 3rd month subscription payment.

**Implication for modeling**: Payment failure rate and downgrade/cancel events are strong churn signals. Payment frequency decline over time is a key feature.

## Noise & Non-Stationarity

### Hot event weeks (2–3 random weeks)
- Activity multiplier: 1.3x–1.8x on session count (all users)
- Simulates new game launches — adds non-stationarity

### Noise injection
- **2% AFK sessions**: duration 3–12 hours, fps 1–10, exit_type = timeout
- **3% latency spikes**: latency 150–400ms, jitter 50–150ms, packet_loss 0.05–0.2
- **1% late night sessions**: moved to 0:00–5:00

### Other rules
- Free tier: session capped at 60 minutes
- Game selection weighted by popularity (S=10x, A=5x, B=2x, C=1x)
- Peak hours: 19:00–23:59; off-peak: 08:00–18:59
- Resolution distribution varies by tier (free mostly 720p/1080p, ultimate mostly 1440p/4K)

**Implication for modeling**: Noise means the model can't rely on clean thresholds. Hot weeks affect all users equally, so week-over-week changes during hot weeks are less informative — baseline comparison features help compensate.
