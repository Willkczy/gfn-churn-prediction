# Temporal Conventions

> **Single source of truth for week numbering, window boundaries, and the
> churn-decay timeline across this repo.** Any other doc/notebook/code that
> mentions a "week" should be consistent with what is written here.

---

## 1. Week numbering — always 1-indexed calendar weeks

The data spans **12 calendar weeks**: 2024-01-01 to 2024-03-25 (84 days).

| Calendar week | Date range (inclusive start, exclusive end) |
|---|---|
| 1  | 2024-01-01 → 2024-01-08 |
| 2  | 2024-01-08 → 2024-01-15 |
| 3  | 2024-01-15 → 2024-01-22 |
| 4  | 2024-01-22 → 2024-01-29 |
| 5  | 2024-01-29 → 2024-02-05 |
| 6  | 2024-02-05 → 2024-02-12 |
| 7  | 2024-02-12 → 2024-02-19 |
| 8  | 2024-02-19 → 2024-02-26 |
| 9  | 2024-02-26 → 2024-03-04 |
| 10 | 2024-03-04 → 2024-03-11 |
| 11 | 2024-03-11 → 2024-03-18 |
| 12 | 2024-03-18 → 2024-03-25 |

**Rule:** in docs, notebooks, validation reports, and external communication,
always say "week N" using this 1-indexed scheme.

**Internal exception:** Python `for week in range(OBS_WEEKS)` loops produce a
0-indexed counter. In the codebase we always convert immediately:

```python
for week in range(OBS_WEEKS):
    calendar_week = week + 1   # ALWAYS branch on calendar_week
```

Never write `if week >= N` against the raw 0-indexed counter — this is the
source of the historical off-by-one decay bug (see §5).

---

## 2. Window roles

| Window | Calendar weeks | Purpose |
|---|---|---|
| Baseline   | 1–4  | Historical reference for trend features (`*_vs_baseline`, `*_wow_change` from week 1 perspective). Sessions here look "normal" for every persona. |
| Observation | 5–8 | Features are computed here. About-to-churn users show progressive decay (factor 1.00 → 0.25). |
| Prediction | 9–10 | Churn label window. `churn = 1` iff a user has zero sessions in weeks 9–10. About-to-churn users hit decay 0.0 by week 9. |
| Buffer     | 11–12 | Reserved for retraining simulation in Phase 7. Not used in training or labeling. |

### Phase 2 internal `week_num`

The feature-engineering output (`weekly_features.parquet`) uses a **window-local**
`week_num ∈ {1, 2, 3, 4}` that maps to the **observation window only**:

| `week_num` | Calendar week |
|---|---|
| 1 | 5 |
| 2 | 6 |
| 3 | 7 |
| 4 | 8 |

This is the index baked into `wow_change` (`week_num=1` ⇒ null, no prior week in obs),
`new_game_trial_rate` (same rule), and the slope/aggregation logic in Phase 3.

---

## 3. Churn decay timeline (about_to_churn persona)

`generate_session_logs.py` applies a per-week decay factor to about_to_churn
users. The implementation branches on **`calendar_week`** (see §1).

| Calendar week | `decay_factor` | `duration_multiplier` | Effect |
|---|---|---|---|
| 1–5  | 1.00 | 1.00 | Normal — indistinguishable from `regular` persona |
| 6    | 0.75 | 0.70 | Mild drop (mid-obs window) |
| 7    | 0.50 | 0.55 | Clear drop |
| 8    | 0.25 | 0.40 | Strong drop (end of obs window) |
| 9–12 | 0.00 | 0.30 | Zero sessions — yields churn=1 in pred window |

Formulas (from [generate_session_logs.py:160-166](../src/data_generation/generate_session_logs.py)):

```python
decay_factor = max(0.0, 1.0 - 0.25 * (calendar_week - 5))
duration_multiplier = max(0.3, 0.7 - 0.15 * (calendar_week - 6))
```

Exit-type distribution also shifts to "degraded" weights (more crash/disconnect/timeout) whenever decay is active.

**Expected churn rate**: about_to_churn = 15% of users. With decay-zero at
week 9, ≈ 100% of about_to_churn users yield `churn = 1`, plus a small slice
of casual users with naturally sparse activity. Total expected ≈ 15–18%.

---

## 4. Where these constants live in code

| Constant | File | Purpose |
|---|---|---|
| `OBS_START = 2024-01-01`, `OBS_WEEKS = 12` | [src/data_generation/generate_session_logs.py:7-8](../src/data_generation/generate_session_logs.py) | Full data window for generation |
| `DATA_START = 2024-01-01` | notebooks `phase2_feature_engineering.ipynb`, `phase2_feature_engineering_pandas.ipynb` | Week-1 anchor for baseline features |
| `BASELINE_END = OBS_START = 2024-01-29` | same notebooks | Split between baseline (wk 1–4) and obs (wk 5–8) |
| `OBS_END = 2024-02-26` | same notebooks | End of obs window (= start of pred window) |
| `PRED_START = 2024-02-26`, `PRED_END = 2024-03-11` | `notebooks/phase3_dataset_construction.ipynb` | Churn label window (weeks 9–10) |

If you change any of these, update **both** the constant and this doc.

---

## 5. History — the 0-indexed decay bug (2026-05)

**Symptom:** Phase 3 churn rate came out at 3.5% instead of the expected ~15%.

**Root cause:** Pre-fix code at [generate_session_logs.py:154-156](../src/data_generation/generate_session_logs.py) used the 0-indexed loop counter directly:

```python
for week in range(OBS_WEEKS):              # week is 0..11
    if persona == "about_to_churn" and week >= 6:
        decay_factor = 1.0 - 0.25 * (week - 5)
```

Mental model assumed `week` was the 1-indexed calendar week, so decay was
intended to hit zero at week 9 (start of pred window). In reality the 0-indexed
`week=8` (= calendar week 9) gave `decay_factor = 0.25`, not 0.0. Decay-zero
actually landed on calendar week 10. About_to_churn users still produced ~1.5
sessions/user during week 9, so most of them did not get labeled as churn.

**Fix:** Introduce an explicit `calendar_week = week + 1` and branch on that
instead. Decay now hits zero at calendar week 9 as originally intended.

**Lesson — applies to any future timeline change:**
1. Loop counters from `range()` are 0-indexed. Convert at the top of the loop.
2. Branch and compute timeline-dependent quantities on `calendar_week`.
3. If you change boundary weeks, update this doc + `data_design.md` + verify
   the per-persona weekly activity by re-running Phase 3 and checking that
   churn rate is in the expected band.
