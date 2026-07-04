# Implementation Plan — GFN Churn Prediction

> Step-by-step execution plan derived from `DEVELOPMENT_PLAN.md` (scan of 2026-07-05).
> Written to be executed milestone-by-milestone by an AI coding agent. Follow milestones
> in order — later ones depend on earlier ones. Stop at every **CHECKPOINT** and wait for
> the user's decision before continuing.
>
> Decisions already made by the user (do not re-litigate):
> 1. Scope: full roadmap through AWS deployment and MLOps (Phases 4–7).
> 2. Data fork: **harden the synthetic data** (one regeneration cycle, combined with the churn-gap fix).
> 3. Experiment tracking: **MLflow, pulled forward** before LSTM work (local file store).
> 4. Pre-registered success criterion for LSTM/ensemble: **recall at fixed precision** (see M7).

---

## Global rules (apply to every milestone)

**Read before writing any code:** `CLAUDE.md`, `configs/temporal_conventions.md`,
`configs/data_design.md`, `configs/feature_spec.md`. These are canonical; if code and
docs disagree, flag it — do not silently pick one.

**Hard constraints:**
- `persona` column must NEVER appear in model features (data leakage). Any new feature
  table must be checked against this.
- All week arithmetic branches on 1-indexed `calendar_week`, never on a raw 0-indexed
  loop counter (`temporal_conventions.md` §1 and §5 — this exact bug shipped once already).
- Never commit files under `data/`, `models/`, or `mlruns/` (gitignored; keep it that way,
  except where a milestone explicitly says otherwise).
- Never commit AWS credentials, account IDs in code, or `.tfstate` files.
- Package management with `uv` (`uv add <pkg>`, `uv sync`). Do not use pip directly.
- PySpark code (`src/feature_engineering/`) runs only inside the Docker dev container
  (`.devcontainer/` + `infrastructure/docker/docker-compose.yml`). Everything else runs
  in the local uv venv.

**Workflow per milestone:**
- Branch from `develop`: `feature/<milestone-slug>` (or `chore/` for M2/M3-style work).
- Conventional commits: `feat|fix|docs|chore|test|data(scope): message`.
- Open a PR to `develop` at the end of each milestone; update the `PROJECT_PLAN.md`
  progress checklist and decision log (dated) in the same PR.
- After any change to data generation or dataset construction, rerun the affected
  pipeline stages and the test suite before committing.

**Environment sanity check (run once before M1):**
```bash
uv sync
uv run python -c "import pandas, numpy, xgboost, sklearn, torch; print('ok')"
ls data/raw/          # 5 parquet files must exist
ls data/processed/    # xgboost_/lstm_ train/val/test files must exist
```
If `data/` is empty, regenerate: `uv run python -m src.data_generation.run_all`
(then Phase 2 must be rerun in the Spark container — ask the user before doing that).

---

## M1 — Investigate the churn-rate gap (READ-ONLY)

**Why:** actual churn is 12.2% (`models/xgboost_v1_metrics.json`: 917/7,499 val positives);
`configs/temporal_conventions.md` §3 predicts 15–18%, with a theoretical floor of ~15%
(about_to_churn = 15% of users, all of them supposedly reaching zero sessions by week 9).
12.2% is below the floor → something is off in generation, labeling, or the doc.
Everything downstream (hardening parameters in M4) depends on this answer.

**Branch:** none needed — this milestone modifies no source files.

**Steps:**
1. Write a throwaway analysis script at `scratch/churn_gap_analysis.py` (create `scratch/`,
   add it to `.gitignore`). Do not put this in `src/`.
2. In the script:
   a. Load `data/raw/users.parquet` (keep `user_id`, `persona`, `signup_date`) and
      `data/raw/session_logs.parquet` (`user_id`, `start_time`).
   b. Recompute labels with `src.dataset.labels.compute_churn_labels` (import it — do not
      reimplement).
   c. Report overall churn rate and churn rate **per persona**.
   d. For about_to_churn users with `churn == 0`: how many are there, and how many
      sessions do they have inside the prediction window (2024-02-26 ≤ date < 2024-03-11)?
      Sample 5 such users and print their full weekly session counts (calendar weeks 1–12).
   e. Check `signup_date` distribution: are any users signed up after the observation
      window starts? Do late signups correlate with the anomaly?
   f. Cross-check the decay implementation: read `src/data_generation/generate_session_logs.py`
      (especially the decay logic around lines 154–166) and verify the formulas match
      `temporal_conventions.md` §3. Also check whether noise injection (AFK sessions,
      hot event weeks) can create sessions for decayed users in weeks 9–10 — that would
      explain about_to_churn users escaping the churn label.
3. Write findings into `scratch/churn_gap_findings.md`: root cause, evidence, and a
   one-paragraph recommendation (fix generator / fix doc expectation / both).

**Verify:** script runs end-to-end with `uv run python scratch/churn_gap_analysis.py`.

**CHECKPOINT 1:** present findings to the user. Proposed fix (if any) gets folded into M4
(data hardening) so there is only ONE regeneration cycle. Do not change any generator
code in this milestone.

> **STATUS: COMPLETE (2026-07-05). CP1 decision: fix folded into M4.** Root cause found and
> independently verified: `src/data_generation/generate_session_logs.py:185`
> (`weekly_count = max(0, int(rng.normal(weekly_count, 1)))`) jitters the weekly session
> count with no guard, so a fully-decayed user (`weekly_count == 0`) has ≈15.9%/week odds
> (`P(int(N(0,1)) ≥ 1)`) of a ghost session. Result: 1,839 of 7,429 about_to_churn users
> (24.75%) have ≥1 session in weeks 9–10 and escape the churn label (1,533 on exactly one
> session). Per-persona churn: about_to_churn 75.2%, casual 3.3%, regular 0.1%, hardcore 0%
> → overall 12.23%. The decay formula itself is correct. Signup dates ruled out (all 2023).
> Note for M4: these escapees are accidental label noise and currently the hardest cases —
> fixing the bug alone pushes churn to ≈15.9% and makes the data EASIER. The M4 guard must
> land together with the deliberate partial-churner mechanic. Full analysis in
> `scratch/churn_gap_findings.md` (local, gitignored).

---

## M2 — Test suite + CI (safety net before touching the generator)

**Why:** zero tests exist; both historical bugs were silent failures. Tests must exist
BEFORE the M4 regeneration so the regen is verifiable.

**Branch:** `chore/tests-and-ci`

**Steps:**
1. `uv add --dev pytest ruff`
2. Create `tests/__init__.py` (empty) and the following test files. Use small hand-built
   DataFrames (5–10 rows) — never load real data files in unit tests.

   `tests/test_labels.py` — for `src.dataset.labels.compute_churn_labels`:
   - user with sessions only before the prediction window → churn = 1
   - user with ≥1 session inside the window → churn = 0
   - boundary: session exactly at `PRED_START` (2024-02-26 00:00) → churn = 0;
     session exactly at `PRED_END` (2024-03-11 00:00) → churn = 1
   - user with zero sessions anywhere → churn = 1

   `tests/test_split.py` — for `src.dataset.split.stratified_split`:
   - splits are disjoint and their union equals the input `user_ids`
   - sizes ≈ 70/15/15 (allow ±1 for int truncation)
   - churn rate per split within 1pp of the population rate
   - same seed → identical splits; different seed → different splits

   `tests/test_aggregation.py` — for `src.dataset.aggregation`:
   - `aggregate_xgboost`: build 2 users × 4 weeks of synthetic weekly features;
     check mean columns average correctly, `_last` columns come from `week_num == 4`,
     `playtime_slope` matches a hand-computed slope, output index is `user_id`,
     and the `churn` column joins correctly.
   - `reshape_lstm`: X shape is `(n_users, 4, n_features)`, y aligns with `user_ids`
     order, NaN fills match the documented strategy (wow_change→0, vs_baseline→1,
     days_since_last_payment→-1).
   - `reshape_lstm` with a user having only 3 weekly rows → must raise (see step 3).

   `tests/test_evaluate.py` — for `src.models.evaluate`:
   - `compute_metrics` on a tiny known vector → hand-checked confusion matrix and F1
   - `find_best_threshold` returns a threshold in [0.05, 0.95] and the max-F1 point on
     a constructed example where the best threshold is known
   - metrics dict round-trips through `json.dumps`

3. Fix the silent-misalignment hazard: in `src/dataset/aggregation.py::reshape_lstm`,
   before the `.reshape(...)` call add:
   ```python
   if len(lstm_df) != n_users * 4:
       raise ValueError(
           f"Expected {n_users * 4} rows (4 weeks x {n_users} users), got {len(lstm_df)}"
       )
   ```
4. Add a data-invariant test at `tests/test_data_invariants.py`, marked
   `@pytest.mark.data` and skipped automatically when `data/processed/` files are absent:
   - churn rate across splits within the expected band (use 10–20% until M4 finalizes
     the band, then tighten)
   - train/val/test user_ids disjoint (read the three parquet index sets)
   - no `persona` column in any `xgboost_*.parquet`
   Register the marker in `pyproject.toml` so `pytest` doesn't warn.
5. Add ruff config to `pyproject.toml` (`[tool.ruff]`, line-length 100, target py312,
   default rule set + `I` for import sorting). Run `uv run ruff check src/ tests/ --fix`
   and commit mechanical fixes separately (`chore(lint): ...`).
6. Create `.github/workflows/ci.yml`: on push/PR to `develop` and `feature/**` —
   checkout, `astral-sh/setup-uv`, `uv sync`, `uv run ruff check src/ tests/`,
   `uv run pytest -m "not data"` (unit tests only; data files don't exist in CI).
7. README sync (small, do it here): tick Phase 2/3 roadmap checkboxes, split the tech
   stack table into "implemented" vs "planned", replace the placeholder description in
   `pyproject.toml`, note that feature engineering runs in the dev container.

**Verify:** `uv run pytest` green locally (data tests included); push branch and confirm
the GitHub Actions run is green before opening the PR.

---

## M3 — MLflow (Phase 5, pulled forward)

**Why:** the M4 regen produces a new XGBoost baseline; both old and new runs should be
tracked so the hardening effect is documented. `mlruns/` is already gitignored.

**Branch:** `feature/phase5-mlflow`

**Steps:**
1. `uv add mlflow`
2. Create `src/models/tracking.py` with one helper:
   `log_run(model_name, params: dict, metrics: dict, artifacts: list[Path], tags: dict)`
   — sets tracking URI to `<repo_root>/mlruns` (resolve repo root the same way as
   `src/models/config.py`), experiment name `"gfn-churn"`, flattens the nested metrics
   dict from `compute_metrics` (e.g. `val_roc_auc`, `test_pr_auc`).
3. Wire it into the XGBoost flow: after `save_artifacts` in the notebook/pipeline path,
   log params (`XGB_PARAMS` + `scale_pos_weight` + `best_iteration`), val/test metrics,
   and the three artifact files. Tag runs with `data_version` (see M4 step 6).
4. Backfill: log the existing `models/xgboost_v1_metrics.json` as a run tagged
   `data_version=v1-easy` so the pre-hardening baseline is preserved.
5. Smoke-test: `uv run mlflow ui` starts and shows both runs. Document the command in
   README.

**Verify:** two runs visible in MLflow UI; `uv run pytest` still green.

---

## M4 — Harden the data + fix the churn gap (ONE regeneration cycle)

**Why:** XGBoost at 0.969 test AUC leaves no headroom for the LSTM/ensemble story.
Goal: make the problem realistically hard, land churn rate in a sane band, and fix
whatever M1 found — all in a single regen.

**Branch:** `data/harden-generation-v2`

**CHECKPOINT 2 (before writing code):** propose the exact parameter changes to the user,
informed by M1 findings. Starting proposal (adjust with M1 evidence):
- **Randomize decay onset** per about_to_churn user: onset calendar week sampled from
  {6, 7, 8} (currently fixed at 6). Late-onset users show only 1–2 decayed weeks inside
  the observation window → genuinely hard cases.
- **Partial churners:** ~25% of about_to_churn users decay to a floor of 0.15–0.30
  (instead of 0.0) and keep sparse sessions in weeks 9–10 → they do NOT churn by the
  label definition. Near-churn behavior with churn=0 labels = label noise the model
  must tolerate.
- **Persona overlap:** widen casual session-frequency variance so more casuals have
  naturally quiet fortnights (raising false-positive pressure). Keep the free-tier
  60-min cap and all causal mechanics unchanged.
- **Quality-decay softening:** reduce the exit-type shift for decaying users (e.g.
  degraded weights closer to [70% normal, 12% crash, 13% disconnect, 5% timeout]) so
  quality features are a weaker giveaway.
- **M1 bug fix (required, decided at CP1):** guard the count-noise at
  `generate_session_logs.py:185` so it cannot resurrect sessions for an exactly-zero
  decay week — e.g. apply the `rng.normal(weekly_count, 1)` jitter only when
  `weekly_count > 0`, or scale the jitter's σ by `decay_factor`. Design this together
  with the partial-churner floor above: partial churners' weeks 9–10 sessions must come
  from their explicit nonzero floor, never from unguarded noise. Update
  `temporal_conventions.md` §3 (expected churn ≈15.9% after fix, before the partial-churner
  mechanic shifts it) and `data_design.md` in the same commit.

**Acceptance bands (pre-registered — write them into the PR description):**
- Overall churn rate: **13–18%** (final band set at Checkpoint 2 once M1 explains the floor)
- Retrained XGBoost test ROC-AUC: **0.85–0.93**. If AUC still > 0.95 after one knob
  iteration, stop and report at a checkpoint rather than iterating endlessly.

**Steps (after checkpoint approval):**
1. Implement the approved changes in `src/data_generation/generate_session_logs.py`
   (and `generate_users.py` if persona variance changes). Branch on `calendar_week`
   only. Keep all changes behind clearly named module-level constants at the top of the
   file, with a comment block describing v2 hardening.
2. Update the canonical docs in the same commit: `configs/temporal_conventions.md` §3
   (new decay table incl. onset randomization + partial-churn floor),
   `configs/data_design.md`, and a dated row in the `PROJECT_PLAN.md` decision log.
3. Regenerate: `uv run python -m src.data_generation.run_all` (seed 42, small scale).
4. Rerun Phase 2 inside the Spark dev container:
   `python -m src.feature_engineering.pipeline` (container path — see module docstring).
   If the container cannot be started in this environment, STOP and ask the user to run
   this step.
5. Rerun Phase 3: `uv run python -m src.dataset.pipeline` from the repo root. Check the
   printed churn rate against the acceptance band. Run `uv run pytest` (data-invariant
   tests must pass; tighten the churn band in `tests/test_data_invariants.py` now).
6. Retrain XGBoost via the phase4 notebook flow (or a script mirroring it): name the
   model `xgboost_v2`, log to MLflow with `data_version=v2-hardened`.
7. Compare v1 vs v2 in a short markdown table (AUC, PR-AUC, recall@threshold) in the PR
   description.

**CHECKPOINT 3:** report v2 metrics vs acceptance bands. User approves before Phase 4
modeling continues on the new data.

---

## M5 — SHAP feature importance (on v2 baseline)

**Branch:** `feature/phase4-shap`

**Steps:**
1. Create `src/evaluation/shap_analysis.py` (this answers the open module-home question:
   SHAP lives in `src/evaluation/`; `src/models/evaluate.py` keeps threshold/metric
   utilities). Functions:
   - `compute_shap_values(clf, X: pd.DataFrame) -> shap.Explanation` (TreeExplainer)
   - `save_summary_plots(explanation, out_dir: Path)` — beeswarm + bar plot PNGs
   - `top_features(explanation, k=15) -> pd.DataFrame` (mean |SHAP|)
2. Add a "3. SHAP" section to `notebooks/phase4_xgboost.ipynb` that calls these helpers
   on the v2 model, renders the plots, and writes 5–8 bullet insights (which features
   drive churn predictions; do they match `data_design.md` expectations — decay/volume
   features should dominate, timing/diversity weak).
3. Log the plot PNGs as MLflow artifacts on the v2 run.
4. Tick the SHAP checkbox in `PROJECT_PLAN.md`.

**Verify:** notebook runs top-to-bottom; plots exist; insights section references at
least 3 concrete features with direction of effect.

---

## M6 — LSTM model

**Branch:** `feature/phase4-lstm`

**Steps:**
1. Create `src/models/lstm_model.py`:
   - `ChurnLSTM(nn.Module)`: LSTM(input_size=n_features, hidden_size=64, num_layers=2,
     dropout=0.3, batch_first=True) → take last hidden state → Linear(64→32) → ReLU →
     Dropout → Linear(32→1). Parameters come from `LSTM_PARAMS` in `src/models/config.py`
     — read them, don't hardcode.
   - `train_lstm(data: dict, params, train_params) -> tuple[model, dict]`: uses
     `load_lstm_all()` output; `BCEWithLogitsLoss(pos_weight=class_imbalance_ratio(y_train))`;
     Adam(lr=1e-3, weight_decay=1e-5); `ReduceLROnPlateau(factor=0.5, patience=3)` on val
     PR-AUC; early stopping patience 5 epochs on val PR-AUC; grad-clip norm 1.0; max 30
     epochs; seed everything with `SEED` (torch, numpy, random; `torch.use_deterministic_algorithms`
     where feasible). All constants from `LSTM_TRAIN_PARAMS`.
   - `save_artifacts(...)` mirroring the XGBoost version: model `state_dict` (.pt),
     fitted `StandardScaler` (joblib — the scaler MUST be persisted, see
     `data_loaders.py` docstring), predictions parquet, metrics JSON. Same
     val-selected-threshold-frozen-on-test policy — reuse `find_best_threshold` and
     `compute_metrics`.
2. Add `recall_at_precision(y_true, y_proba, precision_floor: float) -> tuple[recall, threshold]`
   to `src/models/evaluate.py`: highest recall achievable at precision ≥ floor (sweep
   thresholds; if the floor is unreachable, return (0.0, nan) and warn). Unit-test it in
   `tests/test_evaluate.py` (constructed case with known answer).
3. Create `notebooks/phase4_lstm.ipynb`: load via `load_lstm_all()`, train, plot
   train/val loss + val PR-AUC per epoch, evaluate on test at the val-frozen threshold,
   report `recall_at_precision(..., 0.75)` alongside the same number for XGBoost v2.
4. Log the run to MLflow (`model=lstm`, `data_version=v2-hardened`).
5. CPU is fine at 35K×4×38; if an epoch takes >5 min, reduce to a smoke run and report.

**Verify:** training converges (val PR-AUC improves over epoch 1 then plateaus);
artifacts saved under `models/lstm_v1_*`; `uv run pytest` green.

---

## M7 — Ensemble + pre-registered evaluation

**Pre-registered success criterion (user's decision — do not change it after seeing
results):** at **precision ≥ 0.75** on the test set (threshold chosen on val), the
ensemble's **recall must exceed XGBoost v2's recall by ≥ 2 percentage points**.
Report the outcome either way — a negative result gets written up honestly.

**Branch:** `feature/phase4-ensemble`

**Steps:**
1. `src/models/ensemble.py`: weighted average `p = w * p_xgb + (1 - w) * p_lstm`.
   Grid-search `w ∈ {0.0, 0.1, ..., 1.0}` maximizing val `recall_at_precision(0.75)`.
   Freeze `w` and the threshold from val; evaluate once on test.
2. Notebook `notebooks/phase4_ensemble.ipynb`: load both models' saved prediction
   parquets (do NOT retrain), align on user_id, run the grid, produce a comparison
   table (XGBoost v2 / LSTM / ensemble × ROC-AUC, PR-AUC, recall@P≥0.75) and a
   precision-recall curve overlay.
3. Log ensemble run to MLflow. Save `models/ensemble_v1_metrics.json` with the chosen
   `w` and threshold.
4. Write the Phase 4 results summary into `PROJECT_PLAN.md` (decision log + progress)
   and a short `docs/phase4_results.md` with the comparison table and the SHAP insights.

**CHECKPOINT 4:** present the criterion outcome. User decides whether Phase 4 closes
(merge + tag `v0.4.0-phase4`) or one more iteration is warranted.

---

## M8 — Path/CLI cleanup + runnable validation

**Branch:** `chore/paths-and-validation`

**Steps:**
1. Create `src/paths.py`: `REPO_ROOT` (resolved from `__file__`), `RAW_DIR`,
   `PROCESSED_DIR`, `MODELS_DIR`, `MLRUNS_DIR`. Refactor `src/dataset/pipeline.py`,
   `src/models/config.py`, and `src/data_generation/run_all.py` to import from it.
   `src/feature_engineering/pipeline.py` keeps its container default but gains a
   `--input-dir/--output-path` CLI override.
2. Add `argparse` mains to the three pipeline entry points (generation: `--scale`,
   `--seed`; dataset: `--raw-dir`, `--processed-dir`; feature engineering as above).
   Env-var behavior of `run_all.py` stays as a fallback so README commands keep working.
3. Port the phase2/phase3 validation notebooks' checks (the "14 checks" from
   `PROJECT_PLAN.md`) into `src/dataset/validate.py`, runnable as
   `uv run python -m src.dataset.validate` — prints PASS/FAIL per check, exits nonzero
   on failure. Wire it into `tests/test_data_invariants.py` (data-marked).
4. Delete orphan intermediates `data/processed/lstm_features.npz` and
   `data/processed/xgboost_features.parquet` AFTER grepping all notebooks for
   references; if any notebook reads them, update it to the split files instead.

**Verify:** all three pipelines run from the repo root without cwd tricks;
`validate.py` passes on current data; pytest green.

---

## M9 — Full-scale data run (200K users)

**Why before AWS:** validates the Spark pipeline at realistic scale while everything is
still local and free; produces the timing numbers for the portfolio writeup.

**Branch:** none (no source changes expected) — document results only.

**Steps:**
1. `SCALE=full uv run python -m src.data_generation.run_all` — but first check disk
   (~1–2 GB needed) and warn the user this overwrites `data/raw/`. **Ask before
   overwriting** if v2-small data hasn't been backed up or is still needed; simplest is
   regenerating small data afterward (seeded, deterministic).
2. Run Phase 2 in the Spark container on the full data; record wall-clock, partition
   behavior, any OOM/tuning needed (this is the interesting part — document fixes).
3. Run Phase 3 + `validate.py`; confirm churn band holds at scale.
4. Optionally retrain XGBoost v2 on full data; log to MLflow as `data_version=v2-full`.
5. Write findings into `docs/full_scale_run.md` (timings, tuning, metric deltas).
6. Regenerate small-scale data so the default dev state is restored.

---

## M10 — AWS deployment (Phase 6)

**CHECKPOINT 5 (before ANY AWS work):** confirm with the user — AWS account/profile,
region, budget ceiling, and whether SageMaker **batch transform** (cheaper, fits the
weekly-batch churn use case) or a **real-time endpoint** (better demo) is wanted.
Nothing in this milestone runs before this checkpoint.

**Branch:** `feature/phase6-aws`

**Steps (adjust to checkpoint answers):**
1. Terraform under `infrastructure/terraform/`: S3 bucket (data + model artifacts),
   ECR repo, IAM role for SageMaker, and the endpoint/batch-transform resources.
   Remote state or local state per user preference (ask at Checkpoint 5). All resources
   tagged `project=gfn-churn`. Include `terraform destroy` instructions in the README
   section — this is a portfolio project; nothing should run 24/7.
2. `src/serving/predict.py`: load `xgboost_v2.json` + expected-column list, accept a
   parquet/CSV of feature rows, return probabilities + churn flag at the frozen
   threshold. Unit-test the pre/post-processing locally (no AWS in tests).
3. Containerize serving (extend `infrastructure/docker/` with a serving Dockerfile
   compatible with SageMaker's inference contract), push to ECR.
4. Upload model artifacts to S3; deploy; run one smoke inference against the deployed
   model; capture the request/response in `docs/phase6_deployment.md`.
5. GitHub Actions job (manual `workflow_dispatch`) that builds and pushes the serving
   image. AWS auth via GitHub OIDC — never long-lived keys in secrets if avoidable.
6. Document teardown and verify `terraform destroy` leaves the account clean.

---

## M11 — MLOps (Phase 7)

**Branch:** `feature/phase7-mlops`

**Steps:**
1. Extend CI: a `training-smoke` job (manual/nightly) that generates a tiny dataset
   (add a `SCALE=tiny` option, ~2K users, to `generate_users.py`), runs dataset
   construction + a short XGBoost fit, and asserts metrics above a sanity floor.
2. Drift monitoring with Evidently (`uv add evidently`): compare feature distributions
   between the observation window and the **weeks 11–12 buffer** (this is what the
   buffer was reserved for — `temporal_conventions.md` §2). Script:
   `src/monitoring/drift_report.py` → HTML report artifact.
3. Retraining simulation: shift the window forward two weeks (obs 7–10, pred 11–12),
   rebuild the dataset with the M8 CLI flags, retrain, compare old vs new model in
   MLflow. Document the design in `docs/phase7_retraining.md`. (Note: window boundaries
   are currently constants in `src/feature_engineering/config.py` and
   `src/dataset/config.py` — parameterizing them is part of this step; update
   `temporal_conventions.md` accordingly.)
4. Optional (ask user): CloudWatch alarm on endpoint latency/error rate if a real-time
   endpoint was chosen in M10.
5. Final README overhaul: architecture diagram reflects reality, results table,
   MLflow/CI/monitoring screenshots, honest "synthetic data limitations" section.

---

## Milestone order and checkpoints — summary

| # | Milestone | Blocking checkpoint |
|---|---|---|
| M1 | ✅ Done 2026-07-05 — root cause: unguarded count-noise at `generate_session_logs.py:185`; fix folded into M4 | CP1 passed |
| M2 | Tests + CI + README sync | — |
| M3 | MLflow early | — |
| M4 | Harden data + regen + retrain XGBoost v2 | CP2: params before code; CP3: metrics after |
| M5 | SHAP on v2 | — |
| M6 | LSTM | — |
| M7 | Ensemble, pre-registered recall@P≥0.75 test | CP4: outcome → user |
| M8 | Paths/CLI/validate.py cleanup | — |
| M9 | Full-scale 200K run | ask before overwriting data/raw |
| M10 | AWS deployment | CP5: account/budget/endpoint-type before any AWS call |
| M11 | MLOps: CI training, Evidently drift, retraining sim | — |
