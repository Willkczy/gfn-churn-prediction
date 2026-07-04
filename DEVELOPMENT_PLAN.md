# Development Plan — GFN Churn Prediction

> Evidence-based roadmap built from a full repo scan on 2026-07-05 (branch `feature/phase4-model-training`, clean tree).
> Sources: `README.md`, `PROJECT_PLAN.md`, `configs/*.md`, all of `src/`, `notebooks/phase4_xgboost.ipynb`, `models/xgboost_v1_metrics.json`, `infrastructure/docker/`, git history (30 commits, 2 merged PRs).

## TLDR

This is a portfolio-grade, end-to-end churn-prediction pipeline on synthetic cloud-gaming data. Phases 1–3 (data generation, PySpark feature engineering, dataset construction) are complete and merged; Phase 4 is mid-flight with a trained XGBoost baseline already at **test AUC 0.969** — far above the project's ensemble target of 0.89.

**Highest-leverage direction:** the modeling problem is currently too easy, and the repo has zero automated tests despite two documented silent-failure bugs. Before investing in the LSTM/ensemble, (1) resolve the churn-rate discrepancy (actual 12.2% vs the 15–18% the design docs predict), (2) decide whether to harden the synthetic data so the dual-model story is meaningful, and (3) add a small pytest + CI layer around the pure-pandas dataset modules. These three items protect everything downstream and cost days, not weeks.

---

## 1. Current-state summary

**Architecture** (verified in code, not just docs):

- `src/data_generation/` — 5 generators + `run_all.py` entry point; `SCALE`/`SEED` env vars; persona-driven behavior with churn decay, causal quality→behavior links, noise injection (design in `configs/data_design.md`).
- `src/feature_engineering/` — PySpark pipeline (`pipeline.py`) producing `weekly_features.parquet`; 6 feature modules over a 4-week observation window; runs inside the Docker Spark dev container (`infrastructure/docker/docker-compose.yml`: Spark 3.5.6 master + 2 workers + Postgres/pgAdmin).
- `src/dataset/` — labels (zero sessions in weeks 9–10 → churn=1, `labels.py`), XGBoost flat aggregation + LSTM 3D tensor (`aggregation.py`), stratified 70/15/15 split seed=42 (`split.py`), end-to-end `pipeline.py`.
- `src/models/` — config, data loaders (with train-fitted StandardScaler for LSTM), shared evaluator (val-selected threshold frozen for test — methodologically correct), XGBoost train/persist. LSTM hyperparameters defined in `config.py` but no LSTM module yet.
- `src/evaluation/`, `src/serving/` — empty placeholders.
- `notebooks/` — 8 notebooks; phase2/phase3 have dedicated validation notebooks; `phase4_xgboost.ipynb` covers EDA (9 sections incl. leakage sanity) + training.

**Phase status** (per `PROJECT_PLAN.md` §3, confirmed against git history and artifacts on disk):

| Phase | Status |
|---|---|
| 1 — Data generation | Done (v0.1.0-phase1); regenerated 2026-05 after decay-timing fix |
| 2 — Feature engineering | Done (PR #1) |
| 3 — Dataset construction | Done (PR #2) |
| 4 — Model training | In progress: XGBoost v1 trained (`models/xgboost_v1_*`); SHAP, LSTM, ensemble pending |
| 5 — MLflow | Not started |
| 6 — AWS deployment | Not started |
| 7 — MLOps (CI/CD, monitoring, retraining) | Not started |

**XGBoost v1 results** (`models/xgboost_v1_metrics.json`): test ROC-AUC 0.969, PR-AUC 0.709, F1 0.799, recall 0.882 at val-selected threshold 0.69. Early stopping at 60 trees.

**Tests:** none. No `tests/` directory, no test files anywhere (verified by `find`). Nothing to run. The README's project-structure section advertises a `tests/` directory that does not exist.

**CI:** none. No `.github/workflows/`. No lint config in `pyproject.toml` (the devcontainer references black/pylint/ruff extensions, but the repo carries no tool config).

## 2. Strengths worth preserving

- **Docs discipline is genuinely unusual for a solo project.** `configs/temporal_conventions.md` is a canonical timeline with a written postmortem of the off-by-one decay bug (§5); `configs/data_design.md` records the generative ground truth with per-section modeling implications; `PROJECT_PLAN.md` keeps a dated decision log with rationale. Keep this pattern for every phase.
- **Notebook-first → `src/` extraction workflow**, with extraction validated (commit 609cd9b claims byte-identical output vs notebook). Both Phase 2 and Phase 3 followed it.
- **Methodological hygiene in Phase 4:** threshold selected on val and frozen for test (`src/models/xgboost_model.py:save_artifacts`), `scale_pos_weight` computed at fit time, seeds fixed at 42 end-to-end, leakage sanity check in the EDA (`persona` exclusion is enforced by the docs and checked in notebook §1.6).
- **Reproducibility:** deterministic seeds, `run_all.py` single entry point, self-contained Docker Spark environment, uv-managed deps with lockfile.
- **Git hygiene:** conventional commits, feature branches per phase, PRs to `develop`. Two subtle bugs (decay off-by-one, gitignore swallowing `src/models/`) were caught, fixed, and documented rather than papered over.

## 3. Key risks and bottlenecks

1. **Zero automated tests, with a track record that proves the need.** Both documented production bugs were silent failures: the 0-indexed decay bug shipped wrong data for weeks (churn rate 3.5% instead of ~15%, `temporal_conventions.md` §5), and the unanchored `.gitignore` rule made commit 2b79a29 land empty (`PROJECT_PLAN.md` decision log 2026-06-07). `src/dataset/` and `src/models/evaluate.py` are pure pandas/numpy functions — cheap to test, currently untested.
2. **Churn rate discrepancy — possible residual labeling/generation issue.** `temporal_conventions.md` §3 predicts ~15–18% churn (about_to_churn = 15% of users, ≈100% of them decaying to zero by week 9, plus sparse casuals). Actual rate is 12.2% (`configs/feature_spec.md:87`; 917/7,499 positives in `models/xgboost_v1_metrics.json`). That is *below* the theoretical floor, which suggests some about_to_churn users still have sessions in weeks 9–10 (or the expectation in the doc is wrong). **Uncertain — needs investigation before LSTM training**, because if generation is subtly off, all downstream artifacts get regenerated again.
3. **The AUC 0.89 ensemble target is already obsolete.** XGBoost alone scores 0.969 test AUC. On this data, the LSTM cannot demonstrate incremental value against a saturated baseline, which undercuts the project's core "dual-model" narrative (`README.md` Key Highlights). This is a strategic fork: harden the data (weaker/noisier decay, overlapping personas, partial churn) or reframe success metrics (e.g., PR-AUC, recall at fixed precision, performance on a hard cohort).
4. **Silent-misalignment hazard in `src/dataset/aggregation.py:reshape_lstm`.** The `.values.reshape(n_users, 4, n_features)` at line ~114 assumes every user has exactly 4 weekly rows. If a future Phase 2 change drops a user-week, features silently shift against labels. One `assert len(lstm_df) == n_users * 4` removes the hazard.
5. **Three path conventions across three phases.** `src/models/config.py` resolves the repo root from `__file__` (robust); `src/dataset/pipeline.py` uses cwd-relative `"data/raw"` (breaks unless run from repo root); `src/feature_engineering/pipeline.py` hardcodes the container path `/home/spark/work/...`. Inconsistent and fragile as more entry points appear.
6. **README is stale and over-promises.** Roadmap checkboxes show Phases 2–3 incomplete (`README.md:123-129`) though both merged; the tech-stack table and structure diagram list MLflow, Optuna, Terraform, Evidently, GitHub Actions, and `tests/` — none present in the repo yet. For a portfolio repo, the README is the landing page; the gap reads badly. `pyproject.toml` still says "Add your description here".
7. **Model metrics are not versioned.** `/models/` is fully gitignored, so `xgboost_v1_metrics.json` exists only on this machine. Until MLflow (Phase 5), experiment results have no durable record.
8. **Orphan artifacts:** `data/processed/lstm_features.npz` and `xgboost_features.parquet` (pre-split intermediates, May 20) are referenced by no `src/` code. Uncertain whether any notebook still reads them; likely stale.
9. **PySpark absent from `pyproject.toml`** — intentional (Spark lives in the container per `infrastructure/docker/requirements.txt`), but it means `src/feature_engineering` is only runnable in the dev container, and the container's `requirements.txt` pins only pyspark. Fine for now; worth a note in README's setup section.

## 4. Recommended roadmap

### Now — small, high-confidence (days)

1. **Investigate the 12.2% vs 15–18% churn-rate gap** before any further model work. Check: do about_to_churn users have sessions in weeks 9–10? Does signup timing interact with the window? Outcome is either a doc fix or a data regen — both cheap now, expensive after LSTM/ensemble are trained.
2. **Finish the open Phase 4 XGBoost item: SHAP feature importance** (`PROJECT_PLAN.md:83`, already planned; loaders return DataFrames specifically to keep column names for SHAP).
3. **Bootstrap `tests/` with pytest over the pure functions:** `labels.compute_churn_labels`, `split.stratified_split` (ratios, stratification, determinism), `aggregation.aggregate_xgboost`/`reshape_lstm` (tiny fixture frames), `evaluate.compute_metrics`/`find_best_threshold`. Add a data test that asserts the churn rate lands in the expected band — this codifies the §5 postmortem lesson permanently.
4. **Add the assert in `reshape_lstm`** (risk #4).
5. **README sync:** tick Phase 2–3 checkboxes, split the tech stack into "implemented" vs "planned", fix `pyproject.toml` description, note that feature engineering runs in the dev container.
6. **Add ruff config + a minimal GitHub Actions workflow** (ruff + pytest on PR). This is Phase 7 work brought forward at trivial cost, and it starts the CI/CD portfolio story early.

### Next — medium (1–2 weeks)

7. **Decide the data-hardness question** (open question #2 below), then act: if hardening, adjust `generate_session_logs.py` decay/noise parameters and regenerate; the two prior regen cycles show the pipeline reruns cleanly.
8. **LSTM model + training loop** — `LSTM_PARAMS`/`LSTM_TRAIN_PARAMS` already defined in `src/models/config.py`, loaders and scaler handling done. Define up front what "LSTM adds value" means (e.g., recall at fixed precision, or performance on users whose decay starts late in the window).
9. **Ensemble + comparative evaluation writeup** against the pre-registered success criterion, not raw AUC.
10. **Phase 5 MLflow** — planned at 1 day; do it before big LSTM hyperparameter sweeps so runs are tracked, and it fixes the unversioned-metrics problem (risk #7).
11. **Unify path handling** into one shared config (repo-root resolution as in `src/models/config.py`), add `argparse` CLI entry points to the three pipelines.
12. **Promote the phase2/phase3 validation notebooks into a runnable `validate.py`** so the 14 dataset checks can run in CI after any regen.

### Later — larger bets

13. **Phase 6 AWS deployment** (SageMaker, S3, ECR) + populate `src/serving/` with batch/endpoint inference.
14. **Phase 7 MLOps:** CI/CD deploy pipeline, Evidently drift monitoring, retraining simulation using the reserved weeks 11–12 buffer (`temporal_conventions.md` §2 — designed for exactly this, currently unused).
15. **Full-scale run (200K users / 30M+ sessions)** on the Spark cluster — the two-tier scale design exists for this; do it once the pipeline is test-covered.
16. **Deferred feature:** `quality_downgrade_count` (`configs/feature_spec.md:39`) if SHAP shows quality features matter.
17. Clean up orphan intermediates in `data/processed/` after confirming no notebook depends on them.

## 5. Testing and quality plan

Current state: **no tests exist and nothing can be run** — stated plainly.

- **Unit tests (Now):** pure-pandas modules in `src/dataset/` and `src/models/evaluate.py` with hand-built 5–10 row fixtures. No Spark, no data files, sub-second suite.
- **Data/invariant tests (Now):** after-generation checks — churn rate in expected band, 4 rows per user in `weekly_features`, no persona column in model features, split disjointness and stratification. These target the failure class this project has actually experienced.
- **Spark feature tests (Later):** heavier; either run `local[*]` PySpark in CI or lean on the pandas-equivalent notebook as a parity oracle. Not worth blocking on now.
- **CI (Now):** GitHub Actions — ruff + pytest on push/PR to `develop`. Extend in Phase 7 with data-validation and training-smoke jobs.
- **Lint/format:** adopt ruff (already in the devcontainer extension list) with config in `pyproject.toml`.

## 6. Open questions to answer before implementation

1. **Why is churn 12.2% when the design predicts ≥15%?** Bug, signup-timing interaction, or stale doc? Determines whether a third data regen is needed. (Highest priority.)
2. **Harden the data or reframe the target?** With XGBoost at 0.969 AUC, what result would make the LSTM/ensemble chapter convincing to a portfolio reviewer? Pick the success criterion before building the LSTM.
3. **Where should experiment results live until Phase 5?** Commit metrics JSONs, start MLflow early, or accept local-only?
4. **When to do the 200K full-scale run** — before AWS (validates Spark pipeline at scale locally) or on AWS (EMR/Glue story)?
5. **Is the `evaluation/` module meant to absorb `src/models/evaluate.py` plus SHAP**, or stay separate? Two plausible homes exist today.
