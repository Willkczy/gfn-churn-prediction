# GFN Churn Prediction

An end-to-end machine learning pipeline for predicting user churn on a cloud gaming streaming platform (modeled after NVIDIA GeForce NOW). Built from synthetic data generation through full MLOps deployment on AWS.

## Motivation

This project recreates and extends a real-world churn prediction system originally developed in a research collaboration between Taiwan Mobile and NTU (Jan–Jun 2024). The goal is to build a complete, production-grade ML pipeline that can be understood, reproduced, and extended by anyone.

## Key Highlights

- **Dual-model framework**: XGBoost for interpretable feature analysis + LSTM for capturing sequential behavior patterns across 4-week windows
- **Realistic synthetic data**: 3.5M+ streaming sessions generated with persona-driven behavior, dynamic causal relationships (e.g., poor streaming quality → early session termination), and injected noise/outliers
- **42 engineered features** (flat per-user format) from session patterns, engagement decay, streaming quality, game diversity, playtime volatility, and payment behavior — plus a 4-week sequential tensor (38 features/week) for the LSTM
- **SHAP-based explainability** to translate model insights into actionable product retention strategies *(in progress — Phase 4)*
- **Full MLOps on AWS** *(planned — Phases 5–7)*: SageMaker deployment, CI/CD via GitHub Actions, model monitoring with Evidently AI, infrastructure as code with Terraform

## Architecture

Target architecture — stages through dataset construction and XGBoost training are built; MLflow, AWS deployment, and monitoring are planned (Phases 5–7).

```
Synthetic Data ──→ PySpark Feature ──→ Dataset ──→ Model Training ──→ AWS Deployment
 Generation         Engineering       Construction  (XGBoost+LSTM)    (SageMaker)
     │                                                   │                  │
     │                                              MLflow Tracking    Monitoring
     │                                                                (Evidently AI)
     └── 5 tables ──────────────────────────────────── CI/CD (GitHub Actions) ──┘
```

## Data Overview

The pipeline generates 5 relational tables simulating a cloud gaming streaming service:

| Table | Description | Rows (small) | Key Characteristics |
|---|---|---|---|
| `users` | User profiles with demographic info | 50K | 4 behavioral personas, Taiwan county regions |
| `session_logs` | Streaming session records | 3.55M | Causal quality→duration impact, noise injection |
| `game_catalog` | Game metadata | 200 | Popularity power law distribution |
| `subscription_events` | Plan changes (upgrade/downgrade/cancel) | 38.7K | Temporally-biased toward churn patterns |
| `payments` | Transaction records | 238.9K | Persona-driven failure/refund rates |

Data generation supports two scales — `small` (50K users) for fast development iteration and `full` (200K users, 30M+ sessions) for PySpark practice at realistic scale.

## Tech Stack

| Layer | Technology | Status |
|---|---|---|
| Data Generation | Python, NumPy, Pandas, Faker | ✅ Implemented |
| Storage | Parquet (local) | ✅ Implemented (S3 planned) |
| Feature Engineering | PySpark (Docker Spark cluster) | ✅ Implemented |
| ML Training | XGBoost, scikit-learn | ✅ Baseline trained |
| ML Training | PyTorch (LSTM), ensemble | 🔨 In progress |
| Explainability | SHAP | 🔨 In progress |
| Experiment Tracking | MLflow | Planned (Phase 5) |
| Containerization | Docker | ✅ Spark dev environment |
| Model Serving | AWS SageMaker | Planned (Phase 6) |
| Testing | pytest, ruff | ✅ Implemented |
| CI/CD | GitHub Actions | ✅ Lint + unit tests on push/PR |
| Infrastructure | Terraform | Planned (Phase 6) |
| Monitoring | Evidently AI, AWS CloudWatch | Planned (Phase 7) |

## Project Structure

```
gfn-churn-prediction/
├── src/
│   ├── data_generation/       # Phase 1: Synthetic data generators
│   ├── feature_engineering/   # Phase 2: PySpark feature pipeline (runs in the Spark dev container)
│   ├── dataset/               # Phase 3: Labeling, splitting, XGBoost/LSTM formatting
│   ├── models/                # Phase 4: XGBoost (trained), LSTM + ensemble (in progress)
│   ├── evaluation/            # SHAP analysis (planned)
│   └── serving/               # Inference scripts (planned — Phase 6)
├── data/
│   ├── raw/                   # Generated synthetic data (parquet, gitignored)
│   └── processed/             # Weekly features + train/val/test splits (gitignored)
├── models/                    # Trained model artifacts + metrics (gitignored)
├── notebooks/                 # Phase-by-phase EDA, development, and validation notebooks
├── configs/                   # Canonical docs: data design, feature spec, temporal conventions
├── tests/                     # Unit tests (pytest) + data-invariant checks (marked `data`)
├── .github/workflows/         # CI: ruff lint + unit tests on push/PR
├── infrastructure/
│   └── docker/                # Spark dev cluster (terraform/ planned for Phase 6)
├── PROJECT_PLAN.md            # Phases, progress checklist, decision log
├── DEVELOPMENT_PLAN.md        # Evidence-based roadmap (from repo audit)
├── IMPLEMENTATION_PLAN.md     # Milestone-by-milestone execution plan
└── pyproject.toml             # Dependencies (managed with uv)
```

Planned but not yet present: `infrastructure/terraform/` (see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)).

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for package management

### Setup

```bash
git clone https://github.com/<your-username>/gfn-churn-prediction.git
cd gfn-churn-prediction
uv sync
```

### Generate Synthetic Data

```bash
# Small scale (50K users, ~3.5M sessions) — takes ~3 minutes
uv run python -m src.data_generation.run_all

# Full scale (200K users, ~30M+ sessions) — for Spark practice
SCALE=full uv run python -m src.data_generation.run_all
```

Generated data is saved to `data/raw/` in Parquet format.

### Feature Engineering & Dataset Construction

Feature engineering uses PySpark and runs inside the Docker Spark dev container (open the repo in a Dev Container, or start `infrastructure/docker/docker-compose.yml`):

```bash
# Inside the Spark container — writes data/processed/weekly_features.parquet
python -m src.feature_engineering.pipeline
```

Dataset construction (labels, splits, model-ready formats) runs locally from the repo root:

```bash
# Writes xgboost_/lstm_ train/val/test files to data/processed/
uv run python -m src.dataset.pipeline
```

You can also generate individual tables:

```bash
uv run python -m src.data_generation.generate_game_catalog
uv run python -m src.data_generation.generate_users
uv run python -m src.data_generation.generate_session_logs
uv run python -m src.data_generation.generate_subscription_events
uv run python -m src.data_generation.generate_payments
```

### Tests & Lint

```bash
uv run ruff check src/ tests/     # lint
uv run pytest                     # full suite (data-invariant tests skip if data/processed/ is absent)
uv run pytest -m "not data"       # unit tests only (what CI runs)
```

## Roadmap

- [x] **Phase 1** — Synthetic data generation (v0.1.0)
- [x] **Phase 2** — Feature engineering with PySpark
- [x] **Phase 3** — Dataset construction (labeling, train/val/test split)
- [ ] **Phase 4** — Model training (XGBoost baseline trained; SHAP, LSTM, ensemble in progress)
- [ ] **Phase 5** — Experiment tracking with MLflow
- [ ] **Phase 6** — AWS deployment (SageMaker, S3, ECR)
- [ ] **Phase 7** — MLOps (CI/CD, monitoring, drift detection, auto-retraining)

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for progress detail and the decision log, and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the milestone-by-milestone execution plan.

## License

This project is for educational and portfolio purposes.