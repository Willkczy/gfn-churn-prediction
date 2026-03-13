# GFN Churn Prediction

An end-to-end machine learning pipeline for predicting user churn on a cloud gaming streaming platform (modeled after NVIDIA GeForce NOW). Built from synthetic data generation through full MLOps deployment on AWS.

## Motivation

This project recreates and extends a real-world churn prediction system originally developed in a research collaboration between Taiwan Mobile and NTU (Jan–Jun 2024). The goal is to build a complete, production-grade ML pipeline that can be understood, reproduced, and extended by anyone.

## Key Highlights

- **Dual-model framework**: XGBoost for interpretable feature analysis + LSTM for capturing sequential behavior patterns across 4-week windows
- **Realistic synthetic data**: 3.5M+ streaming sessions generated with persona-driven behavior, dynamic causal relationships (e.g., poor streaming quality → early session termination), and injected noise/outliers
- **50+ engineered features** from session patterns, engagement decay, streaming quality, game diversity, playtime volatility, and payment behavior
- **SHAP-based explainability** to translate model insights into actionable product retention strategies
- **Full MLOps on AWS**: SageMaker deployment, CI/CD via GitHub Actions, model monitoring with Evidently AI, infrastructure as code with Terraform

## Architecture

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

| Layer | Technology |
|---|---|
| Data Generation | Python, NumPy, Pandas, Faker |
| Storage | Parquet, AWS S3 |
| Feature Engineering | PySpark |
| ML Training | XGBoost, PyTorch, scikit-learn, Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Model Serving | AWS SageMaker |
| CI/CD | GitHub Actions |
| Infrastructure | Terraform |
| Monitoring | Evidently AI, AWS CloudWatch |

## Project Structure

```
gfn-churn-prediction/
├── src/
│   ├── data_generation/       # Phase 1: Synthetic data generators
│   ├── feature_engineering/   # Phase 2: PySpark feature pipelines
│   ├── dataset/               # Phase 3: Labeling, splitting, formatting
│   ├── models/                # Phase 4: XGBoost, LSTM, ensemble
│   ├── evaluation/            # Metrics and SHAP analysis
│   └── serving/               # Inference scripts
├── data/
│   ├── raw/                   # Generated synthetic data (parquet)
│   ├── processed/             # Feature-engineered data
│   └── splits/                # Train/val/test sets
├── notebooks/                 # EDA and experimentation
├── configs/                   # Hyperparameters, feature lists
├── tests/                     # Unit and integration tests
├── infrastructure/
│   ├── terraform/             # AWS infrastructure as code
│   └── docker/                # Dockerfiles
├── .github/workflows/         # CI/CD pipelines
├── PROJECT_PLAN.md            # Detailed project plan and data schemas
└── pyproject.toml             # Dependencies (managed with uv)
```

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

You can also generate individual tables:

```bash
uv run python -m src.data_generation.generate_game_catalog
uv run python -m src.data_generation.generate_users
uv run python -m src.data_generation.generate_session_logs
uv run python -m src.data_generation.generate_subscription_events
uv run python -m src.data_generation.generate_payments
```

## Roadmap

- [x] **Phase 1** — Synthetic data generation (v0.1.0)
- [ ] **Phase 2** — Feature engineering with PySpark
- [ ] **Phase 3** — Dataset construction (labeling, train/val/test split)
- [ ] **Phase 4** — Model training (XGBoost + LSTM + ensemble)
- [ ] **Phase 5** — Experiment tracking with MLflow
- [ ] **Phase 6** — AWS deployment (SageMaker, S3, ECR)
- [ ] **Phase 7** — MLOps (CI/CD, monitoring, drift detection, auto-retraining)

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full project plan, data schemas, generation logic, and technical decisions.

## License

This project is for educational and portfolio purposes.