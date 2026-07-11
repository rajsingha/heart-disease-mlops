# Heart Disease Prediction — End-to-End MLOps Project

Predicts the risk of heart disease from patient health data (UCI Heart Disease
dataset, Cleveland subset) and serves the model as a containerized, monitored,
Kubernetes-ready FastAPI service.

**Pipeline:** data download → EDA → preprocessing pipeline → model training +
hyper-parameter tuning (MLflow-tracked) → FastAPI serving → Docker →
GitHub Actions CI → Kubernetes → Prometheus/Grafana monitoring.

See [docs/REPORT.md](docs/REPORT.md) for the full project report.

## Project structure

```
├── .github/workflows/ci.yml     # CI: lint → test → train → docker smoke-test
├── data/
│   ├── raw/                     # downloaded UCI data (gitignored, re-downloadable)
│   └── processed/               # cleaned dataset (committed for reproducible CI)
├── docs/                        # assignment brief + final report
├── k8s/                         # Deployment + Service manifests + instructions
├── monitoring/                  # Prometheus config, Grafana provisioning + dashboard
├── models/                      # trained pipeline + metadata (created by training)
├── notebooks/eda.ipynb          # executed EDA notebook
├── reports/                     # EDA figures, model comparison, evaluation plots
├── src/
│   ├── config.py                # paths, feature schema, constants
│   ├── data/download.py         # dataset acquisition (ucimlrepo + URL fallback)
│   ├── data/preprocess.py       # cleaning + sklearn ColumnTransformer pipeline
│   ├── eda.py                   # scripted EDA (saves figures)
│   ├── models/train.py          # GridSearchCV + MLflow tracking + model export
│   └── api/                     # FastAPI app (/predict, /health, /metrics)
├── tests/                       # pytest suite (data, model, API)
├── Dockerfile                   # slim serving image (python:3.12-slim)
├── docker-compose.yml           # API + Prometheus + Grafana stack
├── requirements.txt             # full dev/training environment
└── requirements-api.txt         # pinned serving-only dependencies
```

## Quickstart (clean setup)

Requires Python 3.12+ and (optionally) Docker.

```bash
# 1. Environment
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Data
python -m src.data.download       # fetch UCI Heart Disease -> data/raw/
python -m src.data.preprocess     # clean -> data/processed/heart_disease_clean.csv

# 3. EDA (figures land in reports/figures/)
python -m src.eda

# 4. Train + tune + track (MLflow store: ./mlflow.db, artifacts: ./mlruns)
python -m src.models.train        # exports best model -> models/model.joblib

# 5. Inspect experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://127.0.0.1:5000
```

> **Note (Python 3.14 only):** mlflow 3.14.0's UI server crashes on Python 3.14
> because `importlib.abc.Traversable` was removed from the stdlib. Until the
> upstream fix ships, patch one line in
> `.venv/Lib/site-packages/mlflow/assistant/skill_installer.py`:
> `from importlib.abc import Traversable` →
> `from importlib.resources.abc import Traversable`.
> Python 3.12/3.13 (and the Docker image/CI) are unaffected.

```bash

# 6. Serve the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs (Swagger UI) and try `/predict`:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"age\":57,\"sex\":1,\"cp\":4,\"trestbps\":140,\"chol\":241,\"fbs\":0,\"restecg\":0,\"thalach\":123,\"exang\":1,\"oldpeak\":0.2,\"slope\":2,\"ca\":0,\"thal\":7}"
```

Response:

```json
{"prediction": 1, "label": "heart_disease", "probability": 0.79, "model_family": "..."}
```

## Tests & linting

```bash
pytest -v          # unit tests: data cleaning, pipeline, training, API contract
ruff check src tests
```

Tests run against deterministic synthetic data — no network or trained model
required, so they work in CI from a bare checkout.

## Docker

```bash
python -m src.models.train                     # model must exist first
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
curl http://localhost:8000/health
```

## Monitoring stack (API + Prometheus + Grafana)

```bash
docker compose up --build
```

| Service    | URL                    | Notes                                  |
|------------|------------------------|----------------------------------------|
| API        | http://localhost:8000  | Swagger at /docs, metrics at /metrics  |
| Prometheus | http://localhost:9090  | scrapes the API every 5 s              |
| Grafana    | http://localhost:3000  | admin/admin — dashboard auto-provisioned |

The "Heart Disease API" Grafana dashboard shows request rate, p95 latency,
predictions by class, and HTTP status codes. Every request is also logged
(method, path, status, latency) by the API's logging middleware.

## Kubernetes

Manifests and step-by-step instructions (Docker Desktop / Minikube / cloud)
are in [k8s/](k8s/README.md):

```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
```

## CI/CD

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on
every push/PR to `main`:

1. **Lint** — ruff over `src` and `tests`
2. **Test** — full pytest suite (pipeline fails on any test failure)
3. **Train** — retrains from the committed cleaned dataset, uploads the model,
   MLflow runs, and evaluation reports as workflow artifacts
4. **Docker** — builds the serving image with the freshly trained model, boots
   the container, and smoke-tests `/health` and `/predict`

## API reference

| Method | Path       | Description                                    |
|--------|------------|------------------------------------------------|
| GET    | `/`        | Service info                                   |
| GET    | `/health`  | Liveness/readiness (used by k8s probes)        |
| POST   | `/predict` | Patient JSON → prediction + probability        |
| GET    | `/metrics` | Prometheus metrics                             |
| GET    | `/docs`    | Interactive Swagger UI                         |
