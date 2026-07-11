# Screenshots

Captures referenced by `docs/REPORT.md`, taken from the live system.

| File | Shows |
|---|---|
| `mlflow_runs.png` | MLflow experiment `heart-disease-classification` — run table with the three tuned model families |
| `mlflow_run_detail.png` | Winning `logistic_regression` run — all CV/test metrics, best hyper-parameters, run metadata |
| `swagger_ui.png` | FastAPI interactive docs (`/docs`) listing `/predict`, `/health`, `/metrics` |
| `predict_response.png` | A real `POST /predict` call with request payload and JSON response (prediction + probability) |
| `docker_run.png` | `docker compose ps` — API (healthy), Prometheus, and Grafana containers + `/health` response |
| `ci_pipeline.png` | Green GitHub Actions run — Lint → Unit tests → Train model → Docker build & smoke test, with uploaded artifacts |
| `kubectl_get_pods.png` | Kubernetes: 2/2 API pods Running + LoadBalancer service |
| `k8s_predict.png` | `/health` and `/predict` served through the Kubernetes LoadBalancer on localhost:80 |
| `grafana_dashboard.png` | Grafana "Heart Disease API" dashboard — request rate, p95 latency, predictions by class, HTTP status codes |

To re-capture: bring the stack up (`docker compose up --build`, `kubectl apply -f k8s/`,
`mlflow ui --backend-store-uri sqlite:///mlflow.db`), send a few `/predict` requests,
and screenshot the pages/commands above.
