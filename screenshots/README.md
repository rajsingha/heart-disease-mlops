# Screenshots for the report

Capture these and reference them from `docs/REPORT.md`:

| File | What to capture | How |
|---|---|---|
| `mlflow_runs.png` | Experiment run comparison table | `mlflow ui --backend-store-uri sqlite:///mlflow.db` → open http://127.0.0.1:5000, select the 3 runs → Compare |
| `mlflow_run_detail.png` | One run's params/metrics/artifacts (ROC + confusion matrix) | click the `logistic_regression` run |
| `swagger_ui.png` | Interactive API docs | `uvicorn src.api.main:app --port 8000` → http://localhost:8000/docs |
| `predict_response.png` | A successful `/predict` call | Swagger "Try it out" or the curl from README |
| `docker_run.png` | Container running + curl responses | `docker run --rm -p 8000:8000 heart-disease-api:latest` |
| `ci_pipeline.png` | Green workflow run (all 4 jobs) | GitHub → Actions tab after pushing |
| `kubectl_get_pods.png` | 2/2 pods Running + Service | `kubectl get pods,svc -l app=heart-disease-api` |
| `k8s_predict.png` | curl against the LoadBalancer URL | see `k8s/README.md` |
| `grafana_dashboard.png` | "Heart Disease API" dashboard with traffic | `docker compose up --build`, send a few predictions, open http://localhost:3000 |

Note: EDA figures and model evaluation plots don't need screenshots — they are
generated files in `reports/figures/` and inside MLflow artifacts.
