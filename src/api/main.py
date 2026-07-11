"""FastAPI service exposing the trained heart-disease classifier.

Endpoints:
    GET  /          basic service info
    GET  /health    liveness/readiness probe (checks model is loaded)
    POST /predict   JSON patient record -> prediction + probability
    GET  /metrics   Prometheus metrics (via instrumentator)

Env vars:
    MODEL_PATH           path to the joblib pipeline (default models/model.joblib)
    MODEL_METADATA_PATH  path to the metadata JSON (default models/model_metadata.json)

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import HealthResponse, PatientData, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("heart_disease_api")

PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Number of predictions served, by predicted class",
    ["predicted_class"],
)

state: dict = {"model": None, "metadata": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Resolved at startup (not import) so tests can point env vars elsewhere.
    model_path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    metadata_path = Path(os.getenv("MODEL_METADATA_PATH", "models/model_metadata.json"))
    if model_path.exists():
        state["model"] = joblib.load(model_path)
        if metadata_path.exists():
            state["metadata"] = json.loads(metadata_path.read_text())
        logger.info(
            "Loaded model '%s' from %s",
            state["metadata"].get("model_family", "unknown"),
            model_path,
        )
    else:
        logger.error(
            "Model file not found at %s — /predict will return 503", model_path
        )
    yield
    state["model"] = None


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Predicts the risk of heart disease from patient health data "
        "(UCI Heart Disease dataset). Built for the MLOps assignment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        '%s %s -> %d (%.1f ms)',
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/", tags=["info"])
def root():
    return {
        "service": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    loaded = state["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_family=state["metadata"].get("model_family"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(patient: PatientData):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = pd.DataFrame([patient.model_dump()])
    try:
        probability = float(state["model"].predict_proba(features)[0, 1])
    except Exception as exc:
        logger.exception("Inference failed for payload %s", patient.model_dump())
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    prediction = int(probability >= 0.5)
    label = "heart_disease" if prediction else "no_heart_disease"
    PREDICTION_COUNTER.labels(predicted_class=label).inc()
    logger.info("Prediction: %s (p=%.3f)", label, probability)

    return PredictionResponse(
        prediction=prediction,
        label=label,
        probability=round(probability, 4),
        model_family=state["metadata"].get("model_family", "unknown"),
    )
