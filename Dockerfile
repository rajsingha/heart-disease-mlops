# Serving image: FastAPI + trained model. Train first (python -m src.models.train)
# so models/model.joblib exists, then: docker build -t heart-disease-api .
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY models/model.joblib models/model_metadata.json models/

ENV MODEL_PATH=/app/models/model.joblib \
    MODEL_METADATA_PATH=/app/models/model_metadata.json \
    PYTHONUNBUFFERED=1

# Run as an unprivileged user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
