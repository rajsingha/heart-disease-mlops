"""API contract tests against a model trained on synthetic data."""

VALID_PAYLOAD = {
    "age": 57,
    "sex": 1,
    "cp": 4,
    "trestbps": 140,
    "chol": 241,
    "fbs": 0,
    "restecg": 0,
    "thalach": 123,
    "exang": 1,
    "oldpeak": 0.2,
    "slope": 2,
    "ca": 0,
    "thal": 7,
}


def test_root_lists_endpoints(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/health"


def test_health_reports_model_loaded(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_prediction_and_probability(api_client):
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("heart_disease", "no_heart_disease")


def test_predict_label_consistent_with_probability(api_client):
    body = api_client.post("/predict", json=VALID_PAYLOAD).json()
    assert body["prediction"] == (1 if body["probability"] >= 0.5 else 0)
    expected = "heart_disease" if body["prediction"] == 1 else "no_heart_disease"
    assert body["label"] == expected


def test_predict_rejects_missing_field(api_client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    assert api_client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_out_of_range_value(api_client):
    payload = VALID_PAYLOAD | {"age": -5}
    assert api_client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_wrong_type(api_client):
    payload = VALID_PAYLOAD | {"chol": "not-a-number"}
    assert api_client.post("/predict", json=payload).status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(api_client):
    api_client.post("/predict", json=VALID_PAYLOAD)
    response = api_client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "model_predictions_total" in response.text
