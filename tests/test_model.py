"""Tests for model training, evaluation, and persistence."""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data.preprocess import build_preprocessor, split_data
from src.models.train import evaluate

EXPECTED_METRICS = {
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
}


def _fit_quick_pipeline(clean_df):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    ).fit(X_train, y_train)
    return pipeline, X_test, y_test


def test_evaluate_returns_all_metrics_in_range(clean_df):
    pipeline, X_test, y_test = _fit_quick_pipeline(clean_df)
    metrics = evaluate(pipeline, X_test, y_test)
    assert set(metrics) == EXPECTED_METRICS
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def test_model_learns_signal_better_than_chance(clean_df):
    pipeline, X_test, y_test = _fit_quick_pipeline(clean_df)
    metrics = evaluate(pipeline, X_test, y_test)
    assert metrics["test_roc_auc"] > 0.7
    assert metrics["test_accuracy"] > 0.6


def test_saved_model_roundtrip_preserves_predictions(clean_df, tmp_path):
    pipeline, X_test, _ = _fit_quick_pipeline(clean_df)
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    reloaded = joblib.load(path)
    original = pipeline.predict_proba(X_test)[:, 1]
    restored = reloaded.predict_proba(X_test)[:, 1]
    assert (original == restored).all()


def test_pipeline_predicts_on_single_record(clean_df):
    pipeline, X_test, _ = _fit_quick_pipeline(clean_df)
    record = X_test.head(1)
    proba = pipeline.predict_proba(record)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_train_candidate_logs_to_mlflow(clean_df, tmp_path, monkeypatch):
    """End-to-end: grid search one small candidate with MLflow in a temp store."""
    import mlflow

    from src.models import train as train_module

    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.chdir(tmp_path)  # keep MLflow artifacts out of the repo
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("test-experiment")

    X_train, X_test, y_train, y_test = split_data(clean_df)
    spec = {
        "estimator": LogisticRegression(max_iter=2000, solver="liblinear"),
        "param_grid": {"model__C": [0.1, 1.0]},
    }
    result = train_module.train_candidate(
        "logreg_test", spec, X_train, X_test, y_train, y_test
    )

    assert result["test_metrics"]["test_roc_auc"] > 0.6
    run = mlflow.get_run(result["run_id"])
    assert run.data.params["cv_folds"] == str(config.CV_FOLDS)
    assert "test_roc_auc" in run.data.metrics
    assert "cv_roc_auc" in run.data.metrics


def test_save_best_writes_model_and_metadata(clean_df, tmp_path, monkeypatch):
    from src.models import train as train_module

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "model.joblib")
    monkeypatch.setattr(config, "MODEL_METADATA_FILE", tmp_path / "meta.json")

    pipeline, X_test, y_test = _fit_quick_pipeline(clean_df)
    result = {
        "name": "logreg_test",
        "pipeline": pipeline,
        "best_params": {"model__C": 1.0},
        "cv_metrics": {"cv_roc_auc": 0.9},
        "test_metrics": evaluate(pipeline, X_test, y_test),
        "run_id": "dummy",
    }
    train_module.save_best(result)

    assert (tmp_path / "model.joblib").exists()
    metadata = pd.read_json(tmp_path / "meta.json", typ="series")
    assert metadata["model_family"] == "logreg_test"
