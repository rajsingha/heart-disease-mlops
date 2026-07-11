"""Shared fixtures: deterministic synthetic data shaped like the UCI dataset.

Tests never hit the network — they run against generated data whose target is
a noisy function of a few features, so models can genuinely learn from it.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data.preprocess import build_preprocessor


def make_synthetic_clean(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """Synthetic frame matching the cleaned dataset schema (with `target`)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(54, 9, n).clip(29, 77).round(),
            "sex": rng.binomial(1, 0.68, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.normal(131, 17, n).clip(94, 200).round(),
            "chol": rng.normal(246, 50, n).clip(126, 564).round(),
            "fbs": rng.binomial(1, 0.15, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.normal(150, 23, n).clip(71, 202).round(),
            "exang": rng.binomial(1, 0.33, n),
            "oldpeak": rng.exponential(1.0, n).clip(0, 6.2).round(1),
            "slope": rng.integers(1, 4, n),
            "ca": rng.integers(0, 4, n).astype(float),
            "thal": rng.choice([3.0, 6.0, 7.0], n, p=[0.55, 0.1, 0.35]),
        }
    )
    score = (
        0.04 * (df["age"] - 54)
        - 0.03 * (df["thalach"] - 150)
        + 0.9 * df["exang"]
        + 0.6 * df["oldpeak"]
        + 0.8 * (df["cp"] == 4)
        + 0.7 * (df["thal"] == 7.0)
        + 0.5 * df["ca"]
        - 1.5
    )
    prob = 1 / (1 + np.exp(-score))
    df[config.TARGET] = rng.binomial(1, prob)
    return df


def make_synthetic_raw(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """Raw-shaped frame: 0-4 `num` label, missing values in `ca`/`thal`."""
    df = make_synthetic_clean(n, seed)
    rng = np.random.default_rng(seed + 1)
    df["num"] = np.where(df[config.TARGET] == 1, rng.integers(1, 5, n), 0)
    df = df.drop(columns=[config.TARGET])
    df.loc[df.sample(frac=0.03, random_state=seed).index, "ca"] = np.nan
    df.loc[df.sample(frac=0.02, random_state=seed + 2).index, "thal"] = np.nan
    return df


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    return make_synthetic_clean()


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_synthetic_raw()


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory, clean_df):
    """A quick fitted pipeline + metadata saved to a temp dir (for API tests)."""
    model_dir = tmp_path_factory.mktemp("model")
    X = clean_df[config.ALL_FEATURES]
    y = clean_df[config.TARGET]
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    ).fit(X, y)
    joblib.dump(pipeline, model_dir / "model.joblib")
    (model_dir / "model_metadata.json").write_text(
        json.dumps({"model_family": "logistic_regression_test"})
    )
    return model_dir


@pytest.fixture(scope="session")
def api_client(trained_model_dir):
    """TestClient wired to the temp model via env vars."""
    saved = {
        "MODEL_PATH": os.environ.get("MODEL_PATH"),
        "MODEL_METADATA_PATH": os.environ.get("MODEL_METADATA_PATH"),
    }
    os.environ["MODEL_PATH"] = str(trained_model_dir / "model.joblib")
    os.environ["MODEL_METADATA_PATH"] = str(trained_model_dir / "model_metadata.json")
    try:
        from fastapi.testclient import TestClient

        from src.api.main import app

        # env vars are read in the app's lifespan, which TestClient triggers here
        with TestClient(app) as client:
            yield client
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
