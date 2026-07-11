"""Batch/CLI inference against the trained pipeline — no API required.

Usage:
    python -m src.models.predict                     # demo on a sample patient
    python -m src.models.predict --json patient.json # single record from file
    python -m src.models.predict --csv patients.csv  # batch scoring
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src import config

SAMPLE_PATIENT = {
    "age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 241, "fbs": 0,
    "restecg": 0, "thalach": 123, "exang": 1, "oldpeak": 0.2, "slope": 2,
    "ca": 0, "thal": 7,
}


def load_model(model_path: Path = config.MODEL_FILE):
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run `python -m src.models.train` first"
        )
    return joblib.load(model_path)


def predict_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Score a dataframe of patient records; returns predictions + probability."""
    features = frame[config.ALL_FEATURES]
    result = frame.copy()
    result["probability"] = model.predict_proba(features)[:, 1].round(4)
    result["prediction"] = (result["probability"] >= 0.5).astype(int)
    result["label"] = result["prediction"].map(
        {1: "heart_disease", 0: "no_heart_disease"}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart disease inference")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--json", type=Path, help="path to a single-record JSON file")
    source.add_argument("--csv", type=Path, help="path to a CSV of patient records")
    args = parser.parse_args()

    if args.csv:
        frame = pd.read_csv(args.csv)
    elif args.json:
        frame = pd.DataFrame([json.loads(args.json.read_text())])
    else:
        frame = pd.DataFrame([SAMPLE_PATIENT])
        print("No input given — scoring a built-in sample patient.\n")

    scored = predict_frame(load_model(), frame)
    columns = ["prediction", "label", "probability"]
    print(scored[columns].to_string(index=(len(scored) > 1)))


if __name__ == "__main__":
    main()
