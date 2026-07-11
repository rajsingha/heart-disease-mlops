"""Central configuration: paths, feature schema, and training constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "heart_disease_raw.csv"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "heart_disease_clean.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# UCI Heart Disease (Cleveland) column schema. The raw file has no header row.
RAW_COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# `num` (0-4) is binarised into `target` (0 = no disease, 1 = disease).
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

UCI_DATASET_ID = 45  # ucimlrepo id for "Heart Disease"
UCI_FALLBACK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

MLFLOW_EXPERIMENT = "heart-disease-classification"
