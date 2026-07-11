"""Data cleaning and the reusable preprocessing pipeline.

Cleaning (done once, result committed to ``data/processed``):
    * binarise the 0-4 ``num`` label into ``target``
    * coerce every feature to numeric (raw file uses ``?`` for missing)
    * drop duplicate rows

Preprocessing (fit on train only, shipped inside the model pipeline):
    * numeric features  -> median imputation + standard scaling
    * categorical codes -> most-frequent imputation + one-hot encoding

Usage:
    python -m src.data.preprocess
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw UCI frame into a modelling-ready frame."""
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "num" in df.columns:
        df[config.TARGET] = (df["num"] > 0).astype(int)
        df = df.drop(columns=["num"])

    df = df.drop_duplicates().reset_index(drop=True)
    return df[config.ALL_FEATURES + [config.TARGET]]


def build_preprocessor() -> ColumnTransformer:
    """Preprocessing transformer reused verbatim at inference time."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, config.NUMERIC_FEATURES),
            ("categorical", categorical, config.CATEGORICAL_FEATURES),
        ]
    )


def split_data(df: pd.DataFrame):
    """Stratified train/test split on the cleaned frame."""
    X = df[config.ALL_FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def run() -> pd.DataFrame:
    """Clean the raw file and persist the processed CSV."""
    raw = pd.read_csv(config.RAW_DATA_FILE)
    cleaned = clean(raw)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(config.CLEAN_DATA_FILE, index=False)
    logger.info(
        "Saved cleaned data: %d rows, %d columns -> %s",
        cleaned.shape[0],
        cleaned.shape[1],
        config.CLEAN_DATA_FILE,
    )
    return cleaned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = run()
    print(frame[config.TARGET].value_counts().rename("class counts"))
