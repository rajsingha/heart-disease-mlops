"""Download the UCI Heart Disease (Cleveland) dataset.

Tries the `ucimlrepo` package first and falls back to the raw file on the
UCI archive. Writes a headered CSV with `?` markers converted to NaN.

Usage:
    python -m src.data.download
"""

import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def _fetch_via_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    heart = fetch_ucirepo(id=config.UCI_DATASET_ID)
    df = pd.concat([heart.data.features, heart.data.targets], axis=1)
    df.columns = config.RAW_COLUMN_NAMES
    return df


def _fetch_via_url() -> pd.DataFrame:
    return pd.read_csv(
        config.UCI_FALLBACK_URL,
        header=None,
        names=config.RAW_COLUMN_NAMES,
        na_values="?",
    )


def download(force: bool = False) -> pd.DataFrame:
    """Download the raw dataset to ``data/raw`` and return it."""
    if config.RAW_DATA_FILE.exists() and not force:
        logger.info("Raw data already present at %s", config.RAW_DATA_FILE)
        return pd.read_csv(config.RAW_DATA_FILE)

    try:
        df = _fetch_via_ucimlrepo()
        logger.info("Fetched dataset via ucimlrepo (id=%d)", config.UCI_DATASET_ID)
    except Exception as exc:  # noqa: BLE001 - fall back to direct download
        logger.warning("ucimlrepo failed (%s); falling back to direct URL", exc)
        df = _fetch_via_url()
        logger.info("Fetched dataset from %s", config.UCI_FALLBACK_URL)

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DATA_FILE, index=False)
    logger.info("Saved %d rows to %s", len(df), config.RAW_DATA_FILE)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = download()
    print(f"Downloaded {frame.shape[0]} rows x {frame.shape[1]} columns")
    print(frame.head())
