"""Unit tests for cleaning and the preprocessing pipeline."""

import numpy as np
import pandas as pd

from src import config
from src.data.preprocess import build_preprocessor, clean, split_data


class TestClean:
    def test_binarises_target(self, raw_df):
        cleaned = clean(raw_df)
        assert config.TARGET in cleaned.columns
        assert "num" not in cleaned.columns
        assert set(cleaned[config.TARGET].unique()) <= {0, 1}

    def test_num_greater_than_zero_maps_to_one(self, raw_df):
        cleaned = clean(raw_df)
        assert (cleaned[config.TARGET] == (raw_df["num"] > 0).astype(int)).all()

    def test_coerces_question_marks_to_nan(self):
        df = pd.DataFrame(
            {c: [1.0, 2.0] for c in config.ALL_FEATURES} | {"num": [0, 2]}
        )
        df["ca"] = ["?", "1"]
        cleaned = clean(df)
        assert np.isnan(cleaned.loc[0, "ca"])
        assert cleaned.loc[1, "ca"] == 1.0

    def test_drops_duplicates(self, raw_df):
        doubled = pd.concat([raw_df, raw_df], ignore_index=True)
        assert len(clean(doubled)) == len(clean(raw_df))

    def test_output_column_order(self, raw_df):
        cleaned = clean(raw_df)
        assert list(cleaned.columns) == config.ALL_FEATURES + [config.TARGET]


class TestPreprocessor:
    def test_fit_transform_removes_missing_values(self, raw_df):
        cleaned = clean(raw_df)
        assert cleaned.isna().any().any(), "fixture should contain missing values"
        transformed = build_preprocessor().fit_transform(cleaned[config.ALL_FEATURES])
        assert not np.isnan(np.asarray(transformed, dtype=float)).any()

    def test_numeric_features_are_standardised(self, clean_df):
        transformed = build_preprocessor().fit_transform(
            clean_df[config.ALL_FEATURES]
        )
        numeric_block = np.asarray(transformed, dtype=float)[
            :, : len(config.NUMERIC_FEATURES)
        ]
        assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-8)
        assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-8)

    def test_handles_unseen_categories(self, clean_df):
        preprocessor = build_preprocessor()
        preprocessor.fit(clean_df[config.ALL_FEATURES])
        oddball = clean_df[config.ALL_FEATURES].head(1).copy()
        oddball["thal"] = 99.0  # category never seen during fit
        transformed = preprocessor.transform(oddball)
        assert not np.isnan(np.asarray(transformed, dtype=float)).any()


class TestSplit:
    def test_split_sizes_and_stratification(self, clean_df):
        X_train, X_test, y_train, y_test = split_data(clean_df)
        assert len(X_train) + len(X_test) == len(clean_df)
        assert abs(len(X_test) / len(clean_df) - config.TEST_SIZE) < 0.02
        overall = clean_df[config.TARGET].mean()
        assert abs(y_train.mean() - overall) < 0.05
        assert abs(y_test.mean() - overall) < 0.05

    def test_split_is_reproducible(self, clean_df):
        first = split_data(clean_df)
        second = split_data(clean_df)
        pd.testing.assert_frame_equal(first[0], second[0])
