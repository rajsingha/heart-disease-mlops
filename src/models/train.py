"""Train, tune, and compare classifiers with full MLflow tracking.

For each candidate model family a GridSearchCV (stratified 5-fold, refit on
ROC-AUC) is run inside its own MLflow run, logging best hyper-parameters,
cross-validation metrics, held-out test metrics, confusion-matrix and ROC
plots, and the fitted sklearn pipeline. The best family by test ROC-AUC is
exported to ``models/model.joblib`` together with a metadata JSON.

Usage:
    python -m src.models.train
"""

import json
import logging
import platform
from datetime import UTC, datetime

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.data.preprocess import build_preprocessor, split_data

logger = logging.getLogger(__name__)

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

CANDIDATES = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=5000, solver="liblinear"),
        "param_grid": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            # l1_ratio 0.0 == L2 penalty, 1.0 == L1 (sklearn >= 1.8 API)
            "model__l1_ratio": [0.0, 1.0],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=config.RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
        },
    },
}


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    """Held-out test metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }


def _log_plots(pipeline: Pipeline, X_test, y_test, name: str) -> None:
    """Log confusion matrix and ROC curve to MLflow and reports/figures."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix — {name}")
    mlflow.log_figure(fig, f"plots/confusion_matrix_{name}.png")
    fig.savefig(
        config.FIGURES_DIR / f"confusion_matrix_{name}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"ROC curve — {name}")
    mlflow.log_figure(fig, f"plots/roc_curve_{name}.png")
    fig.savefig(
        config.FIGURES_DIR / f"roc_curve_{name}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def train_candidate(name: str, spec: dict, X_train, X_test, y_train, y_test) -> dict:
    """Grid-search one model family inside an MLflow run."""
    pipeline = Pipeline(
        [("preprocessor", build_preprocessor()), ("model", spec["estimator"])]
    )
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    search = GridSearchCV(
        pipeline,
        spec["param_grid"],
        scoring=SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name=name):
        logger.info("Tuning %s over %s", name, spec["param_grid"])
        search.fit(X_train, y_train)

        mlflow.set_tag("model_family", name)
        mlflow.log_params(
            {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        )
        mlflow.log_param("cv_folds", config.CV_FOLDS)

        best_idx = search.best_index_
        cv_metrics = {
            f"cv_{metric}": search.cv_results_[f"mean_test_{metric}"][best_idx]
            for metric in SCORING
        }
        cv_metrics["cv_roc_auc_std"] = search.cv_results_["std_test_roc_auc"][best_idx]
        test_metrics = evaluate(search.best_estimator_, X_test, y_test)
        mlflow.log_metrics({**cv_metrics, **test_metrics})

        _log_plots(search.best_estimator_, X_test, y_test, name)
        mlflow.log_text(
            classification_report(y_test, search.best_estimator_.predict(X_test)),
            f"reports/classification_report_{name}.txt",
        )
        mlflow.sklearn.log_model(
            search.best_estimator_,
            name="model",
            input_example=X_train.head(3).astype("float64"),
            serialization_format="cloudpickle",
        )

        logger.info(
            "%s: cv_roc_auc=%.4f test_roc_auc=%.4f",
            name,
            cv_metrics["cv_roc_auc"],
            test_metrics["test_roc_auc"],
        )
        return {
            "name": name,
            "pipeline": search.best_estimator_,
            "best_params": search.best_params_,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "run_id": mlflow.active_run().info.run_id,
        }


def save_best(result: dict) -> None:
    """Persist the winning pipeline and its metadata for serving."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["pipeline"], config.MODEL_FILE)

    metadata = {
        "model_family": result["name"],
        "best_params": {
            k.removeprefix("model__"): v for k, v in result["best_params"].items()
        },
        "cv_metrics": {k: round(float(v), 4) for k, v in result["cv_metrics"].items()},
        "test_metrics": {
            k: round(float(v), 4) for k, v in result["test_metrics"].items()
        },
        "mlflow_run_id": result["run_id"],
        "trained_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "features": {
            "numeric": config.NUMERIC_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
        },
    }
    config.MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved best model (%s) to %s", result["name"], config.MODEL_FILE)


def run() -> dict:
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    df = pd.read_csv(config.CLEAN_DATA_FILE)
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    results = [
        train_candidate(name, spec, X_train, X_test, y_train, y_test)
        for name, spec in CANDIDATES.items()
    ]

    comparison = pd.DataFrame(
        [{"model": r["name"], **r["cv_metrics"], **r["test_metrics"]} for r in results]
    ).set_index("model")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv")
    logger.info("Model comparison:\n%s", comparison.round(4).to_string())

    best = max(results, key=lambda r: r["test_metrics"]["test_roc_auc"])
    save_best(best)
    return best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    winner = run()
    print(f"\nBest model: {winner['name']}")
    print(json.dumps(winner["test_metrics"], indent=2))
