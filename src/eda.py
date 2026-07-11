"""Exploratory data analysis: saves publication-quality figures to reports/figures.

Usage:
    python -m src.eda
"""

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="deep")

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1 = male)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST",
    "ca": "Major vessels colored (0-3)",
    "thal": "Thalassemia result",
}


def _save(fig: plt.Figure, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_class_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[config.TARGET].value_counts().sort_index()
    ax.bar(
        ["No disease (0)", "Disease (1)"],
        counts.values,
        color=["#4c72b0", "#c44e52"],
    )
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / len(df):.1%})", ha="center", fontweight="bold")
    ax.set_ylabel("Patients")
    ax.set_title("Class distribution — heart disease target")
    _save(fig, "class_distribution.png")


def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    missing.sort_values(ascending=False).plot.bar(ax=ax, color="#dd8452")
    ax.set_ylabel("Missing values")
    ax.set_title(f"Missing values per column (n = {len(df)} rows)")
    _save(fig, "missing_values.png")


def plot_histograms(df: pd.DataFrame) -> None:
    features = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, features, strict=False):
        sns.histplot(data=df, x=col, hue=config.TARGET, kde=True, ax=ax, alpha=0.6)
        ax.set_xlabel(FEATURE_LABELS.get(col, col))
        ax.set_title(col)
    fig.suptitle("Numeric feature distributions by target", fontsize=14)
    fig.tight_layout()
    _save(fig, "histograms_numeric.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation heatmap (Pearson)")
    _save(fig, "correlation_heatmap.png")


def plot_categorical_vs_target(df: pd.DataFrame) -> None:
    features = config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flat, features, strict=False):
        rates = df.groupby(col)[config.TARGET].mean()
        rates.plot.bar(ax=ax, color="#55a868")
        ax.set_ylabel("P(disease)")
        ax.set_ylim(0, 1)
        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=10)
    axes.flat[-1].axis("off")
    fig.suptitle("Disease rate by categorical feature level", fontsize=14)
    fig.tight_layout()
    _save(fig, "categorical_vs_target.png")


def plot_feature_relationships(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(
        data=df, x="age", y="thalach", hue=config.TARGET, alpha=0.7, ax=axes[0]
    )
    axes[0].set_title("Max heart rate vs age")
    sns.boxplot(data=df, x=config.TARGET, y="oldpeak", hue=config.TARGET, ax=axes[1])
    axes[1].set_title("ST depression by target")
    axes[1].set_xticks([0, 1], ["No disease", "Disease"])
    fig.tight_layout()
    _save(fig, "feature_relationships.png")


def run() -> None:
    df = pd.read_csv(config.CLEAN_DATA_FILE)
    logger.info("Loaded cleaned data: %s", df.shape)
    plot_class_distribution(df)
    plot_missing_values(df)
    plot_histograms(df)
    plot_correlation_heatmap(df)
    plot_categorical_vs_target(df)
    plot_feature_relationships(df)

    summary = df.describe().T
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.REPORTS_DIR / "eda_summary_statistics.csv")
    logger.info("EDA complete: figures in %s", config.FIGURES_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
