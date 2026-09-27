"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_CANDIDATES = ["Class", "class", "target", "Target", "label", "Label"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the Kaggle dataset and save it as data/train.csv."
        )
    return pd.read_csv(path)


def detect_target_column(df: pd.DataFrame) -> str:
    """Detect a likely target column."""
    for column in TARGET_CANDIDATES:
        if column in df.columns:
            return column

    # Fall back to a common Kaggle pattern where the first non-feature
    # column is a class label.
    for column in df.columns:
        normalized = str(column).strip().lower()
        if normalized in {"class", "target", "label", "category"}:
            return column

    raise ValueError(
        "Could not automatically identify the target column. "
        f"Available columns: {list(df.columns)}"
    )


def validate_dataset(df: pd.DataFrame, target_column: str) -> None:
    """Validate the assignment requirements."""
    if len(df) < 100:
        raise ValueError(f"Dataset has only {len(df)} rows; at least 100 are required.")

    feature_count = len(df.columns) - 1
    if feature_count < 5:
        raise ValueError(
            f"Dataset has only {feature_count} features; at least 5 are required."
        )

    class_count = df[target_column].nunique(dropna=True)
    if class_count < 3:
        raise ValueError(
            f"Target has only {class_count} classes; at least 3 are required."
        )


def prepare_features(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate numeric features and target."""
    working = df.copy()

    # Remove obvious index columns created by CSV exports.
    index_like = [
        c for c in working.columns
        if str(c).lower().startswith("unnamed:")
    ]
    if index_like:
        working = working.drop(columns=index_like)

    X = working.drop(columns=[target_column])
    y = working[target_column]

    # Naive Bayes in this project is designed for numeric wine measurements.
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            "Non-numeric feature columns found: "
            f"{non_numeric}. Remove identifier/text columns before training."
        )

    if X.isnull().any().any():
        # Median imputation keeps the project robust if a CSV has missing values.
        X = X.fillna(X.median(numeric_only=True))

    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Create a stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def make_scaling_pipeline(model):
    """Return a standardization + model pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
