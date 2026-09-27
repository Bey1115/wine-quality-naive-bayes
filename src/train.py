"""Train and save the Gaussian Naive Bayes model."""

from pathlib import Path

import joblib
from sklearn.naive_bayes import GaussianNB

from preprocessing import (
    detect_target_column,
    load_data,
    make_scaling_pipeline,
    make_train_test_split,
    prepare_features,
    validate_dataset,
)


def train_model(
    data_path: str = "data/train.csv",
    model_path: str = "models/gaussian_naive_bayes.joblib",
):
    df = load_data(data_path)
    target = detect_target_column(df)
    validate_dataset(df, target)

    X, y = prepare_features(df, target)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    pipeline = make_scaling_pipeline(GaussianNB())
    pipeline.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "feature_names": X.columns.tolist(),
            "target_column": target,
        },
        model_path,
    )

    return pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pipeline, X_train, X_test, y_train, y_test = train_model()
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print("Model saved to models/gaussian_naive_bayes.joblib")
