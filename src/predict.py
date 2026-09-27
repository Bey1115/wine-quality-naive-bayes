"""Load the trained model and predict a new wine sample."""

from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path="models/gaussian_naive_bayes.joblib"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run src/train.py first."
        )
    return joblib.load(path)


def predict_sample(sample: pd.DataFrame, model_path="models/gaussian_naive_bayes.joblib"):
    """Predict the class of one or more wine samples."""
    artifact = load_model(model_path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    missing = [column for column in feature_names if column not in sample.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    sample = sample[feature_names]
    predictions = model.predict(sample)
    probabilities = model.predict_proba(sample)

    return predictions, probabilities


if __name__ == "__main__":
    artifact = load_model()
    print("Expected feature order:")
    print(artifact["feature_names"])
    print("\nCreate a DataFrame with these columns and call predict_sample().")
