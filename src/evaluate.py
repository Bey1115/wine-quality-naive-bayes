"""Evaluation utilities for the Naive Bayes model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(model, X_test, y_test, figure_path="reports/figures/confusion_matrix.png"):
    """Evaluate a fitted classifier and save a confusion matrix."""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
    }

    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall: {recall:.4f}")
    print(f"Weighted F1: {f1:.4f}")

    cm = confusion_matrix(y_test, predictions, labels=model.classes_)

    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )
    plt.title("Gaussian Naive Bayes Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()

    return metrics, predictions


def error_analysis(X_test, y_test, predictions):
    """Return incorrectly classified observations."""
    result = X_test.copy()
    result["actual"] = y_test.values
    result["predicted"] = predictions
    return result[result["actual"] != result["predicted"]].copy()
