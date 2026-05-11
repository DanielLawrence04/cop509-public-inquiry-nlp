"""
Evaluation: compare system predictions against ground-truth labels.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TypedDict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .classification import Label, normalize_label

LABELS: list[Label] = ["accepted", "partially_accepted", "rejected", "not_addressed"]


class EvaluationResult(TypedDict):
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    report: str


def compare_to_ground_truth(
    predictions: list[Label],
    ground_truth: list[Label],
) -> EvaluationResult:
    """
    Compute classification metrics comparing *predictions* to *ground_truth*.

    Parameters
    ----------
    predictions : list[Label]
        System-predicted labels.
    ground_truth : list[Label]
        Human-annotated reference labels.

    Returns
    -------
    EvaluationResult
        Dictionary containing scalar metrics, per-class breakdowns,
        a confusion matrix, and a formatted classification report.

    Raises
    ------
    ValueError
        If *predictions* and *ground_truth* have different lengths or are empty.
    """
    predictions = [normalize_label(label) for label in predictions]
    ground_truth = [normalize_label(label) for label in ground_truth]

    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truth)} ground truth labels."
        )
    if not predictions:
        raise ValueError("Prediction and ground truth lists must not be empty.")

    report = classification_report(
        ground_truth,
        predictions,
        labels=LABELS,
        zero_division=0,
    )
    cm = confusion_matrix(ground_truth, predictions, labels=LABELS).tolist()

    per_class: dict[str, dict[str, float]] = {}
    for label in LABELS:
        y_true_bin = [1 if g == label else 0 for g in ground_truth]
        y_pred_bin = [1 if p == label else 0 for p in predictions]
        per_class[label] = {
            "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
            "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
            "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
            "support": int(sum(y_true_bin)),
        }

    return EvaluationResult(
        accuracy=float(accuracy_score(ground_truth, predictions)),
        precision_macro=float(
            precision_score(ground_truth, predictions, labels=LABELS, average="macro", zero_division=0)
        ),
        recall_macro=float(
            recall_score(ground_truth, predictions, labels=LABELS, average="macro", zero_division=0)
        ),
        f1_macro=float(
            f1_score(ground_truth, predictions, labels=LABELS, average="macro", zero_division=0)
        ),
        per_class=per_class,
        confusion_matrix=cm,
        report=report,
    )


def results_to_dataframe(result: EvaluationResult):
    """
    Convert per-class metrics to a ``pandas.DataFrame`` for display.

    Parameters
    ----------
    result : EvaluationResult
        Output of :func:`compare_to_ground_truth`.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    rows = []
    for label, metrics in result["per_class"].items():
        rows.append({"label": label, **metrics})
    return pd.DataFrame(rows).set_index("label")
