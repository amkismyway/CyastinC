"""Evaluation helpers for targeted cystatin C testing."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def classification_metrics(y_true, probability, threshold=0.5):
    prediction = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "AUROC": roc_auc_score(y_true, probability),
        "AUPRC": average_precision_score(y_true, probability),
        "Brier": brier_score_loss(y_true, probability),
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "PPV": tp / (tp + fp) if tp + fp else np.nan,
        "NPV": tn / (tn + fn) if tn + fn else np.nan,
    }


def testing_budget(y_true, probability, budgets=(0.10, 0.20, 0.30, 0.50)):
    y = np.asarray(y_true)
    p = np.asarray(probability)
    order = np.argsort(-p)
    total_cases = y.sum()
    rows = []
    for fraction in budgets:
        n_test = max(1, int(np.ceil(len(y) * fraction)))
        found = y[order[:n_test]].sum()
        rows.append({
            "testing_budget": fraction,
            "people_tested": n_test,
            "cases_detected": int(found),
            "case_detection": found / total_cases if total_cases else np.nan,
            "positive_yield": found / n_test if n_test else np.nan,
            "number_needed_to_test": n_test / found if found else np.nan,
        })
    return pd.DataFrame(rows)
