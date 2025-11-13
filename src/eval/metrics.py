"""Metrics computation for classification."""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray | List[int],
    y_pred: np.ndarray | List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names for per-class metrics

    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
    }

    if class_names:
        metrics["per_class"]["class_names"] = class_names
        # Add per-class dict
        per_class_dict = {}
        for i, name in enumerate(class_names):
            per_class_dict[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        metrics["per_class_dict"] = per_class_dict

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray | List[int],
    y_pred: np.ndarray | List[int],
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names

    Returns:
        Confusion matrix array
    """
    return confusion_matrix(y_true, y_pred)


def compute_metrics_from_tensor(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute metrics from PyTorch tensors."""
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.argmax(dim=1).cpu().numpy() if y_pred.dim() > 1 else y_pred.cpu().numpy()
    return compute_metrics(y_true_np, y_pred_np, class_names)
