"""Plotting utilities for metrics visualization."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set_style("whitegrid")


def plot_confusion_matrix(
    y_true: np.ndarray | List[int],
    y_pred: np.ndarray | List[int],
    class_names: List[str],
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        output_path: Optional path to save figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized)")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves(
    history: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 5),
) -> None:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
        output_path: Optional path to save figure
        figsize: Figure size
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    axes[0].plot(epochs, history.get("train_loss", []), "b-", label="Train Loss")
    axes[0].plot(epochs, history.get("val_loss", []), "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy/F1 plot
    if "val_macro_f1" in history:
        axes[1].plot(epochs, history.get("val_macro_f1", []), "g-", label="Val Macro F1")
    if "train_acc" in history:
        axes[1].plot(epochs, history.get("train_acc", []), "b-", label="Train Acc")
    if "val_acc" in history:
        axes[1].plot(epochs, history.get("val_acc", []), "r-", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Training and Validation Metrics")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to: {output_path}")
    else:
        plt.show()

    plt.close()


def save_metrics_json(metrics: Dict, output_path: Path) -> None:
    """Save metrics dictionary to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {output_path}")

