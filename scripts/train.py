#!/usr/bin/env python3
"""
Training script for LADOS classification.

Usage:
    python scripts/train.py --config configs/train_resnet50.yaml
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data import LadosImageDataset, get_train_transforms, get_val_transforms
from src.eval import compute_metrics, plot_confusion_matrix, plot_training_curves, save_metrics_json
from src.models import create_model


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_macro_f1",
        mode: str = "max",
        min_delta: float = 0.001,
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "max":
            if score < self.best_score + self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == "min"
            if score > self.best_score - self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


def find_optimal_batch_size(
    model: nn.Module,
    device: torch.device,
    dataset: LadosImageDataset,
    max_batch_size: int = 32,
    min_batch_size: int = 4,
) -> int:
    """Find optimal batch size that fits in GPU memory."""
    print("Tuning batch size...")
    model.eval()

    for batch_size in [max_batch_size, 16, 8, min_batch_size]:
        try:
            # Create a small dummy batch
            dummy_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )
            sample_batch, _ = next(iter(dummy_loader))
            sample_batch = sample_batch.to(device)

            # Try forward pass
            with torch.no_grad():
                _ = model(sample_batch)

            print(f"✓ Batch size {batch_size} fits in memory")
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ Batch size {batch_size} too large, trying smaller...")
                torch.cuda.empty_cache()
                continue
            else:
                raise

    print("Warning: Even minimum batch size failed. Falling back to CPU.")
    return min_batch_size


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    use_amp: bool,
    accum_steps: int,
) -> Dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader)

    return metrics


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> Dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, class_names=class_names)
    metrics["loss"] = total_loss / len(loader)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train LADOS classification model")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config file"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run one batch only")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup device
    use_gpu = config["device"]["use_gpu"] and torch.cuda.is_available()
    device = torch.device(f"cuda:{config['device']['cuda_device']}" if use_gpu else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["output"]["run_dir"]) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load datasets
    train_dataset = LadosImageDataset(
        root=config["data"]["root"],
        split="train",
        transform=get_train_transforms(img_size=config["data"]["img_size"]),
        manifest_path=config["data"].get("train_manifest"),
    )

    val_dataset = LadosImageDataset(
        root=config["data"]["root"],
        split="val",
        transform=get_val_transforms(img_size=config["data"]["img_size"]),
        manifest_path=config["data"].get("val_manifest"),
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Dataset classes: {class_names}")
    print(f"Number of classes: {num_classes}")

    # Small sample mode
    if config.get("small_sample", {}).get("enabled", False):
        max_samples = config["small_sample"]["max_samples"]
        train_dataset.samples = train_dataset.samples[:max_samples]
        val_dataset.samples = val_dataset.samples[:max_samples // 4]

    # Create model - use actual number of classes from dataset
    model = create_model(
        backbone=config["model"]["backbone"],
        num_classes=num_classes,  # Use actual number of classes from dataset
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"].get("freeze_backbone", False),
    ).to(device)

    # Batch size tuning
    batch_size = config["training"]["batch_size"]
    if config["training"]["batch_size_tuner"]["enabled"] and use_gpu:
        batch_size = find_optimal_batch_size(
            model,
            device,
            train_dataset,
            max_batch_size=config["training"]["batch_size_tuner"]["max_batch_size"],
            min_batch_size=config["training"]["batch_size_tuner"]["min_batch_size"],
        )
        print(f"Using batch size: {batch_size}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=use_gpu,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=use_gpu,
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Ensure lr and weight_decay are floats (YAML might parse scientific notation as string)
    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"]["weight_decay"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler
    total_steps = len(train_loader) * config["training"]["epochs"]
    warmup_steps = config["training"].get("warmup_steps")
    if warmup_steps is None:
        warmup_epochs = config["training"].get("warmup_epochs", 5)
        warmup_steps = len(train_loader) * warmup_epochs

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
    )

    # Mixed precision
    use_amp = config["training"]["mixed_precision"] and use_gpu
    scaler = GradScaler() if use_amp else None

    # Early stopping
    early_stopping = None
    if config["training"]["early_stopping"]["enabled"]:
        early_stopping = EarlyStopping(
            patience=config["training"]["early_stopping"]["patience"],
            metric=config["training"]["early_stopping"]["metric"],
            mode=config["training"]["early_stopping"]["mode"],
            min_delta=config["training"]["early_stopping"]["min_delta"],
        )

    # Training loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    best_val_f1 = 0.0
    best_epoch = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(1, config["training"]["epochs"] + 1):
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            config["training"]["gradient_accumulation_steps"],
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, class_names)

        # Update scheduler
        scheduler.step()

        # Log metrics
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        print(
            f"Epoch {epoch}/{config['training']['epochs']}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_f1,
                "class_names": class_names,
            }
            torch.save(checkpoint, run_dir / "checkpoint.pt")
            print(f"✓ Saved best model (val_macro_f1={best_val_f1:.4f})")

        # Early stopping
        if early_stopping:
            if early_stopping(val_metrics["macro_f1"]):
                print(f"Early stopping at epoch {epoch}")
                break

        # Dry run
        if args.dry_run:
            break

    # Final evaluation
    print("\nFinal evaluation on validation set...")
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt")["model_state_dict"])
    final_metrics = validate(model, val_loader, criterion, device, class_names)

    # Save metrics
    all_metrics = {
        "history": history,
        "final_metrics": final_metrics,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "training_time_seconds": time.time() - start_time,
    }
    save_metrics_json(all_metrics, run_dir / "metrics.json")

    # Plot curves
    plot_training_curves(history, run_dir / "training_curves.png")

    # Plot confusion matrix
    val_preds = []
    val_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    plot_confusion_matrix(
        val_labels, val_preds, class_names, run_dir / "confusion_matrix.png"
    )

    print(f"\nTraining complete! Results saved to: {run_dir}")
    print(f"Best val_macro_f1: {best_val_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()

