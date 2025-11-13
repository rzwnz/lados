"""PyTorch dataset for LADOS classification."""

import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class LadosImageDataset(Dataset):
    """
    LADOS image classification dataset.

    Supports ImageFolder-style directory structure:
        data/
            train/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    ...
            val/
                ...
            test/
                ...
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        transform: Optional[Callable] = None,
        manifest_path: Optional[Path | str] = None,
    ):
        """
        Initialize dataset.

        Args:
            root: Root directory containing train/val/test splits
            split: Dataset split ('train', 'val', 'test')
            transform: Optional image transforms
            manifest_path: Optional path to CSV manifest (if None, scans directories)
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if manifest_path and Path(manifest_path).exists():
            # Load from manifest
            df = pd.read_csv(manifest_path)
            df = df[df["split"] == split]
            self.samples = [
                (self.root / row["image_path"], row["class"]) for _, row in df.iterrows()
            ]
        else:
            # Scan directory structure
            split_dir = self.root / split
            if not split_dir.exists():
                raise ValueError(f"Split directory not found: {split_dir}")

            self.samples = []
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                for img_path in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
                    self.samples.append((img_path, class_name))

        # Create class to index mapping
        self.classes = sorted(set(label for _, label in self.samples))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        print(f"Loaded {len(self.samples)} images from {split} split")
        print(f"Classes: {self.classes}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get item by index.

        Returns:
            (image_tensor, class_idx)
        """
        img_path, class_name = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get class index
        class_idx = self.class_to_idx[class_name]

        return image, class_idx

    def get_class_weights(self) -> dict[str, float]:
        """Compute class weights for balanced training."""
        from collections import Counter

        class_counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        weights = {cls: total / (len(self.classes) * count) for cls, count in class_counts.items()}
        return weights
