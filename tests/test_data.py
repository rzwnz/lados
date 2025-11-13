"""Tests for data loading."""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data import LadosImageDataset, get_train_transforms, get_val_transforms


@pytest.fixture
def dummy_dataset(tmp_path):
    """Create a dummy dataset for testing."""
    # Create directory structure
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    classes = ["class1", "class2"]
    for split_dir in [train_dir, val_dir]:
        for cls in classes:
            cls_dir = split_dir / cls
            cls_dir.mkdir(parents=True)
            # Create dummy images
            for i in range(5):
                img = Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50))
                img.save(cls_dir / f"img_{i}.jpg")

    return tmp_path


def test_dataset_loading(dummy_dataset):
    """Test dataset loading."""
    dataset = LadosImageDataset(root=dummy_dataset, split="train")
    assert len(dataset) == 10  # 5 images per class * 2 classes
    assert len(dataset.classes) == 2


def test_dataset_getitem(dummy_dataset):
    """Test dataset __getitem__."""
    dataset = LadosImageDataset(root=dummy_dataset, split="train", transform=get_val_transforms())
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < len(dataset.classes)


def test_transforms():
    """Test image transforms."""
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    train_tensor = train_transform(img)
    val_tensor = val_transform(img)

    assert train_tensor.shape == (3, 224, 224)
    assert val_tensor.shape == (3, 224, 224)
