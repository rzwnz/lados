"""Image transforms for training and validation."""

from typing import Any, Dict

import torch
from torchvision import transforms


def get_train_transforms(
    img_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get training transforms with data augmentation.

    Args:
        img_size: Target image size (square)
        mean: ImageNet mean for normalization
        std: ImageNet std for normalization

    Returns:
        Composed transforms
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_val_transforms(
    img_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        img_size: Target image size (square)
        mean: ImageNet mean for normalization
        std: ImageNet std for normalization

    Returns:
        Composed transforms
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_transforms_config() -> Dict[str, Any]:
    """Get default transforms configuration."""
    return {
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
