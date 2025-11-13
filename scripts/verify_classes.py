#!/usr/bin/env python3
"""Verify that processed dataset has more than 3 classes with data."""

from pathlib import Path
from collections import defaultdict


def verify_classes(data_root: Path = Path("data/processed"), min_classes: int = 3):
    """Verify dataset has at least min_classes with samples."""
    class_counts = defaultdict(int)

    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            file_count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
            class_counts[class_name] += file_count

    print("Class distribution across all splits:")
    print("-" * 40)
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_name:20s}: {count:6d} files")

    classes_with_data = [name for name, count in class_counts.items() if count > 0]
    num_classes = len(classes_with_data)

    print("-" * 40)
    print(f"Total classes with data: {num_classes}")
    print(f"Classes: {', '.join(sorted(classes_with_data))}")

    if num_classes >= min_classes:
        print(f"✓ Dataset has {num_classes} classes (≥ {min_classes} required)")
        return True
    else:
        print(f"✗ Dataset has only {num_classes} classes (need ≥ {min_classes})")
        return False


if __name__ == "__main__":
    import sys

    min_classes = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    success = verify_classes(min_classes=min_classes)
    sys.exit(0 if success else 1)
