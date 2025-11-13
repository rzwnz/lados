#!/usr/bin/env python3
"""
Download and prepare LADOS dataset for classification.

Supports:
- Roboflow API download
- Zenodo direct download (fallback)
- Small-sample mode for quick testing (~200 images)
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from tqdm import tqdm

try:
    from roboflow import Roboflow
except ImportError:
    Roboflow = None


def compute_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_from_roboflow(
    api_key: Optional[str],
    workspace: str,
    project: str,
    version: Union[int, None],
    output_dir: Path,
    small_sample: bool = False,
) -> Path:
    """Download LADOS dataset from Roboflow."""
    if Roboflow is None:
        raise ImportError("roboflow package required. Install with: pip install roboflow")

    if api_key is None:
        print("Warning: No Roboflow API key provided. Trying public access...")
        # For public datasets, we can try without API key or use a placeholder
        api_key = os.getenv("ROBOFLOW_API_KEY", "")

    rf = Roboflow(api_key=api_key if api_key else None)
    try:
        project_obj = rf.workspace(workspace).project(project)

        # If version is None, try to find available versions by testing common numbers
        if version is None:
            print("Version not specified, attempting to auto-detect available version...")
            versions_to_try = [2, 3, 4, 5, 1]  # Try higher versions first, then fallback to 1
            version = None

            for v in versions_to_try:
                try:
                    print(f"Testing version {v}...")
                    # Try to access the version (don't download yet)
                    test_version = project_obj.version(v)
                    # If we get here, the version exists
                    version = v
                    print(f"Found available version: {version}")
                    break
                except Exception:
                    continue

            if version is None:
                print("Could not auto-detect version, will try version 1...")
                version = 1

        print(f"Downloading from workspace: {workspace}, project: {project}, version: {version}")

        # Try to download the specified version
        try:
            dataset = project_obj.version(version).download(
                "coco-segmentation", location=str(output_dir)
            )
        except Exception as ve:
            # If version doesn't exist, try to find available versions
            error_msg = str(ve).lower()
            if "not found" in error_msg or "version" in error_msg or "does not exist" in error_msg:
                print(f"Version {version} not found. Attempting to find available versions...")
                versions_to_try = [2, 3, 4, 5, 1] if version == 1 else [1, 2, 3, 4, 5]
                dataset = None
                found_version = None

                for v in versions_to_try:
                    if v == version:
                        continue  # Skip the one we already tried
                    try:
                        print(f"Trying version {v}...")
                        dataset = project_obj.version(v).download("coco", location=str(output_dir))
                        found_version = v
                        print(f"✓ Successfully found and downloaded version {v}")
                        break
                    except Exception as e:
                        print(f"  Version {v} failed: {str(e)[:100]}")
                        continue

                if dataset is None:
                    print(
                        f"\nError: Could not find any valid version after trying: {versions_to_try}"
                    )
                    print("Please check the Roboflow project page to see available versions:")
                    print(f"https://universe.roboflow.com/{workspace}/{project}")
                    raise ve
                version = found_version
            else:
                raise

    except Exception as e:
        print(f"Error accessing Roboflow project: {e}")
        print(f"Workspace: {workspace}, Project: {project}, Attempted Version: {version}")
        print("Note: Some datasets may require an API key. Get one from: https://roboflow.com/")
        print("You can also try specifying a different version with --version")
        raise

    # Handle dataset path - Roboflow typically creates a directory like {project}-{version}
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset location - Roboflow might download to a different location than specified
    try:
        dataset_path = Path(dataset.location).resolve()
    except:
        # If dataset.location doesn't exist, try to find it
        dataset_path = None

    final_path = output_dir / "lados"

    print(f"Dataset location reported by Roboflow: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target location: {final_path}")

    # First, search for where Roboflow actually put the files
    # Roboflow often creates directories like: {project}-{version} or just downloads to current dir
    search_locations = [
        dataset_path if dataset_path and dataset_path.exists() else None,
        output_dir / f"{project}-{version}",
        output_dir / f"{project}-{version}-{version}",
        output_dir / project,
        output_dir,
        Path.cwd() / f"{project}-{version}",
        Path.cwd() / project,
    ]

    # Remove None values and duplicates
    search_locations = [loc for loc in search_locations if loc is not None]
    search_locations = list(
        dict.fromkeys(search_locations)
    )  # Remove duplicates while preserving order

    actual_dataset_path = None
    for search_loc in search_locations:
        if not search_loc.exists():
            continue
        # Check if this location contains train/val/test directories
        if (search_loc / "train").exists() or (search_loc / "train" / "images").exists():
            actual_dataset_path = search_loc
            print(f"✓ Found dataset at: {actual_dataset_path}")
            break
        # Also check for subdirectories
        for subdir in search_loc.iterdir():
            if subdir.is_dir() and (
                (subdir / "train").exists() or (subdir / "train" / "images").exists()
            ):
                actual_dataset_path = subdir
                print(f"✓ Found dataset at: {actual_dataset_path}")
                break
        if actual_dataset_path:
            break

    if not actual_dataset_path:
        print("Warning: Could not find dataset directory structure!")
        print("Searched in:")
        for loc in search_locations:
            print(f"  - {loc} (exists: {loc.exists() if loc else False})")
        # Try to list what's actually in output_dir
        if output_dir.exists():
            print(f"\nContents of {output_dir}:")
            try:
                for item in output_dir.iterdir():
                    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
            except:
                pass
        raise FileNotFoundError(
            "Could not locate downloaded dataset. Check Roboflow download location."
        )

    # Now move the dataset to final_path if needed
    if actual_dataset_path.resolve() == final_path.resolve():
        print(f"✓ Dataset already at target location: {final_path}")
    elif actual_dataset_path == output_dir:
        # Dataset is directly in output_dir - need to move via temp to avoid "move into itself"
        print("Dataset files are in output_dir, moving via temporary directory...")
        temp_dir = output_dir.parent / f"temp_lados_{project}_{version}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        # Move all dataset files/directories to temp (except lados if it exists)
        for item in output_dir.iterdir():
            if item.name != "lados":
                shutil.move(str(item), str(temp_dir / item.name))

        # Now move temp to final location
        if final_path.exists():
            shutil.rmtree(final_path)
        shutil.move(str(temp_dir), str(final_path))
        print(f"✓ Moved dataset to {final_path}")
    else:
        # Dataset is in a subdirectory or different location - move it
        if final_path.exists():
            shutil.rmtree(final_path)
        shutil.move(str(actual_dataset_path), str(final_path))
        print(f"✓ Moved dataset from {actual_dataset_path} to {final_path}")

    if small_sample:
        # Keep only first 200 images
        print("Small sample mode: keeping first 200 images...")
        images_dir = final_path / "train" / "images"
        if images_dir.exists():
            images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))[:200]
            # Keep corresponding annotations
            for img in images_dir.glob("*"):
                if img not in images:
                    img.unlink()

    return final_path


def download_from_zenodo(record_id: str, output_dir: Path, small_sample: bool = False) -> Path:
    """Download LADOS dataset from Zenodo (fallback method)."""
    # This is a placeholder - actual Zenodo API integration would go here
    # For now, we'll document the expected structure
    raise NotImplementedError(
        "Zenodo download not yet implemented. "
        "Please use Roboflow or download manually from: "
        "https://zenodo.org/record/[LADOS_RECORD_ID]"
    )


def convert_coco_to_classification(
    coco_path: Path, output_path: Path, small_sample: bool = False
) -> None:
    """
    Convert COCO format to ImageFolder structure for classification.

    For classification, we convert segmentation masks to dominant class labels.
    """
    from pycocotools.coco import COCO

    output_path.mkdir(parents=True, exist_ok=True)

    # LADOS classes - using actual COCO categories to preserve more classes
    # COCO has: oils-emulsions, emulsion, oil, oil-platform, sheen, ship
    # We'll create distinct classes from these to have >3 classes
    class_names = [
        "oil",  # Pure oil spills
        "emulsion",  # Oil emulsions (different from pure oil)
        "oil_platform",  # Oil platforms
        "sheen",  # Oil sheen (thin film)
        "ship",  # Ships
        "background",  # No annotations / background
    ]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Handle both "val" and "valid" naming conventions
    split_mapping = {"train": "train", "val": "val", "test": "test"}  # Try "val" first

    for split_name, split_dir_name in split_mapping.items():
        # Try both "val" and "valid" for validation split
        if split_name == "val":
            possible_dirs = ["val", "valid"]
        else:
            possible_dirs = [split_dir_name]

        ann_file = None
        images_dir = None
        found_split = None

        for dir_name in possible_dirs:
            test_ann_file = coco_path / f"{dir_name}/_annotations.coco.json"
            # Images might be in {dir_name}/images or directly in {dir_name}/
            test_images_dir1 = coco_path / f"{dir_name}/images"
            test_images_dir2 = coco_path / dir_name
            if test_ann_file.exists():
                # Check which images directory exists
                if test_images_dir1.exists():
                    images_dir = test_images_dir1
                elif test_images_dir2.exists():
                    images_dir = test_images_dir2
                else:
                    continue
                ann_file = test_ann_file
                found_split = dir_name
                break

        if not ann_file or not ann_file.exists():
            print(
                f"Warning: Annotation file for {split_name} not found (tried: {possible_dirs}), skipping {split_name}"
            )
            continue

        print(f"Processing {split_name} from {found_split} directory...")
        print(f"  Annotation file: {ann_file}")
        print(f"  Images directory: {images_dir}")

        coco = COCO(str(ann_file))
        # images_dir already set above
        output_split_dir = output_path / split_name  # Use split_name, not found_split

        # Verify images_dir has image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not image_files:
            print(f"Warning: No image files found in {images_dir}, skipping {split_name}")
            continue
        print(f"  Found {len(image_files)} image files")

        # Create class directories
        for class_name in class_names:
            (output_split_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Process images
        image_ids = coco.getImgIds()
        if small_sample and split_name == "train":
            image_ids = image_ids[:200]

        for img_id in tqdm(image_ids, desc=f"Processing {split_name}"):
            img_info = coco.loadImgs(img_id)[0]
            img_path = images_dir / img_info["file_name"]

            if not img_path.exists():
                continue

            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)

            # Determine dominant class (simplified - in practice, load mask and count pixels)
            # For now, use first non-background category or background
            dominant_class = "background"
            if annotations:
                # In real implementation, load mask and count pixels
                # For now, use category_id
                cats = [ann["category_id"] for ann in annotations]
                if cats:
                    cat_id = max(set(cats), key=cats.count)
                    cat_info = coco.loadCats(cat_id)[0]
                    cat_name = cat_info["name"].lower().replace(" ", "_").replace("-", "_")

                    # Map COCO category names to our class names
                    # Preserve distinct classes: oil, emulsion, oil-platform, sheen, ship
                    category_mapping = {
                        # Direct mappings
                        "oil": "oil",
                        "emulsion": "emulsion",
                        "oil_platform": "oil_platform",
                        "oil-platform": "oil_platform",
                        "sheen": "sheen",
                        "ship": "ship",
                        # oils-emulsions is a parent category, map to oil
                        "oils_emulsions": "oil",
                        "oils-emulsions": "oil",
                        # Legacy mappings (if present)
                        "oil_spill": "oil",
                        "algae_bloom": "background",  # Not in this dataset
                        "algae": "background",
                        "organic_matter": "background",
                        "organic": "background",
                        "land": "background",
                        "background": "background",
                    }

                    # Try direct match first, then mapping
                    if cat_name in class_to_idx:
                        dominant_class = cat_name
                    elif cat_name in category_mapping:
                        mapped_name = category_mapping[cat_name]
                        if mapped_name in class_to_idx:
                            dominant_class = mapped_name
                        else:
                            dominant_class = "background"
                    else:
                        # Unknown category, use background
                        print(f"  Warning: Unknown category '{cat_name}', using background")
                        dominant_class = "background"

            # Copy image to class directory
            output_img_path = output_split_dir / dominant_class / img_info["file_name"]
            try:
                shutil.copy2(img_path, output_img_path)
            except Exception as e:
                print(f"Error copying {img_path} to {output_img_path}: {e}")
                continue

    print(f"Classification dataset created at: {output_path}")


def create_manifests(data_root: Path, output_dir: Path) -> None:
    """Create train/val/test CSV manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        # Also check for "valid" if "val" doesn't exist
        if not split_dir.exists() and split == "val":
            split_dir = data_root / "valid"
        if not split_dir.exists():
            continue

        records = []
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in class_dir.glob("*.jpg"):
                records.append(
                    {
                        "image_path": str(img_path.relative_to(data_root)),
                        "class": class_name,
                        "split": split,
                    }
                )
            for img_path in class_dir.glob("*.png"):
                records.append(
                    {
                        "image_path": str(img_path.relative_to(data_root)),
                        "class": class_name,
                        "split": split,
                    }
                )

        df = pd.DataFrame(records)
        manifest_path = output_dir / f"{split}.csv"
        df.to_csv(manifest_path, index=False)
        print(f"Created manifest: {manifest_path} ({len(df)} images)")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare LADOS dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for raw dataset",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed classification dataset",
    )
    parser.add_argument(
        "--roboflow-key",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="konstantinos-gkountakos",
        help="Roboflow workspace name",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="lados",
        help="Roboflow project name",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Roboflow dataset version (default: auto-detect latest)",
    )
    parser.add_argument(
        "--small-sample",
        action="store_true",
        help="Download only ~200 images for quick testing",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip COCO to classification conversion",
    )

    args = parser.parse_args()

    # Get API key from env if not provided
    api_key = args.roboflow_key or os.getenv("ROBOFLOW_API_KEY")

    # Download dataset
    print("Downloading LADOS dataset...")
    try:
        dataset_path = download_from_roboflow(
            api_key=api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            output_dir=args.output_dir,
            small_sample=args.small_sample,
        )
        print(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Error downloading from Roboflow: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if you need an API key: https://roboflow.com/")
        print("2. Verify the workspace and project names are correct")
        print("3. Try specifying a different version: --version 2 (or 3, 4, etc.)")
        print(
            "4. Check the dataset URL: https://universe.roboflow.com/konstantinos-gkountakos/lados"
        )
        print("\nFalling back to manual download instructions...")
        print(
            "Please download LADOS manually from: "
            "https://universe.roboflow.com/konstantinos-gkountakos/lados"
        )
        print("Then extract it to:", args.output_dir)
        return

    # Convert to classification format
    if not args.skip_conversion:
        print("Converting to classification format...")
        convert_coco_to_classification(
            dataset_path, args.processed_dir, small_sample=args.small_sample
        )

        # Create manifests
        print("Creating manifests...")
        create_manifests(args.processed_dir, Path("manifests"))

    print("Done!")


if __name__ == "__main__":
    main()
