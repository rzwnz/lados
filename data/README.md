# LADOS Dataset

## Dataset Information

LADOS (Large-scale Aerial Dataset for Oil Spill detection) contains **3,388** pixel-level annotated aerial images.

## Classes

The dataset is preprocessed into **6 classification classes** (≥3 classes as required):

1. **oil** - Pure oil spills
2. **emulsion** - Oil emulsions (different from pure oil)
3. **oil_platform** - Oil platforms
4. **sheen** - Oil sheen (thin film on water surface)
5. **ship** - Ships/vessels
6. **background** - No annotations / background

These classes are derived from the COCO annotation categories:
- `oil` (ID: 2)
- `emulsion` (ID: 1)
- `oil-platform` (ID: 3)
- `sheen` (ID: 4)
- `ship` (ID: 5)
- `oils-emulsions` (ID: 0, parent category, mapped to `oil`)

## Dataset Sources

- **Roboflow**: https://universe.roboflow.com/konstantinos-gkountakos/lados
- **Zenodo**: [LADOS Dataset Release]
- **MDPI**: [Publication Link]

## Citation

If you use this dataset, please cite:

```
[LADOS Dataset Citation - Add actual citation from paper]
```

## Data Format

The dataset is provided in COCO format with segmentation masks. For classification experiments, we convert pixel-level masks to single class labels using the following heuristic:

### Classification Label Conversion

1. **Primary rule**: Choose the non-background label with the largest pixel area
2. **Fallback**: Assign `background` if background pixels > 80% of total pixels
3. **Category mapping**: COCO categories are mapped to our 6-class structure:
   - `oil` → `oil`
   - `emulsion` → `emulsion`
   - `oil-platform` → `oil_platform`
   - `sheen` → `sheen`
   - `ship` → `ship`
   - `oils-emulsions` → `oil` (parent category)

This conversion is implemented in `scripts/download_lados.py` and documented in the preprocessing pipeline.

## Directory Structure

After processing, the dataset is organized as:

```
data/processed/
├── train/
│   ├── oil/
│   ├── emulsion/
│   ├── oil_platform/
│   ├── sheen/
│   ├── ship/
│   └── background/
├── val/
│   └── [same structure]
└── test/
    └── [same structure]
```

## Manifests

CSV manifests are created in `manifests/`:
- `train.csv`: Training set manifest
- `val.csv`: Validation set manifest
- `test.csv`: Test set manifest

Each manifest contains:
- `image_path`: Relative path to image
- `class`: Class label
- `split`: Dataset split

## Class Distribution

After preprocessing, the class distribution should show:
- Multiple classes with non-zero samples (≥3 classes)
- Balanced or imbalanced distribution depending on the dataset

To check class distribution:
```bash
for class_dir in data/processed/train/*/; do 
    echo "$(basename $class_dir): $(find $class_dir -type f | wc -l) files"
done
```

