# EntocellularAnnotate

Cell annotation toolkit for building training data for [Cellpose](https://github.com/MouseLand/cellpose) fine-tuning on phase contrast microscopy images.

Built for the UMass Boston IMPACT program's cell segmentation pipeline. Includes a Napari-based annotation tool with **model-assisted pre-labeling** — pre-computed segmentation masks are loaded as starting points so annotators correct predictions instead of painting from scratch.

## Repository Structure

```
├── annotate.py              # Napari annotation tool
├── img0/
│   ├── patches/             # 196 image patches (256×256, 50% overlap)
│   ├── masks/               # Hand-annotated cell masks
│   └── pred_masks/          # Model-predicted masks (for assisted annotation)
├── img1/
│   ├── patches/             # 196 image patches
│   ├── masks/               # Hand-annotated cell masks
│   └── pred_masks/          # Model-predicted masks
└── models/
    └── finetuned_best       # Fine-tuned Cellpose cyto3 model
```

Patches are 256×256 crops extracted from full phase contrast microscopy TIFFs with 50% overlap (stride 128). Each `.npy` file is a single patch.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

### Annotate from scratch

```bash
python annotate.py -i img0/patches -o img0/masks
```

### Model-assisted annotation (recommended)

Pre-computed predictions are included in `pred_masks/` directories. These are loaded as the starting labels — you correct mistakes instead of painting every cell by hand.

```bash
python annotate.py -i img0/patches -o img0/masks -p img0/pred_masks
python annotate.py -i img1/patches -o img1/masks -p img1/pred_masks
```

### Resume a session

Already-annotated patches are automatically skipped. To jump to a specific index:

```bash
python annotate.py -i img0/patches -o img0/masks -p img0/pred_masks --start 50
```

### All options

| Flag | Description |
|------|-------------|
| `-i`, `--input` | Directory containing `img_*.npy` patches (required) |
| `-o`, `--output` | Directory to save `mask_*.npy` annotations (required) |
| `-p`, `--preds` | Directory containing `pred_*.npy` predictions (optional) |
| `--start` | Patch index to start from (default: 0) |

## Napari Controls

| Action | Control |
|--------|---------|
| Paint cell | Left click / drag |
| Erase | Right click / drag |
| New cell label | Press `M` (auto-increments to next unused label) |
| Pick existing label | Press `L`, then click a cell |
| Toggle mask visibility | Click eye icon or press `V` |
| Adjust brush size | Toolbar slider |
| Save and next patch | Close the Napari window |

## Annotation Guidelines

- Each cell gets a unique integer label (1, 2, 3, ...). Background is 0.
- If you can see **any boundary** between cells, annotate them separately.
- If cells form one indistinguishable blob, annotate as one.

## Model Details

The included model (`models/finetuned_best`) is a Cellpose cyto3 model fine-tuned on hand-annotated phase contrast microscopy patches.

- **Base model:** Cellpose cyto3
- **Training config:** 500 epochs, lr=0.001, weight_decay=1e-5
- **Cell diameter:** 17 px
- **Base AP@0.5:** 0.523 → **Fine-tuned AP@0.5: 0.624**

## Pipeline Overview

This tool is step 2 in the fine-tuning pipeline:

1. **Generate patches** — Chunk full microscopy TIFFs into 256×256 patches (`Generate_patches.ipynb`)
2. **Annotate** — Label cells in Napari using this tool (`annotate.py`)
3. **Fine-tune** — Train Cellpose on hand-labeled patches, evaluate with AP@0.5 (`step3_finetune.ipynb`)
4. **Iterate** — Generate new predictions with the improved model, annotate more patches, retrain
