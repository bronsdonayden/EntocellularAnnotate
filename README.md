# EntocellularAnnotate

Cell annotation toolkit for [Cellpose](https://github.com/MouseLand/cellpose) fine-tuning on phase contrast microscopy images. Built for the UMass Boston IMPACT program.

Includes a Napari-based annotation tool with **model-assisted pre-labeling** — pre-computed segmentation masks are loaded as starting points so you correct predictions instead of painting from scratch.

## Repository Structure

```
├── annotate.py                 # Napari annotation tool
├── generate_patches.ipynb      # Chunk TIFFs into patches (Colab)
├── finetune.ipynb              # Fine-tune Cellpose + evaluate (Colab)
├── generate_predictions.ipynb  # Generate pred_masks for annotation (Colab)
├── img0/
│   ├── patches/                # 196 image patches (256×256, 50% overlap)
│   ├── masks/                  # Hand-annotated cell masks
│   └── pred_masks/             # Model predictions for assisted annotation
├── img1/
│   ├── patches/                # 196 image patches
│   ├── masks/                  # Hand-annotated cell masks
│   └── pred_masks/             # Model predictions for assisted annotation
└── models/
    └── finetuned_best          # Fine-tuned Cellpose cyto3 model (AP@0.5: 0.624)
```

## Setup

```bash
pip install -r requirements.txt
```

## Annotating

### From scratch

```bash
python annotate.py -i img0/patches -o img0/masks
```

### Model-assisted (recommended)

```bash
python annotate.py -i img0/patches -o img0/masks -p img0/pred_masks
python annotate.py -i img1/patches -o img1/masks -p img1/pred_masks
```

### Resume

Already-annotated patches are automatically skipped. Jump to a specific index:

```bash
python annotate.py -i img0/patches -o img0/masks -p img0/pred_masks --start 50
```

### Options

| Flag | Description |
|------|-------------|
| `-i`, `--input` | Directory with `img_*.npy` patches (required) |
| `-o`, `--output` | Directory to save `mask_*.npy` annotations (required) |
| `-p`, `--preds` | Directory with `pred_*.npy` predictions (optional) |
| `--start` | Patch index to start from (default: 0) |

## Napari Controls

| Action | Control |
|--------|---------|
| Paint cell | Left click / drag |
| Erase | Right click / drag |
| New cell label | `M` (auto-increments) |
| Pick existing label | `L` then click a cell |
| Toggle mask | Eye icon or `V` |
| Brush size | Toolbar slider |
| Save + next | Close the window |

## Annotation Guidelines

- Each cell gets a unique integer label. Background is 0.
- If you can see **any boundary** between cells, annotate separately.
- If cells form one indistinguishable blob, annotate as one.

## Model

The included model is Cellpose cyto3 fine-tuned on 51 hand-annotated phase contrast patches.

| | AP@0.5 |
|---|---|
| Base cyto3 | 0.523 |
| Fine-tuned | 0.624 |
| Base cpsam | 0.568 |

Training config: 500 epochs, lr=0.001, weight_decay=1e-5, diameter=17px.
