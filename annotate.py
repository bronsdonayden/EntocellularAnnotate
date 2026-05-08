"""
annotate.py — Manual cell annotation tool using Napari
Run locally: python annotate.py
             python annotate.py --start 50   ← jump to patch 50

Controls:
  - Select the Labels layer in the layer list (left panel)
  - Pick a label number in the toolbar (increment for each new cell)
  - Paint with left click/drag, erase with right click/drag
  - Brush size adjustable in toolbar
  - When done with a patch, close the Napari window → auto-saves → loads next patch

Clump rule: if you can see ANY boundary between cells, annotate separately.
            If it's truly one indistinguishable blob, annotate as one.
"""

import napari
import numpy as np
import glob
import os
import argparse

# ── CONFIG ───────────────────────────────────────────────────────────────────
ANNOTATE_DIR = './content/annotate'
MASK_OUT_DIR = './content/annotate_masks'
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0,
                    help='Patch index to start from (e.g. --start 50)')
args = parser.parse_args()

os.makedirs(MASK_OUT_DIR, exist_ok=True)

img_files = sorted(glob.glob(os.path.join(ANNOTATE_DIR, 'img_*.npy')))
assert len(img_files) > 0, f"No patches found in {ANNOTATE_DIR}"

total = len(img_files)

# Apply --start offset
if args.start > 0:
    if args.start >= total:
        print(f"--start {args.start} is out of range (only {total} patches). Exiting.")
        exit()
    img_files = img_files[args.start:]
    print(f"Starting from patch index {args.start}")

remaining = [
    f for f in img_files
    if not os.path.exists(
        os.path.join(MASK_OUT_DIR, os.path.basename(f).replace('img_', 'mask_'))
    )
]

done = total - len(remaining) - args.start

print(f"Total patches:     {total}")
print(f"Already annotated: {done}")
print(f"Remaining:         {len(remaining)}")
print(f"Masks saving to:   {MASK_OUT_DIR}\n")

if len(remaining) == 0:
    print("All patches already annotated. Nothing to do.")
    exit()

for i, img_path in enumerate(remaining):
    patch_name = os.path.basename(img_path)
    mask_name  = patch_name.replace('img_', 'mask_')
    mask_path  = os.path.join(MASK_OUT_DIR, mask_name)

    img        = np.load(img_path)
    global_idx = int(patch_name.replace('img_', '').replace('.npy', ''))

    print(f"[{global_idx + 1}/{total}] Annotating: {patch_name}")
    print("  Close window when done with this patch")

    viewer = napari.Viewer(title=f"[{global_idx + 1}/{total}] {patch_name}")
    viewer.add_image(img, colormap='gray', name='image')

    labels_layer = viewer.add_labels(
        np.zeros(img.shape, dtype=np.int32),
        name='cells'
    )

    napari.run()

    mask    = labels_layer.data.astype(np.int32)
    n_cells = len(np.unique(mask)) - 1
    np.save(mask_path, mask)
    print(f"  Saved: {mask_name} ({n_cells} cells annotated)\n")

print("Done for this session!")
print(f"Zip {MASK_OUT_DIR} and upload to the fine-tuning notebook when ready.")