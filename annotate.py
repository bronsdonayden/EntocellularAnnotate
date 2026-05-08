"""
annotate.py — Cell annotation tool with optional model-assisted pre-labeling

Loads image patches one at a time in Napari for manual cell labeling.
Supports pre-computed segmentation masks as starting points so you
correct predictions instead of painting from scratch.

Masks are saved incrementally — close the Napari window to save and
advance to the next patch. Already-annotated patches are automatically
skipped, so you can stop and resume at any time.
"""

import napari
import numpy as np
import glob
import os
import sys
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manual cell annotation tool with optional model-assisted pre-labeling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s -i patches/ -o masks/
  %(prog)s -i patches/ -o masks/ -p pred_masks/
  %(prog)s -i patches/ -o masks/ --start 50
  %(prog)s -i patches/ -o masks/ -p pred_masks/ --start 50

controls (napari):
  - select the 'cells' labels layer in the left panel
  - pick a label number (increment for each new cell)
  - paint: left click/drag    erase: right click/drag
  - toggle mask visibility: click the eye icon or press 'v'
  - adjust brush size in the toolbar
  - close the window when done → auto-saves → loads next patch
        """,
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Directory containing img_*.npy image patches",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Directory to save mask_*.npy annotation masks",
    )
    parser.add_argument(
        "-p", "--preds", type=str, default=None,
        help="Directory containing pred_*.npy pre-computed masks "
             "(optional — enables model-assisted annotation)",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Patch index to start from, e.g. --start 50 (default: 0)",
    )
    return parser.parse_args()


def discover_patches(input_dir, output_dir, start):
    """Find all patches, filter out already-annotated ones, apply --start offset."""
    img_files = sorted(glob.glob(os.path.join(input_dir, "img_*.npy")))
    if len(img_files) == 0:
        print(f"Error: no img_*.npy files found in {input_dir}")
        sys.exit(1)

    total = len(img_files)

    if start > 0:
        if start >= total:
            print(f"--start {start} is out of range (only {total} patches).")
            sys.exit(1)
        img_files = img_files[start:]

    remaining = [
        f for f in img_files
        if not os.path.exists(
            os.path.join(output_dir, os.path.basename(f).replace("img_", "mask_"))
        )
    ]

    return img_files, remaining, total


def load_prediction(preds_dir, patch_name, img_shape):
    """Load a pre-computed prediction if available, otherwise return zeros."""
    if preds_dir is None:
        return np.zeros(img_shape[:2], dtype=np.int32), None

    pred_name = patch_name.replace("img_", "pred_")
    pred_path = os.path.join(preds_dir, pred_name)

    if os.path.exists(pred_path):
        labels = np.load(pred_path).astype(np.int32)
        n_pred = len(np.unique(labels)) - 1
        return labels, n_pred

    return np.zeros(img_shape[:2], dtype=np.int32), None


def annotate_patch(img, init_labels, title):
    """Open Napari with the image and initial labels, return final labels on close."""
    viewer = napari.Viewer(title=title)
    viewer.add_image(img, colormap="gray", name="image")
    labels_layer = viewer.add_labels(init_labels, name="cells")
    napari.run()
    return labels_layer.data.astype(np.int32)


def main():
    args = parse_args()

    input_dir  = args.input
    output_dir = args.output
    preds_dir  = args.preds

    os.makedirs(output_dir, exist_ok=True)

    # Validate predictions directory
    if preds_dir is not None:
        pred_count = len(glob.glob(os.path.join(preds_dir, "pred_*.npy")))
        if pred_count == 0:
            print(f"Error: no pred_*.npy files found in {preds_dir}")
            sys.exit(1)
        print(f"Model-assisted mode: {pred_count} predictions loaded from {preds_dir}")

    # Discover work
    _, remaining, total = discover_patches(input_dir, output_dir, args.start)
    done_before = total - len(remaining) - args.start

    print(f"Total patches:     {total}")
    print(f"Already annotated: {done_before}")
    print(f"Remaining:         {len(remaining)}")
    print(f"Saving to:         {output_dir}")
    print(f"Mode:              {'MODEL-ASSISTED' if preds_dir else 'BLANK CANVAS'}")
    print()

    if len(remaining) == 0:
        print("All patches already annotated. Nothing to do.")
        return

    session_count = 0
    session_start = time.time()

    for i, img_path in enumerate(remaining):
        patch_name = os.path.basename(img_path)
        mask_name  = patch_name.replace("img_", "mask_")
        mask_path  = os.path.join(output_dir, mask_name)
        global_idx = int(patch_name.replace("img_", "").replace(".npy", ""))

        img = np.load(img_path)
        init_labels, n_pred = load_prediction(preds_dir, patch_name, img.shape)

        # Status line
        progress = f"[{global_idx + 1}/{total}]"
        if n_pred is not None:
            print(f"{progress} {patch_name} — {n_pred} predicted cells, correct as needed")
        else:
            print(f"{progress} {patch_name}")
        print("  Close window when done")

        # Annotate
        mask = annotate_patch(
            img, init_labels,
            title=f"{progress} {patch_name}",
        )

        n_cells = len(np.unique(mask)) - 1
        np.save(mask_path, mask)
        session_count += 1

        elapsed = time.time() - session_start
        rate = elapsed / session_count
        left = len(remaining) - (i + 1)

        print(f"  Saved: {mask_name} ({n_cells} cells)")
        print(f"  Session: {session_count} done, {left} left, ~{rate:.0f}s/patch\n")

    elapsed = time.time() - session_start
    minutes = elapsed / 60

    print("=" * 50)
    print(f"Session complete!")
    print(f"  Annotated: {session_count} patches in {minutes:.1f} min")
    print(f"  Avg pace:  {elapsed / max(session_count, 1):.0f}s per patch")
    print(f"  Masks in:  {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
