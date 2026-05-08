"""
cpsam_inf.py — Cellpose-SAM inference / prediction generator.

Usage:
    python cpsam_inf.py                                  # all patches
    python cpsam_inf.py --root img0                      # all patches in img0
    python cpsam_inf.py --root img0 --n 5 --view         # first 5, napari preview
    python cpsam_inf.py --root img0 --indices 0 7 42     # specific indices
    python cpsam_inf.py --model models/cpsam_finetuned   # use fine-tuned weights
    python cpsam_inf.py --out-name pred_masks            # refresh canonical seeds (overwrites)
"""

import argparse
import glob
import os
import re
from datetime import datetime
import numpy as np

from cellpose import models


def list_patches(patch_dir):
    paths = sorted(glob.glob(os.path.join(patch_dir, 'img_*.npy')))
    out = []
    for p in paths:
        m = re.match(r'img_(\d+)\.npy', os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, nargs='+', default=['img0', 'img1'],
                    help='One or more image roots, each with patches/ (and optionally masks/, pred_masks/)')
    ap.add_argument('--model', type=str, default='cpsam',
                    help="Pretrained name ('cpsam') or path to a fine-tuned model")
    ap.add_argument('--n', type=int, default=None,
                    help='Limit per-root to first N patches (default: all)')
    ap.add_argument('--indices', type=int, nargs='+', default=None,
                    help='Specific patch indices, applied per-root, e.g. --indices 0 7 42')
    ap.add_argument('--out-name', type=str, default=None,
                    help="Subdir name under each root to save preds to "
                         "(default: 'preds_<timestamp>'; pass 'pred_masks' to overwrite the canonical seeds)")
    ap.add_argument('--view', action='store_true',
                    help='Open Napari to inspect each prediction')
    ap.add_argument('--no-gpu', action='store_true')
    args = ap.parse_args()

    out_name = args.out_name or 'preds_' + datetime.now().strftime('%Y%m%d_%H%M%S')

    plan = []  # list of (root, idx, img_path)
    for root in args.root:
        patch_dir = os.path.join(root, 'patches')
        patches = list_patches(patch_dir)
        if not patches:
            print(f'Warning: no patches in {patch_dir}, skipping')
            continue
        if args.indices is not None:
            wanted = set(args.indices)
            selected = [(i, p) for i, p in patches if i in wanted]
            missing = wanted - {i for i, _ in selected}
            if missing:
                print(f'Warning: indices not in {root}: {sorted(missing)}')
        elif args.n is not None:
            selected = patches[: args.n]
        else:
            selected = patches
        for idx, p in selected:
            plan.append((root, idx, p))

    if not plan:
        raise SystemExit('No patches to run.')

    print(f'Loading model: {args.model}')
    model = models.CellposeModel(gpu=not args.no_gpu, pretrained_model=args.model)

    for root, idx, img_path in plan:
        pred_dir = os.path.join(root, out_name)
        mask_dir = os.path.join(root, 'masks')
        os.makedirs(pred_dir, exist_ok=True)

        img = np.load(img_path).astype(np.float32)
        masks, _, _ = model.eval(img)
        n_cells = int(masks.max())

        pred_path = os.path.join(pred_dir, f'pred_{idx:04d}.npy')
        np.save(pred_path, masks.astype(np.int32))
        print(f'[{root}/{idx}] {os.path.basename(img_path)} -> {n_cells} cells  (saved {pred_path})')

        if args.view:
            import napari
            gt_path = os.path.join(mask_dir, f'mask_{idx:04d}.npy')
            viewer = napari.Viewer(title=f'{root} patch {idx} — pred ({n_cells} cells)')
            viewer.add_image(img, colormap='gray', name='image')
            viewer.add_labels(masks.astype(np.int32), name='prediction')
            if os.path.exists(gt_path):
                viewer.add_labels(np.load(gt_path).astype(np.int32),
                                  name='hand_mask', visible=False)
            napari.run()

    print(f'\nDone. Ran {len(plan)} patches. Output dir per root: {out_name}/')


if __name__ == '__main__':
    main()
