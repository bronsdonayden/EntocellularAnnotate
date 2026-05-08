"""
cpsam_train.py — fine-tune Cellpose 4 (cpsam) on annotated patches.

Pairs `<root>/patches/img_NNNN.npy` with `<root>/masks/mask_NNNN.npy` across
one or more image roots, splits off a validation set, and fine-tunes from
the pretrained cpsam checkpoint.

Usage:
    python cpsam_train.py                              # all roots: img0, img1
    python cpsam_train.py --root img0                  # only img0
    python cpsam_train.py --epochs 200 --lr 5e-5 --model-name my_cells
"""

import argparse
import glob
import os
import re
from datetime import datetime
import numpy as np

from cellpose import models, train

MODEL_OUT = './models'


def load_pairs(roots):
    pairs = []
    for root in roots:
        patch_dir = os.path.join(root, 'patches')
        mask_dir = os.path.join(root, 'masks')
        for img_path in sorted(glob.glob(os.path.join(patch_dir, 'img_*.npy'))):
            m = re.match(r'img_(\d+)\.npy', os.path.basename(img_path))
            if not m:
                continue
            mask_path = os.path.join(mask_dir, f'mask_{m.group(1)}.npy')
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
    return pairs


def split(pairs, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(pairs) * val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


def load(pairs):
    imgs = [np.load(p).astype(np.float32) for p, _ in pairs]
    masks = [np.load(m).astype(np.int32) for _, m in pairs]
    return imgs, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, nargs='+', default=['img0', 'img1'],
                    help='One or more image roots, each with patches/ + masks/')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--weight-decay', type=float, default=0.1)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model-name', type=str, default=None,
                    help="Name for the saved model under ./models (default: 'cpsam_finetuned_<timestamp>')")
    ap.add_argument('--no-gpu', action='store_true')
    args = ap.parse_args()

    model_name = args.model_name or 'cpsam_finetuned_' + datetime.now().strftime('%Y%m%d_%H%M%S')

    pairs = load_pairs(args.root)
    if not pairs:
        raise SystemExit(f'No (img, mask) pairs found under: {args.root}')

    train_pairs, val_pairs = split(pairs, args.val_frac, args.seed)
    train_imgs, train_masks = load(train_pairs)
    val_imgs, val_masks = load(val_pairs)

    print(f'Pairs found: {len(pairs)}  ->  train: {len(train_pairs)}  val: {len(val_pairs)}')

    os.makedirs(MODEL_OUT, exist_ok=True)
    model = models.CellposeModel(gpu=not args.no_gpu, pretrained_model='cpsam')

    train.train_seg(
        model.net,
        train_data=train_imgs,
        train_labels=train_masks,
        test_data=val_imgs,
        test_labels=val_masks,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        save_path=MODEL_OUT,
        model_name=model_name,
    )

    print(f'Saved model under: {os.path.join(MODEL_OUT, model_name)}')


if __name__ == '__main__':
    main()
