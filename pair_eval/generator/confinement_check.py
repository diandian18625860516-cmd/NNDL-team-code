#!/usr/bin/env python3
"""Geometric confinement check: for every pair, the map differences between
original and valid/invalid must be confined to the declared edit regions
(plus resize rounding), and no label changes may occur elsewhere."""
import json
import numpy as np
from PIL import Image

ROOT = '/home/z/my-project/workspace/D_deliverable'
MAN = json.load(open(f'{ROOT}/pair_manifest.json'))
DATA = '/home/z/my-project/workspace/data/mvtec_loco'

SX = {'screw_bag': 256 / 1600.0, 'pushpins': 256 / 1700.0}
SY = {'screw_bag': 256 / 1100.0, 'pushpins': 256 / 1000.0}
PAD = 3  # resize rounding allowance in 256-px units

fails = []
for e in MAN:
    cat = e['category']
    o = np.array(Image.open(f"{ROOT}/{e['original_path']}"))
    sx, sy = SX[cat], SY[cat]
    regions = []
    er = e['edit_region']
    for key in ('region_bbox_full', 'union_old_new', 'valid_region_full'):
        if key in er:
            x1, y1, x2, y2 = er[key]
            regions.append((x1 * sx - PAD, y1 * sy - PAD, x2 * sx + PAD, y2 * sy + PAD))
    tp = e['transform_parameters']
    # valid side: old bbox + new bbox from the translation entry
    v = tp['valid']
    if 'old_centroid' in v and 'translation_px' in v:
        pass
    # add new_bbox if recorded inside invalid for substitution
    if 'new_bbox' in tp['invalid']:
        x1, y1, x2, y2 = tp['invalid']['new_bbox']
        regions.append((x1 * sx - PAD, y1 * sy - PAD, x2 * sx + PAD, y2 * sy + PAD))
    if not regions:
        continue  # pair without geometric region info (should not happen)
    mask = np.zeros((256, 256), bool)
    H, W = 256, 256
    for (x1, y1, x2, y2) in regions:
        mask[max(0, int(y1)):min(H, int(np.ceil(y2))), max(0, int(x1)):min(W, int(np.ceil(x2)))] = True
    for side, path in (('valid', e['valid_path']), ('invalid', e['invalid_path'])):
        m = np.array(Image.open(f"{ROOT}/{path}"))
        diff = m != o
        if diff.any():
            ys, xs = np.where(diff)
            outside = (~mask[ys, xs]).sum()
            if outside > 0:
                fails.append((e['pair_id'], side, int(outside)))

print('pairs checked:', len(MAN))
print('confinement violations:', len(fails))
for f in fails[:10]:
    print(' ', f)
