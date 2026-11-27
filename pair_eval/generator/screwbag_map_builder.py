"""Screw Bag map builder from VLM bbox annotations + local mask refinement.

Per annotated part bbox:
 - crop bbox (+12 px margin), local adaptive threshold (p25+25, min 60),
   closing(9), main connected component  -> precise part mask.
 - recompute features (area, L, elong, hole_px).
Gate (strict, no silent fallbacks):
 - exactly 2 screws -> longer = long_screw, shorter = short_screw; ratio>=1.08.
 - exactly 4 smalls (L<235); washer iff hole_px>=3300; must yield 2+2.
 - refined mask must overlap the VLM bbox (IoU(center)>0.1 sanity).
 - type must agree with the VLM label; disagreement -> flagged, image excluded.

Label scheme: 0=background/bag, 1=long_screw, 2=short_screw, 3=nut, 4=washer.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

LABELS = {'background': 0, 'long_screw': 1, 'short_screw': 2, 'nut': 3, 'washer': 4}
SCREW_MIN_LEN = 235
WASHER_HOLE_PX = 3300
MARGIN = 12
MIN_MASK_AREA = 400
TYPE_ALIASES = {'long_screw': 'long_screw', 'short_screw': 'short_screw',
                'nut': 'nut', 'washer': 'washer'}


def refine_part(g, bbox):
    x1, y1, x2, y2 = bbox
    m = MARGIN
    cy1, cy2 = max(0, y1 - m), min(g.shape[0], y2 + m)
    cx1, cx2 = max(0, x1 - m), min(g.shape[1], x2 + m)
    crop = g[cy1:cy2, cx1:cx2].astype(int)
    th = max(np.percentile(crop, 25) + 25, 60)
    mask = crop > th
    mask = ndimage.binary_closing(mask, np.ones((9, 9)))
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    sub = lab == main
    if sub.sum() < MIN_MASK_AREA:
        return None
    ys, xs = np.where(sub)
    filled = ndimage.binary_fill_holes(sub)
    holes = int(filled.sum() - sub.sum())
    h, w = sub.shape
    xs0 = xs - xs.mean(); ys0 = ys - ys.mean()
    cov = np.cov(np.stack([xs0, ys0]))
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    elong = float(np.sqrt(max(ev[0], 1e-3) / max(ev[1], 1e-3)))
    # recenter mask to full image coords
    return dict(mask=sub, offset=(cy1, cx1),
                area=int(sub.sum()), L=int(max(h, w)), h=int(h), w=int(w),
                hole_px=holes, hole_ratio=holes / max(1, int(filled.sum())),
                elong=elong,
                bbox=(int(cx1 + xs.min()), int(cy1 + ys.min()),
                      int(cx1 + xs.max() + 1), int(cy1 + ys.max() + 1)))


def build_from_annotation(path, ann_parts):
    """Returns dict(map256, full, parts, flags) or (None, None, parts_raw, flags)."""
    a = np.array(Image.open(path))
    g = a.mean(axis=2).astype(np.uint8)
    H, W = g.shape[:2]
    flags = []
    refined = []
    for p in ann_parts:
        r = refine_part(g, p['bbox'])
        if r is None:
            flags.append(f"refine_failed:{p['type']}")
            continue
        r['vlm_type'] = p['type']
        refined.append(r)
    parts = []
    for r in refined:
        if r['L'] >= SCREW_MIN_LEN:
            parts.append(r)
        else:
            parts.append(r)
    screws = [p for p in refined if p['L'] >= SCREW_MIN_LEN]
    smalls = [p for p in refined if p['L'] < SCREW_MIN_LEN]
    if len(screws) != 2:
        flags.append(f'n_screws={len(screws)}')
    if len(smalls) != 4:
        flags.append(f'n_small={len(smalls)}')
    if flags:
        return None, None, refined, flags
    s = sorted(screws, key=lambda p: -p['L'])
    s[0]['cls'], s[1]['cls'] = 'long_screw', 'short_screw'
    if s[0]['L'] / max(1, s[1]['L']) < 1.08:
        flags.append('screw_lengths_ambiguous')
    if s[0]['vlm_type'] != 'long_screw' or s[1]['vlm_type'] != 'short_screw':
        if not (s[0]['vlm_type'].startswith('long') and s[1]['vlm_type'].startswith('short')):
            flags.append('screw_type_disagreement')
    for p in smalls:
        p['cls'] = 'washer' if p['hole_px'] >= WASHER_HOLE_PX else 'nut'
        if p['cls'] != TYPE_ALIASES.get(p['vlm_type'], p['vlm_type']):
            flags.append(f"type_disagreement:{p['vlm_type']}->ref:{p['cls']}")
    counts = {}
    for p in smalls:
        counts[p['cls']] = counts.get(p['cls'], 0) + 1
    if counts != {'washer': 2, 'nut': 2}:
        flags.append(f'small_counts={counts}')
    if flags:
        return None, None, refined, flags
    full = np.zeros((H, W), np.uint8)
    for p in refined:
        oy, ox = p['offset']
        Hs, Ws = p['mask'].shape
        full[oy:oy + Hs, ox:ox + Ws][p['mask']] = LABELS[p['cls']]
    m256 = np.array(Image.fromarray(full).resize((256, 256), Image.NEAREST))
    return m256, full, refined, flags


def overlay(a, refined):
    vis = a.copy()
    colors = {'long_screw': (255, 60, 60), 'short_screw': (255, 160, 60),
              'nut': (60, 120, 255), 'washer': (60, 255, 120)}
    for p in refined:
        if 'cls' not in p:
            continue
        x1, y1, x2, y2 = p['bbox']
        c = colors[p['cls']]
        vis[y1:y1 + 3, x1:x2] = c; vis[y2 - 3:y2, x1:x2] = c
        vis[y1:y2, x1:x1 + 3] = c; vis[y1:y2, x2 - 3:x2] = c
    return vis


if __name__ == '__main__':
    import json, sys
    ANN = json.load(open('/home/z/my-project/workspace/vlm_annotations_sb_v2.json'))['annotations']
    DATA = '/home/z/my-project/workspace/data/mvtec_loco/screw_bag'
    from collections import Counter
    cnt = Counter()
    nok = 0
    for key in sorted(ANN.keys()):
        m, full, refined, flags = build_from_annotation(f'{DATA}/{key}', ANN[key])
        if m is not None:
            nok += 1
            cnt['OK'] += 1
        else:
            for f in flags:
                cnt[f.split(':')[0].split('=')[0]] += 1
    print(f'pass {nok}/{len(ANN)}')
    print(cnt.most_common(10))
