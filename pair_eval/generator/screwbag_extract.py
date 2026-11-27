"""Screw Bag composition map extraction v5 (D module).

Learned rules (from VLM-annotated reference set):
 - part pixels: (gray > 130) OR (local std > 9)   [parts are bright OR textured]
 - fold/bag pixels: gray <= 130 AND lstd <= 6.2 (p90)
Component processing:
 - crimp strip removed by elongation; seal flap excluded via crimp line.
 - big compact blobs (merged parts and/or fold bridges) are Voronoi-split;
   sub-blobs are kept only if part-like (mean lstd >= 7) and size-valid.
 - classification: 2 screws -> longer=long, shorter=short (relative);
   4 smalls -> washer if hole_px >= 3300 else nut (relative fallback).
 - composition gate: exactly {1 long, 1 short, 2 nuts, 2 washers}.

Label scheme: 0=background/bag, 1=long_screw, 2=short_screw, 3=nut, 4=washer.
"""
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

LABELS = {'background': 0, 'long_screw': 1, 'short_screw': 2, 'nut': 3, 'washer': 4}
EXPECTED = {'long_screw': 1, 'short_screw': 1, 'nut': 2, 'washer': 2}

G_T = 130
LSTD_T = 9.0
LSTD_WIN = 11
MEAN_LSTD_T = 13.0      # per-component texture check (real parts >= 15)
MIN_PART_AREA = 2400
MAX_PART_AREA = 90000
SCREW_MIN_LEN = 235
SMALL_MAX_LEN = 215
WASHER_HOLE_PX = 3300
INSET = 50
CRIMP_INSET_FALLBACK = 300
MAX_ELONG = 9.5
SPLIT_AREA = 11000
SPLIT_ELONG = 2.6
SPLIT_SEED_DIST = 25
CLOSING = 15


def local_std(g):
    f = g.astype(np.float64)
    m1 = cv2.blur(f, (LSTD_WIN, LSTD_WIN))
    m2 = cv2.blur(f * f, (LSTD_WIN, LSTD_WIN))
    return np.sqrt(np.maximum(m2 - m1 * m1, 0))


def bag_bbox_of(g):
    bagmask = g > 32
    lab, _ = ndimage.label(bagmask)
    if lab.max() == 0:
        return None
    sizes = ndimage.sum(bagmask, lab, range(1, lab.max() + 1))
    main = int(np.argmax(sizes)) + 1
    sl = ndimage.find_objects(lab == main)[0]
    return (int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop))


def crimp_left_x(g, bag_bbox):
    x1, y1, x2, y2 = bag_bbox
    bright = (g[y1:y2, x1:x2] > 85)
    colfrac = bright.sum(axis=0) / max(1, bright.shape[0])
    dense = colfrac > 0.62
    runs, in_run, start = [], False, 0
    for i, d in enumerate(dense):
        if d and not in_run:
            in_run, start = True, i
        elif not d and in_run:
            in_run = False
            if i - start >= 25:
                runs.append((start, i))
    if in_run and len(dense) - start >= 25:
        runs.append((start, len(dense)))
    if not runs:
        return x2 - CRIMP_INSET_FALLBACK
    right = max(runs, key=lambda r: r[1])
    return x1 + right[0]


def voronoi_split(sub):
    u = sub.astype(np.uint8)
    dist = cv2.distanceTransform(u, cv2.DIST_L2, 5)
    peaks = (dist == ndimage.maximum_filter(dist, 21)) & (dist >= SPLIT_SEED_DIST)
    plab, npeak = ndimage.label(peaks)
    if npeak <= 1:
        return [sub]
    _, inds = ndimage.distance_transform_edt(~peaks, return_indices=True)
    assign = plab[inds[0], inds[1]]
    out = []
    for i in range(1, npeak + 1):
        m = (assign == i) & sub
        if m.sum() >= MIN_PART_AREA:
            out.append(m)
    return out if out else [sub]


def part_like(props, mean_lstd):
    L, elong = props['L'], props['elong']
    if mean_lstd < MEAN_LSTD_T:
        return False
    if SMALL_MAX_LEN >= L:
        return True  # compact small part (nut/washer)
    if L >= SCREW_MIN_LEN and 2.0 <= elong <= MAX_ELONG:
        return True  # screw
    return False


def detect(a):
    g = a.mean(axis=2).astype(np.uint8)
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    bag = bag_bbox_of(g)
    if bag is None:
        return dict(ok=False, error='no_bag')
    cx = crimp_left_x(g, bag)
    interior = (bag[0] + INSET, bag[1] + INSET, min(cx - 12, bag[2] - 60), bag[3] - INSET)
    ix1, iy1, ix2, iy2 = interior
    if ix2 - ix1 < 500 or iy2 - iy1 < 350:
        return dict(ok=False, error='interior_too_small', bag_bbox=bag, interior=interior)
    inside = np.zeros(g.shape, bool)
    inside[iy1:iy2, ix1:ix2] = True
    ls = local_std(g)
    metal = ((g > G_T) | (ls > LSTD_T)) & inside
    metal = ndimage.binary_closing(metal, np.ones((CLOSING, CLOSING)))
    metal = ndimage.binary_opening(metal, np.ones((5, 5)))
    lab, n = ndimage.label(metal)
    comps = []  # (slice, mask)
    for i, s in enumerate(ndimage.find_objects(lab), 1):
        sub = lab[s] == i
        area = int(sub.sum())
        if area < MIN_PART_AREA or area > 600000:
            continue
        h, w = sub.shape
        if min(h, w) < 28:
            continue
        ys, xs = np.where(sub)
        xs0 = xs - xs.mean(); ys0 = ys - ys.mean()
        cov = np.cov(np.stack([xs0, ys0]))
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = float(np.sqrt(max(ev[0], 1e-3) / max(ev[1], 1e-3)))
        if elong > MAX_ELONG:
            continue  # crimp strip / seal artifacts
        if area > SPLIT_AREA:
            for m in voronoi_split(sub):
                comps.append((s, m))
            continue
        comps.append((s, sub))
    parts = []
    for s, m in comps:
        area = int(m.sum())
        if area < MIN_PART_AREA or area > MAX_PART_AREA:
            continue
        h, w = m.shape
        filled = ndimage.binary_fill_holes(m)
        holes = int(filled.sum() - m.sum())
        ys, xs = np.where(m)
        xs0 = xs - xs.mean(); ys0 = ys - ys.mean()
        cov = np.cov(np.stack([xs0, ys0]))
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = float(np.sqrt(max(ev[0], 1e-3) / max(ev[1], 1e-3)))
        mlstd = float(ls[s][m].mean())
        props = dict(area=area, L=int(max(h, w)), h=int(h), w=int(w), hole_px=holes,
                     hole_ratio=holes / max(1, int(filled.sum())), elong=elong, mean_lstd=mlstd)
        if not part_like(props, mlstd):
            continue  # fold residue after split
        parts.append(dict(area=area, L=int(max(h, w)), h=int(h), w=int(w), hole_px=holes,
                          hole_ratio=holes / max(1, int(filled.sum())), elong=elong, mean_lstd=mlstd,
                          bbox=(int(s[1].start), int(s[0].start), int(s[1].stop), int(s[0].stop)),
                          mask=m, slice=s))
    return dict(ok=True, parts=parts, bag_bbox=bag, interior=interior, crimp_x=int(cx))


def classify(parts):
    flags = []
    screws = [p for p in parts if p['L'] >= SCREW_MIN_LEN]
    smalls = [p for p in parts if p['L'] < SCREW_MIN_LEN]
    if len(screws) != 2:
        flags.append(f'n_screws={len(screws)}')
        return None, flags
    if len(smalls) != 4:
        flags.append(f'n_small={len(smalls)}')
        return None, flags
    s = sorted(screws, key=lambda p: -p['L'])
    s[0]['cls'], s[1]['cls'] = 'long_screw', 'short_screw'
    if s[0]['L'] / max(1, s[1]['L']) < 1.08:
        flags.append('screw_lengths_ambiguous')
    for p in smalls:
        p['cls'] = 'washer' if p['hole_px'] >= WASHER_HOLE_PX else 'nut'
    if sum(p['cls'] == 'washer' for p in smalls) != 2:
        ranked = sorted(smalls, key=lambda p: -p['hole_px'])
        for p in smalls:
            p['cls'] = 'nut'
        for p in ranked[:2]:
            p['cls'] = 'washer'
        flags.append('washer_nut_relative_fallback')
    counts = {}
    for p in parts:
        counts[p['cls']] = counts.get(p['cls'], 0) + 1
    if counts != EXPECTED:
        flags.append(f'counts={counts}')
        return None, flags
    return parts, flags


def build_map(path):
    a = np.array(Image.open(path))
    det = detect(a)
    if not det['ok']:
        return None, None, [det.get('error')], det
    parts, flags = classify(det['parts'])
    if parts is None:
        return None, None, flags, det
    H, W = a.shape[:2]
    full = np.zeros((H, W), np.uint8)
    for p in parts:
        x1, y1, x2, y2 = p['bbox']
        full[y1:y2, x1:x2][p['mask']] = LABELS[p['cls']]
    m256 = np.array(Image.fromarray(full).resize((256, 256), Image.NEAREST))
    return m256, parts, flags, det


def overlay(a, parts):
    vis = a.copy()
    colors = {'long_screw': (255, 60, 60), 'short_screw': (255, 160, 60),
              'nut': (60, 120, 255), 'washer': (60, 255, 120)}
    for p in parts:
        if 'cls' not in p:
            continue
        x1, y1, x2, y2 = p['bbox']
        c = colors[p['cls']]
        vis[y1:y1 + 3, x1:x2] = c; vis[y2 - 3:y2, x1:x2] = c
        vis[y1:y2, x1:x1 + 3] = c; vis[y1:y2, x2 - 3:x2] = c
    return vis


if __name__ == '__main__':
    import glob, sys
    srcs = sys.argv[1:]
    if not srcs:
        srcs = sorted(glob.glob('/home/z/my-project/workspace/data/mvtec_loco/screw_bag/train/good/*.png'))[:60]
    nok = 0
    for src in srcs:
        m, parts, flags, det = build_map(src)
        if parts is not None:
            nok += 1
        elif len(sys.argv) > 1:
            print(f"{src.split('/')[-1]}: FAIL {flags}")
    print(f'PASS {nok}/{len(srcs)}')
