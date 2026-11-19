"""Pair edit generation for D's legal/violation pair list (composition maps).

Convention: every part/pin is stored as a TIGHT-CROPPED boolean mask plus its
global bounding box (x1, y1, x2, y2) — mask.shape == (y2-y1, x2-x1).
`fmask_of` materializes a full-canvas mask on demand.

Screw Bag (labels: 0 bg, 1 long_screw, 2 short_screw, 3 nut, 4 washer):
  pair A: valid = relocate a loose part; invalid = remove the SAME part.
  pair B: valid = relocate the other loose part class;
          invalid = short screw replaced by the long screw (count preserved).

Pushpins (labels: 0 bg, 1 pushpin, 2 separator):
  pair A: valid = in-cell move of pin i;
          invalid = cross-cell move of the SAME pin i (count preserved).
  pair B: valid = in-cell move of another pin; invalid = remove one pin.

Deterministic candidate grids; order derived from a caller-supplied rng seed.
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def resize_map(full):
    return np.array(Image.fromarray(full.astype(np.uint8)).resize((256, 256), Image.NEAREST))


def fmask_of(tight, bbox, shape):
    H, W = shape
    m = np.zeros(shape, bool)
    x1, y1, x2, y2 = bbox
    m[y1:y2, x1:x2] = tight
    return m


def translate_tight(tight, bbox, dxy, shape):
    """Translate a tight mask by (dx, dy); returns (full-canvas mask, new bbox)
    or None if it leaves the canvas."""
    H, W = shape
    x1, y1, x2, y2 = bbox
    h, w = tight.shape
    nx1, ny1 = int(round(x1 + dxy[0])), int(round(y1 + dxy[1]))
    nx2, ny2 = nx1 + w, ny1 + h
    if nx1 < 0 or ny1 < 0 or nx2 > W or ny2 > H:
        return None
    out = np.zeros(shape, bool)
    out[ny1:ny2, nx1:nx2] = tight
    return out, (nx1, ny1, nx2, ny2)


def region_ok(new_mask, new_bbox, other_fulls, bounds, dilate=8, margin=10):
    """Translated mask must stay inside bounds and overlap nothing else."""
    x1, y1, x2, y2 = new_bbox
    if not (bounds[0] + margin <= x1 and bounds[1] + margin <= y1
            and x2 <= bounds[2] - margin and y2 <= bounds[3] - margin):
        return False
    if not other_fulls:
        return True
    blocked = np.zeros(new_mask.shape, bool)
    for om in other_fulls:
        blocked |= om
    blocked = ndimage.binary_dilation(blocked, np.ones((dilate, dilate)))
    return not (new_mask & blocked).any()


# ---------------- Screw Bag edits ----------------

def sb_rearrange(full, tight, bbox, other_fulls, interior, rng):
    """Deterministically relocate one loose part. Returns edit dict or None."""
    base_angle = int(rng * 360)
    cands = []
    for k in range(16):
        ang = np.deg2rad(base_angle + k * 22.5)
        for dist in (320, 240, 180, 130):
            cands.append((dist * np.cos(ang), dist * np.sin(ang)))
    for dx, dy in cands:
        res = translate_tight(tight, bbox, (dx, dy), full.shape)
        if res is None:
            continue
        nm, nb = res
        if not region_ok(nm, nb, other_fulls, interior):
            continue
        ox, oy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        nx, ny = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
        dist = float(np.hypot(nx - ox, ny - oy))
        if dist < 110:
            continue
        return dict(new_mask=nm, new_bbox=nb, translation=(float(dx), float(dy)),
                    distance=dist, old_centroid=(float(ox), float(oy)),
                    new_centroid=(float(nx), float(ny)))
    return None


def apply_rearrange(full, tight, bbox, edit, label):
    out = full.copy()
    out[bbox[1]:bbox[3], bbox[0]:bbox[2]][tight] = 0
    out[edit['new_mask']] = label
    return out


def apply_remove(full, tight, bbox):
    out = full.copy()
    out[bbox[1]:bbox[3], bbox[0]:bbox[2]][tight] = 0
    return out


def rotate_tight(tight, angle_deg, target_c, shape):
    """Rotate a tight mask and center it at target_c. Returns (full mask, bbox)."""
    m = tight.astype(np.uint8) * 255
    rot = ndimage.rotate(m, angle_deg, reshape=True, order=0) > 0
    h, w = rot.shape
    cx, cy = target_c
    x1 = int(round(cx - w / 2)); y1 = int(round(cy - h / 2))
    x2, y2 = x1 + w, y1 + h
    H, W = shape
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None
    out = np.zeros(shape, bool)
    out[y1:y2, x1:x2] = rot
    return out, (x1, y1, x2, y2)


def sb_spec_substitution(full, long_tight, long_bbox, short_tight, short_bbox,
                         other_fulls, interior, rng):
    """Replace the short screw with the long screw (rotated to fit)."""
    scx = (short_bbox[0] + short_bbox[2]) / 2
    scy = (short_bbox[1] + short_bbox[3]) / 2
    start = int(rng * 360)
    angles = [(start + 15 * k) % 360 for k in range(24)]
    # deterministic offset grid: first the exact replacement position, then
    # nearby offsets (recorded in the manifest) when the fit fails
    offs = [(0.0, 0.0)]
    for ang0 in range(0, 360, 45):
        for r in (110.0, 70.0):
            offs.append((r * np.cos(np.deg2rad(ang0)), r * np.sin(np.deg2rad(ang0))))
    for dx, dy in offs:
        for ang in angles:
            res = rotate_tight(long_tight, ang, (scx + dx, scy + dy), full.shape)
            if res is None:
                continue
            nm, nb = res
            if not region_ok(nm, nb, other_fulls, interior, dilate=8, margin=10):
                continue
            return dict(new_mask=nm, new_bbox=nb, rotation_deg=float(ang),
                        placement_offset_px=(float(dx), float(dy)),
                        source_centroid=(float((long_bbox[0] + long_bbox[2]) / 2),
                                         float((long_bbox[1] + long_bbox[3]) / 2)),
                        target_centroid=(float(scx + dx), float(scy + dy)),
                        area=int(nm.sum()))
    return None


def apply_spec_substitution(full, long_tight, long_bbox, short_tight, short_bbox, edit, long_label):
    """Replace the short screw with a rotated COPY of the long screw; the
    original long screw stays in place (total part count preserved at 6)."""
    out = full.copy()
    out[short_bbox[1]:short_bbox[3], short_bbox[0]:short_bbox[2]][short_tight] = 0
    out[edit['new_mask']] = long_label
    return out


# ---------------- Pushpins edits ----------------

def pp_move_within_cell(full, tight, bbox, cell, rng, min_dist=25):
    """Translate a pin within its own cell (centroid stays inside the cell region;
    wall overlap tolerated up to 10% of the pin area). Picks the farthest move."""
    x1, y1, x2, y2 = bbox
    wall = full == 2
    cx1, cy1, cx2, cy2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
    ccx, ccy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
    base = int(rng * 360)
    best = None
    for k in range(12):
        ang = np.deg2rad((base + k * 30) % 360)
        for frac in (1.0, 0.7, 0.45):
            dx = np.cos(ang) * 260 * frac
            dy = np.sin(ang) * 260 * frac
            res = translate_tight(tight, bbox, (dx, dy), full.shape)
            if res is None:
                continue
            nm, nb = res
            ncx, ncy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
            # centroid must remain inside its own cell region (small inset)
            if not (cx1 + 12 <= ncx <= cx2 - 12 and cy1 + 12 <= ncy <= cy2 - 12):
                continue
            overlap_wall = int((nm & wall).sum())
            if overlap_wall > 0.10 * tight.sum():
                continue
            ox, oy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = float(np.hypot(ncx - ox, ncy - oy))
            if dist < min_dist:
                continue
            if best is None or dist > best['distance']:
                best = dict(new_mask=nm, new_bbox=nb, translation=(float(dx), float(dy)),
                            distance=dist, old_centroid=(float(ox), float(oy)),
                            new_centroid=(float(ncx), float(ncy)),
                            wall_overlap_px=overlap_wall)
    return best


def pp_move_cross_cell(full, tight, bbox, target_cell, resident_full, rng):
    """Move the pin into the target (adjacent) cell: centroid lands inside the
    target cell region, no overlap with the resident pin (dilated)."""
    x1, y1, x2, y2 = bbox
    wall = full == 2
    cx1, cy1, cx2, cy2 = target_cell['x1'], target_cell['y1'], target_cell['x2'], target_cell['y2']
    blocked = None
    if resident_full is not None:
        blocked = ndimage.binary_dilation(resident_full, np.ones((8, 8)))
    ox, oy = (x1 + x2) / 2, (y1 + y2) / 2
    spots = []
    for sx in (0.2, 0.35, 0.5, 0.65, 0.8):
        for sy in (0.2, 0.35, 0.5, 0.65, 0.8):
            spots.append((cx1 + (cx2 - cx1) * sx, cy1 + (cy2 - cy1) * sy))
    for tx, ty in spots:
        dx, dy = tx - ox, ty - oy
        res = translate_tight(tight, bbox, (dx, dy), full.shape)
        if res is None:
            continue
        nm, nb = res
        ncx, ncy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
        if not (cx1 + 12 <= ncx <= cx2 - 12 and cy1 + 12 <= ncy <= cy2 - 12):
            continue
        overlap_wall = int((nm & wall).sum())
        if overlap_wall > 0.10 * tight.sum():
            continue
        if blocked is not None and (nm & blocked).any():
            continue
        dist = float(np.hypot(ncx - ox, ncy - oy))
        return dict(new_mask=nm, new_bbox=nb, translation=(float(dx), float(dy)),
                    distance=dist, old_centroid=(float(ox), float(oy)),
                    new_centroid=(float(ncx), float(ncy)),
                    wall_overlap_px=overlap_wall)
    return None
