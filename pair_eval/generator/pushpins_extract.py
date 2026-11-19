"""Pushpins composition map extraction v3 — geometry-accurate cells.

The 3x5 grid of cells is a physical structure (milky plastic walls). We detect
the wall lattice as the largest bright connected component; the 15 cell REGIONS
are the holes of that lattice (fill_holes minus lattice). Each yellow head is
assigned to the cell region containing its centroid (tilt-proof). Needle pixels
are bright pixels not in walls, assigned to the nearest head with a distance
cap. The separator label = the actual wall-lattice pixels.

Label scheme: 0=background, 1=pushpin, 2=separator.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

LABELS = {'background': 0, 'pushpin': 1, 'separator': 2}
HEAD_AREA = (1400, 7000)
NEEDLE_T = 95
NEEDLE_CAP = 260
WALL_T = 60
N_ROWS, N_COLS = 3, 5


def yellow_heads(a):
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    y = (R > 140) & (G > 80) & (B < 130) & ((R - B) > 60) & ((G - B) > 20)
    lab, n = ndimage.label(y)
    hd = []
    for i in range(1, n + 1):
        m = lab == i
        s = int(m.sum())
        if HEAD_AREA[0] <= s <= HEAD_AREA[1]:
            cy, cx = ndimage.center_of_mass(m)
            sl = ndimage.find_objects(m)[0]
            hd.append(dict(area=s, cx=float(cx), cy=float(cy), comp=i,
                           bbox=(int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop))))
    return hd, lab


def wall_lattice(g, yellow):
    """Detect the wall grid structure. Returns (wall_mask, box) or (None, None)."""
    bright = (g > WALL_T) & (g < 245)
    bright = bright & ~ndimage.binary_dilation(yellow, np.ones((7, 7)))
    lab, n = ndimage.label(bright)
    if n == 0:
        return None, None
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    wall = lab == main
    ys, xs = np.where(wall)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return wall, box


def extract(path):
    a = np.array(Image.open(path)).astype(int)
    g = a.mean(2)
    hd, ylab = yellow_heads(a)
    flags = []
    if len(hd) != 15:
        return None, [f'n_heads={len(hd)}']
    yellow = ylab > 0
    wall, box = wall_lattice(g, yellow)
    if wall is None:
        return None, ['no_wall_lattice']
    H, W = g.shape
    # cell regions = holes of the wall lattice; keep the 15 largest only
    # (small holes inside the wall body are plastic texture, not cells)
    wall = ndimage.binary_closing(wall, np.ones((11, 11)))  # bridge small gaps
    filled = ndimage.binary_fill_holes(wall)
    cellarea = filled & ~wall
    clab, ncells = ndimage.label(cellarea)
    if ncells < 15:
        return None, [f'n_cells={ncells}']
    sizes = ndimage.sum(cellarea, clab, range(1, ncells + 1))
    top15 = sorted(range(1, ncells + 1), key=lambda i: -sizes[i - 1])[:15]
    if sizes[top15[-1] - 1] < 25000:
        return None, [f'cell_too_small={int(sizes[top15[-1] - 1])}']
    keep = np.zeros(clab.shape, np.int32)
    for j, i in enumerate(sorted(top15), 1):
        keep[clab == i] = j
    clab = keep
    ncells = 15
    # order cells into 3x5 grid by centroid
    cents = ndimage.center_of_mass(clab > 0, clab, range(1, 16))
    order = sorted(range(15), key=lambda i: (cents[i][0], cents[i][1]))
    cells = {}
    for pos, i in enumerate(order):
        r, c = divmod(pos, 5)
        sl = ndimage.find_objects(clab == i + 1)[0]
        cells[(r, c)] = dict(x1=int(sl[1].start), y1=int(sl[0].start),
                             x2=int(sl[1].stop), y2=int(sl[0].stop),
                             cx=float(cents[i][1]), cy=float(cents[i][0]),
                             comp=i + 1)
    # assign each head to the cell region containing its centroid
    occ = {}
    assign_ok = True
    for h in hd:
        ci = clab[int(round(h['cy'])), int(round(h['cx']))]
        if ci == 0:
            # head not inside any cell region (on a wall): fall back to nearest
            dists = [np.hypot(h['cx'] - cc['cx'], h['cy'] - cc['cy']) for cc in cells.values()]
            k = list(cells.keys())[int(np.argmin(dists))]
            if min(dists) > 120:
                assign_ok = False
                break
        else:
            k = next(key for key, cc in cells.items() if cc['comp'] == ci)
        h['cell'] = k
        occ[k] = occ.get(k, 0) + 1
    if not assign_ok:
        return None, ['head_outside_grid']
    if len(occ) != 15 or any(v != 1 for v in occ.values()):
        return None, [f'occupancy={len(occ)}']
    # needles: bright, not yellow, not wall
    cand = (g > NEEDLE_T) & ~yellow & ~wall
    hy = np.array([h['cy'] for h in hd]); hx = np.array([h['cx'] for h in hd])
    pts = np.argwhere(cand)
    assign = np.zeros((H, W), np.int8)
    if len(pts):
        d = np.stack([np.hypot(pts[:, 1] - hx[k], pts[:, 0] - hy[k]) for k in range(len(hd))])
        nearest = d.argmin(0); dmin = d.min(0)
        keep = dmin <= NEEDLE_CAP
        for k, (py, px) in zip(nearest[keep], pts[keep]):
            assign[py, px] = k + 1
    # full canvas
    full = np.zeros((H, W), np.uint8)
    full[wall] = LABELS['separator']
    pins = []
    for k, h in enumerate(hd):
        pm = (ylab == h['comp']) | (assign == k + 1)
        full[pm] = LABELS['pushpin']
        ys, xs = np.where(pm)
        pins.append(dict(idx=k, area=int(pm.sum()), cell=h['cell'],
                         head_area=h['area'], cx=float(xs.mean()), cy=float(ys.mean()),
                         bbox=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
                         mask=pm))
    m256 = np.array(Image.fromarray(full).resize((256, 256), Image.NEAREST))
    grid = dict(box=box, cells=cells)
    return dict(map=m256, full=full, pins=pins, grid=grid, heads=hd, wall=wall), flags


def build_map(path):
    res, flags = extract(path)
    if res is None:
        return None, None, flags
    return res['map'], res, flags


if __name__ == '__main__':
    import glob, sys
    srcs = sys.argv[1:] or sorted(glob.glob('/home/z/my-project/workspace/data/mvtec_loco/pushpins/train/good/*.png'))[:12]
    nok = 0
    for s in srcs:
        m, rec, flags = build_map(s)
        ok = rec is not None
        nok += ok
        print(f"{s.split('/')[-1]}: {'OK' if ok else 'FAIL'} {flags}")
    print(f'PASS {nok}/{len(srcs)}')
