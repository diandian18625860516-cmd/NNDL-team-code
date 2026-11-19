#!/usr/bin/env python3
"""D deliverable orchestrator: builds the complete legal/violation pair package.

 1. Build source records (screw_bag: VLM-annotation-based refinement; pushpins:
    geometry-based extraction) — tight-mask convention, cached to disk.
 2. Fixed source-purpose assignment (train / development / final_diagnostic),
    decided BEFORE any variant is generated.
    v2: purposes come from B's OFFICIAL split manifests (per-image, interleaved),
    included at generator/official_split/*.csv — NOT from contiguous ranges and
    NOT from validation/good (B assigns every edit-source role inside train/good).
 3. Generate edits (valid + invalid) on full-resolution label canvases; save
    256x256 single-channel PNGs (original / valid / invalid).
 4. Verify every pair (sizes, labels, counts, hashes) -> qa/qa_report.json.
 5. pair_manifest.json (README v1 schema array), source_status.json, stats.

Deterministic: candidate orders derived from CRC32(pair_id); no RNG state.
"""
import csv
import hashlib
import json
import os
import pickle
import shutil
import zlib
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

import pair_generator as pg
import screwbag_map_builder as sbmb
import pushpins_extract as ppe
from screwbag_extract import bag_bbox_of, crimp_left_x

DATA_ROOT = '/home/z/my-project/workspace/data/mvtec_loco'
ANN_SB = '/home/z/my-project/workspace/vlm_annotations_sb_v2.json'
OUT_ROOT = '/home/z/my-project/workspace/D_deliverable'
CACHE = '/home/z/my-project/workspace/records_cache.pkl'

LABELS_SB = {'background': 0, 'long_screw': 1, 'short_screw': 2, 'nut': 3, 'washer': 4}

# ---------------- assignment policy (v2): B's official split manifest ----------------
# v1 used contiguous filename ranges (train 000-240 / dev 240-300 / FD validation).
# B has since delivered the OFFICIAL split lists (per-image, interleaved, covering
# ALL train/good images); v2 adopts them verbatim as the authoritative assignment.
OFFICIAL_SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'official_split')

SPLIT_NAME_TO_PURPOSE = {
    'train_edit_source': 'train',
    'dev_source': 'development',
    'final_diag_source': 'final_diagnostic',
}


def load_official_split():
    """Read B's split CSVs -> {category: {image_index: purpose}} (verbatim)."""
    out = {}
    for cat in ('screw_bag', 'pushpins'):
        path = os.path.join(OFFICIAL_SPLIT_DIR, f'{cat}_split_manifest.csv')
        if not os.path.exists(path):
            raise SystemExit(f'[ERROR] B official split manifest missing: {path}')
        m = {}
        with open(path, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                idx = int(r['image_id'].split('.')[0])
                m[idx] = SPLIT_NAME_TO_PURPOSE[r['split'].strip()]
        out[cat] = m
    return out


def purpose_ids(official, cat, purpose):
    """Sorted train/good image indices B assigned to this purpose."""
    return sorted(i for i, p in official[cat].items() if p == purpose)

RULE_REF = {
    'sb_rearrange': ("MVTec LOCO AD (Bergmann et al., IJCV 2022) screw_bag: loose parts "
                     "(nuts/washers) may lie anywhere inside the bag; relocating one "
                     "preserves the required composition 1 long screw + 1 short screw + "
                     "2 nuts + 2 washers, so the product remains valid. "
                     "Basis: all 360 normal training images audited in the progress report."),
    'sb_missing': ("MVTec LOCO AD screw_bag official logical anomaly types "
                   "1_missing_long_screw / 1_missing_short_screw / 1_missing_nut / "
                   "1_missing_washer (defects_config.json pixel values 245/244/243/241). "
                   "Removing a required component violates the composition constraint "
                   "even though every remaining part is intact."),
    'sb_spec': ("MVTec LOCO AD screw_bag official logical anomaly type screw_too_long "
                "(defects_config.json pixel value 255): the screw-length specification is "
                "violated. Our variant keeps the total part count at 6 (count preserved) "
                "while replacing the short screw with a second long screw, isolating the "
                "specification violation from count changes."),
    'pp_incell': ("MVTec LOCO AD (Bergmann et al., IJCV 2022) pushpins: each of the 15 "
                  "cells of the 3x5 grid holds exactly one pushpin in normal images; a pin "
                  "lying at a different position within its own cell keeps one-pin-per-cell "
                  "and the total count of 15, so the product remains valid."),
    'pp_crosscell': ("MVTec LOCO AD pushpins: the one-pin-per-cell constraint is the "
                     "documented normal configuration; moving one pin into a neighbouring "
                     "cell leaves one cell empty and one with two pins while the total "
                     "count stays 15 - a pure allocation violation (analogous in kind to "
                     "the official 1_additional_pushpin / missing_pushpin pair)."),
    'pp_missing': ("MVTec LOCO AD pushpins official logical anomaly type missing_pushpin "
                   "(defects_config.json pixel value 253)."),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def rng_of(pair_id):
    return zlib.crc32(pair_id.encode()) % 10000 / 10000.0


def save_map_png(arr256, path):
    Image.fromarray(arr256.astype(np.uint8), mode='L').save(path)


# ---------------- source records ----------------

def sb_source_record(rel, ann_parts):
    path = f'{DATA_ROOT}/screw_bag/{rel}'
    m256, full, refined, flags = sbmb.build_from_annotation(path, ann_parts)
    if m256 is None:
        return None, flags
    a = np.array(Image.open(path))
    g = a.mean(axis=2).astype(np.uint8)
    bag = bag_bbox_of(g)
    cx = crimp_left_x(g, bag) if bag else None
    interior = (bag[0] + 50, bag[1] + 50, min(cx - 12, bag[2] - 60), bag[3] - 50) if bag else None
    H, W = full.shape
    parts = []
    for p in refined:
        x1, y1, x2, y2 = p['bbox']
        fmask = np.zeros((H, W), bool)
        oy, ox = p['offset']
        Hs, Ws = p['mask'].shape
        fmask[oy:oy + Hs, ox:ox + Ws][p['mask']] = True
        tight = fmask[y1:y2, x1:x2]
        # effective ownership: only pixels that actually carry this part's
        # label in the painted canvas (overlaps resolved by paint order)
        cls_id = LABELS_SB[p['cls']]
        eff = tight & (full[y1:y2, x1:x2] == cls_id)
        parts.append(dict(cls=p['cls'], bbox=(int(x1), int(y1), int(x2), int(y2)),
                          area=int(eff.sum()), L=p['L'], hole_px=p['hole_px'],
                          tight=eff))
    return dict(map=m256, full=full, parts=parts, interior=interior,
                canvas_shape=(H, W)), flags


def pp_source_record(rel):
    path = f'{DATA_ROOT}/pushpins/{rel}'
    m256, rec, flags = ppe.build_map(path)
    if rec is None:
        return None, flags
    pins = []
    for p in rec['pins']:
        x1, y1, x2, y2 = p['bbox']
        tight = p['mask'][y1:y2, x1:x2]
        # effective ownership: only pixels carrying the pushpin label
        eff = tight & (rec['full'][y1:y2, x1:x2] == 1)
        pins.append(dict(cell=p['cell'], bbox=(int(x1), int(y1), int(x2), int(y2)),
                         area=int(eff.sum()), tight=eff))
    return dict(map=m256, full=rec['full'], pins=pins,
                cells={k: v for k, v in rec['grid']['cells'].items()},
                canvas_shape=rec['full'].shape), flags


# ---------------- pair generation ----------------

# v2 fix: minimum effective ownership (full-res px) for a part chosen as the
# edit target. Below this the part is too overlapped to yield a visible edit
# (v1 only used images 000-112; B's interleaved split surfaced image 107 whose
# nut had ~1211 effective px and failed the pixel-level QA). 2600 full-res px
# ≈ 120 map pixels — 2x margin over the QA threshold (removal must drop >60 map px).
MIN_EDIT_PART_AREA = 2600


def sb_gen_pair_a(rec, source_id, pair_id):
    parts = rec['parts']
    interior = rec['interior']
    rng = rng_of(pair_id)
    idx = int(source_id.split('-')[-1])
    want = 'washer' if idx % 2 == 0 else 'nut'
    pool = [p for p in parts if p['cls'] == want and p['area'] >= MIN_EDIT_PART_AREA]
    if not pool:
        pool = [p for p in parts if p['cls'] in ('washer', 'nut')
                and p['area'] >= MIN_EDIT_PART_AREA]
    if not pool:
        return None, ['loose_part_too_overlapped']
    P = max(pool, key=lambda p: p['area'])
    others = [pg.fmask_of(p['tight'], p['bbox'], rec['canvas_shape'])
              for p in parts if p is not P]
    edit = pg.sb_rearrange(rec['full'], P['tight'], P['bbox'], others, interior, rng)
    if edit is None:
        return None, ['rearrange_infeasible']
    lab = LABELS_SB[P['cls']]
    valid_full = pg.apply_rearrange(rec['full'], P['tight'], P['bbox'], edit, lab)
    invalid_full = pg.apply_remove(rec['full'], P['tight'], P['bbox'])
    tp = dict(
        valid=dict(operation='translate', part_class=P['cls'], part_area_px=int(P['area']),
                   translation_px=[round(edit['translation'][0], 1), round(edit['translation'][1], 1)],
                   move_distance_px=round(edit['distance'], 1),
                   old_centroid=[round(edit['old_centroid'][0], 1), round(edit['old_centroid'][1], 1)],
                   new_centroid=[round(edit['new_centroid'][0], 1), round(edit['new_centroid'][1], 1)]),
        invalid=dict(operation='remove', part_class=P['cls'], part_area_px=int(P['area']),
                     removed_bbox=[int(v) for v in P['bbox']]),
        controls=dict(same_part_both_sides=True, edit_area_identical=True,
                      composition_after_valid={'long_screw': 1, 'short_screw': 1, 'nut': 2, 'washer': 2},
                      composition_after_invalid={'long_screw': 1, 'short_screw': 1, 'nut': 2, 'washer': 2,
                                                 P['cls']: 1}),
        canvas=dict(size=[int(rec['canvas_shape'][1]), int(rec['canvas_shape'][0])],
                    map_size=[256, 256], units='full-resolution pixels'),
    )
    er = dict(edit_part=P['cls'],
              region_bbox_full=[int(v) for v in P['bbox']],
              union_old_new=[int(min(P['bbox'][0], edit['new_bbox'][0])),
                             int(min(P['bbox'][1], edit['new_bbox'][1])),
                             int(max(P['bbox'][2], edit['new_bbox'][2])),
                             int(max(P['bbox'][3], edit['new_bbox'][3]))])
    return dict(valid_full=valid_full, invalid_full=invalid_full, tp=tp, edit_region=er,
                rule=('sb_rearrange', 'sb_missing'))


def sb_gen_pair_b(rec, source_id, pair_id):
    parts = rec['parts']
    interior = rec['interior']
    rng = rng_of(pair_id)
    long_p = next((p for p in parts if p['cls'] == 'long_screw'), None)
    short_p = next((p for p in parts if p['cls'] == 'short_screw'), None)
    if long_p is None or short_p is None:
        return None, ['missing_screw_for_substitution']
    # obstacles for the substituted copy: everything except the short screw
    # (which is being replaced); the ORIGINAL long screw must also be avoided
    others = [pg.fmask_of(p['tight'], p['bbox'], rec['canvas_shape'])
              for p in parts if p is not short_p]
    edit = pg.sb_spec_substitution(rec['full'], long_p['tight'], long_p['bbox'],
                                   short_p['tight'], short_p['bbox'], others, interior, rng)
    if edit is None:
        return None, ['substitution_infeasible']
    idx = int(source_id.split('-')[-1])
    want = 'nut' if idx % 2 == 0 else 'washer'
    pool = [p for p in parts if p['cls'] == want and p['area'] >= MIN_EDIT_PART_AREA]
    if not pool:
        pool = [p for p in parts if p['cls'] in ('washer', 'nut')
                and p['area'] >= MIN_EDIT_PART_AREA]
    if not pool:
        return None, ['rearrange_pool_empty_or_too_overlapped']
    P = max(pool, key=lambda p: p['area'])
    others2 = [pg.fmask_of(p['tight'], p['bbox'], rec['canvas_shape'])
               for p in parts if p is not P]
    redit = pg.sb_rearrange(rec['full'], P['tight'], P['bbox'], others2, interior,
                            rng_of(pair_id + 'v'))
    if redit is None:
        return None, ['rearrange_infeasible']
    valid_full = pg.apply_rearrange(rec['full'], P['tight'], P['bbox'], redit,
                                    LABELS_SB[P['cls']])
    invalid_full = pg.apply_spec_substitution(rec['full'], long_p['tight'], long_p['bbox'],
                                               short_p['tight'], short_p['bbox'], edit,
                                               LABELS_SB['long_screw'])
    tp = dict(
        valid=dict(operation='translate', part_class=P['cls'], part_area_px=int(P['area']),
                   translation_px=[round(redit['translation'][0], 1), round(redit['translation'][1], 1)],
                   move_distance_px=round(redit['distance'], 1)),
        invalid=dict(operation='replace_short_screw_with_long_screw',
                     rotation_deg=round(edit['rotation_deg'], 1),
                     placement_offset_px=[round(edit['placement_offset_px'][0], 1),
                                          round(edit['placement_offset_px'][1], 1)],
                     new_bbox=[int(v) for v in edit['new_bbox']],
                     new_area_px=int(edit['area']),
                     removed_short_screw_area_px=int(short_p['area']),
                     note='original long screw kept in place; a rotated copy replaces '
                          'the short screw, so the total part count stays 6'),
        controls=dict(count_preserved=True,
                      composition_after_valid={'long_screw': 1, 'short_screw': 1, 'nut': 2, 'washer': 2},
                      composition_after_invalid={'long_screw': 2, 'short_screw': 0, 'nut': 2, 'washer': 2}),
        canvas=dict(size=[int(rec['canvas_shape'][1]), int(rec['canvas_shape'][0])],
                    map_size=[256, 256], units='full-resolution pixels'),
    )
    er = dict(edit_part='short_screw->long_screw',
              region_bbox_full=[int(v) for v in short_p['bbox']],
              union_old_new=[int(min(short_p['bbox'][0], edit['new_bbox'][0])),
                             int(min(short_p['bbox'][1], edit['new_bbox'][1])),
                             int(max(short_p['bbox'][2], edit['new_bbox'][2])),
                             int(max(short_p['bbox'][3], edit['new_bbox'][3]))],
              valid_region_full=[int(min(P['bbox'][0], redit['new_bbox'][0])),
                                 int(min(P['bbox'][1], redit['new_bbox'][1])),
                                 int(max(P['bbox'][2], redit['new_bbox'][2])),
                                 int(max(P['bbox'][3], redit['new_bbox'][3]))])
    return dict(valid_full=valid_full, invalid_full=invalid_full, tp=tp, edit_region=er,
                rule=('sb_rearrange', 'sb_spec'))


def pp_gen_pair_a(rec, source_id, pair_id):
    pins = rec['pins']
    cells = rec['cells']
    idx = int(source_id.split('-')[-1])
    cell_key = (idx % 3, idx % 5)
    pin = next((p for p in pins if p['cell'] == cell_key), None)
    if pin is None:
        return None, [f'pin_missing_at_{cell_key}']
    rng = rng_of(pair_id)
    vedit = pg.pp_move_within_cell(rec['full'], pin['tight'], pin['bbox'],
                                   cells[cell_key], rng)
    if vedit is None:
        return None, ['incell_infeasible']
    r, c = cell_key
    targets = []
    if c - 1 >= 0:
        targets.append((r, c - 1))
    if c + 1 <= 4:
        targets.append((r, c + 1))
    if r - 1 >= 0:
        targets.append((r - 1, c))
    if r + 1 <= 2:
        targets.append((r + 1, c))
    k0 = int(rng_of(pair_id + 't') * len(targets)) % len(targets)
    targets = targets[k0:] + targets[:k0]
    iedit = None
    tgt_used = None
    for tk in targets:
        resident = next((p for p in pins if p['cell'] == tk), None)
        res_full = pg.fmask_of(resident['tight'], resident['bbox'],
                               rec['canvas_shape']) if resident is not None else None
        iedit = pg.pp_move_cross_cell(rec['full'], pin['tight'], pin['bbox'],
                                      cells[tk], res_full, rng)
        if iedit is not None:
            tgt_used = tk
            break
    if iedit is None:
        return None, ['crosscell_infeasible']
    valid_full = pg.apply_rearrange(rec['full'], pin['tight'], pin['bbox'], vedit, 1)
    invalid_full = pg.apply_rearrange(rec['full'], pin['tight'], pin['bbox'], iedit, 1)
    tp = dict(
        valid=dict(operation='translate_within_cell', source_cell=list(cell_key),
                   pin_area_px=int(pin['area']),
                   translation_px=[round(vedit['translation'][0], 1), round(vedit['translation'][1], 1)],
                   move_distance_px=round(vedit['distance'], 1),
                   wall_overlap_px=int(vedit.get('wall_overlap_px', 0))),
        invalid=dict(operation='translate_to_adjacent_cell', source_cell=list(cell_key),
                     target_cell=list(tgt_used), pin_area_px=int(pin['area']),
                     translation_px=[round(iedit['translation'][0], 1), round(iedit['translation'][1], 1)],
                     move_distance_px=round(iedit['distance'], 1),
                     wall_overlap_px=int(iedit.get('wall_overlap_px', 0))),
        controls=dict(same_pin_both_sides=True, edit_area_identical=True,
                      total_pin_count_valid=15, total_pin_count_invalid=15,
                      occupancy_valid='one_pin_per_cell',
                      occupancy_invalid='one_cell_empty_one_cell_two_pins',
                      distance_note=('cross-cell distance is necessarily larger; '
                                     'area and edit process are matched exactly')),
        canvas=dict(size=[int(rec['canvas_shape'][1]), int(rec['canvas_shape'][0])],
                    map_size=[256, 256], units='full-resolution pixels'),
    )
    er = dict(edit_part=f'pushpin_cell_{cell_key[0]}_{cell_key[1]}',
              region_bbox_full=[int(v) for v in pin['bbox']],
              union_old_new=[int(min(pin['bbox'][0], vedit['new_bbox'][0], iedit['new_bbox'][0])),
                             int(min(pin['bbox'][1], vedit['new_bbox'][1], iedit['new_bbox'][1])),
                             int(max(pin['bbox'][2], vedit['new_bbox'][2], iedit['new_bbox'][2])),
                             int(max(pin['bbox'][3], vedit['new_bbox'][3], iedit['new_bbox'][3]))])
    return dict(valid_full=valid_full, invalid_full=invalid_full, tp=tp, edit_region=er,
                rule=('pp_incell', 'pp_crosscell'))


def pp_gen_pair_b(rec, source_id, pair_id):
    pins = rec['pins']
    cells = rec['cells']
    idx = int(source_id.split('-')[-1])
    vkey = ((idx + 1) % 3, (idx + 2) % 5)
    mkey = ((idx + 2) % 3, (idx + 3) % 5)
    vpin = next((p for p in pins if p['cell'] == vkey), None)
    mpin = next((p for p in pins if p['cell'] == mkey), None)
    if vpin is None or mpin is None:
        return None, ['pin_missing']
    vedit = pg.pp_move_within_cell(rec['full'], vpin['tight'], vpin['bbox'],
                                   cells[vkey], rng_of(pair_id))
    if vedit is None:
        return None, ['incell_infeasible']
    valid_full = pg.apply_rearrange(rec['full'], vpin['tight'], vpin['bbox'], vedit, 1)
    invalid_full = pg.apply_remove(rec['full'], mpin['tight'], mpin['bbox'])
    tp = dict(
        valid=dict(operation='translate_within_cell', source_cell=list(vkey),
                   pin_area_px=int(vpin['area']),
                   translation_px=[round(vedit['translation'][0], 1), round(vedit['translation'][1], 1)],
                   move_distance_px=round(vedit['distance'], 1)),
        invalid=dict(operation='remove_pin', source_cell=list(mkey),
                     removed_area_px=int(mpin['area']),
                     removed_bbox=[int(v) for v in mpin['bbox']]),
        controls=dict(total_pin_count_valid=15, total_pin_count_invalid=14,
                      composition_after_invalid='14 pins, one cell empty'),
        canvas=dict(size=[int(rec['canvas_shape'][1]), int(rec['canvas_shape'][0])],
                    map_size=[256, 256], units='full-resolution pixels'),
    )
    er = dict(edit_part_valid=f'pushpin_cell_{vkey[0]}_{vkey[1]}',
              edit_part_invalid=f'pushpin_cell_{mkey[0]}_{mkey[1]}',
              region_bbox_full=[int(v) for v in mpin['bbox']],
              valid_region_full=[int(min(vpin['bbox'][0], vedit['new_bbox'][0])),
                                 int(min(vpin['bbox'][1], vedit['new_bbox'][1])),
                                 int(max(vpin['bbox'][2], vedit['new_bbox'][2])),
                                 int(max(vpin['bbox'][3], vedit['new_bbox'][3]))])
    return dict(valid_full=valid_full, invalid_full=invalid_full, tp=tp, edit_region=er,
                rule=('pp_incell', 'pp_missing'))


# ---------------- verification ----------------

def verify_pair(entry, files):
    checks = []
    orig = np.array(Image.open(files['original']))
    val = np.array(Image.open(files['valid']))
    inv = np.array(Image.open(files['invalid']))
    for name, m in (('original', orig), ('valid', val), ('invalid', inv)):
        checks.append(dict(check='size_256', target=name, passed=m.shape == (256, 256),
                           detail=str(m.shape)))
    domain = ({0, 1, 2, 3, 4} if entry['category'] == 'screw_bag' else {0, 1, 2})
    ok_dom = set(np.unique(orig)) <= domain and set(np.unique(val)) <= domain and set(np.unique(inv)) <= domain
    checks.append(dict(check='label_domain', passed=bool(ok_dom),
                       detail=f'orig={sorted(set(orig.ravel().tolist()))}'))
    if entry['category'] == 'screw_bag':
        op = entry['transform_parameters']['invalid']['operation']
        n_o, n_v, n_i = int((orig > 0).sum()), int((val > 0).sum()), int((inv > 0).sum())
        if op == 'remove':
            cls_name = entry['transform_parameters']['invalid']['part_class']
            cls_id = {'long_screw': 1, 'short_screw': 2, 'nut': 3, 'washer': 4}[cls_name]
            others_ids = [v for k, v in {'long_screw': 1, 'short_screw': 2, 'nut': 3, 'washer': 4}.items()
                         if k != cls_name]
            d_cls = int((orig == cls_id).sum()) - int((inv == cls_id).sum())
            stable = all(abs(int((orig == oid).sum()) - int((inv == oid).sum())) <= 70
                         for oid in others_ids)
            checks.append(dict(check='invalid_removes_part', passed=d_cls > 60 and stable,
                               detail=f'{cls_name} {int((orig == cls_id).sum())}->{int((inv == cls_id).sum())}'))
        else:  # replace_short_screw_with_long_screw
            # long_screw pixels should increase, short_screw disappear
            lo, so = int((orig == 1).sum()), int((orig == 2).sum())
            li, si = int((inv == 1).sum()), int((inv == 2).sum())
            checks.append(dict(check='invalid_spec_substitution', passed=li > lo and si < so * 0.35,
                               detail=f'long {lo}->{li}, short {so}->{si}'))
        checks.append(dict(check='valid_keeps_area', passed=abs(n_v - n_o) <= max(60, n_o * 0.02),
                           detail=f'{n_o}->{n_v}'))
    else:
        pin_o, pin_v, pin_i = int((orig == 1).sum()), int((val == 1).sum()), int((inv == 1).sum())
        checks.append(dict(check='valid_pin_area', passed=abs(pin_v - pin_o) <= pin_o * 0.12,
                           detail=f'{pin_o}->{pin_v}'))
        checks.append(dict(check='invalid_pin_area', passed=abs(pin_i - pin_o) <= pin_o * 0.12,
                           detail=f'{pin_o}->{pin_i}'))
        sep_o, sep_v, sep_i = int((orig == 2).sum()), int((val == 2).sum()), int((inv == 2).sum())
        # moved pins may cover a small fraction of wall pixels (recorded in the manifest)
        checks.append(dict(check='separator_stable', passed=(sep_v >= sep_o * 0.85 and sep_v <= sep_o)
                                                 and (sep_i >= sep_o * 0.85 and sep_i <= sep_o),
                           detail=f'{sep_o}/{sep_v}/{sep_i}'))
    checks.append(dict(check='files_exist', passed=all(os.path.exists(f) for f in files.values())))
    return checks


# ---------------- main ----------------

def build_records():
    import pickle
    sb_records, sb_flags = {}, {}
    pp_records, pp_flags = {}, {}
    for pth in (CACHE, CACHE + '.part'):
        if os.path.exists(pth):
            try:
                with open(pth, 'rb') as f:
                    cache = pickle.load(f)
                sb_records.update(cache['sb'][0]); sb_flags.update(cache['sb'][1])
                pp_records.update(cache['pp'][0]); pp_flags.update(cache['pp'][1])
            except Exception:
                pass  # truncated checkpoint; ignore
    if os.path.exists(CACHE + '.part'):
        os.remove(CACHE + '.part')
    print(f'loaded cached records: sb={len(sb_records)}, pp={len(pp_records)}')
    ann = json.load(open(ANN_SB))['annotations']
    done = 0
    for split in ['train/good', 'validation/good']:
        for i in range(1000):
            rel = f'{split}/{i:03d}.png'
            if rel not in ann:
                continue
            if rel in sb_flags or rel in sb_records:
                continue
            rec, fl = sb_source_record(rel, ann[rel])
            if rec is None:
                sb_flags[rel] = fl
            else:
                sb_records[rel] = rec
            done += 1
            if done % 30 == 0:
                print(f'sb records {done}', flush=True)
    done = 0
    for cat_range in [('train/good', 0, 372)]:
        split, lo, hi = cat_range
        for i in range(lo, hi):
            rel = f'{split}/{i:03d}.png'
            path = f'{DATA_ROOT}/pushpins/{rel}'
            if not os.path.exists(path):
                continue
            if rel in pp_flags or rel in pp_records:
                continue
            rec, fl = pp_source_record(rel)
            if rec is None:
                pp_flags[rel] = fl
            else:
                pp_records[rel] = rec
            done += 1
            if done % 40 == 0:
                with open(CACHE + '.part', 'wb') as f:
                    pickle.dump(dict(sb=(sb_records, sb_flags), pp=(pp_records, pp_flags)), f, protocol=4)
                os.replace(CACHE + '.part', CACHE)
                print(f'pp records {done} (checkpoint)', flush=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(dict(sb=(sb_records, sb_flags), pp=(pp_records, pp_flags)), f, protocol=4)
    print('records built and cached')
    return (sb_records, sb_flags), (pp_records, pp_flags)


def main():
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    os.makedirs(OUT_ROOT)
    manifest = []
    source_status = {}
    stats = defaultdict(Counter)
    (sb_records, sb_flags), (pp_records, pp_flags) = build_records()
    print(f'screw_bag usable: {len(sb_records)} (flagged {len(sb_flags)})')
    print(f'pushpins usable: {len(pp_records)} (flagged {len(pp_flags)})')

    pair_specs = []
    TAGS = {'train': 'TR', 'development': 'DV', 'final_diagnostic': 'FD'}

    def pairs_for(cat, records, flags, purpose, ids):
        tag = TAGS[purpose]
        for i in ids:
            rel = f'train/good/{i:03d}.png'
            if rel not in records:
                reason = (['vlm_annotation_pending'] if cat == 'screw_bag' and rel not in flags
                          else list(flags.get(rel, ['not_available'])))
                source_status[f'{cat}/{rel}'] = dict(purpose=purpose, status='extraction_failed',
                                                     flags=reason)
                stats[cat]['sources_failed'] += 1
                continue
            source_status[f'{cat}/{rel}'] = dict(purpose=purpose, status='ok')
            stats[cat]['sources_ok'] += 1
            sid = f'{"sb" if cat == "screw_bag" else "pp"}-train-{i:03d}'
            genA = sb_gen_pair_a if cat == 'screw_bag' else pp_gen_pair_a
            genB = sb_gen_pair_b if cat == 'screw_bag' else pp_gen_pair_b
            pair_specs.append((cat, rel, sid, f'{"SB" if cat == "screw_bag" else "PP"}-{tag}-{i:03d}-A',
                               purpose, 'A', genA))
            if purpose == 'train' and i % 3 == 0:
                pair_specs.append((cat, rel, sid,
                                   f'{"SB" if cat == "screw_bag" else "PP"}-TR-{i:03d}-B',
                                   purpose, 'B', genB))

    official = load_official_split()
    for cat in ['screw_bag', 'pushpins']:
        for purpose in ['train', 'development', 'final_diagnostic']:
            records = sb_records if cat == 'screw_bag' else pp_records
            flags = sb_flags if cat == 'screw_bag' else pp_flags
            pairs_for(cat, records, flags, purpose, purpose_ids(official, cat, purpose))
    print(f'pair specs: {len(pair_specs)}')

    qa_all = {}
    for cat, rel, sid, pid, purpose, ptag, gen in pair_specs:
        rec = sb_records[rel] if cat == 'screw_bag' else pp_records[rel]
        res = gen(rec, sid, pid)
        if not isinstance(res, dict):
            fl = res[1] if isinstance(res, tuple) else ['generation_failed']
            stats[cat][f'pairs_failed_{ptag}'] += 1
            source_status[f'{cat}/{rel}'].setdefault('pair_failures', []).append(
                dict(pair_id=pid, flags=fl))
            continue
        split_dir = 'train' if 'train/good' in rel else 'validation'
        idx = rel.split('/')[-1].replace('.png', '')
        orig_dir = f'{OUT_ROOT}/data/composition_maps/{cat}/{split_dir}'
        val_dir = f'{OUT_ROOT}/data/edits/{cat}/valid'
        inv_dir = f'{OUT_ROOT}/data/edits/{cat}/invalid'
        for d in (orig_dir, val_dir, inv_dir):
            os.makedirs(d, exist_ok=True)
        orig_path = f'{orig_dir}/{idx}.png'
        vpath = f'{val_dir}/{pid}.png'
        ipath = f'{inv_dir}/{pid}.png'
        if not os.path.exists(orig_path):
            save_map_png(rec['map'], orig_path)
        save_map_png(pg.resize_map(res['valid_full']), vpath)
        save_map_png(pg.resize_map(res['invalid_full']), ipath)
        rule_v, rule_i = res['rule']
        entry = dict(
            pair_id=pid,
            source_id=sid,
            source_path=f'{cat}/{rel}',
            category=cat,
            source_split='train' if 'train/good' in rel else 'validation',
            original_path=f'data/composition_maps/{cat}/{split_dir}/{idx}.png',
            valid_path=f'data/edits/{cat}/valid/{pid}.png',
            invalid_path=f'data/edits/{cat}/invalid/{pid}.png',
            transform_parameters=res['tp'],
            edit_region=res['edit_region'],
            rule_reference=dict(valid=RULE_REF[rule_v], invalid=RULE_REF[rule_i]),
            review_status='pending',
            review_reason=None,
            representation='composition_map',
            purpose=purpose,
            generator=dict(name='D_pair_generator', version='1.1.0',
                           determinism='crc32(pair_id)-ordered candidate grid'),
            file_sha256=dict(original=sha256_file(orig_path), valid=sha256_file(vpath),
                             invalid=sha256_file(ipath)),
        )
        manifest.append(entry)
        files = dict(original=orig_path, valid=vpath, invalid=ipath)
        checks = verify_pair(entry, files)
        qa_all[pid] = dict(checks=checks, all_passed=all(c['passed'] for c in checks))
        stats[cat][f'pairs_ok_{ptag}'] += 1

    os.makedirs(f'{OUT_ROOT}/qa', exist_ok=True)
    json.dump(manifest, open(f'{OUT_ROOT}/pair_manifest.json', 'w'), indent=1)
    json.dump(source_status, open(f'{OUT_ROOT}/source_status.json', 'w'), indent=1)
    json.dump(qa_all, open(f'{OUT_ROOT}/qa/qa_report.json', 'w'), indent=1)
    json.dump({k: dict(v) for k, v in stats.items()}, open(f'{OUT_ROOT}/qa/stats.json', 'w'), indent=1)
    print('manifest pairs:', len(manifest))
    print('stats:', {k: dict(v) for k, v in stats.items()})
    nfail = sum(1 for v in qa_all.values() if not v['all_passed'])
    print('pairs with failed checks:', nfail)


if __name__ == '__main__':
    main()
