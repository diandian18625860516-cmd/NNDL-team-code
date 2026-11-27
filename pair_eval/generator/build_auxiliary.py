#!/usr/bin/env python3
"""Build D's auxiliary deliverables:
 - qa/contact_sheets/*.jpg   (paged pair overviews for B's review)
 - qa/triplets/*.png         (doc figures: RGB original + map original/valid/invalid)
 - review_workbook.csv      (one row per pair with review fields for B)
 - r1_r2_sample_list.json    (fixed violation-sample list for the ranking-control
                              experiment R1 vs R2 — identical inputs, one switch)
 - integration/run_template_pair_eval.json (E's paired-run JSON template)
 - qa/stats.md               (human-readable statistics)
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/z/my-project/workspace/D_deliverable'
DATA_ROOT = '/home/z/my-project/workspace/data/mvtec_loco'
MAN = json.load(open(f'{ROOT}/pair_manifest.json'))
QA = json.load(open(f'{ROOT}/qa/qa_report.json'))
SRC = json.load(open(f'{ROOT}/source_status.json'))

PALETTE = np.array([[20, 20, 20], [255, 60, 60], [255, 170, 60],
                    [60, 120, 255], [60, 255, 120], [160, 60, 200]], np.uint8)
CLASS_CN = {'screw_bag': {0: '背景/袋', 1: '长螺丝', 2: '短螺丝', 3: '螺母', 4: '垫圈'},
            'pushpins': {0: '背景', 1: '图钉', 2: '隔板'}}


def font(sz=13):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', sz)
    except Exception:
        return ImageFont.load_default()


def colorize(m):
    return Image.fromarray(PALETTE[m.astype(np.uint8)])


def contact_sheets():
    out = f'{ROOT}/qa/contact_sheets'
    os.makedirs(out, exist_ok=True)
    CELL, LH = 200, 20
    bygroup = defaultdict(list)
    for e in MAN:
        bygroup[(e['category'], e['purpose'])].append(e)
    n_files = 0
    for (cat, purpose), entries in sorted(bygroup.items()):
        entries = sorted(entries, key=lambda e: e['pair_id'])
        pages = [entries[i:i + 6] for i in range(0, len(entries), 6)]
        for pi, page in enumerate(pages, 1):
            W = CELL * 3 + 30
            H = (CELL + LH + 8) * len(page)
            canvas = Image.new('RGB', (W, H), (255, 255, 255))
            d = ImageDraw.Draw(canvas)
            for r, e in enumerate(page):
                y0 = r * (CELL + LH + 8)
                d.text((4, y0 + 3), f"{e['pair_id']}  {e['purpose']}  "
                        f"valid={e['transform_parameters']['valid']['operation']}  "
                        f"invalid={e['transform_parameters']['invalid']['operation']}",
                        fill=(0, 0, 0), font=font(13))
                for c, (k, lab) in enumerate([('original_path', 'original'),
                                              ('valid_path', 'VALID'),
                                              ('invalid_path', 'INVALID')]):
                    m = np.array(Image.open(f"{ROOT}/{e[k]}"))
                    img = colorize(m).resize((CELL, CELL), Image.NEAREST)
                    canvas.paste(img, (c * (CELL + 15), y0 + LH))
            fn = f'{out}/{cat}_{purpose}_page{pi:02d}.jpg'
            canvas.save(fn, quality=80)
            n_files += 1
    return n_files


def triplets():
    """RGB original + map original/valid/invalid figures (3 per category/purpose)."""
    out = f'{ROOT}/qa/triplets'
    os.makedirs(out, exist_ok=True)
    made = 0
    for cat in ['screw_bag', 'pushpins']:
        for purpose in ['train', 'development', 'final_diagnostic']:
            ents = sorted([e for e in MAN if e['category'] == cat and e['purpose'] == purpose],
                          key=lambda e: e['pair_id'])[:1]
            for e in ents:
                rgb = Image.open(f"{DATA_ROOT}/{e['source_path']}").convert('RGB')
                rgb_small = rgb.resize((460, int(460 * rgb.height / rgb.width)))
                maps = []
                for k in ['original_path', 'valid_path', 'invalid_path']:
                    m = np.array(Image.open(f"{ROOT}/{e[k]}"))
                    maps.append(colorize(m).resize((460, 460), Image.NEAREST))
                W = 460 * 4 + 60
                H = max(rgb_small.height, 460) + 30
                canvas = Image.new('RGB', (W, H), (255, 255, 255))
                d = ImageDraw.Draw(canvas)
                canvas.paste(rgb_small, (0, 30))
                d.text((4, 6), f"{cat} / {e['source_path']} / {e['pair_id']} ({purpose})", fill=(0, 0, 0), font=font(14))
                labs = ['RGB 原图 (source)', '组成图 original', '组成图 VALID (合法变化)', '组成图 INVALID (违规变化)']
                for i, (im, lab) in enumerate(zip([rgb_small] + maps, labs)):
                    canvas.paste(im, (i * (460 + 20), 30))
                    d.text((i * (460 + 20) + 4, 486 + 30 if False else H - 22), lab, fill=(0, 0, 0), font=font(13))
                fn = f'{out}/{cat}_{purpose}_{e["pair_id"]}.png'
                canvas.save(fn)
                made += 1
    return made


def workbook():
    import csv
    with open(f'{ROOT}/review_workbook.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['pair_id', 'category', 'purpose', 'pair_kind', 'source_path',
                    'valid_op', 'invalid_op', 'self_check', 'review_status',
                    'review_reason_B', 'reviewer', 'date'])
        for e in sorted(MAN, key=lambda e: e['pair_id']):
            qa = QA.get(e['pair_id'], {})
            w.writerow([e['pair_id'], e['category'], e['purpose'],
                        e['pair_id'][-1], e['source_path'],
                        e['transform_parameters']['valid']['operation'],
                        e['transform_parameters']['invalid']['operation'],
                        'PASS' if qa.get('all_passed') else 'FAIL',
                        e['review_status'], '', '', ''])


def r1_r2_list():
    """Fixed violation-sample list for the ranking-control experiment.
    R1 (no ranking) and R2 (ranking on) must consume EXACTLY this list."""
    entries = []
    for e in MAN:
        if e['category'] == 'screw_bag' and e['purpose'] == 'train':
            entries.append(dict(pair_id=e['pair_id'], source_id=e['source_id'],
                                source_path=e['source_path'],
                                invalid_path=e['invalid_path'],
                                invalid_op=e['transform_parameters']['invalid']['operation'],
                                original_path=e['original_path'],
                                sha256=e['file_sha256']['invalid']))
    json.dump(dict(
        description=('R1/R2 排序控制实验的固定违规样本清单：R1(不加排序) 与 R2(加排序) 必须消费'
                     '完全相同的这批违规样本（含原始图与编辑参数），两组配置只差排序损失开关。'
                     '样本即 pair_manifest.json 中 screw_bag/train 用途对的 invalid 输入'
                     '（v2：train 用途按 B 官方划分清单逐图指定）。'),
        n_pairs=len(entries),
        entries=entries), open(f'{ROOT}/r1_r2_sample_list.json', 'w'), indent=1)
    return len(entries)


def run_template():
    tpl = {
        "_comment": ("E 的配对运行 JSON 模板（按 B/D/E→C 交付约定 v1）。同一文件同时携带"
                     "通用六字段与配对评价必需字段；null 表示待 E 填写，不能凭默认值补成已知。"),
        "run_id": None,
        "model": "SALAD",
        "category": "screw_bag",
        "stage": None,
        "score_name": "composition_branch_image_score",
        "score_direction": "higher_is_anomalous",
        "initialization": None,
        "code_version": None,
        "seed": None,
        "data_split": "D_deliverable/source_status.json (purpose assignment)",
        "preprocessing": "composition map 256x256, single channel integer labels",
        "steps": None,
        "sample_budget": None,
        "score_definition": None,
        "branch_scores": None,
        "map_location": None,
        "original_results": None,
        "representation": "composition_map",
        "score_branch": "composition",
        "model_seen_source_paths": [
            "_REQUIRED: 起点检查点训练中见过的完整原始 RGB 路径数组；"
            "v2 起全部配对源图都在 train/good（SALAD train 划分，必然被见过），"
            "必须完整列入；SALAD 官方训练同时使用 validation 划分，若见过也须列入。"
        ],
        "seen_sources_complete": True,
        "protocol_frozen": None,
        "pair_manifest": "pair_manifest.json",
        "pair_selection": {"category": "screw_bag", "purpose": "final_diagnostic",
                           "review_status": "approved"}
    }
    os.makedirs(f'{ROOT}/integration', exist_ok=True)
    json.dump(tpl, open(f'{ROOT}/integration/run_template_pair_eval.json', 'w'),
              indent=2, ensure_ascii=False)


def stats_md():
    lines = []
    per = defaultdict(Counter)
    for e in MAN:
        per[(e['category'], e['purpose'])][e['pair_id'][-1]] += 1
    src_per = defaultdict(Counter)
    for k, v in SRC.items():
        cat = k.split('/')[0]
        src_per[(cat, v['purpose'])][v['status']] += 1
    lines.append('# D 交付统计 (pair generation v1.1.0 · 用途划分 = B 官方清单 v2)\n')
    lines.append('## 配对 (pair) 数量\n')
    lines.append('| 类别 | 用途 | A 对 (重摆/移格 vs 违规) | B 对 (替换/缺件) | 合计 |')
    lines.append('|---|---|---|---|---|')
    for (cat, pur), c in sorted(per.items()):
        a = c['A']; b = c['B']
        lines.append(f'| {cat} | {pur} | {a} | {b} | {a+b} ')
    lines.append(f"| **合计** | | | | **{len(MAN)}** |")
    lines.append('\n## 源图 (source) 状态\n')
    lines.append('| 类别 | 用途 | 可用 | 失败/未标注 |')
    lines.append('|---|---|---|---|')
    for (cat, pur), c in sorted(src_per.items()):
        lines.append(f"| {cat} | {pur} | {c['ok']} | {c['extraction_failed']} |")
    nfail = sum(1 for v in QA.values() if not v['all_passed'])
    lines.append(f'\n- 自动校验 (QA)：{len(MAN)} 对中 {len(MAN)-nfail} 对全部通过，'
                 f'{nfail} 对存在未通过项（见 qa_report.json）。\n'
                 f'- 审核状态：全部为 pending，等待 B 验收后更新为 approved/rejected。\n'
                 f'- 统计单位：同时报告源图数与编辑对数；一张源图最多派生 2 对（A/B），'
                 f'不把变体当作独立源图计数。')
    open(f'{ROOT}/qa/stats.md', 'w').write('\n'.join(lines))
    return len(lines)


if __name__ == '__main__':
    n1 = contact_sheets()
    n2 = triplets()
    workbook()
    n3 = r1_r2_list()
    run_template()
    stats_md()
    print(f'contact sheets: {n1}; triplets: {n2}; r1_r2 entries: {n3}')
