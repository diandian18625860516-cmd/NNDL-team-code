#!/usr/bin/env python3
"""VLM annotation runner for screw_bag normal images (chunked, resumable).

Workers (daemon threads) pull image keys from a queue, call `z-ai vision`
(subprocess) with a robust JSON prompt, parse part bboxes+types, and append
to a checkpoint JSON. Safe to stop at any time; rerun resumes.

v2 note: pair sources now follow B's official split manifest, which assigns
ALL edit-source roles (train_edit / dev / final_diag) to train/good images.
Therefore annotation now covers train/good 000-359 only (validation/good
has no assigned role and is NOT annotated).

Usage: python3 vlm_annotate.py [--budget SEC] [--limit N]
"""
import json
import os
import subprocess
import sys
import threading
import time
from PIL import Image

DATA = '/home/z/my-project/workspace/data/mvtec_loco/screw_bag'
OUT = '/home/z/my-project/workspace/vlm_annotations_sb_v2.json'
TMP = '/home/z/my-project/workspace/vlm_tmp'
os.makedirs(TMP, exist_ok=True)

PROMPT = ("这张图是 MVTec LOCO AD screw_bag 数据集的一张正常图：透明自封袋里有恰好6个金属件："
          "1根长螺丝、1根短螺丝、2个六角螺母、2个圆形垫圈（圆环）。"
          "请只输出JSON，格式：{\"parts\":[{\"type\":\"long_screw|short_screw|nut|washer\","
          "\"x1\":int,\"y1\":int,\"x2\":int,\"y2\":int}]}。坐标用原图像素（1600宽x1100高）。"
          "注意：长螺丝是细长杆约300-460像素长；短螺丝约260-320像素；螺母是实心六角形约110-130像素；"
          "垫圈是带大中心孔的圆环约110-140像素。部件可能互相部分重叠，仍分别给出各自边界框。只输出JSON。")

LOCK = threading.Lock()
STOP = threading.Event()


def image_files():
    files = []
    d = f'{DATA}/train/good'          # v2: only train/good carries B-assigned roles
    for f in sorted(os.listdir(d)):
        if f.endswith('.png'):
            files.append(f'train/good/{f}')
    return files


def parse_parts(content):
    s = content.strip()
    if s.startswith('```'):
        s = s.split('```')[1].lstrip('json').strip()
    s = s[s.find('{'):s.rfind('}') + 1]
    parts = json.loads(s)['parts']
    out = []
    for p in parts:
        t = p.get('type')
        if t not in ('long_screw', 'short_screw', 'nut', 'washer'):
            continue
        x1, y1, x2, y2 = int(p['x1']), int(p['y1']), int(p['x2']), int(p['y2'])
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(1599, x2), min(1099, y2)
        if x2 - x1 > 20 and y2 - y1 > 20:
            out.append(dict(type=t, bbox=[x1, y1, x2, y2]))
    return out


def annotate(rel):
    img_path = f'{DATA}/{rel}'
    jpg = f'{TMP}/{rel.replace("/", "_")}.jpg'
    if not os.path.exists(jpg):
        Image.open(img_path).convert('RGB').save(jpg, quality=85)
    last = 'unknown'
    for attempt in range(8):
        if STOP.is_set():
            return None, 'stopped'
        out = f'{TMP}/resp_{threading.get_ident()}_{attempt % 3}.json'
        if os.path.exists(out):
            os.remove(out)
        try:
            r = subprocess.run(['z-ai', 'vision', '-p', PROMPT, '-i', jpg, '-o', out],
                               capture_output=True, timeout=120)
            if os.path.exists(out):
                d = json.load(open(out))
                parts = parse_parts(d['choices'][0]['message']['content'])
                if parts:
                    time.sleep(1.2)
                    return parts, None
            err = r.stderr.decode()[:160]
            last = f'bad_output:{err}'
            # rate-limited: long backoff; otherwise short
            if '429' in err:
                time.sleep(45 + 15 * attempt)
            else:
                time.sleep(2 + attempt * 2)
        except Exception as e:
            last = str(e)[:120]
            time.sleep(3)
    return None, last


def load_state():
    if os.path.exists(OUT):
        d = json.load(open(OUT))
        return d.get('annotations', {}), d.get('errors', {})
    return {}, {}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=1e9)
    ap.add_argument('--limit', type=int, default=1e9)
    args = ap.parse_args()
    files = image_files()
    results, errors = load_state()
    todo = [f for f in files if f not in results]  # errored ones are retried each run
    todo.sort()                                   # plain filename order (all train/good)
    print(f'total {len(files)}, done {len(results)}, err {len(errors)}, todo {len(todo)}', flush=True)
    if not todo:
        return
    import queue
    q = queue.Queue()
    for rel in todo:
        q.put(rel)
    n_done = [0]
    t0 = time.time()

    def save():
        with LOCK:
            json.dump({'annotations': results, 'errors': errors}, open(OUT, 'w'),
                      ensure_ascii=False, indent=0)

    def worker():
        while not STOP.is_set():
            try:
                rel = q.get_nowait()
            except queue.Empty:
                return
            parts, err = annotate(rel)
            if STOP.is_set() and parts is None:
                q.put(rel)
                return
            with LOCK:
                if parts:
                    results[rel] = parts
                else:
                    errors[rel] = err
                n_done[0] += 1
            if n_done[0] % 5 == 0:
                save()
                el = time.time() - t0
                print(f'[{n_done[0]}/{len(todo)}] ok={len(results)} err={len(errors)} el={el:.0f}s', flush=True)
            time.sleep(1.0)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(1)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        if time.time() - t0 > args.budget or n_done[0] >= args.limit:
            STOP.set()
        time.sleep(2)
    save()
    print(f'stop ok={len(results)} err={len(errors)}', flush=True)


if __name__ == '__main__':
    main()
