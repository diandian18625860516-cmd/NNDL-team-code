"""Stage verified provenance, run real GPU inference, then evaluate the new CSV."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / 'src/salad_author_eval'
REPORT = ROOT / 'reports/salad_author_five'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--category', default='pushpins', choices=sorted(x.name for x in (HERE/'weight_metadata').iterdir() if x.is_dir()))
    for key in ('archive','rgb','maps','weights','out'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): p.error('--out must be a new directory')
    cat = a.category
    source = HERE/'weight_metadata'/cat/'SOURCE.json'
    record = json.loads(source.read_text('utf-8'))
    for member in record['members']:
        f = a.weights/member['path']
        if not f.is_file() or f.stat().st_size != member['bytes']:
            p.error(f'Missing or wrong size: {f}')
        crc = 0
        with f.open('rb') as stream:
            while chunk := stream.read(4*1024**2):
                crc = zlib.crc32(chunk,crc)
        if f'{crc:08x}' != member['zip_crc32']: p.error(f'Checksum mismatch: {f}')
    dst = a.weights/cat/'SOURCE.json'
    if dst.exists():
        if json.loads(dst.read_text('utf-8-sig')) != record: p.error(f'Conflicting source record: {dst}')
    else:
        shutil.copyfile(source,dst)
    env = {**os.environ,'PYTHONUTF8':'1','OPENBLAS_NUM_THREADS':'1'}
    subprocess.run([sys.executable,'-B','-u',SRC/'run_salad_author.py','--category',cat,
                    '--archive',a.archive,'--rgb',a.rgb,'--maps',a.maps,'--weights',a.weights,'--out',a.out],
                   env=env,check=True)
    sys.path.insert(0,str(SRC))
    from common import metadata
    from image_level import evaluate
    result,_ = evaluate(a.out/'predictions.csv',REPORT/'manifest.csv',metadata(a.out/'run.json'))
    (a.out/'demo_metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f"{result['prediction_count']} official test images; macro AUROC = {100*result['metrics']['macro']['auroc']:.4f}%")

if __name__=='__main__':
    main()
