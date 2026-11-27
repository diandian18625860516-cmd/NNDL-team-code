"""Recompute the five-category report on CPU from the archived per-image scores."""
import csv
from math import isclose
from pathlib import Path
from statistics import mean

from common import CATEGORIES, metadata
from image_level import evaluate

REPORT = Path(__file__).resolve().parents[2] / "reports/salad_author_five"

def main():
    rows = []
    for cat in CATEGORIES:
        run = REPORT / cat
        result, _ = evaluate(run/"predictions.csv", REPORT/"manifest.csv", metadata(run/"run.json"))
        row = {"test": result["prediction_count"], **{k:100*v["auroc"] for k,v in result["metrics"].items()}}
        with (run/"results.csv").open(encoding="utf-8-sig") as f:
            assert isclose(row["macro"],float(next(csv.DictReader(f))["AUC"]),abs_tol=1e-8)
        rows.append((cat,row))
    rows.append(("five_category_mean", {k:(sum if k=="test" else mean)(r[k] for _,r in rows) for k in rows[0][1]}))
    with (REPORT/"summary_percent.csv").open(encoding="utf-8-sig") as f:
        expected={r["category"]:r for r in csv.DictReader(f)}
    assert set(expected)=={cat for cat,_ in rows}
    print("category               images  logical  structural   macro  pooled (%)")
    for cat,r in rows:
        assert all(isclose(v,float(expected[cat][k]),abs_tol=1e-8) for k,v in r.items())
        print(f"{cat:22} {r['test']:5}  {r['logical']:7.2f}  {r['structural']:9.2f}  {r['macro']:6.2f}  {r['pooled']:6.2f}")

if __name__=="__main__":
    main()
