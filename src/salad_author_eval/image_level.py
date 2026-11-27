"""LOCO image AUROC, preserving the legacy sklearn algorithm and group definitions."""

from pathlib import Path
from sklearn.metrics import roc_auc_score

from common import GROUPS, check_coverage, read_csv, read_json, read_predictions, relative_path


def identity(row, category):
    path = relative_path(row["image_path"])
    group = row["anomaly_type"]
    if group not in GROUPS or str(row["label"]) not in ("0", "1"):
        raise ValueError(f"Unknown anomaly_type or nonbinary label: {path}")
    label = int(row["label"])
    if label != GROUPS[group]:
        raise ValueError(f"Label/anomaly_type conflict: {path}")
    parts = path.split("/")
    if len(parts) != 4 or parts[:3] != [category, "test", group]:
        raise ValueError(f"Expected category/test/anomaly_type/image: {path}")
    if "split" in row and row["split"] != "test":
        raise ValueError(f"Expected test split: {path}")
    if "category" in row and row["category"] != category:
        raise ValueError(f"Manifest category mismatch: {path}")
    return path, group, label


def test_manifest(path, category):
    if Path(path).suffix.lower() == ".json":
        # Legacy data_manifest.json contains audits for all three splits.
        rows = read_json(path)["test"]["images"]
    else:
        all_rows = read_csv(path, {"image_path", "category", "split", "anomaly_type", "label"})
        rows = [r for r in all_rows if r["category"] == category and r["split"] == "test"]
    expected = {}
    for row in rows:
        key, group, label = identity(row, category)
        if key in expected:
            raise ValueError(f"Duplicate manifest entry: {key}")
        expected[key] = (group, label)
    if not expected:
        raise ValueError("Empty test manifest for category")
    return expected


def image_metrics(records):
    result = {}
    for name, groups in {
        "pooled": set(GROUPS),
        "structural": {"good", "structural_anomalies"},
        "logical": {"good", "logical_anomalies"},
    }.items():
        selected = [r for r in records if r["anomaly_type"] in groups]
        labels = [r["label"] for r in selected]
        if set(labels) != {0, 1}:
            raise ValueError(f"Both normal and anomalous examples required for {name}")
        result[name] = {"auroc": float(roc_auc_score(labels, [r["score"] for r in selected])),
                        "normal_count": labels.count(0), "anomaly_count": labels.count(1),
                        "total_count": len(labels)}
    result["macro"] = {"auroc": (result["structural"]["auroc"] + result["logical"]["auroc"]) / 2,
                       "component_count": 2, "components": ["structural", "logical"]}
    return result


def evaluate(prediction_path, manifest_path, meta):
    predictions = read_predictions(prediction_path, meta, image_labels=True)
    expected = test_manifest(manifest_path, meta["category"])
    check_coverage(predictions, expected)
    for row in predictions.values():
        key, group, label = identity(row, meta["category"])
        if expected[key] != (group, label):
            raise ValueError(f"Prediction/manifest label conflict: {key}")
        row["label"] = label
    metrics = image_metrics(list(predictions.values()))
    summary = {"evaluation": "official_test_image_level", "coverage": "exact_manifest_match",
               "counts": {g: sum(r["anomaly_type"] == g for r in predictions.values()) for g in GROUPS},
               "prediction_count": len(predictions), "metrics": metrics,
               "legacy_name_mapping": {"overall": "pooled"}}
    table = [{"run_id": meta["run_id"], "model": meta["model"], "category": meta["category"],
              "stage": meta["stage"], "score_name": meta["score_name"],
              "metric": name, "auroc": value["auroc"],
              "normal_count": value.get("normal_count", ""),
              "anomaly_count": value.get("anomaly_count", ""),
              "total_count": value.get("total_count", ""),
              "component_count": value.get("component_count", "")}
             for name, value in metrics.items()]
    return summary, {"summary.csv": table}
