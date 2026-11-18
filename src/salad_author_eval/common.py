"""Small shared readers; no model, image loading, hashing or training side effects."""

import csv
import json
import math
from pathlib import Path

CATEGORIES = ("breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors")
GROUPS = {"good": 0, "structural_anomalies": 1, "logical_anomalies": 1}
STAGES = ("author_weights", "from_scratch", "continued_training", "short_validation", "memory_bank_fit")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required) <= set(fields):
            raise ValueError(f"Invalid CSV header; required: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty CSV")
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Malformed CSV row")
    return rows


def relative_path(value):
    value = str(value).replace("\\", "/")
    parts = value.split("/")
    if len(parts) < 2 or any(p in ("", ".", "..") for p in parts) or ":" in value:
        raise ValueError(f"Expected qualified relative path, not basename/absolute path: {value}")
    return value


def finite(value):
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"Nonfinite score: {value}")
    return score


def metadata(path):
    data = read_json(path)
    required = ("run_id", "model", "category", "stage", "score_direction", "score_name")
    if any(not isinstance(data.get(k), str) or not data[k].strip() for k in required):
        raise ValueError(f"Metadata requires nonempty strings: {required}")
    if data["category"] not in CATEGORIES or data["stage"] not in STAGES:
        raise ValueError("Unknown category or stage")
    if data["score_direction"] not in ("higher_is_anomalous", "lower_is_anomalous"):
        raise ValueError("Unknown score direction")
    return data


def read_predictions(path, meta, image_labels=False):
    required = {"image_path", "score"}
    if image_labels:
        required |= {"anomaly_type", "label"}
    rows = read_csv(path, required)
    result = {}
    sign = 1 if meta["score_direction"] == "higher_is_anomalous" else -1
    has_run_id = "run_id" in rows[0]
    for row in rows:
        if has_run_id and row["run_id"] != meta["run_id"]:
            raise ValueError("Prediction run_id differs from metadata")
        key = relative_path(row["image_path"])
        if key in result:
            raise ValueError(f"Duplicate prediction: {key}")
        if key.split("/")[0] != meta["category"]:
            raise ValueError(f"Prediction category mismatch: {key}")
        raw = finite(row["score"])
        result[key] = {**row, "image_path": key, "raw_score": raw, "score": sign * raw}
    return result


def check_coverage(actual, expected):
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"Coverage mismatch: missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}")


def write_outputs(directory, summary, tables):
    # A new directory is required: neither input evidence nor previous outputs are overwritten.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, rows in tables.items():
        if not rows:
            continue
        with (directory / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def provenance(meta, predictions, manifest):
    return {"schema_version": 1, "run": meta,
            "prediction_file": str(Path(predictions).resolve()),
            "manifest_file": str(Path(manifest).resolve()),
            "score_transform": "identity" if meta["score_direction"] == "higher_is_anomalous" else "negate",
            "evaluated_score_direction": "higher_is_anomalous"}
