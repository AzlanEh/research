import csv
import json
import shutil
from pathlib import Path
from collections import Counter

import numpy as np

DATASET_ROOT = Path("/mnt/external_disk/gaurav_project/wifi_csi")
ANNOTATION = DATASET_ROOT.parent / "annotation.csv"
AMP_DIR = DATASET_ROOT / "amp"

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "data" / "metadata" / "wimans"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "wimans"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

samples = []
lengths = []
errors = []

with ANNOTATION.open(newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    label = row["label"].strip()
    csi_path = AMP_DIR / f"{label}.npy"

    if not csi_path.is_file():
        errors.append(f"Missing CSI file: {label}")
        continue

    try:
        # Memory-map: inspect array metadata without loading 33 GB into RAM.
        arr = np.load(csi_path, mmap_mode="r")
        shape = list(arr.shape)
        dtype = str(arr.dtype)
    except Exception as exc:
        errors.append(f"Cannot read {label}: {exc}")
        continue

    if len(shape) != 4 or shape[1:] != [3, 3, 30]:
        errors.append(f"Unexpected shape for {label}: {shape}")
        continue

    users = []
    for i in range(1, 7):
        location = row[f"user_{i}_location"].strip()
        activity = row[f"user_{i}_activity"].strip()

        if location and activity:
            users.append({
                "slot": i,
                "location": location,
                "activity": activity
            })

    group_id, sample_index = label.split("_")[1:]

    sample = {
        "sample_id": label,
        "group_id": int(group_id),
        "sample_index": int(sample_index),
        "environment": row["environment"].strip(),
        "wifi_band": row["wifi_band"].strip(),
        "number_of_users": int(row["number_of_users"]),
        "users": users,
        "csi_path": str(csi_path),
        "shape": shape,
        "dtype": dtype,
        "num_timesteps": shape[0]
    }

    samples.append(sample)
    lengths.append(shape[0])

# Save one JSON object per sample.
manifest_path = PROCESSED_DIR / "manifest.jsonl"
with manifest_path.open("w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

# Preserve a copy of the source annotations.
shutil.copy2(ANNOTATION, OUTPUT_DIR / "annotation.csv")

report = {
    "dataset": "WiMANS",
    "dataset_root": str(DATASET_ROOT),
    "annotation_file": str(ANNOTATION),
    "total_annotation_rows": len(rows),
    "indexed_samples": len(samples),
    "errors": errors,
    "groups": len({s["group_id"] for s in samples}),
    "environment_distribution": dict(
        Counter(s["environment"] for s in samples)
    ),
    "wifi_band_distribution": dict(
        Counter(s["wifi_band"] for s in samples)
    ),
    "user_count_distribution": dict(
        Counter(str(s["number_of_users"]) for s in samples)
    ),
    "activity_distribution": dict(
        Counter(u["activity"] for s in samples for u in s["users"])
    ),
    "sequence_length": {
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
        "mean": round(sum(lengths) / len(lengths), 2) if lengths else None
    }
}

report_path = OUTPUT_DIR / "manifest_report.json"
report_path.write_text(json.dumps(report, indent=2))

print("========== WiMANS MANIFEST ==========")
print("Annotations:", len(rows))
print("Indexed samples:", len(samples))
print("Groups:", report["groups"])
print("Errors:", len(errors))
print("Sequence lengths:", report["sequence_length"])
print("Manifest:", manifest_path)
print("Report:", report_path)

if errors:
    print("\nFirst errors:")
    for error in errors[:20]:
        print("-", error)

if len(samples) != len(rows) or errors:
    raise SystemExit("Manifest build failed validation.")

print("\nManifest built successfully.")
