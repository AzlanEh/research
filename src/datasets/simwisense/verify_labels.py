"""verify_labels.py

Verify label consistency across the SiMWiSense train/val/test CSVs.

Checks:
  1. Every filename listed in a CSV exists on disk.
  2. The CSV label matches the parent batch folder (e.g.
     Slots/Train/A_batch/batch_0.mat -> label "A").
  3. The set of unique labels matches the expected class set (A, B, C).
  4. No empty labels or malformed rows.

Labels are treated as provisional single letters (A, B, C). No
human-readable activity name is assumed or assigned.

Designed to run from `src/`:
    python dataset/verify_labels.py

Outputs a verification report to
    ../data/metadata/label_verification_report.json

Read-only with respect to the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense"
    "/Data/fine_grained/Classroom/80MHz/3mo/m1"
)
DEFAULT_ACTIVITIES = ["A", "B", "C"]
DEFAULT_SPLITS = ["Train", "Val", "Test"]

CSV_FILENAMES = {
    "Train": "train_set.csv",
    "Val": "val_set.csv",
    "Test": "test_set.csv",
}

# The val_set.csv lives alongside train_set.csv in Slots/Train/ (val is a
# random subset carved out of Train, not a separate batch directory).
SPLIT_CSV_DIR = {
    "Train": "Train",
    "Val": "Train",
    "Test": "Test",
}


def repo_root_from_src() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.dirname(here)
    src_dir = os.path.dirname(datasets_dir)
    return os.path.dirname(src_dir)


def csv_path(dataset_root: str, split: str) -> str:
    return os.path.join(
        dataset_root, "Slots", SPLIT_CSV_DIR[split], CSV_FILENAMES[split]
    )


def load_csv_rows(csv_path: str) -> list[dict]:
    rows = []
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def expected_label_from_filename(filename: str) -> str | None:
    """Derive the expected label from a relative filename like
    'Slots/Train/A_batch/batch_0.mat' -> 'A'.

    Returns None if the pattern cannot be parsed.
    """
    parts = filename.replace("\\", "/").split("/")
    # Expect: [..., Slots, Split, X_batch, file.mat]
    for part in parts:
        if part.endswith("_batch") and len(part) > len("_batch"):
            return part[-len("_batch") - 1 : -len("_batch")]
    return None


def verify(dataset_root: str, activities: list[str], splits: list[str]) -> dict:
    report = {
        "dataset_root": dataset_root,
        "activities": activities,
        "splits": splits,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "per_split": {},
        "summary": {
            "total_csv_rows": 0,
            "rows_missing_file": 0,
            "label_mismatch": 0,
            "unique_labels": set(),
            "expected_labels": sorted(activities),
            "all_files_exist": True,
            "all_labels_match": True,
            "label_set_matches_expected": False,
        },
    }

    all_rows = []

    for split in splits:
        path = csv_path(dataset_root, split)
        split_entry = {
            "csv_path": path,
            "csv_exists": os.path.isfile(path),
            "row_count": 0,
            "missing_files": [],
            "label_mismatches": [],
            "unique_labels": set(),
        }
        report["per_split"][split] = split_entry

        if not split_entry["csv_exists"]:
            print(f"[verify_labels] WARNING: CSV not found: {path}")
            continue

        rows = load_csv_rows(path)
        split_entry["row_count"] = len(rows)
        report["summary"]["total_csv_rows"] += len(rows)

        # CSV filenames (e.g. "A_batch/batch_0.mat") are relative to the
        # directory that contains the CSV (Slots/Train or Slots/Test),
        # not to dataset_root.
        csv_dir = os.path.dirname(path)

        for row in rows:
            filename = row.get("filename", "").strip()
            label = row.get("label", "").strip()
            full_path = os.path.join(csv_dir, filename)
            rel_path = os.path.relpath(full_path, dataset_root)

            split_entry["unique_labels"].add(label)
            report["summary"]["unique_labels"].add(label)
            all_rows.append((split, rel_path, label, full_path))

            # Check file existence
            if not os.path.isfile(full_path):
                split_entry["missing_files"].append(filename)
                report["summary"]["rows_missing_file"] += 1
                report["summary"]["all_files_exist"] = False

            # Check label matches folder name
            expected = expected_label_from_filename(filename)
            if expected is not None and label != expected:
                split_entry["label_mismatches"].append(
                    {"filename": filename, "label": label, "expected": expected}
                )
                report["summary"]["label_mismatch"] += 1
                report["summary"]["all_labels_match"] = False

        # Convert sets to sorted lists for JSON serialization
        split_entry["unique_labels"] = sorted(split_entry["unique_labels"])

    # Label-set comparison
    unique_labels_sorted = sorted(report["summary"]["unique_labels"])
    report["summary"]["unique_labels"] = unique_labels_sorted
    report["summary"]["label_set_matches_expected"] = (
        set(unique_labels_sorted) == set(activities)
    )

    # Per-activity counts across all splits
    per_activity = {a: {"Train": 0, "Val": 0, "Test": 0} for a in activities}
    for split, filename, label, _full_path in all_rows:
        if label in per_activity and split in per_activity[label]:
            per_activity[label][split] += 1
    report["per_activity"] = per_activity

    # Console output
    print(f"[verify_labels] Dataset root: {dataset_root}")
    print(f"[verify_labels] Activities: {activities}")
    print(f"[verify_labels] Splits: {splits}")
    print()
    for split in splits:
        se = report["per_split"][split]
        print(f"Split {split}:")
        print(f"  CSV exists: {se['csv_exists']}  rows: {se['row_count']}")
        print(f"  Missing files: {len(se['missing_files'])}")
        print(f"  Label mismatches: {len(se['label_mismatches'])}")
        print(f"  Unique labels: {se['unique_labels']}")
        if se["missing_files"][:3]:
            print(f"    e.g. missing: {se['missing_files'][:3]}")
        if se["label_mismatches"][:3]:
            print(f"    e.g. mismatch: {se['label_mismatches'][:3]}")
        print()
    print("=== Summary ===")
    print(f"  Total CSV rows: {report['summary']['total_csv_rows']}")
    print(f"  Rows missing file: {report['summary']['rows_missing_file']}")
    print(f"  Label mismatches: {report['summary']['label_mismatch']}")
    print(f"  Unique labels found: {report['summary']['unique_labels']}")
    print(f"  All files exist: {report['summary']['all_files_exist']}")
    print(f"  All labels match folder: {report['summary']['all_labels_match']}")
    print(f"  Label set matches expected {activities}: {report['summary']['label_set_matches_expected']}")
    print()
    print("=== Per-activity counts ===")
    for a in activities:
        print(f"  {a}: Train={per_activity[a]['Train']} "
              f"Val={per_activity[a]['Val']} Test={per_activity[a]['Test']}")

    return report


def write_report(report: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # JSON cannot serialize sets; they were already converted to lists.
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[verify_labels] Report written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify label consistency across SiMWiSense train/val/test CSVs."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ROOT,
        help="Path to the SiMWiSense monitor root (default: %(default)s)",
    )
    parser.add_argument(
        "--activities",
        nargs="+",
        default=DEFAULT_ACTIVITIES,
        help="Expected activity labels (default: A B C)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Splits to verify (default: Train Val Test)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON report path (default: ../data/metadata/label_verification_report.json)",
    )
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(
            repo_root_from_src(), "data", "metadata", "label_verification_report.json"
        )

    report = verify(args.dataset, args.activities, args.splits)
    write_report(report, out_path)


if __name__ == "__main__":
    main()
