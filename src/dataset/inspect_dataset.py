"""inspect_dataset.py

Inspect the SiMWiSense fine-grained CSI dataset layout.

Answers:
  1. Where is the dataset on this machine?
  2. What are the actual activity folders and filenames?
  3. How many batch files exist per activity / split?
  4. What is the shape/dtype of a sample .mat window?

Designed to run from `src/`:
    python dataset/inspect_dataset.py

Outputs a reproducible inspection report to
    ../data/metadata/inspection_report.json

This script is read-only with respect to the source dataset
(/mnt/external_disk). It does not modify any existing decoder or
investigation files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

try:
    import scipy.io
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "scipy is required. Activate the project env and install requirements:\n"
        "  .venv/bin/pip install -r requirements.txt"
    ) from exc

# ---------------------------------------------------------------------------
# Defaults (absolute path to the source dataset, treated as read-only)
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense"
    "/Data/fine_grained/Classroom/80MHz/3mo/m1"
)

DEFAULT_ACTIVITIES = ["A", "B", "C"]
DEFAULT_SPLITS = ["Train", "Val", "Test"]

# The batch directory naming convention, e.g. "Slots/Train/A_batch".
BATCH_SUFFIX = "_batch"


def repo_root_from_src() -> str:
    """Return the repo root assuming this file lives in <root>/src/dataset/."""
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(here)
    return os.path.dirname(src_dir)


def activity_folder_exists(root: str, monitor: str, activity: str) -> bool:
    return os.path.isdir(os.path.join(root, activity))


def count_batch_files(batch_dir: str) -> int:
    if not os.path.isdir(batch_dir):
        return 0
    return sum(1 for f in os.listdir(batch_dir) if f.endswith(".mat"))


def list_batch_files(batch_dir: str) -> list[str]:
    if not os.path.isdir(batch_dir):
        return []
    return sorted(f for f in os.listdir(batch_dir) if f.endswith(".mat"))


def sample_window_shape(path: str) -> tuple:
    """Load the first .mat file found under `path` and return (shape, dtype, complex)."""
    if not os.path.isdir(path):
        return None, None, None
    files = sorted(f for f in os.listdir(path) if f.endswith(".mat"))
    if not files:
        return None, None, None
    mat = scipy.io.loadmat(os.path.join(path, files[0]))
    csi = mat.get("csi_mon")
    if csi is None:
        return None, None, None
    return csi.shape, str(csi.dtype), bool(__import__("numpy").iscomplexobj(csi))


def inspect(dataset_root: str, activities: list[str], splits: list[str]) -> dict:
    report = {
        "dataset_root": dataset_root,
        "monitor": os.path.basename(dataset_root),
        "activities": activities,
        "splits": splits,
        "inspected_at": datetime.now(timezone.utc).isoformat(),
        "structure": {},
        "sample_window": {},
        "totals": {"train": 0, "val": 0, "test": 0, "grand_total": 0},
    }

    exists = os.path.isdir(dataset_root)
    report["root_exists"] = exists
    if not exists:
        print(f"[inspect_dataset] WARNING: dataset root not found: {dataset_root}")
        return report

    print(f"[inspect_dataset] Dataset root: {dataset_root}")
    print(f"[inspect_dataset] Activities: {activities}")
    print(f"[inspect_dataset] Splits: {splits}")
    print()

    for activity in activities:
        entry = {"activity": activity, "folder_exists": False, "splits": {}}
        activity_dir = os.path.join(dataset_root, activity)
        entry["folder_exists"] = os.path.isdir(activity_dir)

        for split in splits:
            batch_dir = os.path.join(
                dataset_root, "Slots", split, f"{activity}{BATCH_SUFFIX}"
            )
            split_entry = {
                "batch_dir": batch_dir,
                "dir_exists": os.path.isdir(batch_dir),
                "batch_count": count_batch_files(batch_dir),
                "files": list_batch_files(batch_dir)[:5],  # first 5 for readability
            }
            # Sample window shape from the Train split representative.
            if split == "Train" and split_entry["dir_exists"]:
                shape, dtype, is_complex = sample_window_shape(batch_dir)
                split_entry["sample_shape"] = list(shape) if shape else None
                split_entry["sample_dtype"] = dtype
                split_entry["sample_is_complex"] = is_complex
            entry["splits"][split] = split_entry
            key = split.lower()
            if key in report["totals"]:
                report["totals"][key] += split_entry["batch_count"]

        report["structure"][activity] = entry
        report["totals"]["grand_total"] += entry["splits"].get("Train", {}).get("batch_count", 0) \
            + entry["splits"].get("Val", {}).get("batch_count", 0) \
            + entry["splits"].get("Test", {}).get("batch_count", 0)

        # Console summary
        print(f"Activity {activity}: folder_exists={entry['folder_exists']}")
        for split in splits:
            se = entry["splits"][split]
            print(
                f"  {split:5s}: dir={se['dir_exists']} "
                f"batches={se['batch_count']}"
                + (f" sample_shape={se.get('sample_shape')}" if se.get("sample_shape") else "")
            )
        print()

    # Print totals
    print("=== Totals ===")
    for k in ("train", "val", "test", "grand_total"):
        print(f"  {k}: {report['totals'][k]}")

    return report


def write_report(report: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[inspect_dataset] Report written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the SiMWiSense fine-grained CSI dataset layout."
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
        help="Activity folders to inspect (default: A B C)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Splits to inspect (default: Train Val Test)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON report path (default: ../data/metadata/inspection_report.json)",
    )
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(
            repo_root_from_src(), "data", "metadata", "inspection_report.json"
        )

    report = inspect(args.dataset, args.activities, args.splits)
    write_report(report, out_path)


if __name__ == "__main__":
    main()
