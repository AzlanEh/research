"""build_manifest.py

Build a reproducible, machine-readable manifest of the SiMWiSense CSI
windows for the activity-recognition POC.

The manifest records, for every batch file:
  filename, label, split, monitor, dataset variant, shape, dtype.

It also performs the meaningful leakage check: overlap between splits is
computed on full relative paths (e.g.
  Slots/Train/A_batch/batch_0.mat  vs  Slots/Test/B_batch/batch_0.mat)
so two files that merely share a basename but live in different activity
folders are NOT counted as overlapping. A naive basename-only overlap is
also reported for comparison.

Labels are provisional single letters (A, B, C). The accompanying
activity_mapping.json explicitly marks each activity name as null and
mapping_verified = false, so no unsupported claim about activity identity
is made.

Designed to run from `src/`:
    python dataset/build_manifest.py

Outputs:
  ../data/metadata/activity_mapping.json
  ../data/metadata/build_manifest_report.json
  ../data/processed/manifest.json

Read-only with respect to the source dataset (/mnt/external_disk).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

try:
    import numpy as np
    import scipy.io
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "numpy/scipy required. Activate the project env and install requirements:\n"
        "  .venv/bin/pip install -r requirements.txt"
    ) from exc

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

# Paths relative to the repo root (<root>/src/dataset/ -> <root>/).
MANIFEST_OUT = os.path.join("data", "processed", "manifest.json")
MAPPING_OUT = os.path.join("data", "metadata", "activity_mapping.json")
REPORT_OUT = os.path.join("data", "metadata", "build_manifest_report.json")


def repo_root_from_src() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(here)
    return os.path.dirname(src_dir)


def abs_path(rel_path: str) -> str:
    """Resolve a path that is relative to the repo root when run from src/."""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(repo_root_from_src(), rel_path)


def csv_path(dataset_root: str, split: str) -> str:
    return os.path.join(
        dataset_root, "Slots", SPLIT_CSV_DIR[split], CSV_FILENAMES[split]
    )


def load_csv_rows(csv_path: str) -> list[dict]:
    rows = []
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def sample_window_shape(dataset_root: str, activity: str, split: str = "Train"):
    """Return (shape, dtype, is_complex) for one representative .mat file."""
    batch_dir = os.path.join(
        dataset_root, "Slots", split, f"{activity}_batch"
    )
    if not os.path.isdir(batch_dir):
        return None, None, None
    files = sorted(f for f in os.listdir(batch_dir) if f.endswith(".mat"))
    if not files:
        return None, None, None
    mat = scipy.io.loadmat(os.path.join(batch_dir, files[0]))
    csi = mat.get("csi_mon")
    if csi is None:
        return None, None, None
    return tuple(csi.shape), str(csi.dtype), bool(np.iscomplexobj(csi))


def build_manifest(dataset_root: str, activities: list[str], splits: list[str]) -> dict:
    # 1. Load all CSV rows with their full relative path as the key.
    #    Key = relative path from dataset_root, e.g. "Slots/Train/A_batch/batch_0.mat"
    entries: dict[str, dict] = {}
    per_split_rows: dict[str, list[dict]] = {s: [] for s in splits}
    per_activity_counts: dict[str, dict[str, int]] = {
        a: {s: 0 for s in splits} for a in activities
    }

    for split in splits:
        path = csv_path(dataset_root, split)
        rows = load_csv_rows(path)
        # CSV filenames (e.g. "A_batch/batch_0.mat") are relative to the
        # directory that contains the CSV (Slots/Train or Slots/Test).
        csv_dir = os.path.dirname(path)
        for row in rows:
            filename = row.get("filename", "").strip()
            label = row.get("label", "").strip()
            if not filename or not label:
                continue
            full_path = os.path.join(csv_dir, filename)
            rel_path = os.path.relpath(full_path, dataset_root)
            entry = {
                "filename": rel_path,
                "label": label,
                "split": split,
                "monitor": os.path.basename(dataset_root),
                "dataset": "fine_grained",
                "environment": "Classroom/80MHz/3mo",
            }
            entries[rel_path] = entry
            per_split_rows[split].append(entry)
            if label in per_activity_counts and split in per_activity_counts[label]:
                per_activity_counts[label][split] += 1

    # 2. Sample window shape/dtype for every label appearing in the data.
    #    (Not limited to --activities, since the CSVs contain all 20 labels.)
    sampled_shape: dict[str, tuple] = {}
    sampled_dtype: dict[str, str] = {}
    sampled_complex: dict[str, bool] = {}
    labels_in_data = {e["label"] for e in entries.values()}
    for activity in sorted(labels_in_data):
        shape, dtype, is_complex = sample_window_shape(dataset_root, activity, "Train")
        sampled_shape[activity] = shape
        sampled_dtype[activity] = dtype
        sampled_complex[activity] = is_complex

    # Annotate every entry with shape/dtype from its activity's sample.
    for entry in entries.values():
        act = entry["label"]
        entry["shape"] = list(sampled_shape.get(act) or [])
        entry["dtype"] = sampled_dtype.get(act)
        entry["is_complex"] = sampled_complex.get(act)

    # ------------------------------------------------------------------
    # 3. Leakage checks.
    #    Three increasingly strict notions of overlap:
    #      a) Exact relative-path overlap (full path incl. split dir).
    #      b) Split-relative path overlap (strip "Slots/<split>/" prefix,
    #         e.g. "A_batch/batch_0.mat" appears in both Train and Test).
    #      c) CSI-content duplicate check: for entries sharing a
    #         split-relative path, load both .mat files and compare
    #         complex CSI arrays exactly.
    #    Source-row overlap is NOT checkable from these .mat files (they
    #    contain only csi_mon, no original row index), so it is reported
    #    as unverified.
    # ------------------------------------------------------------------
    def split_relative_path(full_rel: str, split: str) -> str:
        """Strip 'Slots/<split>/' prefix -> 'A_batch/batch_0.mat'."""
        prefix = f"Slots/{split}/"
        if full_rel.startswith(prefix):
            return full_rel[len(prefix):]
        return full_rel

    # Attach split-relative path to each entry.
    for entry in entries.values():
        entry["split_relative_path"] = split_relative_path(
            entry["filename"], entry["split"]
        )

    # 3a. Exact full-relative-path overlap.
    split_sets_full: dict[str, set[str]] = {
        s: {e["filename"] for e in per_split_rows[s]} for s in splits
    }
    overlap_full_path: dict[str, int] = {}
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            key = f"{s1.lower()}_{s2.lower()}_overlap_full_path"
            overlap_full_path[key] = len(split_sets_full[s1] & split_sets_full[s2])

    # 3b. Split-relative path overlap (the meaningful filename-level check).
    split_sets_rel: dict[str, set[str]] = {
        s: {e["split_relative_path"] for e in per_split_rows[s]} for s in splits
    }
    overlap_split_relative: dict[str, int] = {}
    split_relative_matches: dict[str, list[tuple[str, str]]] = {}
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            key = f"{s1.lower()}_{s2.lower()}_overlap_split_relative"
            common = split_sets_rel[s1] & split_sets_rel[s2]
            overlap_split_relative[key] = len(common)
            split_relative_matches[key] = sorted(
                (f"Slots/{s1}/{p}", f"Slots/{s2}/{p}") for p in common
            )

    # Baseline: basename-only overlap (for transparency, NOT the leakage metric).
    def basename_set(split: str) -> set[str]:
        return {os.path.basename(e["filename"]) for e in per_split_rows[split]}

    basename_overlap: dict[str, int] = {}
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            key = f"{s1.lower()}_{s2.lower()}_overlap_basename_only"
            basename_overlap[key] = len(basename_set(s1) & basename_set(s2))

    # 3c. CSI-content duplicate check for split-relative matches.
    #     Load both files and compare complex arrays exactly.
    csi_content_duplicates: dict[str, int] = {}
    csi_content_check_note = (
        "CSI-content comparison performed only on split-relative path matches "
        "for classes A/B/C. Loading every file is expensive; a full all-pairs "
        "content hash would require reading the entire dataset."
    )
    for key, pairs in split_relative_matches.items():
        # key is like "train_test_overlap_split_relative"
        parts = key.split("_")
        s1 = parts[0]
        s2 = parts[1]
        count = 0
        for p1, p2 in pairs:
            act1 = p1.split("/")[1].replace("_batch", "")
            act2 = p2.split("/")[1].replace("_batch", "")
            if act1 not in ("A", "B", "C") or act2 not in ("A", "B", "C"):
                continue
            full1 = os.path.join(dataset_root, p1)
            full2 = os.path.join(dataset_root, p2)
            try:
                c1 = scipy.io.loadmat(full1)["csi_mon"]
                c2 = scipy.io.loadmat(full2)["csi_mon"]
                if c1.shape == c2.shape and np.array_equal(c1, c2):
                    count += 1
            except Exception:
                continue
        csi_content_duplicates[key] = count

    # 3d. Source-row overlap: not determinable from .mat files alone.
    source_row_overlap_note = (
        "Original source-row indices are not stored in the .mat batch files "
        "(they contain only csi_mon). Overlapping source-row intervals "
        "cannot be determined without re-decoding the PCAP or the original "
        "full-length MAT files. Status: UNVERIFIED."
    )

    # Overall leakage status: not fully verified until content + source-row
    # checks are complete.
    leakage_status = "not_fully_verified"
    leakage_reason = (
        "Zero split-relative path overlap is necessary but not sufficient "
        "for zero leakage. CSI-content duplicates and source-row overlap "
        "remain to be checked."
    )

    # 4. Assemble manifest.
    manifest = {
        "dataset": {
            "name": "SiMWiSense",
            "variant": "fine_grained",
            "environment": "Classroom/80MHz/3mo",
            "monitor": os.path.basename(dataset_root),
            "root": dataset_root,
        },
        "classes": list(activities),
        "label_mapping": {
            act: {"activity_name": None, "mapping_verified": False}
            for act in activities
        },
        "splits": {s: {"count": len(per_split_rows[s])} for s in splits},
        "per_activity": per_activity_counts,
        "window": {
            "shape_per_activity": {a: list(sampled_shape.get(a) or []) for a in activities},
            "dtype_per_activity": sampled_dtype,
            "is_complex_per_activity": sampled_complex,
            "note": "Shape/dtype sampled from one representative .mat per activity; all windows are expected to match.",
        },
        "leakage_check": {
            "overlap_full_path": overlap_full_path,
            "overlap_split_relative": overlap_split_relative,
            "overlap_basename_only": basename_overlap,
            "csi_content_duplicates": csi_content_duplicates,
            "split_relative_matches": split_relative_matches,
            "source_row_overlap": "UNVERIFIED",
            "leakage_status": leakage_status,
            "leakage_reason": leakage_reason,
            "csi_content_check_note": csi_content_check_note,
            "source_row_overlap_note": source_row_overlap_note,
            "method_note": (
                "Leakage is assessed on three levels: (1) full relative path, "
                "(2) split-relative path (strip 'Slots/<split>/' prefix), and "
                "(3) exact CSI-content comparison on split-relative matches. "
                "Source-row overlap cannot be determined from .mat files alone "
                "(they store only csi_mon). Overall status is "
                "'not_fully_verified' until all checks are complete."
            ),
        },
        "files": sorted(entries.values(), key=lambda e: e["filename"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "build_manifest.py",
    }

    # Summary for console
    total_files = len(entries)
    print(f"[build_manifest] Dataset root: {dataset_root}")
    print(f"[build_manifest] Classes: {activities}")
    print(f"[build_manifest] Total manifest entries: {total_files}")
    print("=== Split counts ===")
    for s in splits:
        print(f"  {s}: {len(per_split_rows[s])}")
    print("=== Per-activity counts ===")
    for a in activities:
        print(
            f"  {a}: "
            + " ".join(f"{s}={per_activity_counts[a][s]}" for s in splits)
        )
    print("=== Leakage check (full relative paths) ===")
    for k, v in overlap_full_path.items():
        print(f"  {k}: {v}")
    print("=== Leakage check (split-relative paths) ===")
    for k, v in overlap_split_relative.items():
        print(f"  {k}: {v}")
    print("=== CSI-content duplicates (split-relative matches, A/B/C) ===")
    for k, v in csi_content_duplicates.items():
        print(f"  {k}: {v}")
    print("=== Leakage check (basename only, for comparison) ===")
    for k, v in basename_overlap.items():
        print(f"  {k}: {v}")
    print(f"\n[build_manifest] Leakage status: {leakage_status}")
    print(f"[build_manifest] Reason: {leakage_reason}")

    return manifest


def write_json(obj: dict, rel_path: str) -> str:
    out_path = abs_path(rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reproducible CSI manifest for the SiMWiSense activity POC."
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
        help="Provisional activity labels (default: A B C)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Splits to include (default: Train Val Test)",
    )
    parser.add_argument(
        "--manifest-out",
        default=MANIFEST_OUT,
        help="Manifest JSON output path (default: ../data/processed/manifest.json)",
    )
    parser.add_argument(
        "--mapping-out",
        default=MAPPING_OUT,
        help="Activity mapping output path (default: ../data/metadata/activity_mapping.json)",
    )
    parser.add_argument(
        "--report-out",
        default=REPORT_OUT,
        help="Build report output path (default: ../data/metadata/build_manifest_report.json)",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.dataset, args.activities, args.splits)

    # activity_mapping.json (source of truth for labels, explicitly unverified)
    mapping_path = write_json(manifest["label_mapping"], args.mapping_out)

    # Full manifest (data/processed)
    manifest_path = write_json(manifest, args.manifest_out)

    # Lightweight report (data/metadata) - excludes the huge file list for readability
    report = {
        k: v
        for k, v in manifest.items()
        if k != "files"
    }
    report["file_count"] = len(manifest["files"])
    report_path = write_json(report, args.report_out)

    print(f"\n[build_manifest] Written:")
    print(f"  {mapping_path}")
    print(f"  {manifest_path} ({len(manifest['files'])} entries)")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
