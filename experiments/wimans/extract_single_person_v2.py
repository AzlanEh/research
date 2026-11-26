"""Extract v2 temporal features for the single-person pilot selection.

``extract_manifest_v2`` consumes a JSONL manifest whose rows carry a
``csi_path``. The pilot selection file is a single JSON document of inference
records without paths, so this wrapper joins the selected IDs back to the
source manifest, writes a temporary JSONL in the shape the extractor expects,
and delegates to the existing function.

It never reads or writes the v1 features or the older v2 pilot features, and
it refuses to overwrite an existing output unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.wimans.extract_features_v2 import (  # noqa: E402
    FEATURE_VERSION,
    extract_manifest_v2,
)

SELECTION_PATH = (
    ROOT / "experiments/wimans/single_person_pilot/selection_manifest.json"
)
SOURCE_MANIFEST = ROOT / "data/processed/wimans/manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/wimans/features_v2_single_person.jsonl"

PROTECTED_OUTPUTS = frozenset(
    {
        (ROOT / "data/processed/wimans/features_v1.jsonl").resolve(),
        (ROOT / "data/processed/wimans/features_v2_pilot.jsonl").resolve(),
    }
)

RECORD_KEYS = frozenset(
    {"sample_id", "group_id", "environment", "wifi_band", "csi_shape", "features"}
)
GROUND_TRUTH_KEYS = frozenset(
    {"number_of_users", "num_users", "users", "user", "activity", "activities",
     "label", "labels", "arm", "ground_truth"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_rows(selection_ids: list[str]) -> list[dict[str, Any]]:
    source = {row["sample_id"]: row for row in load_jsonl(SOURCE_MANIFEST)}
    missing = [sid for sid in selection_ids if sid not in source]
    if missing:
        raise ValueError(f"Selection IDs missing from manifest: {missing[:10]}")

    rows = []
    absent_paths = []
    for sample_id in selection_ids:
        entry = source[sample_id]
        csi_path = Path(entry["csi_path"])
        if not csi_path.exists():
            absent_paths.append(str(csi_path))
        rows.append(
            {
                "sample_id": entry["sample_id"],
                "group_id": entry["group_id"],
                "environment": entry.get("environment"),
                "wifi_band": entry.get("wifi_band"),
                "csi_path": str(csi_path),
            }
        )

    if absent_paths:
        raise FileNotFoundError(
            f"{len(absent_paths)} CSI files are missing, e.g. {absent_paths[:5]}"
        )
    return rows


def verify_output(path: Path, selection_ids: list[str]) -> dict[str, Any]:
    records = load_jsonl(path)
    output_ids = [row["sample_id"] for row in records]

    problems: list[str] = []
    if output_ids != selection_ids:
        problems.append("Output sample IDs are not exactly the selection order.")
    if len(output_ids) != len(set(output_ids)):
        problems.append("Duplicate sample IDs in output.")
    if len(records) != len(selection_ids):
        problems.append(
            f"Expected {len(selection_ids)} records, got {len(records)}."
        )

    manifest_shape = {
        row["sample_id"]: row.get("shape")
        for row in load_jsonl(SOURCE_MANIFEST)
    }

    for row in records:
        sample_id = row["sample_id"]
        if set(row) != RECORD_KEYS:
            problems.append(f"{sample_id}: unexpected record keys {sorted(row)}")
        leaked = set(row) & GROUND_TRUTH_KEYS
        if leaked:
            problems.append(f"{sample_id}: ground-truth keys {sorted(leaked)}")

        features = row.get("features")
        if not isinstance(features, dict):
            problems.append(f"{sample_id}: missing features object")
            continue
        if features.get("feature_version") != FEATURE_VERSION:
            problems.append(
                f"{sample_id}: feature_version {features.get('feature_version')!r}"
            )

        csi_shape = row.get("csi_shape")
        if csi_shape != manifest_shape.get(sample_id):
            problems.append(
                f"{sample_id}: csi_shape {csi_shape} != manifest "
                f"{manifest_shape.get(sample_id)}"
            )

        packets = features.get("num_packets")
        if isinstance(csi_shape, list) and csi_shape:
            if packets != csi_shape[0]:
                problems.append(
                    f"{sample_id}: num_packets {packets} != csi_shape[0] "
                    f"{csi_shape[0]}"
                )

        sequence = features.get("temporal_sequence")
        if not isinstance(sequence, list) or len(sequence) != features.get("num_windows"):
            problems.append(f"{sample_id}: temporal_sequence length mismatch")

        matrix = features.get("cross_link_synchronization", {}).get("matrix")
        if (
            not isinstance(matrix, list)
            or len(matrix) != 9
            or any(not isinstance(r, list) or len(r) != 9 for r in matrix)
        ):
            problems.append(f"{sample_id}: synchronization matrix is not 9x9")

    if problems:
        raise RuntimeError(
            "v2 extraction audit failed:\n- " + "\n- ".join(problems[:20])
        )

    return {
        "records": len(records),
        "unique_sample_ids": len(set(output_ids)),
        "selection_order_match": output_ids == selection_ids,
        "feature_version": FEATURE_VERSION,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract v2 features for the single-person pilot selection."
    )
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output if it already exists.",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output in PROTECTED_OUTPUTS:
        raise SystemExit(f"Refusing to write a protected feature file: {output}")
    if output.exists() and not args.force:
        raise SystemExit(
            f"{output} already exists; pass --force to overwrite it."
        )

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selection_ids = [row["sample_id"] for row in selection["records"]]

    print(f"Selected recordings: {len(selection_ids)}")
    rows = build_rows(selection_ids)

    with tempfile.TemporaryDirectory(prefix="wimans_v2_") as tmpdir:
        staged = Path(tmpdir) / "selection_rows.jsonl"
        with staged.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

        extract_manifest_v2(staged, output)

    audit = verify_output(output, selection_ids)
    print()
    print("V2 EXTRACTION AUDIT")
    print("===================")
    print("records:", audit["records"])
    print("unique sample IDs:", audit["unique_sample_ids"])
    print("same IDs/order as selection:", audit["selection_order_match"])
    print("total bytes:", audit["bytes"])
    print("sha256:", audit["sha256"])
    print("status: PASS")

    audit_record = {
        "experiment": "wimans_single_person_pilot",
        "stage": "v2_feature_extraction",
        "feature_version": audit["feature_version"],
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "selection_manifest": str(args.selection.resolve().relative_to(ROOT)),
        "output_features": str(output.relative_to(ROOT)),
        "selected_records": len(selection_ids),
        "extracted_records": audit["records"],
        "unique_sample_ids": audit["unique_sample_ids"],
        "selection_order_match": audit["selection_order_match"],
        "output_bytes": audit["bytes"],
        "output_sha256": audit["sha256"],
        "status": "PASS",
    }

    audit_path = args.selection.resolve().parent / "extraction_audit.json"
    audit_path.write_text(
        json.dumps(audit_record, indent=2) + "\n", encoding="utf-8"
    )
    print("audit written:", audit_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
