"""experiment_01_zero_shot.py

Zero-shot LLM classification of CSI activity labels (A, B, C).

Pipeline:
  1. Load experiment_manifest.json (balanced evaluation set from Test).
  2. Load CSI windows and extract structured features for each evaluation
     entry (using src/preprocessing/load_csi.py and extract_features.py).
  3. Classify each record zero-shot using src/llm/classifier.py.
  4. Save predictions and evaluation metrics to results/experiment_01_zero_shot/.

Run from src/:
    python experiments/experiment_01_zero_shot.py

Environment variables (for the LLM API):
    LLM_API_KEY        (required)
    LLM_API_BASE_URL
    LLM_MODEL
    LLM_TEMPERATURE    (default 0.0)
    LLM_MAX_TOKENS

Labels A, B, C are provisional — the human-readable activity mapping is
unverified. Leakage status is "not_fully_verified".
"""

from __future__ import annotations

import os
import sys

# Allow running as a script from src/.
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(here)
src_dir = os.path.join(repo_root, "src")
for subdir in ("", "llm", "evaluation"):
    p = os.path.join(src_dir, subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets.simwisense.experiment_manifest import (  # noqa: E402
    load_experiment_manifest,
    select_balanced_from_manifest,
)
from evaluation.evaluate import run_evaluation  # noqa: E402
from llm.classifier import classify_record, is_configured  # noqa: E402
from preprocessing.simwisense.extract_features import extract_features  # noqa: E402
from preprocessing.simwisense.load_csi import load_csi_window  # noqa: E402

DEFAULT_DATASET_ROOT = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense"
    "/Data/fine_grained/Classroom/80MHz/3mo/m1"
)

EXPERIMENT_NAME = f"experiment_01_zero_shot"
EXPERIMENT_MANIFEST = os.path.join(
    repo_root, "data", "processed", "experiment_manifest.json"
)

VALID_REPRESENTATIONS = ("structured_features", "amplitude_time_series")


def extract_feature_for_entry(entry: dict, dataset_root: str) -> dict | None:
    """Load one CSI window and extract its feature record."""
    try:
        window = load_csi_window(
            mat_path=os.path.join(dataset_root, entry["filename"]),
            label=entry["label"],
            split=entry["split"],
            source_file=entry["filename"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP {entry['filename']}: {exc}")
        return None
    return extract_features(window)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run zero-shot LLM classification experiment (A/B/C)."
    )
    parser.add_argument(
        "--manifest",
        default=EXPERIMENT_MANIFEST,
        help="Path to experiment_manifest.json",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ROOT,
        help="Path to the SiMWiSense monitor root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N evaluation entries per class (balanced across A/B/C).",
    )
    parser.add_argument(
        "--representation",
        default="structured_features",
        choices=VALID_REPRESENTATIONS,
        help="Input representation: structured_features (E1) or amplitude_time_series (E2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the feature pipeline without calling the LLM API.",
    )
    args = parser.parse_args()

    # Load experiment manifest.
    exp_manifest = load_experiment_manifest(args.manifest)
    labels = tuple(exp_manifest["labels"])

    # Balanced selection across classes from the evaluation split.
    eval_entries = select_balanced_from_manifest(
        exp_manifest,
        labels=labels,
        per_class=args.limit,
        split="evaluation",
    )

    print(f"[experiment_01_zero_shot] Evaluation entries: {len(eval_entries)}")
    print(f"[experiment_01_zero_shot] Labels: {labels}")
    print(f"[experiment_01_zero_shot] Representation: {args.representation}")
    print(f"[experiment_01_zero_shot] Leakage status: {exp_manifest['leakage_status']}")

    # Check LLM configuration.
    if not is_configured() and not args.dry_run:
        print(
            "\nERROR: LLM_API_KEY is not set. "
            "Set it and re-run, or use --dry-run to test the pipeline."
        )
        sys.exit(1)

    # Extract features for each evaluation entry.
    feature_records = []
    for i, entry in enumerate(eval_entries):
        print(
            f"  [{i + 1}/{len(eval_entries)}] Extracting features: {entry['filename']}"
        )
        rec = extract_feature_for_entry(entry, args.dataset)
        if rec is not None:
            feature_records.append(rec)

    print(
        f"\n[experiment_01_zero_shot] Feature records extracted: {len(feature_records)}"
    )

    if args.dry_run:
        print("[experiment_01_zero_shot] Dry run complete. No LLM calls made.")
        # Save feature records for inspection.
        out_dir = os.path.join(repo_root, "results", EXPERIMENT_NAME)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "feature_records.jsonl"), "w") as fh:
            for rec in feature_records:
                fh.write(json.dumps(rec) + "\n")
        print(f"  Feature records saved to {out_dir}/feature_records.jsonl")
        return

    # Classify each record zero-shot.
    print(
        f"\n[experiment_01_zero_shot] Classifying {len(feature_records)} records (zero-shot)..."
    )
    print(f"\n[experiment_01_zero_shot] Classifying {len(feature_records)} records (zero-shot)...")
    predictions = []
    for i, rec in enumerate(feature_records):
        pred = classify_record(
            rec,
            representation=args.representation,
            few_shot_examples=None,  # zero-shot
        )
        pred["record_index"] = i
        predictions.append(pred)
        status = pred["parse_status"]
        label = pred["predicted_label"]
        print(
            f"  [{i + 1}/{len(feature_records)}] "
            f"gt={pred['ground_truth']} pred={label} "
            f"conf={pred['confidence']} status={status}"
        )

    # Evaluate and save.
    exp_name = f"{EXPERIMENT_NAME}_{args.representation}"
    run_evaluation(predictions, exp_name, labels=labels)


if __name__ == "__main__":
    import json

    main()
