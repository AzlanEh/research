"""experiment_02_few_shot.py

Few-shot LLM classification of CSI activity labels (A, B, C).

Pipeline:
  1. Load experiment_manifest.json (few-shot pool from Train, evaluation
     set from Test).
  2. Load CSI windows and extract structured features for all entries.
  3. Classify each evaluation record using the few-shot examples as
     labelled context in the prompt.
  4. Save predictions and evaluation metrics to results/experiment_02_few_shot/.

Run from src/:
    python experiments/experiment_02_few_shot.py

Few-shot examples are drawn only from the Train partition. Evaluation
samples are drawn only from the Test partition. No entry is used as both
an example and an evaluation sample.

Labels A, B, C are provisional. Leakage status is "not_fully_verified".
"""

from __future__ import annotations

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(here)
src_dir = os.path.join(repo_root, "src")
for subdir in ("", "preprocessing", "llm", "evaluation"):
    p = os.path.join(src_dir, subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets.simwisense.experiment_manifest import (  # noqa: E402
    load_experiment_manifest,
    select_balanced_from_manifest,
)
from preprocessing.load_csi import load_csi_window  # noqa: E402
from preprocessing.extract_features import extract_features  # noqa: E402
from llm.classifier import classify_record, is_configured  # noqa: E402
from evaluation.evaluate import run_evaluation  # noqa: E402

DEFAULT_DATASET_ROOT = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense"
    "/Data/fine_grained/Classroom/80MHz/3mo/m1"
)

EXPERIMENT_NAME = "experiment_02_few_shot"
EXPERIMENT_MANIFEST = os.path.join(
    repo_root, "data", "processed", "experiment_manifest.json"
)

VALID_REPRESENTATIONS = ("structured_features", "amplitude_time_series")


def extract_feature_for_entry(
    entry: dict, dataset_root: str
) -> dict | None:
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
    import json

    parser = argparse.ArgumentParser(
        description="Run few-shot LLM classification experiment (A/B/C)."
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
        help="Input representation: structured_features (E3) or amplitude_time_series (E4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the pipeline without calling the LLM API.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the existing prediction checkpoint.",
    )
    args = parser.parse_args()

    # Load experiment manifest.
    exp_manifest = load_experiment_manifest(args.manifest)
    few_shot_entries = exp_manifest["few_shot"]
    labels = tuple(exp_manifest["labels"])

    # Balanced selection across classes from the evaluation split.
    eval_entries = select_balanced_from_manifest(
        exp_manifest,
        labels=labels,
        per_class=args.limit,
        split="evaluation",
    )

    print(f"[experiment_02_few_shot] Few-shot examples: {len(few_shot_entries)}")
    print(f"[experiment_02_few_shot] Evaluation entries: {len(eval_entries)}")
    print(f"[experiment_02_few_shot] Labels: {labels}")
    print(f"[experiment_02_few_shot] Representation: {args.representation}")
    print(f"[experiment_02_few_shot] Leakage status: {exp_manifest['leakage_status']}")

    if not is_configured() and not args.dry_run:
        print(
            "\nERROR: LLM_API_KEY is not set. "
            "Set it and re-run, or use --dry-run to test the pipeline."
        )
        sys.exit(1)

    # Extract features for few-shot examples.
    print("\n[experiment_02_few_shot] Extracting few-shot features...")
    few_shot_records = []
    for i, entry in enumerate(few_shot_entries):
        print(f"  [{i + 1}/{len(few_shot_entries)}] {entry['filename']}")
        rec = extract_feature_for_entry(entry, args.dataset)
        if rec is not None:
            few_shot_records.append(rec)

    # Extract features for evaluation entries.
    print(f"\n[experiment_02_few_shot] Extracting evaluation features ({len(eval_entries)} entries)...")
    eval_records = []
    for i, entry in enumerate(eval_entries):
        print(f"  [{i + 1}/{len(eval_entries)}] {entry['filename']}")
        rec = extract_feature_for_entry(entry, args.dataset)
        if rec is not None:
            eval_records.append(rec)

    print(
        f"\n[experiment_02_few_shot] "
        f"Few-shot records: {len(few_shot_records)}, "
        f"Evaluation records: {len(eval_records)}"
    )

    if args.dry_run:
        out_dir = os.path.join(repo_root, "results", EXPERIMENT_NAME)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "few_shot_feature_records.jsonl"), "w") as fh:
            for rec in few_shot_records:
                fh.write(json.dumps(rec) + "\n")
        with open(os.path.join(out_dir, "eval_feature_records.jsonl"), "w") as fh:
            for rec in eval_records:
                fh.write(json.dumps(rec) + "\n")
        print(f"[experiment_02_few_shot] Dry run complete. Feature records saved to {out_dir}")
        return

    # Classify each evaluation record with checkpoint/resume support.
    exp_name = f"{EXPERIMENT_NAME}_2shot_{args.representation}"
    out_dir = os.path.join(repo_root, "results", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "predictions_checkpoint.jsonl")

    if os.path.exists(checkpoint_path) and not args.resume:
        raise SystemExit(
            f"\nCheckpoint already exists: {checkpoint_path}\n"
            "Use --resume to continue the existing run.\n"
            "Move the checkpoint elsewhere only if you intentionally "
            "want to start a fresh run."
        )

    # Load the most recent checkpoint record for each source file.
    completed: dict[str, dict] = {}
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                saved = json.loads(line)
                source = saved.get("source_file")
                if source:
                    completed[source] = saved

    # Prevent accidentally resuming a checkpoint from another dataset.
    valid_sources = {e["filename"] for e in eval_entries}
    completed = {s: p for s, p in completed.items() if s in valid_sources}

    # Build the list of records still needing classification (or re-classification
    # if they previously ended in client_error).
    pending = []
    for rec in eval_records:
        src = rec.get("source_file")
        if src in completed and completed[src].get("parse_status") == "ok":
            # Already successfully classified — skip.
            continue
        pending.append(rec)

    print(
        f"\n[experiment_02_few_shot] "
        f"Already classified: {len(completed)}, "
        f"Pending: {len(pending)}"
    )

    predictions: list[dict] = list(completed.values())
    for i, rec in enumerate(pending):
        pred = classify_record(
            rec,
            representation=args.representation,
            few_shot_examples=few_shot_records,
        )
        pred["record_index"] = i
        predictions.append(pred)

        # Append to checkpoint immediately so progress is not lost.
        with open(checkpoint_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(pred) + "\n")

        status = pred["parse_status"]
        label = pred["predicted_label"]
        print(
            f"  [{i + 1}/{len(pending)}] "
            f"gt={pred['ground_truth']} pred={label} "
            f"conf={pred['confidence']} status={status}"
        )

    # Evaluate and save using the latest prediction per source file.
    latest_by_source: dict[str, dict] = {}
    for p in predictions:
        src = p.get("source_file")
        if src:
            latest_by_source[src] = p
    final_predictions = list(latest_by_source.values())

    run_evaluation(final_predictions, exp_name, labels=labels)


if __name__ == "__main__":
    main()
