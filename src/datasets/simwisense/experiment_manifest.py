"""experiment_manifest.py

Build a reproducible experiment manifest from the main manifest.json.

Selects balanced samples per class from the Train and Test partitions:

  - Few-shot pool:  samples from Train (used as labelled examples).
  - Evaluation set: samples from Test (used for zero-shot / few-shot eval).

A fixed random seed ensures reproducibility. The same entry is never used
as both a few-shot example and an evaluation sample.

Leakage status is recorded as "not_fully_verified" (see build_manifest.py):
split-relative filename matches exist between Train and Test, but
CSI-content differs. This manifest does not attempt to resolve that; it
simply records the status for transparency.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MANIFEST = str(
    REPO_ROOT / "data" / "processed" / "manifest.json"
)

DEFAULT_OUT = str(
    REPO_ROOT / "data" / "processed" / "experiment_manifest.json"
)


def load_manifest(path: str = DEFAULT_MANIFEST) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def select_balanced(
    entries: list[dict[str, Any]],
    labels: tuple[str, ...],
    per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select up to `per_class` entries per label, deterministically.

    Uses a fixed seed so the same selection is reproducible across runs.
    """
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        lbl = e.get("label")
        if lbl in labels:
            by_label[lbl].append(e)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in sorted(labels):
        pool = list(by_label.get(label, []))
        rng.shuffle(pool)
        selected.extend(pool[:per_class])
    return selected


def build_experiment_manifest(
    manifest_path: str = DEFAULT_MANIFEST,
    out_path: str = DEFAULT_OUT,
    *,
    labels: tuple[str, ...] = ("A", "B", "C"),
    few_shot_per_class: int = 10,
    eval_per_class: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Build and save the experiment manifest.

    Returns the manifest dict.
    """
    manifest = load_manifest(manifest_path)
    all_entries = manifest["files"]

    # Few-shot examples from Train partition.
    train_entries = [e for e in all_entries if e.get("split") == "Train"]
    few_shot = select_balanced(train_entries, labels, few_shot_per_class, seed)

    # Evaluation set from Test partition.
    test_entries = [e for e in all_entries if e.get("split") == "Test"]
    evaluation = select_balanced(test_entries, labels, eval_per_class, seed + 1)

    # Sanity check: no overlap between few-shot and evaluation sets.
    few_shot_keys = {e["filename"] for e in few_shot}
    eval_keys = {e["filename"] for e in evaluation}
    overlap = few_shot_keys & eval_keys
    if overlap:
        raise RuntimeError(
            f"Few-shot and evaluation sets overlap on {len(overlap)} entries"
        )

    experiment_manifest = {
        "source_manifest": manifest_path,
        "labels": list(labels),
        "seed": seed,
        "few_shot_per_class": few_shot_per_class,
        "eval_per_class": eval_per_class,
        "leakage_status": "not_fully_verified",
        "leakage_note": (
            "Split-relative filename matches exist between Train and Test "
            "(see build_manifest_report.json). Few-shot examples come from "
            "Train, evaluation from Test. CSI-content differs for matched "
            "filenames, but source-row overlap is unverified."
        ),
        "few_shot": few_shot,
        "evaluation": evaluation,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(experiment_manifest, fh, indent=2)

    print(f"[experiment_manifest] Written to {out_path}")
    print(f"  Few-shot: {len(few_shot)} entries "
          f"({few_shot_per_class} per class from Train)")
    print(f"  Evaluation: {len(evaluation)} entries "
          f"({eval_per_class} per class from Test)")
    print(f"  Few-shot/eval overlap: {len(overlap)}")

    return experiment_manifest


def load_experiment_manifest(path: str = DEFAULT_OUT) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def select_balanced_from_manifest(
    experiment_manifest: dict[str, Any],
    *,
    labels: tuple[str, ...] = ("A", "B", "C"),
    per_class: int | None = None,
    split: str = "evaluation",
) -> list[dict[str, Any]]:
    """Select up to `per_class` entries per label from a manifest split.

    If `per_class` is None, returns all entries from the split.

    The selection preserves the manifest's deterministic ordering (which was
    already shuffled with a fixed seed at manifest build time).
    """
    entries = experiment_manifest.get(split, [])
    if per_class is None:
        return [e for e in entries if e.get("label") in labels]

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        lbl = e.get("label")
        if lbl in labels:
            by_label[lbl].append(e)

    selected: list[dict[str, Any]] = []
    for label in sorted(labels):
        selected.extend(by_label.get(label, [])[:per_class])
    return selected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a reproducible experiment manifest (balanced A/B/C sampling)."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Path to source manifest.json",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output experiment manifest path",
    )
    parser.add_argument(
        "--few-shot-per-class",
        type=int,
        default=10,
        help="Few-shot examples per class (from Train)",
    )
    parser.add_argument(
        "--eval-per-class",
        type=int,
        default=50,
        help="Evaluation samples per class (from Test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    build_experiment_manifest(
        manifest_path=args.manifest,
        out_path=args.out,
        labels=("A", "B", "C"),
        few_shot_per_class=args.few_shot_per_class,
        eval_per_class=args.eval_per_class,
        seed=args.seed,
    )
