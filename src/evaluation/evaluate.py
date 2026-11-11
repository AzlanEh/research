"""evaluate.py

Run LLM classification on a set of feature records and save evaluation
results.

Saves:
  - results/<experiment_name>/predictions.jsonl  (one line per prediction)
  - results/<experiment_name>/metrics.json       (aggregated metrics)
  - results/<experiment_name>/confusion_matrix.txt (human-readable)

Each prediction record includes ground truth, predicted label, confidence,
parse status, model, latency, token usage, prompt version, representation,
and any error.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from classifier import classify_batch
from confusion_matrix import compute_confusion_matrix, format_confusion_matrix_text
from metrics import compute_metrics


def repo_root_from_src() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(here)
    return os.path.dirname(src_dir)


def results_dir(experiment_name: str) -> str:
    root = repo_root_from_src()
    return os.path.join(root, "results", experiment_name)


def save_predictions(
    predictions: list[dict[str, Any]], out_path: str
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(json.dumps(p) + "\n")


def save_prompts(
    predictions: list[dict[str, Any]], out_path: str
) -> None:
    """Save the exact prompt messages used for each prediction.

    Only predictions that include "prompt_messages" are saved.
    This enables full reproducibility of the LLM input.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    prompts = []
    for p in predictions:
        if "prompt_messages" in p:
            prompts.append({
                "source_file": p.get("source_file"),
                "ground_truth": p.get("ground_truth"),
                "split": p.get("split"),
                "representation": p.get("representation"),
                "prompt_mode": p.get("prompt_mode"),
                "prompt_messages": p["prompt_messages"],
            })
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(prompts, fh, indent=2)


def run_evaluation(
    predictions: list[dict[str, Any]],
    experiment_name: str,
    labels: tuple[str, ...] = ("A", "B", "C"),
) -> dict[str, Any]:
    """Compute metrics and confusion matrix, save all artifacts.

    Returns the combined results dict.
    """
    out_dir = results_dir(experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    metrics = compute_metrics(predictions, labels=labels)
    cm = compute_confusion_matrix(predictions, labels=labels)

    # Save predictions.
    pred_path = os.path.join(out_dir, "predictions.jsonl")
    save_predictions(predictions, pred_path)

    # Save exact prompts for reproducibility.
    prompt_path = os.path.join(out_dir, "prompts.json")
    save_prompts(predictions, prompt_path)

    # Save metrics.
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Save confusion matrix text.
    cm_path = os.path.join(out_dir, "confusion_matrix.txt")
    with open(cm_path, "w", encoding="utf-8") as fh:
        fh.write(format_confusion_matrix_text(cm))

    # Save a combined summary.
    summary = {
        "experiment_name": experiment_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels": list(labels),
        "metrics": metrics,
        "confusion_matrix": cm,
        "artifacts": {
            "predictions": pred_path,
            "prompts": prompt_path,
            "metrics": metrics_path,
            "confusion_matrix": cm_path,
        },
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== Evaluation: {experiment_name} ===")
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"Evaluable: {metrics['evaluable_predictions']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Macro F1: {metrics['macro_f1']}")
    print(f"Macro precision: {metrics['macro_precision']}")
    print(f"Macro recall: {metrics['macro_recall']}")
    print(f"UNKNOWN rate: {metrics['unknown_rate']}")
    print(f"Parse error rate: {metrics['parse_error_rate']}")
    print(f"\nPer-class:")
    for label, m in metrics["per_class"].items():
        print(
            f"  {label}: P={m['precision']} R={m['recall']} F1={m['f1']} support={m['support']}"
        )
    print(f"\nArtifacts written to: {out_dir}")

    return summary
