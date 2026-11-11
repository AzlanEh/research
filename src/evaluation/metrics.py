"""metrics.py

Classification metrics for the Wi-Fi CSI activity-recognition POC.

Computes:
  - Accuracy
  - Macro precision, recall, F1 (per-class then averaged)
  - Per-class precision, recall, F1, support
  - UNKNOWN rate (fraction of predictions that were UNKNOWN)
  - Parse-error rate

The LLM's self-reported confidence is treated as an uncalibrated score,
not a probability, so no calibration metrics are computed here.
"""

from __future__ import annotations

from typing import Any

VALID_LABELS = ("A", "B", "C")


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def compute_metrics(
    predictions: list[dict[str, Any]],
    labels: tuple[str, ...] = VALID_LABELS,
) -> dict[str, Any]:
    """Compute classification metrics from prediction records.

    Args:
        predictions: List of prediction record dicts (from classifier.py).
            Each must have "ground_truth" and "predicted_label" keys.
        labels: Tuple of valid class labels (default ("A","B","C")).

    Returns:
        Dict with accuracy, macro precision/recall/F1, per-class metrics,
        UNKNOWN rate, parse-error rate, and counts.
    """
    total = len(predictions)
    if total == 0:
        return _empty_metrics(labels)

    # Count parse statuses.
    parse_statuses: dict[str, int] = {}
    for p in predictions:
        st = p.get("parse_status", "unknown")
        parse_statuses[st] = parse_statuses.get(st, 0) + 1

    # Count UNKNOWN predictions.
    unknown_count = sum(
        1 for p in predictions if p.get("predicted_label") == "UNKNOWN"
    )
    unknown_rate = _safe_div(unknown_count, total)

    # Parse-error rate (anything not "ok").
    parse_error_count = total - parse_statuses.get("ok", 0)
    parse_error_rate = _safe_div(parse_error_count, total)

    # Build confusion counts only for parsable predictions with valid labels.
    y_true = []
    y_pred = []
    for p in predictions:
        gt = p.get("ground_truth")
        pr = p.get("predicted_label")
        if gt in labels and pr in labels:
            y_true.append(gt)
            y_pred.append(pr)

    evaluable = len(y_true)
    correct = sum(1 for t, pr in zip(y_true, y_pred) if t == pr)
    accuracy = _safe_div(correct, evaluable) if evaluable > 0 else 0.0

    # Per-class precision, recall, F1.
    per_class: dict[str, dict[str, Any]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in labels:
        tp = sum(1 for t, pr in zip(y_true, y_pred) if t == label and pr == label)
        fp = sum(1 for t, pr in zip(y_true, y_pred) if t != label and pr == label)
        fn = sum(1 for t, pr in zip(y_true, y_pred) if t == label and pr != label)
        support = sum(1 for t in y_true if t == label)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    n_labels = len(labels)
    macro_precision = _safe_div(macro_precision, n_labels)
    macro_recall = _safe_div(macro_recall, n_labels)
    macro_f1 = _safe_div(macro_f1, n_labels)

    return {
        "total_predictions": total,
        "evaluable_predictions": evaluable,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "unknown_rate": round(unknown_rate, 4),
        "unknown_count": unknown_count,
        "parse_error_rate": round(parse_error_rate, 4),
        "parse_error_count": parse_error_count,
        "parse_statuses": parse_statuses,
        "per_class": per_class,
        "labels": list(labels),
    }


def _empty_metrics(labels: tuple[str, ...]) -> dict[str, Any]:
    per_class = {l: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0, "tp": 0, "fp": 0, "fn": 0} for l in labels}
    return {
        "total_predictions": 0,
        "evaluable_predictions": 0,
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "unknown_rate": 0.0,
        "unknown_count": 0,
        "parse_error_rate": 0.0,
        "parse_error_count": 0,
        "parse_statuses": {},
        "per_class": per_class,
        "labels": list(labels),
    }
