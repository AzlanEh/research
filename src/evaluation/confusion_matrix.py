"""confusion_matrix.py

Confusion-matrix computation for the Wi-Fi CSI activity-recognition POC.

Only parsable predictions with valid ground-truth and predicted labels are
included. UNKNOWN and parse-error predictions are tracked separately.
"""

from __future__ import annotations

from typing import Any

VALID_LABELS = ("A", "B", "C")


def compute_confusion_matrix(
    predictions: list[dict[str, Any]],
    labels: tuple[str, ...] = VALID_LABELS,
) -> dict[str, Any]:
    """Compute a confusion matrix from prediction records.

    Args:
        predictions: List of prediction record dicts.
        labels: Tuple of valid class labels.

    Returns:
        Dict with:
            "matrix"        : dict[true_label][pred_label] -> count
            "labels"        : list of labels
            "total"         : total predictions
            "evaluable"     : predictions with valid gt and pred labels
            "unknown_count" : predictions labelled UNKNOWN
            "parse_error_count": predictions that failed to parse
    """
    label_list = list(labels)
    matrix = {t: {p: 0 for p in label_list} for t in label_list}

    total = len(predictions)
    unknown_count = 0
    parse_error_count = 0
    evaluable = 0

    for p in predictions:
        gt = p.get("ground_truth")
        pr = p.get("predicted_label")
        status = p.get("parse_status")

        if status != "ok":
            parse_error_count += 1
            continue

        if pr == "UNKNOWN":
            unknown_count += 1
            continue

        if gt in labels and pr in labels:
            matrix[gt][pr] += 1
            evaluable += 1

    return {
        "matrix": matrix,
        "labels": label_list,
        "total": total,
        "evaluable": evaluable,
        "unknown_count": unknown_count,
        "parse_error_count": parse_error_count,
    }


def format_confusion_matrix_text(
    cm: dict[str, Any],
) -> str:
    """Return a human-readable text confusion matrix."""
    labels = cm["labels"]
    matrix = cm["matrix"]

    # Header
    header = "              " + " ".join(f"{l:>8}" for l in labels)
    lines = [header, "Predicted \\ Actual"]
    for true_label in labels:
        row = matrix.get(true_label, {})
        cells = " ".join(f"{row.get(pred, 0):>8}" for pred in labels)
        lines.append(f"Actual {true_label:>4} : {cells}")
    lines.append("")
    lines.append(
        f"Total: {cm['total']}  Evaluable: {cm['evaluable']}  "
        f"UNKNOWN: {cm['unknown_count']}  Parse errors: {cm['parse_error_count']}"
    )
    return "\n".join(lines)
