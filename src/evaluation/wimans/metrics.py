"""Multiset activity-count metrics for WiMANS.

Input records contain a classifier result plus a separate
``ground_truth_activity_counts`` dictionary. Ground truth is joined only after
inference and is never sent back to the model.

Failed or malformed responses are not dropped. They are treated as an
all-zero prediction for error metrics and can never count as an exact match,
even when the ground truth is also all zeros.
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Sequence

from llm.wimans.prompt_builder import ACTIVITY_LABELS

TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _normalized_counts(
    value: Any,
    labels: Sequence[str],
) -> dict[str, int]:
    """Return nonnegative integer counts for every label, defaulting to zero."""
    counts = {label: 0 for label in labels}
    if not isinstance(value, dict):
        return counts

    for label in labels:
        raw = value.get(label, 0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        if not math.isfinite(float(raw)):
            continue
        counts[label] = max(0, int(raw))

    return counts


def evaluate_records(
    records: list[dict[str, Any]],
    labels: Sequence[str] = ACTIVITY_LABELS,
) -> dict[str, Any]:
    """Evaluate activity-count predictions against joined ground truth.

    Each record must provide ``ground_truth_activity_counts``. Prediction
    counts come from ``predicted_activity_counts`` and are used only when
    ``parse_status == "ok"``.

    A non-``ok`` record contributes an all-zero prediction: it lowers accuracy,
    recall, and count MAE rather than being excluded from the denominator.
    """
    sample_count = len(records)
    if sample_count == 0:
        return _empty_metrics(labels)

    exact_matches = 0
    successful_parses = 0

    true_positive = {label: 0 for label in labels}
    false_positive = {label: 0 for label in labels}
    false_negative = {label: 0 for label in labels}
    support = {label: 0 for label in labels}

    absolute_count_errors: list[float] = []
    total_person_errors: list[float] = []
    latencies: list[float] = []

    token_totals = {key: 0 for key in TOKEN_KEYS}
    token_records = 0

    for record in records:
        parse_ok = record.get("parse_status") == "ok"
        if parse_ok:
            successful_parses += 1

        raw_prediction = record.get("predicted_activity_counts")
        predicted = _normalized_counts(
            raw_prediction if parse_ok else None,
            labels,
        )
        ground_truth = _normalized_counts(
            record.get("ground_truth_activity_counts"),
            labels,
        )

        if (
            parse_ok
            and isinstance(raw_prediction, dict)
            and predicted == ground_truth
        ):
            exact_matches += 1

        for label in labels:
            predicted_count = predicted[label]
            actual_count = ground_truth[label]
            matched = min(predicted_count, actual_count)

            true_positive[label] += matched
            false_positive[label] += max(
                0, predicted_count - actual_count
            )
            false_negative[label] += max(
                0, actual_count - predicted_count
            )
            support[label] += actual_count

            absolute_count_errors.append(
                abs(predicted_count - actual_count)
            )

        total_person_errors.append(
            abs(sum(predicted.values()) - sum(ground_truth.values()))
        )

        latency = record.get("latency_ms")
        if (
            isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and math.isfinite(float(latency))
        ):
            latencies.append(float(latency))

        usage = record.get("token_usage")
        observed_usage = False
        if isinstance(usage, dict):
            for key in TOKEN_KEYS:
                value = usage.get(key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    token_totals[key] += value
                    observed_usage = True
        if observed_usage:
            token_records += 1

    per_activity: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = true_positive[label]
        fp = false_positive[label]
        fn = false_negative[label]

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_activity[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support[label],
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }

    total_tp = sum(true_positive.values())
    total_fp = sum(false_positive.values())
    total_fn = sum(false_negative.values())

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = (
        2 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    return {
        "sample_count": sample_count,
        "exact_multiset_accuracy": _safe_div(exact_matches, sample_count),
        "exact_multiset_matches": exact_matches,
        "count_mae": (
            fmean(absolute_count_errors) if absolute_count_errors else 0.0
        ),
        "total_person_count_mae": fmean(total_person_errors),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": (
            fmean(
                values["f1"] for values in per_activity.values()
            )
            if per_activity
            else 0.0
        ),
        "per_activity": per_activity,
        "parse_success_rate": _safe_div(
            successful_parses, sample_count
        ),
        "parse_success_count": successful_parses,
        "mean_latency_ms": fmean(latencies) if latencies else None,
        "latency_sample_count": len(latencies),
        "token_usage": token_totals,
        "token_usage_record_count": token_records,
    }


def _empty_metrics(labels: Sequence[str]) -> dict[str, Any]:
    per_activity = {
        label: {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
        for label in labels
    }

    return {
        "sample_count": 0,
        "exact_multiset_accuracy": 0.0,
        "exact_multiset_matches": 0,
        "count_mae": 0.0,
        "total_person_count_mae": 0.0,
        "micro_precision": 0.0,
        "micro_recall": 0.0,
        "micro_f1": 0.0,
        "macro_f1": 0.0,
        "per_activity": per_activity,
        "parse_success_rate": 0.0,
        "parse_success_count": 0,
        "mean_latency_ms": None,
        "latency_sample_count": 0,
        "token_usage": {key: 0 for key in TOKEN_KEYS},
        "token_usage_record_count": 0,
    }
