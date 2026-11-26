"""Paired v1/v2 evaluation helpers for the WiMANS pilot.

The pilot runs both feature versions on the same recordings. This module does
the two things that must happen *after* inference:

1. Join ground-truth activity counts from the manifest ``users`` annotations.
2. Compare per-sample errors between the two versions.

Ground truth is only ever joined here. ``join_ground_truth`` refuses to add
annotations to a result that already carries them, and the prompt builders
never see manifest records.
"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from llm.wimans.prompt_builder import ACTIVITY_LABELS

from .metrics import _normalized_counts


def ground_truth_counts(
    users: Iterable[Mapping[str, Any]] | None,
    labels: Sequence[str] = ACTIVITY_LABELS,
) -> dict[str, int]:
    """Count activity annotations from a manifest ``users`` list.

    The counts come from individual user annotations, never from
    ``number_of_users``. An unknown or missing activity is an error rather than
    a silent zero, because it would corrupt the ground truth.
    """
    counts = {label: 0 for label in labels}

    for index, user in enumerate(users or []):
        activity = user.get("activity") if isinstance(user, Mapping) else None
        if activity not in counts:
            raise ValueError(
                f"User {index} has unknown or missing activity: {activity!r}"
            )
        counts[activity] += 1

    return counts


def join_ground_truth(
    results: Sequence[Mapping[str, Any]],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str] = ACTIVITY_LABELS,
) -> list[dict[str, Any]]:
    """Return copies of classifier results with ground truth attached.

    The classifier results must not already contain ground truth; that would
    indicate an evaluation-only field had leaked into inference.
    """
    joined: list[dict[str, Any]] = []

    for result in results:
        if "ground_truth_activity_counts" in result:
            raise ValueError(
                "Classifier result already contains ground truth for "
                f"{result.get('sample_id')!r}."
            )

        sample_id = result.get("sample_id")
        manifest_record = manifest_by_id.get(sample_id)
        if manifest_record is None:
            raise KeyError(f"No manifest record for sample {sample_id!r}.")

        joined.append(
            {
                **result,
                "ground_truth_activity_counts": ground_truth_counts(
                    manifest_record.get("users"),
                    labels,
                ),
            }
        )

    return joined


def sample_error_vector(
    result: Mapping[str, Any],
    labels: Sequence[str] = ACTIVITY_LABELS,
) -> dict[str, Any]:
    """Per-sample error summary using the same rules as ``evaluate_records``."""
    parse_ok = result.get("parse_status") == "ok"
    raw_prediction = result.get("predicted_activity_counts")

    predicted = _normalized_counts(
        raw_prediction if parse_ok else None,
        labels,
    )
    ground_truth = _normalized_counts(
        result.get("ground_truth_activity_counts"),
        labels,
    )

    exact = (
        parse_ok
        and isinstance(raw_prediction, dict)
        and predicted == ground_truth
    )

    return {
        "exact": exact,
        "count_abs_error": sum(
            abs(predicted[label] - ground_truth[label]) for label in labels
        ),
        "total_person_abs_error": abs(
            sum(predicted.values()) - sum(ground_truth.values())
        ),
    }


def paired_summary(
    v1_results: Sequence[Mapping[str, Any]],
    v2_results: Sequence[Mapping[str, Any]],
    labels: Sequence[str] = ACTIVITY_LABELS,
) -> dict[str, Any]:
    """Compare the two versions sample by sample.

    ``v1_better`` / ``v2_better`` count samples by total activity-count error,
    which is a stricter per-sample measure than exact-match accuracy.
    """
    v1_by_id = {result["sample_id"]: result for result in v1_results}
    v2_by_id = {result["sample_id"]: result for result in v2_results}

    if len(v1_by_id) != len(v1_results) or len(v2_by_id) != len(v2_results):
        raise ValueError("Duplicate sample ids in paired results.")

    if set(v1_by_id) != set(v2_by_id):
        missing_v2 = sorted(set(v1_by_id) - set(v2_by_id))
        missing_v1 = sorted(set(v2_by_id) - set(v1_by_id))
        raise ValueError(
            f"Paired results differ: missing_v2={missing_v2}, "
            f"missing_v1={missing_v1}"
        )

    rows: list[dict[str, Any]] = []
    v1_better = 0
    v2_better = 0
    ties = 0
    exact_agreement = 0

    for sample_id in v1_by_id:
        v1_errors = sample_error_vector(v1_by_id[sample_id], labels)
        v2_errors = sample_error_vector(v2_by_id[sample_id], labels)

        if v1_errors["count_abs_error"] < v2_errors["count_abs_error"]:
            v1_better += 1
        elif v2_errors["count_abs_error"] < v1_errors["count_abs_error"]:
            v2_better += 1
        else:
            ties += 1

        if v1_errors["exact"] == v2_errors["exact"]:
            exact_agreement += 1

        rows.append(
            {
                "sample_id": sample_id,
                "v1": v1_errors,
                "v2": v2_errors,
                "count_error_delta_v2_minus_v1": (
                    v2_errors["count_abs_error"]
                    - v1_errors["count_abs_error"]
                ),
            }
        )

    deltas = [row["count_error_delta_v2_minus_v1"] for row in rows]

    return {
        "sample_count": len(rows),
        "v1_exact_matches": sum(row["v1"]["exact"] for row in rows),
        "v2_exact_matches": sum(row["v2"]["exact"] for row in rows),
        "v1_mean_count_abs_error": fmean(
            row["v1"]["count_abs_error"] for row in rows
        ),
        "v2_mean_count_abs_error": fmean(
            row["v2"]["count_abs_error"] for row in rows
        ),
        "mean_count_error_delta_v2_minus_v1": fmean(deltas),
        "v1_better_samples": v1_better,
        "v2_better_samples": v2_better,
        "tie_samples": ties,
        "exact_decision_agreement": exact_agreement / len(rows),
        "samples": rows,
        "caveat": (
            "Paired descriptive comparison over 20 recordings. The pilot is "
            "coverage-oriented, not a representative sample, so these counts "
            "do not support significance or generalization claims."
        ),
    }
