import pytest

from evaluation.wimans.paired import (
    ground_truth_counts,
    join_ground_truth,
    paired_summary,
    sample_error_vector,
)
from llm.wimans.prompt_builder import ACTIVITY_LABELS


def counts(**overrides):
    result = {label: 0 for label in ACTIVITY_LABELS}
    result.update(overrides)
    return result


def result(sample_id, predicted=None, status="ok", ground_truth=None):
    record = {
        "sample_id": sample_id,
        "parse_status": status,
        "predicted_activity_counts": predicted,
    }
    if ground_truth is not None:
        record["ground_truth_activity_counts"] = ground_truth
    return record


def test_ground_truth_counts_from_users():
    users = [
        {"activity": "walk"},
        {"activity": "walk"},
        {"activity": "wave"},
    ]

    assert ground_truth_counts(users)["walk"] == 2
    assert ground_truth_counts(users)["wave"] == 1
    assert sum(ground_truth_counts(users).values()) == 3


def test_ground_truth_counts_zero_users():
    assert ground_truth_counts([]) == counts()
    assert ground_truth_counts(None) == counts()


def test_ground_truth_counts_rejects_unknown_activity():
    with pytest.raises(ValueError):
        ground_truth_counts([{"activity": "dance"}])
    with pytest.raises(ValueError):
        ground_truth_counts([{}])


def test_join_ground_truth_does_not_mutate_input():
    original = result("act_1_1", predicted=counts())
    manifest = {"act_1_1": {"users": [{"activity": "jump"}]}}

    joined = join_ground_truth([original], manifest)

    assert "ground_truth_activity_counts" not in original
    assert joined[0]["ground_truth_activity_counts"]["jump"] == 1
    assert joined[0]["sample_id"] == "act_1_1"


def test_join_ground_truth_rejects_preexisting_ground_truth():
    leaked = result("act_1_1", predicted=counts(), ground_truth=counts())

    with pytest.raises(ValueError):
        join_ground_truth([leaked], {"act_1_1": {"users": []}})


def test_join_ground_truth_requires_manifest_row():
    with pytest.raises(KeyError):
        join_ground_truth([result("act_1_1", predicted=counts())], {})


def test_sample_error_vector_failed_parse_is_all_zero_and_not_exact():
    record = result(
        "act_1_1",
        predicted=None,
        status="json_error",
        ground_truth=counts(),
    )

    errors = sample_error_vector(record)

    assert errors["exact"] is False
    assert errors["count_abs_error"] == 0
    assert errors["total_person_abs_error"] == 0


def test_paired_summary_counts_wins_ties_and_agreement():
    ground_truth_walk = counts(walk=1)

    v1 = [
        result("s1", counts(walk=1), ground_truth=ground_truth_walk),
        result("s2", counts(), ground_truth=counts()),
        result("s3", None, status="json_error", ground_truth=counts()),
    ]
    v2 = [
        result("s1", counts(), ground_truth=ground_truth_walk),
        result("s2", counts(), ground_truth=counts()),
        result("s3", counts(), ground_truth=counts()),
    ]

    summary = paired_summary(v1, v2)

    assert summary["sample_count"] == 3
    assert summary["v1_exact_matches"] == 2
    assert summary["v2_exact_matches"] == 2
    assert summary["v1_better_samples"] == 1
    assert summary["v2_better_samples"] == 0
    assert summary["tie_samples"] == 2
    assert summary["mean_count_error_delta_v2_minus_v1"] == pytest.approx(1 / 3)
    assert summary["exact_decision_agreement"] == pytest.approx(1 / 3)


def test_paired_summary_rejects_mismatched_sample_sets():
    v1 = [result("s1", counts())]
    v2 = [result("s2", counts())]

    with pytest.raises(ValueError):
        paired_summary(v1, v2)
