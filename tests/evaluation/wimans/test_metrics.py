import pytest

from evaluation.wimans.metrics import evaluate_records
from llm.wimans.prompt_builder import ACTIVITY_LABELS


def counts(**overrides):
    result = {label: 0 for label in ACTIVITY_LABELS}
    result.update(overrides)
    return result


def test_perfect_prediction():
    record = {
        "ground_truth_activity_counts": counts(walk=2, wave=1),
        "predicted_activity_counts": counts(walk=2, wave=1),
        "parse_status": "ok",
        "latency_ms": 100,
        "token_usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        },
    }

    result = evaluate_records([record])

    assert result["exact_multiset_accuracy"] == 1.0
    assert result["count_mae"] == 0.0
    assert result["total_person_count_mae"] == 0.0
    assert result["micro_f1"] == 1.0
    assert result["parse_success_rate"] == 1.0
    assert result["mean_latency_ms"] == 100.0
    assert result["token_usage"]["total_tokens"] == 70


def test_failed_parse_is_counted_as_incorrect():
    record = {
        "ground_truth_activity_counts": counts(walk=1),
        "predicted_activity_counts": None,
        "parse_status": "json_error",
    }

    result = evaluate_records([record])

    assert result["sample_count"] == 1
    assert result["exact_multiset_accuracy"] == 0.0
    assert result["exact_multiset_matches"] == 0
    assert result["count_mae"] == pytest.approx(1 / 9)
    assert result["total_person_count_mae"] == 1.0
    assert result["parse_success_rate"] == 0.0
    assert result["per_activity"]["walk"]["recall"] == 0.0
    assert result["per_activity"]["walk"]["false_negative"] == 1
    assert result["per_activity"]["walk"]["support"] == 1


def test_failed_parse_cannot_match_all_zero_ground_truth():
    record = {
        "ground_truth_activity_counts": counts(),
        "predicted_activity_counts": None,
        "parse_status": "client_error",
    }

    result = evaluate_records([record])

    assert result["exact_multiset_accuracy"] == 0.0
    assert result["count_mae"] == 0.0
    assert result["total_person_count_mae"] == 0.0
    assert result["parse_success_rate"] == 0.0


def test_counts_are_treated_as_multiplicities():
    record = {
        "ground_truth_activity_counts": counts(walk=2, wave=1),
        "predicted_activity_counts": counts(walk=1, wave=2),
        "parse_status": "ok",
    }

    result = evaluate_records([record])

    walk = result["per_activity"]["walk"]
    assert walk["true_positive"] == 1
    assert walk["false_positive"] == 0
    assert walk["false_negative"] == 1
    assert walk["support"] == 2
    assert walk["precision"] == 1.0
    assert walk["recall"] == 0.5
    assert walk["f1"] == pytest.approx(2 / 3)

    wave = result["per_activity"]["wave"]
    assert wave["true_positive"] == 1
    assert wave["false_positive"] == 1
    assert wave["false_negative"] == 0
    assert wave["precision"] == 0.5
    assert wave["recall"] == 1.0
    assert wave["f1"] == pytest.approx(2 / 3)

    assert result["exact_multiset_accuracy"] == 0.0
    assert result["count_mae"] == pytest.approx(2 / 9)
    assert result["total_person_count_mae"] == 0.0
    assert result["micro_precision"] == pytest.approx(2 / 3)
    assert result["micro_recall"] == pytest.approx(2 / 3)
    assert result["micro_f1"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 9)


def test_latency_and_token_usage_cover_only_available_values():
    records = [
        {
            "ground_truth_activity_counts": counts(),
            "predicted_activity_counts": counts(),
            "parse_status": "ok",
            "latency_ms": 100,
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
        {
            "ground_truth_activity_counts": counts(),
            "predicted_activity_counts": counts(),
            "parse_status": "ok",
            "latency_ms": 50,
            "token_usage": {"total_tokens": 7},
        },
        {
            "ground_truth_activity_counts": counts(),
            "predicted_activity_counts": counts(),
            "parse_status": "ok",
            "latency_ms": None,
            "token_usage": None,
        },
    ]

    result = evaluate_records(records)

    assert result["mean_latency_ms"] == 75.0
    assert result["latency_sample_count"] == 2
    assert result["token_usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 22,
    }
    assert result["token_usage_record_count"] == 2


def test_empty_input_returns_zeroed_metrics():
    result = evaluate_records([])

    assert result["sample_count"] == 0
    assert result["exact_multiset_accuracy"] == 0.0
    assert result["exact_multiset_matches"] == 0
    assert result["count_mae"] == 0.0
    assert result["total_person_count_mae"] == 0.0
    assert result["micro_f1"] == 0.0
    assert result["macro_f1"] == 0.0
    assert result["parse_success_rate"] == 0.0
    assert result["mean_latency_ms"] is None
    assert result["latency_sample_count"] == 0
    assert result["token_usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert result["token_usage_record_count"] == 0
    assert set(result["per_activity"]) == set(ACTIVITY_LABELS)


def test_custom_labels():
    record = {
        "ground_truth_activity_counts": {"X": 1, "Y": 0},
        "predicted_activity_counts": {"X": 0, "Y": 2},
        "parse_status": "ok",
    }

    result = evaluate_records([record], labels=("X", "Y"))

    assert set(result["per_activity"]) == {"X", "Y"}
    assert result["count_mae"] == pytest.approx(1.5)
    assert result["total_person_count_mae"] == 1.0
    assert result["per_activity"]["X"]["support"] == 1
    assert result["per_activity"]["Y"]["false_positive"] == 2
