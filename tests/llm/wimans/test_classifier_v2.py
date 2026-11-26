import json
from unittest.mock import patch

import pytest

from common.llm.client import LLMClientError
from llm.wimans import classifier_v2
from llm.wimans.classifier import ClassifierError
from llm.wimans.prompt_builder import ACTIVITY_LABELS

LINK_NAMES = [
    f"link_{tx}{rx}"
    for tx in range(3)
    for rx in range(3)
]


def make_features():
    links = {
        name: {
            "window_mean": 50.0,
            "window_std": 10.0,
            "temporal_delta": 2.0,
        }
        for name in LINK_NAMES
    }
    matrix = [
        [1.0 if i == j else 0.5 for j in range(9)]
        for i in range(9)
    ]

    return {
        "feature_version": "wimans_recording_v2",
        "window_size_packets": 100,
        "num_packets": 100,
        "num_windows": 1,
        "complete_windows": 1,
        "temporal_sequence": [{
            "window_index": 0,
            "start_packet": 0,
            "end_packet": 100,
            "num_packets": 100,
            "is_partial": False,
            "links": links,
        }],
        "movement_intensity": {
            name: [2.0] for name in LINK_NAMES
        },
        "cross_link_synchronization": {
            "method": "pearson",
            "links": LINK_NAMES,
            "matrix": matrix,
            "unique_off_diagonal_pairs": 36,
            "defined_off_diagonal_pairs": 36,
        },
    }


def make_record(sample_id="act_1_1"):
    return {
        "sample_id": sample_id,
        "group_id": 1,
        "environment": "classroom",
        "wifi_band": "2.4",
        "number_of_users": 2,
        "users": [{"activity": "walk", "location": "A"}],
        "features": make_features(),
    }


def make_llm_result(content=None):
    if content is None:
        content = json.dumps({
            "predicted_activity_counts": {
                label: 0 for label in ACTIVITY_LABELS
            },
            "confidence": 0.8,
            "reason": "Test prediction.",
        })

    return {
        "content": content,
        "model": "test-model",
        "latency_ms": 100,
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "raw_response": {"test": True},
    }


EXPECTED_RESULT_KEYS = {
    "sample_id",
    "group_id",
    "environment",
    "wifi_band",
    "prompt_version",
    "prompt_mode",
    "prompt_messages",
    "predicted_activity_counts",
    "confidence",
    "reason",
    "parse_status",
    "model",
    "latency_ms",
    "token_usage",
    "raw_response",
    "error",
}


def test_success_mirrors_v1_result_contract():
    with patch.object(classifier_v2, "is_configured", return_value=True), \
            patch.object(
                classifier_v2,
                "chat_completion",
                return_value=make_llm_result(),
            ) as chat:
        result = classifier_v2.classify_record(make_record())

    assert set(result) == EXPECTED_RESULT_KEYS
    assert result["sample_id"] == "act_1_1"
    assert result["group_id"] == 1
    assert result["environment"] == "classroom"
    assert result["wifi_band"] == "2.4"
    assert result["prompt_version"] == "wimans_counts_zero_shot_v2_compact"
    assert result["prompt_mode"] == "zero_shot"
    assert result["parse_status"] == "ok"
    assert result["predicted_activity_counts"] == {
        label: 0 for label in ACTIVITY_LABELS
    }
    assert result["confidence"] == 0.8
    assert result["reason"] == "Test prediction."
    assert result["model"] == "test-model"
    assert result["latency_ms"] == 100
    assert result["token_usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
    }
    assert result["raw_response"] == {"test": True}
    assert result["error"] is None

    messages = chat.call_args.args[0]
    assert messages == result["prompt_messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_prompt_contains_no_identifiers_or_ground_truth():
    with patch.object(classifier_v2, "is_configured", return_value=True), \
            patch.object(
                classifier_v2,
                "chat_completion",
                return_value=make_llm_result(),
            ):
        result = classifier_v2.classify_record(make_record())

    user_content = result["prompt_messages"][1]["content"]
    for forbidden in (
        "act_1_1",
        "number_of_users",
        "location",
        '"users"',
        '"walk"',
    ):
        assert forbidden not in user_content

    assert '"prompt_version":"wimans_counts_zero_shot_v2_compact"' in (
        user_content
    )


def test_client_error_returns_structured_result():
    with patch.object(classifier_v2, "is_configured", return_value=True), \
            patch.object(
                classifier_v2,
                "chat_completion",
                side_effect=LLMClientError("HTTP 429 rate limited"),
            ):
        result = classifier_v2.classify_record(make_record())

    assert set(result) == EXPECTED_RESULT_KEYS
    assert result["parse_status"] == "client_error"
    assert result["predicted_activity_counts"] is None
    assert result["confidence"] is None
    assert result["reason"] is None
    assert result["latency_ms"] is None
    assert result["token_usage"] is None
    assert result["raw_response"] is None
    assert "429" in result["error"]
    assert result["prompt_version"] == "wimans_counts_zero_shot_v2_compact"


def test_parse_failure_returns_structured_result():
    with patch.object(classifier_v2, "is_configured", return_value=True), \
            patch.object(
                classifier_v2,
                "chat_completion",
                return_value=make_llm_result("not json"),
            ):
        result = classifier_v2.classify_record(make_record())

    assert result["parse_status"] == "json_error"
    assert result["predicted_activity_counts"] is None
    assert result["error"]


def test_batch_preserves_order_and_adds_record_index():
    with patch.object(classifier_v2, "is_configured", return_value=True), \
            patch.object(
                classifier_v2,
                "chat_completion",
                side_effect=[
                    make_llm_result(),
                    make_llm_result("not json"),
                ],
            ):
        results = classifier_v2.classify_batch([
            make_record("sample_a"),
            make_record("sample_b"),
        ])

    assert [result["record_index"] for result in results] == [0, 1]
    assert [result["sample_id"] for result in results] == [
        "sample_a",
        "sample_b",
    ]
    assert [result["parse_status"] for result in results] == [
        "ok",
        "json_error",
    ]


def test_unconfigured_client_raises_classifier_error():
    with patch.object(classifier_v2, "is_configured", return_value=False), \
            patch.object(classifier_v2, "chat_completion") as chat:
        with pytest.raises(ClassifierError):
            classifier_v2.classify_record(make_record())

    chat.assert_not_called()


def test_malformed_features_raise_value_error_before_api_call():
    record = make_record()
    record["features"]["feature_version"] = "wimans_recording_v1"

    with patch.object(classifier_v2, "is_configured", return_value=True), \
            patch.object(classifier_v2, "chat_completion") as chat:
        with pytest.raises(ValueError):
            classifier_v2.classify_record(record)

    chat.assert_not_called()
