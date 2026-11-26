import json

import pytest

from common.llm.client import LLMClientError
from src.llm.wimans import classifier_activity
from src.llm.wimans.classifier import ClassifierError
from src.llm.wimans.prompt_validation import extract_inference_input
from src.llm.wimans.prompt_validation_activity import (
    audit_activity_only_prompt,
)


def make_record(sample_id="record_001", group_id="group_01"):
    from src.llm.wimans.prompt_builder_v2 import LINK_NAMES

    links = {
        name: {
            "window_mean": 1.0,
            "window_std": 0.2,
            "temporal_delta": 0.1,
        }
        for name in LINK_NAMES
    }

    matrix = [
        [1.0 if i == j else 0.5 for j in range(9)]
        for i in range(9)
    ]

    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "environment": "Office",
        "wifi_band": "5GHz",
        "number_of_users": 1,
        "ground_truth": {"activity": "walk"},
        "features": {
            "feature_version": "wimans_recording_v2",
            "window_size_packets": 100,
            "num_packets": 100,
            "num_windows": 1,
            "complete_windows": 1,
            "temporal_sequence": [
                {
                    "window_index": 0,
                    "start_packet": 0,
                    "end_packet": 100,
                    "num_packets": 100,
                    "is_partial": False,
                    "links": links,
                }
            ],
            "cross_link_synchronization": {
                "method": "pearson",
                "links": LINK_NAMES,
                "matrix": matrix,
                "minimum_complete_windows": 2,
                "unique_off_diagonal_pairs": 36,
                "defined_off_diagonal_pairs": 36,
            },
            "movement_intensity": {
                name: [0.1] for name in LINK_NAMES
            },
        },
    }


def fake_llm_result(content, model="mock-model"):
    return {
        "content": content,
        "model": model,
        "latency_ms": 12.5,
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "raw_response": {"mock": True},
    }


def test_unconfigured_client_raises_without_calling_api(monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("API client must not be called")

    monkeypatch.setattr(classifier_activity, "is_configured", lambda: False)
    monkeypatch.setattr(classifier_activity, "chat_completion", unexpected_call)

    with pytest.raises(ClassifierError, match="LLM client is not configured"):
        classifier_activity.classify_record(make_record())


def test_successful_classification_returns_expected_result(monkeypatch):
    response = (
        '{"activity":"walk","confidence":0.85,'
        '"reason":"Sustained temporal changes."}'
    )

    monkeypatch.setattr(classifier_activity, "is_configured", lambda: True)
    monkeypatch.setattr(
        classifier_activity,
        "chat_completion",
        lambda *args, **kwargs: fake_llm_result(response),
    )

    result = classifier_activity.classify_record(make_record())

    assert result["activity"] == "walk"
    assert result["confidence"] == 0.85
    assert result["reason"] == "Sustained temporal changes."
    assert result["parse_status"] == "ok"
    assert result["model"] == "mock-model"
    assert result["latency_ms"] == 12.5
    assert result["token_usage"]["prompt_tokens"] == 100
    assert result["raw_response"] == {"mock": True}
    assert result["error"] is None
    assert result["prompt_mode"] == "activity_only_zero_shot"
    assert result["feature_version"] == "wimans_recording_v2"


def test_prompt_passes_audit_and_excludes_record_identifiers(monkeypatch):
    response = (
        '{"activity":"nothing","confidence":0.9,'
        '"reason":"No meaningful movement is observed."}'
    )

    monkeypatch.setattr(classifier_activity, "is_configured", lambda: True)
    monkeypatch.setattr(
        classifier_activity,
        "chat_completion",
        lambda *args, **kwargs: fake_llm_result(response),
    )

    record = make_record()
    result = classifier_activity.classify_record(record)
    messages = result["prompt_messages"]

    assert audit_activity_only_prompt(messages, record) == []

    payload = extract_inference_input(messages)
    serialized = json.dumps(payload)

    assert "record_001" not in serialized
    assert "group_01" not in serialized
    assert "number_of_users" not in serialized
    assert "ground_truth" not in serialized


def test_invalid_model_activity_is_reported_as_parse_error(monkeypatch):
    response = (
        '{"activity":"running","confidence":0.8,'
        '"reason":"Movement detected."}'
    )

    monkeypatch.setattr(classifier_activity, "is_configured", lambda: True)
    monkeypatch.setattr(
        classifier_activity,
        "chat_completion",
        lambda *args, **kwargs: fake_llm_result(response),
    )

    result = classifier_activity.classify_record(make_record())

    assert result["activity"] is None
    assert result["parse_status"] == "schema_error"
    assert result["error"] is not None


def test_client_error_is_captured(monkeypatch):
    def fail_request(*args, **kwargs):
        raise LLMClientError("Mock connection failure")

    monkeypatch.setattr(classifier_activity, "is_configured", lambda: True)
    monkeypatch.setattr(
        classifier_activity,
        "chat_completion",
        fail_request,
    )

    result = classifier_activity.classify_record(make_record())

    assert result["activity"] is None
    assert result["confidence"] is None
    assert result["parse_status"] == "client_error"
    assert result["error"] == "Mock connection failure"
    assert result["latency_ms"] is None


def test_batch_preserves_record_order(monkeypatch):
    response = (
        '{"activity":"walk","confidence":0.8,'
        '"reason":"Movement detected."}'
    )

    monkeypatch.setattr(classifier_activity, "is_configured", lambda: True)
    monkeypatch.setattr(
        classifier_activity,
        "chat_completion",
        lambda *args, **kwargs: fake_llm_result(response),
    )

    records = [
        make_record("record_A", "group_A"),
        make_record("record_B", "group_B"),
        make_record("record_C", "group_C"),
    ]

    results = classifier_activity.classify_batch(records)

    assert [item["sample_id"] for item in results] == [
        "record_A",
        "record_B",
        "record_C",
    ]
    assert [item["record_index"] for item in results] == [0, 1, 2]
    assert all(item["parse_status"] == "ok" for item in results)
