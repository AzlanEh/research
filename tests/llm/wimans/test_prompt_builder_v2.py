import json

import pytest

from llm.wimans.prompt_builder_v2 import (
    LINK_NAMES,
    PROMPT_VERSION,
    SEQUENCE_SCHEMA,
    build_zero_shot_prompt_v2,
    prompt_version_v2,
)


def make_record():
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
        "sample_id": "act_1_1",
        "group_id": 1,
        "environment": "classroom",
        "wifi_band": "2.4",
        "number_of_users": 2,
        "users": [{"activity": "walk", "location": "A"}],
        "features": {
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
        },
    }


def get_payload(record):
    messages = build_zero_shot_prompt_v2(record)
    content = messages[1]["content"]
    encoded = content.split("CSI-derived input:\n", 1)[1]
    encoded = encoded.split(
        "\n\nReturn the required JSON object only.", 1
    )[0]
    return json.loads(encoded)


def test_prompt_version():
    assert PROMPT_VERSION == "wimans_counts_zero_shot_v2_compact"
    assert prompt_version_v2() == PROMPT_VERSION


def test_builds_system_and_user_messages():
    messages = build_zero_shot_prompt_v2(make_record())

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_compact_sequence_preserves_all_measurements():
    payload = get_payload(make_record())
    features = payload["features"]

    assert features["feature_version"] == "wimans_recording_v2"
    assert features["prompt_version"] == PROMPT_VERSION
    assert features["link_order"] == LINK_NAMES
    assert features["temporal_sequence_schema"] == SEQUENCE_SCHEMA

    row = features["temporal_sequence"][0]

    assert row[:5] == [0, 0, 100, 100, False]
    assert row[5] == [50.0] * 9
    assert row[6] == [10.0] * 9
    assert row[7] == [2.0] * 9


def test_preserves_synchronization_matrix():
    payload = get_payload(make_record())
    matrix = payload["features"]["cross_link_synchronization"]["matrix"]

    assert len(matrix) == 9
    assert all(len(row) == 9 for row in matrix)
    assert matrix[0][0] == 1.0
    assert matrix[0][1] == 0.5


def test_omits_duplicate_movement_intensity():
    payload = get_payload(make_record())

    assert "movement_intensity" not in payload["features"]


def test_excludes_identifiers_and_ground_truth():
    payload = get_payload(make_record())
    serialized = json.dumps(payload).lower()

    for forbidden in (
        "sample_id",
        "group_id",
        "number_of_users",
        "users",
        "activity",
        "location",
        "act_1_1",
    ):
        assert forbidden not in serialized


def test_includes_inference_metadata():
    payload = get_payload(make_record())

    assert payload["environment"] == "classroom"
    assert payload["wifi_band"] == "2.4"


def test_rejects_wrong_feature_version():
    record = make_record()
    record["features"]["feature_version"] = "wimans_recording_v1"

    with pytest.raises(ValueError):
        build_zero_shot_prompt_v2(record)


def test_rejects_missing_temporal_sequence():
    record = make_record()
    del record["features"]["temporal_sequence"]

    with pytest.raises(ValueError):
        build_zero_shot_prompt_v2(record)


def test_rejects_invalid_synchronization_matrix():
    record = make_record()
    record["features"]["cross_link_synchronization"]["matrix"] = [[1.0]]

    with pytest.raises(ValueError):
        build_zero_shot_prompt_v2(record)


def test_rejects_nan_values():
    record = make_record()
    record["features"]["temporal_sequence"][0]["links"]["link_00"][
        "window_mean"
    ] = float("nan")

    with pytest.raises(ValueError):
        build_zero_shot_prompt_v2(record)
