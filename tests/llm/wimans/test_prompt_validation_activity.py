import copy
import json

from src.llm.wimans.prompt_builder_v2 import LINK_NAMES
from src.llm.wimans.prompt_builder_activity_v2 import (
    build_activity_only_prompt_v2,
)
from src.llm.wimans.prompt_validation import (
    INPUT_PREFIX,
    INPUT_SUFFIX,
    extract_inference_input,
)
from src.llm.wimans.prompt_validation_activity import (
    PROMPT_VERSION_ACTIVITY_V2,
    audit_activity_only_prompt,
)


def make_record():
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
        "sample_id": "record_001",
        "group_id": "group_01",
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


def replace_payload(messages, payload):
    updated = [dict(message) for message in messages]
    user_content = updated[1]["content"]
    prefix, remainder = user_content.split(INPUT_PREFIX, 1)
    _, suffix = remainder.split(INPUT_SUFFIX, 1)

    updated[1]["content"] = (
        prefix
        + INPUT_PREFIX
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + INPUT_SUFFIX
        + suffix
    )
    return updated


def test_valid_activity_only_prompt_passes_audit():
    record = make_record()
    messages = build_activity_only_prompt_v2(record)

    assert audit_activity_only_prompt(messages, record) == []


def test_prompt_uses_activity_only_version():
    record = make_record()
    messages = build_activity_only_prompt_v2(record)
    payload = extract_inference_input(messages)

    assert (
        payload["features"]["prompt_version"]
        == PROMPT_VERSION_ACTIVITY_V2
    )


def test_identifiers_and_ground_truth_are_not_in_payload():
    record = make_record()
    messages = build_activity_only_prompt_v2(record)
    payload = extract_inference_input(messages)
    serialized = json.dumps(payload)

    assert "record_001" not in serialized
    assert "group_01" not in serialized
    assert "number_of_users" not in serialized
    assert "ground_truth" not in serialized


def test_modified_system_prompt_fails_audit():
    record = make_record()
    messages = build_activity_only_prompt_v2(record)
    messages[0]["content"] += "\nPredict the number of people."

    problems = audit_activity_only_prompt(messages, record)

    assert any("System message" in problem for problem in problems)


def test_injected_ground_truth_key_fails_audit():
    record = make_record()
    messages = build_activity_only_prompt_v2(record)
    payload = copy.deepcopy(extract_inference_input(messages))
    payload["features"]["ground_truth_activity"] = "walk"

    modified = replace_payload(messages, payload)
    problems = audit_activity_only_prompt(modified, record)

    assert problems


def test_injected_selection_arm_fails_audit():
    record = make_record()
    messages = build_activity_only_prompt_v2(record)
    payload = copy.deepcopy(extract_inference_input(messages))
    payload["features"]["selection_arm"] = "single_person_activity"

    modified = replace_payload(messages, payload)
    problems = audit_activity_only_prompt(modified, record)

    assert problems


def test_changed_temporal_feature_fails_audit():
    record = make_record()
    messages = build_activity_only_prompt_v2(record)
    payload = copy.deepcopy(extract_inference_input(messages))
    payload["features"]["temporal_sequence"][0][5][0] = 999.0

    modified = replace_payload(messages, payload)
    problems = audit_activity_only_prompt(modified, record)

    assert any("differs from the expected v2 feature payload" in p for p in problems)


def test_wrong_prompt_version_fails_audit():
    record = make_record()
    messages = build_activity_only_prompt_v2(record)
    payload = copy.deepcopy(extract_inference_input(messages))
    payload["features"]["prompt_version"] = "wrong_version"

    modified = replace_payload(messages, payload)
    problems = audit_activity_only_prompt(modified, record)

    assert any("Incorrect activity-only prompt version" in p for p in problems)


def test_null_temporal_deltas_are_preserved():
    record = make_record()
    record["features"]["temporal_sequence"][0]["links"]["link_00"][
        "temporal_delta"
    ] = None

    messages = build_activity_only_prompt_v2(record)
    payload = extract_inference_input(messages)

    assert payload["features"]["temporal_sequence"][0][7][0] is None
    assert audit_activity_only_prompt(messages, record) == []


def test_null_temporal_deltas_are_not_converted_to_zero():
    record = make_record()
    record["features"]["temporal_sequence"][0]["links"]["link_00"][
        "temporal_delta"
    ] = None

    messages = build_activity_only_prompt_v2(record)
    payload = extract_inference_input(messages)

    delta = payload["features"]["temporal_sequence"][0][7][0]

    assert delta is None
    assert delta != 0
