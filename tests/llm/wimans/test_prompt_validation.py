import json

import numpy as np

from llm.wimans.prompt_builder import build_zero_shot_prompt
from llm.wimans.prompt_builder_v2 import build_zero_shot_prompt_v2
from llm.wimans.prompt_validation import (
    audit_inference_input,
    audit_v1_prompt,
    audit_v2_prompt,
    extract_inference_input,
)
from preprocessing.wimans.extract_features_v2 import (
    LINK_NAMES as EXTRACTOR_LINK_NAMES,
    extract_recording_features_v2,
)

STAT_KEYS = ("mean", "std", "p10", "p50", "p90")


def make_link(value):
    return {
        "signal_mean": value,
        "signal_std": value / 2,
        "signal_p10": value - 1,
        "signal_p50": value,
        "signal_p90": value + 1,
        "window_mean_variability": 0.5,
        "temporal_delta_mean": 0.25,
        "temporal_delta_std": 0.1,
    }


def make_v1_record(sample_id="act_999_1", group_id=999):
    summary_window = {key: 1.0 + index for index, key in enumerate(STAT_KEYS)}
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "environment": "classroom",
        "wifi_band": "5",
        "csi_shape": [200, 3, 3, 30],
        "features": {
            "feature_version": "wimans_recording_v1",
            "num_windows": 2,
            "window_size_packets": 100,
            "recording_summary": {
                "signal_mean": 1.0,
                "signal_std": 0.5,
                "window_mean": dict(summary_window),
                "window_std": dict(summary_window),
            },
            "links": {
                name: make_link(float(index) + 1.0)
                for index, name in enumerate(EXTRACTOR_LINK_NAMES)
            },
        },
    }


def make_v2_record(sample_id="act_999_1", group_id=999):
    rng = np.random.default_rng(0)
    csi = rng.random((40, 3, 3, 30)) + 1.0
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "environment": "classroom",
        "wifi_band": "5",
        "csi_shape": list(csi.shape),
        "features": extract_recording_features_v2(csi, window_size=10),
    }


def test_v1_prompt_passes_audit():
    record = make_v1_record()
    messages = build_zero_shot_prompt(record)

    assert audit_v1_prompt(messages, record) == []


def test_v2_prompt_passes_audit():
    record = make_v2_record()
    messages = build_zero_shot_prompt_v2(record)

    assert audit_v2_prompt(messages, record) == []


def test_v2_prompt_omits_movement_intensity():
    record = make_v2_record()
    payload = extract_inference_input(build_zero_shot_prompt_v2(record))

    assert "movement_intensity" not in payload["features"]


def test_v1_prompt_leaks_features_wholesale_and_audit_catches_it():
    record = make_v1_record()
    record["features"]["group_id"] = record["group_id"]

    messages = build_zero_shot_prompt(record)
    problems = audit_v1_prompt(messages, record)

    assert any("group_id" in problem for problem in problems)


def test_audit_rejects_sample_id_value_in_payload():
    record = make_v1_record()
    messages = build_zero_shot_prompt(record)

    payload = extract_inference_input(messages)
    payload["features"]["notes"] = record["sample_id"]

    problems = audit_inference_input(
        payload,
        allowed_feature_keys=frozenset(),
        sample_id=record["sample_id"],
    )

    assert any("notes" in problem for problem in problems)
    assert any("Sample id" in problem for problem in problems)


def test_audit_rejects_activity_label_values():
    record = make_v1_record()
    payload = extract_inference_input(build_zero_shot_prompt(record))
    payload["features"]["feature_version"] = "walk"

    problems = audit_inference_input(
        payload,
        allowed_feature_keys=frozenset({"feature_version"}),
    )

    assert any("Activity labels" in problem for problem in problems)


def test_audit_detects_feature_substitution():
    record = make_v1_record()
    messages = build_zero_shot_prompt(record)

    tampered = make_v1_record()
    tampered["features"]["num_windows"] = 999

    problems = audit_v1_prompt(messages, tampered)

    assert any("differ from the source record" in problem for problem in problems)


def test_v2_audit_rejects_non_compact_synchronization():
    record = make_v2_record()
    messages = build_zero_shot_prompt_v2(record)

    payload = extract_inference_input(messages)
    payload["features"]["cross_link_synchronization"]["links"] = list(
        EXTRACTOR_LINK_NAMES
    )

    user_content = messages[-1]["content"]
    injected = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    tampered_messages = [
        messages[0],
        {
            "role": "user",
            "content": user_content.split("CSI-derived input:\n", 1)[0]
            + "CSI-derived input:\n"
            + injected
            + "\n\nReturn the required JSON object only.",
        },
    ]

    problems = audit_v2_prompt(tampered_messages, record)

    assert any("synchronization" in problem for problem in problems)
    assert any("links" in problem for problem in problems)
