"""Leakage and structure audit for exactly-one-person activity-only prompts."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .prompt_builder_activity_v2 import (
    PROMPT_VERSION_ACTIVITY_V2,
    SYSTEM_PROMPT_ACTIVITY_V2,
    TASK_INSTRUCTION_ACTIVITY_V2,
)
from .prompt_builder_v2 import build_zero_shot_prompt_v2
from .prompt_validation import (
    INPUT_PREFIX,
    ALLOWED_TOP_LEVEL_KEYS,
    V2_FEATURE_KEYS,
    V2_SYNCHRONIZATION_KEYS,
    audit_inference_input,
    collect_keys,
    collect_strings,
    extract_inference_input,
)

ARM_KEYS = frozenset({
    "arm",
    "selection_arm",
    "selection",
    "ground_truth",
    "ground_truth_activity",
    "ground_truth_activity_counts",
    "activity_label",
    "activity_labels",
})

ARM_VALUES = frozenset({
    "single_person_activity",
    "zero_person_detection",
    "single_person",
    "zero_person",
})

USER_PREFIX = TASK_INSTRUCTION_ACTIVITY_V2 + "\n\n" + INPUT_PREFIX


def audit_activity_only_prompt(
    messages: Sequence[Mapping[str, str]],
    record: Mapping[str, Any],
) -> list[str]:
    """Validate the activity-only prompt and its source feature payload."""

    problems: list[str] = []

    if len(messages) != 2:
        return [f"Prompt must contain exactly 2 messages, got {len(messages)}."]

    if messages[0].get("role") != "system":
        problems.append("First message must have role 'system'.")
    if messages[0].get("content") != SYSTEM_PROMPT_ACTIVITY_V2:
        problems.append("System message does not match SYSTEM_PROMPT_ACTIVITY_V2.")

    if messages[1].get("role") != "user":
        problems.append("Second message must have role 'user'.")

    content = messages[1].get("content", "")
    if not content.startswith(USER_PREFIX):
        problems.append("User message must begin with the activity-only instruction.")

    try:
        payload = extract_inference_input(messages)
    except (ValueError, json.JSONDecodeError) as exc:
        return problems + [f"Cannot parse activity-only prompt payload: {exc}"]

    problems += audit_inference_input(
        payload,
        allowed_feature_keys=V2_FEATURE_KEYS | V2_SYNCHRONIZATION_KEYS,
        sample_id=str(record.get("sample_id", "")) or None,
    )

    if set(payload) != ALLOWED_TOP_LEVEL_KEYS:
        return problems

    features = payload.get("features")
    if not isinstance(features, Mapping):
        return problems

    if features.get("feature_version") != "wimans_recording_v2":
        problems.append("Activity-only prompt must use wimans_recording_v2.")

    if features.get("prompt_version") != PROMPT_VERSION_ACTIVITY_V2:
        problems.append("Incorrect activity-only prompt version.")

    leaked_keys = collect_keys(payload) & ARM_KEYS
    if leaked_keys:
        problems.append(f"Leaked selection or ground-truth keys: {sorted(leaked_keys)}.")

    leaked_values = sorted(set(collect_strings(payload)) & ARM_VALUES)
    if leaked_values:
        problems.append(f"Leaked selection-arm values: {leaked_values}.")

    if payload.get("environment") != record.get("environment"):
        problems.append("Prompt environment differs from the source record.")

    if payload.get("wifi_band") != record.get("wifi_band"):
        problems.append("Prompt wifi_band differs from the source record.")

    # Rebuild the expected v2 payload. The only permitted difference is the
    # prompt_version field identifying this activity-only experiment.
    try:
        reference_messages = build_zero_shot_prompt_v2(dict(record))
        reference_payload = extract_inference_input(reference_messages)
        reference_payload["features"]["prompt_version"] = (
            PROMPT_VERSION_ACTIVITY_V2
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return problems + [f"Cannot build reference v2 payload: {exc}"]

    if payload != reference_payload:
        problems.append(
            "Activity-only payload differs from the expected v2 feature payload."
        )

    return problems
