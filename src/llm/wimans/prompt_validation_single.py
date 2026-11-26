"""Dry-run leakage and structure validation for single-person prompts.

The checks reuse the frozen v1/v2 allowlists, forbidden-key rules and payload
extraction from ``prompt_validation``. Only the system message and the task
instruction are new, so the feature block must still match the baseline
representation of the same feature version.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .prompt_builder import ACTIVITY_LABELS
from .prompt_builder_single import (
    SYSTEM_PROMPT_SINGLE,
    TASK_INSTRUCTION,
    build_single_person_prompt,
)
from .prompt_builder_v2 import LINK_NAMES
from .prompt_validation import (
    ALLOWED_TOP_LEVEL_KEYS,
    FORBIDDEN_KEYS,
    INPUT_PREFIX,
    V1_FEATURE_KEYS,
    V1_LINK_KEYS,
    V1_STAT_KEYS,
    V1_SUMMARY_KEYS,
    V2_FEATURE_KEYS,
    V2_SYNCHRONIZATION_KEYS,
    collect_keys,
    collect_strings,
    extract_inference_input,
)

V1_ALLOWED_FEATURE_KEYS = (
    V1_FEATURE_KEYS
    | V1_LINK_KEYS
    | V1_SUMMARY_KEYS
    | V1_STAT_KEYS
    | frozenset(LINK_NAMES)
)

V2_ALLOWED_FEATURE_KEYS = V2_FEATURE_KEYS | V2_SYNCHRONIZATION_KEYS

# Selection-arm markers must never reach the model: they would reveal which
# evaluation arm a recording belongs to, and therefore its occupancy ground
# truth.
ARM_KEYS = frozenset(
    {
        "arm",
        "selection_arm",
        "selection",
        "ground_truth",
        "ground_truth_activity",
        "ground_truth_activity_counts",
    }
)

ARM_VALUES = frozenset(
    {
        "single_person_activity",
        "zero_person_detection",
        "single_person",
        "zero_person",
    }
)

USER_PREFIX = TASK_INSTRUCTION + "\n\n" + INPUT_PREFIX


def audit_single_person_prompt(
    messages: Sequence[Mapping[str, str]],
    record: Mapping[str, Any],
    *,
    feature_version: str,
) -> list[str]:
    """Validate one single-person prompt against its source feature record."""
    problems: list[str] = []

    if feature_version not in {"v1", "v2"}:
        return ["feature_version must be 'v1' or 'v2'."]

    if len(messages) != 2:
        return [f"Prompt must contain exactly 2 messages, got {len(messages)}."]

    if messages[0].get("role") != "system":
        problems.append("First message must have role 'system'.")
    if messages[0].get("content") != SYSTEM_PROMPT_SINGLE:
        problems.append("System message does not match SYSTEM_PROMPT_SINGLE.")

    if messages[1].get("role") != "user":
        problems.append("Second message must have role 'user'.")

    content = messages[1].get("content", "")
    if not content.startswith(USER_PREFIX):
        problems.append("User message must begin with the single-person task instruction.")

    try:
        payload = extract_inference_input(messages)
    except (ValueError, json.JSONDecodeError) as exc:
        return problems + [f"Cannot parse single-person prompt payload: {exc}"]

    if set(payload) != ALLOWED_TOP_LEVEL_KEYS:
        problems.append(
            "Top-level keys must be "
            f"{sorted(ALLOWED_TOP_LEVEL_KEYS)}, got {sorted(payload)}."
        )

    # The feature block must be exactly what the baseline builder sends for
    # this feature version, so v1/v2 remain comparable to their baselines.
    try:
        reference = build_single_person_prompt(
            dict(record),
            feature_version=feature_version,
        )
        reference_payload = extract_inference_input(reference)
    except (ValueError, json.JSONDecodeError) as exc:
        return problems + [f"Cannot build reference payload: {exc}"]

    if payload != reference_payload:
        problems.append(
            "Prompt payload differs from the expected feature block for "
            f"{feature_version}."
        )

    if payload.get("environment") != record.get("environment"):
        problems.append("Prompt environment differs from the source record.")
    if payload.get("wifi_band") != record.get("wifi_band"):
        problems.append("Prompt wifi_band differs from the source record.")

    allowed_feature_keys = (
        V1_ALLOWED_FEATURE_KEYS
        if feature_version == "v1"
        else V2_ALLOWED_FEATURE_KEYS
    )
    allowed_keys = allowed_feature_keys | ALLOWED_TOP_LEVEL_KEYS

    unexpected = collect_keys(payload) - allowed_keys
    if unexpected:
        problems.append(
            f"Unexpected keys in prompt payload: {sorted(unexpected)}."
        )

    leaked_keys = collect_keys(payload) & (FORBIDDEN_KEYS | ARM_KEYS)
    if leaked_keys:
        problems.append(f"Leaked forbidden keys: {sorted(leaked_keys)}.")

    strings = set(collect_strings(payload))

    leaked_labels = sorted(strings & set(ACTIVITY_LABELS))
    if leaked_labels:
        problems.append(f"Activity labels leaked as values: {leaked_labels}.")

    leaked_arm_values = sorted(strings & ARM_VALUES)
    if leaked_arm_values:
        problems.append(
            f"Selection-arm values leaked as strings: {leaked_arm_values}."
        )

    sample_id = str(record.get("sample_id", "") or "")
    if sample_id and sample_id in json.dumps(payload, sort_keys=True):
        problems.append(f"Sample id {sample_id!r} appears in the payload.")

    return problems
