"""Dry-run validation for WiMANS prompts.

These helpers inspect the messages produced by the v1 and v2 prompt builders.
They make no network calls. The pilot dry run and its unit tests share them so
that leakage and structure rules are defined once.

The validation is intentionally strict: it parses the structured input that
the prompt sends to the model and checks that only allowlisted keys appear.
Anything resembling identifiers, split membership, user counts or activity
annotations is reported as an error.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from .prompt_builder import ACTIVITY_LABELS, SYSTEM_PROMPT
from .prompt_builder_v2 import LINK_NAMES, SEQUENCE_SCHEMA

V1_FEATURE_VERSION = "wimans_recording_v1"
V2_FEATURE_VERSION = "wimans_recording_v2"

INPUT_PREFIX = "CSI-derived input:\n"
INPUT_SUFFIX = "\n\nReturn the required JSON object only."

ALLOWED_TOP_LEVEL_KEYS = frozenset({"environment", "wifi_band", "features"})

V1_FEATURE_KEYS = frozenset(
    {
        "feature_version",
        "links",
        "num_windows",
        "recording_summary",
        "window_size_packets",
    }
)

V1_LINK_KEYS = frozenset(
    {
        "signal_mean",
        "signal_p10",
        "signal_p50",
        "signal_p90",
        "signal_std",
        "temporal_delta_mean",
        "temporal_delta_std",
        "window_mean_variability",
    }
)

V1_SUMMARY_KEYS = frozenset(
    {"signal_mean", "signal_std", "window_mean", "window_std"}
)

V1_STAT_KEYS = frozenset({"mean", "std", "p10", "p50", "p90"})

V2_FEATURE_KEYS = frozenset(
    {
        "feature_version",
        "prompt_version",
        "window_size_packets",
        "num_packets",
        "num_windows",
        "complete_windows",
        "link_order",
        "temporal_sequence_schema",
        "temporal_sequence",
        "cross_link_synchronization",
    }
)

V2_SYNCHRONIZATION_KEYS = frozenset({"method", "matrix"})

# Field names that must never appear in a prompt. Matched on exact key names,
# because numeric feature values can coincide with a group id or user count.
FORBIDDEN_KEYS = frozenset(
    {
        "sample_id",
        "group_id",
        "number_of_users",
        "num_users",
        "users",
        "user",
        "user_id",
        "activity",
        "activities",
        "label",
        "labels",
        "split",
        "train",
        "test",
        "ground_truth",
        "ground_truth_activity_counts",
        "predicted_activity_counts",
    }
)


def extract_inference_input(messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Parse the JSON payload embedded in the user message."""
    if len(messages) < 2:
        raise ValueError("Prompt must contain a system and a user message.")

    system_message, user_message = messages[0], messages[-1]

    if system_message.get("role") != "system":
        raise ValueError("First message must have role 'system'.")
    if user_message.get("role") != "user":
        raise ValueError("Last message must have role 'user'.")

    content = user_message.get("content", "")
    if INPUT_PREFIX not in content:
        raise ValueError("User message is missing the CSI input prefix.")

    payload = content.split(INPUT_PREFIX, 1)[1]
    if not payload.endswith(INPUT_SUFFIX):
        raise ValueError("User message is missing the expected suffix.")

    payload = payload[: -len(INPUT_SUFFIX)]
    parsed = json.loads(payload)

    if not isinstance(parsed, dict):
        raise ValueError("CSI input payload must be a JSON object.")
    return parsed


def collect_keys(value: Any) -> set[str]:
    """Return every mapping key found in a nested JSON-like structure."""
    keys: set[str] = set()

    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys |= collect_keys(nested)
    elif isinstance(value, list):
        for item in value:
            keys |= collect_keys(item)

    return keys


def collect_strings(value: Any) -> Iterable[str]:
    """Yield every string value found in a nested JSON-like structure."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from collect_strings(nested)
    elif isinstance(value, list):
        for item in value:
            yield from collect_strings(item)


def audit_inference_input(
    payload: Mapping[str, Any],
    *,
    allowed_feature_keys: frozenset[str],
    sample_id: str | None = None,
) -> list[str]:
    """Check the parsed payload for missing, unexpected or leaked fields."""
    problems: list[str] = []

    top_level = set(payload)
    if top_level != ALLOWED_TOP_LEVEL_KEYS:
        problems.append(
            "Top-level keys must be "
            f"{sorted(ALLOWED_TOP_LEVEL_KEYS)}, got {sorted(top_level)}."
        )

    if not isinstance(payload.get("features"), Mapping):
        problems.append("'features' must be an object.")
        return problems

    allowed_keys = allowed_feature_keys | ALLOWED_TOP_LEVEL_KEYS
    unexpected = collect_keys(payload) - allowed_keys
    if unexpected:
        problems.append(f"Unexpected keys in prompt payload: {sorted(unexpected)}.")

    leaked_keys = collect_keys(payload) & FORBIDDEN_KEYS
    if leaked_keys:
        problems.append(f"Leaked forbidden keys: {sorted(leaked_keys)}.")

    leaked_values = sorted(set(collect_strings(payload)) & set(ACTIVITY_LABELS))
    if leaked_values:
        problems.append(f"Activity labels leaked as values: {leaked_values}.")

    if sample_id is not None:
        serialized = json.dumps(payload, sort_keys=True)
        if sample_id in serialized:
            problems.append(f"Sample id {sample_id!r} appears in the payload.")

    return problems


def audit_system_message(messages: Sequence[Mapping[str, str]]) -> list[str]:
    """The system prompt must be the frozen, shared instruction block."""
    if messages[0].get("content") != SYSTEM_PROMPT:
        return ["System message does not match the frozen SYSTEM_PROMPT."]
    return []


def audit_v1_prompt(
    messages: Sequence[Mapping[str, str]],
    record: Mapping[str, Any],
) -> list[str]:
    """Validate a v1 prompt against its source feature record."""
    problems = audit_system_message(messages)

    try:
        payload = extract_inference_input(messages)
    except (ValueError, json.JSONDecodeError) as exc:
        return problems + [f"Cannot parse v1 prompt payload: {exc}"]

    problems += audit_inference_input(
        payload,
        allowed_feature_keys=(
            V1_FEATURE_KEYS
            | V1_LINK_KEYS
            | V1_SUMMARY_KEYS
            | V1_STAT_KEYS
            | frozenset(LINK_NAMES)
        ),
        sample_id=str(record.get("sample_id", "")) or None,
    )

    features = payload.get("features")
    if not isinstance(features, Mapping):
        return problems

    if features.get("feature_version") != V1_FEATURE_VERSION:
        problems.append(
            "v1 prompt must carry "
            f"{V1_FEATURE_VERSION!r}, got {features.get('feature_version')!r}."
        )

    links = features.get("links")
    if not isinstance(links, Mapping) or set(links) != set(LINK_NAMES):
        problems.append("v1 'links' must contain exactly the nine link names.")
    elif any(set(link) != V1_LINK_KEYS for link in links.values()):
        problems.append("v1 link entries have unexpected keys.")

    summary = features.get("recording_summary")
    if not isinstance(summary, Mapping) or set(summary) != V1_SUMMARY_KEYS:
        problems.append("v1 'recording_summary' keys are incorrect.")
    elif any(
        not isinstance(summary.get(name), Mapping)
        or set(summary[name]) != V1_STAT_KEYS
        for name in ("window_mean", "window_std")
    ):
        problems.append("v1 'recording_summary' statistics have incorrect keys.")

    if features != record.get("features"):
        problems.append("v1 prompt features differ from the source record.")

    if payload.get("environment") != record.get("environment"):
        problems.append("v1 prompt environment differs from the source record.")
    if payload.get("wifi_band") != record.get("wifi_band"):
        problems.append("v1 prompt wifi_band differs from the source record.")

    return problems


def audit_v2_prompt(
    messages: Sequence[Mapping[str, str]],
    record: Mapping[str, Any],
) -> list[str]:
    """Validate that a v2 prompt uses the compact temporal representation."""
    problems = audit_system_message(messages)

    try:
        payload = extract_inference_input(messages)
    except (ValueError, json.JSONDecodeError) as exc:
        return problems + [f"Cannot parse v2 prompt payload: {exc}"]

    problems += audit_inference_input(
        payload,
        allowed_feature_keys=V2_FEATURE_KEYS | V2_SYNCHRONIZATION_KEYS,
        sample_id=str(record.get("sample_id", "")) or None,
    )

    features = payload.get("features")
    if not isinstance(features, Mapping):
        return problems

    if features.get("feature_version") != V2_FEATURE_VERSION:
        problems.append(
            "v2 prompt must carry "
            f"{V2_FEATURE_VERSION!r}, got {features.get('feature_version')!r}."
        )

    if features.get("link_order") != list(LINK_NAMES):
        problems.append("v2 'link_order' must list the nine links in order.")

    if features.get("temporal_sequence_schema") != list(SEQUENCE_SCHEMA):
        problems.append("v2 'temporal_sequence_schema' does not match the schema.")

    if "movement_intensity" in features:
        problems.append("Compact v2 prompt must not duplicate movement_intensity.")

    sequence = features.get("temporal_sequence")
    if not isinstance(sequence, list) or not sequence:
        problems.append("v2 'temporal_sequence' must be a non-empty list.")
    else:
        width = len(SEQUENCE_SCHEMA)
        for window_index, row in enumerate(sequence):
            if not isinstance(row, list) or len(row) != width:
                problems.append(
                    f"v2 window {window_index} must have {width} fields."
                )
                break
            arrays = row[5:]
            if any(not isinstance(arr, list) or len(arr) != len(LINK_NAMES)
                   for arr in arrays):
                problems.append(
                    f"v2 window {window_index} must contain three "
                    f"{len(LINK_NAMES)}-element arrays."
                )
                break

    synchronization = features.get("cross_link_synchronization")
    if not isinstance(synchronization, Mapping):
        problems.append("v2 'cross_link_synchronization' must be an object.")
    else:
        if set(synchronization) != V2_SYNCHRONIZATION_KEYS:
            problems.append(
                "Compact v2 synchronization must contain only "
                f"{sorted(V2_SYNCHRONIZATION_KEYS)}."
            )
        matrix = synchronization.get("matrix")
        if (
            not isinstance(matrix, list)
            or len(matrix) != len(LINK_NAMES)
            or any(
                not isinstance(row, list) or len(row) != len(LINK_NAMES)
                for row in matrix
            )
        ):
            problems.append("v2 synchronization matrix must be 9x9.")

    return problems
