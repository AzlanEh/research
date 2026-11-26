"""Compact zero-shot prompt construction for WiMANS temporal features v2."""

from __future__ import annotations

import json
from typing import Any

from .prompt_builder import SYSTEM_PROMPT

PROMPT_VERSION = "wimans_counts_zero_shot_v2_compact"

LINK_NAMES = [
    f"link_{tx}{rx}"
    for tx in range(3)
    for rx in range(3)
]

SEQUENCE_SCHEMA = [
    "window_index",
    "start_packet",
    "end_packet",
    "num_packets",
    "is_partial",
    "window_mean",
    "window_std",
    "temporal_delta",
]


def build_zero_shot_prompt_v2(
    record: dict[str, Any],
) -> list[dict[str, str]]:
    """Build compact v2 prompt without identifiers or ground-truth labels."""
    features = record.get("features")

    if not isinstance(features, dict):
        raise ValueError("Record must contain a features object.")

    if features.get("feature_version") != "wimans_recording_v2":
        raise ValueError("Expected wimans_recording_v2 features.")

    sequence = features.get("temporal_sequence")
    synchronization = features.get("cross_link_synchronization")

    if not isinstance(sequence, list) or not sequence:
        raise ValueError("Missing temporal_sequence.")

    if not isinstance(synchronization, dict):
        raise ValueError("Missing cross_link_synchronization.")

    compact_sequence = []

    for window in sequence:
        links = window.get("links")

        if not isinstance(links, dict):
            raise ValueError("Each temporal window must contain links.")

        try:
            means = [links[name]["window_mean"] for name in LINK_NAMES]
            stds = [links[name]["window_std"] for name in LINK_NAMES]
            deltas = [links[name]["temporal_delta"] for name in LINK_NAMES]
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid per-link temporal features.") from exc

        compact_sequence.append([
            window["window_index"],
            window["start_packet"],
            window["end_packet"],
            window["num_packets"],
            window["is_partial"],
            means,
            stds,
            deltas,
        ])

    matrix = synchronization.get("matrix")

    if (
        not isinstance(matrix, list)
        or len(matrix) != len(LINK_NAMES)
        or any(not isinstance(row, list) or len(row) != len(LINK_NAMES)
               for row in matrix)
    ):
        raise ValueError("Expected a 9x9 synchronization matrix.")

    compact_features = {
        "feature_version": features["feature_version"],
        "prompt_version": PROMPT_VERSION,
        "window_size_packets": features["window_size_packets"],
        "num_packets": features["num_packets"],
        "num_windows": features["num_windows"],
        "complete_windows": features["complete_windows"],
        "link_order": LINK_NAMES,
        "temporal_sequence_schema": SEQUENCE_SCHEMA,
        "temporal_sequence": compact_sequence,
        "cross_link_synchronization": {
            "method": synchronization["method"],
            "matrix": matrix,
        },
    }

    inference_input = {
        "environment": record.get("environment"),
        "wifi_band": record.get("wifi_band"),
        "features": compact_features,
    }

    user_content = (
        "Infer the activity-count vector from the ordered temporal CSI "
        "features. Each temporal_sequence row follows "
        "temporal_sequence_schema. All nine-value arrays and the "
        "correlation matrix follow link_order.\n\n"
        "CSI-derived input:\n"
        + json.dumps(
            inference_input,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n\nReturn the required JSON object only."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def prompt_version_v2() -> str:
    return PROMPT_VERSION
