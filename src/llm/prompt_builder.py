"""prompt_builder.py

Build classification prompts for the Wi-Fi CSI activity-recognition POC.

Design principle: the LLM must not be told that a particular amplitude or
phase pattern definitely represents a physical activity, because the
A/B/C labels are provisionally assigned and their human-readable activity
mapping is unverified.

The system prompt therefore:
  - Lists only dataset labels A, B, C (no physical activity names).
  - Explicitly states the label-to-activity mapping is unverified.
  - Forbids inventing physical interpretations of the signal.
  - Forbids assuming amplitude changes indicate human movement.
  - Allows UNKNOWN when evidence is insufficient.
  - Requires valid JSON output only.

Prompt versions are tracked so experiments can be compared reproducibly.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Prompt version
# ---------------------------------------------------------------------------
PROMPT_VERSION = "v1_zero_shot"

CLASS_LABELS = ["A", "B", "C"]

SYSTEM_PROMPT = """You are a Wi-Fi CSI activity classification system.

Your task is to classify a CSI-derived feature record into exactly one of these dataset labels:

- A
- B
- C

The input contains measurements extracted from a 50-sample, 242-subcarrier complex CSI window.

Important constraints:

- The activity names associated with A, B and C have not yet been verified.
- Do not invent physical interpretations of the signal.
- Use only patterns supported by the provided measurements.
- Do not assume that amplitude changes necessarily indicate human movement.
- If the evidence is insufficient, return UNKNOWN.

Return valid JSON only:

{
  "predicted_label": "A | B | C | UNKNOWN",
  "confidence": 0.0,
  "reason": "Brief explanation based on the supplied features"
}
"""


# ---------------------------------------------------------------------------
# Representation variants
# ---------------------------------------------------------------------------
REPRESENTATION_STRUCTURED = "structured_features"
REPRESENTATION_AMPLITUDE = "amplitude_time_series"

VALID_REPRESENTATIONS = (REPRESENTATION_STRUCTURED, REPRESENTATION_AMPLITUDE)


def format_feature_record(record: dict[str, Any]) -> str:
    """Format a single feature record as a compact JSON string for the prompt.

    Only scalar and small-list features are included to keep the prompt
    compact. Large per-subcarrier arrays (242 values) are summarized by
    mean/std/min/max rather than emitted raw, to avoid overwhelming the
    model and wasting tokens.

    Used by the ``structured_features`` representation (E1/E3).
    """
    amp = record.get("amplitude_features", {})
    phase = record.get("phase_features", {})
    temp = record.get("temporal_features", {})
    gain = record.get("gain_diagnostics", {})

    summary = {
        "label": record.get("label"),
        "shape": record.get("shape"),
        "source_file": record.get("source_file"),
        "split": record.get("split"),
        "representation": REPRESENTATION_STRUCTURED,
        "amplitude_overall": amp.get("overall"),
        "amplitude_per_time_mean": amp.get("per_time_mean"),
        "amplitude_per_time_std": amp.get("per_time_std"),
        "phase_circular_mean_all": phase.get("circular_mean_all"),
        "phase_circular_std_all": phase.get("circular_std_all"),
        "phase_temporal_variation": phase.get("temporal_phase_variation"),
        "phase_spectral_variation": phase.get("spectral_phase_variation"),
        "phase_wrap_count": phase.get("phase_wrap_count"),
        "temporal_frame_diff": temp.get("frame_to_frame_difference"),
        "temporal_total_energy": temp.get("total_energy"),
        "temporal_amplitude_trend": temp.get("amplitude_trend_correlation"),
        "gain_window_mean": gain.get("window_mean_amplitude"),
        "gain_window_range": gain.get("window_amplitude_range"),
        "gain_window_gain_ratio": gain.get("window_gain_ratio"),
        "gain_temporal_variation": gain.get("temporal_gain_variation"),
    }
    return json.dumps(summary)


def format_amplitude_time_series(record: dict[str, Any]) -> str:
    """Format the amplitude time series for the ``amplitude_time_series``
    representation (E2/E4).

    Uses ``amplitude_features.per_time_mean`` — the mean amplitude across
    all 242 subcarriers at each of the 50 time samples — giving a compact
    50-point time series that preserves temporal signal patterns.

    Optionally includes per-time-sample standard deviation across
    subcarriers to convey fluctuation.
    """
    amp = record.get("amplitude_features", {})
    per_time_mean = amp.get("per_time_mean", [])
    per_time_std = amp.get("per_time_std", [])

    # Build the time series as a list of {t, mean_amplitude, std_amplitude}.
    time_series = []
    for t in range(len(per_time_mean)):
        entry = {"t": t, "mean_amplitude": per_time_mean[t]}
        if t < len(per_time_std):
            entry["std_amplitude"] = per_time_std[t]
        time_series.append(entry)

    summary = {
        "label": record.get("label"),
        "shape": record.get("shape"),
        "source_file": record.get("source_file"),
        "split": record.get("split"),
        "representation": REPRESENTATION_AMPLITUDE,
        "amplitude_time_series": time_series,
        "amplitude_overall": amp.get("overall"),
    }
    return json.dumps(summary)


def format_record_for_representation(
    record: dict[str, Any],
    representation: str = REPRESENTATION_STRUCTURED,
) -> str:
    """Format a feature record according to the chosen representation."""
    if representation == REPRESENTATION_AMPLITUDE:
        return format_amplitude_time_series(record)
    return format_feature_record(record)


def build_zero_shot_prompt(
    record: dict[str, Any],
    *,
    class_labels: list[str] | None = None,
    representation: str = REPRESENTATION_STRUCTURED,
) -> list[dict[str, str]]:
    """Build a zero-shot prompt for a single feature record.

    Args:
        record: Feature record dict (from extract_features.py).
        class_labels: Dataset labels (default A, B, C).
        representation: Input representation — ``structured_features`` (E1)
            or ``amplitude_time_series`` (E2).

    Returns:
        A list of message dicts suitable for the chat-completions API.
    """
    if representation not in VALID_REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Valid: {VALID_REPRESENTATIONS}"
        )

    labels = class_labels or CLASS_LABELS
    labels_str = "\n".join(f"- {l}" for l in labels)

    system = SYSTEM_PROMPT.replace(
        "- A\n- B\n- C",
        labels_str,
    ) if labels != CLASS_LABELS else SYSTEM_PROMPT

    formatted = format_record_for_representation(record, representation)

    rep_label = (
        "amplitude time series (50 time points, mean amplitude per sample)"
        if representation == REPRESENTATION_AMPLITUDE
        else "structured statistical features"
    )

    user_content = (
        "Classify the following CSI feature record into one of the dataset labels above.\n\n"
        f"Input representation: {rep_label}\n"
        "Feature record:\n"
        + formatted
        + "\n\nReturn valid JSON only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_few_shot_prompt(
    record: dict[str, Any],
    examples: list[dict[str, Any]],
    *,
    class_labels: list[str] | None = None,
    representation: str = REPRESENTATION_STRUCTURED,
) -> list[dict[str, str]]:
    """Build a few-shot prompt with labelled examples.

    Args:
        record: The feature record to classify.
        examples: List of feature records, each with a "label" field and
            already-extracted features. These form the labelled examples.
        class_labels: Dataset labels (default A, B, C).
        representation: Input representation — ``structured_features`` (E3)
            or ``amplitude_time_series`` (E4).

    Each example is formatted the same way as the target record, followed
    by its known label, so the LLM can learn the label-feature mapping.
    """
    if representation not in VALID_REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Valid: {VALID_REPRESENTATIONS}"
        )

    labels = class_labels or CLASS_LABELS
    labels_str = "\n".join(f"- {l}" for l in labels)

    system = SYSTEM_PROMPT.replace(
        "- A\n- B\n- C",
        labels_str,
    ) if labels != CLASS_LABELS else SYSTEM_PROMPT

    rep_label = (
        "amplitude time series (50 time points, mean amplitude per sample)"
        if representation == REPRESENTATION_AMPLITUDE
        else "structured statistical features"
    )

    example_blocks = []
    for ex in examples:
        example_blocks.append(
            "Example feature record:\n"
            + format_record_for_representation(ex, representation)
            + f"\nCorrect label: {ex.get('label')}\n"
        )

    examples_text = "\n".join(example_blocks)

    user_content = (
        "Classify the following CSI feature record into one of the dataset labels above.\n\n"
        f"Input representation: {rep_label}\n"
        "Here are labelled examples for reference:\n\n"
        + examples_text
        + "\nTarget feature record to classify:\n"
        + format_record_for_representation(record, representation)
        + "\n\nReturn valid JSON only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    user_content = (
        "Classify the following CSI feature record into one of the dataset labels above.\n\n"
        "Here are labelled examples for reference:\n\n"
        + examples_text
        + "\nTarget feature record to classify:\n"
        + format_feature_record(record)
        + "\n\nReturn valid JSON only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def prompt_version() -> str:
    """Return the current prompt version string."""
    return PROMPT_VERSION
