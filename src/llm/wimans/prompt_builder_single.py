"""Dedicated prompt construction for WiMANS single-person presence/activity.

This module is intentionally independent of the multi-person count prompts.
It reuses each baseline prompt builder only to obtain the exact feature block
that the corresponding feature version already sends, then replaces the task
instruction and the required response schema.

The payload sent to the model contains no activity labels, user counts,
selection arm, group identifiers or ground truth.
"""

from __future__ import annotations

import json
from typing import Any

from .prompt_builder import build_zero_shot_prompt
from .prompt_builder_v2 import build_zero_shot_prompt_v2
from .prompt_validation import extract_inference_input

PROMPT_VERSION = "wimans_presence_activity_v1"

SYSTEM_PROMPT_SINGLE = """
You are analyzing Wi-Fi Channel State Information (CSI) features to estimate
whether any person is present in a recording and, if so, what that person is
doing.

The input contains CSI-derived statistical or temporal features and permitted
contextual metadata. It does not contain ground-truth labels, the true number
of people, or any evaluation metadata.

TASK:
1. Decide whether at least one person is present.
2. If a person is present, choose the single activity that best describes what
   is happening during the recording.
3. If no person is present, report that and give no activity.

ALLOWED ACTIVITY LABELS:
nothing, walk, rotation, jump, wave, lie_down, pick_up, sit_down, stand_up

LABEL SEMANTICS:
- "nothing" means a person is present but inactive.
- "nothing" does not mean an empty room.
- When present is true, activity must be exactly one allowed label.
- When present is false, activity must be null.
- The environment field is contextual metadata. Do not infer presence solely
  from its value, including when it is "empty_room".

INTERPRETING THE CSI INPUT:
- Interpret the supplied features as measurements from a recording over time.
- Temporal windows describe changes in the measured signal. They are not
  separate people.
- CSI links are measurement links, not person identities.
- Signal statistics describe properties of the measured signal; they do not
  directly identify an activity or a person.
- Do not invent numerical thresholds or claim that a feature uniquely proves
  an activity.
- Use only the supplied CSI features and permitted metadata. Do not invent
  identities, locations, or individual trajectories.
- If the evidence is ambiguous, make a conservative estimate and reflect
  uncertainty in confidence and reason.

Return exactly one JSON object with this schema:
{
  "present": true,
  "activity": "walk",
  "confidence": 0.0,
  "reason": "Brief explanation based only on the supplied input."
}

Rules:
- "present" must be true or false.
- "activity" must be one allowed label when present is true, and null when
  present is false.
- "confidence" must be a number from zero to one.
- "reason" must be a non-empty string.
Return only the JSON object. Do not include Markdown fences or other text.
""".strip()

TASK_INSTRUCTION = (
    "Decide whether at least one person is present in this recording and, "
    "if so, identify the single activity that best describes it."
)

FEATURE_VERSION_BUILDERS = {
    "v1": build_zero_shot_prompt,
    "v2": build_zero_shot_prompt_v2,
}


def _feature_payload(
    record: dict[str, Any],
    *,
    feature_version: str,
) -> dict[str, Any]:
    """Return the baseline feature block for this version, unchanged."""
    builder = FEATURE_VERSION_BUILDERS.get(feature_version)
    if builder is None:
        raise ValueError("feature_version must be 'v1' or 'v2'.")

    baseline_messages = builder(record)
    payload = extract_inference_input(baseline_messages)

    if set(payload) != {"environment", "wifi_band", "features"}:
        raise ValueError(
            "Baseline prompt payload has unexpected top-level keys: "
            f"{sorted(payload)}"
        )
    return payload


def build_single_person_prompt(
    record: dict[str, Any],
    *,
    feature_version: str,
) -> list[dict[str, str]]:
    """Build a presence/activity prompt over the unchanged feature block."""
    payload = _feature_payload(record, feature_version=feature_version)

    user_content = (
        TASK_INSTRUCTION
        + "\n\nCSI-derived input:\n"
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n\nReturn the required JSON object only."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT_SINGLE},
        {"role": "user", "content": user_content},
    ]


def prompt_version_single(feature_version: str) -> str:
    if feature_version not in FEATURE_VERSION_BUILDERS:
        raise ValueError("feature_version must be 'v1' or 'v2'.")
    return f"{PROMPT_VERSION}_{feature_version}"
