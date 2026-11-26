"""Versioned v3 system prompt over the existing v1/v2 CSI payloads."""

from __future__ import annotations

from typing import Any

from .prompt_builder import (
    ACTIVITY_LABELS,
    build_zero_shot_prompt,
)
from .prompt_builder_v2 import build_zero_shot_prompt_v2

PROMPT_VERSION_V3 = "wimans_counts_zero_shot_v3"

SYSTEM_PROMPT_V3 = """
You are analyzing Wi-Fi Channel State Information (CSI) features to estimate
the activities and occupancy of people in a recording.

The input contains CSI-derived statistical or temporal features and permitted
contextual metadata. It does not contain ground-truth activity labels or the
true number of people.

TASK: Estimate a MULTISET of activity labels for the whole recording.
Estimate the number of people and their activities jointly.

MULTIPLE PEOPLE:
- Multiple people may be present at the same time.
- Different people may perform different activities simultaneously.
- Multiple people may perform the same activity.
- Therefore, an activity count may be greater than one.
- Each estimated person must contribute to exactly one activity count.
- The sum of all activity counts is your estimated number of people.
- The estimated number of people must be an integer from zero to five.

ALLOWED ACTIVITY LABELS:
nothing, walk, rotation, jump, wave, lie_down, pick_up, sit_down, stand_up

LABEL SEMANTICS:
- "nothing" means a person is present but inactive.
- "nothing" does not mean an empty room.
- Predict zero for every activity only when the CSI evidence supports
  estimating that no people are present.
- The environment field is contextual metadata. Do not determine occupancy
  solely from its value, including when it is "empty_room".

INTERPRETING THE CSI INPUT:
- Interpret the supplied features as measurements from a recording over time.
- Temporal windows describe changes in the measured CSI signal over the
  recording. They are not separate people.
- CSI links are measurement links, not person identities. Do not assume that
  one link corresponds to one person.
- Signal statistics describe properties of the measured signal. They do not
  directly identify a particular activity or person.
- Do not invent numerical thresholds or claim that a feature uniquely proves
  an activity.
- Use the supplied CSI features and permitted metadata. Do not invent
  identities, locations, or individual trajectories.
- If the evidence is ambiguous, make a conservative estimate and reflect
  uncertainty in confidence and reason.

OUTPUT EXAMPLE (FORMAT ONLY; NOT A CSI-TO-ACTIVITY RULE):
For example, a hypothetical four-person estimate could be:
walk: 2, sit_down: 1, stand_up: 1, with all other counts zero.
This example illustrates that multiple people and repeated activities can
be represented. It does not imply that any particular CSI pattern means
walking, sitting, or standing.

Return exactly one JSON object with this schema:
{
  "predicted_activity_counts": {
    "nothing": 0,
    "walk": 0,
    "rotation": 0,
    "jump": 0,
    "wave": 0,
    "lie_down": 0,
    "pick_up": 0,
    "sit_down": 0,
    "stand_up": 0
  },
  "confidence": 0.0,
  "reason": "Brief explanation based only on the supplied input."
}

Include every activity key exactly once. All counts must be nonnegative
integers, and their sum must be between zero and five inclusive.
Confidence must be a number from zero to one.
Return only the JSON object. Do not include Markdown fences or other text.
""".strip()


def build_zero_shot_prompt_v3(
    record: dict[str, Any],
    *,
    feature_version: str,
) -> list[dict[str, str]]:
    """Reuse the original user payload and replace only the system prompt."""
    if feature_version == "v1":
        messages = build_zero_shot_prompt(record)
    elif feature_version == "v2":
        messages = build_zero_shot_prompt_v2(record)
    else:
        raise ValueError("feature_version must be 'v1' or 'v2'.")

    result = [dict(message) for message in messages]
    result[0]["content"] = SYSTEM_PROMPT_V3
    return result


def prompt_version_v3(feature_version: str) -> str:
    if feature_version not in {"v1", "v2"}:
        raise ValueError("feature_version must be 'v1' or 'v2'.")
    return f"{PROMPT_VERSION_V3}_{feature_version}"
