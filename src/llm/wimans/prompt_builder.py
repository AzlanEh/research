"""Prompt construction for WiMANS multi-subject activity-count inference."""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "wimans_counts_zero_shot_v1"

ACTIVITY_LABELS = (
    "nothing",
    "walk",
    "rotation",
    "jump",
    "wave",
    "lie_down",
    "pick_up",
    "sit_down",
    "stand_up",
)

SYSTEM_PROMPT = f"""
You are analyzing Wi-Fi Channel State Information (CSI) features to infer
which activities are present in a recording.

The input contains statistical features extracted from CSI measurements.
It does not contain ground-truth labels or the true number of people.

Predict a MULTISET of activity labels: return the estimated count for each
activity, not a single label and not person identities.

Allowed activity labels:
{", ".join(ACTIVITY_LABELS)}

Important label semantics:
- "nothing" means a person is present but inactive.
- "nothing" does NOT mean an empty room.
- An empty room should have zero for every activity.
- A person should contribute to one activity count.
- Do not invent identities, locations, or individual trajectories.
- Do not assume that each CSI link corresponds to a person.
- Use only the supplied CSI features and metadata.
- Do not claim certainty when the features do not support it.

The dataset contains recordings with between zero and five people.
The actual number of people in this recording is unknown to you.
All activity counts must be nonnegative integers and their sum must not
exceed five.

Return exactly one JSON object with this schema:
{{
  "predicted_activity_counts": {{
    "nothing": 0,
    "walk": 0,
    "rotation": 0,
    "jump": 0,
    "wave": 0,
    "lie_down": 0,
    "pick_up": 0,
    "sit_down": 0,
    "stand_up": 0
  }},
  "confidence": 0.0,
  "reason": "Brief explanation based only on the supplied input."
}}

Include every activity key exactly once. Do not include Markdown fences or
text outside the JSON object. Confidence must be a number from 0 to 1.
""".strip()


def build_zero_shot_prompt(record: dict[str, Any]) -> list[dict[str, str]]:
    """Build a zero-shot prompt without including ground-truth fields."""
    features = record.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError("WiMANS feature record must contain a non-empty 'features' object.")

    # Explicit allowlist: identifiers, labels, user count and user annotations
    # are deliberately excluded from the LLM input.
    inference_input = {
        "environment": record.get("environment"),
        "wifi_band": record.get("wifi_band"),
        "features": features,
    }

    user_content = (
        "Infer the activity-count vector for this recording.\n\n"
        "CSI-derived input:\n"
        + json.dumps(inference_input, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n\nReturn the required JSON object only."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def prompt_version() -> str:
    return PROMPT_VERSION
