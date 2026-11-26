"""Activity-only zero-shot prompt for exactly-one-person WiMANS v2 inference."""

from __future__ import annotations

import json
from typing import Any

from .prompt_builder import ACTIVITY_LABELS
from .prompt_builder_v2 import build_zero_shot_prompt_v2
from .prompt_validation import extract_inference_input


PROMPT_VERSION_ACTIVITY_V2 = "wimans_activity_only_zero_shot_v2"

TASK_INSTRUCTION_ACTIVITY_V2 = (
    "One person is present throughout this recording. "
    "Classify that person's activity using the ordered temporal CSI features."
)

SYSTEM_PROMPT_ACTIVITY_V2 = f"""
You analyze Wi-Fi Channel State Information (CSI) features for human
activity classification.

TASK:
Exactly one person is present in this recording. Identify the single
activity that best describes this person's activity over the recording.

Do not perform person-presence detection or estimate the number of people.

ALLOWED ACTIVITY LABELS:
{", ".join(ACTIVITY_LABELS)}

LABEL SEMANTICS:
- "nothing" means the person is present but inactive or stationary.
- "nothing" does NOT mean an empty room.
- Select exactly one activity label from the allowed labels.

INTERPRETING THE CSI INPUT:
- The input contains measurements extracted from a recording over time.
- Temporal windows describe changes in measured CSI signals. They are not
  separate people or separate activity labels.
- CSI links are measurement links, not person identities.
- The nine-value arrays follow the supplied link_order.
- Each temporal_sequence row follows temporal_sequence_schema.
- window_mean and window_std describe signal statistics within a window.
- temporal_delta describes change between measurements.
- A null temporal_delta means that the change could not be calculated
  from the available packets. It does NOT mean zero movement.
- Cross-link synchronization describes relationships between measured
  CSI links. It does not identify people.
- Signal statistics do not directly prove a particular activity.
- Do not invent numerical thresholds or claim that a feature uniquely
  identifies an activity.
- Use only the supplied CSI features and permitted metadata.
- Do not invent identities, locations, or trajectories.
- If the evidence is ambiguous, choose the best-supported single label
  and reflect uncertainty in confidence and reason.

OUTPUT:
Return exactly one JSON object with this schema:
{{
  "activity": "walk",
  "confidence": 0.0,
  "reason": "Brief explanation based only on the supplied input."
}}

RULES:
- activity must be exactly one allowed label.
- confidence must be a number from zero to one.
- reason must be a non-empty string.
- Do not include additional keys.
- Do not include Markdown fences or text outside the JSON object.
""".strip()


def build_activity_only_prompt_v2(
    record: dict[str, Any],
) -> list[dict[str, str]]:
    """Build an activity-only prompt using the unchanged v2 feature payload."""

    # Reuse v2 solely to obtain its validated, compact CSI representation.
    baseline_messages = build_zero_shot_prompt_v2(record)
    payload = extract_inference_input(baseline_messages)

    if set(payload) != {"environment", "wifi_band", "features"}:
        raise ValueError("Unexpected keys in the v2 inference payload.")

    compact_features = payload["features"]
    compact_features["prompt_version"] = PROMPT_VERSION_ACTIVITY_V2

    user_content = (
        TASK_INSTRUCTION_ACTIVITY_V2
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
        {"role": "system", "content": SYSTEM_PROMPT_ACTIVITY_V2},
        {"role": "user", "content": user_content},
    ]


def prompt_version_activity_v2() -> str:
    return PROMPT_VERSION_ACTIVITY_V2
