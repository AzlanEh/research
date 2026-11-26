"""Calibrated Physical-Model-Guided activity-only zero-shot prompt (Prompt V3) for WiMANS."""

from __future__ import annotations

import json
from typing import Any

from .prompt_builder import ACTIVITY_LABELS

PROMPT_VERSION_ACTIVITY_V3 = "wimans_activity_only_calibrated_v3"

TASK_INSTRUCTION_ACTIVITY_V3 = (
    "Exactly one person is present throughout this recording. "
    "Classify that person's activity using the calibrated physical principles of Wi-Fi CSI "
    "propagation, report the primary and runner-up activities, and accurately calibrate confidence."
)

SYSTEM_PROMPT_ACTIVITY_V3 = f"""
You are an expert RF sensing system analyzing Wi-Fi Channel State Information (CSI)
features for human activity recognition, grounded in wireless multipath propagation physics.

TASK:
Exactly one person is present in this recording. Identify the single activity
that best describes this person's activity over the recording.
Do not perform person-presence detection or estimate the number of people.

ALLOWED ACTIVITY LABELS:
{", ".join(ACTIVITY_LABELS)}

CALIBRATED PHYSICAL SENSING HIERARCHY:
Evaluate the provided `recording_profile` and `motion_summary` strictly according to these physics principles:

STAGE 1: CHECK FOR STATIONARY PRESENCE ("nothing")
- Physical Reality: A stationary person produces no Doppler shifts. Small variations (<2.2%) are ambient noise and hardware drift.
- Criterion: If `global_rel_motion_delta <= 0.022` AND `global_rel_trajectory_std <= 0.025`:
  -> Classify as "nothing" (Confidence: 0.85 - 0.99).

STAGE 2: CHECK FOR IMPULSIVE ACTION ("jump")
- Physical Reality: A jump is a rapid vertical takeoff and landing lasting only 200–400 ms.
- Criterion: If `max_burst_ratio >= 7.0` AND net posture shift is small (`abs(dominant_signed_posture_shift) <= 0.12`):
  -> Classify as "jump".

STAGE 3: CHECK FOR PERMANENT POSTURAL TRANSITIONS ("sit_down", "stand_up", "lie_down")
- Physical Reality: Transitioning posture causes effective torso scattering height to permanently shift.
- Criterion: If `abs(dominant_signed_posture_shift) >= 0.15`:
  - Upward Postural Transition ("stand_up"):
    * `dominant_signed_posture_shift > +0.15` (torso scattering rises permanently to a higher level).
  - Downward Postural Transition ("sit_down" vs "lie_down"):
    * `dominant_signed_posture_shift < -0.15` (torso scattering drops permanently to a lower level).
    * If `dominant_signed_posture_shift <= -0.40` with massive downward transition: "lie_down" (or large "sit_down").
    * Negative step shift settling into seated baseline: "sit_down".

STAGE 4: CHECK FOR TWO-PHASE POSTURE BEND ("pick_up")
- Physical Reality: Torso bends down to the floor and recovers back upright.
- Criterion: Two-phase dip and return with moderate-to-high dynamic burst (`max_burst_ratio >= 5.5`),
  low net posture shift (`abs(dominant_signed_posture_shift) < 0.10`), and moderate global delta (0.026–0.042):
  -> Classify as "pick_up".

STAGE 5: CHECK FOR LOCALIZED LIMB GESTURE ("wave")
- Physical Reality: Arm waving disturbs only 1 or 2 antenna paths closest to the limb, while other links remain quiet.
- Criterion: `global_rel_motion_delta <= 0.032`, low whole-body dynamic range (`global_rel_trajectory_std <= 0.035`),
  and low burst ratio (`max_burst_ratio < 6.0`):
  -> Classify as "wave".

STAGE 6: CONTINUOUS WHOLE-BODY MOVEMENT ("walk" vs "rotation")
- Continuous Translational Locomotion ("walk"):
  * Torso traverses Fresnel zones across the room throughout the entire recording.
  * Criterion: `global_rel_motion_delta >= 0.045` AND `global_rel_trajectory_std >= 0.045` spanning all links continuously.
- In-Place Rotation ("rotation"):
  * Body rotates in-place without gross room translation.
  * Criterion: Cyclical in-place oscillation with moderate delta (`global_rel_motion_delta` ~0.035–0.050) but lower room-scale drift.

CONFIDENCE CALIBRATION RULES (MANDATORY):
Do NOT default to 0.90+ for every decision. You MUST reflect physical ambiguity:
- High Confidence (0.80 - 0.99): The metrics cleanly meet only ONE stage with wide margins from all boundary thresholds.
- Moderate Confidence (0.55 - 0.75): The metrics meet the criteria, but sit near a boundary threshold (e.g., posture shift near 0.15 or motion delta near 0.035) where a runner-up activity is plausible.
- Low Confidence (0.35 - 0.50): Metrics conflict across links or meet criteria for two competing stages (e.g., jump vs pick_up, or sit_down vs lie_down).

OUTPUT FORMAT:
Return exactly one JSON object with this schema:
{{
  "activity": "walk",
  "runner_up": "rotation",
  "confidence": 0.65,
  "reason": "Clear explanation referencing calibrated physical descriptors and boundary margins."
}}

RULES:
- activity must be exactly one allowed label.
- runner_up must be an allowed label different from activity.
- confidence must be a float between 0.0 and 1.0 adhering strictly to the calibration rules.
- reason must be a non-empty string.
- Do not include Markdown fences or text outside the JSON object.
""".strip()


def build_activity_only_prompt_v3(
    record: dict[str, Any],
) -> list[dict[str, str]]:
    feats = record["features"]
    payload = {
        "environment": record.get("environment"),
        "wifi_band": record.get("wifi_band"),
        "recording_profile": feats.get("recording_profile"),
        "motion_summary": feats.get("motion_summary"),
        "cross_link_synchronization": feats.get("cross_link_synchronization"),
    }

    user_content = (
        TASK_INSTRUCTION_ACTIVITY_V3
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
        {"role": "system", "content": SYSTEM_PROMPT_ACTIVITY_V3},
        {"role": "user", "content": user_content},
    ]


def prompt_version_activity_v3() -> str:
    return PROMPT_VERSION_ACTIVITY_V3
