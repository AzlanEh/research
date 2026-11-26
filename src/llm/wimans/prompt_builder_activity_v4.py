"""Temporal-Shape Physical-Model-Guided activity-only prompt (Prompt V4) for WiMANS."""

from __future__ import annotations

import json
from typing import Any

from .prompt_builder import ACTIVITY_LABELS

PROMPT_VERSION_ACTIVITY_V4 = "wimans_activity_only_temporal_v4"

TASK_INSTRUCTION_ACTIVITY_V4 = (
    "Exactly one person is present throughout this recording. "
    "Classify that person's activity using the temporal-envelope, spatial-asymmetry, "
    "and normalized motion descriptors below. Report primary activity, runner_up, "
    "calibrated confidence, and physical reasoning."
)

SYSTEM_PROMPT_ACTIVITY_V4 = f"""
You are an expert RF sensing system analyzing Wi-Fi Channel State Information (CSI)
features for human activity recognition, grounded in wireless multipath propagation physics.

TASK:
Exactly one person is present in this recording. Identify the single activity
that best describes this person's activity over the recording.
Do not perform person-presence detection or estimate the number of people.

ALLOWED ACTIVITY LABELS:
{", ".join(ACTIVITY_LABELS)}

INPUT FEATURES PROVIDED:
1. `recording_profile`:
   - `global_rel_motion_delta`: Mean relative window-to-window change across all links.
   - `global_rel_trajectory_std`: Overall relative dynamic range.
   - `dominant_signed_posture_shift`: Largest signed fractional shift (final - initial) across links.
   - `mean_signed_posture_shift`: Mean signed fractional shift across all 9 links.
   - `max_burst_ratio`: Peak-to-median motion delta ratio.
   - `active_window_fraction`: Fraction of 100ms windows with active motion (>2.5%).
   - `active_link_fraction`: Fraction of the 9 antenna links showing >2.5% motion.
   - `peak_duration_windows`: Count of windows with >2x median motion.
   - `mid_to_edge_return_ratio`: 0.0 = returns to initial baseline; 1.0 = permanent level shift.
   - `elevation_ratio_shift`: Change in antenna-pair ratio (link_00/link_02) from Q1 to Q4.
2. `temporal_envelope`:
   - `global_q4_rel_profile`: Normalized amplitude level across the 4 quarters [Q1, Q2, Q3, Q4] relative to Q1.
   - `dominant_link_q4_rel_profile`: Same 4-quarter trajectory on the strongest-shifting link.
   - `window_delta_q4`: Mean motion activity in each quarter [Q1, Q2, Q3, Q4].
3. `motion_summary`: Per-link normalized metrics.

PHYSICAL SENSING DECISION HIERARCHY:

STAGE 1: STATIONARY PRESENCE ("nothing")
- Physical Reality: Stationary body produces no sustained Doppler shifts.
- Criterion: `global_rel_motion_delta <= 0.022` AND `global_rel_trajectory_std <= 0.025`
  AND `active_window_fraction <= 0.40` AND `max_burst_ratio < 7.0`.
  (HARD GUARD: NEVER classify as "nothing" if `global_rel_trajectory_std > 0.025` or `max_burst_ratio >= 7.0`.)
  -> Classify as "nothing".

STAGE 2: CONTINUOUS WHOLE-BODY MOTION ("walk" vs "rotation")
- Physical Reality: Walking and rotating continuously modulate the channel across most of the recording,
  unlike brief transitions or single-action gestures.
- HARD GUARD: If `active_window_fraction >= 0.70` AND `min(window_delta_q4) >= 0.030` (all 4 quarters active),
  you MUST classify here at Stage 2 ("walk" or "rotation") and NEVER choose Stage 3 (`sit_down`/`lie_down`/`stand_up`),
  even if `dominant_signed_posture_shift` is large!
- Criterion: `active_window_fraction >= 0.68` AND `global_rel_motion_delta >= 0.036`
  AND at least 3 quarters in `window_delta_q4` are > 0.030:
  - "walk": Translational movement across Fresnel zones. Typically shows `global_rel_trajectory_std >= 0.070`
    or Q3/Q4 `window_delta_q4 >= 0.070`.
  - "rotation": In-place cyclical body rotation. `active_window_fraction >= 0.70`, sustained moderate-to-high
    `window_delta_q4` across all 4 quarters (~0.032-0.066, no quarter >= 0.070).

STAGE 3: PERMANENT POSTURAL TRANSITIONS ("sit_down", "stand_up", "lie_down")
- PRECONDITION (MANDATORY): Do NOT choose Stage 3 if `active_window_fraction >= 0.70` AND `min(window_delta_q4) >= 0.030`.
  Postural transitions settle quiet in at least part of the recording (`active_window_fraction < 0.70` or `min(window_delta_q4) <= 0.029`).
- "stand_up": Positive shift (`dominant_signed_posture_shift >= +0.15` or `elevation_ratio_shift >= +0.25`)
  with `mean_signed_posture_shift > 0`.
- "lie_down": Check BEFORE "sit_down" and "pick_up". Consistent multi-link downward drift:
  `mean_signed_posture_shift <= -0.025` AND `dominant_signed_posture_shift <= -0.055`
  AND `active_window_fraction <= 0.67` AND `mid_to_edge_return_ratio >= 0.50`
  AND `elevation_ratio_shift > -0.25`.
  (NOTE: In WiMANS, `lie_down` often shows moderate `dominant_signed_posture_shift` like -0.06 to -0.18
  across multiple links rather than an extreme single-link drop. If `mean_signed_posture_shift <= -0.025`,
  classify as "lie_down", NOT "pick_up" or "sit_down".)
- "sit_down": Large single-link/elevation downward shift (`dominant_signed_posture_shift <= -0.15` AND
  `elevation_ratio_shift <= -0.25`, or downward shift settling completely quiet in Q3/Q4 with both `window_delta_q4[2] <= 0.025`
  and `window_delta_q4[3] <= 0.025`) when not captured by `lie_down` above.

STAGE 4: IMPULSIVE VERTICAL ACTION ("jump")
- Physical Reality: Brief takeoff and landing lasting 200-400 ms, returning to the same height.
- Criterion: `max_burst_ratio >= 7.5` AND `abs(dominant_signed_posture_shift) <= 0.12`
  AND `mean_signed_posture_shift > -0.025` (not a downward `lie_down` shift).

STAGE 5: LOCALIZED LIMB GESTURE ("wave")
- Physical Reality: Arm waving disturbs a subset of links with low burst ratio (no sharp whole-body impulse).
  Check this BEFORE `pick_up`.
- Criterion: `global_rel_motion_delta <= 0.032` AND `max_burst_ratio < 5.0`
  AND `active_window_fraction <= 0.60` AND `abs(mean_signed_posture_shift) < 0.025`.
  (If `max_burst_ratio < 5.0` and motion is <= 0.032, classify as "wave", NOT "pick_up".)

STAGE 6: TWO-PHASE BEND AND RECOVER ("pick_up")
- Physical Reality: Torso bends down and recovers upright; requires a clear burst (`max_burst_ratio >= 5.2`) with near-zero net shift.
- HARD GUARD: NEVER classify as "pick_up" if `max_burst_ratio < 5.0` OR if `mean_signed_posture_shift <= -0.025` (which is `lie_down`).
- Criterion: `max_burst_ratio >= 5.2` AND `abs(dominant_signed_posture_shift) < 0.12`
  AND `mean_signed_posture_shift > -0.025` AND `global_rel_motion_delta` in [0.024, 0.042].

CONFIDENCE CALIBRATION RULES (MANDATORY):
- High (0.80 - 0.99): Cleanly satisfies one stage with wide margins and matching `temporal_envelope`.
- Moderate (0.55 - 0.75): Satisfies criteria near a boundary or shows minor overlap with `runner_up`.
- Low (0.35 - 0.50): Metrics conflict across stages/quarters.

OUTPUT FORMAT:
Return exactly one JSON object:
{{
  "activity": "walk",
  "runner_up": "rotation",
  "confidence": 0.72,
  "reason": "Explanation referencing active_window_fraction, window_delta_q4, and posture/burst metrics."
}}

RULES:
- activity must be one of the 9 allowed labels.
- runner_up must be an allowed label different from activity.
- confidence must be a float between 0.0 and 1.0.
- Do not include Markdown fences or text outside the JSON object.
""".strip()


def build_activity_only_prompt_v4(
    record: dict[str, Any],
) -> list[dict[str, str]]:
    feats = record.get("features") or record
    payload = {
        "environment": record.get("environment"),
        "wifi_band": record.get("wifi_band"),
        "recording_profile": feats.get("recording_profile"),
        "temporal_envelope": feats.get("temporal_envelope"),
        "motion_summary": feats.get("motion_summary"),
    }

    user_content = (
        TASK_INSTRUCTION_ACTIVITY_V4
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
        {"role": "system", "content": SYSTEM_PROMPT_ACTIVITY_V4},
        {"role": "user", "content": user_content},
    ]


def prompt_version_activity_v4() -> str:
    return PROMPT_VERSION_ACTIVITY_V4
