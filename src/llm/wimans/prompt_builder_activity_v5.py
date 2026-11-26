"""Lean V4+V5 Hybrid Physical-Model-Guided activity-only prompt for WiMANS.

Keeps V4's clean 6-stage temporal-shape hierarchy (which scored 51.11% on 45 samples)
and adds ONLY the 3 highest-signal V5 subcarrier/Doppler discriminants:
  - subcarrier_coherence_ratio
  - mid_freq_ratio_2_6hz
  - high_to_low_doppler_ratio
This avoids overloading smaller models with 13+ extra spectral fields.
"""

from __future__ import annotations

import json
from typing import Any

from .prompt_builder import ACTIVITY_LABELS

PROMPT_VERSION_ACTIVITY_V5 = "wimans_activity_only_hybrid_v4_v5"

TASK_INSTRUCTION_ACTIVITY_V5 = (
    "Exactly one person is present throughout this recording. "
    "Classify that person's activity using the temporal-envelope, spatial, and "
    "3 spectral cues below. Report primary activity, runner_up, calibrated confidence, "
    "and physical reasoning."
)

SYSTEM_PROMPT_ACTIVITY_V5 = f"""
You are an expert RF sensing system analyzing Wi-Fi Channel State Information (CSI)
features for human activity recognition, grounded in wireless multipath propagation physics.

TASK:
Exactly one person is present in this recording. Identify the single activity
that best describes this person's activity over the recording.
Do not perform person-presence detection or estimate the number of people.

ALLOWED ACTIVITY LABELS:
{", ".join(ACTIVITY_LABELS)}

INPUT FEATURES PROVIDED:
1. `recording_profile`: 100ms normalized motion (`global_rel_motion_delta`, `global_rel_trajectory_std`),
   signed posture shift (`dominant_signed_posture_shift`, `mean_signed_posture_shift`),
   `max_burst_ratio`, `active_window_fraction`, `active_link_fraction`,
   `mid_to_edge_return_ratio`, `elevation_ratio_shift`.
2. `temporal_envelope`: `global_q4_rel_profile`, `dominant_link_q4_rel_profile`, `window_delta_q4`.
3. `spectral_cues` (3 key subcarrier/Doppler metrics):
   - `subcarrier_coherence_ratio`: ~0.35 = thermal noise; >0.44 = coherent body scattering.
   - `mid_freq_ratio_2_6hz`: 2-6 Hz energy fraction (elevated in jump/wave).
   - `high_to_low_doppler_ratio`: >1.3 = rapid impulse (jump); <1.0 = slow posture/bend.
4. `motion_summary`: Per-link normalized metrics.

PHYSICAL SENSING DECISION HIERARCHY:

STAGE 1: STATIONARY PRESENCE ("nothing")
- Criterion: `global_rel_motion_delta <= 0.022` AND `global_rel_trajectory_std <= 0.025`
  AND `active_window_fraction <= 0.40` AND `subcarrier_coherence_ratio <= 0.40`.
  (NEVER classify as "nothing" if `global_rel_trajectory_std > 0.025` or `max_burst_ratio >= 7.0`.)

STAGE 2: CONTINUOUS WHOLE-BODY MOTION ("walk" vs "rotation")
- HARD GUARD: If `active_window_fraction >= 0.70` AND `min(window_delta_q4) >= 0.030`,
  classify here at Stage 2 ("walk" or "rotation") and NEVER fall through to Stage 3.
- Criterion: `active_window_fraction >= 0.68` AND `global_rel_motion_delta >= 0.036`
  AND at least 3 quarters in `window_delta_q4` > 0.030:
  - "walk": `global_rel_trajectory_std >= 0.070` or Q3/Q4 `window_delta_q4 >= 0.070`
    or clear monotonic drift across `global_q4_rel_profile`.
  - "rotation": Sustained moderate cyclical `window_delta_q4` (~0.032-0.066, no quarter >= 0.070).

STAGE 3: PERMANENT POSTURAL TRANSITIONS ("stand_up", "lie_down", "sit_down")
- PRECONDITION: Only enter Stage 3 if `active_window_fraction < 0.70` OR `min(window_delta_q4) <= 0.029`.
- "stand_up": `dominant_signed_posture_shift >= +0.15` or `elevation_ratio_shift >= +0.25`
  (with `mean_signed_posture_shift > 0`).
- "lie_down": `mean_signed_posture_shift <= -0.025` AND `dominant_signed_posture_shift <= -0.055`
  AND `active_window_fraction <= 0.67` AND `mid_to_edge_return_ratio >= 0.50`
  AND `elevation_ratio_shift > -0.25`.
- "sit_down": `dominant_signed_posture_shift <= -0.15` or `elevation_ratio_shift <= -0.25`
  when not captured by `lie_down`. (NEVER classify as `sit_down` if `abs(dominant_signed_posture_shift) < 0.12`.)

STAGE 4: IMPULSIVE VERTICAL ACTION ("jump")
- Criterion: `max_burst_ratio >= 6.8` AND `active_window_fraction <= 0.65`
  AND `subcarrier_coherence_ratio >= 0.46` AND `high_to_low_doppler_ratio >= 1.5`
  AND `mean_signed_posture_shift > -0.025` AND `abs(dominant_signed_posture_shift) <= 0.12`.
  (NEVER classify as "jump" if `max_burst_ratio < 6.5` or `active_window_fraction > 0.65` or `subcarrier_coherence_ratio < 0.46`.)

STAGE 5: LOCALIZED LIMB GESTURE ("wave")
- Criterion: `global_rel_motion_delta <= 0.032` AND `active_window_fraction <= 0.60`
  AND `abs(mean_signed_posture_shift) < 0.025`
  AND (`max_burst_ratio < 5.0` or `subcarrier_coherence_ratio <= 0.44`).

STAGE 6: TWO-PHASE BEND AND RECOVER ("pick_up")
- HARD GUARD: NEVER classify as "pick_up" if `max_burst_ratio < 5.0` OR `mean_signed_posture_shift <= -0.025`.
- Criterion: `max_burst_ratio >= 5.2` AND `high_to_low_doppler_ratio < 1.2`
  AND `abs(dominant_signed_posture_shift) < 0.12` AND `mean_signed_posture_shift > -0.025`.

CONFIDENCE CALIBRATION RULES:
- High (0.80 - 0.99): Cleanly satisfies one stage with wide margins.
- Moderate (0.55 - 0.75): Near a boundary or minor overlap with `runner_up`.
- Low (0.35 - 0.50): Metrics conflict across stages.

OUTPUT FORMAT (exact JSON only, no Markdown fences):
{{
  "activity": "walk",
  "runner_up": "rotation",
  "confidence": 0.75,
  "reason": "Brief explanation referencing temporal envelope, posture shift, and spectral_cues."
}}
""".strip()


def build_activity_only_prompt_v5(record: dict[str, Any]) -> list[dict[str, str]]:
    feats = record.get("features") or record
    sp = feats.get("subcarrier_profile") or {}
    dp = feats.get("doppler_profile") or {}
    spectral_cues = {
        "subcarrier_coherence_ratio": sp.get("subcarrier_coherence_ratio"),
        "mid_freq_ratio_2_6hz": dp.get("mid_freq_ratio_2_6hz"),
        "high_to_low_doppler_ratio": dp.get("high_to_low_doppler_ratio"),
    }
    payload = {
        "environment": record.get("environment"),
        "wifi_band": record.get("wifi_band"),
        "recording_profile": feats.get("recording_profile"),
        "temporal_envelope": feats.get("temporal_envelope"),
        "spectral_cues": spectral_cues,
        "motion_summary": feats.get("motion_summary"),
    }
    user_content = (
        TASK_INSTRUCTION_ACTIVITY_V5
        + "\n\nCSI-derived input:\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n\nReturn the required JSON object only."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_ACTIVITY_V5},
        {"role": "user", "content": user_content},
    ]


def prompt_version_activity_v5() -> str:
    return PROMPT_VERSION_ACTIVITY_V5
