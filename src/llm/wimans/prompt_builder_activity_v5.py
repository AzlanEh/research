"""V5.1 Hybrid Physical-Model-Guided activity-only prompt for WiMANS.

Builds on V4+V5 hybrid (48.89%, Macro F1 0.4869) by pre-computing explicit
`physical_regime_cues` (deterministic physical flags from the CSI descriptors)
so smaller LLMs like gemini-3.5-flash-lite do not violate multi-threshold
inequalities (which caused 10 of the 23 errors in the 45-sample run).
"""

from __future__ import annotations

import json
from typing import Any

from .prompt_builder import ACTIVITY_LABELS

PROMPT_VERSION_ACTIVITY_V5 = "wimans_activity_only_hybrid_v5_1"

TASK_INSTRUCTION_ACTIVITY_V5 = (
    "Exactly one person is present throughout this recording. "
    "Classify that person's activity using `physical_regime_cues`, `recording_profile`, "
    "`temporal_envelope`, and `spectral_cues`. Follow the stage mapping strictly, and "
    "report primary activity, runner_up, calibrated confidence, and physical reasoning."
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
1. `physical_regime_cues` (exact pre-evaluated physical checks — DO NOT contradict these):
   - `motion_regime`: "stationary" | "continuous_whole_body" | "transient_or_localized"
   - `walk_or_rotation`: "likely_walk" | "likely_rotation" | "n/a" (use when motion_regime == "continuous_whole_body")
   - `posture_regime`: "multi_link_downward" | "strong_downward_shift" | "upward_shift" | "neutral"
   - `spectral_regime`: "high_doppler_burst" | "mid_freq_limb" | "low_freq_bend" | "mixed"
   - `continuous_all_quarters`: true if active_window_fraction >= 0.70 and min(window_delta_q4) >= 0.028
2. `recording_profile`, `temporal_envelope`, `spectral_cues`, `motion_summary`.

PHYSICAL SENSING DECISION HIERARCHY (FOLLOW IN ORDER):

STAGE 1: STATIONARY PRESENCE ("nothing")
- If `motion_regime == "stationary"`: classify as "nothing".
- Never classify as "nothing" if `motion_regime != "stationary"`.

STAGE 2: CONTINUOUS WHOLE-BODY MOTION ("walk" vs "rotation")
- If `motion_regime == "continuous_whole_body"` (or `continuous_all_quarters == true`):
  YOU MUST classify here as "walk" or "rotation". NEVER choose `stand_up`, `sit_down`, `lie_down`, or `pick_up`!
  - Follow `walk_or_rotation`:
    * If `walk_or_rotation == "likely_walk"`: classify as "walk".
    * If `walk_or_rotation == "likely_rotation"`: classify as "rotation".

STAGE 3: PERMANENT POSTURAL TRANSITIONS ("lie_down", "stand_up", "sit_down")
- Only enter Stage 3 when `motion_regime != "continuous_whole_body"` and `continuous_all_quarters == false`:
  - "lie_down": If `posture_regime == "multi_link_downward"`: classify as "lie_down".
  - "stand_up": If `posture_regime == "upward_shift"`: classify as "stand_up".
  - "sit_down": If `posture_regime == "strong_downward_shift"`: classify as "sit_down".

STAGE 4: LOCALIZED LIMB GESTURE ("wave")
- If `spectral_regime == "mid_freq_limb"`: classify as "wave".

STAGE 5: IMPULSIVE VERTICAL ACTION ("jump")
- If `spectral_regime == "high_doppler_burst"`: classify as "jump".

STAGE 6: TWO-PHASE BEND AND RECOVER ("pick_up")
- If `posture_regime == "neutral"` and `max_burst_ratio >= 5.0` and `spectral_regime in ("low_freq_bend", "mixed")`:
  classify as "pick_up".
- NEVER classify as "pick_up" if `posture_regime == "multi_link_downward"` (that is "lie_down").

CONFIDENCE CALIBRATION RULES:
- High (0.80 - 0.99): `physical_regime_cues` cleanly map to one stage.
- Moderate (0.55 - 0.75): Near a boundary or mixed cues with plausible `runner_up`.
- Low (0.35 - 0.50): Conflicting cues across stages.

OUTPUT FORMAT (exact JSON only, no Markdown fences):
{{
  "activity": "walk",
  "runner_up": "rotation",
  "confidence": 0.78,
  "reason": "Brief explanation citing physical_regime_cues and supporting metrics."
}}
""".strip()


def _compute_physical_regime_cues(
    rp: dict[str, Any],
    te: dict[str, Any],
    sp: dict[str, Any],
    dp: dict[str, Any],
) -> dict[str, Any]:
    """Derive deterministic physical regime labels from V4+V5 descriptors."""
    delta = float(rp.get("global_rel_motion_delta", 0.0))
    std = float(rp.get("global_rel_trajectory_std", 0.0))
    dom_sh = float(rp.get("dominant_signed_posture_shift", 0.0))
    m_sh = float(rp.get("mean_signed_posture_shift", 0.0))
    burst = float(rp.get("max_burst_ratio", 0.0))
    act_w = float(rp.get("active_window_fraction", 0.0))
    ret_r = float(rp.get("mid_to_edge_return_ratio", 1.0))
    el_sh = float(rp.get("elevation_ratio_shift", 0.0))
    q4_delta = [float(v) for v in (te.get("window_delta_q4") or [0.0, 0.0, 0.0, 0.0])]
    coh = float(sp.get("subcarrier_coherence_ratio", 0.0))
    mid = float(dp.get("mid_freq_ratio_2_6hz", 0.0))
    h2l = float(dp.get("high_to_low_doppler_ratio", 0.0))

    continuous_all_q = act_w >= 0.70 and min(q4_delta) >= 0.028

    if delta <= 0.022 and std <= 0.025 and act_w <= 0.40 and coh <= 0.40 and burst < 7.0 and mid < 0.30:
        motion_regime = "stationary"
    elif (
        continuous_all_q
        or (act_w >= 0.68 and delta >= 0.042 and (sum(1 for v in q4_delta if v >= 0.030) >= 3 or max(q4_delta) >= 0.090))
    ):
        motion_regime = "continuous_whole_body"
    else:
        motion_regime = "transient_or_localized"

    if motion_regime == "continuous_whole_body":
        if std >= 0.095 or coh >= 0.70 or max(q4_delta[2], q4_delta[3]) >= 0.070:
            walk_or_rotation = "likely_walk"
        else:
            walk_or_rotation = "likely_rotation"
    else:
        walk_or_rotation = "n/a"

    if not continuous_all_q and m_sh <= -0.025 and dom_sh <= -0.055 and act_w <= 0.67 and ret_r >= 0.50 and el_sh > -0.25:
        posture_regime = "multi_link_downward"
    elif not continuous_all_q and (dom_sh >= 0.15 or el_sh >= 0.25) and m_sh > 0.0:
        posture_regime = "upward_shift"
    elif not continuous_all_q and (dom_sh <= -0.15 or el_sh <= -0.25):
        posture_regime = "strong_downward_shift"
    else:
        posture_regime = "neutral"

    if (mid >= 0.30 and coh <= 0.46) or (burst < 4.4 and delta <= 0.032 and act_w <= 0.60 and posture_regime == "neutral"):
        spectral_regime = "mid_freq_limb"
    elif h2l >= 2.1 and mid < 0.30 and posture_regime == "neutral" and burst >= 4.4 and coh >= 0.405:
        spectral_regime = "high_doppler_burst"
    elif h2l < 2.0 and burst >= 5.0:
        spectral_regime = "low_freq_bend"
    else:
        spectral_regime = "mixed"

    return {
        "motion_regime": motion_regime,
        "walk_or_rotation": walk_or_rotation,
        "posture_regime": posture_regime,
        "spectral_regime": spectral_regime,
        "continuous_all_quarters": continuous_all_q,
    }


def build_activity_only_prompt_v5(record: dict[str, Any]) -> list[dict[str, str]]:
    feats = record.get("features") or record
    rp = feats.get("recording_profile") or {}
    te = feats.get("temporal_envelope") or {}
    sp = feats.get("subcarrier_profile") or {}
    dp = feats.get("doppler_profile") or {}

    spectral_cues = {
        "subcarrier_coherence_ratio": sp.get("subcarrier_coherence_ratio"),
        "mid_freq_ratio_2_6hz": dp.get("mid_freq_ratio_2_6hz"),
        "high_to_low_doppler_ratio": dp.get("high_to_low_doppler_ratio"),
    }
    physical_regime_cues = _compute_physical_regime_cues(rp, te, sp, dp)

    payload = {
        "environment": record.get("environment"),
        "wifi_band": record.get("wifi_band"),
        "physical_regime_cues": physical_regime_cues,
        "recording_profile": rp,
        "temporal_envelope": te,
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
