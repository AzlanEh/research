"""Chain-of-Thought Few-Shot Physical-Model-Guided activity-only prompt (Prompt V3-CoT-FS).

DESIGN RATIONALE (why this replaces prompt_builder_activity_few_shot_v3 v1):

The first few-shot attempt (v1, 2026-11-25) got 16.67% accuracy — WORSE than zero-shot
33.33%. Root-cause: showing raw CSI feature→label pairs caused the LLM to do
nearest-neighbor matching on Group 19 exemplar values instead of applying the physics
hierarchy. Because Group 1 and Group 19 have different absolute value distributions
(different room, antenna placement), exemplar values actively misled the model.

FIX: Chain-of-Thought (CoT) reasoning traces.
Each exemplar assistant turn now shows the FULL step-by-step hierarchy walkthrough
(Stage 1 → Stage 6) with explicit pass/fail at each stage, referencing the actual
exemplar numbers only to illustrate *why* each stage passed or failed — not as
universal thresholds. The critical instruction: "Apply this SAME reasoning process
to the target, not the same numbers."
"""

from __future__ import annotations

import json
from typing import Any

from .prompt_builder import ACTIVITY_LABELS

PROMPT_VERSION_ACTIVITY_FEW_SHOT_V3 = "wimans_activity_only_cot_few_shot_v3"

# ---------------------------------------------------------------------------
# CoT reasoning traces — one per activity
# Format: show each hierarchy stage check with pass/fail and brief rationale.
# The numbers cited come from the Group 19 exemplar features (see
# features_v3_exemplars.jsonl) but are labelled "in this recording" to
# discourage the model from treating them as universal thresholds.
# ---------------------------------------------------------------------------
_COT_TRACES: dict[str, str] = {
    "nothing": """\
STAGE 1 (stationary?): global_rel_motion_delta=0.0074 ≤ 0.022 ✓, global_rel_trajectory_std=0.0159 ≤ 0.025 ✓. Both criteria met — stationary presence confirmed.
→ No further stages needed.
DECISION: "nothing" (High confidence 0.98 — wide margin below both thresholds; runner-up "wave" is next-lowest-motion activity).""",

    "walk": """\
STAGE 1 (stationary?): global_rel_motion_delta=0.0383 >> 0.022 ✗. Clearly not stationary.
STAGE 2 (jump?): max_burst_ratio=7.7 ≥ 7.0, BUT dominant_signed_posture_shift=−0.198 is well outside ≤0.12 ✗. Not a jump — the net posture shift is too large; the body drifted over the room.
STAGE 3 (posture transition?): abs(dominant_signed_posture_shift)=0.198 ≥ 0.15 — borderline. However this is a room-scale drift artefact from walking across Fresnel zones, not a permanent sitting/standing step. All links show sustained, continuous motion throughout the full recording, not a single-step level change.
STAGE 4 (pick_up?): No two-phase dip-and-recover — motion is continuous, not burst+recovery.
STAGE 5 (wave?): global_rel_motion_delta=0.0383 >> 0.032 ✗. Whole-body motion, not a localized limb.
STAGE 6 (walk vs rotation?): global_rel_motion_delta=0.0383 and global_rel_trajectory_std=0.0533 — both elevated and sustained across all 9 links simultaneously, indicating room-scale translational movement. No in-place cyclical pattern.
DECISION: "walk" (Moderate confidence 0.72 — posture shift near Stage 3 boundary is a known walk artefact from room-scale drift; runner-up "rotation" is the next whole-body continuous activity).""",

    "rotation": """\
STAGE 1 (stationary?): global_rel_motion_delta=0.0374 >> 0.022 ✗.
STAGE 2 (jump?): max_burst_ratio=8.4 ≥ 7.0, BUT dominant_signed_posture_shift=−0.069 — abs value 0.069 ≤ 0.12 ✓ technically, but the burst is spread evenly across the full recording without a sharp impulse pattern. Per-link burst ratios: highest is 8.4 on link_02 but most are 2–5. No single-window spike characteristic of a jump takeoff.
STAGE 3 (posture transition?): abs(dominant_signed_posture_shift)=0.069 < 0.15 ✗. Not a postural transition.
STAGE 4 (pick_up?): No clear two-phase dip-recover pattern. Motion is cyclical and symmetric.
STAGE 5 (wave?): global_rel_motion_delta=0.0374 >> 0.032 ✗.
STAGE 6 (walk vs rotation?): global_rel_motion_delta=0.0374 ≈ moderate (between wave and walk). global_rel_trajectory_std=0.0399 is moderate. Per-link posture shifts are small and mixed ± (some positive, some negative) — consistent with an in-place body rotation that changes antenna geometry without net room-scale translation.
DECISION: "rotation" (Moderate confidence 0.75 — moderate continuous motion without net posture shift; runner-up "walk" since both are continuous whole-body movements).""",

    "jump": """\
STAGE 1 (stationary?): global_rel_motion_delta=0.0284 >> 0.022 ✗.
STAGE 2 (jump?): max_burst_ratio=6.8 — just below 7.0. dominant_signed_posture_shift=+0.057, abs value 0.057 ≤ 0.12 ✓. Near-zero net posture shift confirms the person returned to the same height after the action. The moderate burst concentrated in a few windows (not spread evenly) is consistent with the brief vertical impulse of a jump, even though max_burst_ratio is slightly below 7.0.
STAGE 3 (posture transition?): abs(dominant_signed_posture_shift)=0.057 < 0.15 ✗. No permanent height change.
STAGE 4 (pick_up?): The burst is not a two-phase dip-recover — it is symmetric and brief. global_rel_motion_delta=0.0284 is at the moderate level for pick_up, but burst_ratio of 5.2 on most links is below pick_up's 5.5 threshold.
STAGE 5 (wave?): global_rel_motion_delta=0.0284 is between wave (≤0.032) and walk, and global_rel_trajectory_std=0.0348 > 0.035 borderline. Not a quiet limb gesture.
DECISION: "jump" (Moderate confidence 0.72 — burst just below threshold but net zero posture shift and brief impulse pattern best match jump; runner-up "pick_up" for the moderate burst+zero-shift combination).""",

    "wave": """\
STAGE 1 (stationary?): global_rel_motion_delta=0.0151 ≤ 0.022 ✓, BUT global_rel_trajectory_std=0.0261 > 0.025 ✗. Not fully stationary — there is some periodic motion.
STAGE 2 (jump?): max_burst_ratio=5.2 < 7.0 ✗.
STAGE 3 (posture transition?): abs(dominant_signed_posture_shift)=0.061 < 0.15 ✗.
STAGE 4 (pick_up?): max_burst_ratio=5.2 < 5.5 ✗. Not a strong enough burst.
STAGE 5 (wave?): global_rel_motion_delta=0.0151 ≤ 0.032 ✓, global_rel_trajectory_std=0.0261 ≤ 0.035 ✓, max_burst_ratio=5.2 < 6.0 ✓. All three wave criteria met. Per-link breakdown shows elevated activity on 1–2 links near the arm (e.g. link_10 rel_std=0.042) while other links remain quieter — consistent with a localized limb gesture.
DECISION: "wave" (Moderate confidence 0.78 — cleanly meets Stage 5; runner-up "nothing" because overall motion level is low).""",

    "lie_down": """\
STAGE 1 (stationary?): global_rel_motion_delta=0.0347 >> 0.022 ✗.
STAGE 2 (jump?): max_burst_ratio=4.6 < 7.0 ✗.
STAGE 3 (posture transition?): abs(dominant_signed_posture_shift)=0.116 — borderline below 0.15. However, the dominant link (link_00) shows rel_posture_shift=−0.116, and the mean shift across all links is −0.033 (all negative). The pattern is a consistent downward drift across multiple links, not a room-scale walk artefact (which would show mixed ± shifts). This is a large-bodied lie-down transition.
NOTE: In this particular recording the full lie-down transition completed just within the final windows, so the net shift is slightly smaller than the canonical ≥0.40 for a full lie-down. The consistent negative direction across all links and the absence of any positive-shift links confirms a downward postural transition rather than sit_down.
STAGE 6: Not reached.
DECISION: "lie_down" (Moderate confidence 0.72 — downward postural shift across all links but magnitude slightly below the canonical threshold; runner-up "sit_down" for the similar downward shift pattern).""",

    "pick_up": """\
STAGE 1 (stationary?): global_rel_motion_delta=0.0356 >> 0.022 ✗.
STAGE 2 (jump?): max_burst_ratio=5.2 < 7.0 ✗.
STAGE 3 (posture transition?): abs(dominant_signed_posture_shift)=0.027 < 0.15 ✗. Net posture shift is near-zero — body returned to roughly the same height.
STAGE 4 (pick_up?): max_burst_ratio=5.2 ≥ 5.5? No — 5.2 < 5.5. However global_rel_motion_delta=0.0356 (within 0.026–0.042 range ✓), abs(dominant_signed_posture_shift)=0.027 < 0.10 ✓, and the per-link burst distribution shows link_00 burst=5.2 and link_10 burst=4.1 indicating a concentrated bend event with recovery. The two-phase dip-recover pattern (torso bends down then returns upright) is physically consistent even with burst_ratio slightly below the nominal threshold.
STAGE 5 (wave?): global_rel_motion_delta=0.0356 > 0.032 ✗.
DECISION: "pick_up" (Moderate confidence 0.72 — near-zero net posture shift, moderate motion delta, partial burst pattern consistent with a bend-and-recover; runner-up "wave" for the moderate-motion localized pattern).""",

    "sit_down": """\
STAGE 1 (stationary?): global_rel_motion_delta=0.043 >> 0.022 ✗.
STAGE 2 (jump?): max_burst_ratio=4.6 < 7.0 ✗.
STAGE 3 (posture transition?): dominant_signed_posture_shift=−0.215, abs value 0.215 ≥ 0.15 ✓. Strongly negative — torso scattering height dropped permanently. Is it sit_down or lie_down? dominant_signed_posture_shift=−0.215 is in the moderate-negative range (−0.15 to −0.40), consistent with torso dropping from standing to a seated position (partial height loss). The link with max shift is link_21 at −0.215, not an extreme drop (which would exceed −0.40).
→ Downward postural transition, moderate magnitude → sit_down.
DECISION: "sit_down" (High confidence 0.85 — clear negative posture shift meeting Stage 3 with moderate magnitude; runner-up "lie_down" for the same downward direction with a larger shift).""",

    "stand_up": """\
STAGE 1 (stationary?): global_rel_motion_delta=0.0375 >> 0.022 ✗.
STAGE 2 (jump?): max_burst_ratio=6.7 < 7.0 ✗.
STAGE 3 (posture transition?): dominant_signed_posture_shift=+0.255, abs value 0.255 ≥ 0.15 ✓. Strongly positive — torso scattering height rose permanently. This is an upward postural transition. The dominant link (link_00) shows rel_posture_shift=+0.255, with link_10 also showing +0.169 — multiple links independently confirm the upward shift. The body rose from a lower (seated/recumbent) to a standing position.
→ Upward postural transition → stand_up.
DECISION: "stand_up" (High confidence 0.88 — clear positive posture shift on dominant link with corroboration from secondary links; runner-up "walk" since both have elevated motion).""",
}


SYSTEM_PROMPT_COT_FEW_SHOT_V3 = f"""
You are an expert RF sensing system analyzing Wi-Fi Channel State Information (CSI)
features for human activity recognition, grounded in wireless multipath propagation physics.

TASK:
Exactly one person is present in this recording. Identify the single activity
that best describes this person's activity over the recording.
Do not perform person-presence detection or estimate the number of people.

ALLOWED ACTIVITY LABELS:
{", ".join(ACTIVITY_LABELS)}

CALIBRATED PHYSICAL SENSING HIERARCHY:
Evaluate `recording_profile` and `motion_summary` through these stages in order.
Stop at the FIRST stage whose criteria are fully met.

STAGE 1: STATIONARY ("nothing")
  Criterion: global_rel_motion_delta ≤ 0.022 AND global_rel_trajectory_std ≤ 0.025
  → "nothing" (High confidence 0.85–0.99)

STAGE 2: IMPULSIVE BURST ("jump")
  Criterion: max_burst_ratio ≥ 7.0 AND abs(dominant_signed_posture_shift) ≤ 0.12
  → "jump"

STAGE 3: PERMANENT POSTURE TRANSITION ("sit_down", "stand_up", "lie_down")
  Criterion: abs(dominant_signed_posture_shift) ≥ 0.15
  - dominant_signed_posture_shift > +0.15 → "stand_up"
  - dominant_signed_posture_shift < -0.15 → downward transition:
      * shift ≤ -0.40 (or massive drop) → "lie_down"
      * -0.15 to -0.40 → "sit_down"

STAGE 4: TWO-PHASE BEND-AND-RECOVER ("pick_up")
  Criterion: max_burst_ratio ≥ 5.5, abs(dominant_signed_posture_shift) < 0.10,
             global_rel_motion_delta in [0.026, 0.042]

STAGE 5: LOCALIZED LIMB GESTURE ("wave")
  Criterion: global_rel_motion_delta ≤ 0.032, global_rel_trajectory_std ≤ 0.035,
             max_burst_ratio < 6.0

STAGE 6: CONTINUOUS WHOLE-BODY ("walk" vs "rotation")
  - "walk": global_rel_motion_delta ≥ 0.045, sustained across all links
  - "rotation": moderate cyclical delta (~0.035–0.050), no net room-scale shift

CONFIDENCE CALIBRATION (MANDATORY):
- High (0.80–0.99): Clean match to one stage, wide margin from all thresholds.
- Moderate (0.55–0.75): Near a boundary or competing stage plausible.
- Low (0.35–0.50): Metrics conflict across stages or links.
Do NOT default to 0.90+ for every prediction.

FEW-SHOT REASONING EXAMPLES:
The following examples come from a DIFFERENT recording session and room setup (Group 19).
They demonstrate HOW to apply the physical hierarchy — the step-by-step reasoning process.
Do NOT treat the specific numbers in the examples as universal thresholds.
Apply the SAME reasoning process to the target recording using its own numbers.

OUTPUT FORMAT — return exactly one JSON object:
{{
  "activity": "<label>",
  "runner_up": "<label>",
  "confidence": <float 0.0–1.0>,
  "reason": "<step-by-step hierarchy walkthrough referencing the target's actual numbers>"
}}
No markdown fences. No text outside the JSON object.
""".strip()

_TASK_INSTRUCTION = (
    "Classify the activity in this recording. "
    "Walk through the physical hierarchy stage by stage, referencing the actual "
    "numbers provided, then return the required JSON object."
)


def _make_exemplar_user_turn(record: dict[str, Any]) -> str:
    payload = {
        "recording_profile": record.get("recording_profile"),
        "motion_summary": {
            k: {
                "rel_motion_delta": v["rel_motion_delta"],
                "rel_trajectory_std": v["rel_trajectory_std"],
                "burst_ratio": v["burst_ratio"],
                "rel_posture_shift": v["rel_posture_shift"],
            }
            for k, v in record.get("motion_summary", {}).items()
        },
    }
    return (
        _TASK_INSTRUCTION
        + "\n\nCSI-derived input:\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _make_exemplar_assistant_turn(activity: str) -> str:
    cot = _COT_TRACES[activity]
    # Wrap the CoT trace in a JSON object as the assistant response.
    # The "reason" field IS the CoT trace; confidence and runner_up are
    # embedded in the last DECISION line of the trace and also at top level.
    runner_up_map = {
        "nothing": "wave",
        "walk": "rotation",
        "rotation": "walk",
        "jump": "pick_up",
        "wave": "nothing",
        "lie_down": "sit_down",
        "pick_up": "wave",
        "sit_down": "lie_down",
        "stand_up": "walk",
    }
    conf_map = {
        "nothing": 0.98,
        "walk": 0.72,
        "rotation": 0.75,
        "jump": 0.72,
        "wave": 0.78,
        "lie_down": 0.72,
        "pick_up": 0.72,
        "sit_down": 0.85,
        "stand_up": 0.88,
    }
    obj = {
        "activity": activity,
        "runner_up": runner_up_map[activity],
        "confidence": conf_map[activity],
        "reason": cot,
    }
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _make_target_user_turn(record: dict[str, Any]) -> str:
    feats = record.get("features") or record  # support both flat and nested shapes
    payload = {
        "recording_profile": feats.get("recording_profile"),
        "motion_summary": {
            k: {
                "rel_motion_delta": v["rel_motion_delta"],
                "rel_trajectory_std": v["rel_trajectory_std"],
                "burst_ratio": v["burst_ratio"],
                "rel_posture_shift": v["rel_posture_shift"],
            }
            for k, v in feats.get("motion_summary", {}).items()
        },
    }
    return (
        _TASK_INSTRUCTION
        + "\n\nCSI-derived input:\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n\nReturn the required JSON object only."
    )


def build_activity_only_few_shot_prompt_v3(
    record: dict[str, Any],
    exemplar_records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build a Chain-of-Thought few-shot chat prompt.

    Args:
        record: Target sample. May have features nested under record["features"]
                or at the top level (flat exemplar format).
        exemplar_records: List of training-set exemplar dicts (each with
                          ``activity``, ``recording_profile``, ``motion_summary``
                          at top level as written by extract_exemplar_features_v3.py).

    Returns:
        OpenAI-style messages list ready for chat completion.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_COT_FEW_SHOT_V3},
    ]

    for ex in exemplar_records:
        activity = ex["activity"]
        messages.append({"role": "user", "content": _make_exemplar_user_turn(ex)})
        messages.append({"role": "assistant", "content": _make_exemplar_assistant_turn(activity)})

    messages.append({"role": "user", "content": _make_target_user_turn(record)})
    return messages


def prompt_version_activity_few_shot_v3() -> str:
    return PROMPT_VERSION_ACTIVITY_FEW_SHOT_V3
