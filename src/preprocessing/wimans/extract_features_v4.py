"""Temporal-shape & spatial WiMANS amplitude-CSI features (v4).

Extends v3 with compact temporal-envelope and spatial-diversity descriptors
designed to resolve v3 scalar overlaps (e.g., walk vs posture shift, jump vs
lie_down, wave vs pick_up):

1. `recording_profile` (v3 base + v4 temporal/spatial):
   - V3 base: `global_rel_motion_delta`, `global_rel_trajectory_std`,
     `mean_signed_posture_shift`, `dominant_signed_posture_shift`, `max_burst_ratio`.
   - `active_window_fraction`: Fraction of window transitions with global relative
     motion delta > 0.025 (separates continuous walk/rotation from brief events).
   - `active_link_fraction`: Fraction of the 9 links with rel_motion_delta > 0.025
     (separates localized limb gestures like wave from whole-body motion).
   - `peak_duration_windows`: Count of window transitions exceeding 2x median
     global motion delta (brief 1-3 windows for jump vs sustained for walk).
   - `mid_to_edge_return_ratio`: |q4 - q1| / max(|q2-q1|, |q3-q1|, |q4-q1|).
     Near 0 for return-to-baseline events (jump, pick_up); near 1 for permanent
     posture transitions (sit_down, stand_up, lie_down).
   - `elevation_ratio_shift`: Change in (link_00 / link_02) amplitude ratio from
     first quarter to last quarter (spatial antenna-pair asymmetry proxy).
2. `temporal_envelope`:
   - `global_q4_rel_profile`: [0.0, (q2-q1)/q1, (q3-q1)/q1, (q4-q1)/q1] across
     all links, showing the 4-quarter trajectory shape.
   - `dominant_link_q4_rel_profile`: Same 4-quarter trajectory on the link with
     the largest absolute posture shift.
   - `window_delta_q4`: Mean relative motion delta in each quarter [q1, q2, q3, q4]
     showing *when* motion occurs (early, middle spike, late, or all quarters).
3. `motion_summary`: Compact per-link summary (normalized fields only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

WINDOW_SIZE = 100
FEATURE_VERSION = "wimans_recording_v4"
LINK_NAMES = [f"link_{tx}{rx}" for tx in range(3) for rx in range(3)]
ACTIVE_DELTA_THRESHOLD = 0.025


def _quarter_indices(num_windows: int) -> list[np.ndarray]:
    """Split window indices [0, num_windows) into 4 contiguous quarters."""
    splits = np.array_split(np.arange(num_windows), 4)
    return [s for s in splits if len(s) > 0]


def _q4_rel_profile(series: np.ndarray, quarters: list[np.ndarray]) -> list[float]:
    """Compute 4-quarter relative level profile anchored at quarter 1."""
    q_means = [float(np.mean(series[q])) for q in quarters]
    q1 = q_means[0] if abs(q_means[0]) > 1.0 else 1.0
    return [round((qm - q_means[0]) / q1, 4) for qm in q_means]


def extract_recording_features_v4_from_window_means(
    win_means_arr: np.ndarray,
    num_packets: int,
) -> dict[str, Any]:
    """Compute V4 temporal-shape & spatial features from (num_windows, 3, 3) window means."""
    win_means_arr = np.asarray(win_means_arr, dtype=np.float64)
    if win_means_arr.ndim != 3 or win_means_arr.shape[1:] != (3, 3):
        raise ValueError(f"Expected shape (W, 3, 3), got {win_means_arr.shape}")
    num_windows = int(win_means_arr.shape[0])
    if num_windows < 2:
        raise ValueError("Recording must contain at least two windows")

    motion_deltas_arr = np.zeros((num_windows, 3, 3), dtype=np.float64)
    motion_deltas_arr[1:] = np.abs(np.diff(win_means_arr, axis=0))

    motion_summary: dict[str, dict[str, float]] = {}
    rel_deltas: list[float] = []
    rel_stds: list[float] = []
    rel_shifts: list[float] = []
    bursts: list[float] = []
    rel_deltas_per_window = np.zeros((max(num_windows - 1, 1), 9), dtype=np.float64)

    link_idx = 0
    for tx in range(3):
        for rx in range(3):
            link_name = f"link_{tx}{rx}"
            means_series = win_means_arr[:, tx, rx]
            deltas_series = motion_deltas_arr[1:, tx, rx]

            base_amp = float(np.mean(means_series))
            safe_base = base_amp if base_amp > 1.0 else 1.0

            med_delta = float(np.median(deltas_series))
            max_delta = float(np.max(deltas_series))
            burst_ratio = max_delta / (med_delta + 1e-4)

            n_edge = min(5, num_windows // 2)
            if n_edge > 0:
                init_level = float(np.mean(means_series[:n_edge]))
                final_level = float(np.mean(means_series[-n_edge:]))
                posture_shift = final_level - init_level
            else:
                posture_shift = 0.0

            traj_std = float(np.std(means_series))
            mean_motion_delta = float(np.mean(deltas_series))

            r_delta = mean_motion_delta / safe_base
            r_std = traj_std / safe_base
            r_shift = posture_shift / safe_base

            rel_deltas.append(r_delta)
            rel_stds.append(r_std)
            rel_shifts.append(r_shift)
            bursts.append(burst_ratio)
            rel_deltas_per_window[:, link_idx] = deltas_series / safe_base
            link_idx += 1

            motion_summary[link_name] = {
                "baseline_amplitude": round(base_amp, 1),
                "rel_motion_delta": round(r_delta, 4),
                "rel_trajectory_std": round(r_std, 4),
                "burst_ratio": round(burst_ratio, 1),
                "rel_posture_shift": round(r_shift, 4),
            }

    dominant_shift_idx = int(np.argmax(np.abs(rel_shifts)))
    dominant_signed_shift = rel_shifts[dominant_shift_idx]
    dom_tx, dom_rx = divmod(dominant_shift_idx, 3)

    # Temporal-shape descriptors across windows
    global_win_rel_deltas = np.mean(rel_deltas_per_window, axis=1)
    active_window_fraction = float(
        np.mean(global_win_rel_deltas > ACTIVE_DELTA_THRESHOLD)
    )
    active_link_fraction = float(
        np.mean(np.array(rel_deltas) > ACTIVE_DELTA_THRESHOLD)
    )
    med_global_win_delta = float(np.median(global_win_rel_deltas))
    peak_duration_windows = int(
        np.sum(global_win_rel_deltas > max(2.0 * med_global_win_delta, ACTIVE_DELTA_THRESHOLD))
    )

    quarters = _quarter_indices(num_windows)
    # Normalized global series: each link divided by its own baseline before averaging
    norm_links = np.zeros((num_windows, 9), dtype=np.float64)
    idx = 0
    for tx in range(3):
        for rx in range(3):
            s = win_means_arr[:, tx, rx]
            b = float(np.mean(s))
            norm_links[:, idx] = s / (b if b > 1.0 else 1.0)
            idx += 1
    global_norm_series = np.mean(norm_links, axis=1)
    global_q4 = _q4_rel_profile(global_norm_series, quarters)
    dom_q4 = _q4_rel_profile(win_means_arr[:, dom_tx, dom_rx], quarters)

    # Return-to-baseline ratio on dominant link (0 = full return, 1 = permanent step)
    dom_max_dev = max(abs(v) for v in dom_q4)
    mid_to_edge_return_ratio = (
        abs(dom_q4[-1]) / (dom_max_dev + 1e-4) if dom_max_dev > 1e-4 else 0.0
    )

    # Quarter-by-quarter motion activity profile
    delta_quarters = _quarter_indices(len(global_win_rel_deltas))
    window_delta_q4 = [
        round(float(np.mean(global_win_rel_deltas[dq])), 4)
        for dq in delta_quarters
    ]

    # Spatial elevation ratio shift (link_00 / link_02 from q1 to q4)
    q1_idx, q4_idx = quarters[0], quarters[-1]
    l00_q1 = float(np.mean(win_means_arr[q1_idx, 0, 0]))
    l02_q1 = float(np.mean(win_means_arr[q1_idx, 0, 2]))
    l00_q4 = float(np.mean(win_means_arr[q4_idx, 0, 0]))
    l02_q4 = float(np.mean(win_means_arr[q4_idx, 0, 2]))
    elev_q1 = l00_q1 / (l02_q1 if abs(l02_q1) > 1.0 else 1.0)
    elev_q4 = l00_q4 / (l02_q4 if abs(l02_q4) > 1.0 else 1.0)
    elevation_ratio_shift = (elev_q4 - elev_q1) / (abs(elev_q1) + 1e-4)

    recording_profile = {
        "global_rel_motion_delta": round(float(np.mean(rel_deltas)), 4),
        "global_rel_trajectory_std": round(float(np.mean(rel_stds)), 4),
        "mean_signed_posture_shift": round(float(np.mean(rel_shifts)), 4),
        "dominant_signed_posture_shift": round(float(dominant_signed_shift), 4),
        "max_burst_ratio": round(float(np.max(bursts)), 1),
        "active_window_fraction": round(active_window_fraction, 3),
        "active_link_fraction": round(active_link_fraction, 3),
        "peak_duration_windows": peak_duration_windows,
        "mid_to_edge_return_ratio": round(float(mid_to_edge_return_ratio), 3),
        "elevation_ratio_shift": round(float(elevation_ratio_shift), 4),
    }

    temporal_envelope = {
        "global_q4_rel_profile": global_q4,
        "dominant_link": f"link_{dom_tx}{dom_rx}",
        "dominant_link_q4_rel_profile": dom_q4,
        "window_delta_q4": window_delta_q4,
    }

    return {
        "feature_version": FEATURE_VERSION,
        "num_packets": num_packets,
        "num_windows": num_windows,
        "recording_profile": recording_profile,
        "temporal_envelope": temporal_envelope,
        "motion_summary": motion_summary,
    }


def extract_recording_features_v4(
    csi: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> dict[str, Any]:
    x = np.asarray(csi)
    if x.ndim != 4 or x.shape[1:] != (3, 3, 30):
        raise ValueError(f"Expected shape (T, 3, 3, 30), got {x.shape}")
    if x.shape[0] < 4:
        raise ValueError("Recording must contain at least four packets")
    if not np.isfinite(x).all():
        raise ValueError("CSI contains NaN or infinite values")

    x = x.astype(np.float64, copy=False)
    num_packets = int(x.shape[0])
    windows = [
        x[start:start + window_size]
        for start in range(0, num_packets, window_size)
    ]
    num_windows = len(windows)
    win_means_arr = np.zeros((num_windows, 3, 3), dtype=np.float64)
    for w_idx, win in enumerate(windows):
        win_means_arr[w_idx] = win.mean(axis=(0, 3))

    return extract_recording_features_v4_from_window_means(win_means_arr, num_packets)
