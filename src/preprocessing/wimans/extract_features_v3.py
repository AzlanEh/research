"""Physical motion WiMANS amplitude-CSI features (v3).

V3 extracts TRUE physical motion descriptors and relative normalized metrics:
1. `recording_profile`: Clear global summary metrics calculated over all links:
   - `global_rel_motion_delta`: Mean relative window-to-window motion.
   - `global_rel_trajectory_std`: Overall relative dynamic range.
   - `signed_posture_shift`: Mean signed fractional shift (final windows - initial windows).
   - `max_signed_posture_shift`: Signed shift on the link with largest change (+ for upward, - for downward).
   - `max_burst_ratio`: Peak-to-median ratio for impulsive burst detection (jump).
2. `motion_summary`: Per-link breakdown of baseline amplitude, relative motion delta,
   relative trajectory std, burst ratio, and relative posture shift.
3. `cross_link_synchronization`: Pearson correlations of window means across links.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

WINDOW_SIZE = 100
FEATURE_VERSION = "wimans_recording_v3"
LINK_NAMES = [f"link_{tx}{rx}" for tx in range(3) for rx in range(3)]


def extract_recording_features_v3(
    csi: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> dict[str, Any]:
    x = np.asarray(csi)
    if x.ndim != 4 or x.shape[1:] != (3, 3, 30):
        raise ValueError(f"Expected shape (T, 3, 3, 30), got {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("Recording must contain at least two packets")
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
    win_stds_arr = np.zeros((num_windows, 3, 3), dtype=np.float64)

    for w_idx, win in enumerate(windows):
        win_means_arr[w_idx] = win.mean(axis=(0, 3))
        win_stds_arr[w_idx] = win.std(axis=(0, 3))

    motion_deltas_arr = np.zeros((num_windows, 3, 3), dtype=np.float64)
    if num_windows > 1:
        motion_deltas_arr[1:] = np.abs(np.diff(win_means_arr, axis=0))

    motion_summary = {}
    rel_deltas = []
    rel_stds = []
    rel_shifts = []
    bursts = []

    for tx in range(3):
        for rx in range(3):
            link_name = f"link_{tx}{rx}"
            means_series = win_means_arr[:, tx, rx]
            deltas_series = motion_deltas_arr[1:, tx, rx] if num_windows > 1 else np.array([0.0])

            base_amp = float(np.mean(means_series))
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
            safe_base = base_amp if base_amp > 1.0 else 1.0

            r_delta = mean_motion_delta / safe_base
            r_std = traj_std / safe_base
            r_shift = posture_shift / safe_base

            rel_deltas.append(r_delta)
            rel_stds.append(r_std)
            rel_shifts.append(r_shift)
            bursts.append(burst_ratio)

            motion_summary[link_name] = {
                "baseline_amplitude": round(base_amp, 1),
                "mean_motion_delta": round(mean_motion_delta, 2),
                "rel_motion_delta": round(r_delta, 4),
                "trajectory_std": round(traj_std, 2),
                "rel_trajectory_std": round(r_std, 4),
                "burst_ratio": round(burst_ratio, 1),
                "posture_shift": round(posture_shift, 2),
                "rel_posture_shift": round(r_shift, 4),
            }

    # Find signed dominant shift:
    dominant_shift_idx = int(np.argmax(np.abs(rel_shifts)))
    dominant_signed_shift = rel_shifts[dominant_shift_idx]

    recording_profile = {
        "global_rel_motion_delta": round(float(np.mean(rel_deltas)), 4),
        "global_rel_trajectory_std": round(float(np.mean(rel_stds)), 4),
        "mean_signed_posture_shift": round(float(np.mean(rel_shifts)), 4),
        "dominant_signed_posture_shift": round(float(dominant_signed_shift), 4),
        "max_burst_ratio": round(float(np.max(bursts)), 1),
    }

    flat_means = win_means_arr.reshape((num_windows, 9))
    with np.errstate(all="ignore"):
        corr_matrix = np.corrcoef(flat_means, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0).tolist()

    return {
        "feature_version": FEATURE_VERSION,
        "num_packets": num_packets,
        "num_windows": num_windows,
        "recording_profile": recording_profile,
        "motion_summary": motion_summary,
        "cross_link_synchronization": {
            "links": LINK_NAMES,
            "matrix": [[round(val, 2) for val in row] for row in corr_matrix],
        },
    }
