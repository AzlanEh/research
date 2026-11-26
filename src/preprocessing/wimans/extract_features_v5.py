"""Subcarrier-resolved & Doppler-band WiMANS amplitude-CSI features (v5).

V5 addresses the physical information ceiling discovered in V4 (~50% on 100ms
subcarrier-averaged window means) by preserving per-subcarrier and
high-rate temporal dynamics from raw (T, 3, 3, 30) CSI amplitude arrays:

1. `recording_profile` + `temporal_envelope` + `motion_summary` (V4 base).
2. `subcarrier_profile` (30 subcarriers preserved):
   - `subcarrier_rms_energy`: RMS of normalized per-subcarrier deviations.
   - `subcarrier_coherence_ratio`: std(subcarrier-mean) / mean(per-subcarrier std).
     High (~0.7-1.0) = coherent whole-body shadowing; low = frequency-selective fade.
   - `top_pca_variance_ratio`: Variance explained by PC1 across 270 (link x sub) streams.
   - `pc1_crest_factor`: Peak |PC1| / RMS(PC1) (sharp impulse detector for jump).
3. `doppler_profile` (25ms-framed FFT across all 270 streams, ~40 Hz sample rate):
   - `low_freq_ratio` (0.3-2.0 Hz): Slow postural shifts (sit_down, stand_up, lie_down).
   - `mid_freq_ratio` (2.0-6.0 Hz): Locomotion & periodic gestures (walk, rotation, wave).
   - `high_freq_ratio` (6.0-20.0 Hz): Impulsive vertical velocity (jump).
   - `spectral_centroid_hz`: Power-weighted mean Doppler frequency (Hz).
   - `high_to_low_doppler_ratio`: high_freq_power / low_freq_power.
4. `impulse_profile` (50ms fine-grained velocity, ~20 Hz):
   - `max_50ms_velocity`: Peak 50ms frame-to-frame subcarrier RMS change.
   - `active_50ms_fraction`: Fraction of 50ms frames above noise floor.
   - `post_peak_quiet_ratio`: Final-quarter 50ms velocity / peak 50ms velocity.

Usage (once /mnt/external_disk is mounted):
    PYTHONPATH=src ./.venv/bin/python src/preprocessing/wimans/extract_features_v5.py --pilot
    PYTHONPATH=src ./.venv/bin/python src/preprocessing/wimans/extract_features_v5.py --full
    PYTHONPATH=src ./.venv/bin/python src/preprocessing/wimans/extract_features_v5.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.wimans.extract_features_v4 import (
    WINDOW_SIZE as V4_WINDOW_SIZE,
    extract_recording_features_v4_from_window_means,
)

FEATURE_VERSION = "wimans_recording_v5"
SAMPLE_RATE_HZ = 1000.0
FRAME_25MS = 25   # 40 Hz effective rate for Doppler FFT (up to 20 Hz)
FRAME_50MS = 50   # 20 Hz effective rate for fine-grained impulse velocity
AMP_DIR_DEFAULT = Path("/mnt/external_disk/gaurav_project/wifi_csi/amp")


def _downsample_frames(x: np.ndarray, frame_size: int) -> np.ndarray:
    """Average (T, 3, 3, 30) into (num_frames, 270) non-overlapping frames."""
    num_frames = x.shape[0] // frame_size
    if num_frames < 4:
        # Fallback if recording is unusually short
        step = max(1, x.shape[0] // 8)
        num_frames = x.shape[0] // step
        trimmed = x[: num_frames * step]
        return trimmed.reshape(num_frames, step, 270).mean(axis=1)
    trimmed = x[: num_frames * frame_size]
    return trimmed.reshape(num_frames, frame_size, 270).mean(axis=1)


def _compute_subcarrier_profile(frames_25ms: np.ndarray) -> dict[str, Any]:
    """Compute subcarrier-resolved coherence & PCA features from (F, 270) frames."""
    means = frames_25ms.mean(axis=0, keepdims=True)
    safe_means = np.where(np.abs(means) > 1.0, means, 1.0)
    norm = (frames_25ms - means) / safe_means  # zero-mean fractional deviation

    subcarrier_rms_energy = float(np.sqrt(np.mean(norm ** 2)))

    # Subcarrier coherence: std of subcarrier-averaged link signals vs mean per-subcarrier std
    norm_links = norm.reshape(norm.shape[0], 9, 30)
    link_avg = norm_links.mean(axis=2)  # (F, 9)
    std_link_avg = float(np.mean(np.std(link_avg, axis=0)))
    mean_sub_std = float(np.mean(np.std(norm_links, axis=0)))
    coherence_ratio = std_link_avg / (mean_sub_std + 1e-6)

    # Top PCA mode across all 270 streams via SVD
    _, s, _ = np.linalg.svd(norm, full_matrices=False)
    var = s ** 2
    total_var = float(np.sum(var))
    top_pca_var_ratio = float(var[0] / total_var) if total_var > 1e-12 else 0.0

    # PC1 time series projection for crest factor (impulse sharpness)
    u, _, _ = np.linalg.svd(norm, full_matrices=False)
    pc1 = u[:, 0] * s[0]
    pc1_rms = float(np.sqrt(np.mean(pc1 ** 2)))
    pc1_crest = float(np.max(np.abs(pc1)) / (pc1_rms + 1e-6))

    return {
        "subcarrier_rms_energy": round(subcarrier_rms_energy, 4),
        "subcarrier_coherence_ratio": round(min(max(coherence_ratio, 0.0), 1.0), 4),
        "top_pca_variance_ratio": round(top_pca_var_ratio, 4),
        "pc1_crest_factor": round(pc1_crest, 2),
    }


def _compute_doppler_profile(frames_25ms: np.ndarray, frame_rate_hz: float) -> dict[str, Any]:
    """Compute Hann-windowed FFT Doppler power bands across all 270 normalized streams."""
    means = frames_25ms.mean(axis=0, keepdims=True)
    safe_means = np.where(np.abs(means) > 1.0, means, 1.0)
    norm = (frames_25ms - means) / safe_means

    n = norm.shape[0]
    hann = np.hanning(n)[:, None]
    windowed = norm * hann
    fft_mag = np.abs(np.fft.rfft(windowed, axis=0)) ** 2
    psd = np.mean(fft_mag, axis=1)  # mean power across 270 streams
    freqs = np.fft.rfftfreq(n, d=1.0 / frame_rate_hz)

    valid = freqs >= 0.3
    freqs_v = freqs[valid]
    psd_v = psd[valid]
    total_power = float(np.sum(psd_v)) + 1e-12

    low_mask = (freqs_v >= 0.3) & (freqs_v < 2.0)
    mid_mask = (freqs_v >= 2.0) & (freqs_v < 6.0)
    high_mask = freqs_v >= 6.0

    low_p = float(np.sum(psd_v[low_mask]))
    mid_p = float(np.sum(psd_v[mid_mask]))
    high_p = float(np.sum(psd_v[high_mask]))

    spectral_centroid = float(np.sum(freqs_v * psd_v) / total_power)
    high_to_low = high_p / (low_p + 1e-12)

    return {
        "low_freq_ratio_0_2hz": round(low_p / total_power, 4),
        "mid_freq_ratio_2_6hz": round(mid_p / total_power, 4),
        "high_freq_ratio_6_20hz": round(high_p / total_power, 4),
        "spectral_centroid_hz": round(spectral_centroid, 2),
        "high_to_low_doppler_ratio": round(min(high_to_low, 99.0), 3),
    }


def _compute_impulse_profile(x: np.ndarray) -> dict[str, Any]:
    """Compute 50ms fine-grained subcarrier velocity profile."""
    frames_50ms = _downsample_frames(x, FRAME_50MS)
    means = frames_50ms.mean(axis=0, keepdims=True)
    safe_means = np.where(np.abs(means) > 1.0, means, 1.0)
    norm = frames_50ms / safe_means

    # Frame-to-frame RMS change across all 270 streams (50ms resolution)
    vel = np.sqrt(np.mean(np.diff(norm, axis=0) ** 2, axis=1))
    max_vel = float(np.max(vel))
    med_vel = float(np.median(vel))
    active_50ms_fraction = float(np.mean(vel > 0.035))

    n_tail = max(1, len(vel) // 4)
    tail_vel = float(np.mean(vel[-n_tail:]))
    post_peak_quiet_ratio = tail_vel / (max_vel + 1e-6)

    return {
        "max_50ms_velocity": round(max_vel, 4),
        "median_50ms_velocity": round(med_vel, 4),
        "burst_50ms_ratio": round(max_vel / (med_vel + 1e-4), 2),
        "active_50ms_fraction": round(active_50ms_fraction, 3),
        "post_peak_quiet_ratio": round(post_peak_quiet_ratio, 3),
    }


def extract_recording_features_v5(
    csi: np.ndarray,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> dict[str, Any]:
    """Extract V4 + V5 Subcarrier & Doppler features from raw (T, 3, 3, 30) CSI."""
    x = np.asarray(csi)
    if x.ndim != 4 or x.shape[1:] != (3, 3, 30):
        raise ValueError(f"Expected shape (T, 3, 3, 30), got {x.shape}")
    if x.shape[0] < 200:
        raise ValueError(f"Recording too short for V5 spectral analysis: {x.shape[0]} packets")
    if not np.isfinite(x).all():
        raise ValueError("CSI contains NaN or infinite values")

    x = x.astype(np.float64, copy=False)
    num_packets = int(x.shape[0])

    # V4 base from 100ms window means
    num_windows = (num_packets + V4_WINDOW_SIZE - 1) // V4_WINDOW_SIZE
    win_means = np.zeros((num_windows, 3, 3), dtype=np.float64)
    for w_idx, start in enumerate(range(0, num_packets, V4_WINDOW_SIZE)):
        win_means[w_idx] = x[start : start + V4_WINDOW_SIZE].mean(axis=(0, 3))
    v4 = extract_recording_features_v4_from_window_means(win_means, num_packets)

    # V5 subcarrier + Doppler + 50ms impulse features
    frames_25ms = _downsample_frames(x, FRAME_25MS)
    eff_rate_hz = sample_rate_hz / FRAME_25MS

    return {
        "feature_version": FEATURE_VERSION,
        "num_packets": num_packets,
        "num_windows": v4["num_windows"],
        "recording_profile": v4["recording_profile"],
        "temporal_envelope": v4["temporal_envelope"],
        "subcarrier_profile": _compute_subcarrier_profile(frames_25ms),
        "doppler_profile": _compute_doppler_profile(frames_25ms, eff_rate_hz),
        "impulse_profile": _compute_impulse_profile(x),
        "motion_summary": v4["motion_summary"],
    }


def _run_self_test() -> None:
    """Verify V5 extractor on synthetic stationary, step-transition, walk, and jump signals."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2.8, 2800)[:, None, None, None]

    # 1. Stationary noise
    csi_nothing = 50.0 + rng.normal(0, 0.3, (2800, 3, 3, 30))
    # 2. Slow postural step (0.5 Hz transition)
    step = 1.0 / (1.0 + np.exp(-8.0 * (t - 1.4)))
    csi_sit = 50.0 * (1.0 - 0.25 * step) + rng.normal(0, 0.3, (2800, 3, 3, 30))
    # 3. Continuous walk (3.5 Hz oscillation)
    csi_walk = 50.0 * (1.0 + 0.12 * np.sin(2 * np.pi * 3.5 * t)) + rng.normal(0, 0.3, (2800, 3, 3, 30))
    # 4. Rapid jump impulse (10 Hz burst at t=1.4s for 0.25s)
    burst_env = np.exp(-0.5 * ((t - 1.4) / 0.10) ** 2)
    csi_jump = 50.0 * (1.0 + 0.30 * burst_env * np.sin(2 * np.pi * 10.0 * t)) + rng.normal(0, 0.3, (2800, 3, 3, 30))

    for name, arr in [("nothing", csi_nothing), ("sit_step", csi_sit), ("walk_3.5hz", csi_walk), ("jump_10hz", csi_jump)]:
        f = extract_recording_features_v5(arr)
        dp = f["doppler_profile"]
        sp = f["subcarrier_profile"]
        ip = f["impulse_profile"]
        print(
            f"{name:<11} | rms={sp['subcarrier_rms_energy']:.4f} coh={sp['subcarrier_coherence_ratio']:.3f} "
            f"pca1={sp['top_pca_variance_ratio']:.3f} | centroid={dp['spectral_centroid_hz']:>5.2f}Hz "
            f"low={dp['low_freq_ratio_0_2hz']:.3f} mid={dp['mid_freq_ratio_2_6hz']:.3f} high={dp['high_freq_ratio_6_20hz']:.3f} | "
            f"vel50={ip['max_50ms_velocity']:.4f} quiet={ip['post_peak_quiet_ratio']:.3f}"
        )


def _extract_batch(sample_ids: list[str], amp_dir: Path, out_path: Path, meta_map: dict[str, dict[str, Any]]) -> None:
    if not amp_dir.exists() or not any(amp_dir.glob("*.npy")):
        print(f"ERROR: Raw amplitude directory not available or empty: {amp_dir}", file=sys.stderr)
        print("Ensure /mnt/external_disk is mounted before running extraction.", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, sid in enumerate(sample_ids, 1):
            csi = np.load(amp_dir / f"{sid}.npy")
            feats = extract_recording_features_v5(csi)
            meta = meta_map.get(sid, {})
            rec = {
                "sample_id": sid,
                "group_id": meta.get("group_id"),
                "environment": meta.get("environment"),
                "wifi_band": meta.get("wifi_band"),
                "features": feats,
            }
            out_f.write(json.dumps(rec) + "\n")
            if idx % 50 == 0 or idx == len(sample_ids):
                print(f"[{idx}/{len(sample_ids)}] Extracted V5 features for {sid}", flush=True)
    print(f"Wrote {len(sample_ids)} V5 feature records to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract V5 Subcarrier + Doppler features.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic signal sanity test")
    parser.add_argument("--pilot", action="store_true", help="Extract 18 pilot samples")
    parser.add_argument("--full", action="store_true", help="Extract all 693 test samples")
    parser.add_argument("--amp-dir", type=Path, default=AMP_DIR_DEFAULT)
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return

    meta_map: dict[str, dict[str, Any]] = {}
    v4_full = Path("data/processed/wimans/features_v4_activity_only.jsonl")
    if v4_full.exists():
        for line in v4_full.open():
            r = json.loads(line)
            meta_map[r["sample_id"]] = r

    if args.pilot:
        sids = [json.loads(l)["sample_id"] for l in Path("data/processed/wimans/features_v3_pilot.jsonl").open()]
        _extract_batch(sids, args.amp_dir, Path("data/processed/wimans/features_v5_pilot.jsonl"), meta_map)
    elif args.full:
        gt = json.loads(Path("experiments/wimans/activity_only_v1/ground_truth.json").read_text())
        _extract_batch(sorted(gt.keys()), args.amp_dir, Path("data/processed/wimans/features_v5_activity_only.jsonl"), meta_map)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
