"""extract_features.py

Extract structured signal features from a CSI window (50 x 242 complex).

Two representations are produced for each window:

  1. Amplitude time series — the raw |csi| matrix (50 x 242), suitable for
     Wi-Chat-style prompting where the LLM sees the signal evolution.
  2. Structured statistical features — compact summaries of amplitude,
     phase, temporal, and subcarrier behaviour, plus gain diagnostics.

The original complex CSI is preserved unchanged. No AGC correction,
normalization, or gain scaling is applied. Gain diagnostics are recorded
for tracking only (to support the separate AGC investigation), not for
correction.

Designed to run from ``src/``:
    python -m preprocessing.simwisense.extract_features --manifest data/processed/manifest.json --limit 10

Outputs a JSONL feature file to ``data/processed/`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from .load_csi import CsiWindow, load_csi_window, amplitude, phase

# Number of temporal segments to divide the 50-sample window into.
NUM_TEMPORAL_SEGMENTS = 5


# ---------------------------------------------------------------------------
# Amplitude features
# ---------------------------------------------------------------------------
def amplitude_features(csi: np.ndarray) -> dict[str, Any]:
    """Statistics of the amplitude matrix |csi| (shape 50 x 242).

    All values are computed on the raw amplitude; no normalization.
    """
    amps = np.abs(csi)

    # Overall amplitude statistics (scalars).
    overall = {
        "mean": float(amps.mean()),
        "std": float(amps.std()),
        "min": float(amps.min()),
        "max": float(amps.max()),
        "median": float(np.median(amps)),
    }

    # Per-subcarrier statistics across time: shape (242,).
    per_subcarrier_mean = amps.mean(axis=0).tolist()
    per_subcarrier_std = amps.std(axis=0).tolist()
    per_subcarrier_min = amps.min(axis=0).tolist()
    per_subcarrier_max = amps.max(axis=0).tolist()

    # Per-time-sample statistics across subcarriers: shape (50,).
    per_time_mean = amps.mean(axis=1).tolist()
    per_time_std = amps.std(axis=1).tolist()

    # Amplitude dynamic range per subcarrier (max / min), for gain tracking.
    # Guard against division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_subcarrier_range = (amps.max(axis=0) / np.where(amps.min(axis=0) == 0, np.nan, amps.min(axis=0)))
    per_subcarrier_range = np.where(np.isfinite(per_subcarrier_range), per_subcarrier_range, 0.0).tolist()

    return {
        "overall": overall,
        "per_subcarrier_mean": per_subcarrier_mean,
        "per_subcarrier_std": per_subcarrier_std,
        "per_subcarrier_min": per_subcarrier_min,
        "per_subcarrier_max": per_subcarrier_max,
        "per_time_mean": per_time_mean,
        "per_time_std": per_time_std,
        "per_subcarrier_dynamic_range": per_subcarrier_range,
    }


# ---------------------------------------------------------------------------
# Phase features
# ---------------------------------------------------------------------------
def phase_features(csi: np.ndarray) -> dict[str, Any]:
    """Statistics of the phase angle(csi) (shape 50 x 242), in radians.

    Phase is circular; means and stds use circular statistics where
    appropriate.
    """
    ph = np.angle(csi)

    # Circular mean across all elements.
    mean_vector = np.mean(np.exp(1j * ph))
    circular_mean = float(np.angle(mean_vector))
    circular_std = float(np.std(np.angle(np.exp(1j * (ph - circular_mean)))))

    # Phase variation between consecutive time samples (temporal).
    # diff along axis 0: shape (49, 242).
    temporal_diff = np.diff(ph, axis=0)
    # Wrap differences to [-pi, pi].
    temporal_diff = (temporal_diff + np.pi) % (2 * np.pi) - np.pi
    temporal_phase_variation = {
        "mean_abs": float(np.mean(np.abs(temporal_diff))),
        "std": float(np.std(temporal_diff)),
        "max_abs": float(np.max(np.abs(temporal_diff))),
    }

    # Phase variation across subcarriers (spectral).
    # diff along axis 1: shape (50, 241).
    spectral_diff = np.diff(ph, axis=1)
    spectral_diff = (spectral_diff + np.pi) % (2 * np.pi) - np.pi
    spectral_phase_variation = {
        "mean_abs": float(np.mean(np.abs(spectral_diff))),
        "std": float(np.std(spectral_diff)),
        "max_abs": float(np.max(np.abs(spectral_diff))),
    }

    # Phase wrap count: number of times phase jumps by > pi/2 between
    # consecutive time samples, summed over all subcarriers.
    wrap_count = int(np.sum(np.abs(temporal_diff) > np.pi / 2))

    # Per-time-sample circular mean phase (shape 50,).
    per_time_circular_mean = []
    for t in range(ph.shape[0]):
        mv = np.mean(np.exp(1j * ph[t, :]))
        per_time_circular_mean.append(float(np.angle(mv)))

    # Per-subcarrier circular mean phase (shape 242,).
    per_subcarrier_circular_mean = []
    for s in range(ph.shape[1]):
        mv = np.mean(np.exp(1j * ph[:, s]))
        per_subcarrier_circular_mean.append(float(np.angle(mv)))

    return {
        "circular_mean_all": circular_mean,
        "circular_std_all": circular_std,
        "temporal_phase_variation": temporal_phase_variation,
        "spectral_phase_variation": spectral_phase_variation,
        "phase_wrap_count": wrap_count,
        "per_time_circular_mean": per_time_circular_mean,
        "per_subcarrier_circular_mean": per_subcarrier_circular_mean,
    }


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------
def temporal_features(csi: np.ndarray) -> dict[str, Any]:
    """Temporal dynamics of the complex CSI window.

    Includes frame-to-frame differences, segment-level summaries, and
    total signal energy.
    """
    amps = np.abs(csi)

    # Frame-to-frame complex difference: csi[t+1] - csi[t], shape (49, 242).
    frame_diff = np.diff(csi, axis=0)
    frame_diff_amp = np.abs(frame_diff)

    temporal_diff = {
        "mean_abs": float(np.mean(frame_diff_amp)),
        "std": float(np.std(frame_diff_amp)),
        "max_abs": float(np.max(frame_diff_amp)),
    }

    # Split the 50-sample window into NUM_TEMPORAL_SEGMENTS segments.
    n_samples = csi.shape[0]
    segment_size = n_samples // NUM_TEMPORAL_SEGMENTS
    segment_amplitudes = []
    for seg in range(NUM_TEMPORAL_SEGMENTS):
        start = seg * segment_size
        end = (seg + 1) * segment_size if seg < NUM_TEMPORAL_SEGMENTS - 1 else n_samples
        seg_amps = amps[start:end, :]
        segment_amplitudes.append({
            "segment": seg,
            "samples": [start, end],
            "mean_amplitude": float(seg_amps.mean()),
            "std_amplitude": float(seg_amps.std()),
            "max_amplitude": float(seg_amps.max()),
        })

    # Total signal energy (sum of squared magnitude).
    total_energy = float(np.sum(amps ** 2))

    # Amplitude trend: correlation of per-time mean amplitude with time index.
    per_time_mean = amps.mean(axis=1)
    time_idx = np.arange(n_samples, dtype=float)
    if np.std(per_time_mean) > 0 and np.std(time_idx) > 0:
        amplitude_trend = float(np.corrcoef(per_time_mean, time_idx)[0, 1])
    else:
        amplitude_trend = 0.0

    return {
        "frame_to_frame_difference": temporal_diff,
        "temporal_segments": segment_amplitudes,
        "total_energy": total_energy,
        "amplitude_trend_correlation": amplitude_trend,
    }


# ---------------------------------------------------------------------------
# Gain diagnostics (recording only — NOT correction)
# ---------------------------------------------------------------------------
def gain_diagnostics(csi: np.ndarray) -> dict[str, Any]:
    """Record gain-related diagnostics for tracking.

    These values describe the amplitude behaviour within the window and are
    intended to support the separate AGC/gain investigation. They are NOT
    used to correct or normalize the CSI.

    The original CSI is preserved unchanged regardless of these values.
    """
    amps = np.abs(csi)

    # Overall window gain indicators.
    window_mean_amplitude = float(amps.mean())
    window_amplitude_range = float(amps.max() - amps.min())
    window_amplitude_std = float(amps.std())

    # Per-subcarrier gain ratio: max amplitude / min amplitude across time.
    # A high ratio may indicate subcarrier-specific gain changes.
    per_subcarrier_max = amps.max(axis=0)
    per_subcarrier_min = amps.min(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_ratio = per_subcarrier_max / np.where(
            per_subcarrier_min == 0, np.nan, per_subcarrier_min
        )
    gain_ratio = np.where(np.isfinite(gain_ratio), gain_ratio, 0.0).tolist()

    # Temporal gain variation: std of per-segment mean amplitude.
    n_samples = csi.shape[0]
    segment_size = n_samples // NUM_TEMPORAL_SEGMENTS
    segment_means = []
    for seg in range(NUM_TEMPORAL_SEGMENTS):
        start = seg * segment_size
        end = (seg + 1) * segment_size if seg < NUM_TEMPORAL_SEGMENTS - 1 else n_samples
        segment_means.append(float(amps[start:end, :].mean()))
    temporal_gain_variation = float(np.std(segment_means))

    # Max-to-min amplitude ratio for the whole window (potential gain swing).
    with np.errstate(divide="ignore", invalid="ignore"):
        window_gain_ratio = amps.max() / amps.min() if amps.min() > 0 else 0.0

    return {
        "window_mean_amplitude": window_mean_amplitude,
        "window_amplitude_range": window_amplitude_range,
        "window_amplitude_std": window_amplitude_std,
        "window_gain_ratio": float(window_gain_ratio),
        "per_subcarrier_gain_ratio": gain_ratio,
        "temporal_gain_variation": temporal_gain_variation,
        "note": (
            "Gain diagnostics are recorded for tracking only. No AGC "
            "correction or normalization is applied to the CSI."
        ),
    }


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------
def extract_features(window: CsiWindow) -> dict[str, Any]:
    """Extract all feature groups from a CsiWindow.

    Returns a dict matching the proposed output schema:

        {
          "label": "A",
          "shape": [50, 242],
          "dtype": "complex128",
          "amplitude_features": {...},
          "phase_features": {...},
          "temporal_features": {...},
          "gain_diagnostics": {...},
          "source_file": "A_batch/batch_0.mat"
        }

    The original complex CSI is NOT included in the output (to keep files
    compact); it remains available via load_csi.load_csi_window().
    """
    csi = window.csi

    return {
        "label": window.label,
        "shape": list(window.shape),
        "dtype": window.dtype,
        "source_file": window.source_file,
        "split": window.split,
        "amplitude_features": amplitude_features(csi),
        "phase_features": phase_features(csi),
        "temporal_features": temporal_features(csi),
        "gain_diagnostics": gain_diagnostics(csi),
    }


def feature_record_to_jsonl(record: dict[str, Any]) -> str:
    """Serialize a feature record to a single JSONL line."""
    return json.dumps(record)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured CSI features from manifest entries."
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest JSON (default: <repo>/data/processed/manifest.json)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset root (overrides manifest entry paths if needed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N entries (for testing).",
    )
    parser.add_argument(
        "--activities",
        nargs="+",
        default=["A", "B", "C"],
        help="Activity labels to include (default: A B C).",
    )
    parser.add_argument(
        "--per-activity",
        type=int,
        default=None,
        help="Sample this many entries per activity (overrides --limit).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: <repo>/data/processed/features.jsonl)",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root:
    # <root>/src/preprocessing/simwisense -> <root>.
    here = os.path.dirname(os.path.abspath(__file__))
    preprocessing_dir = os.path.dirname(here)
    src_dir = os.path.dirname(preprocessing_dir)
    repo_root = os.path.dirname(src_dir)

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = os.path.join(repo_root, "data", "processed", "manifest.json")

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(repo_root, "data", "processed", "features.jsonl")

    dataset_root = args.dataset
    if dataset_root is None:
        dataset_root = (
            "/mnt/external_disk/SiMWiSense/SiMWiSense"
            "/Data/fine_grained/Classroom/80MHz/3mo/m1"
        )

    # Load manifest.
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    all_entries = manifest["files"]

    # Filter by activity if requested.
    activities = set(args.activities)
    filtered = [e for e in all_entries if e["label"] in activities]

    # Per-activity balanced sampling (overrides --limit).
    if args.per_activity is not None:
        from collections import defaultdict
        by_activity: dict[str, list[dict]] = defaultdict(list)
        for e in filtered:
            by_activity[e["label"]].append(e)
        entries = []
        for act in sorted(by_activity):
            entries.extend(by_activity[act][: args.per_activity])
    elif args.limit is not None:
        entries = filtered[: args.limit]
    else:
        entries = filtered

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    processed = 0
    with open(out_path, "w", encoding="utf-8") as out_fh:
        for entry in entries:
            try:
                window = load_csi_window(
                    mat_path=os.path.join(dataset_root, entry["filename"]),
                    label=entry["label"],
                    split=entry["split"],
                    source_file=entry["filename"],
                )
            except Exception as exc:  # pragma: no cover - skip bad files
                print(f"[extract_features] SKIP {entry['filename']}: {exc}")
                continue

            record = extract_features(window)
            out_fh.write(feature_record_to_jsonl(record) + "\n")
            processed += 1

    print(f"[extract_features] Processed {processed} windows -> {out_path}")


if __name__ == "__main__":
    main()
