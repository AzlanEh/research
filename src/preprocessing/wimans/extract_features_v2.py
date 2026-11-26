"""Temporal WiMANS amplitude-CSI features (v2).

V1 summarised a whole recording into a few recording-level statistics, which
discarded the order in which the signal changed. V2 keeps an explicit,
chronological per-window sequence and adds two derived views:

1. ``temporal_sequence`` -- one entry per 100-packet window (including the
   final partial window). Each entry holds ``window_mean``, ``window_std`` and
   ``temporal_delta`` for all nine anonymous links.
2. ``movement_intensity`` -- the per-window ``temporal_delta`` series for each
   link, preserved instead of aggregated across the recording.
3. ``cross_link_synchronization`` -- Pearson correlations between the nine
   movement-intensity series, stored as a symmetric 9 x 9 matrix.

The feature output deliberately contains no activity labels, user counts or
ground-truth annotations. Windows are never labelled individually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


WINDOW_SIZE = 100
FEATURE_VERSION = "wimans_recording_v2"
MIN_COMPLETE_WINDOWS_FOR_SYNCHRONIZATION = 2

LINK_NAMES = [f"link_{tx}{rx}" for tx in range(3) for rx in range(3)]
NUM_UNIQUE_OFF_DIAGONAL_PAIRS = len(LINK_NAMES) * (len(LINK_NAMES) - 1) // 2


def _pearson(values_a: list[float | None], values_b: list[float | None]) -> float | None:
    """Return Pearson r for paired, defined observations, or None.

    ``None`` is used for undefined correlations so that the result stays
    JSON-safe. A constant series has no defined correlation.
    """
    pairs = [
        (a, b)
        for a, b in zip(values_a, values_b)
        if a is not None and b is not None
    ]
    if len(pairs) < 2:
        return None

    a = np.asarray([p[0] for p in pairs], dtype=np.float64)
    b = np.asarray([p[1] for p in pairs], dtype=np.float64)

    if np.all(a == a[0]) or np.all(b == b[0]):
        return None

    correlation = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(correlation):
        return None
    return correlation


def extract_recording_features_v2(
    csi: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> dict[str, Any]:
    """Extract ordered temporal features from a (T, 3, 3, 30) CSI recording.

    The final partial window is retained. All nine anonymous links are kept in
    TX/RX order. The output is JSON-serializable with ``allow_nan=False``.
    """
    x = np.asarray(csi)

    if x.ndim != 4 or x.shape[1:] != (3, 3, 30):
        raise ValueError(f"Expected shape (T, 3, 3, 30), got {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("Recording must contain at least two packets")
    if not np.isfinite(x).all():
        raise ValueError("CSI contains NaN or infinite values")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")

    x = x.astype(np.float64, copy=False)
    num_packets = int(x.shape[0])
    complete_windows = num_packets // window_size

    windows = [
        x[start:start + window_size]
        for start in range(0, num_packets, window_size)
    ]

    temporal_sequence: list[dict[str, Any]] = []
    movement_intensity: dict[str, list[float | None]] = {
        name: [] for name in LINK_NAMES
    }

    for window_index, window in enumerate(windows):
        start_packet = window_index * window_size
        links: dict[str, dict[str, float | None]] = {}

        for tx in range(3):
            for rx in range(3):
                link_name = f"link_{tx}{rx}"
                values = window[:, tx, rx, :]
                packet_means = values.mean(axis=1)

                if len(packet_means) > 1:
                    temporal_delta = float(
                        np.abs(np.diff(packet_means)).mean()
                    )
                else:
                    # A one-packet window has no consecutive differences.
                    temporal_delta = None

                links[link_name] = {
                    "window_mean": float(values.mean()),
                    "window_std": float(values.std()),
                    "temporal_delta": temporal_delta,
                }
                movement_intensity[link_name].append(temporal_delta)

        temporal_sequence.append(
            {
                "window_index": window_index,
                "start_packet": start_packet,
                "end_packet": start_packet + len(window),
                "num_packets": int(len(window)),
                "is_partial": len(window) < window_size,
                "links": links,
            }
        )

    matrix: list[list[float | None]] = [
        [None for _ in LINK_NAMES] for _ in LINK_NAMES
    ]
    defined_off_diagonal_pairs = 0

    if complete_windows >= MIN_COMPLETE_WINDOWS_FOR_SYNCHRONIZATION:
        for i, link_i in enumerate(LINK_NAMES):
            for j, link_j in enumerate(LINK_NAMES):
                if j < i:
                    continue

                if i == j:
                    defined = [
                        value
                        for value in movement_intensity[link_i]
                        if value is not None
                    ]
                    if len(defined) >= 2 and not np.all(
                        np.asarray(defined) == defined[0]
                    ):
                        correlation = 1.0
                    else:
                        correlation = None
                else:
                    correlation = _pearson(
                        movement_intensity[link_i],
                        movement_intensity[link_j],
                    )

                matrix[i][j] = correlation
                matrix[j][i] = correlation
                if i != j and correlation is not None:
                    defined_off_diagonal_pairs += 1

    return {
        "feature_version": FEATURE_VERSION,
        "window_size_packets": window_size,
        "num_packets": num_packets,
        "num_windows": len(windows),
        "complete_windows": complete_windows,
        "temporal_sequence": temporal_sequence,
        "movement_intensity": movement_intensity,
        "cross_link_synchronization": {
            "method": "pearson",
            "links": list(LINK_NAMES),
            "matrix": matrix,
            "minimum_complete_windows": MIN_COMPLETE_WINDOWS_FOR_SYNCHRONIZATION,
            "unique_off_diagonal_pairs": NUM_UNIQUE_OFF_DIAGONAL_PAIRS,
            "defined_off_diagonal_pairs": defined_off_diagonal_pairs,
        },
    }


def extract_manifest_v2(
    manifest_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> None:
    """Extract v2 features for each manifest entry into a JSONL file."""
    with manifest_path.open() as manifest_file:
        entries = [
            json.loads(line)
            for line in manifest_file
            if line.strip()
        ]

    if limit is not None:
        entries = entries[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    with output_path.open("w") as output:
        for entry in entries:
            csi_path = Path(entry["csi_path"])
            csi = np.load(csi_path, mmap_mode="r")

            features = extract_recording_features_v2(csi)

            # Deliberately exclude number_of_users and users/activities.
            record = {
                "sample_id": entry["sample_id"],
                "group_id": entry["group_id"],
                "environment": entry.get("environment"),
                "wifi_band": entry.get("wifi_band"),
                "csi_shape": list(csi.shape),
                "features": features,
            }

            output.write(json.dumps(record, allow_nan=False) + "\n")
            processed += 1

            if processed % 100 == 0:
                print(f"Processed {processed}/{len(entries)}")

    print(f"Processed {processed} recordings -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ordered temporal WiMANS CSI features (v2)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/wimans/manifest.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/wimans/features_v2.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    extract_manifest_v2(args.manifest, args.output, args.limit)


if __name__ == "__main__":
    main()
