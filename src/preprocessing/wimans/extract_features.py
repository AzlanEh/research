"""Recording-level statistical features for WiMANS amplitude CSI."""

import argparse
import json
from pathlib import Path

import numpy as np


WINDOW_SIZE = 100
FEATURE_VERSION = "wimans_recording_v1"


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize a sequence of window-level measurements."""
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
    }


def extract_recording_features(
    csi: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> dict:
    """Extract compact features from a (T, 3, 3, 30) CSI recording."""
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

    links = {}
    window_means = []
    window_stds = []

    # Keep the final partial window.
    windows = [
        x[start:start + window_size]
        for start in range(0, len(x), window_size)
    ]

    for tx in range(3):
        for rx in range(3):
            link_name = f"link_{tx}{rx}"
            link_values = x[:, tx, rx, :]

            link_window_means = []
            link_window_stds = []
            link_window_deltas = []

            for window in windows:
                values = window[:, tx, rx, :]
                packet_means = values.mean(axis=1)

                link_window_means.append(float(values.mean()))
                link_window_stds.append(float(values.std()))

                if len(packet_means) > 1:
                    link_window_deltas.append(
                        float(np.abs(np.diff(packet_means)).mean())
                    )

            links[link_name] = {
                "signal_mean": float(link_values.mean()),
                "signal_std": float(link_values.std()),
                "signal_p10": float(np.percentile(link_values, 10)),
                "signal_p50": float(np.percentile(link_values, 50)),
                "signal_p90": float(np.percentile(link_values, 90)),
                "window_mean_variability": float(
                    np.std(link_window_means)
                ),
                "temporal_delta_mean": float(
                    np.mean(link_window_deltas)
                ),
                "temporal_delta_std": float(
                    np.std(link_window_deltas)
                ),
            }

    for window in windows:
        window_means.append(float(window.mean()))
        window_stds.append(float(window.std()))

    return {
        "feature_version": FEATURE_VERSION,
        "window_size_packets": window_size,
        "num_windows": len(windows),
        "recording_summary": {
            "signal_mean": float(x.mean()),
            "signal_std": float(x.std()),
            "window_mean": summarize(window_means),
            "window_std": summarize(window_stds),
        },
        "links": links,
    }


def extract_manifest(manifest_path: Path, output_path: Path, limit: int | None):
    entries = [
        json.loads(line)
        for line in manifest_path.open()
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

            features = extract_recording_features(csi)

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


def main():
    parser = argparse.ArgumentParser(
        description="Extract recording-level WiMANS CSI features."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/wimans/manifest.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/wimans/features_v1.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    extract_manifest(args.manifest, args.output, args.limit)


if __name__ == "__main__":
    main()
