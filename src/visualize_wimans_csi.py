"""Visualize one WiMANS CSI amplitude recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csi(path: Path) -> np.ndarray:
    """Load and validate a WiMANS CSI recording."""
    if not path.exists():
        raise FileNotFoundError(f"CSI file not found: {path}")

    csi = np.load(path, allow_pickle=False)

    if csi.ndim != 4 or csi.shape[1:] != (3, 3, 30):
        raise ValueError(
            f"Expected shape (T, 3, 3, 30), got {csi.shape}"
        )

    # Convert complex CSI to magnitude if necessary.
    if np.iscomplexobj(csi):
        csi = np.abs(csi)

    csi = csi.astype(np.float32, copy=False)

    if not np.isfinite(csi).all():
        raise ValueError("CSI contains NaN or infinite values.")

    return csi


def visualize(csi: np.ndarray, sample_id: str, output: Path,
              max_points: int = 1500) -> None:

    T = csi.shape[0]

    # Downsample only for plotting. Original data remains unchanged.
    indices = np.linspace(
        0, T - 1, min(T, max_points), dtype=int
    )
    data = csi[indices]

    # Common color scale for all nine links.
    vmin, vmax = np.percentile(csi, [1, 99])

    fig = plt.figure(figsize=(17, 13))
    fig.suptitle(
        f"WiMANS CSI Visualization — {sample_id}\n"
        f"Original shape: {csi.shape}",
        fontsize=17,
        fontweight="bold",
    )

    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[1, 1, 1, 0.9],
        hspace=0.4,
        wspace=0.3,
    )

    # 1. Nine link heatmaps.
    image = None

    for tx in range(3):
        for rx in range(3):
            ax = fig.add_subplot(gs[tx, rx])

            link = data[:, tx, rx, :]

            image = ax.imshow(
                link.T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(f"TX {tx} → RX {rx}")
            ax.set_xlabel("Time index")
            ax.set_ylabel("Subcarrier")

    fig.colorbar(
        image,
        ax=fig.axes[:9],
        label="CSI Amplitude",
        shrink=0.85,
    )

    # 2. Overall temporal amplitude.
    ax = fig.add_subplot(gs[3, :])

    mean_amplitude = np.mean(data, axis=(1, 2, 3))
    std_amplitude = np.std(data, axis=(1, 2, 3))

    ax.plot(
        indices,
        mean_amplitude,
        label="Mean amplitude",
        linewidth=1.2,
    )

    ax.fill_between(
        indices,
        mean_amplitude - std_amplitude,
        mean_amplitude + std_amplitude,
        alpha=0.2,
        label="±1 standard deviation",
    )

    ax.set_title("Overall CSI Amplitude Over Time")
    ax.set_xlabel("Packet / Time index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"\nVisualization saved: {output}")

    plt.show()


def print_statistics(csi: np.ndarray) -> None:
    print("\n" + "=" * 55)
    print("WIMANS CSI DATASET INSPECTION")
    print("=" * 55)

    print(f"Shape:       {csi.shape}")
    print(f"Data type:   {csi.dtype}")
    print(f"Time steps:  {csi.shape[0]}")
    print(f"TX antennas: {csi.shape[1]}")
    print(f"RX antennas: {csi.shape[2]}")
    print(f"Subcarriers: {csi.shape[3]}")

    print("\nAmplitude statistics:")
    print(f"Mean:        {np.mean(csi):.6f}")
    print(f"Std:         {np.std(csi):.6f}")
    print(f"Minimum:     {np.min(csi):.6f}")
    print(f"Maximum:     {np.max(csi):.6f}")
    print(f"Median:      {np.median(csi):.6f}")

    print("\nIntegrity:")
    print(f"NaN values:  {np.isnan(csi).sum()}")
    print(f"Inf values:  {np.isinf(csi).sum()}")

    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize WiMANS CSI amplitude recordings."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to a CSI .npy recording",
    )

    parser.add_argument(
        "--sample-id",
        default=None,
        help="Sample identifier (default: filename)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=1500,
        help="Maximum temporal points displayed",
    )

    args = parser.parse_args()

    if args.max_points < 2:
        parser.error("--max-points must be at least 2")

    csi = load_csi(args.input)

    sample_id = args.sample_id or args.input.stem

    output = args.output or args.input.with_name(
        f"{sample_id}_csi_visualization.png"
    )

    print_statistics(csi)
    visualize(csi, sample_id, output, args.max_points)


if __name__ == "__main__":
    main()
