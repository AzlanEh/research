"""plot_csi_analysis.py

Generate CSI analysis overview plots for multiple SiMWiSense test batch
activities.

Produces one 6-panel overview figure per activity in the ``plots/``
directory:

  * Test_batch_A.png
  * Test_batch_B.png
  * Test_batch_C.png
  * Test_batch_D.png
  * Test_batch_E.png

Each figure contains:
  1. CSI Amplitude Heatmap
  2. CSI Phase Heatmap
  3. Mean Amplitude per Sample
  4. Amplitude Std Dev per Sample
  5. Mean Amplitude per Subcarrier
  6. Phase Std Dev per Subcarrier

All figures are derived from raw complex CSI windows
(50 samples x 242 subcarriers). No AGC correction or normalization is
applied to the stored data.

Designed to run from ``src/``:

    python plot_csi_analysis.py

By default, loads batch_0.mat from Test/A_batch ... Test/E_batch.
Use --activities or --files to override.

Read-only with respect to the source dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense"
    "/Data/fine_grained/Classroom/80MHz/3mo/m1"
)

DEFAULT_ACTIVITIES = ["A", "B", "C", "D", "E"]

# Region boundaries used throughout the investigation scripts.
NORMAL_END = 30
ANOMALY_END = 37
TOTAL_SAMPLES = 50


def csi_path(dataset_root: str, activity: str, batch: int = 0) -> str:
    """Build the path to a test batch .mat file."""
    return (
        f"{dataset_root}/Slots/Test/{activity}_batch/batch_{batch}.mat"
    )


def load_csi(mat_path: str) -> np.ndarray:
    """Load CSI from a .mat file and return a (50, 242) complex array."""
    mat = loadmat(mat_path)
    if "csi_mon" not in mat:
        raise KeyError(
            f"'csi_mon' not found in {mat_path}. "
            f"Available keys: {list(mat.keys())}"
        )
    csi = np.asarray(mat["csi_mon"]).squeeze()
    if csi.ndim != 2:
        raise ValueError(f"Expected 2D CSI, got shape {csi.shape}")
    # Ensure samples are rows, subcarriers are columns.
    if csi.shape == (242, TOTAL_SAMPLES):
        csi = csi.T
    if csi.shape != (TOTAL_SAMPLES, 242):
        raise ValueError(
            f"Expected CSI shape (50, 242), got {csi.shape}"
        )
    return csi[:TOTAL_SAMPLES, :]


def region_color(idx: int) -> str:
    """Return a color for a sample index based on its region."""
    if idx < NORMAL_END:
        return "#2ca02c"   # green
    if idx < ANOMALY_END:
        return "#d62728"   # red
    return "#1f77b4"       # blue


def plot_test_batch_overview(csi: np.ndarray, activity: str, out_dir: Path) -> None:
    """Generate a 6-panel CSI overview figure for one activity batch."""
    amp = np.abs(csi)
    phase = np.angle(csi)

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(
        f"SiMWiSense CSI — Test Batch {activity} Overview "
        f"(50 samples x 242 subcarriers)",
        fontsize=15,
    )

    # 1. CSI amplitude heatmap
    im0 = axes[0, 0].imshow(
        amp, aspect="auto", origin="lower", cmap="viridis"
    )
    axes[0, 0].set_title("CSI Amplitude Heatmap")
    axes[0, 0].set_xlabel("Subcarrier")
    axes[0, 0].set_ylabel("Sample")
    plt.colorbar(im0, ax=axes[0, 0], label="Amplitude")

    # 2. CSI phase heatmap
    im1 = axes[0, 1].imshow(
        phase, aspect="auto", origin="lower", cmap="twilight"
    )
    axes[0, 1].set_title("CSI Phase Heatmap")
    axes[0, 1].set_xlabel("Subcarrier")
    axes[0, 1].set_ylabel("Sample")
    plt.colorbar(im1, ax=axes[0, 1], label="Phase (rad)")

    # 3. Mean amplitude per sample
    mean_amp = amp.mean(axis=1)
    colors = [region_color(i) for i in range(TOTAL_SAMPLES)]
    axes[1, 0].bar(range(TOTAL_SAMPLES), mean_amp, color=colors, alpha=0.7)
    axes[1, 0].axvline(NORMAL_END - 0.5, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].axvline(ANOMALY_END - 0.5, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("Mean Amplitude per Sample")
    axes[1, 0].set_xlabel("Sample")
    axes[1, 0].set_ylabel("Mean Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Amplitude std dev per sample
    std_amp = amp.std(axis=1)
    axes[1, 1].bar(range(TOTAL_SAMPLES), std_amp, color=colors, alpha=0.7)
    axes[1, 1].axvline(NORMAL_END - 0.5, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].axvline(ANOMALY_END - 0.5, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Amplitude Std Dev per Sample")
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].set_ylabel("Std Dev")
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Per-subcarrier mean amplitude
    subcarrier_mean = amp.mean(axis=0)
    axes[2, 0].plot(subcarrier_mean, color="#1f77b4", linewidth=0.8)
    axes[2, 0].set_title("Mean Amplitude per Subcarrier")
    axes[2, 0].set_xlabel("Subcarrier Index")
    axes[2, 0].set_ylabel("Mean Amplitude")
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Per-subcarrier phase std
    subcarrier_phase_std = phase.std(axis=0)
    axes[2, 1].plot(subcarrier_phase_std, color="#d62728", linewidth=0.8)
    axes[2, 1].set_title("Phase Std Dev per Subcarrier")
    axes[2, 1].set_xlabel("Subcarrier Index")
    axes[2, 1].set_ylabel("Phase Std Dev (rad)")
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / f"Test_batch_{activity}.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CSI overview plots for multiple test batch activities."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ROOT,
        help="Path to the SiMWiSense monitor root (default: %(default)s)",
    )
    parser.add_argument(
        "--activities",
        nargs="+",
        default=DEFAULT_ACTIVITIES,
        help="Activity letters to plot (default: A B C D E)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Batch number to load for each activity (default: 0)",
    )
    parser.add_argument(
        "--output",
        default="plots",
        help="Output directory for plots (default: plots/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    activities = args.activities[:5]  # limit to 5 images

    print(f"[plot_csi_analysis] Dataset root: {args.dataset}")
    print(f"[plot_csi_analysis] Activities: {activities}")
    print(f"[plot_csi_analysis] Batch number: {args.batch}")
    print(f"[plot_csi_analysis] Output directory: {out_dir}")

    for activity in activities:
        mat_path = csi_path(args.dataset, activity, args.batch)
        try:
            csi = load_csi(mat_path)
        except FileNotFoundError as exc:
            print(f"  SKIP {activity}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {activity}: {exc}")
            continue

        print(f"  -> Test_batch_{activity}.png (shape={csi.shape})")
        plot_test_batch_overview(csi, activity, out_dir)

    print(f"\n[plot_csi_analysis] All figures written to {out_dir}/")


if __name__ == "__main__":
    main()
