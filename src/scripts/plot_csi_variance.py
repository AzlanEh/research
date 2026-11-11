"""plot_csi_variance.py

Generate variance-focused CSI analysis plots from SiMWiSense batch data.

Produces variance metrics (per-sample, per-subcarrier, temporal, rolling)
from the underlying complex CSI data — not from the rendered PNG image.

Designed to run from ``src/``:

    python plot_csi_variance.py

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
DEFAULT_MAT_PATH = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense"
    "/Data/fine_grained/Classroom/80MHz/3mo/m1/"
    "Slots/Test/A_batch/batch_0.mat"
)

TOTAL_SAMPLES = 50
NORMAL_END = 30
ANOMALY_END = 37


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
    if csi.shape == (242, TOTAL_SAMPLES):
        csi = csi.T
    if csi.shape != (TOTAL_SAMPLES, 242):
        raise ValueError(
            f"Expected CSI shape (50, 242), got {csi.shape}"
        )
    return csi[:TOTAL_SAMPLES, :]


def region_of(idx: int) -> str:
    if idx < NORMAL_END:
        return "Normal"
    if idx < ANOMALY_END:
        return "Anomaly"
    return "Stable"


def region_color(idx: int) -> str:
    if idx < NORMAL_END:
        return "#2ca02c"   # green
    if idx < ANOMALY_END:
        return "#d62728"   # red
    return "#1f77b4"       # blue


def compute_variance_metrics(csi: np.ndarray) -> dict:
    """Compute all variance metrics from complex CSI.

    Returns a dict with:
      per_sample_variance   : (50,) variance of amplitude across subcarriers
      per_subcarrier_variance : (242,) variance of amplitude across samples
      temporal_variance     : variance of per-sample mean amplitude
      rolling_variance      : rolling window variance (window=5)
      amplitude             : raw amplitude matrix (50, 242)
    """
    amp = np.abs(csi)

    # Variance across subcarriers for each sample (per-sample variance)
    per_sample_variance = np.var(amp, axis=1)

    # Variance across samples for each subcarrier (per-subcarrier variance)
    per_subcarrier_variance = np.var(amp, axis=0)

    # Temporal variance of the mean amplitude trace
    mean_amp_trace = amp.mean(axis=1)
    temporal_variance = float(np.var(mean_amp_trace))

    # Rolling variance with window=5
    window = 5
    rolling_var = []
    for i in range(TOTAL_SAMPLES - window + 1):
        window_data = amp[i:i + window, :]
        rolling_var.append(np.var(window_data))
    rolling_var = np.asarray(rolling_var)
    # Pad to align with center of window
    rolling_var_padded = np.full(TOTAL_SAMPLES, np.nan)
    rolling_var_padded[window // 2 : -(window // 2)] = rolling_var

    return {
        "amplitude": amp,
        "per_sample_variance": per_sample_variance,
        "per_subcarrier_variance": per_subcarrier_variance,
        "temporal_variance": temporal_variance,
        "rolling_variance": rolling_var_padded,
        "mean_amp_trace": mean_amp_trace,
    }


def plot_variance_dashboard(csi: np.ndarray, out_dir: Path) -> None:
    """Generate a comprehensive variance dashboard."""
    metrics = compute_variance_metrics(csi)
    amp = metrics["amplitude"]
    per_sample_var = metrics["per_sample_variance"]
    per_subcarrier_var = metrics["per_subcarrier_variance"]
    rolling_var = metrics["rolling_variance"]
    mean_trace = metrics["mean_amp_trace"]

    sample_indices = np.arange(TOTAL_SAMPLES)
    colors = [region_color(i) for i in sample_indices]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "CSI Variance Analysis — Test Batch 0 (50 samples x 242 subcarriers)",
        fontsize=16,
    )

    # 1. Per-sample variance (variance across subcarriers per sample)
    axes[0, 0].bar(sample_indices, per_sample_var, color=colors, alpha=0.7)
    axes[0, 0].axvline(NORMAL_END - 0.5, color="k", linestyle="--", alpha=0.5)
    axes[0, 0].axvline(ANOMALY_END - 0.5, color="k", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Per-Sample Variance (across subcarriers)")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Variance")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Per-subcarrier variance (variance across samples per subcarrier)
    subcarrier_indices = np.arange(amp.shape[1])
    axes[0, 1].plot(subcarrier_indices, per_subcarrier_var, color="#1f77b4", linewidth=0.8)
    axes[0, 1].set_title("Per-Subcarrier Variance (across samples)")
    axes[0, 1].set_xlabel("Subcarrier Index")
    axes[0, 1].set_ylabel("Variance")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Variance heatmap (local variance in sample x subcarrier space)
    # Compute local variance using a 5x5 sliding window
    win = 5
    var_heatmap = np.zeros_like(amp)
    for i in range(amp.shape[0]):
        for j in range(amp.shape[1]):
            i0 = max(0, i - win // 2)
            i1 = min(amp.shape[0], i + win // 2 + 1)
            j0 = max(0, j - win // 2)
            j1 = min(amp.shape[1], j + win // 2 + 1)
            var_heatmap[i, j] = np.var(amp[i0:i1, j0:j1])
    im = axes[1, 0].imshow(
        var_heatmap, aspect="auto", origin="lower", cmap="plasma"
    )
    axes[1, 0].axvline(NORMAL_END - 0.5, color="w", linestyle="--", alpha=0.7)
    axes[1, 0].axvline(ANOMALY_END - 0.5, color="w", linestyle="--", alpha=0.7)
    axes[1, 0].set_title("Local Variance Heatmap (5x5 window)")
    axes[1, 0].set_xlabel("Subcarrier")
    axes[1, 0].set_ylabel("Sample")
    plt.colorbar(im, ax=axes[1, 0], label="Local Variance")

    # 4. Rolling variance over time
    axes[1, 1].bar(sample_indices, rolling_var, color=colors, alpha=0.7)
    axes[1, 1].axvline(NORMAL_END - 0.5, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].axvline(ANOMALY_END - 0.5, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Rolling Variance (window=5 samples)")
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].set_ylabel("Rolling Variance")
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Mean amplitude with variance bands (std as error band)
    std_per_sample = amp.std(axis=1)
    axes[2, 0].plot(sample_indices, mean_trace, color="#1f77b4", linewidth=2, label="Mean amplitude")
    axes[2, 0].fill_between(
        sample_indices,
        mean_trace - std_per_sample,
        mean_trace + std_per_sample,
        color="#1f77b4", alpha=0.3, label="±1 std dev"
    )
    axes[2, 0].axvline(NORMAL_END - 0.5, color="k", linestyle="--", alpha=0.5)
    axes[2, 0].axvline(ANOMALY_END - 0.5, color="k", linestyle="--", alpha=0.5)
    axes[2, 0].set_title("Mean Amplitude with Variance Bands")
    axes[2, 0].set_xlabel("Sample")
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Variance ratio: anomaly vs normal per subcarrier
    normal_var = np.var(amp[:NORMAL_END, :], axis=0)
    anomaly_var = np.var(amp[NORMAL_END:ANOMALY_END, :], axis=0)
    stable_var = np.var(amp[ANOMALY_END:, :], axis=0)
    eps = 1e-12
    ratio_anomaly_normal = anomaly_var / (normal_var + eps)
    ratio_stable_normal = stable_var / (normal_var + eps)

    axes[2, 1].plot(subcarrier_indices, ratio_anomaly_normal, color="#d62728", linewidth=0.8, label="Anomaly/Normal")
    axes[2, 1].plot(subcarrier_indices, ratio_stable_normal, color="#1f77b4", linewidth=0.8, label="Stable/Normal")
    axes[2, 1].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    axes[2, 1].set_title("Variance Ratio vs Normal Region")
    axes[2, 1].set_xlabel("Subcarrier Index")
    axes[2, 1].set_ylabel("Variance Ratio")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / "csi_variance_analysis.png", dpi=200)
    plt.close()

    # Print summary statistics
    print("\nCSI VARIANCE SUMMARY")
    print("=" * 55)
    print(f"Temporal variance (of mean amplitude trace): {metrics['temporal_variance']:.4f}")
    print()
    for name, start, end in [
        ("Normal", 0, NORMAL_END),
        ("Anomaly", NORMAL_END, ANOMALY_END),
        ("Stable", ANOMALY_END, TOTAL_SAMPLES),
    ]:
        region_var = per_sample_var[start:end]
        print(
            f"{name:8s}: mean_var={np.mean(region_var):.2f}, "
            f"median_var={np.median(region_var):.2f}, "
            f"max_var={np.max(region_var):.2f}, "
            f"min_var={np.min(region_var):.2f}, "
            f"std_var={np.std(region_var):.2f}"
        )
    print()
    print(f"Per-subcarrier variance: mean={np.mean(per_subcarrier_var):.2f}, "
          f"max={np.max(per_subcarrier_var):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate variance-focused CSI analysis plots."
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_MAT_PATH,
        help="Path to a batch_*.mat file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="plots",
        help="Output directory for plots (default: plots/)",
    )
    args = parser.parse_args()

    mat_path = args.file
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot_csi_variance] Loading CSI from: {mat_path}")
    csi = load_csi(mat_path)
    print(f"[plot_csi_variance] CSI shape: {csi.shape}, dtype: {csi.dtype}")
    print(f"[plot_csi_variance] Output directory: {out_dir}")

    plot_variance_dashboard(csi, out_dir)
    print(f"  -> csi_variance_analysis.png")
    print(f"\n[plot_csi_variance] Figure written to {out_dir}/csi_variance_analysis.png")


if __name__ == "__main__":
    main()
