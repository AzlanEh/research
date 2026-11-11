"""plot_csi_variance_5.py

Generate 5 focused CSI variance/temporal analysis plots from SiMWiSense batch data.

Outputs (5 figures) in the ``plots/`` directory:

  1. per_sample_variance.png    - Variance across subcarriers per sample (bar chart)
  2. per_subcarrier_variance.png - Variance across samples per subcarrier (line plot)
  3. rolling_variance.png        - Rolling window variance over time
  4. variance_heatmap.png        - Local variance heatmap (sample x subcarrier)
  5. variance_ratio.png          - Variance ratio vs Normal region per subcarrier

All metrics computed from the raw complex CSI; no AGC correction applied.

Run from ``src/``:
    python plot_csi_variance_5.py
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


def region_color(idx: int) -> str:
    if idx < NORMAL_END:
        return "#2ca02c"   # green
    if idx < ANOMALY_END:
        return "#d62728"   # red
    return "#1f77b4"       # blue


def compute_metrics(csi: np.ndarray) -> dict:
    """Compute variance metrics from complex CSI."""
    amp = np.abs(csi)

    per_sample_variance = np.var(amp, axis=1)
    per_subcarrier_variance = np.var(amp, axis=0)

    # Rolling variance (window=5), centered
    window = 5
    rolling_var = []
    for i in range(TOTAL_SAMPLES - window + 1):
        rolling_var.append(np.var(amp[i : i + window, :]))
    rolling_var = np.asarray(rolling_var)
    rolling_var_padded = np.full(TOTAL_SAMPLES, np.nan)
    rolling_var_padded[window // 2 : -(window // 2)] = rolling_var

    # Local variance heatmap (5x5 sliding window)
    win = 5
    var_heatmap = np.zeros_like(amp)
    for i in range(amp.shape[0]):
        for j in range(amp.shape[1]):
            i0 = max(0, i - win // 2)
            i1 = min(amp.shape[0], i + win // 2 + 1)
            j0 = max(0, j - win // 2)
            j1 = min(amp.shape[1], j + win // 2 + 1)
            var_heatmap[i, j] = np.var(amp[i0:i1, j0:j1])

    # Variance ratios vs Normal region
    normal_var = np.var(amp[:NORMAL_END, :], axis=0)
    anomaly_var = np.var(amp[NORMAL_END:ANOMALY_END, :], axis=0)
    stable_var = np.var(amp[ANOMALY_END:, :], axis=0)
    eps = 1e-12
    ratio_anomaly = anomaly_var / (normal_var + eps)
    ratio_stable = stable_var / (normal_var + eps)

    return {
        "amplitude": amp,
        "per_sample_variance": per_sample_variance,
        "per_subcarrier_variance": per_subcarrier_variance,
        "rolling_variance": rolling_var_padded,
        "var_heatmap": var_heatmap,
        "ratio_anomaly": ratio_anomaly,
        "ratio_stable": ratio_stable,
    }


# ---------------------------------------------------------------------------
# Plot 1: Per-sample variance
# ---------------------------------------------------------------------------
def plot_per_sample_variance(metrics: dict, out_dir: Path) -> None:
    var = metrics["per_sample_variance"]
    indices = np.arange(TOTAL_SAMPLES)
    colors = [region_color(i) for i in indices]

    plt.figure(figsize=(14, 5))
    plt.bar(indices, var, color=colors, alpha=0.7)
    plt.axvline(NORMAL_END - 0.5, color="k", linestyle="--", alpha=0.5)
    plt.axvline(ANOMALY_END - 0.5, color="k", linestyle="--", alpha=0.5)
    plt.title("Per-Sample Variance (across 242 subcarriers)")
    plt.xlabel("Sample")
    plt.ylabel("Variance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "per_sample_variance.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Per-subcarrier variance
# ---------------------------------------------------------------------------
def plot_per_subcarrier_variance(metrics: dict, out_dir: Path) -> None:
    var = metrics["per_subcarrier_variance"]
    indices = np.arange(var.shape[0])

    plt.figure(figsize=(14, 5))
    plt.plot(indices, var, color="#1f77b4", linewidth=0.8)
    plt.title("Per-Subcarrier Variance (across 50 samples)")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Variance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "per_subcarrier_variance.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Rolling variance
# ---------------------------------------------------------------------------
def plot_rolling_variance(metrics: dict, out_dir: Path) -> None:
    var = metrics["rolling_variance"]
    indices = np.arange(TOTAL_SAMPLES)
    colors = [region_color(i) for i in indices]

    plt.figure(figsize=(14, 5))
    plt.bar(indices, var, color=colors, alpha=0.7)
    plt.axvline(NORMAL_END - 0.5, color="k", linestyle="--", alpha=0.5)
    plt.axvline(ANOMALY_END - 0.5, color="k", linestyle="--", alpha=0.5)
    plt.title("Rolling Variance (window=5 samples)")
    plt.xlabel("Sample")
    plt.ylabel("Rolling Variance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "rolling_variance.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Plot 4: Variance heatmap
# ---------------------------------------------------------------------------
def plot_variance_heatmap(metrics: dict, out_dir: Path) -> None:
    heatmap = metrics["var_heatmap"]

    plt.figure(figsize=(14, 6))
    im = plt.imshow(
        heatmap, aspect="auto", origin="lower", cmap="plasma"
    )
    plt.axvline(NORMAL_END - 0.5, color="w", linestyle="--", alpha=0.7)
    plt.axvline(ANOMALY_END - 0.5, color="w", linestyle="--", alpha=0.7)
    plt.colorbar(im, label="Local Variance (5x5 window)")
    plt.title("Local Variance Heatmap (sample x subcarrier)")
    plt.xlabel("Subcarrier")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.savefig(out_dir / "variance_heatmap.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Plot 5: Variance ratio
# ---------------------------------------------------------------------------
def plot_variance_ratio(metrics: dict, out_dir: Path) -> None:
    ratio_a = metrics["ratio_anomaly"]
    ratio_s = metrics["ratio_stable"]
    indices = np.arange(ratio_a.shape[0])

    plt.figure(figsize=(14, 5))
    plt.plot(indices, ratio_a, color="#d62728", linewidth=0.8, label="Anomaly / Normal")
    plt.plot(indices, ratio_s, color="#1f77b4", linewidth=0.8, label="Stable / Normal")
    plt.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    plt.title("Variance Ratio vs Normal Region (per subcarrier)")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Variance Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "variance_ratio.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 5 CSI variance analysis plots."
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

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot_csi_variance_5] Loading CSI from: {args.file}")
    csi = load_csi(args.file)
    print(f"[plot_csi_variance_5] CSI shape: {csi.shape}, dtype: {csi.dtype}")

    metrics = compute_metrics(csi)

    plot_per_sample_variance(metrics, out_dir)
    print("  -> per_sample_variance.png")

    plot_per_subcarrier_variance(metrics, out_dir)
    print("  -> per_subcarrier_variance.png")

    plot_rolling_variance(metrics, out_dir)
    print("  -> rolling_variance.png")

    plot_variance_heatmap(metrics, out_dir)
    print("  -> variance_heatmap.png")

    plot_variance_ratio(metrics, out_dir)
    print("  -> variance_ratio.png")

    print(f"\n[plot_csi_variance_5] All 5 figures written to {out_dir}/")


if __name__ == "__main__":
    main()
