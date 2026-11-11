import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description="Visualize a SiMWiSense CSI batch")
    parser.add_argument(
        "--file",
        default="/mnt/external_disk/SiMWiSense/SiMWiSense/Data/fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch/batch_0.mat"
    )
    parser.add_argument("--output", default="plots")
    args = parser.parse_args()

    file_path = Path(args.file)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    mat = loadmat(file_path)

    if "csi_mon" not in mat:
        raise KeyError(f"'csi_mon' not found. Available keys: {list(mat.keys())}")

    csi = np.asarray(mat["csi_mon"]).squeeze()

    if csi.ndim != 2:
        raise ValueError(f"Expected a 2D CSI matrix, got {csi.shape}")

    if csi.shape == (242, 50):
        csi = csi.T

    if csi.shape != (50, 242):
        raise ValueError(f"Expected shape (50, 242), got {csi.shape}")

    amplitude = np.abs(csi)
    mean_amplitude = amplitude.mean(axis=1)
    std_amplitude = amplitude.std(axis=1)

    print(f"File: {file_path}")
    print(f"CSI shape: {csi.shape}")
    print(f"Data type: {csi.dtype}")
    print(f"Finite values: {np.isfinite(csi).all()}")
    print(f"Amplitude range: {amplitude.min():.3f} to {amplitude.max():.3f}")
    print(f"Mean amplitude: {amplitude.mean():.3f}")
    print(f"Mean temporal variation: {std_amplitude.mean():.3f}")

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(f"SiMWiSense CSI Analysis — {file_path.parent.parent.name}", fontsize=15)

    # 1. CSI amplitude heatmap
    image = axes[0, 0].imshow(
        amplitude,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )
    axes[0, 0].set_title("CSI Amplitude Heatmap")
    axes[0, 0].set_xlabel("Subcarrier")
    axes[0, 0].set_ylabel("Time sample")
    fig.colorbar(image, ax=axes[0, 0], label="Amplitude")

    # 2. Mean amplitude over subcarriers
    axes[0, 1].plot(mean_amplitude)
    axes[0, 1].set_title("Mean Amplitude Over Time")
    axes[0, 1].set_xlabel("Time sample")
    axes[0, 1].set_ylabel("Mean |CSI|")
    axes[0, 1].grid(True)

    # 3. Amplitude variation over subcarriers
    axes[1, 0].plot(std_amplitude)
    axes[1, 0].set_title("Amplitude Standard Deviation Over Time")
    axes[1, 0].set_xlabel("Time sample")
    axes[1, 0].set_ylabel("Std |CSI|")
    axes[1, 0].grid(True)

    # 4. Selected subcarrier amplitude traces
    selected = [10, 60, 120, 180, 230]
    for subcarrier in selected:
        axes[1, 1].plot(
            amplitude[:, subcarrier],
            label=f"SC {subcarrier}"
        )
    axes[1, 1].set_title("Selected Subcarrier Amplitudes")
    axes[1, 1].set_xlabel("Time sample")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # 5. Real component
    axes[2, 0].plot(csi[:, 120].real)
    axes[2, 0].set_title("Real Component — Subcarrier 120")
    axes[2, 0].set_xlabel("Time sample")
    axes[2, 0].set_ylabel("Real(CSI)")
    axes[2, 0].grid(True)

    # 6. Imaginary component
    axes[2, 1].plot(csi[:, 120].imag)
    axes[2, 1].set_title("Imaginary Component — Subcarrier 120")
    axes[2, 1].set_xlabel("Time sample")
    axes[2, 1].set_ylabel("Imag(CSI)")
    axes[2, 1].grid(True)

    fig.tight_layout()

    output_file = output_dir / f"{file_path.parent.parent.name}_batch_0.png"
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    print(f"\nSaved visualization: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
