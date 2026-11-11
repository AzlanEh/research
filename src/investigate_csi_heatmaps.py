
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

BASE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch"
)

OUTPUT_DIR = "csi_heatmaps"

BATCHES = {
    0: list(range(20, 28)) + list(range(30, 38)),
    1003: list(range(15)),
    1007: [10, 12, 14, 16, 18, 20, 21, 22, 25, 29, 30, 40, 48, 49],
}


def pair_metrics(a, b, valid):
    a = a[valid]
    b = b[valid]

    # Best complex scalar reconstruction.
    alpha = np.vdot(a, b) / (np.vdot(a, a) + 1e-12)

    residual_ab = (
        np.linalg.norm(b - alpha * a) /
        (np.linalg.norm(b) + 1e-12)
    )

    # Reverse direction for symmetric distance.
    beta = np.vdot(b, a) / (np.vdot(b, b) + 1e-12)

    residual_ba = (
        np.linalg.norm(a - beta * b) /
        (np.linalg.norm(a) + 1e-12)
    )

    complex_distance = (residual_ab + residual_ba) / 2

    # Amplitude-only residual.
    aa = np.abs(a)
    bb = np.abs(b)

    gain = np.dot(aa, bb) / (np.dot(aa, aa) + 1e-12)

    amp_residual = (
        np.linalg.norm(bb - gain * aa) /
        (np.linalg.norm(bb) + 1e-12)
    )

    # Weighted phase coherence.
    phase = np.angle(b * np.conj(a))
    weights = np.abs(a) * np.abs(b)
    weights /= weights.sum() + 1e-12

    coherence = abs(
        np.sum(weights * np.exp(1j * phase))
    )

    return complex_distance, amp_residual, coherence


def calculate_matrices(csi, indices):
    selected = csi[indices]

    # Same valid-subcarrier selection as the original script.
    median_amp = np.median(np.abs(selected), axis=0)

    valid = median_amp > max(
        np.median(median_amp) * 0.02,
        1e-8
    )

    n = len(indices)

    complex_matrix = np.zeros((n, n))
    amplitude_matrix = np.zeros((n, n))
    coherence_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d, a, c = pair_metrics(
                selected[i],
                selected[j],
                valid
            )

            complex_matrix[i, j] = complex_matrix[j, i] = d
            amplitude_matrix[i, j] = amplitude_matrix[j, i] = a
            coherence_matrix[i, j] = coherence_matrix[j, i] = c

    return complex_matrix, amplitude_matrix, coherence_matrix, valid.sum()


def save_heatmaps(batch, indices, matrices, valid_count):
    complex_matrix, amplitude_matrix, coherence_matrix = matrices

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels = [str(i) for i in indices]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(21, 7),
        constrained_layout=True
    )

    plots = [
        (
            complex_matrix,
            "Complex CSI Distance",
            "viridis",
            None,
            None
        ),
        (
            amplitude_matrix,
            "Amplitude Residual",
            "magma",
            0,
            None
        ),
        (
            coherence_matrix,
            "Phase Coherence",
            "cividis",
            0,
            1
        ),
    ]

    for ax, (matrix, title, cmap, vmin, vmax) in zip(axes, plots):
        image = ax.imshow(
            matrix,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sample index")

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))

        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

        # Annotate values inside cells.
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix[i, j]

                ax.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Batch {batch} | Pairwise CSI Similarity "
        f"({valid_count} valid subcarriers)",
        fontsize=16,
        fontweight="bold"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"csi_heatmap_batch_{batch}.png"
    )

    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Export numerical matrices.
    np.savetxt(
        os.path.join(OUTPUT_DIR, f"complex_batch_{batch}.csv"),
        complex_matrix,
        delimiter=",",
        fmt="%.6f"
    )

    np.savetxt(
        os.path.join(OUTPUT_DIR, f"amplitude_batch_{batch}.csv"),
        amplitude_matrix,
        delimiter=",",
        fmt="%.6f"
    )

    np.savetxt(
        os.path.join(OUTPUT_DIR, f"coherence_batch_{batch}.csv"),
        coherence_matrix,
        delimiter=",",
        fmt="%.6f"
    )

    print(f"\nBatch {batch}")
    print(f"Valid subcarriers: {valid_count}")
    print(f"Saved heatmap: {output_path}")


def main():
    for batch, indices in BATCHES.items():
        path = os.path.join(BASE, f"batch_{batch}.mat")

        print(f"\nLoading {path}")

        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            continue

        data = loadmat(path)

        if "csi_mon" not in data:
            print("ERROR: csi_mon not found in MAT file")
            continue

        csi = np.asarray(data["csi_mon"])

        if csi.shape[1] != 242 and csi.shape[0] == 242:
            csi = csi.T

        if max(indices) >= csi.shape[0]:
            print(
                f"ERROR: Requested sample {max(indices)}, "
                f"but batch contains only {csi.shape[0]} samples"
            )
            continue

        print(f"CSI shape: {csi.shape}")

        complex_matrix, amplitude_matrix, coherence_matrix, valid_count = (
            calculate_matrices(csi, indices)
        )

        save_heatmaps(
            batch,
            indices,
            (
                complex_matrix,
                amplitude_matrix,
                coherence_matrix
            ),
            valid_count
        )

    print("\nCSI heatmap investigation complete.")


if __name__ == "__main__":
    main()
