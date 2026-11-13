import os
import numpy as np
from scipy.io import loadmat

BASE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch"
)

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

    # Reverse direction for a symmetric distance.
    beta = np.vdot(b, a) / (np.vdot(b, b) + 1e-12)

    residual_ba = (
        np.linalg.norm(a - beta * b) /
        (np.linalg.norm(a) + 1e-12)
    )

    # Symmetric complex residual.
    distance = (residual_ab + residual_ba) / 2

    # Amplitude-only residual.
    aa = np.abs(a)
    bb = np.abs(b)

    gain = np.dot(aa, bb) / (np.dot(aa, aa) + 1e-12)

    amp_residual = (
        np.linalg.norm(bb - gain * aa) /
        (np.linalg.norm(bb) + 1e-12)
    )

    phase = np.angle(b * np.conj(a))
    weights = np.abs(a) * np.abs(b)
    weights /= weights.sum() + 1e-12

    coherence = abs(
        np.sum(weights * np.exp(1j * phase))
    )

    return distance, amp_residual, coherence


for batch, indices in BATCHES.items():
    path = os.path.join(BASE, f"batch_{batch}.mat")
    csi = np.asarray(loadmat(path)["csi_mon"])

    if csi.shape[1] != 242 and csi.shape[0] == 242:
        csi = csi.T

    selected = csi[indices]

    median_amp = np.median(np.abs(selected), axis=0)
    valid = median_amp > max(np.median(median_amp) * 0.02, 1e-8)

    n = len(indices)

    complex_matrix = np.zeros((n, n))
    amplitude_matrix = np.zeros((n, n))
    coherence_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d, a, c = pair_metrics(
                selected[i], selected[j], valid
            )

            complex_matrix[i, j] = complex_matrix[j, i] = d
            amplitude_matrix[i, j] = amplitude_matrix[j, i] = a
            coherence_matrix[i, j] = coherence_matrix[j, i] = c

    print("\n" + "=" * 75)
    print(f"BATCH {batch}: PAIRWISE CSI SIMILARITY")
    print("=" * 75)

    print("\nClosest complex-CSI pairs:")

    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((
                complex_matrix[i, j],
                indices[i],
                indices[j],
                amplitude_matrix[i, j],
                coherence_matrix[i, j],
            ))

    pairs.sort()

    print(
        f"{'Pair':>12} {'Complex Dist':>14} "
        f"{'Amp Resid':>11} {'Phase Coh':>11}"
    )

    for d, i, j, a, c in pairs[:15]:
        print(
            f"{i:5d} -> {j:<5d} "
            f"{d:13.4f} {a:10.4f} {c:10.4f}"
        )

    print("\nPotential complex-CSI matches (distance < 0.05):")

    found = False

    for d, i, j, a, c in pairs:
        if d < 0.05:
            print(
                f"{i} <-> {j} | "
                f"Complex={d:.4f} | "
                f"Amplitude={a:.4f} | "
                f"Coherence={c:.4f}"
            )
            found = True

    if not found:
        print("No pairs below threshold.")

print("\nCSI clustering investigation complete.")
