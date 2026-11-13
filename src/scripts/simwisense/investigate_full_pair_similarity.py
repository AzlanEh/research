import os
import numpy as np
from scipy.io import loadmat

BASE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch"
)

PAIRS = {
    0: [
        (33, 35), (30, 32), (31, 33),
        (33, 34), (35, 37), (20, 21), (24, 27)
    ],
    1003: [
        (5, 6), (5, 8), (7, 10),
        (10, 11), (13, 14), (0, 4)
    ],
    1007: [
        (20, 21), (22, 25), (22, 30),
        (25, 29), (40, 48), (48, 49)
    ],
}


def compare(a, b, valid):
    a = a[valid]
    b = b[valid]

    # Fit the best complex scalar: b ~= alpha * a
    alpha = np.vdot(a, b) / (np.vdot(a, a) + 1e-12)
    complex_residual = (
        np.linalg.norm(b - alpha * a) /
        (np.linalg.norm(b) + 1e-12)
    )

    # Fit the best amplitude-only gain.
    aa = np.abs(a)
    bb = np.abs(b)

    gain = np.dot(aa, bb) / (np.dot(aa, aa) + 1e-12)
    amplitude_residual = (
        np.linalg.norm(bb - gain * aa) /
        (np.linalg.norm(bb) + 1e-12)
    )

    # Phase coherence.
    phase = np.angle(b * np.conj(a))
    weights = np.abs(a) * np.abs(b)
    weights /= weights.sum() + 1e-12

    phase_coherence = abs(
        np.sum(weights * np.exp(1j * phase))
    )

    return (
        amplitude_residual,
        complex_residual,
        phase_coherence,
        abs(alpha),
        np.degrees(np.angle(alpha)),
    )


for batch, pairs in PAIRS.items():
    path = os.path.join(BASE, f"batch_{batch}.mat")
    csi = np.asarray(loadmat(path)["csi_mon"])

    if csi.shape[1] != 242 and csi.shape[0] == 242:
        csi = csi.T

    print(f"\n{'=' * 100}")
    print(f"BATCH {batch}")
    print("=" * 100)

    indices = sorted(set(i for pair in pairs for i in pair))
    selected = csi[indices]

    median_amp = np.median(np.abs(selected), axis=0)
    valid = median_amp > max(np.median(median_amp) * 0.02, 1e-8)

    print(
        f"{'Pair':>12} {'Amp Resid':>11} "
        f"{'Complex Resid':>15} {'Phase Coh':>11} "
        f"{'Gain':>10} {'Phase':>10}"
    )

    for i, j in pairs:
        amp, comp, coh, gain, phase = compare(
            csi[i], csi[j], valid
        )

        print(
            f"{i:5d} -> {j:<5d} "
            f"{amp:10.4f} "
            f"{comp:14.4f} "
            f"{coh:10.4f} "
            f"{gain:9.4f} "
            f"{phase:9.2f}°"
        )

print("\nFull pair similarity investigation complete.")
