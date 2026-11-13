import os
import numpy as np
from scipy.io import loadmat

BASE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch"
)

BATCHES = {
    0: [20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37],
    1003: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    1007: [10, 12, 14, 16, 18, 20, 21, 22, 25, 29, 30, 40, 48, 49],
}


def compare(a, b, valid):
    x = np.arange(len(a))[valid]

    aa = a[valid]
    bb = b[valid]

    phase = np.angle(bb * np.conj(aa))

    weights = np.abs(aa) * np.abs(bb)
    weights /= weights.sum() + 1e-12

    raw = abs(np.sum(weights * np.exp(1j * phase)))

    # Search for the best linear phase slope.
    slopes = np.linspace(-np.pi, np.pi, 20001)

    scores = np.array([
        abs(np.sum(
            weights * np.exp(1j * (phase - slope * x))
        ))
        for slope in slopes
    ])

    best = np.argmax(scores)
    slope = slopes[best]

    corrected = phase - slope * x
    coherence = abs(np.sum(weights * np.exp(1j * corrected)))

    return np.degrees(slope), raw, coherence


for batch, indices in BATCHES.items():
    path = os.path.join(BASE, f"batch_{batch}.mat")
    csi = np.asarray(loadmat(path)["csi_mon"])

    if csi.shape[1] != 242 and csi.shape[0] == 242:
        csi = csi.T

    # Use one common subcarrier mask for this batch.
    selected = csi[indices]
    median_amp = np.median(np.abs(selected), axis=0)

    valid = median_amp > max(np.median(median_amp) * 0.02, 1e-8)

    print(f"\n{'=' * 80}")
    print(f"BATCH {batch}: DIRECT PAIR COMPARISON")
    print("=" * 80)

    pairs = []

    if batch == 0:
        pairs = [
            (33, 35),
            (30, 32),
            (31, 33),
            (33, 34),
            (35, 37),
            (20, 21),
            (24, 27),
        ]
    elif batch == 1003:
        pairs = [
            (5, 6),
            (5, 8),
            (7, 10),
            (10, 11),
            (13, 14),
            (0, 4),
        ]
    else:
        pairs = [
            (20, 21),
            (22, 25),
            (22, 30),
            (25, 29),
            (40, 48),
            (48, 49),
        ]

    print(
        f"{'Pair':>12} {'Slope':>12} "
        f"{'Raw Coh':>10} {'Corrected':>12}"
    )

    for i, j in pairs:
        slope, raw, corrected = compare(
            csi[i], csi[j], valid
        )

        print(
            f"{i:5d} -> {j:<5d} "
            f"{slope:11.4f}° "
            f"{raw:10.4f} "
            f"{corrected:12.4f}"
        )

print("\nDirect phase-pair investigation complete.")
