import os
import numpy as np
from scipy.io import loadmat

BASE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch"
)

REGIONS = {
    0: {
        "reference": [20, 21, 22, 23, 24, 25, 26, 27],
        "test": [30, 31, 32, 33, 34, 35, 36, 37],
    },
    1003: {
        "reference": [0, 1, 2, 3, 4],
        "test": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    },
    1007: {
        "reference": [10, 12, 14, 16, 18],
        "test": [20, 21, 22, 25, 29, 30, 40, 48, 49],
    },
}


def metrics(a, b):
    aa = np.abs(a)
    bb = np.abs(b)

    valid = (
        (aa > max(np.median(aa) * 0.02, 1e-8))
        & (bb > max(np.median(bb) * 0.02, 1e-8))
    )

    if valid.sum() < 10:
        return None

    a = a[valid]
    b = b[valid]

    x = np.arange(len(valid))[valid]
    phase = np.angle(b * np.conj(a))

    weights = np.abs(a) * np.abs(b)
    weights /= weights.sum() + 1e-12

    raw = abs(np.sum(weights * np.exp(1j * phase)))

    slopes = np.linspace(-np.pi, np.pi, 4001)

    scores = np.array([
        abs(np.sum(
            weights * np.exp(1j * (phase - slope * x))
        ))
        for slope in slopes
    ])

    best = np.argmax(scores)
    slope = slopes[best]

    corrected = phase - slope * x
    coherence = abs(np.sum(
        weights * np.exp(1j * corrected)
    ))

    return np.degrees(slope), raw, coherence


for batch, region in REGIONS.items():
    path = os.path.join(BASE, f"batch_{batch}.mat")
    csi = np.asarray(loadmat(path)["csi_mon"])

    if csi.shape[1] != 242 and csi.shape[0] == 242:
        csi = csi.T

    print(f"\n{'=' * 75}")
    print(f"BATCH {batch}")
    print("=" * 75)

    for idx in region["test"]:
        print(f"\nTest sample {idx}")

        for ref_idx in region["reference"]:
            result = metrics(csi[ref_idx], csi[idx])

            if result is None:
                continue

            slope, raw, corrected = result

            print(
                f"  Ref {ref_idx:02d} | "
                f"Slope {slope:8.3f}° | "
                f"Raw {raw:.4f} | "
                f"Corrected {corrected:.4f}"
            )

print("\nReference stability investigation complete.")
