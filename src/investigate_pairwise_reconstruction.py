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

def fit_complex(a, b):
    denom = np.vdot(a, a)

    if abs(denom) < 1e-12:
        return np.nan

    alpha = np.vdot(a, b) / denom

    return (
        np.linalg.norm(b - alpha * a)
        / (np.linalg.norm(b) + 1e-12)
    )

def fit_amplitude(a, b):
    a = np.abs(a)
    b = np.abs(b)

    alpha = np.dot(a, b) / (np.dot(a, a) + 1e-12)

    return (
        np.linalg.norm(b - alpha * a)
        / (np.linalg.norm(b) + 1e-12)
    )

for batch, region in REGIONS.items():

    path = os.path.join(BASE, f"batch_{batch}.mat")
    csi = np.asarray(loadmat(path)["csi_mon"])

    if csi.shape[1] != 242 and csi.shape[0] == 242:
        csi = csi.T

    amp = np.abs(csi)

    reference_amp = np.median(
        amp[region["reference"]], axis=0
    )

    valid = reference_amp > max(
        np.median(reference_amp) * 0.02, 1e-8
    )

    print(f"\n{'=' * 65}")
    print(f"BATCH {batch}")
    print("=" * 65)

    for idx in region["test"]:

        complex_errors = []
        amplitude_errors = []

        for ref_idx in region["reference"]:

            a = csi[ref_idx, valid]
            b = csi[idx, valid]

            complex_errors.append(fit_complex(a, b))
            amplitude_errors.append(fit_amplitude(a, b))

        print(
            f"Sample {idx:02d} | "
            f"Best complex residual: {np.nanmin(complex_errors):.4f} | "
            f"Best amplitude residual: {np.nanmin(amplitude_errors):.4f}"
        )

print("\nPairwise investigation complete.")
