import os
import numpy as np
from scipy.io import loadmat

BASE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch"
)

# (batch, reference samples, test samples)
REGIONS = [
    (0, [20, 21, 22, 23, 24, 25, 26, 27], [30, 32, 34, 36, 37, 38, 39]),
    (1003, [0, 1, 2, 3, 4], list(range(5, 15))),
    (1007, [10, 12, 14, 16, 18], [20, 21, 22, 25, 29, 30, 40, 48, 49]),
]

def load_csi(batch):
    path = os.path.join(BASE, f"batch_{batch}.mat")
    mat = loadmat(path)
    return np.asarray(mat["csi_mon"])

def metrics(reference, sample):
    # One complex scalar fitted over all selected subcarriers
    denominator = np.vdot(reference, reference)

    if abs(denominator) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan

    scalar = np.vdot(reference, sample) / denominator
    residual = np.linalg.norm(sample - scalar * reference)
    residual /= np.linalg.norm(sample) + 1e-12

    # Amplitude profiles
    a = np.abs(reference)
    b = np.abs(sample)

    # Remove overall amplitude scale
    scale = np.dot(a, b) / (np.dot(a, a) + 1e-12)
    amp_residual = np.linalg.norm(b - scale * a)
    amp_residual /= np.linalg.norm(b) + 1e-12

    corr = np.corrcoef(a, b)[0, 1]

    return abs(scalar), np.angle(scalar, deg=True), residual, amp_residual

for batch, refs, tests in REGIONS:
    print(f"\n{'=' * 65}")
    print(f"BATCH {batch}")
    print(f"{'=' * 65}")

    csi = load_csi(batch)

    # Expected dimensions: samples x subcarriers
    if csi.ndim != 2:
        print("Unexpected CSI dimensions:", csi.shape)
        continue

    ref = np.median(np.abs(csi[refs]), axis=0)
    valid = ref > max(np.median(ref) * 0.02, 1e-8)

    # Use a representative reference CSI sample
    reference = np.mean(csi[refs], axis=0)[valid]

    print("Reference samples:", refs)
    print(
        "Sample | Gain | Phase(deg) | Complex residual | "
        "Amplitude residual"
    )

    for idx in tests:
        sample = csi[idx, valid]
        gain, phase, complex_res, amp_res = metrics(reference, sample)

        print(
            f"{idx:6d} | {gain:5.3f} | {phase:10.2f} | "
            f"{complex_res:16.4f} | {amp_res:18.4f}"
        )
