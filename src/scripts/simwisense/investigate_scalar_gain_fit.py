import numpy as np
from scipy.io import loadmat

DATA_PATH = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/"
    "Test/A_batch/batch_0.mat"
)

csi = np.asarray(loadmat(DATA_PATH)["csi_mon"])

if csi.ndim != 2:
    raise ValueError(f"Unexpected shape: {csi.shape}")

if csi.shape[1] == 242 and csi.shape[0] >= 50:
    pass
elif csi.shape[0] == 242 and csi.shape[1] >= 50:
    csi = csi.T
else:
    raise ValueError(f"Unexpected CSI shape: {csi.shape}")

amp = np.abs(csi[:50])

reference = np.median(amp[:30], axis=0)
valid = reference > max(np.median(reference) * 0.02, 1e-8)

profiles = amp[:, valid]

def fit_scalar(a, b):
    # Fit a ~= alpha * b using least squares.
    alpha = np.dot(a, b) / np.dot(b, b)

    residual = a - alpha * b
    relative_error = np.linalg.norm(residual) / np.linalg.norm(a)

    corr = np.corrcoef(a, b)[0, 1]

    return alpha, relative_error, corr

print("\nSCALAR GAIN FIT: ALTERNATING SAMPLES")
print("=" * 70)

pairs = [(30, 31), (32, 33), (34, 35), (36, 35)]

for high, low in pairs:
    alpha, error, corr = fit_scalar(
        profiles[high], profiles[low]
    )

    print(
        f"Sample {high} vs {low} | "
        f"Fitted gain: {alpha:.4f} | "
        f"Relative residual: {error:.4%} | "
        f"Correlation: {corr:.6f}"
    )

print("\nGROUP-LEVEL SCALAR FIT")
print("-" * 70)

high_profile = np.median(profiles[[30, 32, 34, 36]], axis=0)
low_profile = np.median(profiles[[31, 33, 35]], axis=0)

alpha, error, corr = fit_scalar(high_profile, low_profile)

print(f"Fitted high/low gain: {alpha:.4f}")
print(f"Relative residual: {error:.4%}")
print(f"Correlation: {corr:.6f}")

print("\nSTABLE REGION PROFILE EVOLUTION")
print("-" * 70)

for i in range(38, 50):
    alpha, error, corr = fit_scalar(
        profiles[i], profiles[37]
    )

    print(
        f"Sample 37 vs {i:02d} | "
        f"Fitted gain: {alpha:.4f} | "
        f"Relative residual: {error:.4%} | "
        f"Correlation: {corr:.6f}"
    )

print("\nCompleted.")
