
import numpy as np
import scipy.io as sio

FILE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/"
    "Data/fine_grained/Classroom/80MHz/3mo/m1/"
    "Slots/Test/A_batch/batch_0.mat"
)

H = sio.loadmat(FILE)["csi_mon"]
A = np.abs(H)

print("\nGAIN CONSISTENCY ANALYSIS")
print("=" * 65)

for i in range(29, 36):

    x = A[i]
    y = A[i + 1]

    # Best multiplicative scaling factor
    scale = np.dot(x, y) / (np.dot(x, x) + 1e-12)

    # Residual after scaling
    residual = y - scale * x

    relative_error = (
        np.linalg.norm(residual) /
        (np.linalg.norm(y) + 1e-12)
    )

    # Per-subcarrier gain ratio
    ratios = y / (x + 1e-8)

    print(f"\nTransition {i} -> {i+1}")
    print(f"Gain factor:       {scale:.6f}")
    print(f"Mean ratio:        {np.mean(ratios):.6f}")
    print(f"Median ratio:      {np.median(ratios):.6f}")
    print(f"Ratio std:         {np.std(ratios):.6f}")
    print(f"Relative error:    {relative_error:.6f}")

print("\nALTERNATING GAIN ANALYSIS")
print("=" * 65)

for group, indices in [
    ("HIGH", [30, 32, 34, 36]),
    ("LOW", [31, 33, 35])
]:
    profile = np.mean(A[indices], axis=0)
    print(
        f"{group}: Mean amplitude = "
        f"{np.mean(profile):.3f}"
    )

print("\nANALYSIS COMPLETE")
