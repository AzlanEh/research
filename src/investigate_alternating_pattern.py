import numpy as np
from scipy.io import loadmat

DATA_PATH = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/"
    "Test/A_batch/batch_0.mat"
)

csi = np.asarray(loadmat(DATA_PATH)["csi_mon"])

if csi.ndim != 2:
    raise ValueError(f"Unexpected CSI dimensions: {csi.shape}")

if csi.shape[1] == 242 and csi.shape[0] >= 50:
    pass
elif csi.shape[0] == 242 and csi.shape[1] >= 50:
    csi = csi.T
else:
    raise ValueError(f"Unexpected CSI shape: {csi.shape}")

amp = np.abs(csi[:50])

reference = np.median(amp[:30], axis=0)
valid = reference > max(np.median(reference) * 0.02, 1e-8)

profiles = amp[:, valid] / reference[valid]

gain = np.median(profiles, axis=1)
normalized = profiles / gain[:, None]

def correlation(a, b):
    return np.corrcoef(a, b)[0, 1]

print("\nALTERNATING SAMPLE INVESTIGATION")
print("=" * 70)

print("\n1. ADJACENT SAMPLE COMPARISON")
print("-" * 70)

for i in range(30, 36):
    j = i + 1

    raw_corr = correlation(profiles[i], profiles[j])
    shape_corr = correlation(normalized[i], normalized[j])
    gain_ratio = gain[i] / gain[j]

    print(
        f"{i:02d} -> {j:02d} | "
        f"Gain ratio: {gain_ratio:.4f} | "
        f"Raw correlation: {raw_corr:.6f} | "
        f"Normalized correlation: {shape_corr:.6f}"
    )

print("\n2. EVEN-HIGH VS ODD-LOW COMPARISON")
print("-" * 70)

high = [30, 32, 34, 36]
low = [31, 33, 35]

print("High samples:", high)
print("Low samples:", low)

print(f"High median gain: {np.median(gain[high]):.4f}")
print(f"Low median gain:  {np.median(gain[low]):.4f}")

high_profile = np.median(normalized[high], axis=0)
low_profile = np.median(normalized[low], axis=0)

print(
    "High/low normalized profile correlation:",
    f"{correlation(high_profile, low_profile):.6f}"
)

ratio = profiles[high[0]] / np.maximum(profiles[low[0]], 1e-12)

print("\nFirst high/low pair per-subcarrier ratio:")
print(f"Median: {np.median(ratio):.4f}")
print(f"Std:    {np.std(ratio):.4f}")
print(f"Min:    {np.min(ratio):.4f}")
print(f"Max:    {np.max(ratio):.4f}")

print("\n3. STABLE REGION COMPARISON")
print("-" * 70)

for i in range(37, 49):
    print(
        f"{i:02d} -> {i+1:02d} | "
        f"Gain ratio: {gain[i] / gain[i+1]:.4f} | "
        f"Shape correlation: "
        f"{correlation(normalized[i], normalized[i+1]):.6f}"
    )

print("\nInvestigation complete.")
