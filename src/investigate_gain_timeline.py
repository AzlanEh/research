import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

DATA_PATH = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/"
    "Test/A_batch/batch_0.mat"
)

data = loadmat(DATA_PATH)
csi = np.asarray(data["csi_mon"])

if csi.ndim != 2:
    raise ValueError(f"Expected 2D CSI, got {csi.shape}")

# Ensure samples are rows and subcarriers are columns.
if csi.shape[1] == 242 and csi.shape[0] >= 50:
    pass
elif csi.shape[0] == 242 and csi.shape[1] >= 50:
    csi = csi.T
else:
    raise ValueError(f"Unexpected CSI shape: {csi.shape}")

csi = csi[:50, :]
amp = np.abs(csi)

# Use the median amplitude profile of normal samples as reference.
reference = np.median(amp[:30], axis=0)

valid = reference > max(np.median(reference) * 0.02, 1e-8)

gain = np.median(
    amp[:, valid] / reference[valid],
    axis=1
)

# Profile residual after removing the estimated scalar gain.
residual = []

for i in range(50):
    normalized = amp[i, valid] / max(gain[i], 1e-12)
    error = np.linalg.norm(normalized - reference[valid])
    denominator = np.linalg.norm(reference[valid])
    residual.append(error / denominator)

residual = np.asarray(residual)

print("\nPER-SAMPLE GAIN TIMELINE")
print("=" * 65)

for i in range(50):
    region = (
        "Normal" if i < 30
        else "Anomaly" if i < 37
        else "Stable"
    )
    print(
        f"Sample {i:02d} | {region:8s} | "
        f"Gain: {gain[i]:.4f} | "
        f"Residual: {residual[i]:.4f}"
    )

print("\nREGION GAIN SUMMARY")
print("-" * 65)

for name, start, end in [
    ("Normal", 0, 30),
    ("Anomaly", 30, 37),
    ("Stable", 37, 50),
]:
    values = gain[start:end]
    print(
        f"{name:8s}: median={np.median(values):.4f}, "
        f"std={np.std(values):.4f}, "
        f"min={np.min(values):.4f}, "
        f"max={np.max(values):.4f}"
    )

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

axes[0].plot(range(50), gain, marker="o", markersize=4)
axes[0].axhline(1.0, linestyle="--", label="Normal reference")
axes[0].axvline(30, linestyle=":", label="Region transition")
axes[0].axvline(37, linestyle=":")
axes[0].set_ylabel("Amplitude gain")
axes[0].set_title("Per-sample CSI amplitude gain")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(50), residual, marker="o", markersize=4)
axes[1].axvline(30, linestyle=":")
axes[1].axvline(37, linestyle=":")
axes[1].set_xlabel("Sample index")
axes[1].set_ylabel("Normalized profile residual")
axes[1].set_title("Residual after scalar gain normalization")
axes[1].grid(True, alpha=0.3)

for ax in axes:
    ax.axvspan(0, 29, alpha=0.08)
    ax.axvspan(30, 36, alpha=0.12)
    ax.axvspan(37, 49, alpha=0.08)

plt.tight_layout()
plt.savefig("gain_timeline_analysis.png", dpi=160)
print("\nSaved: gain_timeline_analysis.png")
plt.show()
