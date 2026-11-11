
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

FILE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/"
    "Data/fine_grained/Classroom/80MHz/3mo/m1/"
    "Slots/Test/A_batch/batch_0.mat"
)

H = sio.loadmat(FILE)["csi_mon"]
A = np.abs(H)

# Per-sample RMS
rms = np.sqrt(np.mean(A**2, axis=1))

# Relative gain between consecutive samples
gain = rms[1:] / (rms[:-1] + 1e-12)

# Estimate alternating gain pattern
print("\nSAMPLE RMS AND GAIN PATTERN")
print("=" * 60)

for i in range(len(rms)):
    print(f"Sample {i:02d}: RMS={rms[i]:.3f}")

print("\nCONSECUTIVE GAIN RATIOS")
for i, g in enumerate(gain):
    print(f"{i:02d}->{i+1:02d}: {g:.4f}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(rms, marker="o")
axes[0].set_title("CSI RMS Amplitude Across Samples")
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("RMS Amplitude")
axes[0].grid(True)

axes[1].plot(gain, marker="o")
axes[1].axhline(1, linestyle="--")
axes[1].set_title("Consecutive Sample Gain Ratios")
axes[1].set_xlabel("Transition")
axes[1].set_ylabel("Gain Ratio")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("plots/gain_pattern.png", dpi=200)
plt.show()
