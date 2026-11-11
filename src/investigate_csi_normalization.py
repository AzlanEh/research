
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

FILE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/"
    "Data/fine_grained/Classroom/80MHz/3mo/m1/"
    "Slots/Test/A_batch/batch_0.mat"
)

data = sio.loadmat(FILE)
H = data["csi_mon"]

A = np.abs(H)
P = np.angle(H)

# RMS normalization per sample
rms = np.sqrt(np.mean(A**2, axis=1, keepdims=True))
A_norm = A / (rms + 1e-12)

# Adjacent subcarrier phase difference
phase_diff = np.angle(H[:, 1:] * np.conj(H[:, :-1]))

# Compare amplitude profiles
print("\nNORMALIZATION ANALYSIS")
for i in range(30, 37):
    print(
        f"Sample {i}: "
        f"Original RMS={rms[i, 0]:.3f}, "
        f"Normalized RMS={np.sqrt(np.mean(A_norm[i]**2)):.3f}"
    )

print("\nNORMALIZED PROFILE CORRELATION")
for i in range(29, 36):
    corr = np.corrcoef(A_norm[i], A_norm[i+1])[0, 1]
    print(f"{i}->{i+1}: {corr:.6f}")

# Plot amplitude before and after normalization
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

for i in range(25, 41):
    axes[0].plot(A[i], alpha=0.6, label=str(i))
    axes[1].plot(A_norm[i], alpha=0.6, label=str(i))

axes[0].set_title("Original CSI Amplitude")
axes[1].set_title("RMS-Normalized CSI Amplitude")

for ax in axes:
    ax.set_xlabel("Subcarrier")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/normalization_comparison.png", dpi=200)
plt.show()

# Phase difference visualization
plt.figure(figsize=(14, 5))
plt.imshow(
    phase_diff[25:41],
    aspect="auto",
    cmap="twilight",
    interpolation="nearest"
)
plt.colorbar(label="Phase difference (radians)")
plt.xlabel("Subcarrier")
plt.ylabel("Sample")
plt.title("Adjacent Subcarrier Phase Differences")
plt.savefig("plots/phase_difference.png", dpi=200)
plt.show()
