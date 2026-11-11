import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

path = (
    "/mnt/external_disk/SiMWiSense/"
    "SiMWiSense/Data/fine_grained/Classroom/"
    "80MHz/3mo/m1/Slots/Test/A_batch/batch_0.mat"
)

data = sio.loadmat(path)

print("MAT file variables:")
for key, value in data.items():
    if not key.startswith("__"):
        print(key, value.shape, value.dtype)

# Replace with the actual CSI variable name
csi = data["csi_mon"]

amplitude = np.abs(csi)
phase = np.angle(csi)

# Row-wise statistics
mean = amplitude.mean(axis=1)
std = amplitude.std(axis=1)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

axes[0].plot(mean)
axes[0].set_title("Mean CSI Amplitude")
axes[0].grid()

axes[1].plot(std)
axes[1].set_title("CSI Amplitude Standard Deviation")
axes[1].grid()

axes[2].imshow(
    phase.T,
    aspect="auto",
    cmap="twilight",
    origin="lower"
)
axes[2].set_title("CSI Phase Heatmap")
axes[2].set_xlabel("Time Sample")
axes[2].set_ylabel("Subcarrier")

plt.tight_layout()
plt.savefig("plots/csi_anomaly_investigation.png", dpi=200)
plt.show()

# Detect unusually large amplitude changes
changes = np.abs(np.diff(mean))

print("\nLargest temporal changes:")
for i in np.argsort(changes)[-10:][::-1]:
    print(
        f"Between samples {i} and {i+1}: "
        f"{changes[i]:.2f}"
    )
