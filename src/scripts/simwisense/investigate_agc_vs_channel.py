
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

PATH = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/"
    "Data/fine_grained/Classroom/80MHz/3mo/"
    "m1/Slots/Test/A_batch/batch_0.mat"
)

data = loadmat(PATH)
csi = np.asarray(data["csi_mon"])

if csi.ndim != 2:
    raise ValueError(f"Expected 2D CSI, got {csi.shape}")

if csi.shape[0] < 50:
    raise ValueError(f"Expected at least 50 samples, got {csi.shape}")

amp = np.abs(csi)

# Reference: normal samples
reference = np.median(amp[0:30], axis=0)

# Exclude invalid / near-zero subcarriers
threshold = max(np.median(reference) * 0.02, 1e-8)
valid = (
    np.isfinite(reference)
    & (reference > threshold)
)

ref = reference[valid]
eps = 1e-12

regions = {
    "Normal": range(0, 30),
    "Anomaly": range(30, 37),
    "Stable": range(37, 50),
}

amp_gain = []
complex_gain = []
amp_residual = []
complex_residual = []
phase_rotation = []

print("\nAGC VS CHANNEL INVESTIGATION")
print("=" * 65)
print("Valid subcarriers:", valid.sum(), "/", len(valid))

for name, indices in regions.items():

    results = []

    for i in indices:
        x = csi[i, valid]
        a = np.abs(x)

        # Amplitude-only gain
        g_amp = np.median(a / (ref + eps))

        # Amplitude-normalized profile residual
        normalized = a / (g_amp + eps)
        r_amp = (
            np.linalg.norm(normalized - ref)
            / (np.linalg.norm(ref) + eps)
        )

        # Complex scalar least-squares estimate
        reference_complex = csi[0, valid]
        alpha = (
            np.vdot(reference_complex, x)
            / (np.vdot(reference_complex, reference_complex) + eps)
        )

        reconstructed = alpha * reference_complex

        r_complex = (
            np.linalg.norm(x - reconstructed)
            / (np.linalg.norm(x) + eps)
        )

        results.append((
            g_amp,
            r_amp,
            abs(alpha),
            np.angle(alpha, deg=True),
            r_complex
        ))

    results = np.asarray(results)

    amp_gain.append(results[:, 0])
    amp_residual.append(results[:, 1])
    complex_gain.append(results[:, 2])
    phase_rotation.append(results[:, 3])
    complex_residual.append(results[:, 4])

    print(f"\n{name}")
    print("-" * 45)
    print("Amplitude gain:", np.median(results[:, 0]))
    print("Amplitude residual:", np.median(results[:, 1]))
    print("Complex gain:", np.median(results[:, 2]))
    print("Phase rotation (degrees):", np.median(results[:, 3]))
    print("Complex residual:", np.median(results[:, 4]))

# Per-subcarrier gain profiles
gain_anomaly = (
    np.median(amp[30:37], axis=0)[valid] / (ref + eps)
)

gain_stable = (
    np.median(amp[37:50], axis=0)[valid] / (ref + eps)
)

print("\nPER-SUBCARRIER SUMMARY")
print("Anomaly gain std:", np.std(gain_anomaly))
print("Stable gain std:", np.std(gain_stable))

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(13, 12))

axes[0].plot(np.arange(30, 37), amp_gain[1], "o-", label="Anomaly")
axes[0].plot(np.arange(37, 50), amp_gain[2], "o-", label="Stable")
axes[0].set_title("Estimated amplitude gain")
axes[0].set_xlabel("Sample")
axes[0].legend()
axes[0].grid()

axes[1].plot(np.arange(30, 37), amp_residual[1], "o-", label="Anomaly")
axes[1].plot(np.arange(37, 50), amp_residual[2], "o-", label="Stable")
axes[1].set_title("Amplitude residual after gain compensation")
axes[1].set_xlabel("Sample")
axes[1].legend()
axes[1].grid()

axes[2].plot(gain_anomaly, label="Anomaly")
axes[2].plot(gain_stable, label="Stable")
axes[2].set_title("Frequency-dependent gain")
axes[2].set_xlabel("Valid subcarrier index")
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.savefig("agc_vs_channel_analysis.png", dpi=150)
plt.show()
