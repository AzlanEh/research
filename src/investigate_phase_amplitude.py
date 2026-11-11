import scipy.io as sio
import numpy as np

mat_file = "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch/batch_0.mat"

data = sio.loadmat(mat_file)
csi = data["csi_mon"]

amp = np.abs(csi)
phase = np.angle(csi)

# Phase difference with wrapping
phase_diff = np.angle(
    np.exp(1j * np.diff(phase, axis=0))
)

# Amplitude difference
amp_diff = np.abs(np.diff(amp, axis=0))

# Complex CSI difference
complex_diff = np.abs(np.diff(csi, axis=0))

# Normal and anomaly intervals
normal = slice(0, 29)
anomaly = slice(29, 36)

def report(name, values):
    normal_mean = np.mean(values[normal])
    anomaly_mean = np.mean(values[anomaly])

    print(f"\n{name}")
    print(f"Normal:  {normal_mean:.4f}")
    print(f"Anomaly: {anomaly_mean:.4f}")
    print(f"Ratio:   {anomaly_mean/(normal_mean+1e-8):.4f}")

report("AMPLITUDE CHANGE", amp_diff)
report("PHASE CHANGE", np.abs(phase_diff))
report("COMPLEX CSI CHANGE", complex_diff)

# Inspect individual time transitions
print("\nTRANSITION ANALYSIS")
print("-" * 60)

for i in range(29, 36):
    print(
        f"{i} -> {i+1}: "
        f"Amplitude={np.mean(amp_diff[i]):.2f}, "
        f"Phase={np.mean(np.abs(phase_diff[i])):.4f}, "
        f"Complex={np.mean(complex_diff[i]):.2f}"
    )


from scipy.stats import pearsonr

print("\nAMPLITUDE PROFILE CORRELATION")

for i in range(29, 36):

    a = amp[i]
    b = amp[i+1]

    corr, _ = pearsonr(a, b)

    scale = np.median(b / (a + 1e-8))

    print(
        f"{i}->{i+1}: "
        f"Correlation={corr:.4f}, "
        f"Scale={scale:.4f}"
    )
