
import numpy as np
from scipy.io import loadmat

path = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/"
    "Data/fine_grained/Classroom/80MHz/3mo/"
    "m1/Slots/Test/A_batch/batch_0.mat"
)

data = loadmat(path)
csi = data["csi_mon"]

amp = np.abs(csi)

# Compare normal samples with abnormal samples
normal = amp[0:30]
anomaly = amp[30:37]
stable_after = amp[37:50]

normal_mean = np.mean(normal, axis=0)
anomaly_mean = np.mean(anomaly, axis=0)
stable_mean = np.mean(stable_after, axis=0)

# Robust per-subcarrier gain estimation
eps = 1e-8

gain_anomaly = anomaly_mean / (normal_mean + eps)
gain_stable = stable_mean / (normal_mean + eps)

print("\nPER-SUBCARRIER GAIN ANALYSIS")
print("=" * 65)

print("Subcarrier  Normal  Anomaly  Gain  Stable  Stable Gain")

for i in range(csi.shape[1]):
    print(
        f"{i:5d} "
        f"{normal_mean[i]:8.2f} "
        f"{anomaly_mean[i]:8.2f} "
        f"{gain_anomaly[i]:7.3f} "
        f"{stable_mean[i]:8.2f} "
        f"{gain_stable[i]:7.3f}"
    )

print("\nSUMMARY")
print("Mean anomaly gain:", np.median(gain_anomaly))
print("Mean stable gain:", np.median(gain_stable))
print("Anomaly gain std:", np.std(gain_anomaly))
print("Stable gain std:", np.std(gain_stable))
