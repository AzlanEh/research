import scipy.io as sio
import numpy as np

mat_file = "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch/batch_0.mat"

data = sio.loadmat(mat_file)
csi = data["csi_mon"]

amp = np.abs(csi)

# Calculate changes between consecutive samples
diff = np.abs(np.diff(amp, axis=0))

# Mean change per subcarrier
mean_change = np.mean(diff, axis=0)

# Change specifically around samples 29-36
anomaly_change = np.mean(diff[29:36], axis=0)

# Normal baseline
baseline_change = np.mean(
    np.concatenate([diff[:29], diff[36:]], axis=0),
    axis=0
)

ratio = anomaly_change / (baseline_change + 1e-8)

print("\nTOP 20 SUBCARRIERS WITH LARGEST ANOMALY CHANGE")
print("-" * 65)

indices = np.argsort(anomaly_change)[::-1][:20]

for i in indices:
    print(
        f"SC {i:3d}: "
        f"Anomaly={anomaly_change[i]:9.3f}, "
        f"Baseline={baseline_change[i]:9.3f}, "
        f"Ratio={ratio[i]:8.2f}"
    )

# Overall comparison
print("\nOVERALL CHANGE")
print("Anomaly mean:", np.mean(anomaly_change))
print("Baseline mean:", np.mean(baseline_change))

complex_diff = np.abs(np.diff(csi, axis=0))

print("\nCOMPLEX CSI CHANGE")
print("Normal:",
      np.mean(np.concatenate([
          complex_diff[:29],
          complex_diff[36:]
      ], axis=0)))

print("Anomaly:",
      np.mean(complex_diff[29:36]))
