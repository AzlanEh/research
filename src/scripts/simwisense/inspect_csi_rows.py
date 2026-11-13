import numpy as np
from scipy.io import loadmat

FILE = "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch/batch_0.mat"

mat = loadmat(FILE)
csi = np.asarray(mat["csi_mon"]).squeeze()

if csi.shape == (242, 50):
    csi = csi.T

amp = np.abs(csi)

print("\nROW-WISE CSI ANALYSIS")
print("-" * 75)
print(f"{'Row':<8}{'Mean':<15}{'Std':<15}{'Min':<15}{'Max':<15}")

for i in range(len(amp)):
    print(
        f"{i:<8}"
        f"{amp[i].mean():<15.3f}"
        f"{amp[i].std():<15.3f}"
        f"{amp[i].min():<15.3f}"
        f"{amp[i].max():<15.3f}"
    )

print("\nSUBCARRIER ANALYSIS")
print("-" * 50)

for i in [10, 60, 120, 180, 230]:
    print(
        f"Subcarrier {i}: "
        f"mean={amp[:, i].mean():.3f}, "
        f"std={amp[:, i].std():.3f}, "
        f"min={amp[:, i].min():.3f}, "
        f"max={amp[:, i].max():.3f}"
    )
