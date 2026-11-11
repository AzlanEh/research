import numpy as np
from scipy.io import loadmat

PATH = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/"
    "Test/A_batch/batch_0.mat"
)

csi = np.asarray(loadmat(PATH)["csi_mon"])

if csi.shape[1] == 242:
    csi = csi[:50]
elif csi.shape[0] == 242:
    csi = csi.T[:50]
else:
    raise ValueError(csi.shape)

amp = np.abs(csi)
ref = np.median(amp[:30], axis=0)
valid = ref > max(np.median(ref) * 0.02, 1e-8)

profiles = amp[:, valid] / ref[valid]

def corr(a, b):
    return np.corrcoef(a, b)[0, 1]

print("STABLE REGION CORRELATION VERIFICATION")
print("=" * 65)

for i in range(37, 50):
    print(
        f"37 vs {i}: {corr(profiles[37], profiles[i]):.6f}"
    )

print("\nAdjacent stable samples:")
for i in range(37, 49):
    print(
        f"{i} vs {i+1}: "
        f"{corr(profiles[i], profiles[i+1]):.6f}"
    )

print("\nConsistency check:")
print("37 vs 44:", corr(profiles[37], profiles[44]))
print("37 vs 45:", corr(profiles[37], profiles[45]))
print("44 vs 45:", corr(profiles[44], profiles[45]))
