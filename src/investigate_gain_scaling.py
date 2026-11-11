import scipy.io as sio
import numpy as np

path = "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch/batch_0.mat"

data = sio.loadmat(path)
csi = data["csi_mon"]

amp = np.abs(csi)

print("\nGAIN SCALING INVESTIGATION")
print("=" * 65)

for i in range(29, 36):

    a = amp[i]
    b = amp[i + 1]

    # Least-squares multiplicative scaling factor
    alpha = np.dot(a, b) / (np.dot(a, a) + 1e-12)

    # Relative reconstruction error
    predicted = alpha * a

    error = np.linalg.norm(b - predicted) / (
        np.linalg.norm(b) + 1e-12
    )

    # Profile correlation
    correlation = np.corrcoef(a, b)[0, 1]

    print(f"\nTransition {i} -> {i+1}")
    print(f"Scaling factor: {alpha:.6f}")
    print(f"Correlation:    {correlation:.6f}")
    print(f"Relative error: {error:.8f}")
