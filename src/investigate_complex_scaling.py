import scipy.io as sio
import numpy as np

path = "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch/batch_0.mat"

data = sio.loadmat(path)
csi = data["csi_mon"]

print("\nCOMPLEX CSI SCALING ANALYSIS")
print("=" * 65)

for i in range(29, 36):

    x = csi[i]
    y = csi[i + 1]

    # Best complex multiplicative scaling factor
    alpha = np.vdot(x, y) / (np.vdot(x, x) + 1e-12)

    predicted = alpha * x

    error = np.linalg.norm(y - predicted) / (
        np.linalg.norm(y) + 1e-12
    )

    phase_change = np.angle(alpha)

    print(f"\nTransition {i} -> {i+1}")
    print(f"Complex scaling: {alpha}")
    print(f"Amplitude scaling: {abs(alpha):.6f}")
    print(f"Phase rotation: {phase_change:.6f} radians")
    print(f"Reconstruction error: {error:.8f}")
