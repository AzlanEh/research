import os
import numpy as np
from scipy.io import loadmat

BASE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/"
    "Test/A_batch"
)

BATCHES = [0] + list(range(1000, 1010))

print("\nCROSS-BATCH CSI GAIN ANALYSIS")
print("=" * 90)
print(
    f"{'Batch':12s} {'Samples':>8s} {'Median':>10s} "
    f"{'Std':>10s} {'Even':>10s} {'Odd':>10s} "
    f"{'Even/Odd':>10s}"
)
print("-" * 90)

for batch_id in BATCHES:
    path = os.path.join(BASE, f"batch_{batch_id}.mat")

    if not os.path.exists(path):
        print(f"batch_{batch_id:<6} FILE NOT FOUND")
        continue

    try:
        data = loadmat(path)
        csi = np.asarray(data["csi_mon"])

        if csi.ndim != 2:
            print(f"batch_{batch_id:<6} INVALID SHAPE {csi.shape}")
            continue

        if csi.shape[1] == 242:
            pass
        elif csi.shape[0] == 242:
            csi = csi.T
        else:
            print(f"batch_{batch_id:<6} UNKNOWN SHAPE {csi.shape}")
            continue

        amp = np.abs(csi)

        if amp.shape[0] < 4:
            print(f"batch_{batch_id:<6} TOO FEW SAMPLES")
            continue

        # Normalize each subcarrier by its batch median.
        reference = np.median(amp, axis=0)
        valid = reference > max(np.median(reference) * 0.02, 1e-8)

        profiles = amp[:, valid] / reference[valid]

        # One robust gain estimate per sample.
        gains = np.median(profiles, axis=1)

        even = np.median(gains[::2])
        odd = np.median(gains[1::2])

        parity_ratio = even / max(odd, 1e-12)

        print(
            f"batch_{batch_id:<6} {len(gains):8d} "
            f"{np.median(gains):10.4f} "
            f"{np.std(gains):10.4f} "
            f"{even:10.4f} {odd:10.4f} "
            f"{parity_ratio:10.4f}"
        )

    except Exception as e:
        print(f"batch_{batch_id:<6} ERROR: {e}")

print("-" * 90)
print("Analysis complete.")
