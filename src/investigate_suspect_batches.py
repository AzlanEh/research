import os
import numpy as np
from scipy.io import loadmat

BASE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/"
    "Test/A_batch"
)

BATCHES = [0, 1003, 1007]

for batch_id in BATCHES:
    path = os.path.join(BASE, f"batch_{batch_id}.mat")
    csi = np.asarray(loadmat(path)["csi_mon"])

    if csi.ndim != 2:
        print(f"Invalid shape in batch {batch_id}: {csi.shape}")
        continue

    if csi.shape[1] == 242:
        pass
    elif csi.shape[0] == 242:
        csi = csi.T
    else:
        print(f"Unknown shape in batch {batch_id}: {csi.shape}")
        continue

    amp = np.abs(csi)
    reference = np.median(amp, axis=0)
    valid = reference > max(np.median(reference) * 0.02, 1e-8)

    profiles = amp[:, valid] / reference[valid]
    gains = np.median(profiles, axis=1)

    print(f"\n{'=' * 65}")
    print(f"BATCH {batch_id}")
    print("=" * 65)

    print("Sample gains:")
    for start in range(0, len(gains), 10):
        end = min(start + 10, len(gains))
        values = " ".join(f"{x:.3f}" for x in gains[start:end])
        print(f"{start:02d}-{end-1:02d}: {values}")

    if len(gains) > 1:
        ratios = gains[1:] / np.maximum(gains[:-1], 1e-12)
        log_ratios = np.log(ratios)

        # Positive/negative alternation in consecutive gain changes.
        signs = np.sign(log_ratios)
        alternation = (
            -np.mean(signs[1:] * signs[:-1])
            if len(signs) > 1 else 0
        )

        print("\nAdjacent gain ratios:")
        for start in range(0, len(ratios), 10):
            end = min(start + 10, len(ratios))
            values = " ".join(f"{x:.3f}" for x in ratios[start:end])
            print(f"{start:02d}->{end:02d}: {values}")

        print(f"\nAdjacent ratio median: {np.median(ratios):.4f}")
        print(f"Adjacent ratio std:    {np.std(ratios):.4f}")
        print(f"Alternation score:     {alternation:.4f}")

    print("\nGain statistics:")
    print(f"Median: {np.median(gains):.4f}")
    print(f"Minimum: {np.min(gains):.4f}")
    print(f"Maximum: {np.max(gains):.4f}")
    print(f"Std: {np.std(gains):.4f}")

print("\nInvestigation complete.")
