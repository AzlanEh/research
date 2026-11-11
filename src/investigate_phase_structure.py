import os
import numpy as np
from scipy.io import loadmat

BASE = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch"
)

REGIONS = {
    0: {
        "reference": [20, 21, 22, 23, 24, 25, 26, 27],
        "test": [30, 31, 32, 33, 34, 35, 36, 37],
    },
    1003: {
        "reference": [0, 1, 2, 3, 4],
        "test": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    },
    1007: {
        "reference": [10, 12, 14, 16, 18],
        "test": [20, 21, 22, 25, 29, 30, 40, 48, 49],
    },
}


def compare_phase(reference, sample):
    amp_a = np.abs(reference)
    amp_b = np.abs(sample)

    threshold_a = max(np.median(amp_a) * 0.02, 1e-8)
    threshold_b = max(np.median(amp_b) * 0.02, 1e-8)

    valid = (amp_a > threshold_a) & (amp_b > threshold_b)

    if np.count_nonzero(valid) < 10:
        return None

    a = reference[valid]
    b = sample[valid]

    # Amplitude-profile residual after optimal scalar gain.
    aa = np.abs(a)
    bb = np.abs(b)

    gain = np.dot(aa, bb) / (np.dot(aa, aa) + 1e-12)
    amp_residual = (
        np.linalg.norm(bb - gain * aa) /
        (np.linalg.norm(bb) + 1e-12)
    )

    # Per-subcarrier phase difference.
    phase = np.angle(b * np.conj(a))

    weights = np.abs(a) * np.abs(b)
    weights /= np.sum(weights) + 1e-12

    # Common phase coherence, without slope correction.
    common_phase = np.angle(np.sum(weights * np.exp(1j * phase)))

    raw_coherence = abs(
        np.sum(weights * np.exp(1j * phase))
    )

    # Fit a linear phase slope using a circular search.
    # This avoids directly unwrapping phase across missing subcarriers.
    subcarriers = np.arange(len(reference))[valid]

    slopes = np.linspace(-np.pi, np.pi, 4001)

    scores = np.array([
        abs(np.sum(
            weights * np.exp(1j * (phase - slope * subcarriers))
        ))
        for slope in slopes
    ])

    best_idx = np.argmax(scores)
    best_slope = slopes[best_idx]

    corrected = phase - best_slope * subcarriers

    corrected_common = np.angle(
        np.sum(weights * np.exp(1j * corrected))
    )

    corrected_coherence = abs(
        np.sum(
            weights * np.exp(
                1j * (corrected - corrected_common)
            )
        )
    )

    # Circular phase spread after linear slope removal.
    circular_std = np.degrees(
        np.sqrt(max(0, -2 * np.log(max(corrected_coherence, 1e-12))))
    )

    return {
        "amp_residual": amp_residual,
        "raw_coherence": raw_coherence,
        "corrected_coherence": corrected_coherence,
        "slope_deg": np.degrees(best_slope),
        "circular_std": circular_std,
        "common_phase": np.degrees(common_phase),
    }


for batch, region in REGIONS.items():
    path = os.path.join(BASE, f"batch_{batch}.mat")

    csi = np.asarray(loadmat(path)["csi_mon"])

    if csi.shape[1] != 242 and csi.shape[0] == 242:
        csi = csi.T

    print("\n" + "=" * 105)
    print(f"BATCH {batch}")
    print("=" * 105)

    print(
        f"{'Sample':>7} {'Best Ref':>9} {'Amp Resid':>11} "
        f"{'Raw Coh':>9} {'Slope Coh':>11} "
        f"{'Slope':>11} {'Phase Spread':>14}"
    )

    for idx in region["test"]:
        sample = csi[idx]
        candidates = []

        for ref_idx in region["reference"]:
            result = compare_phase(csi[ref_idx], sample)

            if result is not None:
                candidates.append((ref_idx, result))

        if not candidates:
            print(f"{idx:7d} No valid reference")
            continue

        # Select the reference with the lowest amplitude residual.
        # Phase metrics are then evaluated against that same reference.
        best_ref, best = min(
            candidates,
            key=lambda item: item[1]["amp_residual"]
        )

        print(
            f"{idx:7d} {best_ref:9d} "
            f"{best['amp_residual']:10.4f} "
            f"{best['raw_coherence']:9.4f} "
            f"{best['corrected_coherence']:11.4f} "
            f"{best['slope_deg']:10.3f} "
            f"{best['circular_std']:13.2f}°"
        )

print("\nPhase-slope investigation complete.")
