"""Extract V3 features for few-shot exemplar samples.

Usage (from project root):
    PYTHONPATH=src ./.venv/bin/python src/preprocessing/wimans/extract_exemplar_features_v3.py \\
        --exemplars experiments/wimans/activity_only_v2/few_shot_exemplars.json \\
        --amp-dir /mnt/external_disk/gaurav_project/wifi_csi/amp \\
        --out data/processed/wimans/features_v3_exemplars.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from project root with PYTHONPATH=src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.wimans.extract_features_v3 import extract_recording_features_v3

AMP_DIR_DEFAULT = Path("/mnt/external_disk/gaurav_project/wifi_csi/amp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract V3 features for few-shot exemplars.")
    parser.add_argument("--exemplars", type=Path,
                        default=Path("experiments/wimans/activity_only_v2/few_shot_exemplars.json"))
    parser.add_argument("--amp-dir", type=Path, default=AMP_DIR_DEFAULT)
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/wimans/features_v3_exemplars.jsonl"))
    args = parser.parse_args()

    exemplars: dict[str, str] = json.loads(args.exemplars.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for activity, sample_id in exemplars.items():
        amp_path = args.amp_dir / f"{sample_id}.npy"
        if not amp_path.exists():
            print(f"[ERROR] Amplitude file not found: {amp_path}", flush=True)
            sys.exit(1)

        csi = np.load(amp_path)
        print(f"[{sample_id}] Loaded CSI shape={csi.shape}, activity={activity}", flush=True)
        features = extract_recording_features_v3(csi)
        record = {
            "sample_id": sample_id,
            "activity": activity,
            **features,
        }
        records.append(record)

    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(records)} exemplar feature records to {args.out}", flush=True)
    # Print a compact summary table
    print("\n{:<12} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "sample_id", "activity",
        "rel_delta", "rel_std", "dom_shift", "burst", "packets"
    ))
    print("-" * 80)
    for rec in records:
        rp = rec["recording_profile"]
        print("{:<12} {:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.1f} {:<10}".format(
            rec["sample_id"], rec["activity"],
            rp["global_rel_motion_delta"],
            rp["global_rel_trajectory_std"],
            rp["dominant_signed_posture_shift"],
            rp["max_burst_ratio"],
            rec["num_packets"],
        ))


if __name__ == "__main__":
    main()
