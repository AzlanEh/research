cd ~/workspace/wifi-extend

PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
from collections import Counter

ids = {"act_1_1", "act_1_2", "act_1_3"}

def load_jsonl(path):
    with open(path) as f:
        return {
            r["sample_id"]: r
            for r in map(json.loads, f)
            if r["sample_id"] in ids
        }

features = load_jsonl("data/processed/wimans/features_v1.jsonl")
manifest = load_jsonl("data/processed/wimans/manifest.jsonl")
predictions = load_jsonl("data/processed/wimans/smoke_predictions.jsonl")

for sid in sorted(ids):
    m = manifest[sid]
    f = features[sid]
    p = predictions[sid]

    actual = Counter(u["activity"] for u in m["users"])

    print("\n" + "=" * 65)
    print("SAMPLE:", sid)
    print("Group:", m["group_id"])
    print("Environment:", m["environment"])
    print("Wi-Fi band:", m["wifi_band"])
    print("Ground truth:", dict(actual))
    print("Prediction:", p["predicted_activity_counts"])
    print("Model reason:", p.get("reason"))
    print("Prediction fields:", list(p.keys()))

    summary = f["features"]["recording_summary"]
    print("\nRecording summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nPer-link temporal features:")
    for link, values in f["features"]["links"].items():
        print(
            f"  {link}: "
            f"delta={values['temporal_delta_mean']:.4f}, "
            f"variability={values['window_mean_variability']:.4f}, "
            f"mean={values['signal_mean']:.4f}, "
            f"std={values['signal_std']:.4f}"
        )
PY
