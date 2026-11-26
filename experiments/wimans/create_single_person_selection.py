import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "experiments/wimans/single_person_pilot"

MANIFEST_PATH = ROOT / "data/processed/wimans/manifest.jsonl"
SPLIT_PATH = ROOT / "data/processed/wimans/splits/group_disjoint_80_20.json"

SEED = 20261123
DEPTH_PER_CELL = 1
ZERO_PERSON_SAMPLES = 7

ACTIVITIES = [
    "jump",
    "lie_down",
    "nothing",
    "pick_up",
    "rotation",
    "sit_down",
    "stand_up",
    "walk",
    "wave",
]

EXPECTED_GROUPS = {1, 3, 24, 38, 60, 77, 92}
ZERO_PERSON_GROUP = 203


def load_jsonl(path):
    rows = {}
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ValueError(f"Duplicate sample_id at {path}:{line_number}: {sample_id}")
            rows[sample_id] = row
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = load_jsonl(MANIFEST_PATH)

    with SPLIT_PATH.open() as f:
        split = json.load(f)

    test_ids = split["test"]
    missing = [sample_id for sample_id in test_ids if sample_id not in manifest]
    if missing:
        raise ValueError(f"{len(missing)} test IDs missing from manifest: {missing[:10]}")

    # Build the exact test-split candidate pools.
    activity_cells = defaultdict(list)
    zero_candidates = []

    for sample_id in test_ids:
        row = manifest[sample_id]
        count = row.get("number_of_users")
        users = row.get("users", [])

        if count == 0:
            if users:
                raise ValueError(
                    f"Zero-person sample has user annotations: {sample_id}"
                )
            if row.get("group_id") == ZERO_PERSON_GROUP:
                zero_candidates.append(row)
            continue

        if count != 1:
            continue

        if len(users) != 1:
            raise ValueError(
                f"Single-person sample does not have exactly one user: {sample_id}"
            )

        activity = users[0].get("activity")
        if activity not in ACTIVITIES:
            raise ValueError(
                f"Unexpected or missing activity for {sample_id}: {activity!r}"
            )

        activity_cells[(activity, row["group_id"])].append(row)

    # Validate the expected structure before selecting anything.
    actual_groups = {
        group_id
        for _, group_id in activity_cells
    }
    if actual_groups != EXPECTED_GROUPS:
        raise ValueError(
            f"Unexpected single-person groups: {sorted(actual_groups)}"
        )

    for activity in ACTIVITIES:
        for group_id in sorted(EXPECTED_GROUPS):
            candidates = activity_cells[(activity, group_id)]
            if len(candidates) != 11:
                raise ValueError(
                    f"Expected 11 candidates for ({activity}, group {group_id}); "
                    f"found {len(candidates)}"
                )

    if len(zero_candidates) != 99:
        raise ValueError(
            f"Expected 99 zero-person candidates in group 203; "
            f"found {len(zero_candidates)}"
        )

    # Deterministic selection: one sample from each activity/group cell.
    rng = random.Random(SEED)
    selected = []

    for activity in ACTIVITIES:
        for group_id in sorted(EXPECTED_GROUPS):
            candidates = sorted(
                activity_cells[(activity, group_id)],
                key=lambda row: row["sample_id"],
            )
            chosen = rng.choice(candidates)

            selected.append({
                "sample_id": chosen["sample_id"],
                "arm": "single_person_activity",
                "group_id": chosen["group_id"],
                "environment": chosen["environment"],
                "wifi_band": chosen["wifi_band"],
            })

    # Select seven zero-person recordings from the single available group.
    zero_candidates = sorted(
        zero_candidates,
        key=lambda row: row["sample_id"],
    )
    selected_zero = rng.sample(zero_candidates, ZERO_PERSON_SAMPLES)

    for row in selected_zero:
        selected.append({
            "sample_id": row["sample_id"],
            "arm": "zero_person_detection",
            "group_id": row["group_id"],
            "environment": row["environment"],
            "wifi_band": row["wifi_band"],
        })

    # Keep labels out of the inference manifest.
    ground_truth = {}
    for item in selected:
        row = manifest[item["sample_id"]]

        if item["arm"] == "single_person_activity":
            ground_truth[item["sample_id"]] = {
                "number_of_users": 1,
                "activity": row["users"][0]["activity"],
            }
        else:
            ground_truth[item["sample_id"]] = {
                "number_of_users": 0,
                "activity": None,
            }

    # Audit the selection.
    activity_counts = Counter()
    group_activity_counts = Counter()
    environment_counts = Counter()
    band_counts = Counter()
    zero_group_counts = Counter()

    for item in selected:
        if item["arm"] == "single_person_activity":
            activity = ground_truth[item["sample_id"]]["activity"]
            activity_counts[activity] += 1
            group_activity_counts[(item["group_id"], activity)] += 1
        else:
            zero_group_counts[item["group_id"]] += 1

        environment_counts[item["environment"]] += 1
        band_counts[item["wifi_band"]] += 1

    assert len(selected) == 70
    assert len({x["sample_id"] for x in selected}) == 70
    assert all(activity_counts[a] == 7 for a in ACTIVITIES)
    assert all(
        group_activity_counts[(group_id, activity)] == 1
        for group_id in EXPECTED_GROUPS
        for activity in ACTIVITIES
    )
    assert sum(zero_group_counts.values()) == 7
    assert set(zero_group_counts) == {ZERO_PERSON_GROUP}

    selection_manifest = {
        "experiment": "wimans_single_person_pilot",
        "selection_version": "single_person_selection_v1",
        "seed": SEED,
        "test_split": "group_disjoint_80_20",
        "design": {
            "single_person_groups": sorted(EXPECTED_GROUPS),
            "samples_per_activity_group_cell": DEPTH_PER_CELL,
            "single_person_samples": 63,
            "zero_person_samples": ZERO_PERSON_SAMPLES,
            "total_samples": 70,
        },
        "records": selected,
    }

    audit = {
        "experiment": "wimans_single_person_pilot",
        "selection_version": "single_person_selection_v1",
        "seed": SEED,
        "total_selected": len(selected),
        "unique_sample_ids": len({x["sample_id"] for x in selected}),
        "single_person_samples": sum(activity_counts.values()),
        "zero_person_samples": sum(zero_group_counts.values()),
        "single_person_activity_counts": dict(sorted(activity_counts.items())),
        "single_person_group_ids": sorted(EXPECTED_GROUPS),
        "group_activity_cell_counts": {
            f"{group_id}:{activity}": group_activity_counts[(group_id, activity)]
            for group_id in sorted(EXPECTED_GROUPS)
            for activity in ACTIVITIES
        },
        "zero_person_group_counts": dict(zero_group_counts),
        "environment_counts": dict(environment_counts),
        "wifi_band_counts": dict(band_counts),
        "zero_person_limitation": (
            "All selected zero-person recordings originate from group 203; "
            "this arm is a single-session evaluation and cannot establish "
            "cross-environment or cross-band occupancy generalization."
        ),
        "ground_truth_separated_from_inference_manifest": True,
        "status": "PASS",
    }

    (OUT / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2) + "\n"
    )
    (OUT / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2) + "\n"
    )
    (OUT / "selection_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    print("SELECTION COMPLETE")
    print("==================")
    print("Selected recordings:", len(selected))
    print("Single-person:", sum(activity_counts.values()))
    print("Zero-person:", sum(zero_group_counts.values()))
    print("Unique IDs:", len({x["sample_id"] for x in selected}))

    print("\nACTIVITY COUNTS")
    for activity in ACTIVITIES:
        print(f"{activity:<12} {activity_counts[activity]}")

    print("\nZERO-PERSON GROUPS:", dict(zero_group_counts))
    print("ENVIRONMENTS:", dict(environment_counts))
    print("WIFI BANDS:", dict(band_counts))

    print("\nSaved:")
    print(OUT / "selection_manifest.json")
    print(OUT / "ground_truth.json")
    print(OUT / "selection_audit.json")
    print("\nAudit status:", audit["status"])


if __name__ == "__main__":
    main()
