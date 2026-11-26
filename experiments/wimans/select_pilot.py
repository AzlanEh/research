import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")
MANIFEST_PATH = ROOT / "data/processed/wimans/manifest.jsonl"
SPLIT_PATH = ROOT / "data/processed/wimans/splits/group_disjoint_80_20.json"

OUTPUT_DIR = ROOT / "experiments/wimans"
SELECTION_PATH = OUTPUT_DIR / "pilot_20_selection.json"
PILOT_MANIFEST_PATH = OUTPUT_DIR / "pilot_20_manifest.jsonl"

SEED = 42
USER_ONE_GROUPS_TO_SELECT = 4


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Load the full manifest.
records = {}
with MANIFEST_PATH.open() as f:
    for line in f:
        row = json.loads(line)
        records[row["sample_id"]] = row

# Load the group-disjoint split.
with SPLIT_PATH.open() as f:
    split = json.load(f)

test_ids = split["test"]
test_rows = [records[sample_id] for sample_id in test_ids]

groups = defaultdict(list)
for row in test_rows:
    groups[int(row["group_id"])].append(row)

# Verify the expected group structure.
assert len(groups) == 23, f"Expected 23 groups, got {len(groups)}"

for group_id, rows in groups.items():
    assert len(rows) == 99, f"Group {group_id} has {len(rows)} samples"

    strata = {
        (
            row["environment"],
            str(row["wifi_band"]),
            int(row["number_of_users"]),
        )
        for row in rows
    }
    assert len(strata) == 1, f"Inconsistent group {group_id}: {strata}"

# Select every group except the user-count-1 groups.
mandatory_groups = sorted(
    group_id
    for group_id, rows in groups.items()
    if int(rows[0]["number_of_users"]) != 1
)

user_one_groups = sorted(
    group_id
    for group_id, rows in groups.items()
    if int(rows[0]["number_of_users"]) == 1
)

assert len(mandatory_groups) == 16
assert len(user_one_groups) == 7
assert 203 in mandatory_groups

# Select four user-1 groups deterministically.
group_rng = random.Random(SEED)
selected_user_one = sorted(
    group_rng.sample(user_one_groups, USER_ONE_GROUPS_TO_SELECT)
)

selected_groups = sorted(mandatory_groups + selected_user_one)
assert len(selected_groups) == 20
assert len(set(selected_groups)) == 20

# Select one sample from each group using an independent RNG.
sample_rng = random.Random(SEED + 1)
selected_rows = []

for group_id in selected_groups:
    candidates = sorted(
        groups[group_id],
        key=lambda row: row["sample_id"],
    )
    selected_rows.append(sample_rng.choice(candidates))

selected_rows.sort(key=lambda row: int(row["group_id"]))

# Validate the final selection.
assert len(selected_rows) == 20
assert len({row["sample_id"] for row in selected_rows}) == 20
assert len({int(row["group_id"]) for row in selected_rows}) == 20
assert any(int(row["number_of_users"]) == 0 for row in selected_rows)

user_counts = Counter(
    int(row["number_of_users"]) for row in selected_rows
)
environments = sorted({row["environment"] for row in selected_rows})
bands = sorted({str(row["wifi_band"]) for row in selected_rows})
environment_band_pairs = sorted({
    (row["environment"], str(row["wifi_band"]))
    for row in selected_rows
})

assert user_counts == {
    0: 1, 1: 4, 2: 4, 3: 4, 4: 4, 5: 3
}, user_counts

assert len(environment_band_pairs) == 6, environment_band_pairs

# Save the full selected records for feature extraction.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with PILOT_MANIFEST_PATH.open("w") as f:
    for row in selected_rows:
        f.write(json.dumps(row) + "\n")

# Save a compact, auditable selection record.
selection = {
    "protocol": "wimans_group_disjoint_20_recording_pilot",
    "selection_seed": SEED,
    "sample_selection_seed": SEED + 1,
    "source_split": str(SPLIT_PATH),
    "source_manifest": str(MANIFEST_PATH),
    "source_manifest_sha256": sha256(MANIFEST_PATH),
    "source_split_sha256": sha256(SPLIT_PATH),
    "selection_rule": (
        "Include all test groups with user counts other than 1; "
        "select 4 of the 7 user-count-1 groups using the fixed seed; "
        "select one sample per selected group using an independent fixed seed."
    ),
    "selected_group_ids": selected_groups,
    "selected_user_one_group_ids": selected_user_one,
    "selected_samples": [
        {
            "sample_id": row["sample_id"],
            "group_id": int(row["group_id"]),
            "environment": row["environment"],
            "wifi_band": str(row["wifi_band"]),
            "number_of_users": int(row["number_of_users"]),
        }
        for row in selected_rows
    ],
    "summary": {
        "sample_count": len(selected_rows),
        "distinct_group_count": len(selected_groups),
        "user_count_distribution": dict(sorted(user_counts.items())),
        "environments": environments,
        "wifi_bands": bands,
        "environment_band_combinations": [
            {"environment": env, "wifi_band": band}
            for env, band in environment_band_pairs
        ],
    },
    "caveat": (
        "Coverage-oriented pilot, not a representative random sample. "
        "One sample is selected per group. Not suitable for statistical "
        "significance or generalization claims."
    ),
}

SELECTION_PATH.write_text(json.dumps(selection, indent=2) + "\n")

print("Pilot selection created successfully.")
print(f"Selection record: {SELECTION_PATH}")
print(f"Pilot manifest:   {PILOT_MANIFEST_PATH}")
print()
print("Selected groups:", selected_groups)
print("User-1 groups selected:", selected_user_one)
print("User-count distribution:", dict(sorted(user_counts.items())))
print("Environments:", environments)
print("Wi-Fi bands:", bands)
print("Environment × band combinations:", len(environment_band_pairs))
print()
print("Selected recordings:")
for row in selected_rows:
    print(
        f"  {row['sample_id']:>12} | "
        f"group={int(row['group_id']):>3} | "
        f"{row['environment']:<12} | "
        f"{str(row['wifi_band']):<4} | "
        f"users={int(row['number_of_users'])}"
    )
