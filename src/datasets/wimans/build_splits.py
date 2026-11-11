#!/usr/bin/env python3
"""Build deterministic WiMANS split files.

Two protocols are generated from the validated manifest:

  paper_random_80_20
      A seeded random 80/20 split at the sample level, matching the WiMANS
      paper's recommended random train/test split. Samples from the same
      collection group can land on both sides, so this is a benchmark
      reproduction, not a group-generalization estimate.

  group_disjoint_80_20
      A seeded 80/20 split at the group level: all 99 samples of a group stay
      together. Groups are stratified by (environment, wifi_band) and balanced
      on user count. This is the primary evaluation split.

Split files intentionally contain sample IDs only. All protocol metadata,
distribution summaries, and leakage checks live in split_report.json.

The manifest itself is never modified.

Run from anywhere:
    python src/datasets/wimans/build_splits.py
    python src/datasets/wimans/build_splits.py --seed 42 --train-fraction 0.8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "processed" / "wimans" / "manifest.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "wimans" / "splits"
DEFAULT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.8


def load_manifest(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(sample: dict) -> tuple[int, int]:
    return (sample["group_id"], sample["sample_index"])


def group_table(samples: list[dict]) -> dict[int, dict]:
    """Return group_id -> metadata, validating within-group homogeneity."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["group_id"]].append(sample)

    meta: dict[int, dict] = {}
    for group_id, members in grouped.items():
        environments = {m["environment"] for m in members}
        bands = {m["wifi_band"] for m in members}
        user_counts = {m["number_of_users"] for m in members}
        if len(environments) != 1 or len(bands) != 1 or len(user_counts) != 1:
            raise SystemExit(
                f"Group {group_id} is not homogeneous: "
                f"environments={environments}, bands={bands}, "
                f"user_counts={user_counts}"
            )
        meta[group_id] = {
            "group_id": group_id,
            "environment": next(iter(environments)),
            "wifi_band": next(iter(bands)),
            "number_of_users": next(iter(user_counts)),
            "samples": len(members),
        }

    group_sizes = {entry["samples"] for entry in meta.values()}
    if len(group_sizes) != 1:
        raise SystemExit(
            f"Groups have unequal sample counts: {sorted(group_sizes)}"
        )
    return meta


def summarize_samples(
    sample_ids: list[str], by_id: dict[str, dict]
) -> dict:
    rows = [by_id[sid] for sid in sample_ids]
    lengths = [r["num_timesteps"] for r in rows]
    activity_counts = Counter(
        user["activity"] for row in rows for user in row["users"]
    )
    user_counts = Counter(str(row["number_of_users"]) for row in rows)

    return {
        "samples": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "environment": dict(
            sorted(Counter(row["environment"] for row in rows).items())
        ),
        "wifi_band": dict(
            sorted(Counter(row["wifi_band"] for row in rows).items())
        ),
        "number_of_users": dict(
            sorted(user_counts.items(), key=lambda item: int(item[0]))
        ),
        "activity_slots": dict(sorted(activity_counts.items())),
        "sequence_length": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": round(statistics.fmean(lengths), 2),
            "max": max(lengths),
        },
    }


def summarize_groups(group_ids: list[int], group_meta: dict[int, dict]) -> dict:
    rows = [group_meta[gid] for gid in group_ids]
    user_counts = Counter(str(row["number_of_users"]) for row in rows)
    return {
        "groups": len(rows),
        "environment": dict(
            sorted(Counter(row["environment"] for row in rows).items())
        ),
        "wifi_band": dict(
            sorted(Counter(row["wifi_band"] for row in rows).items())
        ),
        "number_of_users": dict(
            sorted(user_counts.items(), key=lambda item: int(item[0]))
        ),
    }


def check_sample_split(
    split: dict, all_ids: set[str], group_of: dict[str, int]
) -> dict:
    train = set(split["train"])
    test = set(split["test"])
    train_groups = {group_of[sid] for sid in train}
    test_groups = {group_of[sid] for sid in test}
    overlap = train_groups & test_groups
    test_in_train_groups = sum(1 for sid in test if group_of[sid] in train_groups)

    return {
        "train_samples": len(split["train"]),
        "test_samples": len(split["test"]),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "group_overlap": len(overlap),
        "test_samples_in_train_groups": test_in_train_groups,
        "groups_in_both": sorted(overlap),
        "sample_disjoint": train.isdisjoint(test),
        "sample_union_complete": (train | test) == all_ids,
        "train_has_duplicates": len(train) != len(split["train"]),
        "test_has_duplicates": len(test) != len(split["test"]),
    }


def largest_remainder(
    totals: Counter, target_total: int
) -> dict[int, int]:
    """Allocate target_total proportionally, with deterministic tie-breaking."""
    total = sum(totals.values())
    quotas = {key: target_total * count / total for key, count in totals.items()}
    allocation = {key: int(quota) for key, quota in quotas.items()}
    remaining = target_total - sum(allocation.values())
    ranked = sorted(
        quotas,
        key=lambda key: (-(quotas[key] - allocation[key]), key),
    )
    for key in ranked[:remaining]:
        allocation[key] += 1
    return allocation


def improve_spread(
    allocation: dict[tuple[str, str], dict[int, int]],
    cells: list[tuple[str, str]],
    users: list[int],
    capacity: dict[tuple[str, str], Counter],
) -> None:
    """Reduce concentration with marginal-preserving pairwise swaps.

    Row and column sums are fixed by construction; a swap exchanges one unit
    of user count u1 in cell c1 with one unit of u2 in cell c2. The objective
    sum(x^2) is minimised when counts are spread as evenly as possible.
    """

    def cost() -> int:
        return sum(allocation[cell][user] ** 2 for cell in cells for user in users)

    improved = True
    while improved:
        improved = False
        for index, cell_a in enumerate(cells):
            for cell_b in cells[index + 1 :]:
                for user_a in users:
                    if allocation[cell_a][user_a] == 0:
                        continue
                    for user_b in users:
                        if user_a == user_b or allocation[cell_b][user_b] == 0:
                            continue
                        if allocation[cell_a][user_b] + 1 > capacity[cell_a][user_b]:
                            continue
                        if allocation[cell_b][user_a] + 1 > capacity[cell_b][user_a]:
                            continue
                        previous = cost()
                        allocation[cell_a][user_a] -= 1
                        allocation[cell_a][user_b] += 1
                        allocation[cell_b][user_b] -= 1
                        allocation[cell_b][user_a] += 1
                        if cost() < previous:
                            improved = True
                        else:
                            allocation[cell_a][user_a] += 1
                            allocation[cell_a][user_b] -= 1
                            allocation[cell_b][user_b] += 1
                            allocation[cell_b][user_a] -= 1


def allocate_cell_user_counts(
    cells: list[tuple[str, str]],
    users: list[int],
    cell_target: dict[tuple[str, str], int],
    user_target: dict[int, int],
    capacity: dict[tuple[str, str], Counter],
) -> dict[tuple[str, str], dict[int, int]]:
    """Find exact per-cell/per-user test counts via deterministic backtracking."""

    def candidates(cell: tuple[str, str]) -> list[tuple[int, ...]]:
        target = cell_target[cell]
        out: list[tuple[int, ...]] = []

        def build(index: int, remaining: int, vector: list[int]) -> None:
            if index == len(users):
                if remaining == 0:
                    out.append(tuple(vector))
                return
            user = users[index]
            max_take = min(capacity[cell][user], remaining)
            for take in range(max_take + 1):
                vector.append(take)
                build(index + 1, remaining - take, vector)
                vector.pop()

        build(0, target, [])
        # Prefer spreading user counts across a cell over stacking one count.
        out.sort(key=lambda vec: (sum(x > 1 for x in vec), max(vec, default=0), vec))
        return out

    candidate_map = {cell: candidates(cell) for cell in cells}
    order = sorted(cells, key=lambda cell: (len(candidate_map[cell]), cell))

    def feasible(
        remaining_cells: list[tuple[str, str]], remaining_users: dict[int, int]
    ) -> bool:
        if sum(remaining_users.values()) != sum(
            cell_target[cell] for cell in remaining_cells
        ):
            return False
        for user in users:
            available = sum(capacity[cell][user] for cell in remaining_cells)
            if remaining_users[user] > available:
                return False
        for cell in remaining_cells:
            available = sum(
                min(capacity[cell][user], remaining_users[user]) for user in users
            )
            if available < cell_target[cell]:
                return False
        return True

    def search(index: int, remaining_users: dict[int, int]):
        if index == len(order):
            return {}
        cell = order[index]
        rest = order[index + 1 :]
        for vector in candidate_map[cell]:
            if any(
                vector[i] > remaining_users[user] for i, user in enumerate(users)
            ):
                continue
            next_remaining = {
                user: remaining_users[user] - vector[i]
                for i, user in enumerate(users)
            }
            if not feasible(rest, next_remaining):
                continue
            result = search(index + 1, next_remaining)
            if result is not None:
                result[cell] = dict(zip(users, vector))
                return result
        return None

    allocation = search(0, dict(user_target))
    if allocation is None:
        raise SystemExit(
            "Could not find a stratified group allocation for the requested "
            "fraction. Try the default --train-fraction 0.8."
        )
    improve_spread(allocation, cells, users, capacity)
    return allocation


def build_paper_random(
    samples: list[dict], by_id: dict[str, dict], train_fraction: float, seed: int
) -> dict:
    ordered = sorted(samples, key=natural_key)
    n_test = round((1.0 - train_fraction) * len(ordered))
    shuffled = ordered[:]
    random.Random(f"{seed}:paper_random").shuffle(shuffled)

    test_ids = sorted(
        (sample["sample_id"] for sample in shuffled[:n_test]),
        key=lambda sid: natural_key(by_id[sid]),
    )
    test_set = set(test_ids)
    train_ids = sorted(
        (sample["sample_id"] for sample in ordered if sample["sample_id"] not in test_set),
        key=lambda sid: natural_key(by_id[sid]),
    )
    return {"train": train_ids, "test": test_ids}


def build_group_disjoint(
    samples: list[dict],
    group_meta: dict[int, dict],
    train_fraction: float,
    seed: int,
) -> dict:
    groups = sorted(group_meta)
    n_test_groups = round((1.0 - train_fraction) * len(groups))
    cells = sorted(
        {(meta["environment"], meta["wifi_band"]) for meta in group_meta.values()}
    )
    users = sorted({meta["number_of_users"] for meta in group_meta.values()})

    # Balanced cell targets: 23 over 6 cells -> five cells with 4, one with 3.
    base, remainder = divmod(n_test_groups, len(cells))
    cell_order = cells[:]
    random.Random(f"{seed}:cells").shuffle(cell_order)
    cell_target = {cell: base for cell in cells}
    for cell in cell_order[:remainder]:
        cell_target[cell] += 1

    # Balanced user-count targets via largest remainder.
    user_totals = Counter(meta["number_of_users"] for meta in group_meta.values())
    user_target = largest_remainder(user_totals, n_test_groups)

    capacity: dict[tuple[str, str], Counter] = {cell: Counter() for cell in cells}
    buckets: dict[tuple[tuple[str, str], int], list[int]] = defaultdict(list)
    for group_id, meta in group_meta.items():
        cell = (meta["environment"], meta["wifi_band"])
        capacity[cell][meta["number_of_users"]] += 1
        buckets[(cell, meta["number_of_users"])].append(group_id)

    allocation = allocate_cell_user_counts(
        cells, users, cell_target, user_target, capacity
    )

    test_groups: set[int] = set()
    for cell in cells:
        for user in users:
            take = allocation[cell][user]
            if take == 0:
                continue
            bucket = sorted(buckets[(cell, user)])
            random.Random(f"{seed}:group:{cell[0]}:{cell[1]}:{user}").shuffle(bucket)
            test_groups.update(bucket[:take])

    train_groups = set(groups) - test_groups
    test_ids = [
        sample["sample_id"]
        for sample in sorted(
            (s for s in samples if s["group_id"] in test_groups), key=natural_key
        )
    ]
    train_ids = [
        sample["sample_id"]
        for sample in sorted(
            (s for s in samples if s["group_id"] in train_groups), key=natural_key
        )
    ]
    return {
        "train": train_ids,
        "test": test_ids,
        "train_groups": sorted(train_groups),
        "test_groups": sorted(test_groups),
        "cell_target": {f"{cell[0]}|{cell[1]}": cell_target[cell] for cell in cells},
        "user_target": {str(user): user_target[user] for user in users},
        "cell_user_allocation": {
            f"{cell[0]}|{cell[1]}": {
                str(user): allocation[cell][user] for user in users
            }
            for cell in cells
        },
    }


def write_split(path: Path, split: dict) -> None:
    payload = {"train": split["train"], "test": split["test"]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic WiMANS paper-random and group-disjoint splits."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Input manifest.jsonl (default: data/processed/wimans/manifest.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: data/processed/wimans/splits)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Training fraction (default: 0.8)",
    )
    args = parser.parse_args()

    if not 0.0 < args.train_fraction < 1.0:
        raise SystemExit("--train-fraction must be between 0 and 1")

    samples = load_manifest(args.manifest)
    by_id = {sample["sample_id"]: sample for sample in samples}
    if len(by_id) != len(samples):
        raise SystemExit("Manifest contains duplicate sample IDs")

    group_meta = group_table(samples)
    group_size = next(iter({entry["samples"] for entry in group_meta.values()}))
    group_of = {sample["sample_id"]: sample["group_id"] for sample in samples}
    all_ids = set(by_id)

    paper = build_paper_random(samples, by_id, args.train_fraction, args.seed)
    group_disjoint = build_group_disjoint(
        samples, group_meta, args.train_fraction, args.seed
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_pct = round(args.train_fraction * 100)
    test_pct = 100 - train_pct
    paper_path = args.out_dir / f"paper_random_{train_pct}_{test_pct}.json"
    group_path = args.out_dir / f"group_disjoint_{train_pct}_{test_pct}.json"
    report_path = args.out_dir / "split_report.json"

    write_split(paper_path, paper)
    write_split(group_path, group_disjoint)

    paper_checks = check_sample_split(paper, all_ids, group_of)
    group_checks = check_sample_split(group_disjoint, all_ids, group_of)
    group_checks["train_groups_list"] = group_disjoint["train_groups"]
    group_checks["test_groups_list"] = group_disjoint["test_groups"]
    group_checks["cell_target"] = group_disjoint["cell_target"]
    group_checks["user_target"] = group_disjoint["user_target"]
    group_checks["cell_user_allocation"] = group_disjoint["cell_user_allocation"]

    for label, checks in (
        ("paper_random", paper_checks),
        ("group_disjoint", group_checks),
    ):
        if not checks["sample_disjoint"] or not checks["sample_union_complete"]:
            raise SystemExit(f"{label} split failed basic coverage checks: {checks}")
        if checks["train_has_duplicates"] or checks["test_has_duplicates"]:
            raise SystemExit(f"{label} split contains duplicate sample IDs")

    if group_checks["group_overlap"] != 0:
        raise SystemExit("Group-disjoint split leaked groups across train/test")
    if group_checks["test_samples_in_train_groups"] != 0:
        raise SystemExit("Group-disjoint split has test samples in train groups")
    if group_checks["train_groups"] + group_checks["test_groups"] != len(group_meta):
        raise SystemExit("Group-disjoint split does not cover every group")

    paper_name = f"paper_random_{train_pct}_{test_pct}"
    group_name = f"group_disjoint_{train_pct}_{test_pct}"

    report = {
        "dataset": "WiMANS",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "deterministic": True,
        "caveats": {
            "group_definition": (
                "group_id is parsed from the sample label act_<group_id>_"
                f"<sample_index>. Each group has {group_size} samples and is "
                "homogeneous in environment, wifi_band, and number_of_users."
            ),
            "group_disjoint_is_not_person_disjoint": (
                "The six volunteer identities recur across groups. Protocol B "
                "tests unseen collection groups, not unseen people."
            ),
            "zero_user_samples": (
                "Samples with number_of_users == 0 have no activity labels. "
                "They remain in both split files for the no-human baseline and "
                "must be filtered out of the 9-class activity benchmark. "
                "Filtering by user count preserves group disjointness because "
                "user count is constant within a group."
            ),
        },
        "protocols": {
            paper_name: {
                "description": (
                    "Seeded random 80/20 split at the sample level, matching "
                    "the WiMANS paper's recommended random train/test split. "
                    "Groups are not held out; this is a paper-comparison "
                    "benchmark, not a group-generalization estimate."
                ),
                "split_unit": "sample",
                "file": paper_path.name,
                "train": summarize_samples(paper["train"], by_id),
                "test": summarize_samples(paper["test"], by_id),
                "checks": paper_checks,
            },
            group_name: {
                "description": (
                    "Seeded 80/20 split at the group level: all samples from "
                    "a group stay on one side. Groups are stratified by "
                    "(environment, wifi_band) and balanced on user count. "
                    "This is the primary evaluation split."
                ),
                "split_unit": "group",
                "file": group_path.name,
                "train": summarize_samples(group_disjoint["train"], by_id),
                "test": summarize_samples(group_disjoint["test"], by_id),
                "train_groups": summarize_groups(
                    group_disjoint["train_groups"], group_meta
                ),
                "test_groups": summarize_groups(
                    group_disjoint["test_groups"], group_meta
                ),
                "checks": group_checks,
            },
        },
        "output_sha256": {
            paper_path.name: sha256_file(paper_path),
            group_path.name: sha256_file(group_path),
        },
        "leakage_summary": {
            paper_name: {
                "group_overlap": paper_checks["group_overlap"],
                "test_samples_in_train_groups": paper_checks[
                    "test_samples_in_train_groups"
                ],
                "status": "expected_group_leakage",
                "note": (
                    "Sample-level random splitting puts every group on both "
                    "sides. Use for paper comparison only."
                ),
            },
            group_name: {
                "group_overlap": group_checks["group_overlap"],
                "test_samples_in_train_groups": group_checks[
                    "test_samples_in_train_groups"
                ],
                "status": "clean",
                "note": (
                    "No group and no sample appears on both sides. This does "
                    "not imply person-disjoint: the six identities recur "
                    "across groups."
                ),
            },
        },
    }

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("========== WiMANS SPLITS ==========")
    print(f"Manifest: {args.manifest} ({len(samples)} samples)")
    print(f"Seed: {args.seed}  Train fraction: {args.train_fraction}")
    for name, split, checks in (
        (paper_name, paper, paper_checks),
        (group_name, group_disjoint, group_checks),
    ):
        print(f"\n[{name}]")
        print(
            f"  samples: train={checks['train_samples']} "
            f"test={checks['test_samples']}"
        )
        print(
            f"  groups:  train={checks['train_groups']} "
            f"test={checks['test_groups']} overlap={checks['group_overlap']}"
        )
        print(f"  sample_disjoint={checks['sample_disjoint']}")
        print(f"  sample_union_complete={checks['sample_union_complete']}")
        print(
            f"  test_samples_in_train_groups="
            f"{checks['test_samples_in_train_groups']}"
        )

    print(f"\nWritten:")
    print(f"  {paper_path}")
    print(f"  {group_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
