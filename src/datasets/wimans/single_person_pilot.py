"""Deterministic single-person / zero-person selection for WiMANS.

The single-person benchmark uses the group-disjoint test split only. Eligible
recordings are those with exactly one annotated person, plus a separate
zero-person arm. ``empty_room`` is an environment label, never an occupancy
label: occupancy comes from ``number_of_users`` and the per-user annotations.

Selection is balanced by construction. In the current split each of the seven
user-count-1 groups contains 11 samples of every activity, so one sample per
(activity, group) cell yields 63 single-person recordings with exactly seven
examples of each of the nine activities and seven distinct groups per class.

The output contains no model inputs beyond identifiers and metadata; prompts
are built separately from the feature files.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Must match llm.wimans.prompt_builder.ACTIVITY_LABELS. Kept local so the
# dataset layer does not depend on the LLM package; a test asserts equality.
ACTIVITIES = (
    "nothing",
    "walk",
    "rotation",
    "jump",
    "wave",
    "lie_down",
    "pick_up",
    "sit_down",
    "stand_up",
)

PROTOCOL = "wimans_single_person_pilot_v1"
DEFAULT_SEED = 42
DEFAULT_PER_CELL = 1
DEFAULT_ZERO_PERSON = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records[row["sample_id"]] = row
    return records


def load_test_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return list(json.load(handle)["test"])


def _user_count(row: Mapping[str, Any]) -> int:
    return int(row["number_of_users"])


def _activity_of(row: Mapping[str, Any]) -> str | None:
    users = row.get("users") or []
    if len(users) != 1:
        return None
    activity = users[0].get("activity")
    return activity if activity in ACTIVITIES else None


def build_cells(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Return activity -> group_id -> eligible single-person rows."""
    cells: dict[str, dict[int, list[dict[str, Any]]]] = {
        activity: defaultdict(list) for activity in ACTIVITIES
    }
    for row in rows:
        if _user_count(row) != 1:
            continue
        activity = _activity_of(row)
        if activity is None:
            continue
        cells[activity][int(row["group_id"])].append(dict(row))
    return cells


def select_single_person_pilot(
    manifest: Mapping[str, Mapping[str, Any]],
    test_ids: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
    per_cell: int = DEFAULT_PER_CELL,
    zero_person_count: int = DEFAULT_ZERO_PERSON,
) -> dict[str, Any]:
    """Select one balanced sample set per (activity, group) cell plus zeros.

    Raises ``ValueError`` when the split cannot satisfy the requested depth, so
    a shortfall is never silently absorbed by an unbalanced selection.
    """
    if per_cell < 1:
        raise ValueError("per_cell must be at least 1.")
    if zero_person_count < 1:
        raise ValueError("zero_person_count must be at least 1.")

    test_rows = []
    for sample_id in test_ids:
        row = manifest.get(sample_id)
        if row is None:
            raise KeyError(f"Test sample {sample_id!r} missing from manifest.")
        test_rows.append(row)

    cells = build_cells(test_rows)

    single_groups = sorted(
        {
            int(row["group_id"])
            for row in test_rows
            if _user_count(row) == 1
        }
    )
    if not single_groups:
        raise ValueError("Test split contains no single-person groups.")

    for activity in ACTIVITIES:
        for group_id in single_groups:
            available = len(cells[activity].get(group_id, []))
            if available < per_cell:
                raise ValueError(
                    f"Cell ({activity!r}, group {group_id}) has {available} "
                    f"samples, need {per_cell}."
                )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for activity in ACTIVITIES:
        for group_id in single_groups:
            candidates = sorted(
                cells[activity][group_id],
                key=lambda row: row["sample_id"],
            )
            chosen = rng.sample(candidates, per_cell)
            for row in chosen:
                selected.append(
                    {
                        "arm": "single_person",
                        "sample_id": row["sample_id"],
                        "group_id": int(row["group_id"]),
                        "sample_index": row.get("sample_index"),
                        "environment": row["environment"],
                        "wifi_band": str(row["wifi_band"]),
                        "number_of_users": 1,
                        "activity": activity,
                        "csi_path": row.get("csi_path"),
                    }
                )

    zero_candidates = sorted(
        (
            dict(row)
            for row in test_rows
            if _user_count(row) == 0
        ),
        key=lambda row: row["sample_id"],
    )
    if len(zero_candidates) < zero_person_count:
        raise ValueError(
            f"Only {len(zero_candidates)} zero-person samples available, "
            f"need {zero_person_count}."
        )

    zero_rng = random.Random(seed + 1)
    for row in zero_rng.sample(zero_candidates, zero_person_count):
        selected.append(
            {
                "arm": "zero_person",
                "sample_id": row["sample_id"],
                "group_id": int(row["group_id"]),
                "sample_index": row.get("sample_index"),
                "environment": row["environment"],
                "wifi_band": str(row["wifi_band"]),
                "number_of_users": 0,
                "activity": None,
                "csi_path": row.get("csi_path"),
            }
        )

    selected.sort(
        key=lambda row: (
            0 if row["arm"] == "single_person" else 1,
            row["activity"] or "",
            row["group_id"],
            row["sample_id"],
        )
    )
    return {
        "protocol": PROTOCOL,
        "seed": seed,
        "zero_person_seed": seed + 1,
        "per_activity_group_cell": per_cell,
        "single_person_expected": len(ACTIVITIES) * len(single_groups) * per_cell,
        "zero_person_expected": zero_person_count,
        "single_person_groups": single_groups,
        "samples": selected,
    }


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append(
        {"check": name, "passed": bool(passed), "detail": detail}
    )


def audit_selection(
    selection: Mapping[str, Any],
    manifest: Mapping[str, Mapping[str, Any]],
    test_ids: Sequence[str],
) -> dict[str, Any]:
    """Verify annotations, balance, arm separation and split membership."""
    samples = selection["samples"]
    by_id = {row["sample_id"]: row for row in samples}
    test_set = set(test_ids)
    per_cell = int(selection["per_activity_group_cell"])

    single = [row for row in samples if row["arm"] == "single_person"]
    zero = [row for row in samples if row["arm"] == "zero_person"]

    checks: list[dict[str, Any]] = []

    _check(
        checks,
        "unique_sample_ids",
        len(by_id) == len(samples),
        {"samples": len(samples), "unique": len(by_id)},
    )
    _check(
        checks,
        "all_samples_in_test_split",
        all(row["sample_id"] in test_set for row in samples),
        {
            "missing": sorted(
                row["sample_id"]
                for row in samples
                if row["sample_id"] not in test_set
            )
        },
    )
    _check(
        checks,
        "all_samples_in_manifest",
        all(row["sample_id"] in manifest for row in samples),
        {
            "missing": sorted(
                row["sample_id"]
                for row in samples
                if row["sample_id"] not in manifest
            )
        },
    )

    bad_single = []
    for row in single:
        source = manifest[row["sample_id"]]
        users = source.get("users") or []
        if (
            int(source["number_of_users"]) != 1
            or len(users) != 1
            or users[0].get("activity") != row["activity"]
            or row["activity"] not in ACTIVITIES
        ):
            bad_single.append(row["sample_id"])
    _check(
        checks,
        "single_person_annotations_match_manifest",
        not bad_single,
        {"invalid": bad_single},
    )

    bad_zero = []
    for row in zero:
        source = manifest[row["sample_id"]]
        if int(source["number_of_users"]) != 0 or (source.get("users") or []):
            bad_zero.append(row["sample_id"])
    _check(
        checks,
        "zero_person_annotations_match_manifest",
        not bad_zero,
        {"invalid": bad_zero},
    )

    _check(
        checks,
        "arms_disjoint",
        not ({row["sample_id"] for row in single}
             & {row["sample_id"] for row in zero}),
        {"single": len(single), "zero": len(zero)},
    )

    single_groups = sorted({row["group_id"] for row in single})
    activity_counts = Counter(row["activity"] for row in single)
    activity_groups = {
        activity: sorted(
            {row["group_id"] for row in single if row["activity"] == activity}
        )
        for activity in ACTIVITIES
    }
    cell_counts = Counter(
        (row["activity"], row["group_id"]) for row in single
    )
    group_counts = Counter(row["group_id"] for row in single)

    _check(
        checks,
        "per_activity_balance",
        all(
            activity_counts[activity] == len(single_groups) * per_cell
            for activity in ACTIVITIES
        ),
        {activity: activity_counts[activity] for activity in ACTIVITIES},
    )
    _check(
        checks,
        "per_activity_group_diversity",
        all(len(activity_groups[activity]) == len(single_groups)
            for activity in ACTIVITIES),
        {
            activity: len(activity_groups[activity])
            for activity in ACTIVITIES
        },
    )
    _check(
        checks,
        "per_cell_depth",
        all(
            cell_counts[(activity, group_id)] == per_cell
            for activity in ACTIVITIES
            for group_id in single_groups
        ),
        {"per_cell": per_cell, "cells": len(cell_counts)},
    )
    _check(
        checks,
        "group_usage_balance",
        set(group_counts.values()) == {len(ACTIVITIES) * per_cell},
        dict(sorted(group_counts.items())),
    )
    _check(
        checks,
        "single_arm_group_count",
        len(single_groups) == len(selection["single_person_groups"]),
        {"groups": single_groups},
    )
    _check(
        checks,
        "arm_totals",
        len(single) == int(selection["single_person_expected"])
        and len(zero) == int(selection["zero_person_expected"]),
        {
            "single": len(single),
            "single_expected": selection["single_person_expected"],
            "zero": len(zero),
            "zero_expected": selection["zero_person_expected"],
        },
    )

    zero_groups = sorted({row["group_id"] for row in zero})

    _check(
        checks,
        "zero_person_single_source_group",
        len(zero_groups) == 1,
        {"groups": zero_groups},
    )

    summary = {
        "total_samples": len(samples),
        "single_person_samples": len(single),
        "zero_person_samples": len(zero),
        "single_person_groups": single_groups,
        "zero_person_groups": zero_groups,
        "activities": {
            activity: {
                "samples": activity_counts[activity],
                "groups": activity_groups[activity],
            }
            for activity in ACTIVITIES
        },
        "environment": dict(
            sorted(Counter(row["environment"] for row in samples).items())
        ),
        "wifi_band": dict(
            sorted(Counter(row["wifi_band"] for row in samples).items())
        ),
    }

    failures = [
        check["check"] for check in checks if not check["passed"]
    ]
    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "summary": summary,
    }


def selection_sha256(selection: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        selection, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
