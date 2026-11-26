"""Offline validation gate for the WiMANS single-person pilot.

Runs every check that can be made without a model call:

  * selection structure (70 unique IDs, 63 activity / 7 zero-person);
  * balance (seven samples per activity, one per activity/group cell);
  * ground truth separated from the inference manifest;
  * identical feature coverage in v1 and v2;
  * per-sample prompt audits for both feature versions;
  * byte-identical feature blocks versus each baseline prompt builder;
  * no selection-arm or sample-id information in any prompt.

Network access is blocked for the duration of the run; any attempt is recorded
and fails the gate. Exits nonzero if any check fails.
"""

from __future__ import annotations

import json
import re
import socket
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llm.wimans.prompt_builder import (  # noqa: E402
    ACTIVITY_LABELS,
    build_zero_shot_prompt,
)
from llm.wimans.prompt_builder_single import (  # noqa: E402
    build_single_person_prompt,
)
from llm.wimans.prompt_builder_v2 import build_zero_shot_prompt_v2  # noqa: E402
from llm.wimans.prompt_validation import (  # noqa: E402
    INPUT_PREFIX,
    INPUT_SUFFIX,
    extract_inference_input,
)
from llm.wimans.prompt_validation_single import (  # noqa: E402
    audit_single_person_prompt,
)

SELECTION_PATH = (
    ROOT / "experiments/wimans/single_person_pilot/selection_manifest.json"
)
GROUND_TRUTH_PATH = (
    ROOT / "experiments/wimans/single_person_pilot/ground_truth.json"
)
SOURCE_MANIFEST = ROOT / "data/processed/wimans/manifest.jsonl"
SPLIT_PATH = ROOT / "data/processed/wimans/splits/group_disjoint_80_20.json"
V1_FEATURES = ROOT / "data/processed/wimans/features_v1.jsonl"
V2_FEATURES = ROOT / "data/processed/wimans/features_v2_single_person.jsonl"

FEATURE_PATHS = {"v1": V1_FEATURES, "v2": V2_FEATURES}
BASELINE_BUILDERS = {
    "v1": build_zero_shot_prompt,
    "v2": build_zero_shot_prompt_v2,
}

ALLOWED_SELECTION_KEYS = frozenset(
    {"sample_id", "arm", "group_id", "environment", "wifi_band"}
)
GROUND_TRUTH_FIELDS = frozenset(
    {"number_of_users", "num_users", "users", "user", "activity", "activities",
     "label", "labels", "ground_truth"}
)
ARM_VALUES = (
    "single_person_activity",
    "zero_person_detection",
    "single_person",
    "zero_person",
)
ARM_PATTERNS = [
    re.compile(r"\barm\b", re.IGNORECASE),
    re.compile(r"\bground[_-]?truth\b", re.IGNORECASE),
] + [re.compile(re.escape(value), re.IGNORECASE) for value in ARM_VALUES]

NETWORK_ATTEMPTS: list[str] = []


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def payload_text(messages: list[dict[str, str]]) -> str:
    body = messages[-1]["content"].split(INPUT_PREFIX, 1)[1]
    assert body.endswith(INPUT_SUFFIX)
    return body[: -len(INPUT_SUFFIX)]


def check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def run_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    source = {row["sample_id"]: row for row in load_jsonl(SOURCE_MANIFEST)}
    test_ids = set(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["test"])

    records = selection["records"]
    selection_ids = [row["sample_id"] for row in records]
    selection_set = set(selection_ids)

    # 1. Exactly 70 unique IDs.
    check(
        checks,
        "selection_70_unique_ids",
        len(selection_ids) == 70 and len(selection_set) == 70,
        {"records": len(selection_ids), "unique": len(selection_set)},
    )

    # 2. Arm sizes.
    arm_counts = Counter(row["arm"] for row in records)
    check(
        checks,
        "arm_sizes_63_single_7_zero",
        arm_counts == {"single_person_activity": 63, "zero_person_detection": 7},
        dict(sorted(arm_counts.items())),
    )

    # 3. Seven samples per activity.
    activity_counts = Counter(
        ground_truth[sid]["activity"]
        for sid in selection_ids
        if ground_truth[sid]["number_of_users"] == 1
    )
    check(
        checks,
        "seven_samples_per_activity",
        set(activity_counts) == set(ACTIVITY_LABELS)
        and set(activity_counts.values()) == {7},
        dict(sorted(activity_counts.items())),
    )

    # 4. One sample per (activity, group) cell.
    cells = Counter(
        (row["group_id"], ground_truth[row["sample_id"]]["activity"])
        for row in records
        if row["arm"] == "single_person_activity"
    )
    single_groups = sorted({row["group_id"] for row in records
                            if row["arm"] == "single_person_activity"})
    expected_cells = len(ACTIVITY_LABELS) * len(single_groups)
    check(
        checks,
        "one_sample_per_activity_group_cell",
        len(cells) == expected_cells and set(cells.values()) == {1},
        {
            "groups": single_groups,
            "cells": len(cells),
            "expected_cells": expected_cells,
            "depths": sorted(set(cells.values())),
        },
    )

    # 5. Ground truth stays outside the inference manifest and matches source.
    selection_key_problems = [
        row["sample_id"] for row in records if set(row) - ALLOWED_SELECTION_KEYS
    ]
    gt_ids_match = set(ground_truth) == selection_set
    gt_mismatches = []
    for row in records:
        sid = row["sample_id"]
        source_row = source[sid]
        truth = ground_truth[sid]
        if row["arm"] == "single_person_activity":
            users = source_row.get("users") or []
            valid = (
                int(source_row["number_of_users"]) == 1
                and len(users) == 1
                and truth.get("number_of_users") == 1
                and truth.get("activity") == users[0].get("activity")
            )
        else:
            valid = (
                int(source_row["number_of_users"]) == 0
                and not (source_row.get("users") or [])
                and truth == {"number_of_users": 0, "activity": None}
            )
        if not valid:
            gt_mismatches.append(sid)

    leaked_fields = [
        sid
        for sid, truth in ground_truth.items()
        if truth.get("activity") is not None
        and truth["activity"] not in ACTIVITY_LABELS
    ]
    check(
        checks,
        "ground_truth_separated_and_valid",
        not selection_key_problems and gt_ids_match
        and not gt_mismatches and not leaked_fields,
        {
            "unexpected_selection_keys": selection_key_problems,
            "ground_truth_ids_match": gt_ids_match,
            "annotation_mismatches": gt_mismatches,
            "invalid_labels": leaked_fields,
        },
    )

    # 6. Both feature files cover exactly the selected IDs.
    features: dict[str, dict[str, dict[str, Any]]] = {}
    coverage_detail: dict[str, Any] = {}
    coverage_ok = True
    for version, path in FEATURE_PATHS.items():
        file_rows = load_jsonl(path)
        rows = [row for row in file_rows if row["sample_id"] in selection_set]
        ids = [row["sample_id"] for row in rows]
        index = {row["sample_id"]: row for row in rows}
        features[version] = index
        ok = (
            len(rows) == 70
            and len(set(ids)) == 70
            and set(ids) == selection_set
        )
        coverage_ok &= ok
        coverage_detail[version] = {
            "file_records": len(file_rows),
            "selected_records": len(rows),
            "unique": len(set(ids)),
            "missing": sorted(selection_set - set(ids)),
            "extra": sorted(set(ids) - selection_set),
        }
    check(checks, "feature_coverage_v1_equals_v2_equals_selection",
          coverage_ok, coverage_detail)

    # Build both prompt versions once for the remaining checks.
    prompts: dict[tuple[str, str], list[dict[str, str]]] = {}
    build_failures = []
    for version in FEATURE_PATHS:
        for sid in selection_ids:
            try:
                prompts[(version, sid)] = build_single_person_prompt(
                    features[version][sid], feature_version=version
                )
            except Exception as exc:  # noqa: BLE001 - report, do not abort
                build_failures.append((version, sid, str(exc)))
    check(
        checks,
        "prompts_build_for_all_140_version_sample_pairs",
        not build_failures,
        {"built": len(prompts), "failures": build_failures[:5]},
    )

    # 7. Every sample passes its single-person prompt audit.
    audit_failures = []
    for (version, sid), messages in prompts.items():
        problems = audit_single_person_prompt(
            messages, features[version][sid], feature_version=version
        )
        if problems:
            audit_failures.append((version, sid, problems[:3]))
    check(
        checks,
        "single_person_prompt_audit_all_140",
        not audit_failures,
        {"audited": len(prompts), "failures": audit_failures[:5]},
    )

    # 8. Feature blocks byte-identical to each baseline builder.
    identity_failures = []
    for (version, sid), messages in prompts.items():
        baseline = BASELINE_BUILDERS[version](features[version][sid])
        if payload_text(messages) != payload_text(baseline):
            identity_failures.append((version, sid))
    check(
        checks,
        "feature_block_byte_identical_to_baseline",
        not identity_failures,
        {"compared": len(prompts), "failures": identity_failures[:5]},
    )

    # 9. No arm, ground-truth value or sample-id information in any prompt.
    # The system prompt legitimately *warns* about ground truth, so the
    # concept phrase is only checked in the user message; arm values and the
    # word "arm" are checked everywhere.
    arm_only_patterns = [
        re.compile(r"\barm\b", re.IGNORECASE),
    ] + [re.compile(re.escape(value), re.IGNORECASE) for value in ARM_VALUES]
    leakage = []
    for (version, sid), messages in prompts.items():
        hits = []
        for message in messages:
            content = message["content"]
            for pattern in arm_only_patterns:
                if pattern.search(content):
                    hits.append(pattern.pattern)
        user_text = messages[-1]["content"]
        for pattern in ARM_PATTERNS:
            if pattern.search(user_text):
                hits.append(pattern.pattern)
        if sid in user_text:
            hits.append(f"sample_id:{sid}")
        if hits:
            leakage.append((version, sid, sorted(set(hits))))
    check(
        checks,
        "no_arm_ground_truth_or_sample_id_in_prompts",
        not leakage,
        {"scanned": len(prompts), "leaks": leakage[:5]},
    )

    # 10. No network access attempted during the whole gate.
    check(
        checks,
        "no_network_requests",
        not NETWORK_ATTEMPTS,
        {"attempts": list(NETWORK_ATTEMPTS)},
    )

    return checks


def _blocked_connect(self, *args: Any, **kwargs: Any) -> None:
    NETWORK_ATTEMPTS.append("socket.connect")
    raise RuntimeError("Network access attempted during offline validation.")


def _blocked_create_connection(*args: Any, **kwargs: Any) -> None:
    NETWORK_ATTEMPTS.append("socket.create_connection")
    raise RuntimeError("Network access attempted during offline validation.")


def main() -> int:
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection
    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.create_connection = _blocked_create_connection
    try:
        checks = run_checks()
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.create_connection = original_create_connection

    width = max(len(item["check"]) for item in checks)
    print("OFFLINE VALIDATION GATE")
    print("=======================")
    for index, item in enumerate(checks, 1):
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{index:>2}/{len(checks)} {item['check']:<{width}}  {status}")
        if not item["passed"]:
            print(f"     detail: {json.dumps(item['detail'])}")

    failures = [item["check"] for item in checks if not item["passed"]]
    print()
    if failures:
        print(f"OFFLINE VALIDATION FAILED — {len(failures)} CHECK(S): "
              + ", ".join(failures))
        return 1

    print(
        "ALL OFFLINE VALIDATIONS PASSED — "
        "70 RECORDINGS × 2 FEATURE VERSIONS"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
