#!/usr/bin/env python3
"""Four-request smoke test for the WiMANS single-person pipeline.

Two deterministic recordings (one activity sample, one zero-person sample) are
run through both feature representations: 2 samples x 2 versions = 4 requests.

The script repeats the offline prompt audits immediately before inference,
records every outcome incrementally, and never prints or persists the API key.
A technical failure in one request does not stop the remaining requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.llm.client import config_summary  # noqa: E402
from llm.wimans.classifier import ClassifierError  # noqa: E402
from llm.wimans.classifier_single import classify_record  # noqa: E402
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
V1_FEATURES = ROOT / "data/processed/wimans/features_v1.jsonl"
V2_FEATURES = ROOT / "data/processed/wimans/features_v2_single_person.jsonl"

OUTPUT_DIR = ROOT / "results/wimans/single_person_pilot"
JSONL_PATH = OUTPUT_DIR / "smoke_test.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "smoke_test_summary.json"

FEATURE_PATHS = {"v1": V1_FEATURES, "v2": V2_FEATURES}
BASELINE_BUILDERS = {
    "v1": build_zero_shot_prompt,
    "v2": build_zero_shot_prompt_v2,
}
VERSIONS = ("v1", "v2")

# Fixed, structural smoke selection: the first record of each arm in selection
# order. Not chosen from any model output.
SAMPLE_PLAN = (
    {
        "sample_id": "act_1_85",
        "arm": "single_person_activity",
        "expected_ground_truth": {"number_of_users": 1, "activity": "jump"},
    },
    {
        "sample_id": "act_203_40",
        "arm": "zero_person_detection",
        "expected_ground_truth": {"number_of_users": 0, "activity": None},
    },
)

# One fixed setting for all four requests.
TEMPERATURE = 0.0
MAX_TOKENS = 512


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def payload_text(messages: list[dict[str, str]]) -> str:
    body = messages[-1]["content"].split(INPUT_PREFIX, 1)[1]
    assert body.endswith(INPUT_SUFFIX)
    return body[: -len(INPUT_SUFFIX)]


def prompt_fingerprint(messages: list[dict[str, str]]) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def preflight() -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    """Re-run the offline audits for the four planned requests."""
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected = {row["sample_id"]: row for row in selection["records"]}
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

    problems: list[str] = []
    features: dict[str, dict[str, dict[str, Any]]] = {
        version: {row["sample_id"]: row for row in load_jsonl(path)}
        for version, path in FEATURE_PATHS.items()
    }

    for plan in SAMPLE_PLAN:
        sample_id = plan["sample_id"]

        if sample_id not in selected:
            problems.append(f"{sample_id}: not in selection manifest")
            continue
        if selected[sample_id]["arm"] != plan["arm"]:
            problems.append(
                f"{sample_id}: arm {selected[sample_id]['arm']!r} != "
                f"{plan['arm']!r}"
            )

        if ground_truth.get(sample_id) != plan["expected_ground_truth"]:
            problems.append(
                f"{sample_id}: ground truth {ground_truth.get(sample_id)} != "
                f"expected {plan['expected_ground_truth']}"
            )

        for version in VERSIONS:
            record = features[version].get(sample_id)
            if record is None:
                problems.append(
                    f"{sample_id}/{version}: missing from feature file"
                )
                continue

            messages = build_single_person_prompt(
                record, feature_version=version
            )
            problems.extend(
                f"{sample_id}/{version}: {problem}"
                for problem in audit_single_person_prompt(
                    messages, record, feature_version=version
                )
            )

            baseline = BASELINE_BUILDERS[version](record)
            if payload_text(messages) != payload_text(baseline):
                problems.append(
                    f"{sample_id}/{version}: feature block differs from "
                    "baseline"
                )

    return features, problems


def write_entry(handle: Any, entry: dict[str, Any]) -> None:
    handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def normalise_entry(
    plan: dict[str, Any],
    version: str,
    result: dict[str, Any],
    *,
    request_index: int,
    attempted: bool,
) -> dict[str, Any]:
    messages = result.get("prompt_messages") or []
    return {
        "request_index": request_index,
        "sample_id": plan["sample_id"],
        "arm": plan["arm"],
        "feature_version": version,
        "expected_ground_truth": plan["expected_ground_truth"],
        "group_id": result.get("group_id"),
        "environment": result.get("environment"),
        "wifi_band": result.get("wifi_band"),
        "prompt_version": result.get("prompt_version"),
        "prompt_sha256": prompt_fingerprint(messages) if messages else None,
        "prompt_chars": sum(len(m.get("content", "")) for m in messages),
        "request_attempted": attempted,
        "parse_status": result.get("parse_status"),
        "present": result.get("present"),
        "activity": result.get("activity"),
        "confidence": result.get("confidence"),
        "reason": result.get("reason"),
        "model": result.get("model"),
        "latency_ms": result.get("latency_ms"),
        "token_usage": result.get("token_usage"),
        "raw_response": result.get("raw_response"),
        "error": result.get("error"),
    }


def write_summary(
    entries: list[dict[str, Any]], client_config: Mapping[str, Any]
) -> int:
    checks = technical_checks(entries)
    technical_success = all(checks.values())

    observations = []
    for plan in SAMPLE_PLAN:
        for version in VERSIONS:
            entry = next(
                item for item in entries
                if item["sample_id"] == plan["sample_id"]
                and item["feature_version"] == version
            )
            if plan["expected_ground_truth"]["number_of_users"] == 0:
                observations.append(
                    {
                        "sample_id": plan["sample_id"],
                        "feature_version": version,
                        "predicted_present": entry["present"],
                        "expected_present": False,
                        "match": entry["present"] is False,
                    }
                )
            else:
                observations.append(
                    {
                        "sample_id": plan["sample_id"],
                        "feature_version": version,
                        "predicted_activity": entry["activity"],
                        "expected_activity": plan["expected_ground_truth"][
                            "activity"
                        ],
                        "match": entry["activity"]
                        == plan["expected_ground_truth"]["activity"],
                    }
                )

    summary = {
        "experiment": "wimans_single_person_smoke",
        "selection_manifest": str(SELECTION_PATH.relative_to(ROOT)),
        "ground_truth_file": str(GROUND_TRUTH_PATH.relative_to(ROOT)),
        "feature_files": {
            version: str(path.relative_to(ROOT))
            for version, path in FEATURE_PATHS.items()
        },
        "sample_ids": [plan["sample_id"] for plan in SAMPLE_PLAN],
        "requested": len(SAMPLE_PLAN) * len(VERSIONS),
        "completed": sum(1 for e in entries if e["request_attempted"]),
        "parsed_ok": sum(1 for e in entries if e["parse_status"] == "ok"),
        "client_errors": sum(
            1 for e in entries if e["parse_status"] == "client_error"
        ),
        "parse_failures": sum(
            1 for e in entries
            if e["parse_status"] not in {"ok", "client_error"}
        ),
        "settings": {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "model": client_config.get("model"),
            "api_key_configured": client_config.get("api_key_configured"),
        },
        "technical_checks": checks,
        "technical_success": technical_success,
        "accuracy_observations": observations,
        "accuracy_note": (
            "Descriptive only. Four responses cannot establish recognition "
            "accuracy; these fields do not affect technical success."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    print("\nTECHNICAL CHECKS")
    print("================")
    for name, passed in checks.items():
        print(f"{name:<42} {'PASS' if passed else 'FAIL'}")
    print()
    print("SMOKE TEST TECHNICAL SUCCESS:", technical_success)
    print("results:", JSONL_PATH.relative_to(ROOT))
    print("summary:", SUMMARY_PATH.relative_to(ROOT))
    return 0 if technical_success else 1


def technical_checks(entries: list[dict[str, Any]]) -> dict[str, Any]:
    def all_ok(predicate: Any) -> bool:
        return bool(entries) and all(predicate(entry) for entry in entries)

    present_boolean = all_ok(
        lambda entry: isinstance(entry.get("present"), bool)
    )

    def activity_consistent(entry: dict[str, Any]) -> bool:
        if entry.get("present") is True:
            return entry.get("activity") in ACTIVITY_LABELS
        if entry.get("present") is False:
            return entry.get("activity") is None
        return False

    by_version: dict[str, set[str]] = {}
    for entry in entries:
        by_version.setdefault(entry["feature_version"], set()).add(
            entry["sample_id"]
        )

    models = {
        entry["model"]
        for entry in entries
        if entry.get("parse_status") == "ok" and entry.get("model")
    }

    checks = {
        "four_requests_completed": (
            len(entries) == 4
            and all(entry.get("request_attempted") for entry in entries)
        ),
        "four_responses_parsed": all_ok(
            lambda entry: entry.get("parse_status") == "ok"
        ),
        "present_is_boolean": present_boolean,
        "activity_consistent_with_presence": all_ok(activity_consistent),
        "both_versions_valid_for_both_samples": (
            all_ok(lambda entry: entry.get("parse_status") == "ok")
            and len(by_version) == 2
            and by_version["v1"] == by_version["v2"]
            and len(by_version["v1"]) == 2
        ),
        "identical_sample_ids_across_versions": (
            set(by_version) == {"v1", "v2"}
            and by_version["v1"] == by_version["v2"]
        ),
        "consistent_model_across_requests": len(models) == 1,
    }
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Four-request single-person smoke test."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite previous smoke-test results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pre-flight audits only; make no requests or files.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Recompute the summary from an existing smoke_test.jsonl.",
    )
    args = parser.parse_args()

    features, problems = preflight()

    print("PRE-FLIGHT")
    print("==========")
    for plan in SAMPLE_PLAN:
        sid = plan["sample_id"]
        print(
            f"{sid:<12} arm={plan['arm']:<22} "
            f"expected={json.dumps(plan['expected_ground_truth'])}"
        )
    print(f"prompt audits + feature-block identity: "
          f"{'PASS' if not problems else 'FAIL'}")

    if problems:
        for problem in problems:
            print(f"  - {problem}")
        print("\nSMOKE TEST NOT STARTED: pre-flight failed.")
        return 1

    if args.dry_run:
        print("\nDRY RUN: no requests made, no files written.")
        return 0

    if args.summarize_only:
        if not JSONL_PATH.exists():
            print(f"\nNo existing {JSONL_PATH} to summarize.")
            return 1
        entries = load_jsonl(JSONL_PATH)
        return write_summary(entries, config_summary())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.force and (JSONL_PATH.exists() or SUMMARY_PATH.exists()):
        print(
            f"\nRefusing to overwrite existing results in {OUTPUT_DIR}. "
            "Pass --force to replace them."
        )
        return 2
    if args.force:
        for path in (JSONL_PATH, SUMMARY_PATH):
            if path.exists():
                path.unlink()

    client_config = config_summary()
    # config_summary reports settings only; it does not expose the key.
    print("\nCLIENT CONFIG")
    print("=============")
    print(
        f"model={client_config.get('model')} "
        f"temperature={TEMPERATURE} max_tokens={MAX_TOKENS} "
        f"api_key_configured={client_config.get('api_key_configured')}"
    )

    entries: list[dict[str, Any]] = []

    with JSONL_PATH.open("a", encoding="utf-8") as handle:
        request_index = 0
        for plan in SAMPLE_PLAN:
            for version in VERSIONS:
                request_index += 1
                record = features[version][plan["sample_id"]]

                print(
                    f"\n[{request_index}/4] {version} {plan['sample_id']} "
                    f"({plan['arm']})"
                )

                try:
                    result = classify_record(
                        record,
                        feature_version=version,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                except ClassifierError as exc:
                    result = {
                        "sample_id": plan["sample_id"],
                        "group_id": record.get("group_id"),
                        "environment": record.get("environment"),
                        "wifi_band": record.get("wifi_band"),
                        "prompt_version": None,
                        "parse_status": "client_error",
                        "present": None,
                        "activity": None,
                        "confidence": None,
                        "reason": None,
                        "model": None,
                        "latency_ms": None,
                        "token_usage": None,
                        "raw_response": None,
                        "error": str(exc),
                        "prompt_messages": build_single_person_prompt(
                            record, feature_version=version
                        ),
                    }
                except Exception as exc:  # noqa: BLE001 - record, keep going
                    result = {
                        "sample_id": plan["sample_id"],
                        "group_id": record.get("group_id"),
                        "environment": record.get("environment"),
                        "wifi_band": record.get("wifi_band"),
                        "prompt_version": None,
                        "parse_status": "unexpected_error",
                        "present": None,
                        "activity": None,
                        "confidence": None,
                        "reason": None,
                        "model": None,
                        "latency_ms": None,
                        "token_usage": None,
                        "raw_response": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "prompt_messages": build_single_person_prompt(
                            record, feature_version=version
                        ),
                    }

                entry = normalise_entry(
                    plan,
                    version,
                    result,
                    request_index=request_index,
                    attempted=True,
                )
                write_entry(handle, entry)
                entries.append(entry)

                print(
                    f"  status={entry['parse_status']} "
                    f"present={entry['present']} "
                    f"activity={entry['activity']} "
                    f"confidence={entry['confidence']} "
                    f"tokens={entry['token_usage']}"
                )
                if entry["error"]:
                    print(f"  error={entry['error']}")

    checks = technical_checks(entries)
    technical_success = all(checks.values())

    observations = []
    for plan in SAMPLE_PLAN:
        for version in VERSIONS:
            entry = next(
                item for item in entries
                if item["sample_id"] == plan["sample_id"]
                and item["feature_version"] == version
            )
            if plan["expected_ground_truth"]["number_of_users"] == 0:
                observations.append(
                    {
                        "sample_id": plan["sample_id"],
                        "feature_version": version,
                        "predicted_present": entry["present"],
                        "expected_present": False,
                        "match": entry["present"] is False,
                    }
                )
            else:
                observations.append(
                    {
                        "sample_id": plan["sample_id"],
                        "feature_version": version,
                        "predicted_activity": entry["activity"],
                        "expected_activity": plan["expected_ground_truth"][
                            "activity"
                        ],
                        "match": entry["activity"]
                        == plan["expected_ground_truth"]["activity"],
                    }
                )

    summary = {
        "experiment": "wimans_single_person_smoke",
        "selection_manifest": str(SELECTION_PATH.relative_to(ROOT)),
        "ground_truth_file": str(GROUND_TRUTH_PATH.relative_to(ROOT)),
        "feature_files": {
            version: str(path.relative_to(ROOT))
            for version, path in FEATURE_PATHS.items()
        },
        "sample_ids": [plan["sample_id"] for plan in SAMPLE_PLAN],
        "requested": len(SAMPLE_PLAN) * len(VERSIONS),
        "completed": sum(1 for e in entries if e["request_attempted"]),
        "parsed_ok": sum(1 for e in entries if e["parse_status"] == "ok"),
        "client_errors": sum(
            1 for e in entries if e["parse_status"] == "client_error"
        ),
        "parse_failures": sum(
            1 for e in entries
            if e["parse_status"] not in {"ok", "client_error"}
        ),
        "settings": {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "model": client_config.get("model"),
            "api_key_configured": client_config.get("api_key_configured"),
        },
        "technical_checks": checks,
        "technical_success": technical_success,
        "accuracy_observations": observations,
        "accuracy_note": (
            "Descriptive only. Four responses cannot establish recognition "
            "accuracy; these fields do not affect technical success."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    print("\nTECHNICAL CHECKS")
    print("================")
    for name, passed in checks.items():
        print(f"{name:<42} {'PASS' if passed else 'FAIL'}")
    print()
    print("SMOKE TEST TECHNICAL SUCCESS:", technical_success)
    print("results:", JSONL_PATH.relative_to(ROOT))
    print("summary:", SUMMARY_PATH.relative_to(ROOT))
    return 0 if technical_success else 1


if __name__ == "__main__":
    sys.exit(main())
