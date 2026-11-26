#!/usr/bin/env python
"""Dry-run validation for the 20-recording WiMANS v1/v2 pilot.

This script builds every v1 and every v2 prompt for the frozen pilot
selection, audits each prompt for ground-truth leakage, and confirms that the
prompts are accepted by their classifiers. It makes **no network calls**: the
classifier's configuration check and chat-completion call are replaced with
local stubs for the duration of the compatibility check.

Run from anywhere:

    ./.venv/bin/python experiments/wimans/dry_run_prompts.py

Artifacts:

    experiments/wimans/dry_run_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm.wimans import classifier, classifier_v2  # noqa: E402
from llm.wimans.prompt_builder import (  # noqa: E402
    ACTIVITY_LABELS,
    build_zero_shot_prompt,
)
from llm.wimans.prompt_builder_v2 import (  # noqa: E402
    build_zero_shot_prompt_v2,
)
from llm.wimans.prompt_validation import (  # noqa: E402
    audit_v1_prompt,
    audit_v2_prompt,
)

SELECTION_PATH = REPO_ROOT / "experiments/wimans/pilot_20_selection.json"
V1_PATH = REPO_ROOT / "data/processed/wimans/features_v1.jsonl"
V2_PILOT_PATH = REPO_ROOT / "data/processed/wimans/features_v2_pilot.jsonl"
REPORT_PATH = REPO_ROOT / "experiments/wimans/dry_run_report.json"

DRY_RUN_MODEL = "dry-run-model"

CANNED_COMPLETION = json.dumps(
    {
        "predicted_activity_counts": {label: 0 for label in ACTIVITY_LABELS},
        "confidence": 0.5,
        "reason": "Local dry-run stub; no model was called.",
    }
)


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records[row["sample_id"]] = row
    return records


def load_pilot_selection() -> dict[str, Any]:
    with SELECTION_PATH.open() as handle:
        return json.load(handle)


def build_and_audit_v1(
    record: dict[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    messages = build_zero_shot_prompt(record)
    repeat = build_zero_shot_prompt(record)
    problems = audit_v1_prompt(messages, record)
    if repeat != messages:
        problems.append("v1 prompt is not deterministic.")
    return messages, problems


def build_and_audit_v2(
    record: dict[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    messages = build_zero_shot_prompt_v2(record)
    repeat = build_zero_shot_prompt_v2(record)
    problems = audit_v2_prompt(messages, record)
    if repeat != messages:
        problems.append("v2 prompt is not deterministic.")
    return messages, problems


def _stub_chat_completion(
    captured: list[list[dict[str, str]]],
) -> Callable[..., dict[str, Any]]:
    def stub(
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        captured.append([dict(message) for message in messages])
        return {
            "content": CANNED_COMPLETION,
            "model": model or DRY_RUN_MODEL,
            "latency_ms": 0.0,
            "raw_response": {"stub": True},
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "finish_reason": "stop",
            "error": None,
        }

    return stub


def check_classifier(
    module: Any,
    record: dict[str, Any],
    expected_messages: list[dict[str, str]],
    expected_prompt_version: str,
) -> tuple[dict[str, Any], list[str]]:
    """Run a classifier against a stubbed client and validate its contract."""
    captured: list[list[dict[str, str]]] = []
    problems: list[str] = []

    with patch.object(module, "is_configured", return_value=True), patch.object(
        module,
        "chat_completion",
        side_effect=_stub_chat_completion(captured),
    ):
        result = module.classify_record(record, model=DRY_RUN_MODEL)

    if len(captured) != 1:
        problems.append(f"Expected one client call, got {len(captured)}.")
    elif captured[0] != expected_messages:
        problems.append("Classifier did not forward the built prompt unchanged.")

    if result.get("prompt_messages") != expected_messages:
        problems.append("Classifier result prompt_messages differ from the prompt.")

    if result.get("prompt_version") != expected_prompt_version:
        problems.append(
            "Classifier reported prompt_version "
            f"{result.get('prompt_version')!r}, expected "
            f"{expected_prompt_version!r}."
        )

    if result.get("parse_status") != "ok":
        problems.append(
            "Classifier result is not parseable: "
            f"{result.get('parse_status')!r} / {result.get('error')!r}."
        )

    counts = result.get("predicted_activity_counts")
    if not isinstance(counts, dict) or set(counts) != set(ACTIVITY_LABELS):
        problems.append("Classifier result lacks a full activity-count vector.")
    elif sum(counts.values()) > 5:
        problems.append("Classifier result exceeds the five-person limit.")

    return result, problems


def main() -> int:
    selection = load_pilot_selection()
    selected = selection["selected_samples"]
    sample_ids = [row["sample_id"] for row in selected]

    if len(sample_ids) != 20 or len(set(sample_ids)) != 20:
        raise SystemExit("Selection must contain 20 unique sample ids.")

    v1_records = load_jsonl(V1_PATH)
    v2_records = load_jsonl(V2_PILOT_PATH)

    missing_v1 = sorted(set(sample_ids) - set(v1_records))
    missing_v2 = sorted(set(sample_ids) - set(v2_records))
    extra_v2 = sorted(set(v2_records) - set(sample_ids))

    if missing_v1 or missing_v2 or extra_v2:
        raise SystemExit(
            "Feature/selection mismatch: "
            f"missing_v1={missing_v1}, missing_v2={missing_v2}, "
            f"extra_v2={extra_v2}"
        )

    records: list[dict[str, Any]] = []
    failures: list[str] = []

    for row in selected:
        sample_id = row["sample_id"]
        v1_record = v1_records[sample_id]
        v2_record = v2_records[sample_id]

        v1_messages, v1_problems = build_and_audit_v1(v1_record)
        v2_messages, v2_problems = build_and_audit_v2(v2_record)

        v1_result, v1_classifier_problems = check_classifier(
            classifier,
            v1_record,
            v1_messages,
            classifier.prompt_version(),
        )
        v2_result, v2_classifier_problems = check_classifier(
            classifier_v2,
            v2_record,
            v2_messages,
            classifier_v2.prompt_version_v2(),
        )

        v1_problems += v1_classifier_problems
        v2_problems += v2_classifier_problems

        v1_chars = len(v1_messages[1]["content"])
        v2_chars = len(v2_messages[1]["content"])

        if v1_problems:
            failures += [f"[v1] {sample_id}: {p}" for p in v1_problems]
        if v2_problems:
            failures += [f"[v2] {sample_id}: {p}" for p in v2_problems]

        records.append(
            {
                "sample_id": sample_id,
                "group_id": row["group_id"],
                "environment": row["environment"],
                "wifi_band": row["wifi_band"],
                "number_of_users": row["number_of_users"],
                "v1": {
                    "prompt_version": v1_result.get("prompt_version"),
                    "user_prompt_chars": v1_chars,
                    "approx_user_prompt_tokens": v1_chars // 4,
                    "problems": v1_problems,
                },
                "v2": {
                    "prompt_version": v2_result.get("prompt_version"),
                    "user_prompt_chars": v2_chars,
                    "approx_user_prompt_tokens": v2_chars // 4,
                    "problems": v2_problems,
                },
            }
        )

    v1_chars = [row["v1"]["user_prompt_chars"] for row in records]
    v2_chars = [row["v2"]["user_prompt_chars"] for row in records]

    report = {
        "protocol": "wimans_group_disjoint_20_recording_pilot",
        "dry_run": True,
        "network_calls": 0,
        "selection_path": str(SELECTION_PATH.relative_to(REPO_ROOT)),
        "v1_feature_path": str(V1_PATH.relative_to(REPO_ROOT)),
        "v2_feature_path": str(V2_PILOT_PATH.relative_to(REPO_ROOT)),
        "sample_count": len(records),
        "distinct_group_count": len({row["group_id"] for row in records}),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "prompt_chars": {
            "v1": {
                "min": min(v1_chars),
                "max": max(v1_chars),
                "mean": sum(v1_chars) / len(v1_chars),
            },
            "v2": {
                "min": min(v2_chars),
                "max": max(v2_chars),
                "mean": sum(v2_chars) / len(v2_chars),
            },
        },
        "samples": records,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    print("WiMANS pilot prompt dry run")
    print(f"  samples: {len(records)}")
    print(f"  distinct groups: {report['distinct_group_count']}")
    print("  network calls: 0 (client stubbed for compatibility check)")
    print()
    print(
        f"  {'sample':<14} {'group':>5} {'env':<13} {'band':<5} "
        f"{'v1 chars':>9} {'v2 chars':>9} {'status':>7}"
    )
    for row in records:
        status = "PASS" if not (
            row["v1"]["problems"] or row["v2"]["problems"]
        ) else "FAIL"
        print(
            f"  {row['sample_id']:<14} {row['group_id']:>5} "
            f"{row['environment']:<13} {row['wifi_band']:<5} "
            f"{row['v1']['user_prompt_chars']:>9} "
            f"{row['v2']['user_prompt_chars']:>9} {status:>7}"
        )

    print()
    print(f"  v1 user-prompt chars: min={report['prompt_chars']['v1']['min']} "
          f"max={report['prompt_chars']['v1']['max']} "
          f"mean={report['prompt_chars']['v1']['mean']:.1f}")
    print(f"  v2 user-prompt chars: min={report['prompt_chars']['v2']['min']} "
          f"max={report['prompt_chars']['v2']['max']} "
          f"mean={report['prompt_chars']['v2']['mean']:.1f}")
    print()
    print(f"  report: {REPORT_PATH.relative_to(REPO_ROOT)}")

    if failures:
        print()
        print(f"FAILED with {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print()
    print("PASS: prompts are leakage-free and accepted by both classifiers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
