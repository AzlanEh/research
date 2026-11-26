"""Offline audit of all WiMANS activity-only v2 prompts.

This script builds and validates prompts locally. It never calls an LLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm.wimans.prompt_builder_activity_v2 import (
    build_activity_only_prompt_v2,
)
from llm.wimans.prompt_validation import extract_inference_input
from llm.wimans.prompt_validation_activity import (
    audit_activity_only_prompt,
)


DEFAULT_INPUT = Path("data/processed/wimans/features_v2_activity_only.jsonl")
DEFAULT_REPORT = Path(
    "results/wimans/activity_only_v2/prompt_audit.json"
)


def audit_file(input_path: Path) -> dict[str, Any]:
    total = 0
    unique_ids: set[str] = set()
    duplicate_ids: list[str] = []
    missing_ids = 0
    prompt_failures: list[dict[str, Any]] = []

    total_windows = 0
    total_partial_windows = 0
    total_null_deltas = 0
    records_with_null_deltas = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            total += 1

            try:
                record = json.loads(line)
                sample_id = record.get("sample_id")

                if not sample_id:
                    missing_ids += 1
                else:
                    sample_id = str(sample_id)
                    if sample_id in unique_ids:
                        duplicate_ids.append(sample_id)
                    unique_ids.add(sample_id)

                messages = build_activity_only_prompt_v2(record)
                problems = audit_activity_only_prompt(messages, record)

                payload = extract_inference_input(messages)
                sequence = payload["features"]["temporal_sequence"]

                total_windows += len(sequence)
                record_null_deltas = 0

                for window in sequence:
                    if window[4]:
                        total_partial_windows += 1

                    for delta in window[7]:
                        if delta is None:
                            record_null_deltas += 1

                total_null_deltas += record_null_deltas
                if record_null_deltas:
                    records_with_null_deltas += 1

                if problems:
                    prompt_failures.append({
                        "line": line_number,
                        "sample_id": sample_id,
                        "problems": problems,
                    })

            except Exception as exc:
                prompt_failures.append({
                    "line": line_number,
                    "sample_id": None,
                    "problems": [
                        f"{type(exc).__name__}: {exc}"
                    ],
                })

    return {
        "input_file": str(input_path),
        "total_records": total,
        "unique_sample_ids": len(unique_ids),
        "duplicate_sample_ids": sorted(set(duplicate_ids)),
        "missing_sample_ids": missing_ids,
        "total_temporal_windows": total_windows,
        "total_partial_windows": total_partial_windows,
        "total_null_temporal_deltas": total_null_deltas,
        "records_with_null_temporal_deltas": records_with_null_deltas,
        "prompt_failures": prompt_failures,
        "passed": (
            total > 0
            and len(unique_ids) == total
            and not duplicate_ids
            and missing_ids == 0
            and not prompt_failures
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit WiMANS activity-only v2 prompts offline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--expected-records",
        type=int,
        default=693,
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    report = audit_file(args.input)

    report["expected_records"] = args.expected_records
    report["record_count_matches_expected"] = (
        report["total_records"] == args.expected_records
    )
    report["passed"] = (
        report["passed"] and report["record_count_matches_expected"]
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Input: {report['input_file']}")
    print(f"Records: {report['total_records']}")
    print(f"Unique sample IDs: {report['unique_sample_ids']}")
    print(f"Temporal windows: {report['total_temporal_windows']}")
    print(f"Partial windows: {report['total_partial_windows']}")
    print(f"Null temporal deltas: {report['total_null_temporal_deltas']}")
    print(
        "Records with null temporal deltas: "
        f"{report['records_with_null_temporal_deltas']}"
    )
    print(f"Prompt failures: {len(report['prompt_failures'])}")
    print(f"Report: {args.report}")
    print(f"Status: {'PASS' if report['passed'] else 'FAIL'}")

    if report["prompt_failures"]:
        print("\nFirst 10 failures:")
        for failure in report["prompt_failures"][:10]:
            print(json.dumps(failure, indent=2))

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
