"""Two-request smoke test for WiMANS prompt v3."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llm.wimans.classifier_v3 import classify_record
from llm.wimans.prompt_builder import build_zero_shot_prompt
from llm.wimans.prompt_builder_v2 import build_zero_shot_prompt_v2
from llm.wimans.prompt_builder_v3 import build_zero_shot_prompt_v3
from llm.wimans.prompt_validation_v3 import audit_v3_prompt


SAMPLE_ID = "act_17_42"

V1_PATH = ROOT / "data/processed/wimans/features_v1.jsonl"
V2_PATH = ROOT / "data/processed/wimans/features_v2_pilot.jsonl"

OUTPUT_DIR = ROOT / "results/wimans/prompt_v3_smoke"
OUTPUT_PATH = OUTPUT_DIR / "smoke_test.json"


def load_sample(path: Path, sample_id: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("sample_id") == sample_id:
                return record

    raise ValueError(f"{sample_id} not found in {path}")


def main() -> None:
    records = {
        "v1": load_sample(V1_PATH, SAMPLE_ID),
        "v2": load_sample(V2_PATH, SAMPLE_ID),
    }

    # Build original prompts so we can verify the v3 user payload is unchanged.
    original_builders = {
        "v1": build_zero_shot_prompt,
        "v2": build_zero_shot_prompt_v2,
    }

    prepared: dict[str, dict[str, Any]] = {}

    for version, record in records.items():
        original_messages = original_builders[version](record)
        v3_messages = build_zero_shot_prompt_v3(
            record,
            feature_version=version,
        )

        user_payload_unchanged = (
            original_messages[-1]["content"] == v3_messages[-1]["content"]
        )
        if not user_payload_unchanged:
            raise RuntimeError(
                f"{version}: v3 changed the original user-side CSI payload."
            )

        problems = audit_v3_prompt(
            v3_messages,
            record,
            feature_version=version,
        )
        if problems:
            raise RuntimeError(
                f"{version}: v3 prompt audit failed:\n- "
                + "\n- ".join(problems)
            )

        prepared[version] = {
            "record": record,
            "user_payload_unchanged": user_payload_unchanged,
        }

        print(
            f"{version}: payload unchanged = {user_payload_unchanged}; "
            "prompt audit = PASS"
        )

    results = []

    # Exactly one request for each feature representation.
    for version in ("v1", "v2"):
        print(f"\nCalling model with {version} features...")

        result = classify_record(
            prepared[version]["record"],
            feature_version=version,
            temperature=0.0,
            max_tokens=512,
        )

        counts = result.get("predicted_activity_counts")
        total_people = (
            sum(counts.values())
            if isinstance(counts, dict)
            else None
        )

        report = {
            "sample_id": SAMPLE_ID,
            "feature_version": version,
            "prompt_version": result["prompt_version"],
            "payload_unchanged": prepared[version]["user_payload_unchanged"],
            "parse_status": result["parse_status"],
            "predicted_activity_counts": counts,
            "predicted_people": total_people,
            "confidence": result.get("confidence"),
            "reason": result.get("reason"),
            "model": result.get("model"),
            "latency_ms": result.get("latency_ms"),
            "token_usage": result.get("token_usage"),
            "error": result.get("error"),
        }
        results.append(report)

        print(json.dumps(report, indent=2, ensure_ascii=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "wimans_prompt_v3_smoke",
        "sample_id": SAMPLE_ID,
        "network_calls_requested": 2,
        "results": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nSaved smoke-test report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
