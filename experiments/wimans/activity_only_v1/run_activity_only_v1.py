#!/usr/bin/env python3
"""Run the WiMANS single-person, nine-activity, v1-only experiment.

Selection:
  * Existing group_disjoint_80_20.json test sample IDs only.
  * Exactly one person according to annotation.csv.
  * Activity is read from the activity slot matching the one populated location
    slot (user_1..user_6).
  * Expected selection: 693 recordings, 77 per activity. The script fails
    closed if these counts do not match.

Inference:
  * Existing features_v1.jsonl representation, unchanged.
  * Activity-only prompt: assumes one person is present.
  * No sample ID, group ID, user count, or ground-truth activity is sent.
  * One request per selected recording.

Usage:
  python experiments/wimans/run_activity_only_v1.py --dry-run
  python experiments/wimans/run_activity_only_v1.py
  python experiments/wimans/run_activity_only_v1.py --summarize-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from common.llm.client import (  # noqa: E402
    LLMClientError,
    chat_completion,
    config_summary,
    is_configured,
)
from llm.wimans.prompt_builder import build_zero_shot_prompt  # noqa: E402
from llm.wimans.prompt_validation import extract_inference_input  # noqa: E402
from llm.wimans.response_parser import _extract_json_object  # noqa: E402


SPLIT_PATH = ROOT / "data/processed/wimans/splits/group_disjoint_80_20.json"
ANNOTATION_PATH = ROOT / "data/metadata/wimans/annotation.csv"
MANIFEST_PATH = ROOT / "data/processed/wimans/manifest.jsonl"
FEATURES_PATH = ROOT / "data/processed/wimans/features_v1.jsonl"

EXPERIMENT_DIR = ROOT / "experiments/wimans/activity_only_v1"
RESULTS_DIR = ROOT / "results/wimans/activity_only_v1"
SELECTION_PATH = EXPERIMENT_DIR / "selection_manifest.json"
GROUND_TRUTH_PATH = EXPERIMENT_DIR / "ground_truth.json"
RESULTS_PATH = RESULTS_DIR / "activity_results.jsonl"
EVALUATION_PATH = RESULTS_DIR / "activity_evaluation.json"

ACTIVITY_LABELS = (
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
SLOTS = range(1, 7)
EXPECTED_SAMPLES = 693
EXPECTED_PER_CLASS = 77

TEMPERATURE = 0.0
MAX_TOKENS = 512
DEFAULT_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

SYSTEM_PROMPT = """
You are analyzing statistical features derived from Wi-Fi Channel State
Information (CSI) from a recording in which exactly one person is present.

TASK:
Classify the person's activity during the recording. Choose exactly one label
from the allowed activity labels.

ALLOWED ACTIVITY LABELS:
nothing, walk, rotation, jump, wave, lie_down, pick_up, sit_down, stand_up

LABEL SEMANTICS:
- "nothing" means one person is present but inactive or stationary.
- "nothing" does NOT mean an empty room or no person.
- Exactly one person is present. Do not perform person-presence detection.
- Return exactly one of the nine allowed activity labels.
- Use only the supplied CSI-derived features and permitted context.
- Do not invent identities, locations, trajectories, or numerical thresholds.
- CSI links are measurement links, not person identities.
- Temporal windows describe changes in the measured signal, not separate people.
- If evidence is ambiguous, choose the best-supported label and express
  uncertainty through confidence and reason.

Return exactly one JSON object with this schema:
{
  "activity": "walk",
  "confidence": 0.0,
  "reason": "Brief explanation based only on the supplied input."
}

Rules:
- "activity" must be exactly one allowed label.
- "confidence" must be a number from 0 to 1.
- "reason" must be a non-empty string.
Return only the JSON object. Do not include Markdown fences or other text.
""".strip()

TASK_INSTRUCTION = (
    "Exactly one person is present. Classify that person's activity from "
    "the following CSI-derived input. Remember that 'nothing' means the "
    "person is present but inactive, not that the room is empty."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def group_from_sample_id(sample_id: str) -> int:
    match = re.fullmatch(r"act_(\d+)_(\d+)", sample_id)
    if not match:
        raise ValueError(f"Unexpected WiMANS sample ID: {sample_id!r}")
    return int(match.group(1))


def activity_from_annotation_row(row: pd.Series) -> tuple[int, str]:
    occupied = [
        slot for slot in SLOTS
        if pd.notna(row[f"user_{slot}_location"])
    ]
    if len(occupied) != 1:
        raise ValueError(
            f"{row['label']}: expected exactly one populated location slot; "
            f"found {occupied}"
        )

    slot = occupied[0]
    activity = row[f"user_{slot}_activity"]
    if pd.isna(activity):
        raise ValueError(
            f"{row['label']}: user_{slot}_activity is missing"
        )
    activity = str(activity).strip()
    if activity not in ACTIVITY_LABELS:
        raise ValueError(
            f"{row['label']}: unexpected activity {activity!r}"
        )
    return slot, activity


def build_selection() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_ids = set(split["train"])
    test_ids = set(split["test"])

    if train_ids & test_ids:
        raise ValueError("Sample-level train/test leakage found.")

    train_groups = {group_from_sample_id(s) for s in train_ids}
    test_groups = {group_from_sample_id(s) for s in test_ids}
    overlap = train_groups & test_groups
    if overlap:
        raise ValueError(f"Group leakage in split: {sorted(overlap)}")

    manifest_rows = load_jsonl(MANIFEST_PATH)
    manifest = {row["sample_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Duplicate sample IDs found in manifest.")

    annotation = pd.read_csv(ANNOTATION_PATH)
    if annotation["label"].duplicated().any():
        raise ValueError("Duplicate sample IDs found in annotation.csv.")

    single = annotation[annotation["number_of_users"] == 1].copy()
    records: list[dict[str, Any]] = []
    ground_truth: dict[str, Any] = {}

    for _, row in single.iterrows():
        sample_id = str(row["label"])
        if sample_id not in test_ids:
            continue

        slot, activity = activity_from_annotation_row(row)
        manifest_row = manifest.get(sample_id)
        if manifest_row is None:
            raise ValueError(f"{sample_id}: missing from manifest.jsonl")

        parsed_group = group_from_sample_id(sample_id)
        manifest_group = int(manifest_row["group_id"])
        if parsed_group != manifest_group:
            raise ValueError(
                f"{sample_id}: ID group {parsed_group} != "
                f"manifest group {manifest_group}"
            )
        if int(manifest_row["number_of_users"]) != 1:
            raise ValueError(
                f"{sample_id}: annotation says one person but manifest "
                "number_of_users disagrees"
            )

        records.append(
            {
                "sample_id": sample_id,
                "group_id": parsed_group,
                "environment": manifest_row["environment"],
                "wifi_band": str(manifest_row["wifi_band"]),
                "occupied_slot": slot,
            }
        )
        ground_truth[sample_id] = {
            "activity": activity,
            "number_of_users": 1,
            "occupied_slot": slot,
            "group_id": parsed_group,
        }

    records.sort(key=lambda item: item["sample_id"])

    selected_ids = {item["sample_id"] for item in records}
    if not selected_ids <= test_ids:
        raise ValueError("Selection contains samples outside test split.")
    if selected_ids & train_ids:
        raise ValueError("Selected samples overlap the training split.")

    counts = {label: 0 for label in ACTIVITY_LABELS}
    for truth in ground_truth.values():
        counts[truth["activity"]] += 1

    if len(records) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} test single-person samples, "
            f"found {len(records)}."
        )
    if any(counts[label] != EXPECTED_PER_CLASS for label in ACTIVITY_LABELS):
        raise ValueError(
            f"Expected {EXPECTED_PER_CLASS} samples per class; got {counts}"
        )

    selection = {
        "experiment": "wimans_activity_only_v1",
        "split_file": str(SPLIT_PATH.relative_to(ROOT)),
        "annotation_file": str(ANNOTATION_PATH.relative_to(ROOT)),
        "selection_rule": (
            "test split AND number_of_users == 1; activity read from the "
            "activity column matching the sole populated location slot"
        ),
        "expected_samples": EXPECTED_SAMPLES,
        "activity_labels": list(ACTIVITY_LABELS),
        "activity_counts": counts,
        "records": records,
    }
    return records, ground_truth, selection


def freeze_selection(
    selection: dict[str, Any], ground_truth: dict[str, Any]
) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    if SELECTION_PATH.exists():
        existing = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        if canonical_hash(existing) != canonical_hash(selection):
            raise SystemExit(
                "Frozen selection differs from the current annotation/split. "
                "Inspect the changes before proceeding; do not silently "
                "replace the experiment selection."
            )
    else:
        write_json(SELECTION_PATH, selection)

    if GROUND_TRUTH_PATH.exists():
        existing_gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
        if canonical_hash(existing_gt) != canonical_hash(ground_truth):
            raise SystemExit(
                "Frozen ground truth differs from current annotation.csv. "
                "Inspect before changing the experiment."
            )
    else:
        write_json(GROUND_TRUTH_PATH, ground_truth)


def load_features(
    selection_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    wanted = {row["sample_id"] for row in selection_records}
    features: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(FEATURES_PATH):
        sample_id = row["sample_id"]
        if sample_id in wanted:
            if sample_id in features:
                raise ValueError(f"Duplicate feature record: {sample_id}")
            features[sample_id] = row

    missing = sorted(wanted - set(features))
    if missing:
        raise ValueError(
            f"Missing {len(missing)} v1 feature records; first 10: {missing[:10]}"
        )
    if set(features) != wanted:
        raise ValueError("Feature coverage does not exactly match selection.")

    for sample_id, row in features.items():
        if row.get("features", {}).get("feature_version") != "wimans_recording_v1":
            raise ValueError(f"{sample_id}: unexpected feature version.")
    return features


def build_activity_prompt(record: dict[str, Any]) -> list[dict[str, str]]:
    # Reuse the existing v1 baseline feature projection unchanged.
    baseline_messages = build_zero_shot_prompt(record)
    payload = extract_inference_input(baseline_messages)
    if set(payload) != {"environment", "wifi_band", "features"}:
        raise ValueError(
            "Unexpected baseline inference payload keys: "
            f"{sorted(payload)}"
        )

    user_content = (
        TASK_INSTRUCTION
        + "\n\nCSI-derived input:\n"
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n\nReturn the required JSON object only."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Defensive prompt-leakage checks. The only user payload is the feature
    # projection; sample/group IDs and labels are held outside this function.
    serialized_user = messages[1]["content"]
    forbidden = ("sample_id", "group_id", "number_of_users", "occupied_slot")
    if any(token in serialized_user for token in forbidden):
        raise ValueError("Forbidden evaluation metadata found in prompt.")
    return messages


def parse_activity_response(content: str) -> dict[str, Any]:
    if not content or not content.strip():
        return {"parse_status": "empty", "error": "Empty model response."}

    json_text = _extract_json_object(content)
    if json_text is None:
        return {"parse_status": "json_error", "error": "No JSON object found."}

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {"parse_status": "json_error", "error": str(exc)}

    if not isinstance(data, dict) or set(data) != {
        "activity", "confidence", "reason"
    }:
        return {
            "parse_status": "schema_error",
            "error": "Expected exactly activity, confidence, and reason.",
        }

    activity = data["activity"]
    confidence = data["confidence"]
    reason = data["reason"]

    if not isinstance(activity, str) or activity not in ACTIVITY_LABELS:
        return {
            "parse_status": "schema_error",
            "error": "activity is not one of the nine allowed labels.",
        }
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return {
            "parse_status": "value_error",
            "error": "confidence must be a number from 0 to 1.",
        }
    if not isinstance(reason, str) or not reason.strip():
        return {
            "parse_status": "schema_error",
            "error": "reason must be a non-empty string.",
        }

    return {
        "parse_status": "ok",
        "activity": activity,
        "confidence": float(confidence),
        "reason": reason.strip(),
        "error": None,
    }


def is_permanent_client_error(error: str) -> bool:
    text = error.lower()
    markers = (
        "400", "401", "403", "invalid api key", "please pass a valid api key",
        "unauthorized", "forbidden", "authentication",
    )
    return any(marker in text for marker in markers)


def is_transient_client_error(error: str) -> bool:
    text = error.lower()
    markers = (
        "429", "rate limit", "500", "502", "503", "504",
        "timeout", "timed out", "connection", "temporarily unavailable",
        "service unavailable",
    )
    return any(marker in text for marker in markers)


def fingerprint(messages: list[dict[str, str]]) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def append_jsonl(handle: Any, entry: dict[str, Any]) -> None:
    handle.write(
        json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def classify_with_retries(
    record: dict[str, Any],
    *,
    max_attempts: int,
) -> dict[str, Any]:
    messages = build_activity_prompt(record)
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = chat_completion(
                messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            parsed = parse_activity_response(response.get("content", ""))
            return {
                **parsed,
                "model": response.get("model"),
                "latency_ms": response.get("latency_ms"),
                "token_usage": response.get("token_usage"),
                "raw_response": response.get("raw_response"),
                "attempts": attempt,
                "prompt_messages": messages,
            }
        except LLMClientError as exc:
            last_error = str(exc)
            if (
                attempt >= max_attempts
                or is_permanent_client_error(last_error)
                or not is_transient_client_error(last_error)
            ):
                return {
                    "parse_status": "client_error",
                    "error": last_error,
                    "attempts": attempt,
                    "model": None,
                    "latency_ms": None,
                    "token_usage": None,
                    "raw_response": None,
                    "prompt_messages": messages,
                }
        except Exception as exc:  # preserve failure instead of losing the run
            return {
                "parse_status": "unexpected_error",
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": attempt,
                "model": None,
                "latency_ms": None,
                "token_usage": None,
                "raw_response": None,
                "prompt_messages": messages,
            }

        delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
        print(f"    transient error; retrying in {delay:.1f}s")
        time.sleep(delay)

    return {
        "parse_status": "client_error",
        "error": last_error or "Unknown client error",
        "attempts": max_attempts,
        "prompt_messages": messages,
    }


def load_latest_results(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for row in load_jsonl(path):
        latest[row["sample_id"]] = row
    return latest


def run_experiment(
    records: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    *,
    max_attempts: int,
    force: bool,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists() and force:
        RESULTS_PATH.unlink()
        print("Removed previous result JSONL (--force).")

    latest = load_latest_results(RESULTS_PATH)
    completed = {
        sample_id
        for sample_id, row in latest.items()
        if row.get("parse_status") == "ok"
    }
    print(
        f"Resume state: {len(completed)}/{len(records)} successful samples."
    )

    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        for index, selection in enumerate(records, start=1):
            sample_id = selection["sample_id"]
            if sample_id in completed:
                print(f"[{index:03d}/{len(records)}] skip {sample_id} (done)")
                continue

            print(f"[{index:03d}/{len(records)}] v1 {sample_id}")
            result = classify_with_retries(
                features[sample_id],
                max_attempts=max_attempts,
            )
            messages = result.get("prompt_messages", [])
            entry = {
                "request_index": index,
                "sample_id": sample_id,
                "group_id": selection["group_id"],
                "environment": selection["environment"],
                "wifi_band": selection["wifi_band"],
                "feature_version": "v1",
                "prompt_version": "wimans_activity_only_v1",
                "prompt_sha256": fingerprint(messages) if messages else None,
                "prompt_chars": sum(
                    len(message.get("content", "")) for message in messages
                ),
                "parse_status": result.get("parse_status"),
                "predicted_activity": result.get("activity"),
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
                "model": result.get("model"),
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "latency_ms": result.get("latency_ms"),
                "token_usage": result.get("token_usage"),
                "raw_response": result.get("raw_response"),
                "error": result.get("error"),
                "attempts": result.get("attempts", 1),
            }
            append_jsonl(handle, entry)
            print(
                f"    {entry['parse_status']}: "
                f"{entry['predicted_activity']} "
                f"confidence={entry['confidence']}"
            )
            if entry["error"]:
                print(f"    error: {str(entry['error'])[:200]}")


def calculate_metrics(
    records: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    results_path: Path,
) -> dict[str, Any]:
    latest = load_latest_results(results_path)
    confusion: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    group_correct: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    successful = 0
    missing_or_failed = 0
    details: list[dict[str, Any]] = []

    for record in records:
        sample_id = record["sample_id"]
        actual = ground_truth[sample_id]["activity"]
        result = latest.get(sample_id, {})
        parsed_ok = result.get("parse_status") == "ok"
        predicted = result.get("predicted_activity") if parsed_ok else None
        if predicted not in ACTIVITY_LABELS:
            predicted = None

        successful += int(parsed_ok)
        missing_or_failed += int(not parsed_ok)
        is_correct = predicted == actual
        correct += int(is_correct)
        group_id = int(record["group_id"])
        group_correct[group_id][0] += int(is_correct)
        group_correct[group_id][1] += 1

        confusion[actual][predicted or "__no_prediction__"] += 1
        details.append(
            {
                "sample_id": sample_id,
                "group_id": group_id,
                "actual_activity": actual,
                "predicted_activity": predicted,
                "parse_status": result.get("parse_status", "missing"),
                "correct": is_correct,
            }
        )

    per_class: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for label in ACTIVITY_LABELS:
        tp = confusion[label].get(label, 0)
        fp = sum(
            confusion[other].get(label, 0)
            for other in ACTIVITY_LABELS if other != label
        )
        fn = sum(confusion[label].values()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        f1s.append(f1)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values()),
        }

    total = len(records)
    evaluation = {
        "experiment": "wimans_activity_only_v1",
        "model": config_summary().get("model"),
        "feature_version": "v1",
        "prompt_version": "wimans_activity_only_v1",
        "settings": {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "total_selected_samples": total,
            "successful_parses": successful,
            "failed_or_missing_responses": missing_or_failed,
        },
        "metrics": {
            "accuracy": round(correct / total, 4) if total else 0.0,
            "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "correct_predictions": correct,
            "total_samples": total,
        },
        "per_class": per_class,
        "confusion_matrix": {
            actual: {
                prediction: counts.get(prediction, 0)
                for prediction in (*ACTIVITY_LABELS, "__no_prediction__")
                if counts.get(prediction, 0)
            }
            for actual, counts in sorted(confusion.items())
        },
        "group_accuracy": {
            str(group): {
                "correct": values[0],
                "total": values[1],
                "accuracy": round(values[0] / values[1], 4),
            }
            for group, values in sorted(group_correct.items())
        },
        "details": details,
        "evaluation_note": (
            "All selected samples are from the existing group-disjoint test "
            "split and have exactly one annotated person. Failed or missing "
            "model responses count as incorrect predictions."
        ),
    }
    write_json(EVALUATION_PATH, evaluation)
    return evaluation


def print_summary(evaluation: dict[str, Any]) -> None:
    metrics = evaluation["metrics"]
    print("\nACTIVITY-ONLY V1 EVALUATION")
    print("==========================")
    print("Model:", evaluation["model"])
    print("Samples:", metrics["total_samples"])
    print("Successful parses:", evaluation["settings"]["successful_parses"])
    print(
        "Failed/missing:",
        evaluation["settings"]["failed_or_missing_responses"],
    )
    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("\nPER-CLASS")
    print(f"{'activity':<12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>9}")
    for label in ACTIVITY_LABELS:
        item = evaluation["per_class"][label]
        print(
            f"{label:<12} {item['precision']:>10.4f} "
            f"{item['recall']:>10.4f} {item['f1']:>10.4f} "
            f"{item['support']:>9}"
        )
    print("\nCONFUSION MATRIX (rows=actual, columns=predicted)")
    print(json.dumps(evaluation["confusion_matrix"], indent=2))
    print("\nGROUP ACCURACY")
    print(json.dumps(evaluation["group_accuracy"], indent=2))
    print("\nSaved:", EVALUATION_PATH.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WiMANS group-disjoint, single-person activity-only v1."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selection/features/prompts without API calls.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Recompute metrics from existing activity_results.jsonl.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing results JSONL and rerun all samples.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum attempts for transient request failures (default 3).",
    )
    args = parser.parse_args()

    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    records, ground_truth, selection = build_selection()
    freeze_selection(selection, ground_truth)
    features = load_features(records)

    # Build every prompt offline before any request is made.
    for selection_record in records:
        sample_id = selection_record["sample_id"]
        messages = build_activity_prompt(features[sample_id])
        if len(messages) != 2:
            raise ValueError(f"{sample_id}: unexpected prompt structure")

    print("OFFLINE VALIDATION PASSED")
    print("Selected recordings:", len(records))
    print("Activity counts:", selection["activity_counts"])
    print("Groups in test selection:", len({r["group_id"] for r in records}))
    print("Feature coverage:", len(features), "/", len(records))
    print("Prompt audit: all prompts built; IDs and ground truth excluded")
    print("Selection SHA256:", canonical_hash(selection))

    if args.summarize_only:
        if not RESULTS_PATH.exists():
            print("No result file found:", RESULTS_PATH)
            return 1
        evaluation = calculate_metrics(records, ground_truth, RESULTS_PATH)
        print_summary(evaluation)
        return 0

    if args.dry_run:
        latest = load_latest_results(RESULTS_PATH)
        completed = sum(
            1 for sample_id, row in latest.items()
            if row.get("parse_status") == "ok"
        )
        print(
            f"DRY RUN: {completed}/{len(records)} already successful; "
            f"{len(records) - completed} remain."
        )
        print("No API requests were made.")
        return 0

    client_config = config_summary()
    print("\nCLIENT CONFIG")
    print("Model:", client_config.get("model"))
    print("API key configured:", client_config.get("api_key_configured"))
    if not is_configured():
        print("LLM client is not configured. Set LLM_API_KEY.")
        return 1

    run_experiment(
        records,
        features,
        max_attempts=args.max_attempts,
        force=args.force,
    )
    evaluation = calculate_metrics(records, ground_truth, RESULTS_PATH)
    print_summary(evaluation)

    if evaluation["settings"]["successful_parses"] != len(records):
        print(
            "\nWARNING: Not every sample has a successful parsed response. "
            "Fix the reported failures and rerun to resume."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
