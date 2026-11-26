#!/usr/bin/env python3
"""Paired 140-request runner for the WiMANS single-person pilot.

Runs both feature representations (v1, v2) over all 70 selected recordings:

    70 samples x 2 feature versions = 140 requests

The same sample IDs are used for v1 and v2, so every recording yields a
paired comparison of the two feature representations.

Interruption-safety design
--------------------------
- Each result is appended to ``paired_results.jsonl`` immediately after
  completion (fsync on every write).
- Resume skips only ``(sample_id, feature_version)`` pairs whose previous
  result has ``parse_status == "ok"``. Failed or unparsed requests are
  retried.
- Retries use a bounded policy (``--max-retries`` total attempts per request)
  with exponential backoff; only transient client errors are retried.
- Existing result files are never overwritten without ``--force``.

Evaluation (after all 140 requests finish)
------------------------------------------
Ground truth is loaded from ``ground_truth.json`` and joined only during
evaluation. It is never sent to the model.

Two separate evaluations are produced:

1. Presence detection (70 samples): accuracy, precision, recall, specificity,
   confusion matrix. Ground truth: number_of_users > 0.
2. Activity classification (63 single-person samples): accuracy, macro-F1,
   per-class precision/recall/F1, confusion matrix. Ground truth: the single
   annotated activity. Samples where the model fails to detect presence are
   counted as no-prediction errors so activity scores cannot hide
   presence-detection failures.

All seven zero-person samples come from group 203 (one environment, one band).
Presence results are therefore preliminary, not evidence of general zero-person
detection.

No model results or prompt wording are changed based on smoke-test outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.llm.client import config_summary  # noqa: E402
from llm.wimans.classifier import ClassifierError  # noqa: E402
from llm.wimans.classifier_single import classify_record  # noqa: E402
from llm.wimans.prompt_builder_single import build_single_person_prompt  # noqa: E402
from llm.wimans.prompt_validation_single import audit_single_person_prompt  # noqa: E402

SELECTION_PATH = (
    ROOT / "experiments/wimans/single_person_pilot/selection_manifest.json"
)
GROUND_TRUTH_PATH = (
    ROOT / "experiments/wimans/single_person_pilot/ground_truth.json"
)
V1_FEATURES = ROOT / "data/processed/wimans/features_v1.jsonl"
V2_FEATURES = ROOT / "data/processed/wimans/features_v2_single_person.jsonl"

OUTPUT_DIR = ROOT / "results/wimans/single_person_pilot"
RESULTS_PATH = OUTPUT_DIR / "paired_results.jsonl"
EVALUATION_PATH = OUTPUT_DIR / "paired_evaluation.json"

FEATURE_PATHS = {"v1": V1_FEATURES, "v2": V2_FEATURES}
VERSIONS = ("v1", "v2")

TEMPERATURE = 0.0
MAX_TOKENS = 512
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds; attempt_n sleeps base * 2^(n-1)

REQUESTS_EXPECTED = 140  # 70 samples x 2 versions


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_selection() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    return selection["records"], ground_truth


def load_features(
    selection_ids: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    selection_set = set(selection_ids)
    return {
        version: {
            row["sample_id"]: row
            for row in load_jsonl(path)
            if row["sample_id"] in selection_set
        }
        for version, path in FEATURE_PATHS.items()
    }


def verify_feature_coverage(
    selection_records: list[dict[str, Any]],
    features: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    problems: list[str] = []
    selection_ids = [row["sample_id"] for row in selection_records]
    selection_set = set(selection_ids)

    if len(selection_ids) != len(selection_set):
        problems.append(f"Duplicate sample IDs in selection: {len(selection_ids)}")

    for version in VERSIONS:
        file_ids = set(features[version])
        if file_ids != selection_set:
            missing = sorted(selection_set - file_ids)
            extra = sorted(file_ids - selection_set)
            problems.append(
                f"{version}: feature coverage mismatch "
                f"(missing={len(missing)}, extra={len(extra)})"
            )
            if missing:
                problems.append(f"  missing first 10: {missing[:10]}")
            if extra:
                problems.append(f"  extra first 10: {extra[:10]}")

    return problems


def result_key(sample_id: str, feature_version: str) -> tuple[str, str]:
    return (sample_id, feature_version)


def load_completed(results_path: Path) -> set[tuple[str, str]]:
    """Return (sample_id, version) pairs already completed with parse ok."""
    completed: set[tuple[str, str]] = set()
    if not results_path.exists():
        return completed
    for row in load_jsonl(results_path):
        if row.get("parse_status") == "ok":
            completed.add(result_key(row["sample_id"], row["feature_version"]))
    return completed


def fingerprint(result: dict[str, Any]) -> str:
    messages = result.get("prompt_messages") or []
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def append_result(handle: Any, entry: dict[str, Any]) -> None:
    handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def classify_with_retries(
    record: dict[str, Any],
    feature_version: str,
    *,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_BACKOFF_BASE,
) -> dict[str, Any]:
    """Classify one record, retrying transient client errors up to max_retries.

    Only ``client_error`` outcomes are retried; parse/schema errors are
    deterministic for a given input and are returned on the first attempt.

    The returned result carries ``_internal_attempts`` with the number of
    attempts made (1..max_retries).
    """
    last_result: dict[str, Any] | None = None
    attempts = 0

    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            result = classify_record(
                record,
                feature_version=feature_version,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except ClassifierError as exc:
            result = {
                "sample_id": record.get("sample_id"),
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
                    record, feature_version=feature_version
                ),
            }
        except Exception as exc:  # noqa: BLE001 - record and keep going
            result = {
                "sample_id": record.get("sample_id"),
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
                    record, feature_version=feature_version
                ),
            }

        last_result = result
        status = result.get("parse_status")
        if status == "ok":
            result["_internal_attempts"] = attempts
            return result

        if attempt < max_retries:
            sleep_for = retry_delay * (2 ** (attempt - 1))
            print(
                f"    retry {attempt}/{max_retries - 1} after {sleep_for:.1f}s "
                f"(status={status})"
            )
            time.sleep(sleep_for)

    if last_result is not None:
        last_result["_internal_attempts"] = attempts
    return last_result or {}


def run_experiment(
    *,
    selection_records: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    features: dict[str, dict[str, dict[str, Any]]],
    results_path: Path,
    max_retries: int,
    force: bool,
) -> list[dict[str, Any]]:
    """Run (or resume) all 140 requests, appending each result immediately."""
    if results_path.exists() and not force:
        raise SystemExit(
            f"{results_path} already exists; pass --force to overwrite."
        )

    # When forcing, remove the stale file so failed entries from a previous
    # run do not accumulate alongside the new results.
    if results_path.exists() and force:
        results_path.unlink()
        print(f"Removed stale results file: {results_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all previously written entries once for O(1) resume lookup.
    existing_entries: dict[tuple[str, str], dict[str, Any]] = {}
    if results_path.exists():
        for row in load_jsonl(results_path):
            key = result_key(row["sample_id"], row["feature_version"])
            existing_entries[key] = row

    completed = {
        key
        for key, row in existing_entries.items()
        if row.get("parse_status") == "ok"
    }
    print(
        f"Resume state: {len(completed)} of {REQUESTS_EXPECTED} requests "
        f"already completed successfully."
    )

    entries: list[dict[str, Any]] = []
    request_index = 0
    attempted = 0
    skipped = 0
    successes = 0
    failures = 0

    with results_path.open("a", encoding="utf-8") as handle:
        for record in selection_records:
            sample_id = record["sample_id"]
            arm = record["arm"]

            for version in VERSIONS:
                request_index += 1
                key = result_key(sample_id, version)
                feature_record = features[version][sample_id]

                if key in completed:
                    skipped += 1
                    entries.append(existing_entries[key])
                    continue

                attempted += 1
                print(
                    f"[{request_index:>3}/{REQUESTS_EXPECTED}] "
                    f"{version} {sample_id:<12} ({arm}) attempt 1/{max_retries}"
                )

                result = classify_with_retries(
                    feature_record,
                    version,
                    max_retries=max_retries,
                )

                internal_attempts = result.get("_internal_attempts", 1)
                entry = {
                    "request_index": request_index,
                    "sample_id": sample_id,
                    "arm": arm,
                    "feature_version": version,
                    "group_id": result.get("group_id"),
                    "environment": result.get("environment"),
                    "wifi_band": result.get("wifi_band"),
                    "prompt_version": result.get("prompt_version"),
                    "prompt_sha256": fingerprint(result),
                    "prompt_chars": sum(
                        len(m.get("content", ""))
                        for m in result.get("prompt_messages", [])
                    ),
                    "parse_status": result.get("parse_status"),
                    "present": result.get("present"),
                    "activity": result.get("activity"),
                    "confidence": result.get("confidence"),
                    "reason": result.get("reason"),
                    "model": result.get("model"),
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "latency_ms": result.get("latency_ms"),
                    "token_usage": result.get("token_usage"),
                    "raw_response": result.get("raw_response"),
                    "error": result.get("error"),
                    "retries_used": max(0, internal_attempts - 1),
                }

                # Remove internal bookkeeping before persisting.
                entry.pop("_internal_attempts", None)

                append_result(handle, entry)
                entries.append(entry)

                if result.get("parse_status") == "ok":
                    successes += 1
                    completed.add(key)
                else:
                    failures += 1

                print(
                    f"  status={entry['parse_status']} "
                    f"present={entry['present']} "
                    f"activity={entry['activity']} "
                    f"confidence={entry['confidence']} "
                    f"tokens={entry['token_usage']}"
                )
                if entry["error"]:
                    print(f"  error={entry['error'][:120]}")

    print()
    print(
        f"RUN COMPLETE: {successes} succeeded, {failures} failed, "
        f"{skipped} skipped (resumed), {attempted} newly attempted."
    )
    return entries


def compute_presence_metrics_per_version(
    entries: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    version: str,
) -> dict[str, Any]:
    """Presence detection for one feature version (70 samples)."""
    version_entries = [
        e
        for e in entries
        if e.get("feature_version") == version
        and e.get("parse_status") == "ok"
    ]

    y_true: list[bool] = []
    y_pred: list[bool] = []
    details: list[dict[str, Any]] = []

    for entry in version_entries:
        sample_id = entry["sample_id"]
        truth = ground_truth[sample_id]
        actual_present = truth["number_of_users"] > 0
        predicted_present = bool(entry.get("present"))

        y_true.append(actual_present)
        y_pred.append(predicted_present)
        details.append(
            {
                "sample_id": sample_id,
                "arm": entry["arm"],
                "actual_present": actual_present,
                "predicted_present": predicted_present,
                "correct": actual_present == predicted_present,
            }
        )

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "feature_version": version,
        "total_samples": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "actual_present_predicted_present": tp,
            "actual_present_predicted_absent": fn,
            "actual_absent_predicted_present": fp,
            "actual_absent_predicted_absent": tn,
        },
        "details": details,
    }


def compute_activity_metrics_per_version(
    entries: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    version: str,
) -> dict[str, Any]:
    """Activity classification for one feature version (63 single-person samples).

    Samples where the model fails to detect presence are counted as
    no-prediction errors so activity scores cannot hide presence failures.
    """
    version_entries = [
        e
        for e in entries
        if e.get("arm") == "single_person_activity"
        and e.get("feature_version") == version
        and e.get("parse_status") == "ok"
    ]

    y_true: list[str] = []
    y_pred: list[str | None] = []
    details: list[dict[str, Any]] = []

    for entry in version_entries:
        sample_id = entry["sample_id"]
        truth = ground_truth[sample_id]
        actual_activity = truth["activity"]
        predicted_present = bool(entry.get("present"))
        predicted_activity = entry.get("activity") if predicted_present else None

        y_true.append(actual_activity)
        y_pred.append(predicted_activity)
        details.append(
            {
                "sample_id": sample_id,
                "group_id": entry.get("group_id"),
                "actual_activity": actual_activity,
                "presence_detected": predicted_present,
                "predicted_activity": predicted_activity,
                "correct": predicted_present
                and predicted_activity == actual_activity,
            }
        )

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if p == t)
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float]] = {}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for t, p in zip(y_true, y_pred):
        confusion[t][p if p is not None else "__no_prediction__"] += 1

    for label in ACTIVITY_LABELS:
        tp = confusion[label].get(label, 0)
        fp = sum(
            confusion[other].get(label, 0)
            for other in confusion
            if other != label
        )
        fn = sum(
            v
            for k, v in confusion[label].items()
            if k != label
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": confusion[label].get(label, 0)
            + sum(
                v
                for k, v in confusion[label].items()
                if k != label
            ),
        }

    f1_values = [v["f1"] for v in per_class.values()]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    presence_correct = sum(
        1 for d in details if d["presence_detected"]
    )
    presence_recall_on_single = presence_correct / total if total else 0.0

    activity_correct_given_presence = sum(
        1 for d in details
        if d["presence_detected"] and d["predicted_activity"] == d["actual_activity"]
    )
    activity_accuracy_given_presence = (
        activity_correct_given_presence / presence_correct
        if presence_correct
        else 0.0
    )

    return {
        "feature_version": version,
        "total_single_person_samples": total,
        "correct_activity_predictions": correct,
        "accuracy_all_samples": round(accuracy, 4),
        "presence_recall_on_single_person": round(presence_recall_on_single, 4),
        "activity_accuracy_given_presence_detected": round(
            activity_accuracy_given_presence, 4
        ),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {
            actual: dict(pred_counts)
            for actual, pred_counts in sorted(confusion.items())
        },
        "details": details,
    }


def compute_paired_comparison(
    presence_v1: dict[str, Any],
    presence_v2: dict[str, Any],
    activity_v1: dict[str, Any],
    activity_v2: dict[str, Any],
) -> dict[str, Any]:
    """Compare v1 vs v2 per sample: which version got each prediction right."""
    v1_details = {d["sample_id"]: d for d in presence_v1["details"]}
    v2_details = {d["sample_id"]: d for d in presence_v2["details"]}

    v1_better_presence = 0
    v2_better_presence = 0
    both_correct_presence = 0
    both_wrong_presence = 0

    for sample_id in v1_details:
        d1 = v1_details[sample_id]
        d2 = v2_details[sample_id]
        c1 = d1["correct"]
        c2 = d2["correct"]
        if c1 and not c2:
            v1_better_presence += 1
        elif c2 and not c1:
            v2_better_presence += 1
        elif c1 and c2:
            both_correct_presence += 1
        else:
            both_wrong_presence += 1

    v1_activity_details = {d["sample_id"]: d for d in activity_v1["details"]}
    v2_activity_details = {d["sample_id"]: d for d in activity_v2["details"]}

    v1_better_activity = 0
    v2_better_activity = 0
    both_correct_activity = 0
    both_wrong_activity = 0

    for sample_id in v1_activity_details:
        d1 = v1_activity_details[sample_id]
        d2 = v2_activity_details[sample_id]
        c1 = d1["correct"]
        c2 = d2["correct"]
        if c1 and not c2:
            v1_better_activity += 1
        elif c2 and not c1:
            v2_better_activity += 1
        elif c1 and c2:
            both_correct_activity += 1
        else:
            both_wrong_activity += 1

    return {
        "presence": {
            "v1_better": v1_better_presence,
            "v2_better": v2_better_presence,
            "both_correct": both_correct_presence,
            "both_wrong": both_wrong_presence,
        },
        "activity": {
            "v1_better": v1_better_activity,
            "v2_better": v2_better_activity,
            "both_correct": both_correct_activity,
            "both_wrong": both_wrong_activity,
        },
    }


def write_evaluation(
    entries: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    client_config: Mapping[str, Any],
) -> dict[str, Any]:
    # Deduplicate by (sample_id, feature_version), keeping only parse OK entries.
    ok_entries: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        if entry.get("parse_status") != "ok":
            continue
        key = (entry["sample_id"], entry["feature_version"])
        ok_entries[key] = entry
    deduped = list(ok_entries.values())

    presence_by_version = {}
    activity_by_version = {}
    for version in VERSIONS:
        presence_by_version[version] = compute_presence_metrics_per_version(
            deduped, ground_truth, version
        )
        activity_by_version[version] = compute_activity_metrics_per_version(
            deduped, ground_truth, version
        )

    paired = compute_paired_comparison(
        presence_by_version["v1"],
        presence_by_version["v2"],
        activity_by_version["v1"],
        activity_by_version["v2"],
    )

    by_version: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "ok": 0}
    )
    for entry in deduped:
        v = entry["feature_version"]
        by_version[v]["total"] += 1
        if entry.get("parse_status") == "ok":
            by_version[v]["ok"] += 1

    token_totals = {"v1": 0, "v2": 0}
    latency_totals = {"v1": 0.0, "v2": 0.0}
    for entry in deduped:
        v = entry["feature_version"]
        tu = entry.get("token_usage") or {}
        token_totals[v] += tu.get("total_tokens", 0)
        latency_totals[v] += entry.get("latency_ms") or 0.0

    evaluation = {
        "experiment": "wimans_single_person_pilot",
        "stage": "paired_evaluation",
        "selection_manifest": str(SELECTION_PATH.relative_to(ROOT)),
        "ground_truth_file": str(GROUND_TRUTH_PATH.relative_to(ROOT)),
        "feature_files": {
            version: str(path.relative_to(ROOT))
            for version, path in FEATURE_PATHS.items()
        },
        "settings": {
            "model": client_config.get("model"),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "samples": len({e["sample_id"] for e in deduped}),
            "feature_versions": VERSIONS,
            "total_requests": len(deduped),
        },
        "coverage": {v: dict(c) for v, c in sorted(by_version.items())},
        "presence_detection_by_version": {
            v: {
                k: val
                for k, val in presence_by_version[v].items()
                if k != "details"
            }
            for v in VERSIONS
        },
        "activity_classification_by_version": {
            v: {
                k: val
                for k, val in activity_by_version[v].items()
                if k != "details"
            }
            for v in VERSIONS
        },
        "paired_comparison": paired,
        "cost": {
            "total_input_tokens_by_version": {
                v: sum(
                    (e.get("token_usage") or {}).get("prompt_tokens", 0)
                    for e in deduped
                    if e["feature_version"] == v
                )
                for v in VERSIONS
            },
            "total_output_tokens_by_version": {
                v: sum(
                    (e.get("token_usage") or {}).get("completion_tokens", 0)
                    for e in deduped
                    if e["feature_version"] == v
                )
                for v in VERSIONS
            },
            "total_latency_ms_by_version": {
                v: round(latency_totals[v], 2) for v in VERSIONS
            },
        },
        "limitation": (
            "All seven zero-person samples come from group 203 "
            "(meeting_room, 2.4 GHz). Presence results are preliminary and "
            "cannot establish cross-environment or cross-band generalization."
        ),
        "note": (
            "A technically successful response (parse_status == ok) is not a "
            "correct prediction. Accuracy is measured against ground truth "
            "joined only at evaluation time. Metrics are computed per feature "
            "version, not pooled, so v1 and v2 can be compared directly."
        ),
    }

    EVALUATION_PATH.write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return evaluation


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


def print_summary(evaluation: dict[str, Any]) -> None:
    presence_v1 = evaluation["presence_detection_by_version"]["v1"]
    presence_v2 = evaluation["presence_detection_by_version"]["v2"]
    activity_v1 = evaluation["activity_classification_by_version"]["v1"]
    activity_v2 = evaluation["activity_classification_by_version"]["v2"]
    paired = evaluation["paired_comparison"]
    cost = evaluation["cost"]
    coverage = evaluation["coverage"]

    print("\nPAIRED EVALUATION SUMMARY")
    print("=========================")
    print(f"Model: {evaluation['settings']['model']}")
    print(f"Settings: temperature={evaluation['settings']['temperature']}, "
          f"max_tokens={evaluation['settings']['max_tokens']}")
    print(f"Total requests: {evaluation['settings']['total_requests']}")
    print()
    print("COVERAGE")
    for version in VERSIONS:
        c = coverage.get(version, {})
        print(f"  {version}: {c.get('ok', 0)}/{c.get('total', 0)} parsed OK")
    print()
    print("PRESENCE DETECTION (70 samples per version)")
    print(f"  {'metric':<20} {'v1':>8} {'v2':>8}")
    print(f"  {'accuracy':<20} {presence_v1['accuracy']:>8} {presence_v2['accuracy']:>8}")
    print(f"  {'precision':<20} {presence_v1['precision']:>8} {presence_v2['precision']:>8}")
    print(f"  {'recall':<20} {presence_v1['recall']:>8} {presence_v2['recall']:>8}")
    print(f"  {'specificity':<20} {presence_v1['specificity']:>8} {presence_v2['specificity']:>8}")
    cm1 = presence_v1["confusion_matrix"]
    cm2 = presence_v2["confusion_matrix"]
    print(
        f"  {'confusion TP/FN/FP/TN':<20} "
        f"{cm1['actual_present_predicted_present']}/{cm1['actual_present_predicted_absent']}/"
        f"{cm1['actual_absent_predicted_present']}/{cm1['actual_absent_predicted_absent']}  |  "
        f"{cm2['actual_present_predicted_present']}/{cm2['actual_present_predicted_absent']}/"
        f"{cm2['actual_absent_predicted_present']}/{cm2['actual_absent_predicted_absent']}"
    )
    print()
    print("ACTIVITY CLASSIFICATION (63 single-person samples per version)")
    print(f"  {'metric':<20} {'v1':>8} {'v2':>8}")
    print(
        f"  {'accuracy_all':<20} {activity_v1['accuracy_all_samples']:>8} {activity_v2['accuracy_all_samples']:>8}"
    )
    print(
        f"  {'presence_recall':<20} {activity_v1['presence_recall_on_single_person']:>8} {activity_v2['presence_recall_on_single_person']:>8}"
    )
    print(
        f"  {'acc_given_presence':<20} {activity_v1['activity_accuracy_given_presence_detected']:>8} {activity_v2['activity_accuracy_given_presence_detected']:>8}"
    )
    print(f"  {'macro_f1':<20} {activity_v1['macro_f1']:>8} {activity_v2['macro_f1']:>8}")
    print()
    print("PAIRED COMPARISON (per sample: which version was correct?)")
    print(f"  Presence:   v1_better={paired['presence']['v1_better']}, "
          f"v2_better={paired['presence']['v2_better']}, "
          f"both_correct={paired['presence']['both_correct']}, "
          f"both_wrong={paired['presence']['both_wrong']}")
    print(f"  Activity:   v1_better={paired['activity']['v1_better']}, "
          f"v2_better={paired['activity']['v2_better']}, "
          f"both_correct={paired['activity']['both_correct']}, "
          f"both_wrong={paired['activity']['both_wrong']}")
    print()
    print("PER-CLASS METRICS")
    print(f"  {'label':<12} {'v1_f1':>8} {'v2_f1':>8} {'v1_support':>10} {'v2_support':>10}")
    for label in ACTIVITY_LABELS:
        m1 = activity_v1["per_class"].get(label, {})
        m2 = activity_v2["per_class"].get(label, {})
        print(
            f"  {label:<12} {m1.get('f1', 0):>8.4f} {m2.get('f1', 0):>8.4f} "
            f"{m1.get('support', 0):>10} {m2.get('support', 0):>10}"
        )
    print()
    print("COST")
    print(f"  v1 input tokens:  {cost['total_input_tokens_by_version']['v1']:,}")
    print(f"  v2 input tokens:  {cost['total_input_tokens_by_version']['v2']:,}")
    print(f"  v1 output tokens: {cost['total_output_tokens_by_version']['v1']:,}")
    print(f"  v2 output tokens: {cost['total_output_tokens_by_version']['v2']:,}")
    print(f"  v1 latency (ms):  {cost['total_latency_ms_by_version']['v1']:,}")
    print(f"  v2 latency (ms):  {cost['total_latency_ms_by_version']['v2']:,}")
    print()
    print("Evaluation written to:", EVALUATION_PATH.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired 140-request single-person pilot runner."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite previous paired results and evaluation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help="Total attempts per request (default: %(default)s).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=RETRY_BACKOFF_BASE,
        help="Base retry delay in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify coverage and resume state; make no API requests.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Recompute evaluation from existing paired_results.jsonl.",
    )
    args = parser.parse_args()

    selection_records, ground_truth = load_selection()
    selection_ids = [row["sample_id"] for row in selection_records]
    features = load_features(selection_ids)

    coverage_problems = verify_feature_coverage(selection_records, features)
    if coverage_problems:
        print("FEATURE COVERAGE PROBLEMS:")
        for problem in coverage_problems:
            print(f"  - {problem}")
        return 1

    print(f"Selection: {len(selection_records)} recordings "
          f"({sum(1 for r in selection_records if r['arm'] == 'single_person_activity')} "
          f"single-person, {sum(1 for r in selection_records if r['arm'] == 'zero_person_detection')} zero-person)")
    print(f"Feature coverage: v1={len(features['v1'])}, v2={len(features['v2'])}")
    print(f"Expected requests: {REQUESTS_EXPECTED}")

    if args.summarize_only:
        if not RESULTS_PATH.exists():
            print(f"\nNo existing {RESULTS_PATH} to summarize.")
            return 1
        entries = load_jsonl(RESULTS_PATH)
        client_config = config_summary()
        evaluation = write_evaluation(entries, ground_truth, client_config)
        print_summary(evaluation)
        return 0

    if args.dry_run:
        completed = load_completed(RESULTS_PATH)
        print(
            f"DRY RUN: {len(completed)}/{REQUESTS_EXPECTED} already completed; "
            f"{REQUESTS_EXPECTED - len(completed)} would be attempted."
        )
        return 0

    client_config = config_summary()
    print("\nCLIENT CONFIG")
    print("=============")
    print(
        f"model={client_config.get('model')} "
        f"temperature={TEMPERATURE} max_tokens={MAX_TOKENS} "
        f"api_key_configured={client_config.get('api_key_configured')}"
    )

    entries = run_experiment(
        selection_records=selection_records,
        ground_truth=ground_truth,
        features=features,
        results_path=RESULTS_PATH,
        max_retries=args.max_retries,
        force=args.force,
    )

    # Re-read all entries so resumed runs include previously written lines.
    all_entries = load_jsonl(RESULTS_PATH)
    evaluation = write_evaluation(all_entries, ground_truth, client_config)
    print_summary(evaluation)

    ok_count = sum(1 for e in all_entries if e.get("parse_status") == "ok")
    if ok_count != REQUESTS_EXPECTED:
        print(
            f"\nWARNING: {ok_count}/{REQUESTS_EXPECTED} requests parsed OK. "
            "Evaluation is based on available results only."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
