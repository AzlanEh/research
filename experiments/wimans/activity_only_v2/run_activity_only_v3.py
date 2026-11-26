#!/usr/bin/env python3
"""Run the WiMANS single-person, nine-activity, Physical-Model-Guided (v3) experiment.

Selection:
  * Uses the validated group-disjoint single-person selection (693 samples).
  * Supports balanced pilot mode (--pilot [N_PER_CLASS]).

Inference:
  * Ordered temporal windows from features_v2_activity_only.jsonl.
  * System prompt uses physical-model guidance (Doppler, transient vs continuous,
    postural transitions, localized gestures).
  * Outputs written to results/wimans/activity_only_v3/activity_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from common.llm.client import (  # noqa: E402
    LLMClientError,
    chat_completion,
    config_summary,
)
from llm.wimans.prompt_builder import ACTIVITY_LABELS  # noqa: E402
from llm.wimans.prompt_builder_activity_v3 import (  # noqa: E402
    build_activity_only_prompt_v3,
)
from llm.wimans.response_parser_activity import (  # noqa: E402
    parse_response_activity,
)

RESULTS_DIR = ROOT / "results/wimans/activity_only_v3"
GROUND_TRUTH_PATH = ROOT / "experiments/wimans/activity_only_v1/ground_truth.json"
FEATURES_PATH = ROOT / "data/processed/wimans/features_v3_pilot.jsonl"
RESULTS_PATH = RESULTS_DIR / "activity_results.jsonl"
EVALUATION_PATH = RESULTS_DIR / "activity_evaluation.json"

TEMPERATURE = 0.0
MAX_TOKENS = 512
DEFAULT_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(handle: Any, entry: dict[str, Any]) -> None:
    handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def select_samples(
    ground_truth: dict[str, Any],
    pilot_per_class: int | None = None,
) -> list[str]:
    if pilot_per_class is None:
        return sorted(ground_truth.keys())

    by_class: dict[str, list[str]] = defaultdict(list)
    for sid, meta in sorted(ground_truth.items()):
        act = meta["activity"]
        if len(by_class[act]) < pilot_per_class:
            by_class[act].append(sid)

    selected: list[str] = []
    for act in ACTIVITY_LABELS:
        sids = by_class.get(act, [])
        if len(sids) < pilot_per_class:
            raise ValueError(f"Not enough samples for class {act}: {len(sids)}")
        selected.extend(sids)
    return selected


def load_features(wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(FEATURES_PATH):
        sid = row["sample_id"]
        if sid in wanted_ids:
            features[sid] = row
    missing = wanted_ids - set(features)
    if missing:
        raise ValueError(f"Missing {len(missing)} feature records; first 5: {sorted(missing)[:5]}")
    return features


def classify_record(
    record: dict[str, Any],
    *,
    max_attempts: int,
) -> dict[str, Any]:
    messages = build_activity_only_prompt_v3(record)
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = chat_completion(
                messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            parsed = parse_response_activity(response.get("content", ""))
            return {
                **parsed.to_dict(),
                "model": response.get("model"),
                "latency_ms": response.get("latency_ms"),
                "token_usage": response.get("token_usage"),
                "raw_response": response.get("raw_response"),
                "attempts": attempt,
            }
        except LLMClientError as exc:
            last_error = str(exc)
            if attempt >= max_attempts:
                return {
                    "parse_status": "client_error",
                    "error": last_error,
                    "attempts": attempt,
                    "model": None,
                    "latency_ms": None,
                    "token_usage": None,
                    "raw_response": None,
                }
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        except Exception as exc:
            return {
                "parse_status": "client_error",
                "error": f"Unexpected: {exc}",
                "attempts": attempt,
                "model": None,
                "latency_ms": None,
                "token_usage": None,
                "raw_response": None,
            }

    return {
        "parse_status": "client_error",
        "error": last_error or "Unknown error",
        "attempts": max_attempts,
        "model": None,
        "latency_ms": None,
        "token_usage": None,
        "raw_response": None,
    }


def evaluate_predictions(
    results: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"error": "No results to evaluate"}

    correct = 0
    predictions_per_class: dict[str, int] = defaultdict(int)
    support_per_class: dict[str, int] = defaultdict(int)
    tp_per_class: dict[str, int] = defaultdict(int)
    fp_per_class: dict[str, int] = defaultdict(int)
    fn_per_class: dict[str, int] = defaultdict(int)

    for item in results:
        sid = item["sample_id"]
        true_act = ground_truth[sid]["activity"]
        pred_act = item.get("activity")
        status = item.get("parse_status")

        support_per_class[true_act] += 1
        if status == "ok" and pred_act:
            predictions_per_class[pred_act] += 1
            if pred_act == true_act:
                correct += 1
                tp_per_class[true_act] += 1
            else:
                fp_per_class[pred_act] += 1
                fn_per_class[true_act] += 1
        else:
            fn_per_class[true_act] += 1

    accuracy = correct / total
    per_class_metrics = {}
    f1s = []

    for act in ACTIVITY_LABELS:
        tp = tp_per_class[act]
        fp = fp_per_class[act]
        fn = fn_per_class[act]
        supp = support_per_class[act]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

        per_class_metrics[act] = {
            "support": supp,
            "predicted": predictions_per_class[act],
            "true_positive": tp,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "sample_count": total,
        "correct_predictions": correct,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "predictions_by_class": dict(predictions_per_class),
        "per_class_metrics": per_class_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WiMANS activity-only v3 experiment (Physical-Model Guided)")
    parser.add_argument("--pilot", type=int, default=None, help="Number of balanced samples per class (e.g. 1 or 2)")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and print summary without network calls")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth = load_json(GROUND_TRUTH_PATH)
    sample_ids = select_samples(ground_truth, pilot_per_class=args.pilot)
    features = load_features(set(sample_ids))

    print(f"Loaded {len(sample_ids)} target samples for inference.")
    if args.pilot:
        print(f"Pilot mode: {args.pilot} per class across 9 classes = {len(sample_ids)} total.")

    if args.dry_run:
        print("Dry run requested. Checking client config and first prompt:")
        print(config_summary())
        first_sid = sample_ids[0]
        sample_prompt = build_activity_only_prompt_v3(features[first_sid])
        print(f"Sample {first_sid} prompt built successfully ({len(sample_prompt[1]['content'])} chars).")
        print("\n--- SYSTEM PROMPT PREVIEW ---")
        print(sample_prompt[0]["content"][:800] + "...")
        return

    cfg = config_summary()
    if not cfg["api_key_configured"]:
        print("ERROR: LLM_API_KEY is not configured in environment.", file=sys.stderr)
        print("Set LLM_API_KEY to proceed with live inference.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting inference with model: {cfg['model']} at {cfg['api_base_url']}")
    results: list[dict[str, Any]] = []

    with RESULTS_PATH.open("w", encoding="utf-8") as out_f:
        for idx, sid in enumerate(sample_ids, 1):
            record = features[sid]
            true_act = ground_truth[sid]["activity"]
            print(f"[{idx}/{len(sample_ids)}] Classifying {sid} (True: {true_act})...", end="", flush=True)

            res = classify_record(record, max_attempts=args.max_attempts)
            entry = {
                "sample_id": sid,
                "ground_truth_activity": true_act,
                **res,
            }
            results.append(entry)
            append_jsonl(out_f, entry)

            pred_act = res.get("activity")
            status = res.get("parse_status")
            print(f" -> Pred: {pred_act} [{status}] (Confidence: {res.get('confidence')})")

    eval_report = evaluate_predictions(results, ground_truth)
    write_json(EVALUATION_PATH, eval_report)

    print("\n" + "=" * 60)
    print("INFERENCE AND EVALUATION COMPLETE (V3)")
    print(f"Accuracy: {eval_report['accuracy'] * 100:.2f}% ({eval_report['correct_predictions']}/{eval_report['sample_count']})")
    print(f"Macro F1: {eval_report['macro_f1']:.4f}")
    print("Predictions by class:", eval_report["predictions_by_class"])
    print(f"Results written to: {RESULTS_PATH}")
    print(f"Evaluation written to: {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
