#!/usr/bin/env python3
"""Run the WiMANS single-person, nine-activity, Subcarrier + Doppler (v5) experiment."""

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

from common.llm.client import LLMClientError, chat_completion, config_summary  # noqa: E402
from llm.wimans.prompt_builder import ACTIVITY_LABELS  # noqa: E402
from llm.wimans.prompt_builder_activity_v5 import (  # noqa: E402
    build_activity_only_prompt_v5,
    prompt_version_activity_v5,
)
from llm.wimans.response_parser_activity import parse_response_activity  # noqa: E402

RESULTS_DIR = ROOT / "results/wimans/activity_only_v5"
GROUND_TRUTH_PATH = ROOT / "experiments/wimans/activity_only_v1/ground_truth.json"
PILOT_FEATURES_PATH = ROOT / "data/processed/wimans/features_v5_pilot.jsonl"
FULL_FEATURES_PATH = ROOT / "data/processed/wimans/features_v5_activity_only.jsonl"
RESULTS_PATH = RESULTS_DIR / "activity_results.jsonl"
EVALUATION_PATH = RESULTS_DIR / "activity_evaluation.json"

TEMPERATURE = 0.0
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
DEFAULT_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(handle: Any, entry: dict[str, Any]) -> None:
    handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def select_samples(ground_truth: dict[str, Any], pilot_per_class: int | None = None) -> list[str]:
    if pilot_per_class is None:
        return sorted(ground_truth.keys())
    by_class: dict[str, list[str]] = defaultdict(list)
    for sid, meta in sorted(ground_truth.items()):
        if len(by_class[meta["activity"]]) < pilot_per_class:
            by_class[meta["activity"]].append(sid)
    selected: list[str] = []
    for act in ACTIVITY_LABELS:
        sids = by_class.get(act, [])
        if len(sids) < pilot_per_class:
            raise ValueError(f"Not enough samples for class {act}: {len(sids)}")
        selected.extend(sids)
    return selected


def load_features(wanted_ids: set[str], use_full: bool) -> dict[str, dict[str, Any]]:
    path = FULL_FEATURES_PATH if (use_full or not PILOT_FEATURES_PATH.exists()) else PILOT_FEATURES_PATH
    features: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row["sample_id"] in wanted_ids:
            features[row["sample_id"]] = row
    missing = wanted_ids - set(features)
    if missing and path != FULL_FEATURES_PATH and FULL_FEATURES_PATH.exists():
        for row in load_jsonl(FULL_FEATURES_PATH):
            if row["sample_id"] in missing:
                features[row["sample_id"]] = row
        missing = wanted_ids - set(features)
    if missing:
        raise ValueError(f"Missing {len(missing)} V5 feature records; first 5: {sorted(missing)[:5]}")
    return features


def classify_record(record: dict[str, Any], *, max_attempts: int) -> dict[str, Any]:
    messages = build_activity_only_prompt_v5(record)
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = chat_completion(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            parsed = parse_response_activity(response.get("content", ""))
            return {
                **parsed.to_dict(),
                "model": response.get("model"),
                "latency_ms": response.get("latency_ms"),
                "token_usage": response.get("token_usage"),
                "raw_response": response.get("raw_response"),
                "attempts": attempt,
                "prompt_version": prompt_version_activity_v5(),
            }
        except LLMClientError as exc:
            last_error = str(exc)
            if attempt >= max_attempts:
                return {"parse_status": "client_error", "error": last_error, "attempts": attempt}
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        except Exception as exc:
            return {"parse_status": "client_error", "error": f"Unexpected: {exc}", "attempts": attempt}
    return {"parse_status": "client_error", "error": last_error or "Unknown error", "attempts": max_attempts}


def evaluate_predictions(results: list[dict[str, Any]], ground_truth: dict[str, Any]) -> dict[str, Any]:
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
        support_per_class[true_act] += 1
        if item.get("parse_status") == "ok" and pred_act:
            predictions_per_class[pred_act] += 1
            if pred_act == true_act:
                correct += 1
                tp_per_class[true_act] += 1
            else:
                fp_per_class[pred_act] += 1
                fn_per_class[true_act] += 1
        else:
            fn_per_class[true_act] += 1

    f1s = []
    per_class_metrics = {}
    for act in ACTIVITY_LABELS:
        tp, fp, fn = tp_per_class[act], fp_per_class[act], fn_per_class[act]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        per_class_metrics[act] = {
            "support": support_per_class[act],
            "predicted": predictions_per_class[act],
            "true_positive": tp,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    return {
        "sample_count": total,
        "correct_predictions": correct,
        "accuracy": round(correct / total, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "predictions_by_class": dict(predictions_per_class),
        "per_class_metrics": per_class_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WiMANS activity-only v5 experiment (Subcarrier + Doppler)")
    parser.add_argument("--pilot", type=int, default=None, help="Balanced samples per class (e.g. 2 or 5)")
    parser.add_argument("--full", action="store_true", help="Load full 693-sample V5 features")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts without network calls")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth = load_json(GROUND_TRUTH_PATH)
    sample_ids = select_samples(ground_truth, pilot_per_class=args.pilot)
    features = load_features(set(sample_ids), use_full=args.full)

    print(f"Loaded {len(sample_ids)} target samples for inference.")
    if args.dry_run:
        print(config_summary())
        sample_prompt = build_activity_only_prompt_v5(features[sample_ids[0]])
        total_chars = sum(len(m["content"]) for m in sample_prompt)
        print(f"Sample {sample_ids[0]} prompt built ({total_chars} chars, ~{total_chars // 4} tokens).")
        return

    cfg = config_summary()
    if not cfg["api_key_configured"]:
        print("ERROR: LLM_API_KEY is not configured in environment.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting inference with model: {cfg['model']} at {cfg['api_base_url']}")
    print(f"Prompt version: {prompt_version_activity_v5()}")
    results: list[dict[str, Any]] = []

    with RESULTS_PATH.open("w", encoding="utf-8") as out_f:
        for idx, sid in enumerate(sample_ids, 1):
            true_act = ground_truth[sid]["activity"]
            print(f"[{idx}/{len(sample_ids)}] Classifying {sid} (True: {true_act})...", end="", flush=True)
            res = classify_record(features[sid], max_attempts=args.max_attempts)
            entry = {"sample_id": sid, "ground_truth_activity": true_act, **res}
            results.append(entry)
            append_jsonl(out_f, entry)
            print(f" -> Pred: {res.get('activity')} [{res.get('parse_status')}] (Conf: {res.get('confidence')}, Runner-up: {res.get('runner_up', '-')})")

    eval_report = evaluate_predictions(results, ground_truth)
    write_json(EVALUATION_PATH, eval_report)
    print("\n" + "=" * 60)
    print("INFERENCE AND EVALUATION COMPLETE (V5)")
    print(f"Accuracy: {eval_report['accuracy'] * 100:.2f}% ({eval_report['correct_predictions']}/{eval_report['sample_count']})")
    print(f"Macro F1: {eval_report['macro_f1']:.4f}")
    print("Predictions by class:", eval_report["predictions_by_class"])


if __name__ == "__main__":
    main()
