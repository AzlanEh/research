#!/usr/bin/env python
"""Alternating v1/v2 paired pilot runner for WiMANS.

Protocol: 20 pilot recordings, one per test group. For each recording the v1
request is sent first, then the v2 request, then the next recording:

    sample 1: v1 -> v2
    sample 2: v1 -> v2
    ...
    sample 20: v1 -> v2

Both versions use the same model, temperature and max-token settings, which
are passed explicitly and recorded in ``summary.json``.

Safety:
  * Default mode is a dry run. No network calls are made; the client is
    stubbed and results are written to a ``*_dry_run`` directory.
  * A real run requires ``--execute``. It refuses to start when the client is
    not configured.
  * Every prompt is audited for ground-truth leakage before its request.
  * Ground truth is joined only after all inference is complete.

Usage:
    ./.venv/bin/python experiments/wimans/run_paired_pilot.py
    ./.venv/bin/python experiments/wimans/run_paired_pilot.py --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.llm.client import (  # noqa: E402
    config_summary,
    is_configured,
)
from evaluation.wimans.metrics import evaluate_records  # noqa: E402
from evaluation.wimans.paired import (  # noqa: E402
    join_ground_truth,
    paired_summary,
)
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
MANIFEST_PATH = REPO_ROOT / "data/processed/wimans/manifest.jsonl"
V1_PATH = REPO_ROOT / "data/processed/wimans/features_v1.jsonl"
V2_PATH = REPO_ROOT / "data/processed/wimans/features_v2_pilot.jsonl"

RESULTS_ROOT = REPO_ROOT / "results/wimans"

# JSON key tokens that must never reach the model. Prose words such as
# "activity" in the system prompt are intentionally not matched.
LEAK_TOKENS = (
    '"sample_id"',
    '"group_id"',
    '"number_of_users"',
    '"users"',
    '"activity"',
    '"ground_truth_activity_counts"',
)

DRY_RUN_CONTENT = json.dumps(
    {
        "predicted_activity_counts": {label: 0 for label in ACTIVITY_LABELS},
        "confidence": 0.0,
        "reason": "Dry-run stub response; no model was called.",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_pilot_selection() -> dict[str, Any]:
    with SELECTION_PATH.open() as handle:
        return json.load(handle)


def load_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records[row["sample_id"]] = row
    return records


def stub_chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "content": DRY_RUN_CONTENT,
        "model": model or "dry-run-stub",
        "latency_ms": 0.0,
        "raw_response": {"dry_run": True},
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "finish_reason": "stop",
        "error": None,
    }


def check_prompt_leakage(messages: list[dict[str, str]], sample_id: str) -> None:
    """Belt-and-braces text scan of the exact messages about to be sent."""
    serialized = json.dumps(messages, sort_keys=True)
    found = [token for token in LEAK_TOKENS if token in serialized]
    if found:
        raise RuntimeError(
            f"Refusing to send prompt for {sample_id!r}: leaked {found}."
        )


def audit_pair(
    v1_record: dict[str, Any],
    v2_record: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build and audit both prompts before either request is sent."""
    v1_messages = build_zero_shot_prompt(v1_record)
    v2_messages = build_zero_shot_prompt_v2(v2_record)

    problems = [
        f"[v1] {problem}" for problem in audit_v1_prompt(v1_messages, v1_record)
    ] + [
        f"[v2] {problem}" for problem in audit_v2_prompt(v2_messages, v2_record)
    ]

    sample_id = str(v1_record.get("sample_id"))
    check_prompt_leakage(v1_messages, sample_id)
    check_prompt_leakage(v2_messages, sample_id)

    if problems:
        raise RuntimeError(
            f"Prompt audit failed for {sample_id!r}: " + "; ".join(problems)
        )

    return v1_messages, v2_messages


def run(
    *,
    dry_run: bool,
    output_dir: Path,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
) -> int:
    selection = load_pilot_selection()
    selected = selection["selected_samples"]
    sample_ids = [row["sample_id"] for row in selected]

    if len(sample_ids) != 20 or len(set(sample_ids)) != 20:
        raise SystemExit("Pilot selection must contain 20 unique sample ids.")

    manifest = load_jsonl_index(MANIFEST_PATH)
    v1_records = load_jsonl_index(V1_PATH)
    v2_records = load_jsonl_index(V2_PATH)

    for version, index in (("v1", v1_records), ("v2", v2_records)):
        missing = sorted(set(sample_ids) - set(index))
        if missing:
            raise SystemExit(f"Missing {version} features for: {missing}")

    missing_manifest = sorted(set(sample_ids) - set(manifest))
    if missing_manifest:
        raise SystemExit(f"Missing manifest rows for: {missing_manifest}")

    client_config = config_summary()

    # Resolve defaults once so both versions receive byte-identical settings.
    settings = {
        "model": model or client_config["model"],
        "temperature": (
            client_config["temperature"] if temperature is None else temperature
        ),
        "max_tokens": (
            client_config["max_tokens"] if max_tokens is None else max_tokens
        ),
        "max_retries": client_config["max_retries"],
        "timeout": client_config["timeout"],
        "min_request_interval": client_config["min_request_interval"],
    }
    inference_kwargs = {
        "model": settings["model"],
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    # Completed sample pairs are streamed here so a mid-run failure does not
    # discard requests that already consumed API quota.
    stream_path = output_dir / "stream.jsonl"
    stream = stream_path.open("w")

    started = datetime.now(timezone.utc)
    v1_results: list[dict[str, Any]] = []
    v2_results: list[dict[str, Any]] = []
    request_log: list[dict[str, Any]] = []

    print("WiMANS paired pilot")
    print(f"  mode: {'DRY RUN (no network)' if dry_run else 'EXECUTE (network)'}")
    print(f"  samples: {len(selected)}")
    print(f"  requests: {len(selected) * 2}")
    print(f"  order: per-sample v1 then v2")
    print(f"  output: {display_path(output_dir)}")
    print()

    with ExitStack() as stack:
        if dry_run:
            for module in (classifier, classifier_v2):
                stack.enter_context(
                    patch.object(module, "is_configured", return_value=True)
                )
                stack.enter_context(
                    patch.object(
                        module,
                        "chat_completion",
                        side_effect=stub_chat_completion,
                    )
                )

        for position, row in enumerate(selected, start=1):
            sample_id = row["sample_id"]
            v1_record = v1_records[sample_id]
            v2_record = v2_records[sample_id]

            v1_messages, v2_messages = audit_pair(v1_record, v2_record)

            v1_result = classifier.classify_record(
                v1_record,
                **inference_kwargs,
            )
            if v1_result.get("prompt_messages") != v1_messages:
                raise RuntimeError(
                    f"v1 classifier did not use the audited prompt for "
                    f"{sample_id!r}."
                )
            check_prompt_leakage(v1_result["prompt_messages"], sample_id)
            v1_result = {**v1_result, "request_index": 2 * position - 1}
            v1_results.append(v1_result)

            v2_result = classifier_v2.classify_record(
                v2_record,
                **inference_kwargs,
            )
            if v2_result.get("prompt_messages") != v2_messages:
                raise RuntimeError(
                    f"v2 classifier did not use the audited prompt for "
                    f"{sample_id!r}."
                )
            check_prompt_leakage(v2_result["prompt_messages"], sample_id)
            v2_result = {**v2_result, "request_index": 2 * position}
            v2_results.append(v2_result)

            request_log.append(
                {
                    "sample_id": sample_id,
                    "group_id": row["group_id"],
                    "v1_request_index": 2 * position - 1,
                    "v2_request_index": 2 * position,
                    "v1_parse_status": v1_result["parse_status"],
                    "v2_parse_status": v2_result["parse_status"],
                    "v1_latency_ms": v1_result["latency_ms"],
                    "v2_latency_ms": v2_result["latency_ms"],
                }
            )

            stream.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "v1": v1_result,
                        "v2": v2_result,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()

            print(
                f"  [{position:>2}/{len(selected)}] {sample_id:<14} "
                f"v1={v1_result['parse_status']:<12} "
                f"v2={v2_result['parse_status']:<12}"
            )

    stream.close()

    # Ground truth is joined only now, after every request has completed.
    evaluation_v1 = join_ground_truth(v1_results, manifest)
    evaluation_v2 = join_ground_truth(v2_results, manifest)

    metrics = {
        "protocol": selection["protocol"],
        "dry_run": dry_run,
        "sample_count": len(selected),
        "v1": evaluate_records(evaluation_v1),
        "v2": evaluate_records(evaluation_v2),
        "paired": paired_summary(evaluation_v1, evaluation_v2),
    }

    finished = datetime.now(timezone.utc)
    summary = {
        "protocol": selection["protocol"],
        "dry_run": dry_run,
        "network_calls": 0 if dry_run else len(selected) * 2,
        "order": "per-sample v1 then v2",
        "sample_count": len(selected),
        "request_count": len(selected) * 2,
        "inference_settings": settings,
        "client_config": client_config,
        "selection_path": display_path(SELECTION_PATH),
        "stream_path": display_path(stream_path),
        "selection_sha256": sha256_file(SELECTION_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "v1_features_path": display_path(V1_PATH),
        "v2_features_path": display_path(V2_PATH),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "requests": request_log,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    for name, results in (("v1", v1_results), ("v2", v2_results)):
        path = output_dir / f"predictions_{name}.jsonl"
        with path.open("w") as handle:
            for result in results:
                handle.write(
                    json.dumps(result, allow_nan=False) + "\n"
                )

    for name, results in (("v1", evaluation_v1), ("v2", evaluation_v2)):
        path = output_dir / f"evaluation_{name}.jsonl"
        with path.open("w") as handle:
            for result in results:
                handle.write(
                    json.dumps(result, allow_nan=False) + "\n"
                )

    print()
    print("Metrics")
    for version in ("v1", "v2"):
        version_metrics = metrics[version]
        print(
            f"  {version}: exact={version_metrics['exact_multiset_accuracy']:.3f} "
            f"count_mae={version_metrics['count_mae']:.3f} "
            f"person_mae={version_metrics['total_person_count_mae']:.3f} "
            f"parse_ok={version_metrics['parse_success_count']}/"
            f"{version_metrics['sample_count']}"
        )

    paired = metrics["paired"]
    print(
        f"  paired: v1_better={paired['v1_better_samples']} "
        f"v2_better={paired['v2_better_samples']} "
        f"ties={paired['tie_samples']} "
        f"mean_delta={paired['mean_count_error_delta_v2_minus_v1']:+.3f}"
    )
    print()
    print(f"  summary:    {display_path(output_dir / 'summary.json')}")
    print(f"  metrics:    {display_path(output_dir / 'metrics.json')}")
    if dry_run:
        print()
        print("DRY RUN COMPLETE: 0 network calls. No experimental results.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the alternating v1/v2 WiMANS pilot."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send real API requests. Without this flag the run is a dry run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for artifacts (default depends on mode).",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    dry_run = not args.execute

    if dry_run:
        output_dir = args.output_dir or (RESULTS_ROOT / "pilot_paired_dry_run")
    else:
        if not is_configured():
            raise SystemExit(
                "LLM client is not configured. Set LLM_API_KEY before "
                "using --execute."
            )
        output_dir = args.output_dir or (RESULTS_ROOT / "pilot_paired")
        print("WARNING: --execute will send 40 real API requests.")
        print()

    return run(
        dry_run=dry_run,
        output_dir=output_dir,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    raise SystemExit(main())
