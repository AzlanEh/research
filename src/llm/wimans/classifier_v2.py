"""LLM orchestration for WiMANS v2 multi-subject activity-count inference."""

from __future__ import annotations

from typing import Any

from common.llm.client import LLMClientError, chat_completion, is_configured

from .classifier import ClassifierError
from .prompt_builder_v2 import build_zero_shot_prompt_v2, prompt_version_v2
from .response_parser import parse_response


def classify_record(
    record: dict[str, Any],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Infer activity counts for one WiMANS v2 feature record."""
    if not is_configured():
        raise ClassifierError(
            "LLM client is not configured. Set LLM_API_KEY."
        )

    messages = build_zero_shot_prompt_v2(record)
    saved_messages = [dict(message) for message in messages]

    result_base = {
        "sample_id": record.get("sample_id"),
        "group_id": record.get("group_id"),
        "environment": record.get("environment"),
        "wifi_band": record.get("wifi_band"),
        "prompt_version": prompt_version_v2(),
        "prompt_mode": "zero_shot",
        "prompt_messages": saved_messages,
    }

    try:
        llm_result = chat_completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except LLMClientError as exc:
        return {
            **result_base,
            "predicted_activity_counts": None,
            "confidence": None,
            "reason": None,
            "parse_status": "client_error",
            "model": model or "configured_model",
            "latency_ms": None,
            "token_usage": None,
            "raw_response": None,
            "error": str(exc),
        }

    parsed = parse_response(llm_result["content"])

    return {
        **result_base,
        "predicted_activity_counts": parsed.predicted_activity_counts,
        "confidence": parsed.confidence,
        "reason": parsed.reason,
        "parse_status": parsed.status,
        "model": llm_result["model"],
        "latency_ms": llm_result["latency_ms"],
        "token_usage": llm_result["token_usage"],
        "raw_response": llm_result["raw_response"],
        "error": parsed.error_message,
    }


def classify_batch(
    records: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Classify records sequentially, preserving input order."""
    return [
        {
            **classify_record(
                record,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            "record_index": index,
        }
        for index, record in enumerate(records)
    ]
