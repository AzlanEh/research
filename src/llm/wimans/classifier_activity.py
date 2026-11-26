"""LLM orchestration for WiMANS single-person activity-only inference."""

from __future__ import annotations

from typing import Any

from common.llm.client import LLMClientError, chat_completion, is_configured

from .classifier import ClassifierError
from .prompt_builder_activity_v2 import (
    build_activity_only_prompt_v2,
    prompt_version_activity_v2,
)
from .response_parser_activity import parse_response_activity


def classify_record(
    record: dict[str, Any],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Classify the activity of the one person in a WiMANS recording."""
    if not is_configured():
        raise ClassifierError(
            "LLM client is not configured. Set LLM_API_KEY."
        )

    messages = build_activity_only_prompt_v2(record)
    saved_messages = [dict(message) for message in messages]

    result_base = {
        "sample_id": record.get("sample_id"),
        "group_id": record.get("group_id"),
        "environment": record.get("environment"),
        "wifi_band": record.get("wifi_band"),
        "prompt_version": prompt_version_activity_v2(),
        "prompt_mode": "activity_only_zero_shot",
        "feature_version": "wimans_recording_v2",
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
            "activity": None,
            "confidence": None,
            "reason": None,
            "parse_status": "client_error",
            "model": model or "configured_model",
            "latency_ms": None,
            "token_usage": None,
            "raw_response": None,
            "error": str(exc),
        }

    parsed = parse_response_activity(llm_result["content"])

    return {
        **result_base,
        "activity": parsed.activity,
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
