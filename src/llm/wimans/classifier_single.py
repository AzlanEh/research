"""LLM orchestration for WiMANS single-person presence/activity inference."""

from __future__ import annotations

from typing import Any

from common.llm.client import LLMClientError, chat_completion, is_configured

from .classifier import ClassifierError
from .prompt_builder_single import (
    build_single_person_prompt,
    prompt_version_single,
)
from .response_parser_single import parse_response_single


def classify_record(
    record: dict[str, Any],
    *,
    feature_version: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Infer presence and a single activity for one feature record."""
    if not is_configured():
        raise ClassifierError(
            "LLM client is not configured. Set LLM_API_KEY."
        )

    messages = build_single_person_prompt(
        record,
        feature_version=feature_version,
    )
    saved_messages = [dict(message) for message in messages]

    result_base = {
        "sample_id": record.get("sample_id"),
        "group_id": record.get("group_id"),
        "environment": record.get("environment"),
        "wifi_band": record.get("wifi_band"),
        "prompt_version": prompt_version_single(feature_version),
        "prompt_mode": "zero_shot",
        "feature_version": feature_version,
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
            "present": None,
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

    parsed = parse_response_single(llm_result["content"])

    return {
        **result_base,
        "present": parsed.present,
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
    feature_version: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Classify records sequentially, preserving input order."""
    return [
        {
            **classify_record(
                record,
                feature_version=feature_version,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            "record_index": index,
        }
        for index, record in enumerate(records)
    ]
