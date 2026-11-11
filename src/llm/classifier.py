"""classifier.py

Orchestrates LLM-based CSI activity classification.

Ties together:
  - common/llm/client.py         : API call
  - prompt_builder.py            : prompt construction (zero-shot / few-shot)
  - common/llm/response_parser.py: strict JSON parsing

Returns a prediction record containing everything needed for evaluation:

    {
      "source_file": "...",
      "ground_truth": "A",
      "predicted_label": "A",
      "confidence": 0.85,
      "reason": "...",
      "parse_status": "ok",
      "model": "...",
      "latency_ms": 1234.5,
      "token_usage": {...},
      "prompt_version": "v1_zero_shot",
      "representation": "structured_features",
      "error": null
    }
"""

from __future__ import annotations

from typing import Any

from common.llm.client import LLMClientError, chat_completion, is_configured
from .prompt_builder import (
    REPRESENTATION_STRUCTURED,
    REPRESENTATION_AMPLITUDE,
    build_few_shot_prompt,
    build_zero_shot_prompt,
    prompt_version,
)
from common.llm.response_parser import ALLOWED_LABELS, parse_response


class ClassifierError(RuntimeError):
    """Raised when classification cannot proceed (e.g. client not configured)."""


def classify_record(
    record: dict[str, Any],
    *,
    representation: str = "structured_features",
    few_shot_examples: list[dict[str, Any]] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Classify a single CSI feature record.

    Args:
        record: Feature record dict (from extract_features.py).
        representation: Input representation name (for evaluation logging).
        few_shot_examples: Optional list of labelled example records.
        model: Override model name.
        temperature: Override temperature.
        max_tokens: Override max tokens.

    Returns:
        Prediction record dict with ground truth, prediction, confidence,
        parse status, model, latency, token usage, prompt version, and any
        error.
    """
    if not is_configured():
        raise ClassifierError(
            "LLM client is not configured. Set LLM_API_KEY environment variable."
        )

    ground_truth = record.get("label")
    source_file = record.get("source_file")
    split = record.get("split")

    # Build the prompt.
    if few_shot_examples:
        messages = build_few_shot_prompt(
            record,
            few_shot_examples,
            representation=representation,
        )
        prompt_mode = "few_shot"
    else:
        messages = build_zero_shot_prompt(
            record,
            representation=representation,
        )
        prompt_mode = "zero_shot"

    # Save the exact prompt used (for reproducibility).
    prompt_messages = [dict(m) for m in messages]

    # Call the LLM.
    try:
        llm_result = chat_completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except LLMClientError as exc:
        return {
            "source_file": source_file,
            "ground_truth": ground_truth,
            "split": split,
            "predicted_label": None,
            "confidence": None,
            "reason": None,
            "parse_status": "client_error",
            "model": model or "configured_model",
            "latency_ms": None,
            "token_usage": None,
            "prompt_version": prompt_version(),
            "representation": representation,
            "prompt_mode": prompt_mode,
            "error": str(exc),
            "prompt_messages": prompt_messages,
        }

    # Parse the response.
    parse_result = parse_response(llm_result["content"], allowed_labels=ALLOWED_LABELS)

    return {
        "source_file": source_file,
        "ground_truth": ground_truth,
        "split": split,
        "predicted_label": parse_result.predicted_label,
        "confidence": parse_result.confidence,
        "reason": parse_result.reason,
        "parse_status": parse_result.status,
        "model": llm_result["model"],
        "latency_ms": llm_result["latency_ms"],
        "token_usage": llm_result["token_usage"],
        "prompt_version": prompt_version(),
        "representation": representation,
        "prompt_mode": prompt_mode,
        "error": parse_result.error_message,
        "raw_response": llm_result["raw_response"],
        "prompt_messages": prompt_messages,
    }


def classify_batch(
    records: list[dict[str, Any]],
    *,
    representation: str = "structured_features",
    few_shot_examples: list[dict[str, Any]] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Classify a batch of feature records.

    Returns a list of prediction record dicts (one per input record).
    """
    predictions = []
    for i, record in enumerate(records):
        pred = classify_record(
            record,
            representation=representation,
            few_shot_examples=few_shot_examples,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        pred["record_index"] = i
        predictions.append(pred)
    return predictions
