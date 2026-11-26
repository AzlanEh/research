"""Strict JSON parser for WiMANS activity-count predictions."""

from __future__ import annotations

import json
import math
from typing import Any

from .prompt_builder import ACTIVITY_LABELS

MAX_PEOPLE = 5


class ParseResult:
    def __init__(
        self,
        status: str,
        predicted_activity_counts: dict[str, int] | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        raw_content: str = "",
        error_message: str | None = None,
    ):
        self.status = status
        self.predicted_activity_counts = predicted_activity_counts
        self.confidence = confidence
        self.reason = reason
        self.raw_content = raw_content
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_status": self.status,
            "predicted_activity_counts": self.predicted_activity_counts,
            "confidence": self.confidence,
            "reason": self.reason,
            "raw_content": self.raw_content,
            "error_message": self.error_message,
        }


def _extract_json_object(content: str) -> str | None:
    """Extract the first balanced JSON object, respecting quoted strings."""
    start = content.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(content)):
        char = content[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start:index + 1]

    return None


def parse_response(content: str) -> ParseResult:
    if not content or not content.strip():
        return ParseResult(
            "empty",
            raw_content=content or "",
            error_message="LLM response was empty.",
        )

    json_text = _extract_json_object(content)
    if json_text is None:
        return ParseResult(
            "json_error",
            raw_content=content,
            error_message="No complete JSON object found.",
        )

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return ParseResult(
            "json_error",
            raw_content=content,
            error_message=f"Invalid JSON: {exc}",
        )

    if not isinstance(data, dict):
        return ParseResult(
            "schema_error",
            raw_content=content,
            error_message="Response must be a JSON object.",
        )

    required = {"predicted_activity_counts", "confidence", "reason"}
    if set(data) != required:
        return ParseResult(
            "schema_error",
            raw_content=content,
            error_message=f"Expected exactly these top-level keys: {sorted(required)}",
        )

    counts = data["predicted_activity_counts"]
    if not isinstance(counts, dict) or set(counts) != set(ACTIVITY_LABELS):
        return ParseResult(
            "schema_error",
            raw_content=content,
            error_message="Activity counts must contain every allowed activity key exactly once.",
        )

    validated_counts: dict[str, int] = {}
    for label in ACTIVITY_LABELS:
        value = counts[label]
        # bool is a subclass of int in Python, so reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, int):
            return ParseResult(
                "schema_error",
                raw_content=content,
                error_message=f"Count for {label!r} must be an integer.",
            )
        if value < 0 or value > MAX_PEOPLE:
            return ParseResult(
                "value_error",
                raw_content=content,
                error_message=f"Count for {label!r} must be between 0 and {MAX_PEOPLE}.",
            )
        validated_counts[label] = value

    if sum(validated_counts.values()) > MAX_PEOPLE:
        return ParseResult(
            "value_error",
            raw_content=content,
            error_message=f"Total predicted people cannot exceed {MAX_PEOPLE}.",
        )

    confidence = data["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return ParseResult(
            "schema_error",
            raw_content=content,
            error_message="Confidence must be a number.",
        )

    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return ParseResult(
            "value_error",
            raw_content=content,
            error_message="Confidence must be finite and between 0 and 1.",
        )

    reason = data["reason"]
    if not isinstance(reason, str):
        return ParseResult(
            "schema_error",
            raw_content=content,
            error_message="Reason must be a string.",
        )

    return ParseResult(
        "ok",
        predicted_activity_counts=validated_counts,
        confidence=confidence,
        reason=reason,
        raw_content=content,
    )
