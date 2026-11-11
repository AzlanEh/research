"""response_parser.py

Strict JSON response parser for LLM classification output.

The LLM is instructed to return valid JSON only:

    {
      "predicted_label": "A | B | C | UNKNOWN",
      "confidence": 0.0,
      "reason": "Brief explanation based on the supplied features"
    }

This module:
  1. Extracts the JSON object from the LLM response (handles code fences,
     extra prose, or truncated output).
  2. Validates required keys and value types.
  3. Validates that predicted_label is one of the allowed labels.
  4. Clamps confidence to [0, 1].
  5. Returns a parse status so experiments can track parse failures.

Parse statuses:
  "ok"           - Valid JSON, label recognized, confidence in range.
  "json_error"   - Could not parse JSON from the response.
  "schema_error" - JSON parsed but missing keys or wrong types.
  "label_error"  - predicted_label is not one of the allowed labels.
  "empty"        - Response content was empty or whitespace.
"""

from __future__ import annotations

import json
import re
from typing import Any

ALLOWED_LABELS = {"A", "B", "C", "UNKNOWN"}

# Regex to find a JSON object (balanced braces) inside arbitrary text.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class ParseResult:
    """Result of parsing an LLM classification response.

    Attributes:
        status: One of "ok", "json_error", "schema_error", "label_error", "empty".
        predicted_label: The parsed label (or None on failure).
        confidence: The parsed confidence, clamped to [0, 1] (or None).
        reason: The parsed reason string (or None).
        raw_content: The raw LLM response text.
        error_message: Human-readable description of the failure (or None).
    """

    def __init__(
        self,
        status: str,
        predicted_label: str | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        raw_content: str = "",
        error_message: str | None = None,
    ):
        self.status = status
        self.predicted_label = predicted_label
        self.confidence = confidence
        self.reason = reason
        self.raw_content = raw_content
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_status": self.status,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "reason": self.reason,
            "raw_content": self.raw_content,
            "error_message": self.error_message,
        }


def _clamp_confidence(value: Any) -> float | None:
    """Coerce confidence to a float in [0, 1], or None if invalid."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return max(0.0, min(1.0, f))


def extract_json_object(content: str) -> str | None:
    """Attempt to extract a JSON object substring from arbitrary text.

    Strips Markdown code fences first, then looks for the first balanced
    ``{...}`` block.
    """
    if not content or not content.strip():
        return None

    text = content.strip()

    # Remove Markdown code fences if present.
    if text.startswith("```"):
        # Strip opening fence (with optional language tag)
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        # Strip closing fence
        text = re.sub(r"\n```\s*$", "", text)
        text = text.strip()

    # Try parsing the whole text first (fast path for clean JSON).
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Otherwise find the first { ... } block.
    match = _JSON_OBJECT_RE.search(text)
    if match:
        return match.group(0)
    return None


def parse_response(
    content: str,
    allowed_labels: set[str] | None = None,
) -> ParseResult:
    """Parse an LLM classification response into a ParseResult.

    Args:
        content: Raw text from the LLM.
        allowed_labels: Set of valid labels (default: {"A","B","C","UNKNOWN"}).

    Returns:
        ParseResult with status, parsed fields, and any error message.
    """
    labels = allowed_labels or ALLOWED_LABELS

    if not content or not content.strip():
        return ParseResult(
            status="empty",
            raw_content=content or "",
            error_message="LLM response content was empty or whitespace.",
        )

    json_text = extract_json_object(content)
    if json_text is None:
        return ParseResult(
            status="json_error",
            raw_content=content,
            error_message="No JSON object found in LLM response.",
        )

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return ParseResult(
            status="json_error",
            raw_content=content,
            error_message=f"JSON decode error: {exc}",
        )

    if not isinstance(data, dict):
        return ParseResult(
            status="schema_error",
            raw_content=content,
            error_message="Parsed JSON is not an object.",
        )

    # Validate required keys.
    required = {"predicted_label", "confidence", "reason"}
    missing = required - set(data.keys())
    if missing:
        return ParseResult(
            status="schema_error",
            raw_content=content,
            error_message=f"Missing required keys: {sorted(missing)}",
        )

    predicted_label = data.get("predicted_label")
    confidence_raw = data.get("confidence")
    reason = data.get("reason")

    # Validate label.
    if predicted_label not in labels:
        return ParseResult(
            status="label_error",
            predicted_label=predicted_label,
            confidence=_clamp_confidence(confidence_raw),
            reason=reason if isinstance(reason, str) else None,
            raw_content=content,
            error_message=(
                f"predicted_label '{predicted_label}' not in allowed labels {sorted(labels)}"
            ),
        )

    # Validate confidence.
    confidence = _clamp_confidence(confidence_raw)
    if confidence is None:
        return ParseResult(
            status="schema_error",
            predicted_label=predicted_label,
            confidence=None,
            reason=reason if isinstance(reason, str) else None,
            raw_content=content,
            error_message=f"confidence value is not a valid number: {confidence_raw!r}",
        )

    # Validate reason is a string.
    if not isinstance(reason, str):
        return ParseResult(
            status="schema_error",
            predicted_label=predicted_label,
            confidence=confidence,
            reason=None,
            raw_content=content,
            error_message="reason field is not a string.",
        )

    return ParseResult(
        status="ok",
        predicted_label=predicted_label,
        confidence=confidence,
        reason=reason,
        raw_content=content,
    )


def parse_statuses_summary(results: list[ParseResult]) -> dict[str, int]:
    """Count parse statuses across a list of ParseResult objects."""
    summary: dict[str, int] = {}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1
    return summary
