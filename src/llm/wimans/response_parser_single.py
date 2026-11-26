"""Strict parser for WiMANS single-person presence/activity responses.

The multi-person parser is intentionally untouched. This parser validates one
unified schema for both evaluation arms: presence plus at most one activity.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .prompt_builder import ACTIVITY_LABELS
from .response_parser import _extract_json_object

REQUIRED_KEYS = frozenset({"present", "activity", "confidence", "reason"})


class ParseResultSingle:
    def __init__(
        self,
        status: str,
        present: bool | None = None,
        activity: str | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        raw_content: str = "",
        error_message: str | None = None,
    ):
        self.status = status
        self.present = present
        self.activity = activity
        self.confidence = confidence
        self.reason = reason
        self.raw_content = raw_content
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_status": self.status,
            "present": self.present,
            "activity": self.activity,
            "confidence": self.confidence,
            "reason": self.reason,
            "raw_content": self.raw_content,
            "error_message": self.error_message,
        }


def parse_response_single(content: str) -> ParseResultSingle:
    if not content or not content.strip():
        return ParseResultSingle(
            "empty",
            raw_content=content or "",
            error_message="LLM response was empty.",
        )

    json_text = _extract_json_object(content)
    if json_text is None:
        return ParseResultSingle(
            "json_error",
            raw_content=content,
            error_message="No complete JSON object found.",
        )

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return ParseResultSingle(
            "json_error",
            raw_content=content,
            error_message=f"Invalid JSON: {exc}",
        )

    if not isinstance(data, dict):
        return ParseResultSingle(
            "schema_error",
            raw_content=content,
            error_message="Response must be a JSON object.",
        )

    if set(data) != REQUIRED_KEYS:
        return ParseResultSingle(
            "schema_error",
            raw_content=content,
            error_message=(
                "Expected exactly these top-level keys: "
                f"{sorted(REQUIRED_KEYS)}"
            ),
        )

    present = data["present"]
    if not isinstance(present, bool):
        return ParseResultSingle(
            "schema_error",
            raw_content=content,
            error_message="'present' must be a boolean.",
        )

    activity = data["activity"]
    if present:
        if not isinstance(activity, str) or activity not in ACTIVITY_LABELS:
            return ParseResultSingle(
                "schema_error",
                raw_content=content,
                error_message=(
                    "'activity' must be one of the allowed labels when "
                    "'present' is true."
                ),
            )
    elif activity is not None:
        return ParseResultSingle(
            "schema_error",
            raw_content=content,
            error_message="'activity' must be null when 'present' is false.",
        )

    confidence = data["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return ParseResultSingle(
            "schema_error",
            raw_content=content,
            error_message="Confidence must be a number.",
        )

    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return ParseResultSingle(
            "value_error",
            raw_content=content,
            error_message="Confidence must be finite and between 0 and 1.",
        )

    reason = data["reason"]
    if not isinstance(reason, str) or not reason.strip():
        return ParseResultSingle(
            "schema_error",
            raw_content=content,
            error_message="Reason must be a non-empty string.",
        )

    return ParseResultSingle(
        "ok",
        present=present,
        activity=activity,
        confidence=confidence,
        reason=reason,
        raw_content=content,
    )
