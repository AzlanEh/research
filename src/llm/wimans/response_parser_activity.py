"""Strict parser for WiMANS activity-only classification responses."""

from __future__ import annotations

import json
from typing import Any

from .prompt_builder import ACTIVITY_LABELS
from .response_parser import _extract_json_object

REQUIRED_KEYS = frozenset({"activity", "confidence", "reason"})


class ParseResultActivity:
    def __init__(
        self,
        status: str,
        activity: str | None = None,
        runner_up: str | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        raw_content: str = "",
        error_message: str | None = None,
    ):
        self.status = status
        self.activity = activity
        self.runner_up = runner_up
        self.confidence = confidence
        self.reason = reason
        self.raw_content = raw_content
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_status": self.status,
            "activity": self.activity,
            "runner_up": self.runner_up,
            "confidence": self.confidence,
            "reason": self.reason,
            "raw_content": self.raw_content,
            "error_message": self.error_message,
        }


def parse_response_activity(content: str) -> ParseResultActivity:
    """Parse and validate one activity-only JSON response."""

    if not content or not content.strip():
        return ParseResultActivity(
            "empty",
            raw_content=content or "",
            error_message="LLM response was empty.",
        )

    json_text = _extract_json_object(content)
    if json_text is None:
        return ParseResultActivity(
            "json_error",
            raw_content=content,
            error_message="No JSON object found in LLM response.",
        )

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return ParseResultActivity(
            "json_error",
            raw_content=content,
            error_message=f"JSON parse error: {exc}",
        )

    if not isinstance(data, dict):
        return ParseResultActivity(
            "schema_error",
            raw_content=content,
            error_message="Top-level JSON value must be an object.",
        )

    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        return ParseResultActivity(
            "schema_error",
            raw_content=content,
            error_message=f"Missing required keys: {sorted(missing)}",
        )

    activity = data.get("activity")
    runner_up = data.get("runner_up")
    confidence = data.get("confidence")
    reason = data.get("reason")

    if not isinstance(activity, str) or activity not in ACTIVITY_LABELS:
        return ParseResultActivity(
            "schema_error",
            raw_content=content,
            error_message=f"Invalid activity label: {activity!r}",
        )

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not (0.0 <= float(confidence) <= 1.0)
    ):
        return ParseResultActivity(
            "value_error",
            activity=activity,
            raw_content=content,
            error_message=f"Invalid confidence: {confidence!r}",
        )

    if not isinstance(reason, str) or not reason.strip():
        return ParseResultActivity(
            "schema_error",
            activity=activity,
            confidence=float(confidence),
            raw_content=content,
            error_message="Reason must be a non-empty string.",
        )

    return ParseResultActivity(
        "ok",
        activity=activity,
        runner_up=runner_up if isinstance(runner_up, str) and runner_up in ACTIVITY_LABELS else None,
        confidence=float(confidence),
        reason=reason.strip(),
        raw_content=content,
    )
