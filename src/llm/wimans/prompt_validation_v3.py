"""Validation wrapper for v3 prompts using the frozen v1/v2 payload audits."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .prompt_builder import SYSTEM_PROMPT
from .prompt_builder_v3 import SYSTEM_PROMPT_V3
from .prompt_validation import audit_v1_prompt, audit_v2_prompt


def audit_v3_prompt(
    messages: Sequence[Mapping[str, str]],
    record: Mapping[str, Any],
    *,
    feature_version: str,
) -> list[str]:
    """Check v3 instructions and delegate payload checks to baseline audits."""
    if not messages or messages[0].get("content") != SYSTEM_PROMPT_V3:
        return ["System message does not match SYSTEM_PROMPT_V3."]

    baseline_messages = [dict(message) for message in messages]
    baseline_messages[0]["content"] = SYSTEM_PROMPT

    if feature_version == "v1":
        return audit_v1_prompt(baseline_messages, record)

    if feature_version == "v2":
        return audit_v2_prompt(baseline_messages, record)

    return ["feature_version must be 'v1' or 'v2'."]
