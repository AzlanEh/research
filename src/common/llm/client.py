"""client.py

Retry-aware OpenAI-compatible LLM API client for the Wi-Fi CSI activity-recognition POC.

Designed to work with an OpenAI-compatible chat-completions endpoint
(Gemini, OpenCode, NVIDIA, etc.). All configuration is read from environment
variables so the model, endpoint, and credentials can be swapped without
changing code:

  LLM_API_BASE_URL   Base URL of the API (default: Gemini v1beta OpenAI-compatible)
  LLM_API_KEY        API key (required)
  LLM_MODEL          Model name (default: gemini-3.5-flash-lite)
  LLM_TEMPERATURE    Sampling temperature (default: 0.0)
  LLM_MAX_TOKENS     Max completion tokens (default: 512)
  LLM_TIMEOUT        Request timeout in seconds (default: 120)
  LLM_MAX_RETRIES    Max retries for 429/5xx errors (default: 5)
  LLM_RETRY_BASE_SECONDS   Base backoff seconds (default: 5)
  LLM_RETRY_MAX_SECONDS    Max backoff seconds (default: 120)
  LLM_MIN_REQUEST_INTERVAL Min seconds between requests in one process (default: 5)

Only the Python standard library is used (urllib), so no new pip dependency
is required.

Example request format (OpenAI-compatible):

    POST {base_url}/chat/completions
    Headers: Authorization: Bearer {api_key}, Content-Type: application/json
    Body: {"model": ..., "messages": [...], "temperature": ..., "max_tokens": ...}

Response is returned as a dict with the parsed completion plus metadata
(latency, model, raw JSON, token usage).

Retry behavior:
  - HTTP 429 (rate limit) and 5xx (server errors) are retried up to
    LLM_MAX_RETRIES times with exponential backoff.
  - Retry-After header and Google's RetryInfo delay are honored when present.
  - A minimum request interval (LLM_MIN_REQUEST_INTERVAL) is enforced between
    requests in the same process to avoid burst rate limiting.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Any


# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


API_BASE_URL = _env(
    "LLM_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
API_KEY = _env("LLM_API_KEY")
MODEL = _env("LLM_MODEL", "gemini-3.5-flash-lite")
TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.0") or 0.0)
MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "512") or 512)
TIMEOUT = int(_env("LLM_TIMEOUT", "120") or 120)
MAX_RETRIES = int(_env("LLM_MAX_RETRIES", "5") or 5)
RETRY_BASE_SECONDS = float(_env("LLM_RETRY_BASE_SECONDS", "5") or 5)
RETRY_MAX_SECONDS = float(_env("LLM_RETRY_MAX_SECONDS", "120") or 120)
MIN_REQUEST_INTERVAL = float(_env("LLM_MIN_REQUEST_INTERVAL", "5") or 5)

# Module-level state to enforce minimum request interval across calls.
_LAST_REQUEST_TIME = 0.0


class LLMClientError(RuntimeError):
    """Raised when an LLM API request fails after all retries are exhausted."""


def _enforce_min_interval() -> None:
    """Ensure at least MIN_REQUEST_INTERVAL seconds have passed since the last request."""
    global _LAST_REQUEST_TIME
    elapsed = time.monotonic() - _LAST_REQUEST_TIME
    if elapsed < MIN_REQUEST_INTERVAL:
        sleep_for = MIN_REQUEST_INTERVAL - elapsed
        time.sleep(sleep_for)


def _server_retry_delay(headers: Any, body_text: str) -> float | None:
    """Extract Retry-After or Google's RetryInfo delay, if present.

    Returns seconds to wait, or None if no retry delay could be determined.
    """
    # Check standard Retry-After header.
    if headers is not None:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                # Might be an HTTP-date; try parsing it.
                try:
                    retry_date = parsedate_to_datetime(retry_after)
                    if retry_date:
                        return max(0.0, (retry_date - time.gmtime()).total_seconds())
                except Exception:
                    pass

    # Check Google's RetryInfo in the error body.
    try:
        data = json.loads(body_text)
        details = data.get("error", {}).get("details", [])
        for detail in details:
            if isinstance(detail, dict):
                retry_delay = detail.get("retryDelay")
                if retry_delay:
                    # Format like "47.18661934s"
                    match = re.match(r"([\d.]+)s", str(retry_delay))
                    if match:
                        return max(0.0, float(match.group(1)))
    except Exception:
        pass

    return None


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a chat-completions request to the LLM API with retry logic.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: Model name (overrides LLM_MODEL env var).
        temperature: Sampling temperature (overrides LLM_TEMPERATURE).
        max_tokens: Max completion tokens (overrides LLM_MAX_TOKENS).
        timeout: Request timeout in seconds (overrides LLM_TIMEOUT).
        extra_body: Additional JSON body fields merged into the request.

    Returns:
        Dict with keys:
            "content"        : str, the assistant's message content
            "model"          : str, model used
            "latency_ms"     : float, total request latency in milliseconds
            "raw_response"   : dict, the full parsed JSON response
            "token_usage"    : dict | None, usage info if present
            "finish_reason"  : str | None
            "error"          : str | None, set only on failure after all retries

    Raises:
        LLMClientError: if API_KEY is missing or the request fails after all retries.
    """
    if not API_KEY:
        raise LLMClientError(
            "LLM_API_KEY environment variable is not set. "
            "Set it before calling the LLM."
        )

    model = model or MODEL
    temperature = TEMPERATURE if temperature is None else temperature
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    timeout = TIMEOUT if timeout is None else timeout

    url = API_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "OpenCode/1.0 (wifi-extend research project)",
    }

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        body.update(extra_body)

    payload = json.dumps(body).encode("utf-8")

    last_error: str | None = None
    total_start = time.perf_counter()

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            # Exponential backoff before retrying.
            backoff = min(
                RETRY_BASE_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_SECONDS
            )
            # Add small jitter to avoid thundering herd.
            backoff += random.uniform(0, 1.0)
            time.sleep(backoff)

        # Enforce minimum request interval before each attempt.
        _enforce_min_interval()

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        attempt_start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                _LAST_REQUEST_TIME = time.monotonic()
        except urllib.error.HTTPError as exc:  # noqa: BLE001
            body_text = exc.read().decode("utf-8", errors="replace")
            latency_ms = (time.perf_counter() - attempt_start) * 1000

            # Retry on 429 (rate limit) and 5xx (server errors).
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if retryable and attempt < MAX_RETRIES:
                server_delay = _server_retry_delay(
                    exc.headers if hasattr(exc, "headers") else None, body_text
                )
                delay = server_delay if server_delay is not None else None
                last_error = (
                    f"HTTP {exc.code} (retryable, attempt {attempt + 1}/{MAX_RETRIES + 1}): "
                    f"{body_text[:200]}"
                )
                # If we have a server-suggested delay, sleep before the next attempt
                # (the loop's backoff will also apply on the next iteration).
                if delay is not None:
                    time.sleep(delay)
                continue

            # Non-retryable or out of retries.
            raise LLMClientError(
                f"LLM API HTTP {exc.code} for {url}: {body_text}"
            ) from exc
        except urllib.error.URLError as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - attempt_start) * 1000
            if attempt < MAX_RETRIES:
                last_error = (
                    f"Connection error (retryable, attempt {attempt + 1}/{MAX_RETRIES + 1}): {exc}"
                )
                continue
            raise LLMClientError(
                f"LLM API connection error for {url}: {exc}"
            ) from exc

        # Success — parse the response.
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                f"LLM API returned non-JSON response: {raw[:500]}"
            ) from exc

        # Extract the assistant's message content.
        choices = data.get("choices") or []
        content = ""
        finish_reason = None
        if choices:
            first = choices[0]
            message = first.get("message") or {}
            content = message.get("content") or ""
            finish_reason = first.get("finish_reason")

        usage = data.get("usage") or None

        return {
            "content": content,
            "model": model,
            "latency_ms": round((time.perf_counter() - total_start) * 1000, 2),
            "raw_response": data,
            "token_usage": usage,
            "finish_reason": finish_reason,
            "error": last_error if last_error else None,
        }

    # All retries exhausted.
    raise LLMClientError(
        f"LLM API request failed after {MAX_RETRIES + 1} attempts. "
        f"Last error: {last_error}"
    )


def is_configured() -> bool:
    """Return True if the client appears configured (API key present)."""
    return bool(API_KEY)


def config_summary() -> dict[str, Any]:
    """Return a summary of the current client configuration (no secrets)."""
    return {
        "api_base_url": API_BASE_URL,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "timeout": TIMEOUT,
        "max_retries": MAX_RETRIES,
        "retry_base_seconds": RETRY_BASE_SECONDS,
        "retry_max_seconds": RETRY_MAX_SECONDS,
        "min_request_interval": MIN_REQUEST_INTERVAL,
        "api_key_configured": bool(API_KEY),
    }
