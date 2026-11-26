"""Regression tests for ``common.llm.client``.

All HTTP traffic is mocked. Module-level configuration is patched directly
because ``common/llm/client.py`` reads the environment at import time.

Pacing behavior covered here:
  - ``_LAST_REQUEST_TIME`` advances at request start, including failed attempts.
  - The configured minimum interval is enforced between request starts.
  - HTTP-date ``Retry-After`` values are parsed as timezone-aware datetimes.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest import mock

from common.llm import client


def completion_payload(
    content: str = "classification",
    finish_reason: str | None = "stop",
    usage: dict | None = None,
) -> dict:
    return {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


class ClientTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config = mock.patch.multiple(
            client,
            API_KEY="test-key",
            API_BASE_URL="https://example.invalid/v1",
            MODEL="test-model",
            TEMPERATURE=0.0,
            MAX_TOKENS=64,
            TIMEOUT=5,
            MAX_RETRIES=0,
            RETRY_BASE_SECONDS=0.0,
            RETRY_MAX_SECONDS=0.0,
            MIN_REQUEST_INTERVAL=0.0,
            _LAST_REQUEST_TIME=0.0,
        )
        config.start()
        self.addCleanup(config.stop)

        sleep_patcher = mock.patch.object(client.time, "sleep")
        self.sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def response(self, payload: dict) -> io.BytesIO:
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    def http_error(
        self,
        code: int,
        body: bytes = b'{"error": "boom"}',
        headers: dict | None = None,
    ) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions",
            code,
            "error",
            headers or {},
            io.BytesIO(body),
        )

    def call(self, **kwargs):
        messages = [{"role": "user", "content": "hello"}]
        return client.chat_completion(messages, **kwargs)


class ConfigurationTests(ClientTestCase):
    def test_missing_api_key_raises_before_request(self) -> None:
        client.API_KEY = ""

        with mock.patch.object(client.urllib.request, "urlopen") as urlopen:
            with self.assertRaises(client.LLMClientError):
                self.call()
            urlopen.assert_not_called()

    def test_is_configured(self) -> None:
        self.assertTrue(client.is_configured())
        client.API_KEY = ""
        self.assertFalse(client.is_configured())

    def test_config_summary_reports_settings_without_secret(self) -> None:
        summary = client.config_summary()

        self.assertEqual(summary["api_base_url"], "https://example.invalid/v1")
        self.assertEqual(summary["model"], "test-model")
        self.assertEqual(summary["max_retries"], 0)
        self.assertTrue(summary["api_key_configured"])
        self.assertNotIn("test-key", json.dumps(summary))


class RequestConstructionTests(ClientTestCase):
    def test_url_headers_and_default_body(self) -> None:
        payload = completion_payload()

        with mock.patch.object(
            client.urllib.request, "urlopen", return_value=self.response(payload)
        ) as urlopen:
            self.call()

        request = urlopen.call_args.args[0]
        timeout = urlopen.call_args.kwargs["timeout"]

        self.assertEqual(
            request.full_url,
            "https://example.invalid/v1/chat/completions",
        )
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertEqual(timeout, 5)

        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["max_tokens"], 64)

    def test_explicit_arguments_override_defaults(self) -> None:
        payload = completion_payload()

        with mock.patch.object(
            client.urllib.request, "urlopen", return_value=self.response(payload)
        ) as urlopen:
            self.call(model="other-model", temperature=0.7, max_tokens=9, timeout=3)

        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))

        self.assertEqual(body["model"], "other-model")
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["max_tokens"], 9)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 3)

    def test_extra_body_is_merged_last(self) -> None:
        payload = completion_payload()

        with mock.patch.object(
            client.urllib.request, "urlopen", return_value=self.response(payload)
        ) as urlopen:
            self.call(extra_body={"top_p": 0.9, "model": "override-model"})

        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))

        self.assertEqual(body["top_p"], 0.9)
        # Current behaviour: extra_body may overwrite standard fields.
        self.assertEqual(body["model"], "override-model")


class SuccessParsingTests(ClientTestCase):
    def test_success_payload_is_parsed(self) -> None:
        usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        payload = completion_payload(content="A", usage=usage)

        with mock.patch.object(
            client.urllib.request, "urlopen", return_value=self.response(payload)
        ):
            result = self.call()

        self.assertEqual(result["content"], "A")
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["token_usage"], usage)
        self.assertEqual(result["raw_response"], payload)
        self.assertIsNone(result["error"])
        self.assertIsInstance(result["latency_ms"], float)
        self.assertGreaterEqual(result["latency_ms"], 0.0)

    def test_missing_choices_returns_empty_content(self) -> None:
        with mock.patch.object(
            client.urllib.request, "urlopen", return_value=self.response({})
        ):
            result = self.call()

        self.assertEqual(result["content"], "")
        self.assertIsNone(result["finish_reason"])
        self.assertIsNone(result["token_usage"])
        self.assertIsNone(result["error"])

    def test_missing_message_content_returns_empty_string(self) -> None:
        payload = {"choices": [{"message": {}, "finish_reason": "length"}]}

        with mock.patch.object(
            client.urllib.request, "urlopen", return_value=self.response(payload)
        ):
            result = self.call()

        self.assertEqual(result["content"], "")
        self.assertEqual(result["finish_reason"], "length")

    def test_non_json_success_body_raises_without_retry(self) -> None:
        with mock.patch.object(
            client.urllib.request,
            "urlopen",
            return_value=io.BytesIO(b"<html>not json</html>"),
        ) as urlopen:
            with self.assertRaises(client.LLMClientError) as ctx:
                self.call()

        self.assertIn("non-JSON", str(ctx.exception))
        self.assertEqual(urlopen.call_count, 1)

    @mock.patch.object(client.time, "monotonic", side_effect=[100.0, 100.0])
    def test_last_request_time_is_updated(self, _monotonic) -> None:
        with mock.patch.object(
            client.urllib.request,
            "urlopen",
            return_value=self.response(completion_payload()),
        ):
            self.call()

        self.assertEqual(client._LAST_REQUEST_TIME, 100.0)

    @mock.patch.object(client.time, "monotonic", side_effect=[50.0, 50.0])
    def test_failed_request_updates_last_request_time(self, _monotonic) -> None:
        error = self.http_error(400, b'{"error": "bad request"}')

        with mock.patch.object(
            client.urllib.request, "urlopen", side_effect=[error]
        ):
            with self.assertRaises(client.LLMClientError):
                self.call()

        self.assertEqual(client._LAST_REQUEST_TIME, 50.0)

    @mock.patch.object(
        client.time,
        "monotonic",
        side_effect=[100.0, 100.0, 102.0, 103.0],
    )
    def test_second_request_waits_for_remaining_interval(
        self, _monotonic
    ) -> None:
        client.MIN_REQUEST_INTERVAL = 5.0

        with mock.patch.object(
            client.urllib.request,
            "urlopen",
            side_effect=[
                self.response(completion_payload()),
                self.response(completion_payload()),
            ],
        ) as urlopen:
            self.call()
            self.call()

        self.assertEqual(urlopen.call_count, 2)
        self.sleep.assert_called_once_with(3.0)
        self.assertEqual(client._LAST_REQUEST_TIME, 103.0)


class RetryTests(ClientTestCase):
    def test_non_retryable_http_error_raises_immediately(self) -> None:
        error = self.http_error(400, b'{"error": "bad request"}')

        with mock.patch.object(
            client.urllib.request, "urlopen", side_effect=[error]
        ) as urlopen:
            with self.assertRaises(client.LLMClientError) as ctx:
                self.call()

        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertEqual(urlopen.call_count, 1)
        self.sleep.assert_not_called()

    @mock.patch.object(client.random, "uniform", return_value=0.0)
    def test_429_is_retried_and_success_reports_prior_error(
        self, _uniform
    ) -> None:
        client.MAX_RETRIES = 1
        error = self.http_error(429, headers={"Retry-After": "2"})

        with mock.patch.object(
            client.urllib.request,
            "urlopen",
            side_effect=[error, self.response(completion_payload("recovered"))],
        ) as urlopen:
            result = self.call()

        self.assertEqual(result["content"], "recovered")
        self.assertIsNotNone(result["error"])
        self.assertIn("HTTP 429", result["error"])
        self.assertEqual(urlopen.call_count, 2)

        sleep_values = [call.args[0] for call in self.sleep.call_args_list]
        self.assertIn(2.0, sleep_values)

    @mock.patch.object(client.random, "uniform", return_value=0.0)
    def test_500_retries_exhausted_raises(self, _uniform) -> None:
        client.MAX_RETRIES = 1

        with mock.patch.object(
            client.urllib.request,
            "urlopen",
            side_effect=[self.http_error(500), self.http_error(500)],
        ) as urlopen:
            with self.assertRaises(client.LLMClientError) as ctx:
                self.call()

        self.assertIn("HTTP 500", str(ctx.exception))
        self.assertEqual(urlopen.call_count, 2)

    @mock.patch.object(client.random, "uniform", return_value=0.0)
    def test_connection_error_is_retried_and_success_reports_prior_error(
        self, _uniform
    ) -> None:
        client.MAX_RETRIES = 1

        with mock.patch.object(
            client.urllib.request,
            "urlopen",
            side_effect=[
                urllib.error.URLError("connection refused"),
                self.response(completion_payload("recovered")),
            ],
        ) as urlopen:
            result = self.call()

        self.assertEqual(result["content"], "recovered")
        self.assertIn("Connection error", result["error"])
        self.assertEqual(urlopen.call_count, 2)

    @mock.patch.object(client.random, "uniform", return_value=0.0)
    def test_connection_error_exhausted_raises(self, _uniform) -> None:
        client.MAX_RETRIES = 1

        with mock.patch.object(
            client.urllib.request,
            "urlopen",
            side_effect=[
                urllib.error.URLError("down"),
                urllib.error.URLError("still down"),
            ],
        ) as urlopen:
            with self.assertRaises(client.LLMClientError) as ctx:
                self.call()

        self.assertIn("connection error", str(ctx.exception).lower())
        self.assertEqual(urlopen.call_count, 2)

    def test_minimum_interval_sleeps_when_recent_request(self) -> None:
        client.MIN_REQUEST_INTERVAL = 2.0
        client._LAST_REQUEST_TIME = client.time.monotonic()

        client._enforce_min_interval()

        self.sleep.assert_called_once()
        delay = self.sleep.call_args.args[0]
        self.assertGreater(delay, 0.0)
        self.assertLessEqual(delay, 2.0)


class ServerRetryDelayTests(ClientTestCase):
    def test_numeric_retry_after_header(self) -> None:
        self.assertEqual(
            client._server_retry_delay({"Retry-After": "3"}, ""),
            3.0,
        )

    def test_lowercase_retry_after_header(self) -> None:
        self.assertEqual(
            client._server_retry_delay({"retry-after": "2.5"}, ""),
            2.5,
        )

    def test_http_date_retry_after_is_parsed(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(seconds=30)
        header = format_datetime(future, usegmt=True)
        delay = client._server_retry_delay(
            {"Retry-After": header},
            "",
        )

        self.assertIsNotNone(delay)
        self.assertGreater(delay, 0)
        self.assertLessEqual(delay, 30)

    def test_invalid_retry_after_is_ignored(self) -> None:
        self.assertIsNone(
            client._server_retry_delay({"Retry-After": "not-a-delay"}, "")
        )

    def test_google_retry_info_delay(self) -> None:
        body = json.dumps(
            {"error": {"details": [{"retryDelay": "7.5s"}]}}
        )
        self.assertEqual(client._server_retry_delay(None, body), 7.5)

    def test_no_retry_hint_returns_none(self) -> None:
        self.assertIsNone(client._server_retry_delay(None, "not json"))


if __name__ == "__main__":
    unittest.main()
