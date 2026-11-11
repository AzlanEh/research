"""Regression tests for ``common.llm.response_parser``.

These tests pin the parser's current observable behaviour, including known
limitations (greedy JSON extraction, empty-label-set fallback). They are a
baseline for moving the module later; do not "fix" the expectations here
when the implementation is relocated.
"""

from __future__ import annotations

import json
import unittest

from common.llm.response_parser import (
    extract_json_object,
    parse_response,
    parse_statuses_summary,
)


def json_payload(**overrides) -> str:
    """Return a valid parser payload as a JSON string."""
    payload = {
        "predicted_label": "A",
        "confidence": 0.8,
        "reason": "because",
    }
    payload.update(overrides)
    return json.dumps(payload)


class ExtractJsonObjectTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        text = json_payload()
        self.assertEqual(extract_json_object(text), text)

    def test_plain_json_with_surrounding_whitespace(self) -> None:
        inner = json_payload()
        self.assertEqual(extract_json_object(f"  \n{inner}\n  "), inner)

    def test_fenced_json_with_language_tag(self) -> None:
        inner = json_payload()
        self.assertEqual(extract_json_object(f"```json\n{inner}\n```"), inner)

    def test_fenced_json_without_language_tag(self) -> None:
        inner = json_payload()
        self.assertEqual(extract_json_object(f"```\n{inner}\n```"), inner)

    def test_json_embedded_in_prose(self) -> None:
        inner = json_payload()
        text = f"Here is the result: {inner} End of response."
        self.assertEqual(extract_json_object(text), inner)

    def test_fenced_json_with_trailing_prose(self) -> None:
        inner = json_payload()
        text = f"```json\n{inner}\n```\nThanks"
        self.assertEqual(extract_json_object(text), inner)

    def test_multiple_json_objects_are_captured_greedily(self) -> None:
        # Known limitation: the extraction regex is greedy, so text containing
        # two objects is captured as one invalid JSON blob.
        text = 'Result: {"a": 1} and {"b": 2}'
        self.assertEqual(extract_json_object(text), '{"a": 1} and {"b": 2}')
        self.assertEqual(parse_response(text).status, "json_error")

    def test_truncated_json_returns_none(self) -> None:
        self.assertIsNone(extract_json_object('{"predicted_label": "A"'))

    def test_empty_and_whitespace_return_none(self) -> None:
        for text in ("", "   ", "\n\t"):
            with self.subTest(text=repr(text)):
                self.assertIsNone(extract_json_object(text))


class ParseResponseValidTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        result = parse_response(json_payload())

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.predicted_label, "A")
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.reason, "because")
        self.assertIsNone(result.error_message)

    def test_fenced_json(self) -> None:
        inner = json_payload()
        result = parse_response(f"```json\n{inner}\n```")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.predicted_label, "A")

    def test_json_embedded_in_prose(self) -> None:
        inner = json_payload()
        result = parse_response(f"Classification: {inner}")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.predicted_label, "A")

    def test_unknown_is_accepted_by_default(self) -> None:
        result = parse_response(json_payload(predicted_label="UNKNOWN"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.predicted_label, "UNKNOWN")

    def test_confidence_above_one_is_clamped(self) -> None:
        result = parse_response(json_payload(confidence=1.5))
        self.assertEqual(result.confidence, 1.0)

    def test_confidence_below_zero_is_clamped(self) -> None:
        result = parse_response(json_payload(confidence=-0.5))
        self.assertEqual(result.confidence, 0.0)

    def test_numeric_string_confidence_is_coerced(self) -> None:
        result = parse_response(json_payload(confidence="0.5"))
        self.assertEqual(result.confidence, 0.5)

    def test_boolean_confidence_is_coerced(self) -> None:
        # Current behaviour: bool is a float-compatible value.
        result = parse_response(json_payload(confidence=True))
        self.assertEqual(result.confidence, 1.0)

    def test_infinite_confidence_is_clamped_to_one(self) -> None:
        # Current behaviour: inf survives float() and is clamped.
        result = parse_response(json_payload(confidence=float("inf")))
        self.assertEqual(result.confidence, 1.0)

    def test_raw_content_is_preserved(self) -> None:
        text = f"  prefix {json_payload()} suffix  "
        result = parse_response(text)
        self.assertEqual(result.raw_content, text)

    def test_to_dict_shape(self) -> None:
        result = parse_response(json_payload())
        data = result.to_dict()

        self.assertEqual(
            set(data),
            {
                "parse_status",
                "predicted_label",
                "confidence",
                "reason",
                "raw_content",
                "error_message",
            },
        )
        self.assertEqual(data["parse_status"], "ok")


class ParseResponseErrorTests(unittest.TestCase):
    def test_empty_content(self) -> None:
        result = parse_response("")

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.raw_content, "")
        self.assertIsNone(result.predicted_label)

    def test_whitespace_content_preserves_raw_text(self) -> None:
        result = parse_response("   ")

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.raw_content, "   ")

    def test_no_json_object(self) -> None:
        result = parse_response("I cannot classify this record.")

        self.assertEqual(result.status, "json_error")
        self.assertIsNone(result.predicted_label)
        self.assertIn("No JSON object", result.error_message)

    def test_malformed_json(self) -> None:
        result = parse_response('{"predicted_label": }')

        self.assertEqual(result.status, "json_error")
        self.assertIn("JSON decode error", result.error_message)

    def test_non_object_json(self) -> None:
        result = parse_response("42")

        self.assertEqual(result.status, "schema_error")
        self.assertIn("not an object", result.error_message)

    def test_missing_required_keys(self) -> None:
        result = parse_response('{"predicted_label": "A"}')

        self.assertEqual(result.status, "schema_error")
        self.assertIn("confidence", result.error_message)
        self.assertIn("reason", result.error_message)

    def test_invalid_label(self) -> None:
        result = parse_response(json_payload(predicted_label="D"))

        self.assertEqual(result.status, "label_error")
        self.assertEqual(result.predicted_label, "D")
        self.assertEqual(result.confidence, 0.8)
        self.assertIn("not in allowed labels", result.error_message)

    def test_non_string_label(self) -> None:
        result = parse_response(json_payload(predicted_label=1))

        self.assertEqual(result.status, "label_error")
        self.assertEqual(result.predicted_label, 1)

    def test_explicit_allowed_labels_accepts_custom_label(self) -> None:
        result = parse_response(
            json_payload(predicted_label="Z"),
            allowed_labels={"Z"},
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.predicted_label, "Z")

    def test_explicit_allowed_labels_replaces_defaults(self) -> None:
        result = parse_response(
            json_payload(predicted_label="A"),
            allowed_labels={"Z"},
        )

        self.assertEqual(result.status, "label_error")

    def test_empty_allowed_labels_falls_back_to_defaults(self) -> None:
        # Known limitation: ``allowed_labels or ALLOWED_LABELS`` treats an
        # empty set as "use the default labels".
        result = parse_response(
            json_payload(predicted_label="A"),
            allowed_labels=set(),
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.predicted_label, "A")

    def test_empty_allowed_labels_still_rejects_other_labels(self) -> None:
        result = parse_response(
            json_payload(predicted_label="Z"),
            allowed_labels=set(),
        )

        self.assertEqual(result.status, "label_error")

    def test_non_numeric_confidence(self) -> None:
        result = parse_response(json_payload(confidence="abc"))

        self.assertEqual(result.status, "schema_error")
        self.assertEqual(result.predicted_label, "A")
        self.assertIsNone(result.confidence)
        self.assertIn("not a valid number", result.error_message)

    def test_null_confidence(self) -> None:
        result = parse_response(json_payload(confidence=None))

        self.assertEqual(result.status, "schema_error")
        self.assertIsNone(result.confidence)

    def test_nan_confidence(self) -> None:
        result = parse_response(json_payload(confidence=float("nan")))

        self.assertEqual(result.status, "schema_error")
        self.assertIsNone(result.confidence)

    def test_non_string_reason(self) -> None:
        result = parse_response(json_payload(reason=None))

        self.assertEqual(result.status, "schema_error")
        self.assertEqual(result.predicted_label, "A")
        self.assertEqual(result.confidence, 0.8)
        self.assertIsNone(result.reason)
        self.assertIn("reason field is not a string", result.error_message)

    def test_label_is_validated_before_confidence(self) -> None:
        # Current validation order: an invalid label reports label_error even
        # when confidence is also invalid.
        result = parse_response(
            json_payload(predicted_label="D", confidence="abc")
        )

        self.assertEqual(result.status, "label_error")
        self.assertIsNone(result.confidence)

    def test_label_error_drops_non_string_reason(self) -> None:
        result = parse_response(
            json_payload(predicted_label="D", reason=123)
        )

        self.assertEqual(result.status, "label_error")
        self.assertIsNone(result.reason)


class ParseStatusSummaryTests(unittest.TestCase):
    def test_counts_each_status(self) -> None:
        results = [
            parse_response(json_payload()),
            parse_response(json_payload(predicted_label="D")),
            parse_response(""),
            parse_response("not json"),
        ]

        self.assertEqual(
            parse_statuses_summary(results),
            {"ok": 1, "label_error": 1, "empty": 1, "json_error": 1},
        )

    def test_empty_list(self) -> None:
        self.assertEqual(parse_statuses_summary([]), {})


if __name__ == "__main__":
    unittest.main()
