"""Compatibility tests for ``evaluation.metrics`` and
``evaluation.confusion_matrix``.

These tests pin the current metrics semantics, including the deliberate
difference in how the two modules treat a record whose ``parse_status`` is
not ``"ok"`` but whose predicted label is ``"UNKNOWN"``.
"""

from __future__ import annotations

import unittest

from evaluation.confusion_matrix import (
    compute_confusion_matrix,
    format_confusion_matrix_text,
)
from evaluation.metrics import compute_metrics


def prediction(
    ground_truth: str | None,
    predicted_label: str | None,
    parse_status: str | None = "ok",
) -> dict:
    record = {
        "ground_truth": ground_truth,
        "predicted_label": predicted_label,
    }
    if parse_status is not None:
        record["parse_status"] = parse_status
    return record


class MetricsTests(unittest.TestCase):
    def test_accuracy_uses_evaluable_denominator(self) -> None:
        predictions = [
            prediction("A", "A"),
            prediction("B", "B"),
            prediction("C", "A"),
            prediction("A", "UNKNOWN"),
            prediction("B", None, "client_error"),
        ]

        metrics = compute_metrics(predictions)

        self.assertEqual(metrics["total_predictions"], 5)
        self.assertEqual(metrics["evaluable_predictions"], 3)
        self.assertEqual(metrics["accuracy"], 0.6667)
        self.assertEqual(metrics["unknown_count"], 1)
        self.assertEqual(metrics["unknown_rate"], 0.2)
        self.assertEqual(metrics["parse_error_count"], 1)
        self.assertEqual(metrics["parse_error_rate"], 0.2)
        self.assertEqual(metrics["parse_statuses"], {"ok": 4, "client_error": 1})

    def test_unknown_is_counted_even_when_parse_failed(self) -> None:
        predictions = [prediction("A", "UNKNOWN", "json_error")]

        metrics = compute_metrics(predictions)

        self.assertEqual(metrics["unknown_count"], 1)
        self.assertEqual(metrics["parse_error_count"], 1)
        self.assertEqual(metrics["evaluable_predictions"], 0)
        self.assertEqual(metrics["accuracy"], 0.0)

    def test_missing_parse_status_counts_as_parse_error(self) -> None:
        predictions = [prediction("A", "A", parse_status=None)]

        metrics = compute_metrics(predictions)

        self.assertEqual(metrics["parse_statuses"], {"unknown": 1})
        self.assertEqual(metrics["parse_error_count"], 1)
        self.assertEqual(metrics["evaluable_predictions"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_macro_average_includes_zero_support_labels(self) -> None:
        metrics = compute_metrics([prediction("A", "A")])

        self.assertEqual(metrics["per_class"]["A"]["f1"], 1.0)
        self.assertEqual(metrics["per_class"]["B"]["f1"], 0.0)
        self.assertEqual(metrics["per_class"]["C"]["f1"], 0.0)
        self.assertEqual(metrics["macro_precision"], 0.3333)
        self.assertEqual(metrics["macro_recall"], 0.3333)
        self.assertEqual(metrics["macro_f1"], 0.3333)

    def test_per_class_counts(self) -> None:
        predictions = [
            prediction("A", "A"),
            prediction("A", "B"),
            prediction("B", "A"),
            prediction("C", "C"),
        ]

        metrics = compute_metrics(predictions)

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(
            metrics["per_class"]["A"],
            {
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "support": 2,
                "tp": 1,
                "fp": 1,
                "fn": 1,
            },
        )
        self.assertEqual(
            metrics["per_class"]["B"],
            {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 1,
                "tp": 0,
                "fp": 1,
                "fn": 1,
            },
        )
        self.assertEqual(
            metrics["per_class"]["C"],
            {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "support": 1,
                "tp": 1,
                "fp": 0,
                "fn": 0,
            },
        )
        self.assertEqual(metrics["macro_precision"], 0.5)
        self.assertEqual(metrics["macro_recall"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)

    def test_all_unknown_predictions(self) -> None:
        predictions = [
            prediction("A", "UNKNOWN"),
            prediction("B", "UNKNOWN"),
        ]

        metrics = compute_metrics(predictions)

        self.assertEqual(metrics["evaluable_predictions"], 0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["unknown_count"], 2)
        self.assertEqual(metrics["unknown_rate"], 1.0)

    def test_custom_labels(self) -> None:
        predictions = [
            prediction("X", "X"),
            prediction("Y", "X"),
        ]

        metrics = compute_metrics(predictions, labels=("X", "Y"))

        self.assertEqual(metrics["labels"], ["X", "Y"])
        self.assertEqual(set(metrics["per_class"]), {"X", "Y"})
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_precision"], 0.25)
        self.assertEqual(metrics["macro_recall"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.3333)

    def test_empty_predictions(self) -> None:
        metrics = compute_metrics([])

        self.assertEqual(metrics["total_predictions"], 0)
        self.assertEqual(metrics["evaluable_predictions"], 0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["labels"], ["A", "B", "C"])
        self.assertEqual(set(metrics["per_class"]), {"A", "B", "C"})
        for label in ("A", "B", "C"):
            self.assertEqual(metrics["per_class"][label]["support"], 0)


class ConfusionMatrixTests(unittest.TestCase):
    def test_matrix_contents_and_counts(self) -> None:
        predictions = [
            prediction("A", "A"),
            prediction("A", "B"),
            prediction("B", "A"),
            prediction("C", "C"),
            prediction("A", "UNKNOWN"),
            prediction("B", None, "client_error"),
        ]

        cm = compute_confusion_matrix(predictions)

        self.assertEqual(cm["total"], 6)
        self.assertEqual(cm["evaluable"], 4)
        self.assertEqual(cm["unknown_count"], 1)
        self.assertEqual(cm["parse_error_count"], 1)
        self.assertEqual(
            cm["matrix"],
            {
                "A": {"A": 1, "B": 1, "C": 0},
                "B": {"A": 1, "B": 0, "C": 0},
                "C": {"A": 0, "B": 0, "C": 1},
            },
        )

    def test_unknown_with_failed_status_is_parse_error_only(self) -> None:
        # This is intentionally different from metrics.compute_metrics(), which
        # counts the same record as both UNKNOWN and a parse error.
        predictions = [prediction("A", "UNKNOWN", "json_error")]

        cm = compute_confusion_matrix(predictions)

        self.assertEqual(cm["total"], 1)
        self.assertEqual(cm["evaluable"], 0)
        self.assertEqual(cm["unknown_count"], 0)
        self.assertEqual(cm["parse_error_count"], 1)

    def test_invalid_labels_are_excluded_without_error_counts(self) -> None:
        predictions = [
            prediction("Z", "A"),
            prediction("A", "Z"),
            prediction("A", None),
        ]

        cm = compute_confusion_matrix(predictions)

        self.assertEqual(cm["total"], 3)
        self.assertEqual(cm["evaluable"], 0)
        self.assertEqual(cm["unknown_count"], 0)
        self.assertEqual(cm["parse_error_count"], 0)

    def test_custom_labels(self) -> None:
        predictions = [
            prediction("X", "Y"),
            prediction("Y", "Y"),
        ]

        cm = compute_confusion_matrix(predictions, labels=("X", "Y"))

        self.assertEqual(cm["labels"], ["X", "Y"])
        self.assertEqual(
            cm["matrix"],
            {
                "X": {"X": 0, "Y": 1},
                "Y": {"X": 0, "Y": 1},
            },
        )
        self.assertEqual(cm["evaluable"], 2)

    def test_empty_predictions(self) -> None:
        cm = compute_confusion_matrix([])

        self.assertEqual(cm["total"], 0)
        self.assertEqual(cm["evaluable"], 0)
        self.assertEqual(
            cm["matrix"],
            {
                "A": {"A": 0, "B": 0, "C": 0},
                "B": {"A": 0, "B": 0, "C": 0},
                "C": {"A": 0, "B": 0, "C": 0},
            },
        )

    def test_text_format_is_stable(self) -> None:
        predictions = [
            prediction("A", "A"),
            prediction("A", "B"),
            prediction("B", "A"),
            prediction("B", "B"),
        ]
        cm = compute_confusion_matrix(predictions, labels=("A", "B"))

        expected = (
            "                     A        B\n"
            "Predicted \\ Actual\n"
            "Actual    A :        1        1\n"
            "Actual    B :        1        1\n"
            "\n"
            "Total: 4  Evaluable: 4  UNKNOWN: 0  Parse errors: 0"
        )
        self.assertEqual(format_confusion_matrix_text(cm), expected)


class CrossModuleSemanticsTests(unittest.TestCase):
    def test_unknown_parse_failure_is_counted_differently(self) -> None:
        predictions = [prediction("A", "UNKNOWN", "schema_error")]

        metrics = compute_metrics(predictions)
        cm = compute_confusion_matrix(predictions)

        # metrics: counted as both UNKNOWN and a parse error.
        self.assertEqual(metrics["unknown_count"], 1)
        self.assertEqual(metrics["parse_error_count"], 1)

        # confusion matrix: parse error wins, so it is not counted as UNKNOWN.
        self.assertEqual(cm["unknown_count"], 0)
        self.assertEqual(cm["parse_error_count"], 1)


if __name__ == "__main__":
    unittest.main()
