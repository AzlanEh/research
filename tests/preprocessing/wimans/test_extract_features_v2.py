"""Tests for the v2 WiMANS temporal-feature extractor.

These tests cover the v2 contract only; the v1 extractor and its artifacts
are intentionally left untouched.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocessing.wimans.extract_features_v2 import (
    FEATURE_VERSION,
    LINK_NAMES,
    NUM_UNIQUE_OFF_DIAGONAL_PAIRS,
    extract_manifest_v2,
    extract_recording_features_v2,
)


def make_csi(num_packets: int) -> np.ndarray:
    return np.zeros((num_packets, 3, 3, 30), dtype=np.float64)


def set_link_packets(
    csi: np.ndarray,
    tx: int,
    rx: int,
    packet_values: list[float],
) -> None:
    """Set one link to a constant value per packet across all subcarriers."""
    for packet, value in enumerate(packet_values):
        csi[packet, tx, rx, :] = value


def movement_deltas(csi: np.ndarray, window_size: int, tx: int, rx: int) -> list:
    """Independently compute per-window movement intensity for one link."""
    deltas = []
    for start in range(0, len(csi), window_size):
        window = csi[start:start + window_size, tx, rx, :]
        packet_means = window.mean(axis=1)
        if len(packet_means) > 1:
            deltas.append(float(np.abs(np.diff(packet_means)).mean()))
        else:
            deltas.append(None)
    return deltas


class ValidationTests(unittest.TestCase):
    def test_rejects_non_4d_input(self) -> None:
        with self.assertRaises(ValueError):
            extract_recording_features_v2(np.zeros((10, 3, 30)))

    def test_rejects_wrong_trailing_shape(self) -> None:
        with self.assertRaises(ValueError):
            extract_recording_features_v2(np.zeros((10, 3, 2, 30)))

    def test_rejects_fewer_than_two_packets(self) -> None:
        with self.assertRaises(ValueError):
            extract_recording_features_v2(np.zeros((1, 3, 3, 30)))

    def test_rejects_non_finite_values(self) -> None:
        csi = make_csi(10)
        csi[3, 1, 2, 4] = np.nan
        with self.assertRaises(ValueError):
            extract_recording_features_v2(csi)

    def test_rejects_window_size_below_two(self) -> None:
        with self.assertRaises(ValueError):
            extract_recording_features_v2(make_csi(10), window_size=1)


class TemporalSequenceTests(unittest.TestCase):
    def test_windows_are_chronological_and_keep_partial_window(self) -> None:
        csi = make_csi(10)

        result = extract_recording_features_v2(csi, window_size=4)

        self.assertEqual(result["feature_version"], FEATURE_VERSION)
        self.assertEqual(result["num_packets"], 10)
        self.assertEqual(result["num_windows"], 3)
        self.assertEqual(result["complete_windows"], 2)

        sequence = result["temporal_sequence"]
        self.assertEqual(
            [entry["window_index"] for entry in sequence],
            [0, 1, 2],
        )
        self.assertEqual(
            [entry["start_packet"] for entry in sequence],
            [0, 4, 8],
        )
        self.assertEqual(
            [entry["end_packet"] for entry in sequence],
            [4, 8, 10],
        )
        self.assertEqual(
            [entry["num_packets"] for entry in sequence],
            [4, 4, 2],
        )
        self.assertEqual(
            [entry["is_partial"] for entry in sequence],
            [False, False, True],
        )

    def test_all_nine_links_present_in_every_window(self) -> None:
        result = extract_recording_features_v2(make_csi(10), window_size=4)

        for entry in result["temporal_sequence"]:
            self.assertEqual(list(entry["links"]), LINK_NAMES)
            for link in LINK_NAMES:
                self.assertEqual(
                    set(entry["links"][link]),
                    {"window_mean", "window_std", "temporal_delta"},
                )

    def test_window_statistics_and_temporal_delta(self) -> None:
        csi = make_csi(4)
        # Window 0: packet means 1 and 3 -> mean 2, std 1, delta 2.
        # Window 1: packet means 0 and 0 -> mean 0, std 0, delta 0.
        set_link_packets(csi, 0, 0, [1.0, 3.0, 0.0, 0.0])

        result = extract_recording_features_v2(csi, window_size=2)
        sequence = result["temporal_sequence"]

        first = sequence[0]["links"]["link_00"]
        second = sequence[1]["links"]["link_00"]

        self.assertAlmostEqual(first["window_mean"], 2.0)
        self.assertAlmostEqual(first["window_std"], 1.0)
        self.assertAlmostEqual(first["temporal_delta"], 2.0)

        self.assertAlmostEqual(second["window_mean"], 0.0)
        self.assertAlmostEqual(second["window_std"], 0.0)
        self.assertAlmostEqual(second["temporal_delta"], 0.0)

    def test_single_packet_partial_window_has_null_delta(self) -> None:
        csi = make_csi(9)

        result = extract_recording_features_v2(csi, window_size=4)

        last = result["temporal_sequence"][-1]
        self.assertTrue(last["is_partial"])
        self.assertEqual(last["num_packets"], 1)
        for link in LINK_NAMES:
            self.assertIsNone(last["links"][link]["temporal_delta"])
            self.assertIsNone(result["movement_intensity"][link][-1])

    def test_recording_without_labels_or_ground_truth(self) -> None:
        result = extract_recording_features_v2(make_csi(10), window_size=4)

        forbidden = {
            "label",
            "labels",
            "activity",
            "activities",
            "number_of_users",
            "users",
            "ground_truth",
            "split",
        }
        seen_keys = set()

        def walk(value) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    seen_keys.add(key)
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(result)
        self.assertTrue(forbidden.isdisjoint(seen_keys))

    def test_output_is_deterministic(self) -> None:
        csi = make_csi(10)
        set_link_packets(csi, 1, 1, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        first = extract_recording_features_v2(csi, window_size=4)
        second = extract_recording_features_v2(csi, window_size=4)

        self.assertEqual(
            json.dumps(first, sort_keys=True, allow_nan=False),
            json.dumps(second, sort_keys=True, allow_nan=False),
        )


class MovementIntensityTests(unittest.TestCase):
    def test_movement_intensity_matches_temporal_delta(self) -> None:
        csi = make_csi(10)
        set_link_packets(csi, 0, 0, [0.0, 2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0])
        set_link_packets(csi, 2, 1, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])

        result = extract_recording_features_v2(csi, window_size=4)

        for tx in range(3):
            for rx in range(3):
                link = f"link_{tx}{rx}"
                expected = movement_deltas(csi, 4, tx, rx)
                self.assertEqual(result["movement_intensity"][link], expected)
                actual = [
                    entry["links"][link]["temporal_delta"]
                    for entry in result["temporal_sequence"]
                ]
                self.assertEqual(actual, expected)


class CrossLinkSynchronizationTests(unittest.TestCase):
    def test_matrix_is_symmetric_with_36_unique_pairs(self) -> None:
        result = extract_recording_features_v2(make_csi(8), window_size=2)

        sync = result["cross_link_synchronization"]
        matrix = sync["matrix"]

        self.assertEqual(sync["links"], LINK_NAMES)
        self.assertEqual(sync["unique_off_diagonal_pairs"], 36)
        self.assertEqual(NUM_UNIQUE_OFF_DIAGONAL_PAIRS, 36)
        self.assertEqual(len(matrix), 9)
        self.assertTrue(all(len(row) == 9 for row in matrix))

        for i in range(9):
            for j in range(9):
                self.assertEqual(matrix[i][j], matrix[j][i])

    def test_pearson_correlation_values(self) -> None:
        csi = make_csi(8)
        # Deltas for link_00: [1, 2, 3, 4].
        set_link_packets(csi, 0, 0, [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0])
        # Deltas for link_01: [2, 4, 6, 8] -> perfectly correlated.
        set_link_packets(csi, 0, 1, [0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0])
        # Deltas for link_02: [4, 3, 2, 1] -> perfectly anti-correlated.
        set_link_packets(csi, 0, 2, [0.0, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0])
        # Deltas for link_10: [1, 1, 1, 1] -> constant, undefined.
        set_link_packets(csi, 1, 0, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        result = extract_recording_features_v2(csi, window_size=2)
        matrix = result["cross_link_synchronization"]["matrix"]

        self.assertAlmostEqual(matrix[0][1], 1.0)
        self.assertAlmostEqual(matrix[0][2], -1.0)
        self.assertAlmostEqual(matrix[1][2], -1.0)

        self.assertEqual(matrix[0][0], 1.0)
        self.assertIsNone(matrix[0][3])
        self.assertIsNone(matrix[3][3])

        self.assertEqual(
            result["cross_link_synchronization"]["defined_off_diagonal_pairs"],
            3,
        )

    def test_fewer_than_two_complete_windows_yields_null_correlations(self) -> None:
        csi = make_csi(6)
        set_link_packets(csi, 0, 0, [0.0, 1.0, 0.0, 2.0, 0.0, 3.0])

        result = extract_recording_features_v2(csi, window_size=4)

        self.assertEqual(result["complete_windows"], 1)
        matrix = result["cross_link_synchronization"]["matrix"]
        for row in matrix:
            for value in row:
                self.assertIsNone(value)
        self.assertEqual(
            result["cross_link_synchronization"]["defined_off_diagonal_pairs"],
            0,
        )

    def test_constant_recording_is_json_safe(self) -> None:
        result = extract_recording_features_v2(make_csi(10), window_size=2)

        matrix = result["cross_link_synchronization"]["matrix"]
        for row in matrix:
            for value in row:
                self.assertIsNone(value)

        # Must not raise for NaN/Infinity.
        json.dumps(result, allow_nan=False)


class ManifestTests(unittest.TestCase):
    def test_manifest_output_excludes_ground_truth_and_is_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csi_path = root / "csi.npy"
            np.save(csi_path, make_csi(10))

            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "act_1_1",
                        "group_id": 1,
                        "environment": "classroom",
                        "wifi_band": "2.4",
                        "csi_path": str(csi_path),
                        "number_of_users": 1,
                        "users": ["u1"],
                        "activities": [1],
                    }
                )
                + "\n"
            )

            output_path = root / "features_v2.jsonl"
            extract_manifest_v2(manifest_path, output_path)

            record = json.loads(output_path.read_text().strip())

        self.assertEqual(record["sample_id"], "act_1_1")
        self.assertEqual(record["csi_shape"], [10, 3, 3, 30])
        self.assertEqual(record["features"]["feature_version"], FEATURE_VERSION)
        for key in ("number_of_users", "users", "activities"):
            self.assertNotIn(key, record)


if __name__ == "__main__":
    unittest.main()
