"""load_csi.py

Load a single CSI window (.mat) from the SiMWiSense batch files.

Responsibilities:
  1. Read ``csi_mon`` from a ``batch_*.mat`` file.
  2. Validate shape (expected 50 x 242) and dtype (complex128).
  3. Return the original complex CSI unchanged — no AGC correction,
     no normalization, no gain scaling is applied silently.

Designed to run from ``src/``:
    python preprocessing/load_csi.py --mat path/to/batch_0.mat

Read-only with respect to the source dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.io

# Expected CSI window dimensions (from AGENTS.md / dataset inspection).
EXPECTED_SHAPE = (50, 242)
EXPECTED_DTYPE = np.complex128


@dataclass
class CsiWindow:
    """A single CSI window with its metadata.

    Attributes:
        csi: Complex CSI array, shape (50, 242), dtype complex128.
        source_file: Relative path to the .mat file (e.g.
            "Slots/Train/A_batch/batch_0.mat").
        label: Activity label (single letter, e.g. "A").
        split: Dataset split ("Train", "Val", "Test").
        shape: Tuple of csi.shape.
        dtype: String of csi.dtype.
        is_complex: Whether csi is complex-valued.
    """

    csi: np.ndarray
    source_file: str
    label: str
    split: str
    shape: tuple
    dtype: str
    is_complex: bool


def load_csi_window(mat_path: str, label: str, split: str, source_file: str | None = None) -> CsiWindow:
    """Load and validate one CSI window from a .mat file.

    Args:
        mat_path: Absolute or relative path to the .mat file.
        label: Activity label for this window.
        split: Dataset split for this window.
        source_file: Relative path to the .mat file (e.g.
            "Slots/Train/A_batch/batch_0.mat"). If None, defaults to
            ``os.path.relpath(mat_path)``.

    Returns:
        CsiWindow with the original complex CSI unchanged.

    Raises:
        FileNotFoundError: If ``mat_path`` does not exist.
        KeyError: If the .mat file does not contain ``csi_mon``.
        ValueError: If shape or dtype does not match expectations.
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"CSI mat file not found: {mat_path}")

    mat = scipy.io.loadmat(mat_path)
    if "csi_mon" not in mat:
        raise KeyError(f"'csi_mon' key not found in {mat_path}")

    csi = np.asarray(mat["csi_mon"])

    # Validate shape.
    if csi.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected CSI shape {csi.shape} in {mat_path}; "
            f"expected {EXPECTED_SHAPE}"
        )

    # Validate dtype.
    if csi.dtype != EXPECTED_DTYPE:
        raise ValueError(
            f"Unexpected CSI dtype {csi.dtype} in {mat_path}; "
            f"expected {EXPECTED_DTYPE}"
        )

    if not np.iscomplexobj(csi):
        raise ValueError(f"CSI is not complex-valued in {mat_path}")

    if source_file is None:
        source_file = os.path.relpath(mat_path)

    return CsiWindow(
        csi=csi,
        source_file=source_file,
        label=label,
        split=split,
        shape=tuple(csi.shape),
        dtype=str(csi.dtype),
        is_complex=bool(np.iscomplexobj(csi)),
    )


def load_from_manifest_entry(
    dataset_root: str, entry: dict[str, Any]
) -> CsiWindow:
    """Load a CSI window from a manifest entry dict.

    Args:
        dataset_root: Root directory of the dataset (contains Slots/...).
        entry: Manifest entry with keys "filename", "label", "split".

    Returns:
        CsiWindow with the original complex CSI unchanged.
    """
    mat_path = os.path.join(dataset_root, entry["filename"])
    return load_csi_window(
        mat_path=mat_path,
        label=entry["label"],
        split=entry["split"],
        source_file=entry["filename"],
    )


def amplitude(csi: np.ndarray) -> np.ndarray:
    """Return amplitude (magnitude) of complex CSI, shape (50, 242)."""
    return np.abs(csi)


def phase(csi: np.ndarray) -> np.ndarray:
    """Return phase (angle) of complex CSI in radians, shape (50, 242)."""
    return np.angle(csi)


def real_imag(csi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return real and imaginary components, each shape (50, 242)."""
    return csi.real, csi.imag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and validate one CSI window from a .mat file."
    )
    parser.add_argument(
        "--mat",
        required=True,
        help="Path to a batch_*.mat file.",
    )
    parser.add_argument(
        "--label",
        default="A",
        help="Activity label (default: A).",
    )
    parser.add_argument(
        "--split",
        default="Train",
        help="Dataset split (default: Train).",
    )
    args = parser.parse_args()

    window = load_csi_window(args.mat, args.label, args.split)
    print(f"Loaded: {window.source_file}")
    print(f"  label: {window.label}  split: {window.split}")
    print(f"  shape: {window.shape}  dtype: {window.dtype}  complex: {window.is_complex}")
    amps = amplitude(window.csi)
    print(f"  amplitude: mean={amps.mean():.4f} std={amps.std():.4f} "
          f"min={amps.min():.4f} max={amps.max():.4f}")


if __name__ == "__main__":
    main()
