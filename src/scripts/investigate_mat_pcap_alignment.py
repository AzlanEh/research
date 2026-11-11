"""Read-only MAT <-> PCAP CSI alignment investigation.

Purpose
-------
Determine whether CSI windows stored in SiMWiSense ``batch_*.mat`` files can
be located in the matching ``A.pcap`` capture.

Design
------
Pass 1 streams the PCAP once, decodes valid CSI packets, and retains only:
  * a compact normalized amplitude feature vector per packet,
  * timestamp, original packet index, and sequential valid-CSI index.

The compact features are used to score candidate windows for each MAT batch
over a range of packet strides.  Pass 2 re-streams the PCAP and decodes the
full 242-subcarrier CSI only for the neighbourhoods of the top candidates,
then verifies them with the full metrics.

This script is strictly read-only with respect to the dataset.  It never
modifies source CSI and refuses to write output under /mnt/external_disk.

A high similarity score is a *candidate match*, not proof of provenance.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize_scalar

from csi_decoder import (
    EXPECTED_PACKET_SIZE,
    SUBCARRIER_INDICES,
    decode_packet,
    read_pcap,
)

DEFAULT_PCAP = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/fine_grained/"
    "Classroom/80MHz/3mo/m1/CSI_pcap/A.pcap"
)
DEFAULT_MAT_ROOT = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/Data/fine_grained/"
    "Classroom/80MHz/3mo/m1/Slots/Test/A_batch"
)
DATASET_MOUNT_PREFIX = "/mnt/external_disk"
EPS = 1e-12


# ---------------------------------------------------------------------------
# Loading and features
# ---------------------------------------------------------------------------

def load_mat_batch(mat_root: Path, batch: int) -> np.ndarray:
    path = mat_root / f"batch_{batch}.mat"
    if not path.is_file():
        raise FileNotFoundError(path)

    data = loadmat(path)
    if "csi_mon" not in data:
        raise KeyError(f"'csi_mon' not found in {path}")

    csi = np.asarray(data["csi_mon"])
    if csi.shape[1] != 242 and csi.shape[0] == 242:
        csi = csi.T

    if csi.ndim != 2 or csi.shape[1] != 242:
        raise ValueError(f"Unexpected MAT shape {csi.shape} in {path}")

    return csi


def feature_subcarrier_indices(count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("--feature-subcarriers must be >= 2")
    if count > len(SUBCARRIER_INDICES):
        raise ValueError(
            f"--feature-subcarriers cannot exceed {len(SUBCARRIER_INDICES)}"
        )
    idx = np.linspace(
        0, len(SUBCARRIER_INDICES) - 1, count, dtype=int
    )
    return np.unique(idx)


def amplitude_feature(selected_csi: np.ndarray, sub_idx: np.ndarray) -> np.ndarray:
    """Gain-invariant amplitude-shape feature."""
    amp = np.abs(selected_csi[sub_idx]).astype(np.float64)
    amp = amp - amp.mean()
    norm = float(np.linalg.norm(amp))
    if norm < EPS:
        return amp.astype(np.float32)
    return (amp / norm).astype(np.float32)


def mat_feature_matrix(batch_csi: np.ndarray, sub_idx: np.ndarray) -> np.ndarray:
    return np.stack(
        [amplitude_feature(row, sub_idx) for row in batch_csi]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Pass 1: stream PCAP and retain compact features + metadata
# ---------------------------------------------------------------------------

def _grow(array: np.ndarray, capacity: int, new_capacity: int) -> np.ndarray:
    out = np.empty((new_capacity,) + array.shape[1:], dtype=array.dtype)
    out[:capacity] = array[:capacity]
    return out


def scan_pcap(
    pcap: Path,
    sub_idx: np.ndarray,
    start_time: float | None,
    end_time: float | None,
    max_packets: int | None,
    progress_every: int = 100_000,
):
    capacity = max(1024, max_packets or 1_200_000)
    n_features = len(sub_idx)

    feats = np.empty((capacity, n_features), dtype=np.float32)
    timestamps = np.empty(capacity, dtype=np.float64)
    packet_indices = np.empty(capacity, dtype=np.int64)
    valid_indices = np.empty(capacity, dtype=np.int64)
    amp_max = np.empty(capacity, dtype=np.float32)

    total_packets = 0
    valid_packets = 0
    skipped_length = 0
    decoded = 0
    bad_decode = 0
    nonfinite = 0
    all_zero = 0

    first_ts = None
    last_ts = None
    last_packet_index = -1

    t_start = time.time()
    next_progress = progress_every

    for timestamp, caplen, origlen, packet in read_pcap(pcap):
        packet_index = total_packets
        total_packets += 1
        last_packet_index = packet_index

        if caplen != EXPECTED_PACKET_SIZE or origlen != EXPECTED_PACKET_SIZE:
            skipped_length += 1
            continue

        valid_index = valid_packets
        valid_packets += 1

        if start_time is not None and timestamp < start_time:
            continue
        if end_time is not None and timestamp > end_time:
            break
        if max_packets is not None and decoded >= max_packets:
            break

        try:
            csi = decode_packet(packet)[SUBCARRIER_INDICES]
        except (ValueError, OverflowError):
            bad_decode += 1
            continue

        if not np.all(np.isfinite(csi)):
            nonfinite += 1
            continue
        if np.all(csi == 0):
            all_zero += 1

        amp = np.abs(csi)
        peak = float(amp.max())

        if decoded >= capacity:
            new_capacity = capacity * 2
            feats = _grow(feats, capacity, new_capacity)
            timestamps = _grow(timestamps, capacity, new_capacity)
            packet_indices = _grow(packet_indices, capacity, new_capacity)
            valid_indices = _grow(valid_indices, capacity, new_capacity)
            amp_max = _grow(amp_max, capacity, new_capacity)
            capacity = new_capacity

        feats[decoded] = amplitude_feature(csi, sub_idx)
        timestamps[decoded] = timestamp
        packet_indices[decoded] = packet_index
        valid_indices[decoded] = valid_index
        amp_max[decoded] = peak
        decoded += 1

        if first_ts is None:
            first_ts = timestamp
        last_ts = timestamp

        if total_packets >= next_progress:
            elapsed = time.time() - t_start
            rate = decoded / elapsed if elapsed > 0 else 0.0
            print(
                f"  scanned {total_packets:,} packets | decoded {decoded:,} "
                f"| {rate:,.0f} packets/s",
                flush=True,
            )
            next_progress += progress_every

    if decoded == 0:
        raise RuntimeError("No valid CSI packets decoded in the requested range")

    stats = {
        "total_packets": total_packets,
        "valid_packets": valid_packets,
        "skipped_length": skipped_length,
        "decoded": decoded,
        "bad_decode": bad_decode,
        "nonfinite": nonfinite,
        "all_zero": all_zero,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "last_packet_index": last_packet_index,
        "elapsed": time.time() - t_start,
    }

    return (
        feats[:decoded],
        timestamps[:decoded],
        packet_indices[:decoded],
        valid_indices[:decoded],
        amp_max[:decoded],
        stats,
    )


# ---------------------------------------------------------------------------
# Candidate search
# ---------------------------------------------------------------------------

def top_offsets(
    scores: np.ndarray, count: int, span: int
) -> list[int]:
    """Lowest scores with non-overlapping offsets."""
    if scores.size == 0:
        return []

    take = min(len(scores), max(count * 8, count))
    part = np.argpartition(scores, take - 1)[:take]
    part = part[np.argsort(scores[part])]

    chosen: list[int] = []
    for offset in part:
        o = int(offset)
        if all(abs(o - c) >= span for c in chosen):
            chosen.append(o)
        if len(chosen) >= count:
            break
    return chosen


def robust_stats(scores: np.ndarray) -> tuple[float, float]:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    return median, mad


def search_candidates(
    feats: np.ndarray,
    mat_features: dict[int, np.ndarray],
    strides: list[int],
    top_per_stride: int,
    top_candidates: int,
):
    n = feats.shape[0]
    results = {}

    for batch, mat_feat in mat_features.items():
        if mat_feat.shape[0] != 50:
            raise ValueError(f"Batch {batch}: expected 50 MAT samples")

        # G[t, i] = cosine similarity between packet t and MAT sample i.
        G = feats @ mat_feat.T  # (n, 50), float32

        batch_candidates = []
        summaries = {}

        for stride in strides:
            length = n - 49 * stride
            if length <= 0:
                continue

            scores = np.zeros(length, dtype=np.float32)
            for i in range(50):
                scores += 1.0 - G[i * stride : i * stride + length, i]
            scores /= 50.0

            median, mad = robust_stats(scores)
            span = 49 * stride + 1

            for offset in top_offsets(scores, top_per_stride, span):
                best = float(scores[offset])
                count_le = int(np.count_nonzero(scores <= best))
                p_emp = (count_le + 1) / (length + 1)
                z = (median - best) / (1.4826 * mad + EPS)
                batch_candidates.append(
                    {
                        "batch": batch,
                        "stride": stride,
                        "offset": offset,
                        "score": best,
                        "p_empirical": float(p_emp),
                        "robust_z": float(z),
                        "span": span,
                        "windows": length,
                        "median": median,
                    }
                )

            summaries[stride] = {
                "windows": length,
                "min": float(scores.min()),
                "p01": float(np.percentile(scores, 1)),
                "median": median,
            }

        # Non-max suppression across strides (packet spans must not overlap).
        batch_candidates.sort(key=lambda c: c["score"])
        selected = []
        for cand in batch_candidates:
            start = cand["offset"]
            end = start + 49 * cand["stride"]
            overlaps = False
            for other in selected:
                o_start = other["offset"]
                o_end = o_start + 49 * other["stride"]
                if start <= o_end and o_start <= end:
                    overlaps = True
                    break
            if overlaps:
                continue
            selected.append(cand)
            if len(selected) >= top_candidates:
                break

        results[batch] = {"candidates": selected, "summaries": summaries}

    return results


# ---------------------------------------------------------------------------
# Verification metrics
# ---------------------------------------------------------------------------

def amplitude_residual(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.abs(a)
    bb = np.abs(b)
    gain = float(np.dot(aa, bb) / (np.dot(aa, aa) + EPS))
    return float(
        np.linalg.norm(bb - gain * aa) / (np.linalg.norm(bb) + EPS)
    )


def complex_residual(a: np.ndarray, b: np.ndarray) -> float:
    alpha = np.vdot(a, b) / (np.vdot(a, a) + EPS)
    rab = np.linalg.norm(b - alpha * a) / (np.linalg.norm(b) + EPS)
    beta = np.vdot(b, a) / (np.vdot(b, b) + EPS)
    rba = np.linalg.norm(a - beta * b) / (np.linalg.norm(a) + EPS)
    return float((rab + rba) / 2.0)


def phase_coherence(a: np.ndarray, b: np.ndarray) -> float:
    phase = np.angle(b * np.conj(a))
    weights = np.abs(a) * np.abs(b)
    weights = weights / (weights.sum() + EPS)
    return float(abs(np.sum(weights * np.exp(1j * phase))))


def phase_slope_residual(
    a: np.ndarray, b: np.ndarray, k: np.ndarray
) -> tuple[float, float]:
    """Residual after a complex scalar plus linear phase slope."""
    weights = np.abs(a) * np.abs(b)
    cross = b * np.conj(a)

    def objective(beta: float) -> float:
        return -float(
            abs(np.sum(weights * cross * np.exp(-1j * beta * k)))
            / (weights.sum() + EPS)
        )

    result = minimize_scalar(
        objective,
        bounds=(-0.5, 0.5),
        method="bounded",
        options={"xatol": 1e-4},
    )
    beta = float(result.x)

    model = a * np.exp(1j * beta * k)
    scalar = np.vdot(model, b) / (np.vdot(model, model) + EPS)
    residual = np.linalg.norm(b - scalar * model) / (
        np.linalg.norm(b) + EPS
    )
    return float(residual), beta


def amplitude_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.abs(a)
    bb = np.abs(b)
    if aa.std() < EPS or bb.std() < EPS:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def pair_metrics(a: np.ndarray, b: np.ndarray, k: np.ndarray):
    phase_res, beta = phase_slope_residual(a, b, k)
    return {
        "amplitude_residual": amplitude_residual(a, b),
        "complex_residual": complex_residual(a, b),
        "phase_slope_residual": phase_res,
        "phase_slope": beta,
        "coherence": phase_coherence(a, b),
        "amplitude_correlation": amplitude_correlation(a, b),
    }


def best_circular_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    a0 = a - a.mean()
    b0 = b - b.mean()
    if a0.std() < EPS or b0.std() < EPS:
        return 0.0, 0
    best_corr = -1.0
    best_shift = 0
    for shift in range(len(a0)):
        corr = float(np.corrcoef(a0, np.roll(b0, shift))[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return best_corr, best_shift


def verify_candidate(
    mat: np.ndarray,
    pcap: np.ndarray,
    k: np.ndarray,
):
    rows = [pair_metrics(mat[i], pcap[i], k) for i in range(50)]

    def aggregate(key: str) -> dict:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    pooled_amp = amplitude_residual(mat.ravel(), pcap.ravel())
    pooled_complex = complex_residual(mat.ravel(), pcap.ravel())

    mat_profile = np.mean(np.abs(mat), axis=0)
    pcap_profile = np.mean(np.abs(pcap), axis=0)
    profile_corr = amplitude_correlation(mat_profile, pcap_profile)
    reverse_corr = amplitude_correlation(mat_profile, pcap_profile[::-1])
    shift_corr, shift = best_circular_shift(mat_profile, pcap_profile)

    profile_gain = float(
        np.dot(mat_profile, pcap_profile)
        / (np.dot(mat_profile, mat_profile) + EPS)
    )
    profile_residual = float(
        np.linalg.norm(pcap_profile - profile_gain * mat_profile)
        / (np.linalg.norm(pcap_profile) + EPS)
    )

    return {
        "per_pair": rows,
        "amplitude_residual": aggregate("amplitude_residual"),
        "complex_residual": aggregate("complex_residual"),
        "phase_slope_residual": aggregate("phase_slope_residual"),
        "coherence": aggregate("coherence"),
        "amplitude_correlation": aggregate("amplitude_correlation"),
        "pooled_amplitude_residual": pooled_amp,
        "pooled_complex_residual": pooled_complex,
        "profile_correlation": profile_corr,
        "profile_correlation_reversed": reverse_corr,
        "profile_correlation_best_shift": shift_corr,
        "profile_best_shift": shift,
        "profile_residual": profile_residual,
    }


# ---------------------------------------------------------------------------
# Pass 2: decode only candidate neighbourhoods
# ---------------------------------------------------------------------------

def refine_candidates(
    pcap: Path,
    candidates: list[dict],
    mat_by_batch: dict[int, np.ndarray],
    valid_indices: np.ndarray,
    margin: int,
):
    needed: set[int] = set()
    max_needed = 0

    for cand in candidates:
        batch = cand["batch"]
        offset = cand["offset"]
        stride = cand["stride"]
        for i in range(50):
            base = int(valid_indices[offset + i * stride])
            for delta in range(-margin, margin + 1):
                idx = base + delta
                if idx >= 0:
                    needed.add(idx)
                    max_needed = max(max_needed, idx)

    store: dict[int, tuple[float, int, np.ndarray]] = {}
    valid_index = -1
    packet_index = -1

    print(
        f"\nPass 2: re-streaming to decode {len(needed):,} candidate packets...",
        flush=True,
    )

    for timestamp, caplen, origlen, packet in read_pcap(pcap):
        packet_index += 1
        if caplen != EXPECTED_PACKET_SIZE or origlen != EXPECTED_PACKET_SIZE:
            continue
        valid_index += 1
        if valid_index > max_needed:
            break
        if valid_index not in needed:
            continue
        try:
            csi = decode_packet(packet)[SUBCARRIER_INDICES]
        except (ValueError, OverflowError):
            continue
        store[valid_index] = (timestamp, packet_index, csi)

    verified: dict[tuple[int, int, int], dict] = {}
    k = SUBCARRIER_INDICES.astype(np.float64)

    for cand in candidates:
        batch = cand["batch"]
        offset = cand["offset"]
        stride = cand["stride"]
        indices = [int(valid_indices[offset + i * stride]) for i in range(50)]
        if any(idx not in store for idx in indices):
            continue
        pcap_window = np.stack([store[idx][2] for idx in indices])
        mat = mat_by_batch[batch][:50]
        metrics = verify_candidate(mat, pcap_window, k)
        metrics["timestamps"] = [store[idx][0] for idx in indices]
        metrics["packet_indices"] = [store[idx][1] for idx in indices]
        metrics["valid_indices"] = indices
        verified[(batch, stride, offset)] = metrics

    return verified, store


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_scan_report(stats: dict, amp_max: np.ndarray, feature_count: int):
    duration = (
        stats["last_timestamp"] - stats["first_timestamp"]
        if stats["first_timestamp"] is not None
        else 0.0
    )
    print("\n=== Scan summary ===")
    print(f"Total PCAP packets:      {stats['total_packets']:,}")
    print(f"Valid 1084-byte packets: {stats['valid_packets']:,}")
    print(f"Skipped wrong length:    {stats['skipped_length']:,}")
    print(f"Decoded CSI packets:     {stats['decoded']:,}")
    print(f"Bad decodes:             {stats['bad_decode']:,}")
    print(f"Non-finite samples:      {stats['nonfinite']:,}")
    print(f"All-zero samples:        {stats['all_zero']:,}")
    print(f"Feature subcarriers:     {feature_count}")
    print(f"Duration:                {duration:.3f} s")
    if stats["decoded"] > 1 and duration > 0:
        print(f"Mean packet interval:    {duration / (stats['decoded'] - 1) * 1000:.3f} ms")
    print(f"Scan time:               {stats['elapsed']:.1f} s")

    if amp_max.size:
        print("\n=== Amplitude sanity (length-valid packets only) ===")
        print(
            f"Per-packet peak amplitude: median={np.median(amp_max):.3f} "
            f"p99={np.percentile(amp_max, 99):.3f} max={amp_max.max():.3f}"
        )
        threshold = 10.0 * (np.median(amp_max) + EPS)
        suspicious = int(np.count_nonzero(amp_max > threshold))
        print(
            f"Packets with peak > 10x median: {suspicious:,} "
            f"(threshold {threshold:.3f})"
        )
        print(
            "A length match is not proof of valid CSI; these are candidates, "
            "not provenance conclusions."
        )


def print_batch_report(
    batch: int,
    search: dict,
    verified: dict,
    mat_shape: tuple[int, int],
):
    print(f"\n{'=' * 78}")
    print(f"Batch {batch}  (MAT shape {mat_shape})")
    print(f"{'=' * 78}")

    summaries = search["summaries"]
    for stride, summary in summaries.items():
        print(
            f"  stride {stride}: windows={summary['windows']:,} "
            f"min={summary['min']:.5f} p01={summary['p01']:.5f} "
            f"median={summary['median']:.5f}"
        )

    candidates = search["candidates"]
    if not candidates:
        print("  No candidate windows found.")
        return

    for rank, cand in enumerate(candidates, 1):
        key = (batch, cand["stride"], cand["offset"])
        metrics = verified.get(key)
        print(
            f"\n  Candidate {rank}: stride={cand['stride']} "
            f"offset={cand['offset']:,} "
            f"score={cand['score']:.5f} "
            f"p_emp={cand['p_empirical']:.2e} "
            f"robust_z={cand['robust_z']:.1f}"
        )

        if metrics is None:
            print("    verification: incomplete (packets missing)")
            continue

        print(
            f"    valid indices: {metrics['valid_indices'][0]}.."
            f"{metrics['valid_indices'][-1]} | "
            f"packet indices: {metrics['packet_indices'][0]}.."
            f"{metrics['packet_indices'][-1]}"
        )
        print(
            f"    timestamps: {metrics['timestamps'][0]:.6f}.."
            f"{metrics['timestamps'][-1]:.6f} "
            f"({metrics['timestamps'][-1] - metrics['timestamps'][0]:.3f} s)"
        )
        print(
            "    amplitude residual: "
            f"mean={metrics['amplitude_residual']['mean']:.4f} "
            f"median={metrics['amplitude_residual']['median']:.4f} "
            f"max={metrics['amplitude_residual']['max']:.4f}"
        )
        print(
            "    complex residual:   "
            f"mean={metrics['complex_residual']['mean']:.4f} "
            f"median={metrics['complex_residual']['median']:.4f} "
            f"max={metrics['complex_residual']['max']:.4f}"
        )
        print(
            "    phase-slope residual: "
            f"mean={metrics['phase_slope_residual']['mean']:.4f} "
            f"median={metrics['phase_slope_residual']['median']:.4f} "
            f"max={metrics['phase_slope_residual']['max']:.4f}"
        )
        print(
            "    coherence:          "
            f"mean={metrics['coherence']['mean']:.4f} "
            f"min={metrics['coherence']['min']:.4f}"
        )
        print(
            "    amplitude corr:     "
            f"mean={metrics['amplitude_correlation']['mean']:.4f} "
            f"min={metrics['amplitude_correlation']['min']:.4f}"
        )
        print(
            "    pooled residuals:   "
            f"amp={metrics['pooled_amplitude_residual']:.4f} "
            f"complex={metrics['pooled_complex_residual']:.4f}"
        )
        print(
            "    subcarrier profile: "
            f"corr={metrics['profile_correlation']:.4f} "
            f"reversed={metrics['profile_correlation_reversed']:.4f} "
            f"best_shift={metrics['profile_correlation_best_shift']:.4f} "
            f"(shift {metrics['profile_best_shift']}) "
            f"residual={metrics['profile_residual']:.4f}"
        )


def write_csv(path: Path, rows: list[dict]):
    resolved = path.resolve()
    if str(resolved).startswith(DATASET_MOUNT_PREFIX):
        raise SystemExit(
            "Refusing to write output under the dataset mount: "
            f"{resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "batch",
        "rank",
        "stride",
        "offset",
        "score",
        "p_empirical",
        "robust_z",
        "valid_start",
        "valid_end",
        "packet_start",
        "packet_end",
        "timestamp_start",
        "timestamp_end",
        "amp_res_mean",
        "amp_res_median",
        "complex_res_mean",
        "phase_slope_res_mean",
        "coherence_mean",
        "amp_corr_mean",
        "pooled_amp_res",
        "pooled_complex_res",
        "profile_corr",
        "profile_corr_reversed",
        "profile_corr_best_shift",
        "profile_best_shift",
        "profile_residual",
    ]

    with resolved.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"\nSaved candidate table: {resolved}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Read-only MAT<->PCAP CSI alignment search. "
            "Candidate matches are hypotheses, not provenance proof."
        )
    )
    parser.add_argument("--pcap", type=Path, default=Path(DEFAULT_PCAP))
    parser.add_argument("--mat-root", type=Path, default=Path(DEFAULT_MAT_ROOT))
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[0, 1003, 1007],
        help="MAT batch IDs to search for.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Only decode packets with timestamp >= this value.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Stop decoding packets with timestamp > this value.",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Maximum valid CSI packets to decode in pass 1 (smoke tests).",
    )
    parser.add_argument(
        "--feature-subcarriers",
        type=int,
        default=64,
        help="Number of uniformly spaced subcarriers for the coarse feature.",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help=(
            "Packet strides to test between consecutive MAT samples. "
            "Stride 1 = consecutive packets; larger strides test downsampling."
        ),
    )
    parser.add_argument("--top-per-stride", type=int, default=3)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument(
        "--verify-margin",
        type=int,
        default=2,
        help="Extra valid-CSI packets to decode around each candidate.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional candidate table path (refused under /mnt/external_disk).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Progress print interval in total PCAP packets.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    pcap = args.pcap
    mat_root = args.mat_root

    if not pcap.is_file():
        raise FileNotFoundError(pcap)
    if not mat_root.is_dir():
        raise NotADirectoryError(mat_root)
    if any(stride < 1 for stride in args.strides):
        raise ValueError("--strides values must be >= 1")

    sub_idx = feature_subcarrier_indices(args.feature_subcarriers)

    print("=== MAT <-> PCAP alignment (read-only) ===")
    print(f"PCAP:     {pcap}")
    print(f"MAT root: {mat_root}")
    print(f"Batches:  {args.batches}")
    print(f"Strides:  {args.strides}")
    print(f"Features: {len(sub_idx)} subcarriers")
    if args.start_time is not None or args.end_time is not None:
        print(f"Time window: {args.start_time} .. {args.end_time}")
    if args.max_packets is not None:
        print(f"Max decoded packets: {args.max_packets:,}")

    mat_by_batch: dict[int, np.ndarray] = {}
    mat_features: dict[int, np.ndarray] = {}
    for batch in args.batches:
        csi = load_mat_batch(mat_root, batch)
        mat_by_batch[batch] = csi
        mat_features[batch] = mat_feature_matrix(csi, sub_idx)
        print(f"Loaded MAT batch {batch}: shape={csi.shape}")

    print(
        "\nPass 1: streaming PCAP and retaining compact features "
        "(~1.7k packets/s expected)...",
        flush=True,
    )
    feats, timestamps, packet_indices, valid_indices, amp_max, stats = scan_pcap(
        pcap=pcap,
        sub_idx=sub_idx,
        start_time=args.start_time,
        end_time=args.end_time,
        max_packets=args.max_packets,
        progress_every=args.progress_every,
    )

    print_scan_report(stats, amp_max, len(sub_idx))

    print("\n=== Candidate search ===")
    results = search_candidates(
        feats=feats,
        mat_features=mat_features,
        strides=args.strides,
        top_per_stride=args.top_per_stride,
        top_candidates=args.top_candidates,
    )

    selected_candidates = []
    for batch in args.batches:
        selected_candidates.extend(results[batch]["candidates"])

    verified, _store = refine_candidates(
        pcap=pcap,
        candidates=selected_candidates,
        mat_by_batch=mat_by_batch,
        valid_indices=valid_indices,
        margin=args.verify_margin,
    )

    csv_rows = []
    for batch in args.batches:
        print_batch_report(
            batch,
            results[batch],
            verified,
            mat_by_batch[batch].shape,
        )
        for rank, cand in enumerate(results[batch]["candidates"], 1):
            key = (batch, cand["stride"], cand["offset"])
            metrics = verified.get(key)
            row = {
                "batch": batch,
                "rank": rank,
                "stride": cand["stride"],
                "offset": cand["offset"],
                "score": cand["score"],
                "p_empirical": cand["p_empirical"],
                "robust_z": cand["robust_z"],
            }
            if metrics is not None:
                row.update(
                    {
                        "valid_start": metrics["valid_indices"][0],
                        "valid_end": metrics["valid_indices"][-1],
                        "packet_start": metrics["packet_indices"][0],
                        "packet_end": metrics["packet_indices"][-1],
                        "timestamp_start": metrics["timestamps"][0],
                        "timestamp_end": metrics["timestamps"][-1],
                        "amp_res_mean": metrics["amplitude_residual"]["mean"],
                        "amp_res_median": metrics["amplitude_residual"]["median"],
                        "complex_res_mean": metrics["complex_residual"]["mean"],
                        "phase_slope_res_mean": metrics["phase_slope_residual"]["mean"],
                        "coherence_mean": metrics["coherence"]["mean"],
                        "amp_corr_mean": metrics["amplitude_correlation"]["mean"],
                        "pooled_amp_res": metrics["pooled_amplitude_residual"],
                        "pooled_complex_res": metrics["pooled_complex_residual"],
                        "profile_corr": metrics["profile_correlation"],
                        "profile_corr_reversed": metrics["profile_correlation_reversed"],
                        "profile_corr_best_shift": metrics[
                            "profile_correlation_best_shift"
                        ],
                        "profile_best_shift": metrics["profile_best_shift"],
                        "profile_residual": metrics["profile_residual"],
                    }
                )
            csv_rows.append(row)

    print("\nInterpretation guardrails:")
    print("- Low residual / high correlation means a candidate match only.")
    print("- MAT data may be resampled, phase-sanitized, or otherwise")
    print("  preprocessed relative to the raw PCAP; that remains a hypothesis.")
    print("- Do not treat any candidate as proven dataset provenance.")

    if args.save_csv is not None:
        write_csv(args.save_csv, csv_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
