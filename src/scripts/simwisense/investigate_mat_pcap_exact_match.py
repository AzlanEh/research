"""Read-only exact-row matching between SiMWiSense MAT batches and PCAP CSI.

Motivation
----------
The MAT ``csi_mon`` values and the decoder output are both integer-valued.
If the MAT batches were produced from the same capture with the same
Broadcom 4366c0 unpacking, each MAT row should equal one decoded 242-subcarrier
packet vector exactly, up to:

  * a global power-of-two scale (different decoder shift convention),
  * an overall sign or conjugation convention,
  * a small subcarrier-column offset.

This script streams the PCAP once, decodes every valid 1084-byte packet, and
looks up exact fingerprints of the requested MAT rows.  A match identifies the
source packet index directly; it does not rely on similarity thresholds.

It is strictly read-only with respect to the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

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

# Central columns used for shifted fingerprint comparison.  Keeping a margin
# lets us test column offsets without running off either end of the vector.
COL_LO = 8
COL_HI = 234


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


def to_int_pairs(csi: np.ndarray) -> np.ndarray:
    """Return a (2, n) int64 array of real/imag components."""
    real = np.rint(csi.real)
    imag = np.rint(csi.imag)
    if not np.array_equal(real, csi.real) or not np.array_equal(imag, csi.imag):
        raise ValueError("CSI contains non-integer values")
    return np.stack(
        [real.astype(np.int64), imag.astype(np.int64)]
    )


def odd_normalize(vec: np.ndarray) -> np.ndarray:
    """Divide out the largest common power of two.

    Handles a global decoder scale difference of 2**k exactly.  A vector whose
    components are all zero is returned unchanged.
    """
    orb = int(np.bitwise_or.reduce(np.abs(vec).ravel()))
    if orb == 0:
        return vec.astype(np.int32)
    trailing = int((orb & -orb).bit_length() - 1)
    return (vec >> trailing).astype(np.int32)


def digest(vec: np.ndarray) -> bytes:
    return hashlib.blake2b(vec.tobytes(), digest_size=16).digest()


def build_targets(
    mat_by_batch: dict[int, np.ndarray],
    shifts: list[int],
    conjugates: list[bool],
    negates: list[bool],
):
    targets = []
    lookup: dict[bytes, list[int]] = {}

    for batch, mat in mat_by_batch.items():
        for row in range(mat.shape[0]):
            base = to_int_pairs(mat[row])
            for conj in conjugates:
                v = base.copy()
                if conj:
                    v[1] *= -1
                for neg in negates:
                    vv = -v if neg else v
                    for shift in shifts:
                        # MAT column j is compared with packet column j+shift.
                        target_slice = vv[:, COL_LO:COL_HI]
                        packet_slice_desc = (
                            COL_LO + shift,
                            COL_HI + shift,
                        )
                        for odd in (False, True):
                            norm = (
                                odd_normalize(target_slice)
                                if odd
                                else target_slice.astype(np.int32)
                            )
                            target = {
                                "batch": batch,
                                "row": row,
                                "conj": conj,
                                "neg": neg,
                                "shift": shift,
                                "odd": odd,
                                "vector": norm,
                                "packet_slice": packet_slice_desc,
                            }
                            index = len(targets)
                            targets.append(target)
                            lookup.setdefault(digest(norm), []).append(index)

    return targets, lookup


def packet_variants(
    base: np.ndarray,
    shifts: list[int],
    conjugates: list[bool],
    negates: list[bool],
):
    """Yield (conj, neg, shift, odd, normalized_vector) for one packet."""
    for conj in conjugates:
        v = base.copy()
        if conj:
            v[1] *= -1
        for neg in negates:
            vv = -v if neg else v
            for shift in shifts:
                lo = COL_LO + shift
                hi = COL_HI + shift
                sub = vv[:, lo:hi]
                yield conj, neg, shift, False, sub.astype(np.int32)
                yield conj, neg, shift, True, odd_normalize(sub)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Read-only exact MAT<->PCAP CSI row matching.  A hit is direct "
            "evidence of packet provenance; no hit is not proof of absence "
            "if the dataset applied undocumented preprocessing."
        )
    )
    parser.add_argument("--pcap", type=Path, default=Path(DEFAULT_PCAP))
    parser.add_argument("--mat-root", type=Path, default=Path(DEFAULT_MAT_ROOT))
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[0, 1003, 1007]
    )
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Stop after this many decoded valid CSI packets (smoke tests).",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        nargs="+",
        default=[-1, 0, 1],
        help="Subcarrier column offsets to test.",
    )
    parser.add_argument(
        "--no-conjugate",
        action="store_true",
        help="Disable the conjugated-imaginary variant.",
    )
    parser.add_argument(
        "--no-negate",
        action="store_true",
        help="Disable the overall-sign variant.",
    )
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.pcap.is_file():
        raise FileNotFoundError(args.pcap)
    if not args.mat_root.is_dir():
        raise NotADirectoryError(args.mat_root)
    if COL_LO + min(args.shifts) < 0 or COL_HI + max(args.shifts) > 242:
        raise ValueError("--shifts exceed the retained 242-column vector")

    mat_by_batch = {
        batch: load_mat_batch(args.mat_root, batch) for batch in args.batches
    }

    conjugates = [False, True] if not args.no_conjugate else [False]
    negates = [False, True] if not args.no_negate else [False]

    targets, lookup = build_targets(
        mat_by_batch, args.shifts, conjugates, negates
    )

    print("=== Exact MAT <-> PCAP row matching (read-only) ===")
    print(f"PCAP:         {args.pcap}")
    print(f"MAT root:     {args.mat_root}")
    print(f"Batches:      {args.batches}")
    print(f"Shifts:       {args.shifts}")
    print(f"Conjugate:    {conjugates}")
    print(f"Negate:       {negates}")
    print(f"Target rows:  {len(targets):,} fingerprints")
    if args.max_packets is not None:
        print(f"Max packets:  {args.max_packets:,}")

    hits: list[dict] = []
    total_packets = 0
    valid_packets = 0
    decoded = 0
    bad_decode = 0
    first_ts = None
    last_ts = None
    t0 = time.time()
    next_progress = args.progress_every

    for timestamp, caplen, origlen, packet in read_pcap(args.pcap):
        total_packets += 1
        if caplen != EXPECTED_PACKET_SIZE or origlen != EXPECTED_PACKET_SIZE:
            continue
        valid_index = valid_packets
        valid_packets += 1

        if args.start_time is not None and timestamp < args.start_time:
            continue
        if args.end_time is not None and timestamp > args.end_time:
            break
        if args.max_packets is not None and decoded >= args.max_packets:
            break

        try:
            csi = decode_packet(packet)[SUBCARRIER_INDICES]
        except (ValueError, OverflowError):
            bad_decode += 1
            continue

        base = to_int_pairs(csi)

        for conj, neg, shift, odd, vec in packet_variants(
            base, args.shifts, conjugates, negates
        ):
            found = lookup.get(digest(vec))
            if not found:
                continue
            for target_index in found:
                target = targets[target_index]
                if (
                    target["conj"] != conj
                    or target["neg"] != neg
                    or target["shift"] != shift
                    or target["odd"] != odd
                ):
                    continue
                if not np.array_equal(target["vector"], vec):
                    continue
                hits.append(
                    {
                        "batch": target["batch"],
                        "row": target["row"],
                        "conj": conj,
                        "neg": neg,
                        "shift": shift,
                        "odd": odd,
                        "valid_index": valid_index,
                        "packet_index": total_packets - 1,
                        "timestamp": timestamp,
                    }
                )

        decoded += 1
        if first_ts is None:
            first_ts = timestamp
        last_ts = timestamp

        if args.progress_every and total_packets >= next_progress:
            elapsed = time.time() - t0
            print(
                f"  scanned {total_packets:,} | decoded {decoded:,} | "
                f"hits {len(hits)} | {decoded / max(elapsed, 1e-9):,.0f} pkt/s",
                flush=True,
            )
            next_progress += args.progress_every

    elapsed = time.time() - t0
    print("\n=== Scan summary ===")
    print(f"Total PCAP packets: {total_packets:,}")
    print(f"Valid 1084-byte:    {valid_packets:,}")
    print(f"Decoded:            {decoded:,}")
    print(f"Bad decodes:        {bad_decode:,}")
    if first_ts is not None:
        print(f"Timestamp range:    {first_ts:.6f} .. {last_ts:.6f}")
    print(f"Elapsed:            {elapsed:.1f} s")
    print(f"Exact hits:         {len(hits):,}")

    if not hits:
        print(
            "\nNo exact row matches. This does not prove the MAT data came "
            "from another capture: undocumented phase/amplitude preprocessing "
            "or a different decoder would also prevent exact matching."
        )
        return 0

    print("\n=== Hits by batch ===")
    for batch in args.batches:
        batch_hits = [h for h in hits if h["batch"] == batch]
        rows = sorted({h["row"] for h in batch_hits})
        print(
            f"\nBatch {batch}: {len(batch_hits)} hits across "
            f"{len(rows)}/{mat_by_batch[batch].shape[0]} rows"
        )
        for variant in sorted(
            {(h["conj"], h["neg"], h["shift"], h["odd"]) for h in batch_hits}
        ):
            vh = [
                h
                for h in batch_hits
                if (h["conj"], h["neg"], h["shift"], h["odd"]) == variant
            ]
            print(
                f"  variant conj={variant[0]} neg={variant[1]} "
                f"shift={variant[2]} odd={variant[3]}: {len(vh)} hits"
            )

        by_row: dict[int, list[int]] = {}
        for h in batch_hits:
            by_row.setdefault(h["row"], []).append(h["valid_index"])
        for row in sorted(by_row):
            idx = sorted(by_row[row])
            deltas = np.diff(idx).tolist()
            print(
                f"  row {row:2d}: packets {idx}"
                + (f" deltas {deltas}" if len(idx) > 1 else "")
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
