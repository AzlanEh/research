import argparse
import struct
from pathlib import Path

import numpy as np


PCAP_PATH = (
    "/mnt/external_disk/SiMWiSense/SiMWiSense/"
    "Data/fine_grained/Classroom/80MHz/3mo/m1/"
    "CSI_pcap/A.pcap"
)

CSI_OFFSET = 60       # Bytes; MATLAB HOFFSET=16 (1-based uint32 words)
NFFT = 256
EXPECTED_PACKET_SIZE = CSI_OFFSET + NFFT * 4

# MATLAB: non_zero = [7:128, 132:251]
# Convert MATLAB's 1-based indices to Python's 0-based indices.
SUBCARRIER_INDICES = np.r_[
    np.arange(6, 128),
    np.arange(131, 251),
]


def read_pcap(path):
    """Stream packets from a classic PCAP file."""
    with open(path, "rb") as f:
        magic = f.read(4)

        formats = {
            b"\xd4\xc3\xb2\xa1": "<",  # Little-endian, microseconds
            b"\xa1\xb2\xc3\xd4": ">",  # Big-endian, microseconds
            b"\x4d\x3c\xb2\xa1": "<",  # Little-endian, nanoseconds
            b"\xa1\xb2\x3c\x4d": ">",  # Big-endian, nanoseconds
        }

        if magic not in formats:
            raise ValueError(f"Unsupported PCAP magic: {magic.hex()}")

        endian = formats[magic]

        # Remaining 20 bytes of the 24-byte global header.
        header = f.read(20)
        if len(header) != 20:
            raise ValueError("Incomplete PCAP global header")

        major, minor, thiszone, sigfigs, snaplen, network = (
            struct.unpack(endian + "HHiiII", header)
        )

        print(f"PCAP version: {major}.{minor}")
        print(f"Snaplen: {snaplen}, link type: {network}")

        record_struct = struct.Struct(endian + "IIII")

        while True:
            record = f.read(16)

            if not record:
                break

            if len(record) != 16:
                raise ValueError("Incomplete PCAP packet header")

            ts_sec, ts_fraction, incl_len, orig_len = (
                record_struct.unpack(record)
            )

            packet = f.read(incl_len)

            if len(packet) != incl_len:
                raise ValueError("Incomplete PCAP packet data")

            timestamp = ts_sec + ts_fraction / (
                1_000_000_000 if magic in (
                    b"\x4d\x3c\xb2\xa1",
                    b"\xa1\xb2\x3c\x4d",
                ) else 1_000_000
            )

            yield timestamp, incl_len, orig_len, packet



def decode_4366c0(words):
    """Port of unpack_float.c for Broadcom 4366c0."""

    words = np.asarray(words, dtype=np.uint32)

    if words.shape != (NFFT,):
        raise ValueError(f"Expected {NFFT} words, got {words.shape}")

    nbits = 10
    nman = 12
    nexp = 6

    iq_mask = (1 << (nman - 1)) - 1
    e_mask = (1 << nexp) - 1
    e_p = 1 << (nexp - 1)

    real_sign_mask = 1 << (nexp + 2 * nman - 1)
    imag_sign_mask = real_sign_mask >> nman

    mantissas = np.zeros((NFFT, 2), dtype=np.int64)
    signs = np.ones((NFFT, 2), dtype=np.int64)
    exponents = np.zeros(NFFT, dtype=np.int64)

    maxbit = -e_p

    for i, raw_word in enumerate(words):
        word = int(raw_word)

        vi = (word >> (nexp + nman)) & iq_mask
        vq = (word >> nexp) & iq_mask

        exponent = word & e_mask

        if exponent >= e_p:
            exponent -= e_p << 1

        exponents[i] = exponent

        x = vi | vq
        e = exponent

        if x:
            m = 0xFFFF0000
            b = 0xFFFF
            s = 16

            while s > 0:
                if x & m:
                    e += s
                    x >>= s

                s >>= 1
                m = (m >> s) & b
                b >>= s

            if e > maxbit:
                maxbit = e

        if word & real_sign_mask:
            signs[i, 0] = -1

        if word & imag_sign_mask:
            signs[i, 1] = -1

        mantissas[i, 0] = vi
        mantissas[i, 1] = vq

    shift = nbits - maxbit
    output = np.zeros((NFFT, 2), dtype=np.int64)

    for i in range(NFFT):
        exponent = int(exponents[i]) + shift

        for component in range(2):
            value = int(mantissas[i, component])

            if exponent < -nman:
                value = 0
            elif exponent < 0:
                value >>= -exponent
            else:
                value <<= exponent

            output[i, component] = value * signs[i, component]

    limits = np.iinfo(np.int32)

    if np.any(output < limits.min) or np.any(output > limits.max):
        raise OverflowError("Decoded CSI exceeds int32 range")

    output = output.astype(np.int32)

    return (
        output[:, 0].astype(np.float64)
        + 1j * output[:, 1].astype(np.float64)
    )



def decode_packet(packet):
    """Extract and decode one CSI packet."""
    if len(packet) != EXPECTED_PACKET_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_PACKET_SIZE} bytes, got {len(packet)}"
        )

    words = np.frombuffer(
        packet,
        dtype="<u4",
        count=NFFT,
        offset=CSI_OFFSET,
    )

    return decode_4366c0(words)


def main():
    parser = argparse.ArgumentParser(
        description="Decode Broadcom 4366c0 CSI from SiMWiSense PCAP"
    )
    parser.add_argument("--pcap", default=PCAP_PATH)
    parser.add_argument("--packets", type=int, default=100)
    args = parser.parse_args()

    path = Path(args.pcap)
    if not path.is_file():
        raise FileNotFoundError(path)

    decoded = []
    timestamps = []
    invalid = 0

    for timestamp, caplen, origlen, packet in read_pcap(path):
        if len(decoded) + invalid >= args.packets:
            break

        if origlen != EXPECTED_PACKET_SIZE or caplen != EXPECTED_PACKET_SIZE:
            invalid += 1
            continue

        try:
            csi = decode_packet(packet)
        except (ValueError, OverflowError) as exc:
            print(f"Skipping packet: {exc}")
            invalid += 1
            continue

        decoded.append(csi)
        timestamps.append(timestamp)

    if not decoded:
        raise RuntimeError("No valid CSI packets decoded")

    csi = np.stack(decoded)
    selected = csi[:, SUBCARRIER_INDICES]

    print("\n--- CSI decoding report ---")
    print(f"PCAP: {path}")
    print(f"Decoded packets: {len(decoded)}")
    print(f"Skipped packets: {invalid}")
    print(f"Raw CSI shape: {csi.shape}")
    print(f"Selected CSI shape: {selected.shape}")
    print(f"Complex dtype: {selected.dtype}")
    print(f"Finite values: {np.isfinite(selected).all()}")
    print(f"All-zero packets: {np.all(selected == 0, axis=1).sum()}")
    print(f"First timestamp: {timestamps[0]:.6f}")
    print(f"Last timestamp: {timestamps[-1]:.6f}")
    print(f"First CSI sample: {selected[0, :5]}")
    print(
        "Amplitude range:",
        float(np.abs(selected).min()),
        "to",
        float(np.abs(selected).max()),
    )

    # Save only this small test sample.
    np.save("csi_test_100.npy", selected)
    print("\nSaved: csi_test_100.npy")


if __name__ == "__main__":
    main()
