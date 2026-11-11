
import subprocess
import numpy as np

from csi_decoder import (
    read_pcap,
    decode_4366c0,
    CSI_OFFSET,
    NFFT,
)

PCAP = (
    "/mnt/external_disk/SiMWiSense/Data/"
    "fine_grained/Classroom/80MHz/3mo/"
    "m1/CSI_pcap/A.pcap"
)

NUM_PACKETS = 100

total_values = 0
different_values = 0
max_difference = 0.0
packets_checked = 0

for packet_number, packet in enumerate(read_pcap(PCAP)):
    if packet_number >= NUM_PACKETS:
        break

    timestamp, captured_length, original_length, packet_bytes = packet

    words = np.frombuffer(
        packet_bytes,
        dtype="<u4",
        count=NFFT,
        offset=CSI_OFFSET,
    )

    python_csi = decode_4366c0(words)

    input_data = "\n".join(map(str, words)) + "\n"

    result = subprocess.run(
        ["./csi_reference"],
        input=input_data,
        text=True,
        capture_output=True,
        check=True,
    )

    c_values = np.fromstring(
        result.stdout,
        sep=" ",
        dtype=np.int32,
    )

    if c_values.size != 512:
        raise ValueError(
            f"Packet {packet_number}: expected 512 C values, "
            f"received {c_values.size}"
        )

    c_csi = (
        c_values[0::2].astype(np.float64)
        + 1j * c_values[1::2].astype(np.float64)
    )

    differences = np.abs(python_csi - c_csi)

    total_values += len(python_csi)
    different_values += np.count_nonzero(differences)
    max_difference = max(
        max_difference,
        float(np.max(differences)),
    )

    packets_checked += 1

print("\n--- Python vs C comparison ---")
print(f"Packets checked: {packets_checked}")
print(f"CSI values checked: {total_values}")
print(f"Different values: {different_values}")
print(f"Maximum absolute difference: {max_difference}")

if different_values == 0:
    print("\nSUCCESS: All tested packets match exactly.")
else:
    print("\nFAILURE: Decoder mismatch detected.")
