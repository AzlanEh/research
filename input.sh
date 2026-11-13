python - <<'PY'
from pathlib import Path
from scipy.io import loadmat
import numpy as np

ROOT = Path("/mnt/external_disk/gaurav_project/wifi_csi/mat")

for sample_id in ["act_15_13", "act_39_30", "act_71_20"]:
    print(f"\n{'='*60}\n{sample_id}")
    mat = loadmat(ROOT / f"{sample_id}.mat")
    trace = mat["trace"]

    for i in range(min(5, len(trace))):
        packet = trace[i, 0]

        print(f"\nPacket {i}")
        for field in ["bfee_count", "timestamp", "Nrx", "Ntx", "perm"]:
            if field in packet.dtype.names:
                value = packet[field]
                print(f"  {field}: {np.asarray(value).ravel().tolist()}")

        csi = packet["csi"][0, 0]
        print("  CSI shape:", csi.shape)
PY
