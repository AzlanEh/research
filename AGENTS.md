# AGENTS.md

## Project overview

`wifi-extend` is an experimental Wi-Fi Channel State Information (CSI) analysis
project with two tracks:

1. **CSI decoding / inspection** — decode Broadcom 4366c0 CSI from SiMWiSense
   PCAP/MAT data, compare the Python decoder against the C reference, visualize
   amplitude/phase/coherence, and investigate amplitude (possible gain) changes.
2. **Wi-Chat-style activity recognition** — planned LLM engine using
   physical-model-guided prompts. **Not implemented in this repo yet.**

Current stage: exploratory signal investigation. The decoder is **not yet
validated**, the suspected AGC behaviour is **not proven**, and there is **no
end-to-end classifier**.

Before doing meaningful work, read `.agent/contex.md` — the full research
handoff (papers, hypotheses, reported measurements, roadmap, caveats).

## Environment

The project uses a self-contained conda env at `./.venv` (Python 3.11.18).
It does not touch conda `base` or other projects.

```bash
cd ~/workspace/wifi-extend
conda activate /home/iotlabnew/workspace/wifi-extend/.venv
```

Without conda active:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/iotlabnew/workspace/wifi-extend/.venv
```

Never install into `base`. Use the project env explicitly:

```bash
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install <package>   # then pin it in requirements.txt
```

Pinned deps: `numpy==2.4.6`, `scipy==1.17.1`, `matplotlib==3.11.2`.

## Layout

```text
wifi-extend/
├── .agent/contex.md        # research context & handoff (read-only reference)
├── .venv/                  # project conda env (gitignored)
├── requirements.txt
├── README.md
└── src/
    ├── csi_decoder.py      # PCAP reader + Broadcom 4366c0 decoder (canonical)
    ├── csi_reference.c     # C reference implementation
    ├── csi_reference       # compiled binary (reads 256 words on stdin)
    ├── compare_decoder.py  # Python vs C parity check
    ├── inspect_csi_rows.py # quick row/shape inspection
    ├── plot_csi_batch.py   # batch plotting
    ├── investigate_*.py    # ~20 one-off investigation scripts
    ├── cross_batch_gain_analysis.py
    ├── verify_stable_correlations.py
    ├── csi_test_100.npy    # saved 100-packet decode sample
    ├── csi_heatmaps/       # exported amplitude/coherence/complex CSVs + PNGs
    └── plots/              # investigation plots
```

## Commands

**Run scripts from `src/`.** Most scripts use paths relative to the current
directory (`./csi_reference`, `plots/`, `csi_heatmaps/`, `csi_test_100.npy`),
so running from the repo root writes outputs to the wrong place or fails.

```bash
cd ~/workspace/wifi-extend/src

# Decode 100 packets (default PCAP is hardcoded)
python csi_decoder.py --packets 100
python csi_decoder.py --pcap /path/to/A.pcap --packets 100

# Python vs C decoder parity
python compare_decoder.py

# Inspect a batch / regenerate plots
python inspect_csi_rows.py
python plot_csi_batch.py
python investigate_gain_pattern.py
```

Rebuild the C reference if `csi_reference.c` changes:

```bash
cd src && gcc -O2 -o csi_reference csi_reference.c
```

There is no test suite; verification is script-based and manual.

## Data facts

| Item | Value |
|---|---|
| PCAP | `/mnt/external_disk/SiMWiSense/SiMWiSense/Data/fine_grained/Classroom/80MHz/3mo/m1/CSI_pcap/A.pcap` (1.2 GB) |
| MAT batches | `.../m1/Slots/Test/A_batch/batch_<id>.mat`, array key `csi_mon` |
| Format | Broadcom 4366c0, `CSI_OFFSET = 60`, `NFFT = 256`, packet = 1084 bytes |
| Subcarriers | `np.r_[np.arange(6, 128), np.arange(131, 251)]` → 242 used |
| MAT shape | normally `samples × subcarriers`; scripts transpose if `shape[0] == 242` |
| Batches of interest | `0`, `1003`, `1007` |

Both `/mnt/external_disk/SiMWiSense/Data/...` and
`.../SiMWiSense/SiMWiSense/Data/...` exist; individual scripts hardcode one or
the other. Do not modify anything under `/mnt/external_disk` — treat it as
read-only source data.

## Scientific guardrails (important)

- Distinguish **observed** results, **inferred** explanations, and
  **hypotheses** in comments, docs and responses.
- **Do not call the amplitude behaviour "AGC"** as a proven fact. It is one
  unconfirmed hypothesis among channel change, phase variation, decoder error,
  calibration/scaling, etc.
- **Do not normalize or gain-correct CSI yet.** It may remove real activity
  signal, and the cause is unresolved.
- Preserve raw CSI; write derived outputs to separate files.
- Record exact batch ID, sample indices, subcarrier mask and metric definition
  with every result.
- Do not mix the earlier `1.42%` residual with the later `0.65–0.99` residuals;
  they used different references/aggregation (complex averaging can cause
  phase cancellation). Use pairwise, per-sample fits.
- Paper-reported numbers (e.g. Wi-Chat's 90%) are references, not project
  results.
- Prefer consolidating into a reusable pipeline over adding more one-off
  `investigate_*.py` scripts.

## Known issues / gotchas

- `compare_decoder.py` previously failed with a generator-context error after
  printing PCAP metadata; parity is **unverified**. Capture the full traceback
  before changing decoder math.
- `compare_decoder.py` invokes `./csi_reference`, so it must run from `src/`.
- Decoder defaults and analysis constants are hardcoded (paths, batches,
  subcarrier indices, reference/test regions); keep changes consistent or
  parameterize explicitly.
- Plots use `plt.show()` and relative `savefig` paths; on headless runs prefer
  `MPLBACKEND=Agg` and run from `src/`.
- `csi_decoder.py` saves `csi_test_100.npy` in the CWD, not next to the script.

## Recommended next work (in order)

1. **Validate the decoder** — reproduce and fix the `compare_decoder.py`
   failure; compare packet counts, dimensions, subcarrier order, real/imag
   values and tolerances against the C reference.
2. **Reconcile gain/shape analysis** — same sample pairs, same subcarrier mask,
   individual references, report best/median/worst residuals and per-subcarrier
   distributions.
3. **Consolidate preprocessing** — one loader/validator/windower command with
   logged, configurable transforms.
4. **Build the LLM engine** — only after the input representation is trusted;
   structured JSON output, logged prompts/model/version, held-out evaluation.
5. **Multi-subject exploration** — SiMWiSense-informed, only after the
   single-stream pipeline is stable.

## Code style

- Standalone Python scripts; `argparse` for CLIs; `numpy`/`scipy`/`matplotlib`.
- Keep numeric conventions (offsets, indices, dtype) explicit and documented.
- Flat imports inside `src/` (e.g. `from csi_decoder import ...`); run from
  `src/` rather than converting to a package unless you also fix relative paths.
- Pin any new dependency in `requirements.txt`.
