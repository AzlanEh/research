# Wi-Fi CSI Extension / Wi-Chat Project — Context & Handoff

> **Purpose:** A consolidated handoff of the project context available from prior conversations, uploaded notes, research papers, and the latest repository tree shared in chat.
>
> **Last context snapshot:** 10 November 2026  
> **Primary repository:** `~/workspace/wifi-extend`  
> **Working host shown in terminal output:** `iotlabnew@iotlabnew-Super-Server`
>
> This document distinguishes **observed work**, **reported measurements**, **hypotheses**, and **future plans**. It is not a claim that every experiment has been independently rerun or every result has been validated.

---

## 1. Executive summary

The project is an experimental Wi-Fi Channel State Information (CSI) analysis pipeline, with two research directions:

1. **CSI signal investigation and preprocessing**
   - Decode/inspect CSI data.
   - Compare a Python CSI decoder against a C reference implementation.
   - Visualize CSI amplitude, complex values, phase/coherence and batch-level patterns.
   - Investigate suspicious changes in amplitude and complex CSI across samples.
   - Determine whether observed amplitude changes could be receiver gain/AGC behaviour, channel changes, phase effects, or another artifact.

2. **Wi-Chat-style activity recognition**
   - Use CSI amplitude time series to infer human activity.
   - Incorporate physical Wi-Fi propagation knowledge into prompts to an LLM.
   - Explore classes discussed in the Wi-Chat paper, including walking, falling, breathing and no-event.
   - The LLM engine is a planned next stage; it is **not currently integrated into the `wifi-extend` repository tree**.

The current work is primarily in the **signal inspection / anomaly investigation stage**. The Python-vs-C decoder comparison has been attempted but is not confirmed to pass. The suspected gain/AGC behaviour has not been proven. No validated correction/normalization pipeline or end-to-end activity classifier is established yet.

**Immediate engineering priority:** establish that the CSI values entering the pipeline are decoded correctly, then reconcile the CSI shape/gain experiments using explicit sample pairs and consistent metrics. Avoid correcting or normalizing the suspected anomaly until its cause and impact on activity information are understood.

---

## 2. Research references and how they relate to this project

### 2.1 Wi-Chat paper (`LLM_based.pdf`)

The uploaded paper describes **Wi-Chat**, an LLM-powered Wi-Fi human-activity-recognition approach.

Key ideas extracted from the paper:

- Feed textual or visual representations of raw Wi-Fi signals to LLMs.
- Use physical models of Wi-Fi sensing as prompt/domain guidance.
- Explore zero-shot recognition and in-context learning (ICL).
- The paper describes a base setting using raw CSI amplitude as a time series.
- The physical-model explanations include:
  - **Walking:** movement changes propagation paths and CSI amplitude over time.
  - **Breathing:** small, smooth and gradual changes, often over a longer interval.
  - **No event:** mostly static propagation components and relatively stable amplitude.
- The paper reports a best accuracy of 90% on unseen data in its experiments. This is a paper-reported result, **not a result achieved by this project**.
- The paper discusses prompting strategies including base prompting, in-context learning and physical-model/domain guidance.

Project implementation direction:

1. Build a reliable CSI-to-time-series preprocessing path.
2. Convert an observation window into a compact, interpretable signal representation.
3. Build an LLM inference wrapper with an OpenAI-compatible API.
4. Add physical-model-guided prompts and structured output.
5. Evaluate against labelled examples and compare predictions with ground truth.

The paper does not establish that an arbitrary CSI capture device, dataset or decoder will reproduce its results. Dataset, hardware, preprocessing and prompt differences must be documented.

### 2.2 SiMWiSense paper (`SiMWiSense_Simultaneous_Multi-Subject_Activity_Classification_Through_Wi-Fi_Signals.pdf`)

The uploaded paper describes **SiMWiSense**, a framework for simultaneous multi-subject activity classification through Wi-Fi.

Key points visible in the paper material:

- It targets simultaneous activity classification for multiple people.
- It uses Channel State Information (CSI).
- It addresses generalization across subjects and environments.
- The paper discusses using CSI from the transceiver positioned closest to a subject.
- It is a research reference for the multi-subject direction, not an implemented component of the current repository.

Potential relevance:

- It can inform a later multi-person extension.
- It should not be treated as interchangeable with Wi-Chat: Wi-Chat is the LLM/prompting activity-recognition direction, while SiMWiSense is a multi-subject CSI classification framework.
- The current `wifi-extend` tree does not show a completed SiMWiSense reproduction or multi-subject classifier.

### 2.3 Relationship between the papers and the implementation

| Area                     | Wi-Chat                                                      | SiMWiSense                                         | Current project state                      |
| ------------------------ | ------------------------------------------------------------ | -------------------------------------------------- | ------------------------------------------ |
| Main focus               | LLM-based Wi-Fi activity recognition                         | Simultaneous multi-subject activity classification | Research/investigation                     |
| Input                    | CSI signal representations, especially amplitude time series | CSI                                                | CSI-related data and batch outputs present |
| Main method              | LLM prompting + physical-model guidance + ICL                | CSI-based multi-subject classification framework   | Not reproduced end-to-end                  |
| Multi-person focus       | Not the central focus described here                         | Central focus                                      | Future research direction                  |
| LLM integration          | Central                                                      | Not the central method                             | Planned; not in repo tree                  |
| Validated project result | None established yet                                         | None established yet                               | Not yet                                    |

---

## 3. Project goals

### Near-term goals

- Make CSI decoding reproducible and trustworthy.
- Understand the data dimensions, valid subcarriers, sample ordering and metadata.
- Visualize representative batches.
- Investigate amplitude and phase behaviour without prematurely removing potentially meaningful activity signal.
- Decide whether a gain correction is justified based on evidence.

### Medium-term goals

- Build a stable preprocessing pipeline:
  - load CSI,
  - validate shape and numeric values,
  - identify invalid/weak subcarriers,
  - derive amplitude and phase,
  - form time windows,
  - produce a consistent model input representation.
- Implement a Wi-Chat-style LLM engine using physical-model-guided prompts.
- Evaluate predictions against labelled data.

### Longer-term goals

- Investigate generalization across batches, subjects and environments.
- Explore multi-subject recognition informed by SiMWiSense.
- Separate device/receiver artifacts from actual human-motion signal.
- Produce a reproducible experimental report with clear metrics and limitations.

---

## 4. Environment, hardware and data context

### 4.1 Current repository host

The terminal prompt in the recent debugging conversation showed:

```text
iotlabnew@iotlabnew-Super-Server
```

The project path was:

```text
/home/iotlabnew/workspace/wifi-extend
```

A virtual environment was active in the failed decoder run:

```text
/home/iotlabnew/workspace/wifi-extend/.venv
```

Do not assume that this host is the same machine as the user's personal Omarchy laptop.

### 4.2 Separate laptop Wi-Fi hardware context

A separate prior hardware investigation on the user's Omarchy laptop identified:

```text
MediaTek MT7921 802.11ax PCIe Wireless Network Adapter
PCI ID: 14c3:7961
Driver: mt7921e
```

This was discussed in the context of CSI extraction and the `extra-csi` / MediaTek ecosystem. It is **not confirmed** that the current `wifi-extend` dataset was captured with this adapter, or that the super-server uses the same adapter.

### 4.3 CSI dataset path mentioned in investigation scripts

The AGC investigation scripts used this dataset directory:

```text
/mnt/external_disk/SiMWiSense/SiMWiSense/Data/fine_grained/Classroom/80MHz/3mo/m1/Slots/Test/A_batch
```

The scripts loaded files named:

```text
batch_0.mat
batch_1003.mat
batch_1007.mat
```

and extracted:

```python
loadmat(path)["csi_mon"]
```

The scripts expected a two-dimensional CSI array, normally interpreted as:

```text
samples × subcarriers
```

They also contained a transpose check for 242 subcarriers:

```python
if csi.shape[1] != 242 and csi.shape[0] == 242:
    csi = csi.T
```

This indicates that the investigation scripts were designed around 242 subcarriers. Confirm the actual dimensions from the source MAT files before assuming this is universally true.

---

## 5. Latest known repository structure

The latest tree shared in chat was:

```text
wifi-extend/
├── README.md
├── requirements.txt
└── src/
    ├── agc_vs_channel_analysis.png
    ├── compare_decoder.py
    ├── cross_batch_gain_analysis.py
    ├── csi_decoder.py
    ├── csi_heatmaps/
    │   ├── amplitude_batch_0.csv
    │   ├── amplitude_batch_1003.csv
    │   ├── amplitude_batch_1007.csv
    │   ├── coherence_batch_0.csv
    │   ├── coherence_batch_1003.csv
    │   ├── coherence_batch_1007.csv
    │   ├── complex_batch_0.csv
    │   ├── complex_batch_1003.csv
    │   ├── complex_batch_1007.csv
    │   ├── csi_heatmap_batch_0.png
    │   ├── csi_heatmap_batch_1003.png
    │   └── csi_heatmap_batch_1007.png
    ├── csi_reference
    ├── csi_reference.c
    ├── csi_test_100.npy
    ├── gain_timeline_analysis.png
    ├── inspect_csi_rows.py
    ├── investigate_agc_vs_channel.py
    ├── investigate_alternating_pattern.py
    ├── investigate_complex_scaling.py
    ├── investigate_csi_anomaly.py
    ├── investigate_csi_clusters.py
    ├── investigate_csi_heatmaps.py
    ├── investigate_csi_normalization.py
    ├── investigate_direct_phase_pairs.py
    ├── investigate_full_pair_similarity.py
    ├── investigate_gain_consistency.py
    ├── investigate_gain_pattern.py
    ├── investigate_gain_scaling.py
    ├── investigate_gain_timeline.py
    ├── investigate_pairwise_reconstruction.py
    ├── investigate_phase_amplitude.py
    ├── investigate_phase_structure.py
    ├── investigate_reference_stability.py
    ├── investigate_scalar_gain_fit.py
    ├── investigate_shape_vs_gain.py
    ├── investigate_subcarrier_changes.py
    ├── investigate_subcarrier_gain.py
    ├── investigate_suspect_batches.py
    ├── plot_csi_batch.py
    ├── plots/
    │   ├── csi_anomaly_investigation.png
    │   ├── gain_pattern.png
    │   ├── normalization_comparison.png
    │   ├── phase_difference.png
    │   └── Test_batch_0.png
    └── verify_stable_correlations.py
```

`__pycache__/` was also present in the actual tree but is omitted above because it is generated runtime cache, not a source module.

### 5.1 What the file groups indicate

| Group                        | Files                                                        | Purpose inferred from names and prior work                   |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Decoder                      | `csi_decoder.py`, `csi_reference.c`, `csi_reference`, `compare_decoder.py` | Python decoder, C reference executable/source, and comparison |
| CSI inspection               | `inspect_csi_rows.py`, `plot_csi_batch.py`                   | Inspect rows and plot batch CSI                              |
| Heatmaps/exports             | `investigate_csi_heatmaps.py`, `csi_heatmaps/*.csv`, `*.png` | Export amplitude, coherence, complex CSI and heatmaps        |
| Anomaly                      | `investigate_csi_anomaly.py`, `investigate_csi_clusters.py`, `investigate_suspect_batches.py` | Find unusual samples/patterns                                |
| Gain/AGC                     | `investigate_agc_vs_channel.py`, `investigate_gain_*.py`, `cross_batch_gain_analysis.py` | Study gain patterns, timeline and cross-batch consistency    |
| Phase/complex CSI            | `investigate_phase_*.py`, `investigate_complex_scaling.py`, `investigate_direct_phase_pairs.py`, `investigate_full_pair_similarity.py` | Understand phase and complex CSI differences                 |
| Reconstruction/normalization | `investigate_pairwise_reconstruction.py`, `investigate_shape_vs_gain.py`, `investigate_csi_normalization.py`, `investigate_scalar_gain_fit.py` | Test whether sample changes can be explained by scalar scaling |
| Output plots                 | `src/*.png`, `src/plots/*.png`                               | Visual evidence from investigations                          |

The exact implementation and current correctness of each script must be checked in the repository. File names alone do not establish what a script calculates.

---

## 6. Work completed so far

### 6.1 CSI decoder work

Files present:

- `src/csi_decoder.py`
- `src/csi_reference.c`
- `src/csi_reference` (compiled executable)
- `src/compare_decoder.py`
- `src/csi_test_100.npy`

What has been done:

- A Python CSI decoder exists.
- A C reference implementation/source and compiled executable exist.
- A comparison script was created to compare decoder outputs.
- The comparison was run in the project virtual environment.

Known issue:
A previous run of:

```bash
python compare_decoder.py
```

printed PCAP metadata and then failed with a Python generator-context error. The captured output began:

```text
PCAP version: 2.4
Snaplen: 65535, link type: 1
Traceback (most recent call last):
  File ".../src/compare_decoder.py", line 29, in <module>
```

The complete traceback is not available in this context. The error was not confirmed fixed.

**Status:** Decoder comparison exists, but Python/C equivalence is **not verified**. Do not treat the decoded CSI as validated until this test passes on identical input and the comparison covers the same packets/frames and fields.

### 6.2 CSI visualization and exports

Existing outputs indicate that CSI data from batches `0`, `1003`, and `1007` has been inspected/exported.

Per batch, the tree contains:

- amplitude CSV,
- coherence CSV,
- complex CSI CSV,
- CSI heatmap PNG.

Other plots include:

- `agc_vs_channel_analysis.png`
- `gain_timeline_analysis.png`
- `plots/csi_anomaly_investigation.png`
- `plots/gain_pattern.png`
- `plots/normalization_comparison.png`
- `plots/phase_difference.png`
- `plots/Test_batch_0.png`

**Status:** Visualization and data-export work has been performed. The presence of output files does not by itself validate the decoder or prove a physical interpretation.

### 6.3 Signal anomaly and sample-pattern investigation

A number of scripts were created to examine:

- CSI amplitude anomalies,
- clusters and suspicious batches,
- alternating patterns,
- subcarrier-wise changes,
- phase structure and phase differences,
- complex scaling,
- gain consistency and timeline,
- relationships between gain and channel variation,
- pairwise reconstruction and reference stability.

This is substantial exploratory analysis, but it should now be consolidated into a smaller reproducible analysis pipeline rather than expanded indefinitely with one-off scripts.

---

## 7. Gain / possible AGC investigation

### 7.1 Important framing

The project observed amplitude changes that may be consistent with receiver gain-state changes. **Automatic Gain Control (AGC) is a hypothesis, not a confirmed cause.**

Other possible explanations or contributing factors include:

- actual channel changes,
- movement or changing multipath,
- phase variation,
- CSI calibration or scaling behaviour,
- packet/measurement differences,
- preprocessing or decoder errors.

Do not label the observed patterns “AGC switching” as a proven result.

### 7.2 Reported gain patterns

The project note `Next AGC Investigation.txt` described the following results.

#### Batch 0 — strongest candidate for a gain-state transition

Reported pattern:

- Samples `0–29`: gain close to `1.0`.
- Samples `30`, `32`, `34`, `36`: approximately `1.28–1.32`.
- Intervening odd samples `31`, `33`, `35`: approximately `0.95`.
- From sample `37` onward: gain around `1.30`.

The note also reported an earlier scalar-fit residual of approximately `1.42%` for a selected comparison.

Interpretation in the note:

- The pattern is consistent with a possible amplitude/gain-state transition.
- It is the highest-priority batch for further testing.
- It is not proof of AGC.

#### Batch 1003 — irregular amplitude variation

Reported pattern:

- Gain varies approximately `0.85–1.22`.
- Frequent high/low changes.
- No clear sustained transition like batch 0.

Interpretation:

- Irregular amplitude variation.
- Not enough evidence to call it a gain-state switch.

#### Batch 1007 — multiple amplitude states

Reported pattern:

- Samples `0–1`: approximately `0.89`.
- Samples `2–19`: mostly `1.07–1.17`.
- Samples `20–29`: mixed low values around `0.82–0.86` and higher values.
- Samples `30–47`: mostly `0.82–0.92`.
- Samples `48–49`: jump to approximately `1.10–1.12`.

Interpretation:

- Possible multiple amplitude states or a mixture of channel variation and gain changes.
- Second priority after batch 0.

### 7.3 Reported priority

| Priority | Batch  | Reported behaviour                                     | Interpretation                 |
| -------: | ------ | ------------------------------------------------------ | ------------------------------ |
|        1 | `0`    | Alternating high/low region followed by elevated state | Strongest gain-state candidate |
|        2 | `1007` | Several localized amplitude states and reversals       | Worth investigating            |
|        3 | `1003` | Irregular high/low variation                           | No clear sustained transition  |

The note warns that an alternation score based on consecutive ratio signs can be inflated by noisy fluctuations. It cannot independently establish periodic behaviour.

### 7.4 The crucial shape-versus-scale test

The next scientific question is:

> Are the samples showing the same underlying CSI profile at a different overall amplitude, or is the channel response itself changing across subcarriers?

Two comparisons were designed:

1. **Amplitude-shape similarity**
   - Compare `abs(CSI)` profiles after fitting/removing one overall amplitude scale.
2. **Complex CSI similarity**
   - Compare complex CSI after fitting one complex scalar, which can account for a common amplitude scale and common phase rotation.

Interpretation:

- Low amplitude residual + low complex residual: one common complex scaling may explain the change.
- Low amplitude residual + high complex residual: amplitude shape is similar but phase/complex channel response differs.
- High amplitude residual + high complex residual: one common scalar does not explain the change.

### 7.5 Conflicting residual measurements — preserve this caveat

The notes contain two different kinds of reported results.

**Earlier group-reference experiment:**

- One selected batch-0 comparison reported a scalar-fit residual around `1.42%`.
- The reference was formed by averaging multiple complex CSI samples:

```python
reference = np.mean(csi[refs], axis=0)
```

**Later shape-vs-gain experiment:**

- Reported complex residuals:
  - Batch 0: `0.656–0.999`
  - Batch 1003: `0.496–0.890`
  - Batch 1007: `0.526–0.960`
- Reported amplitude residuals:
  - Batch 0: `0.398–0.416`
  - Batch 1003: `0.384–0.432`
  - Batch 1007: `0.369–0.426`

The notes explicitly say these results do not agree with the earlier `1.42%` result because the experiments used different reference construction and aggregation methods.

A possible issue is that averaging complex CSI samples can cause phase cancellation. The next note therefore proposes comparing individual reference samples against individual test samples and taking the best-fitting scalar residual for each test sample.

**Do not combine these numbers into one conclusion.** Before drawing a scientific conclusion:

- inspect the exact scripts,
- verify which samples and subcarriers are used,
- define the residual denominator consistently,
- compare the same sample pairs,
- report best/median/worst pairwise residuals,
- inspect subcarrier-wise residuals,
- confirm that the CSI data is correctly decoded.

### 7.6 Pairwise reconstruction experiment

The pairwise experiment used these reference/test regions:

| Batch  | Reference samples    | Test samples                         |
| ------ | -------------------- | ------------------------------------ |
| `0`    | `20–27`              | `30–37`                              |
| `1003` | `0–4`                | `5–14`                               |
| `1007` | `10, 12, 14, 16, 18` | `20, 21, 22, 25, 29, 30, 40, 48, 49` |

The pairwise method fits one complex scalar:

```python
alpha = np.vdot(a, b) / np.vdot(a, a)
residual = np.linalg.norm(b - alpha * a) / (np.linalg.norm(b) + 1e-12)
```

For amplitude-only fitting:

```python
a = np.abs(a)
b = np.abs(b)
alpha = np.dot(a, b) / (np.dot(a, a) + 1e-12)
residual = np.linalg.norm(b - alpha * a) / (np.linalg.norm(b) + 1e-12)
```

This avoids using a single complex mean as the only reference. The note's purpose is to test whether the earlier mismatch was caused by reference averaging.

**Status:** The investigation script is present in the repository tree, but its latest output is not available here. Do not assume the pairwise test has resolved the discrepancy.

### 7.7 Do not normalize/correct yet

The investigation notes explicitly recommend not applying gain correction yet.

Reason:

- A correction might remove meaningful human-activity information.
- The suspected artifact has not been identified conclusively.
- The measurements are not yet reconciled.
- It is not established that the observed behaviour is independent of the activity labels.

---

## 8. Wi-Chat LLM engine planning

### 8.1 Previous implementation discussion

The project previously moved from explaining the Wi-Chat paper toward implementing it as a practical project. A later decision was to build the LLM engine next.

A code example was shared using an NVIDIA-hosted OpenAI-compatible endpoint:

```python
import json
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

completion = client.chat.completions.create(
    model="deepseek-ai/deepseek-v4.1-flash",
    messages=json.loads(
        '[{"role":"user","content":[{"type":"text","text":"..."}]}]'
    )
)
```

This is evidence of a proposed API/model integration direction. It does **not** establish that an LLM service, inference pipeline or activity classifier has been completed.

### 8.2 Suggested engine architecture

A practical first implementation should be small and modular:

```text
CSI source
   |
   v
CSI validation / preprocessing
   |
   v
Windowing and feature representation
   |
   v
Signal summary + physical-model context
   |
   v
LLM prompt builder
   |
   v
OpenAI-compatible NVIDIA API client
   |
   v
Structured prediction
   |
   v
Validation, logging and evaluation
```

Possible module layout (proposed, not current repo contents):

```text
src/
├── csi/
│   ├── loader.py
│   ├── validation.py
│   ├── preprocessing.py
│   └── windowing.py
├── features/
│   ├── amplitude.py
│   ├── phase.py
│   └── summarize.py
├── llm/
│   ├── client.py
│   ├── prompts.py
│   ├── schemas.py
│   └── classifier.py
├── evaluation/
│   ├── metrics.py
│   └── confusion_matrix.py
└── main.py
```

### 8.3 Suggested first LLM task

Start with a controlled, small task rather than an ambitious multi-person system.

Candidate labels from the Wi-Chat paper discussion:

- `walking`
- `falling`
- `breathing`
- `no_event`
- `unknown` (recommended safety/uncertainty fallback)

Input should include:

- a fixed time window,
- sampling/window metadata where known,
- amplitude time series or a documented compact representation,
- missing/invalid-data information,
- a concise physical-model guide.

Output should be structured, for example:

```json
{
  "activity": "walking",
  "confidence": 0.0,
  "evidence": [
    "short explanation tied to observed signal behaviour"
  ],
  "alternatives": ["unknown"],
  "data_quality": "acceptable"
}
```

The confidence field should not be treated as calibrated probability unless calibration is tested. Prefer a schema that requires the model to state uncertainty and avoid inventing evidence.

### 8.4 Evaluation requirements

Before claiming recognition performance:

- use ground-truth labels,
- split data carefully to avoid leakage between closely related windows,
- keep subject/session/environment separation where possible,
- report per-class precision, recall and F1,
- include a confusion matrix,
- compare with simple baselines,
- track invalid/ambiguous windows,
- record the exact prompt, model identifier and preprocessing configuration.

Do not compare the project's early prototype directly with the paper's reported 90% unless dataset and evaluation protocols are comparable.

---

## 9. CSI capture and hardware acquisition context

Prior conversations discussed MediaTek CSI extraction and an `extra-csi`-style approach.

A note shared earlier described:

```text
extra-csi
Full-metadata CSI extraction for MediaTek mt76 chipsets on OpenWrt.
```

It also noted that `MtkCSIdump` could retrieve CSI but strip metadata such as source MAC and RSSI. This raised a concern that CSI samples without packet metadata may be difficult to associate with devices, transmitters or capture conditions.

At that time:

- hardware availability was reported as **none** for the planned router-side capture setup,
- router model was **unknown**,
- a practical Wi-Chat implementation plan was requested despite that limitation.

Important distinction:

- The existing `wifi-extend` analysis appears to use a SiMWiSense dataset path on an external disk.
- The separate MT7921 laptop and MediaTek/OpenWrt CSI-capture discussion is a hardware acquisition path.
- Do not assume these are the same data source or that live CSI capture is already working.

A robust future capture pipeline should preserve, where available:

- timestamp,
- source/transmitter identity or anonymized identifier,
- receiver/interface identity,
- RSSI,
- channel and bandwidth,
- packet sequence/ordering,
- CSI dimensions and subcarrier mapping,
- device/firmware/driver metadata,
- capture session and environment identifiers.

---

## 10. Key known issues and risks

| Issue                    | Current understanding                                        | Required action                                              |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Python/C decoder parity  | Comparison script previously failed with generator-context error | Fix traceback and validate on identical PCAP data            |
| CSI correctness          | Not established end-to-end                                   | Validate decoder output, dimensions, packet mapping and numeric values |
| AGC hypothesis           | Patterns observed; cause unconfirmed                         | Compare same sample pairs and test alternate explanations    |
| Residual discrepancy     | Earlier ~1.42% vs later high residuals use different methods | Reconcile references, sample ranges and metric definitions   |
| Complex averaging        | May cause phase cancellation                                 | Use pairwise references and report distributions             |
| Normalization            | Could remove real activity signal                            | Do not apply until validated                                 |
| LLM engine               | API direction discussed, not present in repo tree            | Implement only after input representation is reliable        |
| Activity labels          | No completed project-level classifier/evaluation established | Define labels and ground truth                               |
| Hardware capture         | Separate MediaTek path discussed; availability uncertain     | Confirm actual capture hardware and metadata                 |
| Research reproducibility | Many one-off scripts and outputs                             | Consolidate into reproducible experiments/configs            |

---

## 11. Recommended roadmap

### Phase 1 — Validate decoder and CSI source (highest priority)

1. Open and inspect `compare_decoder.py`.
2. Re-run it and capture the full traceback.
3. Fix the generator/context-manager error.
4. Run the Python decoder and C reference on the exact same PCAP/input packets.
5. Compare:
   - packet count and packet identifiers,
   - CSI dimensions,
   - subcarrier ordering,
   - real and imaginary values,
   - amplitude and phase,
   - missing/invalid values.
6. Define numerical tolerances and produce a pass/fail summary.

**Exit criterion:** decoder comparison is reproducible and passes for the selected test input, with mismatches explained.

### Phase 2 — Reconcile gain/shape analysis

1. Inspect the implementations of:
   - `investigate_shape_vs_gain.py`
   - `investigate_pairwise_reconstruction.py`
   - `investigate_scalar_gain_fit.py`
   - `investigate_gain_pattern.py`
2. Use identical valid subcarriers and sample pairs.
3. Compare individual references, not only complex averages.
4. Report:
   - fitted scalar magnitude,
   - fitted scalar phase,
   - complex residual,
   - amplitude residual,
   - amplitude correlation,
   - per-subcarrier residual distribution.
5. Check whether the observed state changes line up with labels, capture boundaries or metadata.
6. Only then decide whether gain normalization or calibration is justified.

**Exit criterion:** a documented, reproducible explanation of what the amplitude and complex CSI changes do and do not support.

### Phase 3 — Consolidate preprocessing

Create a small reusable pipeline rather than adding more independent investigation scripts.

Recommended components:

- input loader,
- data/shape validator,
- subcarrier validity mask,
- amplitude/phase derivation,
- optional filtering (only when justified),
- fixed-length windowing,
- feature/summary generation,
- export and plotting.

Every transform should be configurable and logged.

**Exit criterion:** one command can load a known batch, validate it, produce a windowed representation and save reproducible outputs.

### Phase 4 — Build Wi-Chat-style LLM engine

1. Implement the API client using environment variables for credentials.
2. Implement prompt templates with physical-model guidance.
3. Send a small, labelled set of windows.
4. Require structured JSON output and validate it.
5. Log prompt/model/version/input window and response.
6. Evaluate against ground truth and baselines.

**Exit criterion:** repeatable inference and an evaluation report on a held-out set.

### Phase 5 — Explore multi-subject direction

After the single-stream pipeline is stable:

- study the SiMWiSense methodology in detail,
- identify what receiver/monitor placement and data are required,
- establish subject-level labels and separation,
- test generalization across people and environments,
- do not claim multi-subject classification until a controlled evaluation supports it.

---

## 12. Suggested immediate commands

From the project root:

```bash
cd ~/workspace/wifi-extend
source .venv/bin/activate
```

Inspect the decoder comparison script:

```bash
sed -n '1,240p' src/compare_decoder.py
```

Run it while preserving the full error:

```bash
python -X dev src/compare_decoder.py 2>&1 | tee compare_decoder.log
```

Inspect the relevant analysis scripts:

```bash
sed -n '1,260p' src/investigate_shape_vs_gain.py
sed -n '1,280p' src/investigate_pairwise_reconstruction.py
sed -n '1,240p' src/investigate_gain_pattern.py
```

Check the available files and outputs:

```bash
find src -maxdepth 2 -type f | sort
```

Check installed dependencies:

```bash
cat requirements.txt
python --version
```

These are inspection/debugging commands. They do not assume the scripts are correct or that the decoder test will pass.

---

## 13. Working conventions for future sessions

- Be precise about what is **observed**, what is **inferred**, and what is **hypothesized**.
- Do not call a pattern AGC unless receiver metadata or controlled experiments support that conclusion.
- Do not apply normalization just to make plots look stable.
- Keep exact batch IDs, sample indices, valid-subcarrier masks and metric definitions with every reported result.
- Preserve raw CSI and create derived outputs separately.
- Keep a record of decoder version, capture source, dataset path and preprocessing configuration.
- Avoid adding more one-off scripts until the current investigations are consolidated.
- When reporting model performance, include data split methodology and per-class metrics.
- Treat paper-reported results as references, not as project achievements.
- Prefer short, reproducible steps and exact commands when debugging.

---

## 14. Uploaded/reference materials known in this project context

The following files were attached or referenced in the project conversation:

### Research papers

- `SiMWiSense_Simultaneous_Multi-Subject_Activity_Classification_Through_Wi-Fi_Signals.pdf`
- `LLM_based.pdf` — Wi-Chat paper material

### Investigation notes

- `Next AGC Investigation.txt`
- `Branch · Next AGC Investigation.txt`
- Several branch copies of `Explain .txt`

### Images / plots shared in conversation

- `Test_batch_0.png`
- `csi_anomaly_investigation.png`
- `gain_pattern.png`

### Repository-generated or repository-listed plots

- `agc_vs_channel_analysis.png`
- `gain_timeline_analysis.png`
- `plots/normalization_comparison.png`
- `plots/phase_difference.png`
- batch CSI heatmaps for batches `0`, `1003`, `1007`

The uploaded files and repository outputs may represent different points in the investigation. Confirm their latest versions before treating any one image or note as the final result.

---

## 15. Current project status at a glance

| Workstream                   | Status                                                       |
| ---------------------------- | ------------------------------------------------------------ |
| Research review: Wi-Chat     | Paper reviewed/discussed; implementation planned             |
| Research review: SiMWiSense  | Paper reviewed/discussed; potential multi-subject direction  |
| CSI decoder                  | Python and C implementations exist; parity test unresolved   |
| CSI visualizations           | Outputs exist for batches 0, 1003, 1007                      |
| Anomaly investigation        | Many exploratory scripts and plots exist                     |
| Gain/AGC analysis            | Candidate patterns found; AGC not confirmed                  |
| Complex CSI/phase analysis   | Investigated; residual discrepancy unresolved                |
| Normalization/correction     | Not validated; should not be applied yet                     |
| LLM engine                   | NVIDIA-compatible API/model direction discussed; not integrated in current repo tree |
| Activity classification      | No validated end-to-end result established                   |
| Multi-subject classification | Research direction only; not implemented                     |
| Live CSI capture             | Hardware/capture path not confirmed for this repository      |

### Bottom line

The project has completed a meaningful amount of CSI inspection and exploratory analysis. The next step is **not** to add more model complexity. It is to make the CSI input trustworthy, reconcile the gain/shape measurements, and then build the Wi-Chat inference layer on top of a validated signal pipeline.
