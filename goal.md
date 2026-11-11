I recommend building this as a small, reproducible research experiment, not immediately attempting all 20 activities.

Our first objective should be to prove one thing:

> Can a general-purpose LLM distinguish three human activities using CSI data from SiMWiSense, without training a dedicated activity-classification neural network?

Below is the implementation plan I would follow.

# 1. Define the proof of concept

POC scope

Project: LLM-based Wi-Fi Activity Recognition

Dataset: SiMWiSense

Input: Wi-Fi CSI measurements

Classification: 3 activities

Recognition engine: General-purpose LLM using prompting

Learning strategy: Zero-shot first, then few-shot prompting

Output: Predicted activity, confidence/uncertainty, and explanation

### Select three activities

I suggest starting with these:

[Side view of handsome muscled young man pushing something heavy](https://images.openai.com/static-rsc-4/kOLBwLrqQJxTCvCbR67Ztq8Ry02PdqvGN5iOT9hdFEPcCouDRVdeK2PfvdhgQ5D5pv6J5vgGwdegobn5JXoE8U11sJvPcGd2I8VyRSrTKwQ2srBDRXg-wC6rIA9V4bQyKi9s1H8JV9P8vnwLEEOoQ3LCW5IX1UGHsDp1ANMMZH6MSOO9_fH2krC6j1nCFxfs?purpose=inline)

1\. Push forward

A forward arm/body movement. Useful for studying directional movement.

[【簡単ストレッチ】2週間でカラダがしなやかに。前屈で腰痛も姿勢も治る！ | 美ST ONLINE](https://images.openai.com/static-rsc-4/gdiHAj4mQqtNhUNHnv1Qro7OocB5ucb5TN5BlS94Y962Ei7EzzcKo65dv8YnnIKmYeAZ0SOBYRq4VDIvDZHmbG_j8U87f5NrptuPDCJRASHj65W1Dxj_F5ZyAlAxcIkbtOlIufm4zGO_YZM9bIL5wa7z7bMCpw0r6fMKLXTn8C0?purpose=inline)

2\. Rotate

Rotational body movement. Useful for studying changing movement direction.

[Handsome man clapping](https://images.openai.com/static-rsc-4/IvE4VYEKUa1DMO5DwtHeEyIxFHtV6o3ym39NGOwZr1Clk2z-U9CWsQd2YbVR-iUVtRMbLJXyQcCT4FGonCdPsL6adyOerI04IpfbqRxZ6m7A6Hh3RveMArmodROht1RNUZ35xkYjj7zC6tYQ4S12J26SaGTQ7Gqodelf-L4AsKWpCEehykh5gWwGhxdBbMJA?purpose=inline)

3\. Clap

Repeated hand movement. Useful for studying short, repeated motion patterns.

These are proposed pilot classes, not confirmed selections from our available files yet. First, we must verify their folder-to-activity mapping.

Why these three? They provide different movement characteristics, making the first experiment more informative than selecting three very similar activities.

# 2. Understand and verify the dataset

Our existing dataset location is:

```
/mnt/external_disk/SiMWiSense/SiMWiSense/
```

The relevant directory structure we've previously worked with is:

```
Data/
└── fine_grained/
    └── Classroom/
        └── 80MHz/
            └── 3mo/
                └── m1/
                    ├── A/
                    ├── B/
                    └── ...
```

Our previously generated batches are organized under:

```
Slots/
├── Train/
│   └── A_batch/
│       ├── batch_0.mat
│       └── ...
└── Test/
    └── A_batch/
        ├── batch_0.mat
        └── ...
```

We have already established that:

- The original CSI is complex-valued.
- The original dataset has a shape of `(1, 123914, 242)` in the previously examined file.
- The generated batch files contain CSI matrices of shape `(50, 242)`.
- The standalone CSI decoder matched the C reference on 100 packets.
- Activity A has previously been mapped to specific source rows, but its actual human-readable activity identity still needs verification.&#x20;

### Important: verify ground truth

The SiMWiSense paper states that synchronized video streams were captured to establish activity ground truth.&#x20;

SiMWiSense_Simultaneous_Multi-Subject_Activity_Classification_Through_Wi-Fi_Signals.pdf



We must establish the relationship between:

```
Activity name
    ↓
Original dataset folder
    ↓
Original CSI source rows
    ↓
Generated batch files
    ↓
Actual activity label
```

We should inspect the dataset README, original repository metadata, and video/annotation files.

Do not assume A means Push forward, B means Rotate, etc.

### Deliverable

Create an activity mapping file:

```
{
  "push_forward": {
    "dataset_folder": "verified_folder",
    "train_batches": [],
    "test_batches": [],
    "monitor": "m1"
  },
  "rotate": {
    "dataset_folder": "verified_folder",
    "train_batches": [],
    "test_batches": [],
    "monitor": "m1"
  },
  "clap": {
    "dataset_folder": "verified_folder",
    "train_batches": [],
    "test_batches": [],
    "monitor": "m1"
  }
}
```

This becomes the source of truth for our experiment.

# 3. Build the CSI preprocessing pipeline

This is where we convert CSI into something useful for an LLM.

The paper describes a 50-sample window and a 0.1-second time window. It also describes the CSI input using amplitude and phase components, with 242 subcarriers in the full-resolution experiments.&#x20;

SiMWiSense_Simultaneous_Multi-Subject_Activity_Classification_Through_Wi-Fi_Signals.pdf



Our generated `(50, 242)` matrices are a useful starting point.

### Step 3.1 — Load CSI

For the existing batch files:

```
from scipy.io import loadmatimport numpy as npdata = loadmat("batch_0.mat")csi = np.asarray(data["csi_mon"])print("Shape:", csi.shape)print("Data type:", csi.dtype)amplitude = np.abs(csi)phase = np.angle(csi)
```

We retain the original complex CSI.

### Step 3.2 — Extract signal information

For every 50 × 242 window, calculate:

| Feature                       | What it tells us                               |
| ----------------------------- | ---------------------------------------------- |
| Mean amplitude                | Average signal strength                        |
| Amplitude standard deviation  | Signal fluctuation                             |
| Minimum and maximum amplitude | Signal range                                   |
| Temporal variation            | How quickly the signal changes                 |
| Amplitude difference          | Changes between consecutive samples            |
| Phase variation               | Changes in phase                               |
| Subcarrier variation          | How signal behavior differs across frequencies |
| Temporal energy               | Overall movement-related signal variation      |

We should also calculate features across smaller temporal sections, rather than only summarizing the entire window.

For example:

```
Window: 50 samples

Segment 1: Samples  0–9
Segment 2: Samples 10–19
Segment 3: Samples 20–29
Segment 4: Samples 30–39
Segment 5: Samples 40–49
```

This allows the LLM to receive information about how the signal evolves over time.

### Step 3.3 — Preserve the anomaly investigation

This is especially important in our project.

We have already observed unusual gain behavior in some CSI batches. We should:

- Preserve the original CSI.
- Record gain-related statistics.
- Avoid blindly applying normalization.
- Test any normalization as a separate experimental variant.
- Compare raw and processed signal representations.

We must not accidentally remove the very signal patterns that distinguish one activity from another.

### Deliverable

A reusable preprocessing script:

```
src/
└── preprocessing/
    ├── load_csi.py
    ├── extract_amplitude.py
    ├── extract_phase.py
    ├── extract_features.py
    └── segment_csi.py
```

# 4. Design the signal-to-LLM representation

This is probably the most important technical component of the entire proof of concept.

An ordinary LLM is not automatically capable of interpreting a 50 × 242 complex CSI matrix.

We need to translate the signal into a structured representation without losing useful temporal information.

I recommend testing two representations.

### Approach A — CSI amplitude time-series prompting

This follows the central idea of Wi-Chat, which provides CSI amplitude time-series data to an LLM and uses domain knowledge to guide its activity recognition.&#x20;

LLM_based.pdf



Example:

```
{
  "window_id": "sample_001",
  "sampling_window": "0.1 seconds",
  "subcarriers": [10, 20, 30, 40, 50],
  "amplitude_sequence": [
    [0.82, 0.84, 0.91, 0.96],
    [0.73, 0.76, 0.88, 0.92],
    [0.81, 0.83, 0.86, 0.95]
  ]
}
```

This is a simplified illustration, not actual dataset values.

We can start with 20 representative subcarriers and all 50 time samples, then compare against a higher-resolution representation.

### Approach B — Feature-based prompting

Instead of sending the entire matrix, send extracted features:

```
{
  "window_id": "sample_001",
  "amplitude_statistics": {
    "mean": 0.84,
    "std": 0.12,
    "min": 0.42,
    "max": 1.31
  },
  "temporal_pattern": {
    "early_variation": 0.08,
    "middle_variation": 0.19,
    "late_variation": 0.11
  },
  "phase_statistics": {
    "mean": 0.0,
    "std": 0.31
  }
}
```

Again, these values are illustrative.

My recommendation: Implement both approaches. We need to establish whether the LLM performs better with time-series information or engineered signal features.

# 5. Build the LLM classification engine

For the first experiment, use a normal general-purpose instruction-following LLM through an API.

We have previously explored NVIDIA's OpenAI-compatible API, so we can keep the model endpoint configurable rather than tying the project to one provider.

Proposed architecture:

```
CSI Preprocessing
       |
       v
Signal Representation
       |
       v
Prompt Builder
       |
       v
General-purpose LLM API
       |
       v
Response Parser
       |
       v
Predicted Activity
```

### Step 5.1 — Define activity knowledge

The prompt should explain the three activity definitions and their expected movement characteristics.

### Step 5.2 — Build the zero-shot prompt

First, provide the activity descriptions without any labeled CSI examples.

Initial classification prompt

ZERO-SHOT EXPERIMENT

```
You are a Wi-Fi sensing activity recognition system.

You will receive CSI measurements collected from a
Wi-Fi sensing device.

Your task is to classify the human activity using
only the provided CSI information.

Possible activities:

1. Push forward
2. Rotate
3. Clap

Activity descriptions:

Push forward:
Forward-directed arm/body movement.

Rotate:
Rotational movement of the body.

Clap:
Repeated movement of the hands towards each other.

Analyze the temporal variation in CSI amplitude
and other provided signal features.

Do not assume that every signal change represents
human movement.

If the signal does not provide enough evidence,
return UNKNOWN.

Return JSON:

{
  "predicted_activity": "...",
  "confidence": 0.0,
  "reason": "..."
}
```

This is our starting prompt, not a proven classifier. The LLM may struggle to distinguish these movements from numerical features alone.

### Step 5.3 — Implement few-shot prompting

After zero-shot testing, give the LLM examples of known CSI patterns and their corresponding activities.

For example:

```
Known example 1:
CSI representation: ...
Correct activity: Push forward

Known example 2:
CSI representation: ...
Correct activity: Rotate

Known example 3:
CSI representation: ...
Correct activity: Clap

Now classify this unknown CSI window:

CSI representation: ...
```

Use examples exclusively from the training split.

The key experiment is whether providing a few labeled examples improves classification on previously unseen CSI windows.

# 6. Prepare the training and testing strategy

This is where we need to be particularly careful.

The LLM itself does not need conventional model training for the initial experiment, but it does need properly selected examples for few-shot prompting and independently labeled test data.

### Proposed dataset allocation

For an initial pilot, aim for:

| Dataset portion                | Per activity | Three activities |
| ------------------------------ | ------------ | ---------------- |
| Few-shot examples              | 5            | 15               |
| Additional development samples | 15           | 45               |
| Final test samples             | 50           | 150              |
| Total                          | 70           | 210              |

These are target counts, subject to the availability of verified, independent labeled windows.

### Data splitting rules

1. Training examples must not appear in testing.
2. Use different recordings or non-overlapping source intervals wherever possible.
3. Keep adjacent windows from the same continuous recording in the same split.
4. Check for duplicated CSI batches and overlapping source rows.
5. Keep the monitor, environment, and subject information in the experiment metadata.

We have previously identified exact source-row correspondences between some generated batches. We should use that mapping to detect accidental train/test leakage.

### Why this matters

Suppose we send the LLM a CSI window during few-shot prompting and then evaluate it on the same window.

It may appear to achieve 100% accuracy, but that would not demonstrate recognition of unseen activity.

We need genuine held-out examples.

# 7. Design the experiments

I recommend four experiments.

| Experiment | Input to LLM                                    | Purpose                                |
| ---------- | ----------------------------------------------- | -------------------------------------- |
| E1         | Activity descriptions + CSI features            | Zero-shot baseline                     |
| E2         | Activity descriptions + CSI amplitude sequences | Time-series baseline                   |
| E3         | CSI + 5 labeled examples per activity           | Few-shot learning                      |
| E4         | CSI + few-shot examples + Wi-Fi domain guidance | Test the benefit of physical knowledge |

This gives us a clear research comparison.

### What we should measure

- Overall accuracy
- Per-class precision
- Per-class recall
- Macro F1-score
- Confusion matrix
- Unknown/abstention rate
- API latency and token usage

For instance, if the model recognizes 45 of 50 Push forward windows, 35 of 50 Rotate windows, and 40 of 50 Clap windows:

\\[ Accuracy = \frac{45+35+40}{150}\times100 \\]

\\[ Accuracy = 80\\% \\]

We should also compare the LLM's performance against a simple majority-class baseline. That tells us whether the LLM is actually learning anything useful from the CSI representation.

Important: The LLM's self-reported confidence is not necessarily calibrated probability. We should not treat it as proof of correctness.

# 8. Suggested project structure

I would keep the initial implementation in Python.

```
wifi-llm-activity-recognition/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│       └── activity_mapping.json
│
├── src/
│   ├── dataset/
│   │   ├── inspect_dataset.py
│   │   ├── verify_labels.py
│   │   └── split_dataset.py
│   │
│   ├── preprocessing/
│   │   ├── load_csi.py
│   │   ├── extract_amplitude.py
│   │   ├── extract_phase.py
│   │   ├── extract_features.py
│   │   └── segment_csi.py
│   │
│   ├── llm/
│   │   ├── client.py
│   │   ├── prompt_builder.py
│   │   ├── zero_shot.py
│   │   ├── few_shot.py
│   │   └── response_parser.py
│   │
│   └── evaluation/
│       ├── evaluate.py
│       ├── metrics.py
│       └── confusion_matrix.py
│
├── experiments/
│   ├── experiment_01_zero_shot.py
│   ├── experiment_02_timeseries.py
│   ├── experiment_03_few_shot.py
│   └── experiment_04_domain_guidance.py
│
├── results/
│   ├── predictions.csv
│   ├── metrics.json
│   └── confusion_matrix.png
│
├── requirements.txt
└── README.md
```

# 9. Execution roadmap

Here is the order I recommend following.

1. Phase 1 — Dataset verification

   First priority
   - Identify the actual activity names corresponding to the dataset folders.
   - Verify training and testing files.
   - Confirm the ground-truth labels.
   - Check source-row overlap and duplicates.
   Deliverable: verified dataset mapping.
2. Phase 2 — CSI preprocessing
   - Load the three activities.
   - Generate CSI windows.
   - Extract amplitude, phase, and temporal features.
   - Preserve original CSI.
   - Generate visualizations.
   Deliverable: reproducible CSI feature dataset.
3. Phase 3 — LLM integration
   - Connect the general-purpose LLM.
   - Implement the prompt builder.
   - Implement zero-shot classification.
   - Validate structured JSON responses.
   Deliverable: first end-to-end prediction.
4. Phase 4 — Few-shot experiments
   - Select five labeled examples per class.
   - Build the in-context prompt.
   - Test unseen CSI windows.
   - Compare with zero-shot performance.
   Deliverable: zero-shot versus few-shot results.
5. Phase 5 — Evaluation and research report
   - Calculate accuracy and macro F1.
   - Generate confusion matrix.
   - Compare signal representations.
   - Analyze failure cases.
   - Document limitations.
   Deliverable: POC results and technical report.

# 10. What success looks like

At the end of this POC, we should be able to demonstrate:

Input

A previously unseen CSI window from the SiMWiSense dataset.

Processing

CSI preprocessing → signal representation → LLM prompt.

Output

```
{
  "predicted_activity": "Rotate",
  "confidence": 0.82,
  "reason": "The observed temporal signal variations
  are more consistent with the provided rotation
  examples."
}
```

Illustrative output only, not an actual prediction.

We should consider the POC successful if it produces reproducible predictions on unseen, correctly labeled CSI windows and demonstrates performance meaningfully above the majority-class baseline. We should not promise a particular accuracy before running the experiments.

# 11. My recommended implementation priorities

There is one decision I feel strongly about: do not start by connecting the LLM API. Start by verifying the dataset labels.

Otherwise, even a technically correct prediction could be evaluated against the wrong activity.

Our immediate milestones should be:

| Priority | Task                                   | Status  |
| -------- | -------------------------------------- | ------- |
| 1        | Verify activity-folder mapping         | Pending |
| 2        | Select three verified activities       | Pending |
| 3        | Prepare independent train/test windows | Pending |
| 4        | Build CSI feature extraction           | Pending |
| 5        | Implement zero-shot LLM                | Pending |
| 6        | Implement few-shot LLM                 | Pending |
| 7        | Evaluate and compare                   | Pending |

My proposed first implementation task: Build `inspect_dataset.py` and `verify_labels.py`. They should inspect the existing dataset, identify activity folders, inspect the MATLAB files, and produce a verified mapping of the first three activities to their CSI batches.

Once that mapping is correct, we can build the rest without having to redo the experiment.

I can take the next step and turn Phase 1 into actual runnable Python scripts for your existing SiMWiSense directory, including checks for label mapping, source-row overlap, and train/test leakage.
