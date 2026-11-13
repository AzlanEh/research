# WiMANS research and implementation plan



Active research track

Planning phase



Our goal should be to build a reproducible multi-user Wi-Fi activity-recognition system using WiMANS, establish reliable machine-learning baselines, and then investigate whether an LLM-based approach can improve or complement them.

I recommend a staged approach. We should not start building a neural network or LLM classifier immediately. First, we need to make the dataset and evaluation protocol scientifically reliable.

## 1. Research objective

Primary research question

Can Wi-Fi CSI amplitude signals identify the activities of multiple people performing different activities simultaneously, and how does recognition performance change with the number of people, environment, and Wi-Fi band?

We will investigate:

- Multi-user activity recognition.
- Activity recognition with 1–5 simultaneous users.
- Generalization to unseen user groups.
- Differences between 2.4 GHz and 5 GHz CSI.
- Classical deep-learning models versus LLM-based recognition.

The WiMANS paper identifies occlusion and mutual interference as major challenges in multi-user sensing. It also reports that activity recognition is harder than identity recognition and localization.&#x20;

2402.09430v2.pdf

2402.09430v2.pdf



## 2. Understand the dataset

WiMANS dataset overview

# 11,286

CSI samples

# 9

Activity classes

# 3

Environments

# 0–5

Users per sample

CSI tensor dimensions

`3000 × 3 × 3 × 30`

Wi-Fi bands: 2.4 GHz and 5 GHz

Environments: Classroom, Meeting Room, Empty Room

Activities: Nothing, Walking, Rotation, Jumping, Waving, Lying Down, Picking Up, Sitting Down, Standing Up.

2402.09430v2.pdf

The dataset includes synchronized videos and annotations for identities, locations, and activities. We will initially use CSI amplitude only; video is not required for the first model.

## 3. Proposed research architecture

WiMANS dataset

CSI amplitude + annotations + metadata

Dataset validation and preprocessing

Tensor validation · padding · masks · feature extraction

ML/DL branch

ST-RF · CNN · LSTM · CLSTM · THAT

LLM branch

Window statistics · structured CSI representation · LLM inference

Multi-user evaluation

Per-user activity metrics · exact-set accuracy · group generalization

Research findings and ablation studies

The ML/DL and LLM branches will remain separate experiments. This lets us compare their results without assuming that an LLM is automatically better.

## 4. Implementation roadmap

### Phase 0 — Research and repository preparation

First priority

Objectives:

- Establish WiMANS-only working scope.
- Review existing WiMANS manifest and split files.
- Verify repository state without changing historical work.
- Identify dataset paths and current preprocessing scripts.
- Document the exact input and annotation schemas.

Deliverables:

- WiMANS dataset specification.
- Experiment configuration.
- Verified split inventory.
- Research directory structure.

Exit condition: We know exactly which files represent the dataset, which split files are authoritative, and how each sample is labelled.

### Phase 1 — Dataset validation

Before preprocessing, validate the actual data.

| Validation     | What we need to establish              |
| -------------- | -------------------------------------- |
| Sample count   | 11,286 records                         |
| Tensor shape   | `(T, 3, 3, 30)`                        |
| Data type      | Expected numerical representation      |
| Time length    | Distribution of available time steps   |
| Labels         | Correct activity and user associations |
| User count     | 0–5 users per sample                   |
| Band           | Correct 2.4/5 GHz metadata             |
| Environment    | Correct environment assignment         |
| Data alignment | MAT and NPY sample correspondence      |

The paper specifies 3,000 intended time steps per sample, but packet loss means actual samples can be shorter. It reports average lengths of 2,864.39 time steps for 2.4 GHz and 2,930.72 for 5 GHz.&#x20;

2402.09430v2.pdf



Important: We must inspect the actual data before assuming every sample has the same temporal length.

Deliverables:

- Dataset validation report.
- Tensor shape distribution.
- Label distribution.
- Packet-loss statistics.
- MAT/NPY alignment findings.

### Phase 2 — Preprocessing pipeline

First implementation milestone

Start with amplitude-only preprocessing.

Proposed contract:

| Component        | Decision                    |
| ---------------- | --------------------------- |
| Input            | Original CSI amplitude      |
| Data type        | float32                     |
| Target length    | 3,000                       |
| Padding          | Right padding               |
| Mask             | Boolean valid-timestep mask |
| Normalization    | None initially              |
| Original data    | Preserve unchanged          |
| User annotations | Preserve all six user slots |

We will create:

```
src/
├── datasets/
│   └── wimans/
│       ├── loader.py
│       ├── validate.py
│       └── annotations.py
│
└── preprocessing/
    └── wimans/
        ├── padding.py
        ├── amplitude.py
        ├── masks.py
        └── features.py
```

The mask is important. Padding must never be interpreted as genuine CSI observations.

For LLM experiments, we will create a second representation using temporal windows. A starting point is 100 ms windows with mean, standard deviation, range, and temporal variation. The full-resolution tensors remain available for deep-learning models.&#x20;

CSI Dataset Setup.txt



Exit condition: The preprocessing pipeline is deterministic, tested, and does not modify the original dataset.

### Phase 3 — Define the multi-user prediction task

This is a critical research decision.

We should not treat WiMANS as an ordinary nine-class single-label classification problem.

Our proposed primary task is:

Per-user activity recognition: predict the activity associated with each occupied user in a sample.

For example:

One sample — three simultaneous users

| User   | Ground-truth activity |
| ------ | --------------------- |
| User 1 | Walking               |
| User 5 | Picking Up            |
| User 6 | Sitting Down          |

Expected prediction

| User   | Predicted activity |
| ------ | ------------------ |
| User 1 | Walking            |
| User 5 | Picking Up         |
| User 6 | Sitting Down       |

We must first verify whether user-slot ordering is stable in the annotation format. If it is not, predictions need a matching strategy rather than assuming that output slot 1 always represents the same person.

Our evaluation should include:

- Per-user activity accuracy.
- Macro precision, recall and F1.
- Per-class F1.
- Exact-set accuracy as a secondary metric.
- Performance grouped by number of simultaneous users.

Zero-user samples remain in the dataset but are excluded from the nine-class activity-recognition task. Presence detection will be a separate task.

### Phase 4 — Machine-learning baselines

We need a reliable benchmark before claiming that an LLM approach contributes anything.

The WiMANS paper evaluates eight Wi-Fi models:

| Model  | Priority                   |
| ------ | -------------------------- |
| ST-RF  | Initial classical baseline |
| MLP    | Baseline                   |
| LSTM   | Baseline                   |
| CNN-1D | High                       |
| CNN-2D | Medium                     |
| CLSTM  | High                       |
| ABLSTM | Medium                     |
| THAT   | High / advanced            |

The paper identifies CLSTM as a strong HAR model and evaluates THAT as a multi-scale convolution/Transformer architecture.&#x20;

2402.09430v2.pdf



My recommended implementation order:

1. Majority-class baseline.
2. ST-RF or simple classical baseline.
3. CNN-1D.
4. LSTM.
5. CLSTM.
6. THAT, after the simpler models are validated.

We should reproduce the paper's relevant model results where feasible, while documenting any differences in architecture, preprocessing, and evaluation.

### Phase 5 — LLM-based multi-user recognition

Only after the supervised baseline is working.

Proposed pipeline:

1. Divide each three-second CSI sample into temporal windows.
2. Extract window-level amplitude statistics.
3. Preserve temporal ordering.
4. Construct a structured representation of the CSI sample.
5. Supply the representation to an LLM.
6. Request predictions for the occupied users.
7. Parse predictions into a fixed, validated output structure.
8. Evaluate with the same ground-truth protocol used by the supervised models.

Potential experiments:

- Zero-shot LLM.
- Few-shot LLM.
- Domain-knowledge-enhanced prompting.
- Different statistical representations.
- Different window sizes.
- Different LLM models.

The Wi-Chat paper can inform this branch, but its reported results come from a self-collected dataset. They must not be treated as expected WiMANS performance.&#x20;

LLM_based.pdf



### Phase 6 — Generalization and ablation studies

We should investigate the factors that make multi-user activity recognition difficult.

| Experiment     | Question                                     |
| -------------- | -------------------------------------------- |
| 2.4 GHz only   | How well does the noisier band perform?      |
| 5 GHz only     | Does cleaner CSI improve recognition?        |
| Dual-band      | Does combining bands help?                   |
| One user       | What is the single-user baseline?            |
| 2–5 users      | How does interference affect performance?    |
| Environment    | Does the model generalize across rooms?      |
| Group-disjoint | Can it generalize to entirely unseen groups? |
| Window size    | How much temporal detail is useful?          |
| Normalization  | Does normalization improve performance?      |

## 5. Evaluation protocol

We already have two split protocols in the project.

Protocol A — Paper comparison

Secondary benchmark

- Sample-level random 80/20 split.
- Training: 9,029 samples.
- Testing: 2,257 samples.
- Designed for comparison with the paper's random-split approach.

Protocol B — Group-disjoint

Primary generalization evaluation

- 91 training groups.
- 23 completely held-out test groups.
- Training: 9,009 samples.
- Testing: 2,277 samples.
- Designed to test generalization to unseen groups.

These are our project's existing protocols, not two equivalent protocols from the original paper. The original paper randomly splits samples within each environment.&#x20;

2402.09430v2.pdf

CSI Dataset Setup.txt



One important limitation: all 99 zero-user samples in the group-disjoint test set belong to `meeting_room|2.4`. We must not use that split to make broad zero-person detection claims.&#x20;

CSI Dataset Setup.txt



We should use the group-disjoint protocol for our main generalization claims and retain the random split for paper comparison.

## 6. Experimental artifact structure

Every experiment must be independently reproducible.

```
experiments/
└── wimans/
    ├── baselines/
    │   ├── st_rf/
    │   ├── cnn1d/
    │   ├── lstm/
    │   ├── clstm/
    │   └── that/
    │
    ├── llm/
    │   ├── zero_shot/
    │   ├── few_shot/
    │   └── ablations/
    │
    └── configs/

results/
└── wimans/
    ├── baselines/
    ├── llm/
    └── comparisons/

reports/
└── wimans/
    ├── dataset_validation.md
    ├── preprocessing.md
    ├── baseline_results.md
    └── final_analysis.md
```

Each run should record:

- Dataset manifest version/hash.
- Split identifier.
- Model and configuration.
- Random seed.
- Preprocessing version.
- Training history.
- Prediction records.
- Evaluation metrics.
- Runtime and hardware information.

## 7. Suggested schedule

An initial planning estimate, assuming consistent access to the dataset and compute.

| Period | Milestone                                  |
| ------ | ------------------------------------------ |
| Day 1 | Dataset and repository audit               |
| Day 2 | Validation and preprocessing               |
| Day 3 | Baseline models                            |
| Day 4 | Multi-user model and evaluation            |
| Day 5 | LLM representation and initial experiments |
| Day 6 | Ablations and generalization               |
| Day 7 | Analysis and research report               |

The timeline is a target, not a promise. Dataset annotation or model-shape problems could extend the first two weeks.

## 8. Immediate next action

Start here

I recommend that our first task be a WiMANS-only read-only audit.

We should inspect:

1. Current Git status and repository instructions.
2. WiMANS manifest and both split files.
3. One amplitude `.npy` sample.
4. One corresponding MAT file, particularly its `trace` field.
5. The existing preprocessing scripts and their outputs.

Do not modify anything during this audit. The objective is to confirm that the existing data really matches the proposed preprocessing contract.

My strongest recommendation: get the data contract, user-slot semantics, and evaluation protocol right before writing model code. A sophisticated model trained on incorrectly aligned multi-user labels will produce impressive-looking but scientifically meaningless results.

Once that audit is complete, we can move directly into Phase 1 and build the WiMANS pipeline.
