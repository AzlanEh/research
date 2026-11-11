# ADR 0001: SiMWiSense LLM classifier boundary and result validity

- **Status:** Accepted
- **Date:** 2026-11-11
- **Scope:** LLM classification and evaluation architecture. Documentation only.
- **Supersedes:** none
- **Related commits:** `6f2b0a4`, `3ce740f`, `2dfebb5`

## Context

The repository supports two datasets with fundamentally different input and
label structures:

- **SiMWiSense** — complex CSI windows of shape `50 x 242`, labelled `A`, `B`,
  `C`. Labels are dataset identifiers whose physical activity mapping is
  unverified. Evaluation is single-label.
- **WiMANS** — amplitude tensors of shape `(T, 3, 3, 30)`, with 9 activities,
  0–5 users per sample, and per-user activity records. Evaluation is natively
  multi-user / multi-label.

The LLM code was originally written only for SiMWiSense. A refactor separated
the API client and response parser into `src/common/llm/`. The open question
was whether to continue toward a generic classifier abstraction in preparation
for WiMANS.

An inspection of the SiMWiSense classifier, prompt builder, evaluation code,
and saved experiment artifacts found that **the target record's ground truth is
included in the model-visible prompt**. This invalidates the existing accuracy
figures as evidence of classification performance and must be recorded before
further implementation.

## Decision summary

| Area | Decision |
|---|---|
| Context | The two datasets have different input shapes, label cardinality, and evaluation semantics. |
| SiMWiSense boundary | The classifier, prompt builder, and evaluation assumptions remain dataset-specific. |
| Shared infrastructure | Only the API client and response parser are shared, with documented caveats. |
| Data leakage | Target `label`, `source_file`, and `split` must not enter model-visible target data. |
| Historical results | All four existing experiment runs are contaminated and unsuitable for classification claims. |
| Evaluation | API failures must be distinguished from model parsing failures and valid predictions. |
| Non-goals | This ADR makes no code changes, deletes no artifacts, and does not implement WiMANS. |

## 1. SiMWiSense boundary

`src/llm/classifier.py` and `src/llm/prompt_builder.py` remain
SiMWiSense-specific. They encode assumptions that do not transfer to WiMANS:

- fixed labels `A`, `B`, `C` (plus `UNKNOWN`);
- a `50 x 242` window description;
- SiMWiSense feature-group names and summary statistics;
- a single-label JSON response contract;
- single-label accuracy, precision, recall, F1, and confusion matrix.

### Input contract

`classify_record()` performs no schema validation. It reads fields with
`.get()`, so missing data silently becomes `null` or an empty list.

| Field | Status | Use |
|---|---|---|
| `label` | Assumed present | Copied to `ground_truth`; **also sent to the model** |
| `source_file` | Assumed present | Prediction identity; **also sent to the model** |
| `split` | Optional | Copied to prediction; **also sent to the model** |
| `shape` | Optional | Sent to the model, otherwise unused |
| `dtype` | Optional | Not used by prompts or evaluation |
| `amplitude_features.overall` | Optional | Structured representation |
| `amplitude_features.per_time_mean` | Required for amplitude mode in practice | `[]` fallback yields an empty series |
| `amplitude_features.per_time_std` | Optional | Per-time-point standard deviation |
| `phase_features.*` | Optional | Five summary fields |
| `temporal_features.*` | Optional | Three summary fields |
| `gain_diagnostics.*` | Optional | Four summary fields |

This contract is not generalised. A WiMANS record has different fields and
different label semantics, and must define its own validated contract.

### Label handling

Label sources are not connected:

| Location | Value | Applied? |
|---|---|---|
| `prompt_builder.CLASS_LABELS` | `["A", "B", "C"]` | Default bullet list |
| `build_*_prompt(class_labels=...)` | caller-supplied | Partially — replaces only the bullet block |
| `classifier.ALLOWED_LABELS` | `{"A", "B", "C", "UNKNOWN"}` | Response validation |
| `metrics.VALID_LABELS` / `confusion_matrix.VALID_LABELS` | `("A", "B", "C")` | Metrics |
| `experiment_manifest["labels"]` | `["A", "B", "C"]` | Passed to `run_evaluation` only |

`class_labels` does not change the `50 x 242` description, the
`"A | B | C | UNKNOWN"` JSON example, or the `"A, B and C"` constraint
sentence. `classify_record()` never accepts or forwards `class_labels` and
always parses with `ALLOWED_LABELS`. The public classifier path is therefore
hard-wired to SiMWiSense labels.

### Prompt builder

Observations, not fixed by this ADR:

- `PROMPT_VERSION = "v1_zero_shot"` is returned for both zero-shot and
  few-shot runs; `prompt_mode` is the only distinguishing field.
- `build_few_shot_prompt()` contains an unreachable second `user_content`
  block after its first `return`.
- The amplitude representation sends 50 `(t, mean, std)` points, not the raw
  `50 x 242` window, despite the system prompt describing a 242-subcarrier
  window.
- System-prompt construction is duplicated between the zero-shot and few-shot
  builders.

### Prediction contract

Always present:

```text
source_file, ground_truth, split, predicted_label, confidence, reason,
parse_status, model, latency_ms, token_usage, prompt_version,
representation, prompt_mode, error, prompt_messages
```

Only on success: `raw_response`.

`parse_status` domain:

```text
ok | empty | json_error | schema_error | label_error   (response_parser)
client_error                                           (classifier)
```

Evaluation requires only `ground_truth`, `predicted_label`, and
`parse_status`. `save_prompts` additionally requires `prompt_messages`.

On `client_error`, `model` is the placeholder `"configured_model"`, not the
actual model.

### Evaluation dependencies

`compute_metrics(predictions, labels=("A", "B", "C"))`:

- `parse_error_count = total - count(parse_status == "ok")` — conflates API
  failure with model output failure;
- `unknown_count = count(predicted_label == "UNKNOWN")` — no status filter;
- `evaluable = count(ground_truth in labels and predicted_label in labels)` —
  no status filter;
- `accuracy = correct / evaluable`;
- macro precision, recall, F1 over `labels`.

`compute_confusion_matrix(predictions, labels=("A", "B", "C"))`:

- `status != "ok"` is counted as a parse error and skipped **first**;
- then `predicted_label == "UNKNOWN"` is counted as unknown;
- then valid `ground_truth`/`predicted_label` pairs enter the matrix.

Observed divergence: a non-`ok` row carrying a valid `predicted_label` is
counted as evaluable by `compute_metrics()` but as a parse error by
`compute_confusion_matrix()`. This is broader than the previously recorded
`non-ok + UNKNOWN` case. It is not fixed here.

## 2. Shared infrastructure boundary

| Component | Verdict | Caveats |
|---|---|---|
| `src/common/llm/client.py` | Reusable as-is | No dataset semantics. Import-time environment configuration and the pinned retry defects remain. |
| `src/common/llm/response_parser.py` | Reusable with explicit labels | Defaults to `{A, B, C, UNKNOWN}`; an empty `allowed_labels` set falls back to defaults; `UNKNOWN` is a hard-coded semantic; JSON extraction uses a greedy regex. |
| `src/evaluation/metrics.py` | Conditionally reusable | Single-label only. WiMANS is natively multi-user. |
| `src/evaluation/confusion_matrix.py` | Conditionally reusable | Same limitation and same divergence quirk. |
| `src/evaluation/evaluate.py` | SiMWiSense-owned for now | Artifact schema and naming are SiMWiSense-shaped; no second consumer exists. |
| `src/llm/classifier.py` | SiMWiSense-owned | Hard-wired labels; no `class_labels` propagation. |
| `src/llm/prompt_builder.py` | SiMWiSense-owned | Fixed dimensions, fixed JSON example, SiMWiSense feature translation. |

No generic `Classifier` base class, protocol, or registry is introduced. Two
datasets with different label cardinality, input tensors, and evaluation
strategies share no classifier behaviour beyond "takes a record, returns a
prediction". A generic abstraction would add indirection without a second
real implementation.

**Boundary rule:** prompt builders may receive ground-truth labels for
*example* records. They must never receive ground-truth labels, class-revealing
paths, or split metadata for the *target* record.

## 3. Data leakage

### Observed

Both prompt representations serialize the target record's `label`,
`source_file`, and `split`:

```python
# src/llm/prompt_builder.py — format_feature_record()
# src/llm/prompt_builder.py — format_amplitude_time_series()
"label":       record.get("label"),
"source_file": record.get("source_file"),
"split":       record.get("split"),
```

A saved prompt shows:

```json
{"label": "A", ..., "source_file": "Slots/Test/A_batch/batch_714.mat", "split": "Test", ...}
```

`source_file` independently encodes the class (`A_batch`) and the split
(`Slots/Test`).

### Model-visible effect

Model-generated `reason` strings confirm the metadata was read:

```text
"The label is explicitly provided in the input record as 'A'."
"The input record explicitly includes the matching label 'A' in its metadata fields (source_file and label)."
"The input record explicitly specifies the dataset label as 'A' in its metadata fields ('label': 'A', 'source_file': 'Slots/Test/A_batch/batch_594.mat')."
```

Counts of `ok` predictions whose `reason` matches
`/label|source_file|source file|metadata|file path|path/i`:

| Run | `ok` | Reasons citing leak |
|---|---:|---:|
| `experiment_01_zero_shot_structured_features` | 66 | 62 |
| `experiment_01_zero_shot_amplitude_time_series` | 86 | 85 |

### Decision

The existing results are **contaminated and invalid for
classification-performance claims**. They must not be cited as evidence that
the model can classify CSI activity.

This ADR does **not** claim that every prediction was necessarily copied from
the supplied label. The defensible statement is that the prompts contained
ground truth and class-revealing metadata, and the model's explanations show
that metadata was used. That is sufficient to invalidate the accuracy figures.

### Rules going forward

1. Target serialization must exclude `label`, `source_file`, `split`, and any
   other ground-truth-revealing metadata.
2. Ground truth remains available locally in the prediction record for
   evaluation.
3. Legitimate few-shot example labels remain in the prompt; only the target
   record is sanitized.
4. Regression tests for the future fix must assert that the serialized target
   payload contains no ground-truth information.

## 4. Historical results

| Run | Total | `ok` | `client_error` | Evaluable | UNKNOWN | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `experiment_01_zero_shot_structured_features` | 150 | 66 | 84 | 61 | 5 | 1.0 |
| `experiment_01_zero_shot_amplitude_time_series` | 150 | 86 | 64 | 84 | 2 | 1.0 |
| `experiment_02_few_shot_structured_features` | 150 | 12 | 138 | 12 | 0 | 1.0 |
| `experiment_02_few_shot_amplitude_time_series` | 150 | 12 | 138 | 12 | 0 | 1.0 |

Three independent problems:

1. **Ground-truth leakage** — the target's `label` is in the prompt.
2. **Metadata leakage** — `source_file` reveals the class and split.
3. **API-success selection** — accuracy is conditional on requests that
   completed successfully. In exp02, only 12 of 150 requests succeeded.

The third issue means that even a leakage-free number from these runs would
describe a non-random subset, not the full evaluation set.

### Decisions

- Keep the original result artifacts unchanged for historical traceability.
- Do not use the `1.0` accuracy figures in reports, comparisons, or
  publications.
- Do not present these runs as a baseline for future leakage-free experiments.
- In all four artifacts the only non-`ok` status is `client_error`; none of
  the reported parse errors are model parsing failures.

## 5. Evaluation

Future evaluation must report at least three separate quantities:

| Quantity | Definition |
|---|---|
| API failures | Requests that never produced a model response (`client_error`) |
| Parse / schema failures | Responses that arrived but were empty, non-JSON, invalid schema, or invalid label |
| Valid predictions | Parsed predictions evaluated against ground truth |

Decisions:

- `client_error` must not be aggregated into "parse error".
- UNKNOWN handling must be reported explicitly and must not be hidden by an
  accuracy denominator that excludes it.
- Metrics and confusion matrix must agree on which rows are evaluable.
- Any new evaluation must state its denominator and the count of excluded
  records.

These are requirements for the eventual leakage-free evaluation. The pinned
defects are not fixed in this ADR.

## 6. Non-goals

This ADR explicitly does **not**:

- change any code;
- change prompt text or prompt versions;
- delete, rewrite, or move result artifacts;
- regenerate experiments or call the LLM API;
- fix the pinned client, parser, or evaluation defects;
- introduce a generic classifier abstraction;
- implement a WiMANS classifier, prompt, loader, or evaluation;
- modify datasets, manifests, splits, or external data;
- push to a remote.

## 7. Consequences and follow-up

The next change is a separate leakage-removal implementation:

1. Define a target-only serialization contract.
2. Exclude `label`, `source_file`, `split`, and other ground-truth-revealing
   metadata from model-visible target input.
3. Retain ground truth locally for evaluation.
4. Add regression tests proving the target payload contains no ground truth.
5. Increment the prompt version.
6. Regenerate the SiMWiSense experiments.
7. Report API failures, parse failures, and valid predictions separately.

**Research rule:** the leakage-free accuracy must never be compared against the
old `1.0` as if the two were equivalent experimental conditions. They differ in
prompt content, prompt version, successful-request population, and validity.
Any comparison requires a new, leakage-free baseline.

## Evidence and reproduction

All findings above are read-only and reproducible from commit `2dfebb5`:

```bash
# Prompt leakage
PYTHONPATH=src ./.venv/bin/python - <<'PY'
from llm.prompt_builder import build_zero_shot_prompt
record = {"label": "A", "source_file": "A_batch/batch_0.mat", "split": "Test"}
print(build_zero_shot_prompt(record)[1]["content"])
PY
```

```bash
# Saved prompts and reasons
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("results/experiment_01_zero_shot_structured_features")
prompts = json.loads((base / "prompts.json").read_text())
print(prompts[0]["prompt_messages"][1]["content"][:400])

for line in (base / "predictions.jsonl").read_text().splitlines():
    row = json.loads(line)
    if row.get("parse_status") == "ok" and "label" in str(row.get("reason", "")).lower():
        print(row["ground_truth"], "->", row["predicted_label"], "|", row["reason"])
        break
PY
```

```bash
# Result counts
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

for run in sorted(Path("results").glob("experiment_0*")):
    metrics_path = run / "metrics.json"
    if not metrics_path.exists():
        continue
    m = json.loads(metrics_path.read_text())
    print(run.name, m["total_predictions"], m["evaluable_predictions"],
          m["unknown_count"], m["accuracy"], m["parse_statuses"])
PY
```
