# Multi-model prompting benchmark

`python -m llm_kbc.benchmark` runs one explicitly selected Hugging Face
causal/instruction model and prompting condition at a time. Repeat the same
protocol for each of the four researcher-selected models; there is no automatic
model selection, model registry, GPU scheduling, or RAG. Existing pinned Qwen
runners, prompts, archived outputs, QLoRA, and the official scorer are unchanged.
No new inference or benchmark results are established by this implementation.

## Frozen conditions

| `--shots` | Prompt version | Demonstration policy |
|---|---|---|
| 0 | `lm-kbc-2026-zero-shot-v2` | None; unchanged `zero_shot.build_messages()` |
| 1 | `lm-kbc-2026-few-shot-v2` | Unchanged v2 lower-median nonempty example |
| 3 | `lm-kbc-2026-few-shot-v1` | Unchanged v1 cardinality/value diversity policy |
| 5 | `lm-kbc-2026-few-shot-v2` | Unchanged v1 prefix plus v2 rank-quartile additions |

Selection delegates to `few_shot.py` (3) and `few_shot_v2.py` (1/5), without
model-dependent changes. Canonical first aliases and complete demonstration
answers retain spelling/order. The reviewed training SHA256 must match
`ad37cd30d1ff4b9f1ef2579b25e64093b202c40da11e8c412e13386f1e5d332e`.
All six relations must have sufficient eligible training examples. See the
README's Few-shot sections for the exact previously committed selection rules.
All conditions use the same existing system message and relation questions.

Only subject/relation fields enter inference prompts. Input gold fields are
ignored, including absent or withheld labels. Few-shot demonstrations use only
training gold. The **entire** input file is checked before a smoke prefix is
selected: supported relations, unique NFKC/casefold/whitespace identities, and
normalized demonstration pair leakage. Cross-relation subject-only overlaps are
reported without reselection. There is no alias-based contamination detection.

## Commands

Help and offline dry-run need only the base Python package. Actual GPU runs use
the existing optional dependencies; offline reports also require `evaluation`:

```bash
python -m pip install -e ".[inference,evaluation]"
python -m llm_kbc.benchmark --help
python -m llm_kbc.benchmark run --help
```

Choose a model and **exact 40-character lowercase Hub commit SHA** explicitly.
Replace these placeholders; no default model or floating `main` revision is
selected. Use identical seed, dtype, token budget, dataset and demonstration
snapshot for the four models. Repeat the commands with each approved model/SHA.

```bash
MODEL='organization/approved-instruction-model'
REVISION='REPLACE_WITH_EXACT_40_CHARACTER_COMMIT_SHA'

# Offline: no network, tokenizer, weights, GPU dependencies, or output creation.
python -m llm_kbc.benchmark run --model "$MODEL" --revision "$REVISION" \
  --shots 3 --dataset data/lm_kbc_2026/val.jsonl --run-id smoke-v1 --dry-run

# GPU smoke: optional cache-dir points to a configurable local/HPC cache.
python -m llm_kbc.benchmark run --model "$MODEL" --revision "$REVISION" \
  --shots 3 --dataset data/lm_kbc_2026/val.jsonl --run-id smoke-v1 --max-examples 5

# Individual complete runs, only after inspecting smoke artifacts.
python -m llm_kbc.benchmark run --model "$MODEL" --revision "$REVISION" \
  --shots 0 --dataset data/lm_kbc_2026/val.jsonl --run-id val-v1
python -m llm_kbc.benchmark run --model "$MODEL" --revision "$REVISION" \
  --shots 1 --dataset data/lm_kbc_2026/val.jsonl --run-id val-v1
python -m llm_kbc.benchmark run --model "$MODEL" --revision "$REVISION" \
  --shots 3 --dataset data/lm_kbc_2026/val.jsonl --run-id val-v1
python -m llm_kbc.benchmark run --model "$MODEL" --revision "$REVISION" \
  --shots 5 --dataset data/lm_kbc_2026/val.jsonl --run-id val-v1
```

All four conditions accept `--model`, `--revision`, `--seed` (default 0),
`--max-new-tokens` (512), `--dtype` (`float16`, or explicit `bfloat16`),
`--cache-dir`, `--data-dir`, `--output-root`, and `--max-examples`.
Use `--dry-run` with any condition. Dry-run validates data, demonstrations,
leakage, configuration and paths; **it cannot verify model availability, access,
chat-template support, context fit, CUDA support, or GPU memory**. It explicitly
reports `tokenizer_checked: false`. It does not reserve a run directory.

Actual runs load the pinned native tokenizer/config first and check **all**
templated input lengths plus the generation budget before loading model weights.
The context ceiling is the minimum finite declared model/tokenizer limit.
`--context-limit` can lower that ceiling or provide an explicitly verified limit
when none is declared; it cannot expand a declared context window. Nothing is
truncated. Model-native EOS IDs (including multiple stop IDs) are preserved;
padding falls back to a declared EOS ID only when necessary. Generation uses
greedy decoding, one beam/sequence, batch size one, eager attention, deterministic
PyTorch algorithms, TF32 disabled, and an explicit seed. Determinism is not a
promise of identical outputs across GPU types, library versions or models.

## Artifacts and offline reporting

Default layout (contents ignored by the existing `results/*` rule):

```text
results/model_benchmark/<model-slug>-<hash-of-model-and-revision>/<shots>-shot/<run-id>/
  predictions.jsonl
  predictions.jsonl.manifest.json
  predictions.jsonl.metadata.json
  predictions.jsonl.raw.jsonl
  predictions.jsonl.failures.jsonl
```

The printed JSON `output` field gives the exact prediction path. Model/revision
hashes avoid collisions between similar model names; conditions and explicit run
IDs separate smoke/full/repeated runs. Existing run directories, even empty or
interrupted ones, are refused. Ancestor symlinks are refused. A fresh run directory
is reserved exclusively before artifact publication; do not run competing tools
inside it. No resume or overwrite option exists. Prefer local/HPC storage and
archive all five artifacts together. This is not a multi-file transaction or a
guarantee against power loss or an external process replacing files.

The manifest retains ordered demonstrations, every preflighted message list,
prompt/selection versions, full-prompt hash, training/dataset hashes, explicit
configuration, source commit/dirty status/source hashes, and runtime metadata.
Native template text/hash, special tokens, templating arguments, tokenizer class,
context limits, actual token counts, model commit, generation configuration,
package versions and GPU/CUDA/cuDNN details are recorded. No Qwen context size or
special token is assumed. The shared `zero_shot._run_prepared()` preserves numeric
token spelling, strict JSON parsing, raw/failure logs, explicit parse-failure
empty-list abstentions, complete ordered coverage checks and final publication.
OOM/interruption leaves diagnostics and no incomplete prediction file.

Offline scoring is an **explicit separate command**; substitute a completed run's
printed prediction path:

```bash
python -m llm_kbc.benchmark report --data-dir data/lm_kbc_2026 --split val \
  --predictions /absolute/path/to/completed/run/predictions.jsonl
```

Reports require completion, matching prediction/manifest hashes, the exact
scoring-file fingerprint, full ordered coverage and raw-response/parser/counter
agreement. Partial smoke runs are rejected. Gold is accessed only by this
separate offline command and the existing official evaluation adapter.
Public test scoring is rejected. The unchanged official scorer supplies overall
and per-relation `macro-p`, `macro-r`, `macro-f1`, and micro metrics at full float
precision. Macro scores average per-query scores, not relation means; macro F1
is not the harmonic mean of macro P/R. JSON format validity is valid strict
responses / total inputs, counting accepted numeric normalization as valid and
parse-failure abstentions as invalid, even when empty gold makes them score well.
`inference_runtime_seconds` sums wall time inside generation calls (templating,
GPU generation/synchronization and decoding), excluding loading, preflight and
artifact writes; `model_load_seconds` is separate. See [evaluation.md](evaluation.md).

## Compatibility and experimental limitations

This supports text causal instruction models implemented by Transformers 4.57.1,
with native default chat templates and safetensors, on a single CUDA device.
No remote model code, quantization, custom templates, automatic dtype changes,
device sharding, or manual special-token concatenation is enabled. Missing or
ambiguous templates, unsupported system roles, omitted/changed message content,
unknown context limits and deterministic-kernel failures are explicit errors.
Some model families therefore need a separately reviewed prompt variant or a
future compatibility change; the CLI will not silently adapt their prompts.
Model access/license/gating and FP16/BF16 suitability must be checked per model.
Large models can OOM; no four-model GPU feasibility claim has been established.

Native templates may introduce reasoning defaults or different stopping behavior.
Inspect the recorded template and smoke outputs before freezing the four-model
protocol. Equal token budgets are comparable settings, not equal text lengths:
tokenizers differ and 512 tokens may truncate high-cardinality award answers.
Selection policies differ across shot counts by design; differences are not
causal estimates of demonstration count alone. Format repair is a separate
exploratory ablation and is never applied by this benchmark.

Freeze the four model IDs/SHAs, all conditions and the model-selection criterion
before inspecting new scores. Validation has already informed this thesis;
results on it are development/model-selection findings, not held-out proof.
Never use query gold for prompts, adaptive selection or retrieval; choose demos
only by the committed training policy. Reserve an untouched labeled benchmark or
official hidden-test procedure for final testing after model/method selection.
Report all 16 model/condition outcomes, failed runs, uncertainty and costs rather
than just the winner; avoid repeatedly tuning on final evaluation data. This
closed-book comparison does not establish RAVEC novelty or superiority.

API references: [Transformers 4.57.1 chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating),
[generation configuration](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation).
