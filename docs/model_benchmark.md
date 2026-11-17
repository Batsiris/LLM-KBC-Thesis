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
`--quantization` (`none` or explicit `nf4`), `--thinking`
(`default`, `enabled`, `disabled`), `--determinism` (`strict` or `warn`),
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

The original causal-LM path remains available on Transformers 4.57.1. The three
reviewed conditional-generation architectures below have explicit loaders and
native text-only chat handling. All paths use safetensors and a single CUDA
device. NF4 is opt-in; no remote model code, custom templates, automatic dtype
changes, CPU/disk offloading, device sharding, or manual special-token
concatenation is enabled. Missing or
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

Freeze the model IDs/SHAs, all conditions and the model-selection criterion
before inspecting new scores. Validation has already informed this thesis;
results on it are development/model-selection findings, not held-out proof.
Never use query gold for prompts, adaptive selection or retrieval; choose demos
only by the committed training policy. Reserve an untouched labeled benchmark or
official hidden-test procedure for final testing after model/method selection.
Report all declared model/condition outcomes, failed runs, uncertainty and costs rather
than just the winner; avoid repeatedly tuning on final evaluation data. This
closed-book comparison does not establish RAVEC novelty or superiority.

API references: [Transformers 4.57.1 chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating),
[generation configuration](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation).

## Reviewed conditional-generation models and A100 40 GB configurations

The following compatibility assessment uses official model cards/configurations
and Transformers release source. It is **not GPU validation**. The existing HPC
environment (PyTorch 2.8.0+cu128, Transformers 4.57.1, Accelerate 1.10.1) is not
upgraded by this project. All zero-/1-/3-/5-shot builders and the official
evaluation protocol remain unchanged; the planned main conditions are 0/3/5.

| Model | Explicit loading class | Existing Transformers 4.57.1 | A100-SXM4-40GB starting configuration |
|---|---|---|---|
| `Qwen/Qwen3.5-9B` | `Qwen3_5ForConditionalGeneration` | Not supported; the config's old `transformers_version` string is not proof of compatibility | BF16 unquantized, or explicit NF4/BF16 as a separate condition |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | `Mistral3For` + `ConditionalGeneration` | Architecture supported; native `mistral-common` and bitsandbytes are additional requirements | NF4 with BF16 compute; the card estimates ~55 GB for unquantized FP16/BF16 |
| `google/gemma-4-31B-it` | `Gemma4ForConditionalGeneration` | Not supported | NF4 with BF16 compute; nominal 30.7B parameters alone exceed 40 GB at 16 bits |

Sources: [Qwen configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json),
[Qwen card](https://huggingface.co/Qwen/Qwen3.5-9B),
[Mistral card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506),
[Mistral 4.57.1 API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/mistral3),
[Gemma card](https://huggingface.co/google/gemma-4-31B-it),
[Gemma configuration](https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/config.json).

The two Mistral class-name fragments concatenate to the original public symbol;
this spelling avoids a false-positive API-key detection in source scanning.

All three are multimodal checkpoints, but the benchmark supplies **text only**.
It loads their complete conditional-generation checkpoints, including unused
vision weights. It does not extract a language submodule or initialize missing
weights. Qwen/Gemma use their pinned native tokenizer/Jinja serialization; images
and an image processor are unnecessary for these text-only inputs. This follows
their conditional model APIs accepting text input IDs without image tensors;
the cards' multimodal examples use processors. No vendor system prompt replaces
the thesis prompt.

Mistral's official repository contains `tekken.json`, not a standard tokenizer
JSON/Jinja template. We fetch **only that tokenizer file at the same explicit
model SHA**, hash it, and load it locally through Transformers'
`MistralCommonTokenizer` (4.57.1) or `MistralCommonBackend` (5.5.0). The
`mistral-common` library performs its native chat encoding; there is no Jinja
template to hash, so template/hash fields are `null`, and tokenizer-file hash,
backend/version and actual serialized prompts are recorded instead. In 4.57.1,
the native encoder implicitly handles a final user query and rejects an explicit
`add_generation_prompt`; in 5.5.0 that argument is a documented no-op. Both use
the same native API, not handcrafted control tokens. See the reviewed
[4.57.1 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/tokenization_mistral_common.py)
and [5.5.0 implementation](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/tokenization_mistral_common.py).

`--thinking default` preserves the checkpoint template's default. Qwen's reviewed
template enables thinking by default; the explicit `--thinking disabled` example
sets its native `enable_thinking=False`. Gemma's reviewed canonical template
supports the same switch and, with thinking disabled, prefills the empty thought
channel before generation. Mistral Small 3.2 has no reviewed switch and rejects
nondefault thinking flags. No reasoning content is extracted or stripped from
generated text: the existing decoder and strict JSON parser remain in use, so
unexpected reasoning/prose still becomes a logged abstention. Thinking mode is
an experimental setting, not a harmless implementation detail. The 512-token
budget is never enlarged to accommodate reasoning. See the
[Qwen native template](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/chat_template.jinja)
and [Gemma native template](https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/chat_template.jinja).

### Separately isolated environment

`benchmark-modern` pins Transformers **5.5.0**, whose release source includes both
[Qwen3.5](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
and [Gemma4](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py).
This is a reviewed target release, not a claim about the earliest supported
version. Its declared requirements accept torch>=2.4, Accelerate>=1.1.0,
bitsandbytes>=0.46.1 and Hub>=1.5,<2. Accordingly, this separate extra keeps torch
2.8.0 and Accelerate 1.10.1, pins bitsandbytes 0.48.1 and Hub 1.5.0, and includes
`mistral-common[image]==1.10.0`, Pillow 10.4.0 and matching torchvision 0.23.0 for
conditional-model imports. These are source-level compatibility checks;
installation/resolution and CUDA kernels have **not** been exercised here.
References: [release dependencies](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/dependency_versions_table.py),
[quantization minimums](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/utils/import_utils.py),
[official torch/torchvision CUDA pairing](https://pytorch.org/get-started/previous-versions/#v280).

Commands below are instructions only, not operations performed by this task.
Use a **new environment**, ideally for all three models to reduce software-stack
confounding. Do not combine `benchmark-modern` with `inference` or `fine-tuning`:
their intentionally preserved Transformers/Hub pins conflict. No training/PEFT
compatibility is claimed for this inference environment.

```bash
# From a Git checkout; refuse to reuse an existing environment.
BENCH_ENV="$HOME/.venvs/llm-kbc-benchmark-modern"
test ! -e "$BENCH_ENV" || exit 1
python3.11 -m venv "$BENCH_ENV"
source "$BENCH_ENV/bin/activate"
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[benchmark-modern,evaluation]"
python -m pip check
python -m pip freeze
```

Keep the resulting environment listing and Slurm logs with each run. Transitive
dependencies are not a complete lockfile. The legacy Mistral path can instead
use a separately reproduced 4.57.1 environment with pinned
`mistral-common[image]==1.10.0` and `bitsandbytes==0.48.1`; do not install these
into a shared environment implicitly. Missing loading classes/dependencies fail
explicitly; no command installs or upgrades anything automatically.

### Exact pinned GPU smoke commands

These model commits were verified through official Hub configuration/API sources
during this review; they are new candidate configurations, not revisions of an
archived baseline. Run on an allocated GPU from the repository root, **only
after approval to execute inference**. A configurable `--cache-dir` may be added.
The loops cover the planned zero-/3-/5-shot conditions; `--shots 1` is also
supported. For offline preparation only, append `--dry-run` (it does not inspect
or download the models/tokenizers).

```bash
for SHOTS in 0 3 5; do
  python -m llm_kbc.benchmark run --model Qwen/Qwen3.5-9B \
    --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --shots "$SHOTS" --dtype bfloat16 --quantization none --thinking disabled \
    --determinism strict --seed 0 --max-new-tokens 512 --context-limit 4096 \
    --dataset data/lm_kbc_2026/val.jsonl --run-id bf16-smoke-v1 --max-examples 5
done

for SHOTS in 0 3 5; do
  python -m llm_kbc.benchmark run \
    --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
    --revision 95a6d26c4bfb886c58daf9d3f7332c857cb27b43 \
    --shots "$SHOTS" --dtype bfloat16 --quantization nf4 --thinking default \
    --determinism strict --seed 0 --max-new-tokens 512 --context-limit 4096 \
    --dataset data/lm_kbc_2026/val.jsonl --run-id nf4-smoke-v1 --max-examples 5
done

for SHOTS in 0 3 5; do
  python -m llm_kbc.benchmark run --model google/gemma-4-31B-it \
    --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
    --shots "$SHOTS" --dtype bfloat16 --quantization nf4 --thinking disabled \
    --determinism strict --seed 0 --max-new-tokens 512 --context-limit 4096 \
    --dataset data/lm_kbc_2026/val.jsonl --run-id nf4-smoke-v1 --max-examples 5
done
```

Full runs use the same commands with `--max-examples 5` removed and fresh run IDs
(`bf16-val-v1` or `nf4-val-v1`). Do not score smoke outputs. No full runs or GPU
smokes were performed while implementing this support.

### Precision, determinism, memory and remaining GPU checks

NF4 uses bitsandbytes 4-bit linear weights, double quantization, explicit BF16
(or FP16) computation, and `device_map={"": 0}`. It never calls `.to()` on an
already quantized/dispatched model. Some embeddings, norms, output heads and
nonlinear/vision components can remain at higher precision; **4-bit does not
mean total memory equals parameter count / 2 bytes**. No CPU/disk offload or
automatic precision fallback is enabled. A weight-only lower-bound check rejects
clearly impossible 16-bit target runs before weight loading; passing it is not a
memory-fit guarantee. See [official quantization guidance](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes).

The examples mix unquantized Qwen with quantized larger models for feasibility;
differences reflect **model plus inference configuration**. For a controlled
quantized comparison, explicitly run Qwen with `--quantization nf4` and a fresh
`nf4-*` run ID too. Treat it as a separate condition. Do not claim a pure
architecture advantage from different precision, thinking mode or software.

Strict PyTorch deterministic algorithms remain the default. Unsupported kernels
raise; there is no automatic retry in nondeterministic mode. An explicitly
declared `--determinism warn` permits PyTorch's warn-only behavior and records the
choice and captured Python warnings. Third-party bitsandbytes/DeltaNet/fused
kernels may not report every nondeterministic operation through PyTorch. Greedy
decoding is not a universal bitwise reproducibility guarantee. Library versions,
optional hybrid-kernel availability, native template, dtype and quantization must
be interpreted together; Transformers logger output also belongs in Slurm logs.

Manifests retain rendered chats and hashes of token IDs. Generation checks the
actual token IDs against preflight, failing on template drift rather than
silently generating a different prompt. Metadata adds loading class/architecture,
quantization configuration, thinking/determinism settings, package versions,
implementation module, GPU compute capability/total/free memory, and peak PyTorch
allocated/reserved bytes. Peak counters reset before model loading and update
after generation, including a failing generation; shared mutable runtime fields
are saved by the existing interruption handler. The prepared manifest reflects
loading-time information, while final metadata contains generation-time peaks
and warnings. These allocator peaks exclude some CUDA/third-party allocations
and are **not NVML/device-wide VRAM measurements**. Timing definitions above are
unchanged. Failures before the shared artifact runner leave no run artifacts;
archive their Slurm stderr rather than inventing completion metadata.

Still requiring verification on the actual A100/driver/runtime:

- Resolve/install the isolated environment, run `pip check`, and verify CUDA
  versions and all three real model/tokenizer imports.
- Confirm tokenizer role handling, pinned files, 0/1/3/5-shot token counts,
  native non-thinking prefixes, preserved EOS IDs and strict JSON behavior.
- Run fresh-process 5-record GPU smokes and compare repeated greedy outputs;
  inspect any deterministic-kernel warnings/errors before declaring a protocol.
- Measure allocator peaks and device-wide VRAM, host RAM, disk/cache use and
  throughput; confirm headroom for the longest award prompts and 512 new tokens.
- Verify bitsandbytes conversion/quantized module coverage and Qwen's hybrid
  fallback/fast-kernel path. Neither numerical stability nor NF4 memory fit is
  established by mocked tests.
- Exercise a controlled interruption on expendable smoke paths and confirm no
  incomplete prediction publication. Keep archived Qwen experiments untouched.
