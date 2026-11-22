# Multi-model prompting benchmark

`python -m llm_kbc.benchmark` runs one explicitly selected Hugging Face
causal/instruction model and prompting condition at a time. Repeat the same
protocol for each of the five researcher-selected models; there is no automatic
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
snapshot for the five models. Repeat the commands with each approved model/SHA.

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
`--quantization` (`none`, explicit `nf4`, or reviewed `fp8-dequantize`), `--thinking`
(`default`, `enabled`, `disabled`), `--determinism` (`strict` or `warn`),
`--cache-dir`, `--data-dir`, `--output-root`, and `--max-examples`.
Use `--expected-records 475` for the official validation snapshot (also before a smoke prefix is selected). Use `--dry-run` with any condition. Dry-run validates data, demonstrations,
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
device. NF4 is opt-in; no remote model code, custom templates, implicit dtype
changes, CPU/disk offloading, device sharding, or manual special-token
concatenation is enabled. Missing or
ambiguous templates, unsupported system roles, omitted/changed message content,
unknown context limits and deterministic-kernel failures are explicit errors.
Some model families therefore need a separately reviewed prompt variant or a
future compatibility change; the CLI will not silently adapt their prompts.
Model access/license/gating and FP16/BF16 suitability must be checked per model.
Large models can OOM; no five-model GPU feasibility claim has been established.

Native templates may introduce reasoning defaults or different stopping behavior.
Inspect the recorded template and smoke outputs before freezing the five-model
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

## Reviewed five-model roster

Verified on 2026-11-22 against official Hub repositories/API metadata and pinned
small configuration files; **no weights were downloaded and no GPU run was made**.
The active main conditions are 0/3/5 shots; 1-shot remains supported. All use
seed 0, 512 new tokens, context limit 4096, unchanged messages/selection policies,
eager attention, batch size one and strict deterministic-algorithm checks.

| Model | Immutable revision | Loading path | Explicit configuration / status |
|---|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Existing `AutoModelForCausalLM` / `qwen2` | Archived baseline; preserve original runner/environment/artifacts |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `Qwen3_5ForConditionalGeneration` | Completed 0/3/5-shot per researcher; preserve actual archived configuration/results |
| `mistralai/Ministral-3-14B-Instruct-2512` | `29439f81c2be264d8d393273f99e7db9c0961120` | `Mistral3For` + `ConditionalGeneration`, inner `ministral3` | **BF16 dequantization of supplied FP8 weights**, full checkpoint, single GPU |
| `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | `Gemma4ForConditionalGeneration` | Unchanged explicit NF4/BF16, thinking disabled |
| `ibm-granite/granite-4.1-8b` | `1504002f650e656a0a3789d99574df12e3e94ed0` | `GraniteForCausalLM`; upstream also registers `AutoModelForCausalLM` | Native BF16, no quantization, native template defaults |

The split spelling of the Mistral class concatenates to its exact public symbol;
it avoids a previously confirmed GitHub secret-scanning false positive.
**Historical only:** `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, revision
`95a6d26c4bfb886c58daf9d3f7332c857cb27b43`, was the former candidate. It is no
longer in `TARGETS`, proposed commands or model selection. Its implementation
history remains in Git (e.g. `be371ae`); no historical outputs are removed.

### Upstream verification and licenses

New models' official metadata and cards declare **Apache-2.0**. Preserve their
attribution/terms with any redistributed weights; this repository vendors none.
The older Qwen2.5 model has a custom Qwen license; Qwen3.5 and this Gemma4
checkpoint declare Apache-2.0. Check the actual pinned repository's terms/access
before redistribution rather than assuming terms from older model families.

- [Ministral official API](https://huggingface.co/api/models/mistralai/Ministral-3-14B-Instruct-2512),
  [pinned card](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512/blob/29439f81c2be264d8d393273f99e7db9c0961120/README.md),
  [configuration](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512/blob/29439f81c2be264d8d393273f99e7db9c0961120/config.json).
- [Granite official API](https://huggingface.co/api/models/ibm-granite/granite-4.1-8b),
  [pinned card](https://huggingface.co/ibm-granite/granite-4.1-8b/blob/1504002f650e656a0a3789d99574df12e3e94ed0/README.md),
  [configuration](https://huggingface.co/ibm-granite/granite-4.1-8b/blob/1504002f650e656a0a3789d99574df12e3e94ed0/config.json),
  [Transformers 5.5 Auto mapping](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/auto/modeling_auto.py).
- Retained [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct),
  [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)
  and [Gemma4](https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/README.md)
  official repositories. Their previous loaders/settings are unchanged.

Small-file SHA256s independently computed from the pinned upstream bytes:

| File | Ministral | Granite |
|---|---|---|
| `config.json` | `fc51295bdc0b39c015c9c3ca5d649de183eae73a23b8cbab1a0e055be8d5fa04` | `dea9d856cb57018117fe2fe3366f37cb4aa39424890061db2c0045a6a4efbda0` |
| `generation_config.json` | `e0923390059f84a9180b00e5501778acc45ea9856cd7f2fd68208b360927c677` | `9117fb03fed79dbb459373edeef9a3eec966bce52a1842e03a6716e83117f0d1` |
| `tokenizer_config.json` | `4a05af595aa11c3436d3b68a616219ae99b303426f708a12b31bc9ce32530728` | `a5ec5daab12ba090a90f3dd169c8f9c275557013a87b9c1258dc7cb497a35c86` |

### Ministral: FP8 source, explicit BF16 runtime

The pinned repository supplies `tekken.json`, HF tokenizer/config/Jinja files,
four HF safetensors shards plus index, and separate consolidated weights. Use
the HF loader's complete checkpoint, including the unused Pixtral vision tower;
no submodel extraction, missing-weight initialization or vision inputs.
The configuration declares static FP8, per-tensor scaling (`weight_block_size:
null`), BF16 for remaining weights, and a 262144-token text context.

An A100 (compute capability 8.0) cannot use the reviewed native FP8 path.
Transformers 5.5 would automatically dequantize below capability 8.9; we instead
**require** `--quantization fp8-dequantize --dtype bfloat16`. The adapter passes
`FineGrainedFP8Config(dequantize=True)` with all original scaling/skip-module
settings preserved, and `device_map={"": 0}`. Both implicit conversion (`none`)
and unreviewed FP8-to-NF4 conversion are rejected for this model. BF16 expansion
cannot recover information lost during the upstream FP8 quantization; this is
not an original unquantized BF16 checkpoint. Native FP8 is not offered here.
Sources: [card BF16 instructions](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512/blob/29439f81c2be264d8d393273f99e7db9c0961120/README.md),
[5.5 FP8 hardware checks](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/quantizers/quantizer_finegrained_fp8.py),
[dequantization implementation](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/integrations/finegrained_fp8.py),
[configuration/loading attributes](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/utils/quantization_config.py).

Use the native `MistralCommonBackend` in 5.5, with pinned local `tekken.json`,
not a handcrafted template or a silent switch to HF Jinja. The card requires
mistral-common >=1.8.6; the existing isolated pin 1.10.0 meets that requirement.
Its encoder handles system/user/assistant turns and the final assistant boundary;
`add_generation_prompt` is a documented no-op. Template/hash are null for this
backend: record tokenizer-file SHA, library version, rendered chats and token-ID
hashes instead. Original message contents must survive serialization unchanged.
No vendor system prompt replaces ours. `--thinking default` is required.
The pinned generation config declares BOS 1, EOS 2, PAD 11; native tokenizer pad
availability may differ, so the existing declared-EOS fallback remains recorded.
[Native tokenizer implementation](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/tokenization_mistral_common.py).

The official parameter inventory totals 13,945,031,680, including vision weights:
BF16 weight storage alone is about **27.9 GB decimal (26.0 GiB)**. A single A100
40GB is plausible at batch one/context 4096, but KV cache, conversion temporaries,
activations and CUDA overhead need measurement. The card's smaller FP8 memory
claims do not establish A100 BF16 fit. No automatic offload or retry is allowed.

### Granite: native causal LM and chat formatting

The pinned checkpoint is `model_type=granite`, `architectures=[GraniteForCausalLM]`,
BF16, with a 131072-token context and no upstream quantization configuration.
Transformers' Auto mapping supports it; we select the explicit Granite class.
The repository includes tokenizer JSON/config, GPT2 vocab/merges, Jinja template,
special-token map, generation config and four safetensors shards plus index.
Native `AutoTokenizer.apply_chat_template()` inserts
`<|start_of_role|>ROLE<|end_of_role|>`, then content and `<|end_of_text|>`;
`add_generation_prompt=True` provides the assistant header. These tokens are
**never assembled in benchmark code**. No thinking/template override is applied.
The pinned config declares BOS/EOS 100257 (`<|end_of_text|>`) and PAD 100256
(`<|pad|>`); preserve the model's EOS configuration. Sources:
[pinned template](https://huggingface.co/ibm-granite/granite-4.1-8b/blob/1504002f650e656a0a3789d99574df12e3e94ed0/chat_template.jinja),
[tokenizer config](https://huggingface.co/ibm-granite/granite-4.1-8b/blob/1504002f650e656a0a3789d99574df12e3e94ed0/tokenizer_config.json).
Its 8,791,592,960 parameters need about **17.6 GB decimal (16.4 GiB)** for BF16
weights, before runtime overhead. BF16/no quantization is the proposed A100
configuration, not a GPU-verified fit guarantee.

### Separately isolated environment (unchanged pins)

Keep the existing QLoRA/inference Transformers 4.57.1 environment untouched.
`benchmark-modern` already pins Transformers 5.5.0, torch 2.8.0, torchvision
0.23.0, Accelerate 1.10.1, bitsandbytes 0.48.1, Hub 1.5.0,
mistral-common[image] 1.10.0 and Pillow 10.4.0. No dependency changes are needed.
These are source/API compatibility checks; installing the modern stack and
exercising its CUDA kernels remain HPC checks. Do not combine these extras with
`inference` or `fine-tuning` (intentionally conflicting Transformers/Hub pins).
[Release dependencies](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/dependency_versions_table.py),
[torch/torchvision pairing](https://pytorch.org/get-started/previous-versions/#v280).

For a new environment only (do not recreate an existing completed-run environment):

```bash
BENCH_ENV="$HOME/.venvs/llm-kbc-benchmark-modern"
test ! -e "$BENCH_ENV" || exit 1
python3.11 -m venv "$BENCH_ENV"
"$BENCH_ENV/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$BENCH_ENV/bin/python" -m pip install -e ".[benchmark-modern,evaluation]"
"$BENCH_ENV/bin/python" -m pip check
```

Verify the actual GPU driver supports the chosen CUDA wheel. Site documentation
can lag the installed driver; this is not permission to upgrade a shared stack.
Record `pip freeze`, driver/runtime and Slurm logs for each run. Dependencies are
pinned at the top level, not a complete transitive lockfile.

### Exact Aristotle smoke/full commands

Run from the repository root, using the existing verified modern environment.
The single parameterized [Slurm template](../scripts/slurm/benchmark.sbatch)
provides both new-model smoke/full configurations. It requests **one ampere GPU**
(`--gpus=1`), 96GB host RAM and 05:55:00 per array task. Each shot condition gets
its own job; the array throttles to one concurrent task. It does not put all
three full conditions into one six-hour allocation. Sources:
[Aristotle GPU directives](https://hpc.it.auth.gr/jobs/gpu-slurm/),
[six-hour ampere limit](https://hpc.it.auth.gr/nodes-summary/).

```bash
export BENCH_PYTHON="$HOME/.venvs/llm-kbc-benchmark-modern/bin/python"
# Optional: configure scratch/cache paths explicitly; no username is hardcoded.
# export BENCH_CACHE_DIR=/your/existing/huggingface-cache
# export BENCH_OUTPUT_ROOT=/your/new/benchmark-results
mkdir -p results/slurm
bash -n scripts/slurm/benchmark.sbatch
"$BENCH_PYTHON" -m pip check

# CPU-only preparation: no tokenizer/network/weights/output creation.
"$BENCH_PYTHON" -m llm_kbc.benchmark run \
  --model mistralai/Ministral-3-14B-Instruct-2512 \
  --revision 29439f81c2be264d8d393273f99e7db9c0961120 --shots 3 \
  --dtype bfloat16 --quantization fp8-dequantize --thinking default \
  --seed 0 --max-new-tokens 512 --context-limit 4096 --determinism strict \
  --dataset data/lm_kbc_2026/val.jsonl --expected-records 475 --run-id preflight-v1 --dry-run

# Three five-record GPU smokes each, in distinct model/condition/run directories.
sbatch scripts/slurm/benchmark.sbatch ministral smoke v1
sbatch scripts/slurm/benchmark.sbatch granite smoke v1

# ONLY after all smoke artifacts and the checklist below have been reviewed:
sbatch scripts/slurm/benchmark.sbatch ministral full v1
sbatch scripts/slurm/benchmark.sbatch granite full v1
```

The script pins both new revisions and verifies exact train/validation file
hashes, then requires all **475** validation identities before selecting any
smoke prefix. It records environment packages and GPU inventory in append-only
Slurm output. Add site-required account flags to `sbatch` if needed; no account
is guessed. Cached snapshots are reusable, but run directories are never reused;
choose a fresh reviewed tag for repeats. Output IDs include model/revision,
shots, precision, smoke/full and tag. No automatic scoring/resume is performed.
The pre-timeout INT requests diagnostic finalization; a hard kill may leave
metadata at running, but cannot publish a partial final prediction file.
If throughput predicts >05:55 per full condition, stop and plan an explicitly
reviewed collection protocol; do not silently shard/resume or extend budgets.

Gemma's existing explicit configuration stays unchanged. On an allocated GPU:

```bash
for SHOTS in 0 3 5; do
  "$BENCH_PYTHON" -m llm_kbc.benchmark run --model google/gemma-4-31B-it \
    --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
    --shots "$SHOTS" --dtype bfloat16 --quantization nf4 --thinking disabled \
    --determinism strict --seed 0 --max-new-tokens 512 --context-limit 4096 \
    --dataset data/lm_kbc_2026/val.jsonl --expected-records 475 \
    --run-id nf4-smoke-v1 --max-examples 5
done
```

Full Gemma commands remove `--max-examples 5` and use a fresh `nf4-val-v1` run ID;
run each condition within its own reviewed allocation. The completed Qwen3.5
and archived Qwen2.5 experiments are references, not requests to rerun them.
Their old entry points, prompt versions, manifests, outputs and metrics remain
untouched. Do not overwrite historical artifacts to make configurations match.

### Fairness and verification checklist

This comparison concerns **model plus inference configuration**, not architecture
alone: FP8-origin/BF16 Ministral, native BF16 Granite, NF4/BF16 Gemma and actual
archived Qwen configurations differ. Thinking modes and tokenizers also differ.
Retain the actual archived settings; do not relabel prior results as if produced
with this new stack. A separately declared matched-precision study is optional,
not a reason to change completed experiments. Native templates add different
headers/EOS and equal token budgets do not imply equal answer lengths. Awards
may truncate at 512 tokens; strict failures stay logged abstentions, without
format repair/retries. Validation-based model selection is not held-out evidence.

Before full runs, verify on the **actual A100 and isolated environment**:

- [ ] `pip check`, native class/config/tokenizer imports, access/licenses, exact
  resolved commits, CUDA driver/wheel compatibility and available scratch space.
- [ ] All 475 identities, reviewed training SHA, all six relations, unchanged
  ordered demonstrations, no gold in queries and no demonstration-pair leakage.
- [ ] All full-file 0/3/5-shot template/token counts fit input + 512 <=4096;
  native system/user/assistant serialization, token-ID hashes and EOS are intact.
- [ ] Ministral really loads the full FP8-origin checkpoint as BF16 with original
  per-tensor scales and no CPU/disk offload; confirm converted module coverage
  and finite logits. Source inspection/mocks cannot verify conversion numerics.
- [ ] Granite native causal model/template, Gemma NF4 coverage, and unchanged
  Qwen paths behave as recorded; do not silently substitute loading classes.
- [ ] Review JSON validity, raw/failure logs, explicit abstentions, completeness,
  hashes and metadata; never score smoke outputs or the unlabeled test split.
- [ ] Measure load/generation time, host RAM, disk, allocator peaks and device-wide
  VRAM. PyTorch peaks exclude some CUDA/third-party allocations. Confirm headroom
  for longest prompts and throughput within each job's six-hour partition limit.
- [ ] Inspect deterministic-kernel errors/warnings and repeated fresh-process
  smoke outputs. Strict algorithms raise on unsupported kernels; no warn-mode
  fallback/retry is automatic. Third-party kernels may escape PyTorch checks.
- [ ] Test interruption on expendable smoke IDs; archive all five artifacts and
  Slurm stderr. Before-run failures have no artifacts; hard kills can interrupt
  diagnostic writes. No multi-file transaction/power-loss guarantee is claimed.

No GPU compatibility, memory fit, bitwise determinism or performance result is
established by CPU-only tests. The official evaluator and scientific limitations
above remain unchanged.
