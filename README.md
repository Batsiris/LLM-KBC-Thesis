# Knowledge Base Construction with Large Language Models

An MSc thesis project investigating how large language models can construct
knowledge bases. The first planned benchmark is **LM-KBC 2026**.

## Installation

Requires Python 3.11 or newer. From the project root, create and activate a
virtual environment:

```sh
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```sh
source .venv/bin/activate
```

Install the package with development dependencies:

```sh
python -m pip install -e ".[dev]"
```

## Usage

Display the CLI help:

```sh
python -m llm_kbc --help
```

Inspect the locally available LM-KBC 2026 splits:

```sh
python -m llm_kbc.dataset_inspector --data-dir data/lm_kbc_2026
```

The directory must contain `train.jsonl`, `val.jsonl`, and `test.jsonl`. The utility
reads UTF-8 JSONL without modifying the files or downloading anything. Each line
must be a JSON object containing `SubjectEntity`, `Relation`, and `ObjectEntities`;
invalid records produce errors with the file path and physical line number.
Blank lines are rejected. Subjects, relations, and aliases must be non-empty
strings. Empty outer object lists are valid; empty inner alias groups are not.

The report includes record and relation counts, the distribution of gold-object
counts, empty object lists, and duplicate subject–relation pairs within each
split. One outer `ObjectEntities` group counts as one object, regardless of how
many aliases it contains. Duplicate matching uses exact strings and reports each
repeated pair's total occurrences plus the number of extra records. Output is
ordered consistently for reproducible inspection.

Train and validation empty lists represent true empty gold answers. **All test
records are unlabeled**: their withheld placeholders are reported separately, and
test gold-object distributions and empty gold-answer counts are unavailable.
The split name determines this policy; empty lists alone do not imply negatives.

Evaluate an existing prediction file using the official LM-KBC 2026 protocol:

```sh
python -m pip install -e ".[dev,evaluation]"
python -m llm_kbc.evaluation --benchmark lm-kbc-2026 --split val --data-dir data/lm_kbc_2026 --predictions results/predictions.jsonl
```

Evaluation accepts explicit `train` or `val` gold selection only; public test
labels are withheld. Predictions use a flat list of strings, unlike nested gold
alias lists. The CLI emits full-precision JSON with overall/per-relation macro
and micro metrics and provenance. See [evaluation protocols](docs/evaluation.md)
for the pinned upstream source, Apache-2.0 license, exact metric definitions,
known edge cases, and adapter architecture. Metrics from different benchmarks
may have different matching and aggregation semantics and must not be directly
compared without qualification.

Run the complete test suite with both `dev` and `evaluation` extras installed:

```sh
python -m pytest
```

## Zero-shot inference in Google Colab

After committing and pushing this implementation yourself, select a **T4 GPU**
runtime in Colab. Run the following in a code cell. Dataset files are ignored by
Git, so this fresh-clone setup fetches train/validation from the same pinned
upstream revision as our evaluator. You may instead supply your existing copies.
These commands are instructions; no model inference has been run during development.

```bash
%%bash
set -e
cd /content
git clone https://github.com/Batsiris/LLM-KBC-Thesis.git
cd LLM-KBC-Thesis
python -m pip install -e ".[inference,evaluation]"
mkdir -p data/lm_kbc_2026
for split in train val; do
  curl --fail --location \
    "https://raw.githubusercontent.com/lm-kbc/dataset2026/e079024b809568f28cded5537203dfd52f5b04ae/data/${split}.jsonl" \
    --output "data/lm_kbc_2026/${split}.jsonl"
done
```

Run five records first, in a new cell:

```bash
%%bash
set -e
cd /content/LLM-KBC-Thesis
python -m llm_kbc.zero_shot \
  --dataset data/lm_kbc_2026/val.jsonl --output results/zero_shot_smoke.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --max-examples 5 --max-new-tokens 512
```

Inspect the smoke predictions and metadata before proceeding. Selection uses the
first records in file order, which currently means five area questions, not a
representative sample of all relations. Do not score this partial file against
full validation: the evaluator treats missing pairs as empty predictions.

Run full validation, then evaluate only if inference succeeds:

```bash
%%bash
set -e
cd /content/LLM-KBC-Thesis
python -m llm_kbc.zero_shot \
  --dataset data/lm_kbc_2026/val.jsonl --output results/zero_shot_val.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --max-new-tokens 512
python -m llm_kbc.evaluation \
  --benchmark lm-kbc-2026 --split val --data-dir data/lm_kbc_2026 \
  --predictions results/zero_shot_val.jsonl > results/zero_shot_val.metrics.json
```

`zero_shot.py` uses the [Qwen chat template](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
with FP16 weights, batch size one, eager attention, greedy decoding
(`do_sample=False`, one beam), a fixed seed, and deterministic PyTorch algorithms.
No quantization, retrieval, examples, gold answers, fine-tuning, or automatic
retries are used. The fixed prompt `lm-kbc-2026-zero-shot-v2` uses short questions
for all six [official relations](https://github.com/lm-kbc/dataset2026#relation-definitions),
with numerical units where applicable. The system prompt sets the reference date
to 1 July 2026. Only subject and
relation enter the prompt; nested gold answers are validated and then ignored.

Each successful run writes one exact-schema prediction record per selected
input, preserving names and order. Duplicate subject-relation input pairs are
rejected before model loading. Complete JSON arrays of strings and finite numbers,
optionally entirely fenced, are accepted. Numeric tokens become strings with
their original spelling and order; answers are not deduplicated. Commentary,
booleans, null, nesting, non-finite numbers, and malformed JSON fail parsing.
An explicit model response containing an empty array is a valid abstention.

Three sidecars accompany the requested output path:

- `<output>.metadata.json`: model ID and requested/resolved Hub commit, prompt
  version/text/hash, chat-template hash, generation configuration, source/input/
  output hashes, selection and counts, library versions, GPU/CUDA details, and status.
  `total_inputs` counts selected records; `valid_records` counts successfully
  parsed responses, separately from `parse_failures` and
  `numeric_normalized_responses`. `format_validity_rate` is valid/processed
  responses (null before any response is processed), so interrupted runs must
  also be checked for coverage using `processed_records` and `total_inputs`.
- `<output>.raw.jsonl`: every original raw response, one-based input line,
  subject/relation, `parsing_status`, and error (null for valid responses).
  Each entry is flushed before generating the next response.
- `<output>.failures.jsonl`: input line, subject/relation, raw response, and error
  for every parse failure; empty when all responses parse successfully.

The explicit methodological policy `parse-failure-as-abstention-v1` is recorded
in metadata as `parse_failure_policy`: each malformed generation produces a scoring
row with `ObjectEntities: []`, while its original response and error remain in both
logs. No answer is recovered from malformed text and no retry occurs. This is our
documented policy, not an official rule for processing raw LLM generations.
The unchanged evaluator gives an empty prediction F1 = 0 for nonempty gold and
F1 = 1 for empty gold. Report parse failures and format validity alongside official
metrics; do not drop failed records or score only the valid subset.

After all selected inputs are generated, coverage, order, unique identities, and
prediction schema are checked before atomic publication. Parse failures alone
allow a complete run and a zero exit code. Runtime errors, OOM, or interruptions
prevent publication; available raw records and diagnostics remain. Large
award-recipient sets may exceed 512 tokens. Existing outputs and sidecars are
never overwritten. Abrupt process termination may leave metadata marked `running`;
only status `complete` denotes a finished run. A smoke run is complete only for
its selected inputs and must not be evaluated against the full validation split.

The default Qwen revision is pinned; other `--model` values default to resolving
`main` unless `--revision` is supplied. Both tokenizer and model use the same
resolved commit. Transformers is pinned in the optional `inference` extra;
Colab's CUDA-enabled PyTorch is reused when compatible, and actual versions are
recorded. [PyTorch does not guarantee identical results across hardware or library
versions](https://docs.pytorch.org/docs/2.9/notes/randomness.html). GPU loading and
performance still require the Colab smoke check; automated tests use fakes and
need neither the `inference` extra nor a GPU. No notebook is required.

## Current status

The project scaffold, dataset inspector, official evaluation adapter, zero-shot
inference pipeline, and synthetic tests are implemented. LM-KBC 2026 dataset files
are available locally. Generation and evaluation are separate commands; no
experiment orchestration framework is implemented. **No model inference, baseline
run, or experiment has been performed during development.** Colab/GPU execution
has not yet been verified.

Planned experiments include running the zero-shot baseline, few-shot and further
relation-aware prompting; fine-tuning and retrieval-augmented generation;
self-consistency and candidate aggregation; evidence-grounded factual verification;
and ontology-aware semantic validation. These remain research directions beyond
the implemented zero-shot pipeline.

## Layout

- `src/llm_kbc/`: inspector, prediction parser, zero-shot inference, and evaluation interface/CLI.
- `src/llm_kbc/_vendor/`: unchanged upstream evaluator with license and provenance.
- `tests/`: smoke, dataset-inspection, and evaluation-protocol tests.
- `docs/`: evaluation usage, provenance, and metric definitions.
- `configs/`: reserved for experiment configuration.
- `data/`: local datasets; contents are ignored by Git.
- `results/`: reserved for generated results; contents are ignored by Git.

The project uses setuptools with a `src` layout and no base runtime dependencies.
Pytest is an optional development dependency; evaluation optionally requires
pandas, and inference optionally requires PyTorch and Transformers. Transitive
dependencies are not locked; future experiments should record their configurations
and environments.
