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

## Few-shot Prompting v1: preparation only

`llm_kbc.few_shot` selects training demonstrations and constructs prompts; it does
not load weights, generate predictions, or score anything. It reuses the unchanged
zero-shot v2 system prompt and six questions. Each query has three same-relation
user/assistant examples followed by the query: eight chat messages in total.
Assistant examples are complete JSON arrays of strings. This is a future inference
condition, distinct from format repair, which post-processes existing generations.
No repair rules are applied to few-shot prompts or demonstration answers.

The fixed policy `relation-aware-3-shot-v1` reads **only `train.jsonl`**:

- Represent each gold object by the first string in its alias group, preserving
  spelling and group order. This is our representative-label convention; the
  schema has alias groups rather than a separate canonical-label field. Never
  concatenate aliases, convert numeric units, or truncate an answer set.
- Exclude whole examples exceeding 32 objects or 1,024 UTF-8 bytes of serialized
  representative-label JSON. These relation-independent caps keep award examples
  short. Fail if fewer than three eligible examples remain for any relation.
- For nonnumeric relations with empty and nonempty examples, target cardinalities
  zero, smallest positive, and largest eligible. Without empty examples, use the
  smallest, lower-median **distinct** cardinality, and largest. If there is only
  one cardinality, use it three times. When a bucket is exhausted, choose the
  nearest remaining cardinality, preferring the larger count on a distance tie.
  All remaining ties use original subject strings in Unicode code-point order.
- For numeric relations, rank singleton answers by exact finite decimal value
  (ignoring grouping commas for ranking only), then original subject. Choose
  positions 0, `(n-1)//2`, and `n-1`. If numeric empty examples exist, use the
  lexicographically first empty example and the minimum/maximum nonempty examples;
  require at least two nonempty candidates. No empty examples are invented.
- Keep these selection-slot orders in the prompt. There is no random seed,
  query-adaptive selection, validation ranking, or replacement on leakage.

The local training snapshot has SHA-256
`ad37cd30d1ff4b9f1ef2579b25e64093b202c40da11e8c412e13386f1e5d332e`.
Its selected subjects and answer cardinalities are:

| Relation | Ordered demonstrations (object count) |
|---|---|
| countryLandBordersCountry | Cuba (0); Canada (1); Germany (9) |
| personHasCityOfDeath | Adel Emam (0); Alessandra Melucco Vaccaro (1); Angelo Rinaldi (1) |
| hasCapacity | Martin Family Stadium in Virginia (1); Cowboy Stadium in Louisiana (1); Allegiant Stadium in Clark County (1) |
| awardWonBy | UEFA Team of the Year (9); FAI Gold Air Medal (19); Order of the Golden Fleece (Georgia) (29) |
| companyTradesAtStockExchange | Ablex Publishing (0); ANY Security Printing Company (1); ING Group (3) |
| hasArea | Poveglia (1); Zakynthos (1); Angola (1) |

Training links to validation/test files, duplicate identities, malformed labels,
and unsupported relations fail. File naming is not proof of provenance: use the
official training snapshot and retain its hash. Evaluation files contribute only
`SubjectEntity` and `Relation`; their gold-answer fields are never accessed.
All 18 demonstrations are checked against the entire input set before preparation
succeeds. Subject matching uses NFKC, casefolding, and collapsed whitespace.
Same-relation overlap fails; cross-relation subject overlap is reported without
reselection. The selected 18 have no such overlaps with local validation inputs,
although the full train/validation files share 11 exact subject names across
different relations. These checks do not resolve entity aliases, reciprocal facts,
object-to-subject connections, or pretraining contamination.

```python
from pathlib import Path
from transformers import AutoTokenizer
from llm_kbc import few_shot, zero_shot

tokenizer = AutoTokenizer.from_pretrained(
    zero_shot.DEFAULT_MODEL, revision=zero_shot.DEFAULT_REVISION,
    trust_remote_code=False,
)
manifest = few_shot.prepare_prompts(
    Path("data/lm_kbc_2026"), Path("data/lm_kbc_2026/val.jsonl"),
    tokenizer=tokenizer,
)
```

This optional CPU tokenizer preflight needs Transformers and Jinja2, available in
the existing inference environment; it downloads no model weights. The returned
manifest contains full demonstrations/messages, versions, input/source hashes,
tokenizer/template details, content lengths, and token counts. Without a supplied
tokenizer, token counts are `None` and `tokenizer_checked` is false. Before any
future inference, use the pinned tokenizer and check the actual chat template,
including generation markers, without truncation. The check reserves the explicit
`max_new_tokens` budget and respects both model and tokenizer limits. No generation
parameter is changed; the preparation default remains 512 new tokens.
Archive the manifest together with the repository commit before future inference.

A CPU preflight of all 475 local validation questions with Transformers 4.57.1,
tokenizers 0.22.2, and the pinned Qwen revision measured 175–573 input tokens.
The maximum plus 512 reserved output tokens is 1,085, within the pinned model's
[32,768-position configuration](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/config.json).
The tokenizer advertises 131,072 tokens; that larger value does not override the
model configuration, and the preflight uses the smaller limit.
This is a prompt-length measurement, not inference or model evaluation.

**512 output tokens are not sufficient for every long award answer.** Using only
training labels, complete representative-label arrays for the two Nobel awards
require 1,545 and 1,559 tokenizer tokens even before an end-of-turn token. Five of
the ten training awards are eligible under the demonstration caps. Short examples
do not establish that validation award answers will fit. Preserve the 512-token
condition for comparability; any larger-budget experiment must be explicit and
reported separately. No validation/test labels were used for this assessment.

## Exploratory format-repair ablation

This independent post-processing ablation uses saved generations from an
**already inspected validation split**. It is exploratory, not an unbiased official
benchmark submission. It performs no inference, factual correction, retrieval,
gold-label lookup, or metric-based selection. The strict baseline remains intact.

```sh
python -m llm_kbc.format_repair --raw-generations results/zero_shot_val.jsonl.raw.jsonl --baseline-predictions results/zero_shot_val.jsonl --output results/zero_shot_val_repaired.jsonl
python -m llm_kbc.evaluation --benchmark lm-kbc-2026 --split val --data-dir data/lm_kbc_2026 --predictions results/zero_shot_val_repaired.jsonl
```

Policy `lm-kbc-format-repair-v1` applies only to strict parse failures:

- A standalone JSON number such as `1.2300` becomes `["1.2300"]`. JSON number
  spelling is preserved without floating-point conversion; NaN, infinity,
  grouping commas, decimal commas, Unicode numeric punctuation, and stray
  characters (including `24522⟩`) are rejected.
- A bracketed list of at least two unquoted names separated by exactly `, `,
  such as `[France, Germany]`, becomes a JSON string array. Each space-separated
  word must start with an uppercase letter and otherwise contain letters or
  internal apostrophes, hyphens, or periods. A nonfinal alphabetic word may end
  in a period, allowing `St. Louis`. Accented letters and internal `’` are allowed.
  Digits, reserved JSON/non-finite words, mixed delimiters, quotes, nesting,
  commentary, truncation, and other characters are rejected. This fixed comma
  interpretation still has semantic ambiguity: `[Paris, Texas]` becomes two
  strings even though it could denote one place, and `[Acme, Incorporated]` can
  split one company name. The grammar cannot resolve this without interpreting
  contents. Lowercase names/connectors and many legitimate punctuation patterns
  are deliberately ineligible too; semicolon-separated lists are not repaired.

Valid responses are checked with the existing strict parser and must agree with
the baseline prediction; their original prediction lines remain byte-for-byte
unchanged. In particular, valid numeric arrays such as `[1,234]` keep the baseline's
two objects; this ablation never reinterprets them as a grouped number. Ineligible
or unsuccessful repairs retain the original empty-list scoring abstention.

Inputs must have equal nonzero lengths, matching ordered unique subject-relation
pairs, contiguous one-based raw indices, and statuses/errors consistent with the
strict parser. Failed baseline predictions must be empty lists. Coverage is
verified **against the supplied baseline**, not an independent dataset manifest;
jointly truncated inputs cannot be detected. Supply the complete validation run,
not smoke outputs. Parsing must agree with the current strict-baseline code.

The separate prediction file is accompanied by `<output>.audit.jsonl` (every
failed response inspected, raw text, original error, rule, success/rejection, and
resulting prediction) and `<output>.metadata.json` (policy/source/parser hashes,
input/output hashes, overall and per-relation counts). An attempt means one strict
parse failure inspected by the rule set, including an ineligible response.
Report attempts, successes, remaining failures, and both strict/repaired official
scores; empty abstentions can score perfectly when gold is empty.

All validation finishes before writing artifacts. Files are never overwritten;
closed temporary files are published using atomic hard links on NTFS/Linux.
Unsupported filesystems fail safely. Predictions are published last; an I/O
failure may leave complete audit/report sidecars without a prediction file.
The name grammar is a syntactic restriction, not proof that a string is an entity
or that its answers are factually correct. These rules are fixed independently
of relations and gold answers, but inspecting validation outputs already limits
the conclusions that can be drawn about generalization.

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
- `src/llm_kbc/few_shot.py`: training-only demonstration selection and prompt preparation.
- `src/llm_kbc/_vendor/`: unchanged upstream evaluator with license and provenance.
- `tests/`: CLI, schema, evaluation, inference, format-repair, and prompt-selection tests.
- `docs/`: evaluation usage, provenance, and metric definitions.
- `configs/`: reserved for experiment configuration.
- `data/`: local datasets; contents are ignored by Git.
- `results/`: reserved for generated results; contents are ignored by Git.

The project uses setuptools with a `src` layout and no base runtime dependencies.
Pytest is an optional development dependency; evaluation optionally requires
pandas, and inference optionally requires PyTorch and Transformers. Transitive
dependencies are not locked; future experiments should record their configurations
and environments.
