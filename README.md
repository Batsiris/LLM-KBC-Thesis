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

Run the complete test suite with the `dev`, `evaluation` and `preprocessing` extras installed:

```sh
python -m pip install -e ".[dev,evaluation,preprocessing]"
python -m pytest
```

## Multi-model prompting benchmark

Use `python -m llm_kbc.benchmark` for explicit model/revision-configurable
zero-, 1-, 3-, and 5-shot comparisons. It reuses the committed prompt policies
(v1 for 3-shot, v2 for 1/5-shot), strict parser, audited output writer and official
offline evaluator. Existing pinned Qwen runners and archived results are unchanged.
Help and offline dry-run require no GPU dependencies or network access.
See [multi-model benchmark commands and protocol](docs/model_benchmark.md) for
all four conditions, isolated run paths, reproducibility metadata, offline
reporting, compatibility limits and model-selection/held-out safeguards.
The revised five-model roster retains archived Qwen2.5 and completed Qwen3.5
conditions, replaces Mistral Small 3.2 with **Ministral 3 14B**, retains Gemma4,
and adds **Granite 4.1 8B**. Qwen completion status is researcher-reported; this
change does not rerun or alter those artifacts. Ministral requires explicit
FP8-checkpoint-to-BF16 conversion on A100; Granite uses native BF16 causal-LM
inference. Their [verified revisions, configurations and Slurm commands](docs/model_benchmark.md#reviewed-five-model-roster)
are documented separately, including a 475-record input guard and GPU verification
checklist. Existing inference/training and isolated modern dependency pins are preserved.
No candidate models are selected automatically and no new inference is run.

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

## Few-shot Prompting v1

`llm_kbc.few_shot` selects training demonstrations and constructs prompts; it does
not load weights, generate predictions, or score anything. It reuses the unchanged
zero-shot v2 system prompt and six questions. Each query has three same-relation
user/assistant examples followed by the query: eight chat messages in total.
Assistant examples are complete JSON arrays of strings. This is a separate inference
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
tokenizer, token counts are `None` and `tokenizer_checked` is false. Few-shot
inference always uses the pinned tokenizer and checks the actual chat template,
including generation markers, without truncation. The check reserves the explicit
`max_new_tokens` budget and respects both model and tokenizer limits. No generation
parameter is changed; the preparation default remains 512 new tokens.
The inference command automatically archives this manifest with source provenance.

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

### Few-shot inference in Google Colab

Use the T4 runtime, repository clone, installation, and pinned dataset-download
commands in the zero-shot Colab section above. Run from a **Git checkout**; no
notebook or additional dependency is needed. The inference CLI enforces the
training SHA256 above, so another training snapshot requires a separately reviewed
experiment. It calls the existing zero-shot GPU loader with the fixed Qwen model
and revision, FP16, eager attention, batch size 1, seed 0, deterministic algorithms,
`do_sample=False`, and **512 new tokens**. These settings cannot be overridden by
this CLI. The original zero-shot CLI and prompt remain unchanged.

Start with five records, writing to Colab's local filesystem:

```bash
%%bash
set -e
cd /content/LLM-KBC-Thesis
python -m llm_kbc.few_shot_inference \
  --data-dir data/lm_kbc_2026 --dataset data/lm_kbc_2026/val.jsonl \
  --output /content/few_shot_v1/smoke.jsonl --max-examples 5
```

Before generation, **every input in the supplied file** is checked for schema,
unique normalized identities, demonstration leakage, and actual templated context
length with all 512 output tokens reserved. Even the smoke command checks all
475 validation inputs, then generates only the first five in file order. Those
five area queries do not check GPU generation for every relation; GPU memory and
latency for award prompts still need measurement. Preflight itself is CPU-only,
but real inference requires CUDA. Full validation is a separate command:

```bash
%%bash
set -e
cd /content/LLM-KBC-Thesis
python -m llm_kbc.few_shot_inference \
  --data-dir data/lm_kbc_2026 --dataset data/lm_kbc_2026/val.jsonl \
  --output /content/few_shot_v1/val.jsonl
python -m llm_kbc.evaluation --benchmark lm-kbc-2026 --split val \
  --data-dir data/lm_kbc_2026 --predictions /content/few_shot_v1/val.jsonl
```

Do not evaluate smoke output as a complete run or score withheld test labels.
Inspect `status: complete`, coverage, and format validity before scoring. There
are no retries, resume, or syntactic repairs. The unchanged strict parser preserves
numeric token spelling; malformed generations become explicit scoring abstentions,
which can score perfectly when the official gold answer is empty.

Each prediction file has `.raw.jsonl`, `.failures.jsonl`, `.metadata.json`, and
`.manifest.json` sidecars appended to its name. Raw generations are flushed for
every response; interruption/OOM retains diagnostics but publishes no incomplete
prediction file. Final schema, order, and coverage checks and publication use the
same runner as zero-shot. Existing artifacts are refused, so use fresh output
names. The immutable manifest records all preflighted inputs/messages, exactly
18 selected demonstrations, selection policy/version, training/input/source
hashes, prompt version, source Git commit and dirty flag, pinned/resolved model
revision, tokenizer versions/context/template hash, and actual generation/runtime
settings. Metadata links its SHA256 and records completion and parsing counts.
A manifest alone does not mean generation completed; pre-generation or I/O failure
may leave sidecars. Archive all five files together. Use a clean, committed clone
for reported runs: source hashes and a dirty flag cannot reconstruct uncommitted
changes. Atomic replacement and flushes are not a multi-file transaction or a
power-loss guarantee; prefer local `/content` over a mounted Drive filesystem.

This experiment changes only demonstration conditioning relative to zero-shot;
it is **not** the format-repair ablation. Selection uses training labels only,
but the validation split has already been inspected during development. Report
that limitation, unresolved alias-level leakage/pretraining contamination, the
demonstration length caps, truncation risk for long award answers, format validity,
and hardware/library differences alongside scores. CPU tests use fake generation;
no actual few-shot GPU inference or evaluation has been performed during development.

## Few-shot Prompting v2: 1-shot and 5-shot

`few_shot_v2.py` prepares the two new conditions; `few_shot_v2_inference.py`
reuses the unchanged zero-shot GPU generator, strict numeric-token parser, and
audited artifact runner. **Reuse the completed v1 3-shot results; do not rerun
3-shot.** The new CLI accepts only `--shots {1,5}`. No existing experiment or
prompt implementation is changed, and no format repair is applied.

The fixed `relation-aware-1-or-5-shot-v2` policy uses only the reviewed training
snapshot (SHA256 above). Eligibility, canonical first-alias labels, numeric
validation, and whole-answer caps (32 objects / 1,024 serialized UTF-8 bytes)
are identical to v1. No strings, order, numeric spelling, or units are changed.

- **1-shot:** sort eligible nonempty examples by `(object count, original
  subject)`, or `(finite decimal value, original subject)` for numeric relations.
  Choose rank `(n-1)//2`, the lower median over **records**, not distinct
  cardinalities. If all eligible examples are empty, choose the first original
  subject in Unicode code-point order. This intentionally prioritizes a
  nonempty representative answer over teaching abstention in the single slot.
- **5-shot:** call the unchanged v1 selector for the first three examples,
  preserving their exact order. For nonnumeric relations, rank all eligible
  records (including empties) by `(object count, original subject)`. Append the
  nearest unused ranks to `(n-1)//4` and `3*(n-1)//4`, in that order; ties choose
  the lower rank. For numeric relations, rank nonempty records by `(finite
  decimal value, original subject)` and compute the same targets using their
  count. Prefer unused nonempty records; if exhausted, use empty records ordered
  by original subject after the nonempty ranks. Existing v1 requirements still
  apply, including two nonempty numeric candidates when empties exist.
- Require enough eligible distinct records in every relation; otherwise fail.
  Ties never depend on training file order. Neither selection receives query
  gold, predictions, or metrics, and leakage never triggers replacement.

The actual training snapshot selects the following subjects. The two additions
are in prompt order; each 5-shot prefix is exactly the v1 table above.

| Relation | 1-shot | 5-shot additions (4th; 5th) |
|---|---|---|
| countryLandBordersCountry | Vietnam (3 objects) | Macau (1); Argentina (5) |
| personHasCityOfDeath | Jūrō Kara (1) | Lasse Åberg (0); Liudmila Belavenets (1) |
| hasCapacity | Cowboy Stadium in Louisiana (`12226`) | Luther "Nick" Jeralds Stadium in North Carolina (`5520`); American Legion Memorial Stadium in North Carolina (`21000`) |
| awardWonBy | Fulbright Prize (27) | Fulbright Prize (27); Premier League Player of the Season (27) |
| companyTradesAtStockExchange | Nautilus, Inc. (1) | PACCO (0); Sodexo (1) |
| hasArea | Zakynthos (`405.55`) | Lake Iseo (`65.3`); Kosovo (`10909.02992`) |

Every prompt has the unchanged zero-shot system message, exactly `shots`
same-relation user/assistant turn pairs with complete JSON string arrays, and a
final query (4 or 12 messages). Only input subject/relation fields are used.
Before loading GPU weights, the full input file is checked for unique normalized
identities, demonstration pair leakage, and exact templated token length without
truncation. Cross-relation subject-only overlaps are reported. Matching remains
NFKC/casefold/whitespace based, without alias resolution or contamination checks.

A CPU-only preflight using the cached pinned tokenizer, Transformers 4.57.1,
tokenizers 0.22.2, and Jinja2 3.1.6 checked all 475 local validation inputs
(input-file SHA256 `ba86b53ac38eb4b23b80391b291e5987ff4bbfe79827596fc09751b1bb0ce2be`).
The 1-shot range is **124–305 tokens**, and 5-shot is **227–959**; the largest
prompt plus the fixed 512-token output budget is 1,471, below 32,768.
Neither set has normalized pair or subject-only overlap with these inputs.
These are prompt measurements, not inference or evaluation results.

### Future GPU commands

Use the existing inference environment and run from a Git checkout on local
storage, not a Drive mount. No new dependencies or notebook are required.
Model/revision, FP16, eager attention, batch size 1, deterministic runtime,
greedy generation, and **512 new tokens** are fixed exactly as in zero-shot/v1.
For example, after cloning and installing the existing `[inference]` extra:

```bash
# Smoke runs: preflight all inputs, then generate the first five only.
python -m llm_kbc.few_shot_v2_inference --shots 1 \
  --data-dir data/lm_kbc_2026 --dataset data/lm_kbc_2026/val.jsonl \
  --output experiments/few_shot_v2/1_shot_smoke.jsonl --max-examples 5
python -m llm_kbc.few_shot_v2_inference --shots 5 \
  --data-dir data/lm_kbc_2026 --dataset data/lm_kbc_2026/val.jsonl \
  --output experiments/few_shot_v2/5_shot_smoke.jsonl --max-examples 5

# Separate complete validation runs, after reviewing smoke artifacts.
python -m llm_kbc.few_shot_v2_inference --shots 1 \
  --data-dir data/lm_kbc_2026 --dataset data/lm_kbc_2026/val.jsonl \
  --output experiments/few_shot_v2/1_shot_val.jsonl
python -m llm_kbc.few_shot_v2_inference --shots 5 \
  --data-dir data/lm_kbc_2026 --dataset data/lm_kbc_2026/val.jsonl \
  --output experiments/few_shot_v2/5_shot_val.jsonl
```

Omitting `--output` defaults to `experiments/few_shot_v2/1_shot.jsonl` or
`5_shot.jsonl`. Existing predictions or any of the four sidecars are refused.
Archive predictions plus `.manifest.json`, `.metadata.json`, `.raw.jsonl`, and
`.failures.jsonl` together. The manifest records shot count, selection/prompt
versions, ordered demonstrations and prompts, training/input SHA256, pinned and
resolved model revision, tokenizer versions and chat-template hash, source commit
and hashes, generation settings, and runtime information. Raw/failure logging,
numeric normalization, explicit parse-failure abstentions, completion status,
and final ordered coverage/schema validation use the existing zero-shot runner.
OOM/interruption retains diagnostics but publishes no incomplete predictions.
As before, multi-file transactions, concurrent-writer safety, and power-loss
durability are not guaranteed. Use fresh filenames and archive completed runs.

This is an exploratory demonstration-count study on already inspected validation
inputs. The 1-shot example need not belong to the v1 prefix, and adding two
examples also changes their identities, prompt length, and empty-answer exposure;
this is not an isolated causal estimate of count alone. Reusing all five eligible
awards also increases award prompt length. The unchanged 512-token budget can
truncate long award answers, even though input contexts fit. GPU behavior remains
unverified for v2. Do not score smoke outputs or withheld test labels, select
demonstrations by validation performance, or alter earlier baseline artifacts.

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

## QLoRA supervised fine-tuning v1 (implementation only)

`llm_kbc.qlora_train` reuses training-only preparation, strict chat-template
tokenization, and padding. It reads targets only from `train.jsonl`, preserves
first-alias spelling and file order, and performs no validation/test evaluation.
The full training file is validated before selecting any smoke prefix. System,
user, and assistant-header tokens have label `-100`; assistant JSON, end-of-turn,
and the template's **trailing newline remain supervised**. Padding labels are
`-100`. Python lists become CPU `torch.long` tensors only in the Trainer collator;
Trainer moves batches to the GPU and applies the model's standard causal loss.

The new optional `fine-tuning` extra pins PyTorch 2.8.0, Transformers 4.57.1,
PEFT 0.17.1, Accelerate 1.10.1, and bitsandbytes 0.48.1. Existing extras are unchanged.
The runtime requires those exact versions (CUDA wheel suffixes are recorded and
allowed). Compatibility was reviewed against the versioned
[Transformers dependencies](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/dependency_versions_table.py),
[PEFT requirements](https://github.com/huggingface/peft/blob/v0.17.1/setup.py),
[Accelerate requirements](https://github.com/huggingface/accelerate/blob/v1.10.1/setup.py),
and [bitsandbytes requirements](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.48.1/pyproject.toml).
This is source/API review, **not a tested GPU environment**. No packages were
installed and no training was run while implementing it. Before a future run,
prepare a separate environment with the extra and a suitable CUDA-enabled PyTorch
wheel. [PyTorch 2.8.0 provides CUDA 12.6/12.8/12.9 wheels](https://pytorch.org/get-started/previous-versions/#v280);
check the HPC driver and the matching bitsandbytes native library first. Transitive
packages are not fully locked; all installed package versions are recorded.

The fixed base is `Qwen/Qwen2.5-3B-Instruct` at revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, with NF4 4-bit weights, double
quantization, BF16 compute, and eager attention. Exactly one visible NVIDIA CUDA
GPU/process with native BF16 (Ampere or newer) is required; there is no CPU/FP16,
multi-GPU, or automatic batch-size fallback. The initial target is an A100 40GB.

| Initial configuration | Value |
|---|---|
| LoRA | rank 16, alpha 32, dropout 0.05, bias none |
| Target projections | q/k/v/o and gate/up/down |
| Learning rate / epochs | `2e-4` / 3 |
| Micro batch / accumulation | 1 / 16; nominal effective batch 16 |
| Sequence limit | 2,048; reject overflow, never truncate |
| Gradient checkpointing | enabled; `use_reentrant=False`; model cache disabled |
| Optimizer | `adamw_torch`; betas 0.9/0.999, epsilon `1e-8`, weight decay 0 |
| Schedule / clipping | linear, warmup ratio 0.03, gradient norm 1.0 |
| Ordering / seed | sequential file order each epoch, workers 0, seed 42 |

These values are **initial experimental choices, not scientifically optimized**.
An incomplete final accumulation window is retained and has a smaller effective
batch. Sequential ordering also preserves the dataset's relation grouping; its
effect is a methodological limitation. Strict PyTorch determinism and disabled
TF32 do not guarantee bitwise reproducibility across GPUs, libraries, or custom
bitsandbytes kernels. Memory requirements and native-kernel execution remain
unverified. The separate identity-overlap audit is available in
`fine_tuning_data.audit_validation_overlap()`; training does not read validation
labels or select checkpoints using validation performance.

Future HPC invocation, **not executed during implementation** (set the three
environment paths to your allocated scratch directories; no username is built in):

```sh
cd ~/LLM-KBC-Thesis
python -m llm_kbc.qlora_train --data-dir data/lm_kbc_2026 \
  --output-dir results/qlora_v1_smoke --smoke \
  --cache-dir "${HF_CACHE:?Set the shared model cache}" \
  --checkpoint-dir "${CHECKPOINT_ROOT:?Set checkpoint root}/qlora_v1_smoke" \
  --tmp-dir "${TMP_ROOT:?Set temporary-file root}"
```

`--smoke` explicitly uses the first eight training records, one epoch, accumulation
4 (nominal effective batch 4), and a two-optimizer-step limit. Other hyperparameters
remain as above. This area-only prefix is not representative of every relation.
For a later full run, omit `--smoke` and use new output/checkpoint names.
`--config path.json` accepts explicit `TrainingConfig` overrides; unknown/invalid
fields fail, and combining it with `--smoke` is rejected. A positive `max_steps`
explicitly overrides the epoch-based stopping condition. Model/revision,
quantization, BF16, attention, and projection targets are fixed for this version.

Output and checkpoint directories must be new, even if an existing directory is
empty. There is no automatic resume or overwrite. Checkpoints are saved each epoch
and retained; no checkpoint retention/deletion policy is applied. Final LoRA
weights/config are saved in `adapter.incomplete` and renamed to `adapter` only
after both files are verified. Base weights are never merged, saved, or modified.
`metadata.json` records status, source Git commit/dirty flag and hashes, training
dataset/feature hashes, selected identities, tokenizer template hash, all package
versions, GPU/runtime details, quantization/LoRA/training settings, and training
loss/steps/log history. These are training results, not benchmark evaluation.
Failures retain diagnostics, checkpoints, and possibly an incomplete adapter.
Metadata updates and final adapter rename are not one filesystem transaction;
only `status: complete` establishes success. SIGKILL can leave `status: running`,
even with a saved adapter. Use a clean committed checkout for reported experiments,
retain the run artifacts and environment, and keep scratch paths separate from
datasets, source, and experiment outputs. Cache/tmp roots may already exist and
are created as needed only during an actual run; temporary files are not deleted
by this command. Nothing in this implementation launches a job automatically.

## QLoRA adapter inference (implementation only)

`llm_kbc.qlora_inference` loads the original pinned Qwen base on CUDA in FP16,
attaches a local PEFT LoRA adapter, and leaves it **unmerged**. It uses the unchanged
zero-shot v2 system prompt, questions, base tokenizer, and chat template. There
are no few-shot demonstrations or evaluation labels in prompts. Dataset inputs
contribute only `SubjectEntity` and `Relation`; gold fields can be absent or nested
and are not accessed. Generation matches zero-shot's seed 0, greedy decoding,
eager attention, batch size 1, deterministic PyTorch flags, and default 512-token
output budget. Model/revision are fixed; token/example limits are explicit CLI
options. Context overflow fails instead of truncating inputs.

Use the existing `fine-tuning` environment, which includes PEFT 0.17.1; this command
does not use bitsandbytes quantization or require BF16 hardware. Both base and
adapter tensors use FP16, with PEFT's automatic FP32 adapter promotion disabled.
The original QLoRA training used an NF4/BF16 base; evaluating the adapter on the
FP16 base is an explicit experimental condition, not numerically identical to
training-time quantization. GPU execution and memory use have not been verified
locally; no inference or evaluation was run while implementing this command.

Point `--adapter-dir` at the completed training run's `adapter` directory. Require
`adapter_config.json` and `adapter_model.safetensors`; base identifier, exact
revision, LoRA type, and causal-LM task must agree. Missing adapter tensors or
shape mismatches fail, and files are hashed before/after model loading to detect
ordinary concurrent changes. Compatibility checks do not certify training
completion or artifact authenticity: retain/check the training run's completed
metadata and archived hashes too. Adapter files are read-only inputs; predictions
and the shared model cache must be outside that directory.

Future smoke command (not executed):

```sh
python -m llm_kbc.qlora_inference \
  --dataset data/lm_kbc_2026/val.jsonl --output results/qlora_val_smoke.jsonl \
  --adapter-dir /path/to/completed/run/adapter --cache-dir /path/to/huggingface \
  --max-examples 5 --max-new-tokens 512
```

For full validation, omit `--max-examples` and choose a new output path. Do not
score a five-record smoke file as full validation or evaluate withheld test labels.
Predictions keep the existing flat-string schema. The shared zero-shot runner
preserves numeric token spelling, converts malformed outputs to explicit `[]`
abstentions, flushes every raw response, and retains failure diagnostics. It
publishes only complete, validated, ordered predictions. Existing predictions
and `.raw.jsonl`, `.failures.jsonl`, or `.metadata.json` sidecars are refused.
Metadata records base revision, adapter path/config and weight/config SHA256s,
dataset/output hashes, generation settings, prompt version/hash, source hashes,
chat-template hash, package versions, and GPU details. Archive all four artifacts
with the adapter/training provenance. There are no retries, repairs, scoring,
merging, or resume operations. Cross-hardware bitwise identity and multi-file
power-loss atomicity are not guaranteed, as with the zero-shot pipeline.

## RAVEC-00: offline candidate and error audit

This CPU-only development analysis compares immutable, complete 5-shot and QLoRA
validation predictions. It does not run inference, retrieval, training, repairs,
or a controller. Gold labels are used **only for explicit offline scoring/error
analysis**, never for constructing candidates or runtime controller features.
The validation split has already been inspected; this is not held-out testing.

Use the existing `evaluation` dependency group. On Aristotle, activate the
project's existing Python environment and run the following Bash command from
`~/LLM-KBC-Thesis`. Set the three environment variables to the real archived
prediction paths and a **new** output path in an existing local/scratch directory:
the archive locations are not available in this checkout, so they are not guessed.

```sh
cd ~/LLM-KBC-Thesis
: "${FIVE_SHOT:?Set to the archived complete 5-shot prediction JSONL}"
: "${QLORA:?Set to the archived complete QLoRA prediction JSONL}"
: "${AUDIT_OUT:?Set to a new audit JSON path in an existing output directory}"
python -m llm_kbc.candidate_audit \
  --gold data/lm_kbc_2026/val.jsonl \
  --five-shot "$FIVE_SHOT" --qlora "$QLORA" \
  --five-shot-raw "${FIVE_SHOT}.raw.jsonl" \
  --five-shot-failures "${FIVE_SHOT}.failures.jsonl" \
  --five-shot-metadata "${FIVE_SHOT}.metadata.json" \
  --qlora-raw "${QLORA}.raw.jsonl" \
  --qlora-failures "${QLORA}.failures.jsonl" \
  --qlora-metadata "${QLORA}.metadata.json" \
  --null-iterations 1000 --seed 0 \
  --expected-queries 475 --expected-raw-additions 381 --expected-tp-gain 77 \
  --output "$AUDIT_OUT"
```

Sidecars are optional **explicit inputs**, not automatically discovered. If one
is unavailable, omit its option: the report says `missing`, rather than assuming
zero failures. If an explicitly supplied file is missing/inconsistent, the audit
fails. Add `--union /absolute/path/to/archived_union.jsonl` to compare the archived
union's object order and official scores with the deterministic constructed union.
The reference figures above are researcher-reported targets, **not locally
reproduced findings**. A mismatch is saved in the report and returns exit code 1;
invalid inputs return 2 without publishing output. Success returns 0.

One full-precision JSON report contains hashes/byte sizes/paths for every supplied
artifact, scorer provenance, code hash, Python/pandas versions, query-level
candidate attribution, relation-level counts, macro/micro metrics, diagnostics,
null controls, and a concise Markdown `summary` also printed to stdout. No timestamp
or random global state is used. Identical files, paths, configuration and software
produce identical output. The output parent must exist. A same-directory temporary
file and exclusive hard-link publication refuse overwrites; use a filesystem that
supports hard links, rather than Google Drive FUSE. Existing artifacts are read
only and hashed again before publication to detect ordinary concurrent changes.
Hashes establish identity/integrity, not authenticity; compare them against the
independently retained archives. Missing metadata hashes remain unverified.

Accounting policy (`ravec-00-candidate-audit-v1`):

- Gold must be labeled `val.jsonl`; public test paths/links are rejected. Both
  prediction files must have the exact gold identities in file order, with no
  duplicate subject–relation rows. Identity keys are exact, as in the official
  evaluator; object normalization is not applied to query identities.
- The union keeps 5-shot strings first, then unseen exact QLoRA strings in their
  original order. Raw additions count **distinct exact strings per query** absent
  from 5-shot. Occurrence counts, exact duplicates and normalized duplicates are
  reported separately. Normalized additions use upstream `normalize_string`,
  not numeric equivalence or gold alias groups. No original prediction is changed.
- All scoring, deduplication, alias matching, numeric 5% tolerance, empty-answer
  conventions and aggregation call the unchanged official evaluator. Reports
  retain its numeric zero/negative-gold and normalization edge cases described in
  [evaluation documentation](docs/evaluation.md). Metrics are official row-macro
  and count-micro, including per-relation metrics.
- Each added raw string is scored alone for **individual matchability** and in
  the growing prefix for **marginal TP credit**. Prefix credit depends on the
  fixed source/order; it is not a unique gold assignment or factual certification.
  Aliases can each match individually but collectively receive one TP. Therefore
  net TP gain divided by raw additions is **not candidate precision**. FP/FN counts
  are official unmatched prediction/gold counts, not adjudicated factual errors.
- Available raw logs are reparsed with the existing strict parser, including
  numeric-token spelling, and checked against predictions and failure logs.
  Partial raw logs are flagged, never treated as complete. Metadata output/dataset
  hashes and available coverage/count fields are checked. Empty arrays alone
  cannot distinguish a deliberate empty answer from parse-failure abstention.
  An unclosed failed array is only a possible incomplete-JSON signal: current
  raw logs lack generated-token counts/finish reasons, so budget exhaustion cannot
  be confirmed. Missing manifests/tokenizer files are not implicitly read.
- Numeric nulls shuffle whole gold alias groups across queries, separately
  **within each numeric relation** and **within relation plus magnitude decade**
  (`floor(log10(first gold alias))`). Pools, candidate order and official scoring
  stay fixed. Only positive finite singleton gold is eligible; excluded zero,
  negative, nonfinite, unparseable, empty or multi-object gold remains in the real
  evaluation. The same eligible subset supplies observed and shuffled TP counts.
  Seeded permutations retain fixed points; report fixed-point rates, bucket sizes
  and movable queries. Small/singleton bins can make a control uninformative.
  The report includes null means, empirical 2.5/97.5 percentile bounds and a
  plus-one upper-tail fraction for TP gain. These are descriptive diagnostics,
  **not calibrated significance tests** or proof that correct numbers were known.

Interpret relation-level headroom before choosing verification versus discovery:
unmatched additions and FP growth motivate verification; nontrivial incremental
coverage surviving numeric controls motivates testing targeted discovery. These
are proposed follow-up experiments, not evidence of RAVEC novelty or superiority.
No archived 475-record audit has been executed in this local checkout.

## Offline BM25 retrieval

`python -m llm_kbc.bm25_retrieval --help` describes the CPU-only retriever over one
validated Wikipedia passage snapshot. It produces deterministic ranked candidates
and a provenance manifest using subject/relation identities only. See
[BM25 retrieval](docs/bm25_retrieval.md) for the fixed scoring/tokenization policy,
offline pilot command, artifact schema and temporal limitations. Context packing
and RAG inference are separate future tasks; existing baselines are unchanged.

Wikipedia article candidates can be prepared separately with
`python -m llm_kbc.wikipedia_discovery --help`. See
[article discovery](docs/wikipedia_discovery.md) for the versioned direct/search
policy, metadata-only cache replay and historical limitations. The researcher
reports the completed v1 18-training-identity discovery pilot and offline replay;
its archived artifacts remain unchanged.

Dataset-agnostic discovery is opt-in with `--policy-version v2`. Generic JSONL
accepts `subject`, `relation_id`, optional `relation_namespace` and
`relation_description`; `--input-format lm-kbc-2026` explicitly projects the
legacy identity fields into the benchmark namespace. V2 defaults to five eligible
unique articles from ten search results, with configurable integer limits
`1 <= candidate_limit <= search_limit <= 10`. The CLI still defaults to v1's
three-candidate policy. For example, using a **previously populated** metadata
cache and a new output path:

```sh
python -m llm_kbc.wikipedia_discovery \
  --policy-version v2 --input-format lm-kbc-2026 \
  --identities /path/to/pilot/identities.jsonl \
  --output /path/to/new/discovery_5.jsonl --cache-dir /path/to/pilot/cache \
  --candidate-limit 5 --search-limit 10
```

This is offline by default. Selection-limit changes reuse complete cached API
responses; query/search-parameter changes can cause controlled cache misses.
V2 records full query provenance and rejects unsupported control/format characters
without altering text; MediaWiki search expressions are submitted verbatim, not
made literal by URL encoding. See the discovery documentation for generic
examples, backend/duplicate rules, artifact-version validation and exact proposed
Aristotle expansion commands. These are article candidates, not final evidence
contexts; no RAG inference or new live pilot is implemented by this stage.

## Current status

The project scaffold, dataset inspector, official evaluation adapter, zero-shot
and few-shot inference pipelines, and synthetic tests are implemented. LM-KBC 2026
dataset files are available locally. Generation and evaluation are separate
commands; no experiment orchestration framework is implemented.

**Zero-shot v2, Few-shot v1 (3-shot), and QLoRA have completed validation
evaluation, as reported by the researcher; Format Repair v1 is also complete.**
The existing external artifacts remain unchanged. Few-shot v2 (1-shot/5-shot)
has only CPU tokenizer preflight and synthetic generation verification; no v2
GPU inference or evaluation has been performed.

Planned experiments include the new few-shot counts and further
relation-aware prompting; additional fine-tuning and retrieval-augmented generation;
self-consistency and candidate aggregation; evidence-grounded factual verification;
and ontology-aware semantic validation. These remain research directions beyond
the implemented inference pipelines.

## Layout

- `src/llm_kbc/`: inspector, prediction parser, zero-shot inference, and evaluation interface/CLI.
- `src/llm_kbc/few_shot.py`: training-only demonstration selection and prompt preparation.
- `src/llm_kbc/few_shot_inference.py`: fixed few-shot inference CLI and reproducibility manifest.
- `src/llm_kbc/few_shot_v2.py`, `few_shot_v2_inference.py`: separate 1/5-shot preparation and inference; v1 remains unchanged.
- `src/llm_kbc/_vendor/`: unchanged upstream evaluator with license and provenance.
- `tests/`: CLI, schema, evaluation, inference, format-repair, and prompt-selection tests.
- `docs/`: evaluation usage, provenance, and metric definitions.
- `configs/`: reserved for experiment configuration.
- `data/`: local datasets; contents are ignored by Git.
- `results/`: reserved for generated results; contents are ignored by Git.
- `experiments/few_shot_v2/`: future 1/5-shot artifacts; ignored by Git.

The project uses setuptools with a `src` layout and no base runtime dependencies.
Pytest is an optional development dependency; evaluation optionally requires
pandas, and inference optionally requires PyTorch and Transformers. Transitive
dependencies are not locked; future experiments should record their configurations
and environments.
