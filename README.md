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

## Current status

The project scaffold, minimal CLI, read-only dataset inspector, official LM-KBC
2026 evaluation adapter, and synthetic tests are implemented. LM-KBC 2026 dataset
files are available locally. Evaluation consumes supplied predictions and does
not generate answers or orchestrate experiments. The inspector uses no model,
external API, or additional runtime dependency.
**No baseline or experiment has been run.** No ML baseline is implemented.

Planned experiments include zero-shot, few-shot, and relation-aware prompting;
fine-tuning and retrieval-augmented generation; self-consistency and candidate
aggregation; evidence-grounded factual verification; and ontology-aware semantic
validation. These are research directions, not implemented features.

## Layout

- `src/llm_kbc/`: package, inspector, prediction parser, and evaluation interface/CLI.
- `src/llm_kbc/_vendor/`: unchanged upstream evaluator with license and provenance.
- `tests/`: smoke, dataset-inspection, and evaluation-protocol tests.
- `docs/`: evaluation usage, provenance, and metric definitions.
- `configs/`: reserved for experiment configuration.
- `data/`: local datasets; contents are ignored by Git.
- `results/`: reserved for generated results; contents are ignored by Git.

The project uses setuptools with a `src` layout and no base runtime dependencies.
Pytest is an optional development dependency; evaluation optionally requires
pandas at a pinned version. Transitive dependencies are not locked; future
experiments should record their configurations and environments.
