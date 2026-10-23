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

Run the smoke test after installation:

```sh
python -m pytest
```

## Current status

Only the project scaffold, minimal CLI, and CLI smoke test are implemented.
**No baseline or experiment has been run.** No ML baseline is implemented, and
no datasets or models have been downloaded. Installation and tests have not been
run during initialization.

Planned experiments include zero-shot, few-shot, and relation-aware prompting;
fine-tuning and retrieval-augmented generation; self-consistency and candidate
aggregation; evidence-grounded factual verification; and ontology-aware semantic
validation. These are research directions, not implemented features.

## Layout

- `src/llm_kbc/`: Python package and CLI.
- `tests/`: CLI smoke test.
- `configs/`: reserved for experiment configuration.
- `data/`: reserved for local datasets; contents are ignored by Git.
- `results/`: reserved for generated results; contents are ignored by Git.

The project uses setuptools with a `src` layout. It has no runtime dependencies;
pytest is an optional development dependency. Dependency versions are not yet
locked; future experiments should record their configurations and environments.
