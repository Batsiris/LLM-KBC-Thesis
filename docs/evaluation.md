# Evaluation protocols

Evaluation consumes existing prediction files. It does not run a model, generate
answers, retrieve evidence, or orchestrate experiments. Metrics from different
benchmarks may use different matching and aggregation semantics and **must not be
directly compared without qualification**, even when they share names such as F1.

## Architecture

- `llm_kbc.evaluation.Evaluator` is a small structural protocol. An adapter selects
  and parses its benchmark's gold data and returns an `EvaluationResult` containing
  the benchmark, split, overall/per-relation metrics, and provenance metadata.
- `LMKBC2026Evaluator` is the only concrete adapter currently implemented. There
  is no plugin framework or implementation for hypothetical future datasets.
- `dataset_inspector.load_jsonl` validates LM-KBC gold records. The separate
  `prediction_io.load_predictions` validates LM-KBC prediction records. These
  formats differ and are not interchangeable.
- The adapter performs input checks and delegates all matching, F1 calculation,
  and metric aggregation to the unmodified upstream evaluator. JSON serialization
  is confined to the evaluation CLI. Experiment orchestration remains future work.

## Installation and usage

From the project root, with the virtual environment activated:

```sh
python -m pip install -e ".[dev,evaluation]"
python -m llm_kbc.evaluation --help
python -m pytest
```

The optional `evaluation` dependency is `pandas==2.2.3`. Although our adapter does
not use DataFrames, the unchanged upstream module imports pandas at module scope.
The base package and dataset inspector do not require pandas. The complete test
suite requires both the `dev` and `evaluation` extras.

To evaluate an existing validation prediction file:

```sh
python -m llm_kbc.evaluation --benchmark lm-kbc-2026 --split val --data-dir data/lm_kbc_2026 --predictions results/predictions.jsonl
```

Use `--split train` for a train diagnostic. `--benchmark`, `--split`, `--data-dir`,
and `--predictions` are required. Gold selection is always `<data-dir>/<split>.jsonl`;
there is no arbitrary `--gold` option. The CLI emits JSON to stdout, retaining
Python float precision without upstream's three-decimal display rounding. To
save it, redirect stdout to a separate results file using your shell. The CLI
itself never writes files.

**Public test labels are withheld.** The CLI and adapter reject `test` and other
split names, and detect train/val links to the corresponding public `test.jsonl`.
They cannot determine that somebody deliberately copied and renamed an unlabeled
file. Callers must supply authentic labeled train/validation files. Empty labels
alone cannot identify test data because genuine train/validation answers can be
empty. No public test evaluation is part of this task.

## Official evaluator provenance

Upstream repository: <https://github.com/lm-kbc/dataset2026>

Pinned commit: `e079024b809568f28cded5537203dfd52f5b04ae`

Exact source:
<https://github.com/lm-kbc/dataset2026/blob/e079024b809568f28cded5537203dfd52f5b04ae/evaluate.py>

Applicable upstream license: **Apache License 2.0**. The license was established
from the repository's `LICENSE` at the same commit before copying the code. There
was no root `NOTICE` file at that revision. The complete upstream license is
distributed alongside the evaluator in `src/llm_kbc/_vendor/lm_kbc_2026/LICENSE`.
This third-party license does not assert a license for the rest of this project.

The evaluator is preserved byte-for-byte. Its SHA-256 is:

```text
2d592ae177c7b230922bb959da7a8ee1c4c662bf72a99d4dbd0cf62170ff9e22
```

`provenance.json` records source/raw URLs, commit, Git blob hashes, and SHA-256
checksums for both `evaluate.py` and `LICENSE`. Package data configuration includes
the license and provenance in distributions. `.gitattributes` disables newline
conversion for the upstream files, and an automated test verifies their hashes.
Upgrades should explicitly update the pinned files, provenance, documentation,
and protocol tests together; runtime evaluation never fetches upstream code.

Every result includes this provenance, the fixed tolerance, Python/pandas
versions, input paths and SHA-256 hashes, input row counts, and the number of
evaluated pairs. Input hashes identify local data revisions independently of the
evaluator revision. No synthetic test score is an experimental model result.

## Input contract

Both inputs are UTF-8 JSONL, one object per physical line. Required fields are
`SubjectEntity`, `Relation`, and `ObjectEntities`. Subject and relation must be
non-empty strings; pair matching is exact, without subject normalization.
Additional metadata is accepted.

Gold `ObjectEntities` is a list of non-empty alias lists; every alias must be a
non-empty string. An empty outer list is a real empty gold answer. The number of
outer groups is the gold count, not the number of strings across all groups.

Prediction `ObjectEntities` must be a **flat list of strings**:

```json
{"SubjectEntity":"MindMed","Relation":"companyTradesAtStockExchange","ObjectEntities":["Cboe Canada","Nasdaq"]}
```

Numeric predictions are also strings. Empty lists and empty prediction files are
valid. Prediction strings are preserved, including empty strings, whitespace,
duplicates, and unparseable numeric strings, because upstream determines their
effect on scoring. No adapter stripping, deduplication, sorting, unit conversion,
or repair occurs. Malformed JSON, missing fields, nested prediction lists, and
nonstring entries fail with file/line context rather than being silently fixed.

This schema validation is deliberately stricter than upstream, which tolerates
some nested predictions and legacy flat gold. We reject those malformed or
legacy shapes rather than changing how accepted records score. An entirely empty
gold file is rejected because upstream macro aggregation divides by zero; a
nonempty gold file whose records all have empty answers remains valid.

## Matching and metrics

These definitions describe the pinned upstream code, not a new scoring design.

### String relations

`awardWonBy`, `countryLandBordersCountry`, `personHasCityOfDeath`, and
`companyTradesAtStockExchange` use string matching. Upstream also defaults unknown
relation names to string matching; the adapter preserves that behavior.

Normalization strips surrounding whitespace and apostrophe-like marks, applies
Unicode NFKD decomposition followed by case-folding, removes combining marks,
turns Unicode punctuation and `+$<=>|~^` into spaces, then collapses whitespace.
For example, `O’Brien` matches `OBrien`, `Kauaʻi` matches `Kauai`, and `Straße`
matches `STRASSE`. It does not resolve arbitrary semantic synonyms.

Predictions are deduplicated by normalized form. Maximum-cardinality bipartite
matching then matches predictions to the normalized alias sets of gold groups.
Each prediction and gold group can contribute at most one true positive (TP).
This handles overlapping alias sets without depending on prediction order.
Different normalized aliases of the same gold entity remain different predictions
but earn at most one TP, lowering precision if both are submitted.

### Numeric relations and known upstream edge cases

`hasArea` and `hasCapacity` use `float(value.replace(',', '').strip())` and the
first string of each gold alias group. A positive gold value `g` matches a numeric
prediction `p` when `abs(p - g) / g <= 0.05`. The boundary is inclusive using
ordinary floating-point arithmetic, without an added epsilon.

- Parsing accepts scientific notation and comma-separated values; there is no
  extraction from prose, unit conversion, or locale-aware decimal-comma parsing.
- Zero gold values cannot match, even exactly. The signed denominator means
  negative gold can match inappropriate values. These upstream behaviors are
  preserved, not corrected.
- Invalid numeric predictions earn no TP but still count as predictions. Invalid
  gold values remain in the gold denominator. No explicit finite-number check
  exists; NaN/infinity follow Python float/comparison behavior.
- Numeric matching greedily consumes the first eligible unmatched gold value.
  With multiple gold groups, prediction/gold order can affect the result.
- String-normalized deduplication occurs before numeric parsing. Numerically
  equivalent strings may remain distinct, and punctuation can cause a collision
  between different numbers. For example, `-100` and `100` normalize identically;
  whichever appears first is retained. The adapter preserves the input order.

### Pair scores and aggregation

Rows are keyed by exact `(SubjectEntity, Relation)`. Duplicate rows in either
input retain the **last** row. Missing predictions become `[]`; predictions for
pairs absent from gold are ignored. No scoring penalty is invented for extra
pairs. Gold alias groups are not deduplicated across groups.

For each gold pair, `P = TP / number_of_retained_predictions` and
`R = TP / number_of_gold_groups`. A zero prediction denominator yields P=1; a
zero gold denominator yields R=1. Upstream computes `F1 = 2*P*R/(P+R)`, with
F1=0 when P+R=0:

| Gold answers | Predictions | P | R | F1 |
|---|---|---:|---:|---:|
| Empty | Empty | 1 | 1 | 1 |
| Nonempty | Empty | 1 | 0 | 0 |
| Empty | Nonempty | 0 | 1 | 0 |

The adapter invokes `evaluate_per_sr_pair`, `macro_average_per_relation`, and
`micro_average_per_relation`; it never recomputes F1 or averages independently.

- **Macro**: arithmetic means of per-pair P, R, and F1. Macro-F1 is the mean of
  pair F1s, not the harmonic mean of macro-P and macro-R. Overall macro averages
  all pairs equally, rather than giving each relation equal weight.
- **Micro**: sum TP, retained prediction counts, and gold group counts, then use
  pooled P/R and upstream F1. Zero pooled denominators use the same conventions.
- Both sets are reported per relation and overall. JSON keeps full float
  precision; upstream's standalone CLI rounds its presentation to three decimals.

## Verification scope

Automated tests use synthetic fixtures for normalization, alias overlap, numeric
boundaries and edge cases, empty answers, missing/extra pairs, duplicate rows,
unequal-size macro/micro aggregation, CLI errors, and strict parsing. Adapter
metrics are compared directly with calls to the vendored upstream functions.
Tests also verify provenance checksums, input preservation, and full-precision
JSON serialization. These checks validate the software protocol; they do not
constitute a baseline run or model-quality experiment.
