# RAG-01: offline evidence corpus

This foundation stores externally supplied text for future Standard RAG, RAVEC
and learned set-level revision experiments. It does not download, generate,
retrieve, embed, rank, chunk, verify or otherwise interpret evidence. There are
no model/GPU dependencies or network calls. No factual corpus was acquired as
part of implementation; tests use synthetic fixtures.

## Format and validation

`llm_kbc.evidence_corpus.EvidenceDocument` is a frozen typed dataclass. Input is
UTF-8 JSONL: one object per physical line. All output documents contain:

| Field | Meaning and validation |
|---|---|
| `doc_id` | Derived `doc-` plus 64 lowercase SHA256 hex characters. Optional on input; a supplied value must match. |
| `title` | Nonempty source title, preserved exactly. |
| `text` | Nonempty source text, preserved exactly, including whitespace and Unicode. |
| `source` | Explicit external category: `encyclopedia`, `official_website`, `news`, `research_publication` or `other_external`. This is not a reliability score. Publisher/source identity is retained in the original URL. |
| `source_url` | Absolute HTTP(S) source URL; no credentials, whitespace, control characters or invalid URL characters. Fragments and revision query parameters are preserved. No URL is fetched. |
| `retrieved_at` | Acquisition timestamp: `YYYY-MM-DDTHH:MM:SS[.fraction]Z` or an explicit `+/-HH:MM` offset. Must be a real date/time. Unknown offset `-00:00` and leap seconds are not supported. Original spelling/offset/precision is retained. |
| `revision_id` | Optional nonempty string identifying the source's revision, or `null`. Numeric revision IDs must be supplied as strings. Missing revisions serialize as `null`; none is invented. |
| `language` | A structurally checked language tag, e.g. `en`, `el-GR`, `zh-Hant`. Checks 2–8 initial letters, followed by optional hyphen-separated 1–8 character alphanumeric subtags; does not validate against a language registry. |
| `content_sha256` | SHA256 of exact UTF-8 `text` bytes, excluding JSON encoding/escaping. Optional on input; a supplied value must match. |

Strings must not be blank, contain NUL, or contain invalid Unicode surrogates.
There is no case folding, Unicode normalization, stripping, numeric conversion,
alias expansion or line-ending transformation within fields. Input CRLF is
accepted; emitted JSONL uses LF, sorted keys and compact JSON with unescaped
Unicode. Newlines within text are JSON-escaped. Unicode line/paragraph separators
within strings remain text, not record boundaries.

Blank records, non-object records, missing/unknown fields, duplicate JSON keys,
nonstandard JSON constants (`NaN`, `Infinity`), invalid UTF-8, incorrect supplied
hashes/IDs and duplicate documents fail with file/line context where applicable.
An empty input file is valid and produces an empty corpus with a zero-count
manifest. Validation completes before any final output is published.

Minimal input record (placeholders illustrate the schema, not factual evidence):

```json
{"title":"<original title>","text":"<externally supplied source text>","source":"other_external","source_url":"https://example.org/document","retrieved_at":"2026-06-30T12:30:00Z","revision_id":null,"language":"en"}
```

## Identity, duplicates and manifest

IDs hash a JSON identity object containing exactly `source`, `source_url`,
`language`, `revision_id` and `content_sha256`. Encode it with `json.dumps(...,
ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)`, append
one LF, encode UTF-8, then prepend `doc-` to its SHA256 hex digest. This excludes
title and retrieval timestamp: recrawling or retitling an otherwise identical
document does not create another independent source. Source URLs are not
canonicalized; distinct revisions/content/source identities have distinct IDs.

Duplicate IDs **and exact duplicate text hashes across different sources** are
rejected, with the first occurrence's line number. Nothing is silently dropped,
merged, or selected as the preferred source. Curate such inputs explicitly and
retain excluded source provenance in the acquisition archive. Syndicated text
must not count as independent corroboration. Near-duplicates and semantic copies
are not detected; explicit provenance grouping is future work.

The corpus is sorted lexicographically by `doc_id`. The accompanying
`<output>.manifest.json` records:

- Schema version `evidence-corpus-v1`, document count and exact corpus SHA256.
- Exact original input-file SHA256, including its ordering/whitespace/encoding.
- ID, duplicate and ingestion policy identifiers.
- Every document's ID, title, source, URL, retrieval timestamp, revision,
  language and text hash, in corpus order. Full text resides in the corpus.

No wall-clock build time, input/output filesystem path or random value is added.
Identical input bytes produce byte-identical corpus and manifest files at new
output paths, including on another machine using this implementation. Reordering
or reformatting input preserves the canonical corpus but changes the input hash
in the manifest. Archive the manifest, original supplied documents, acquisition
records, and implementation Git revision together. Hashes establish integrity
and identity, not authenticity, independence, factual correctness or licensing.

## Offline CLI and Python API

Use the existing project environment; no additional dependency is required.
For example, after placing reviewed external documents in an existing directory:

```sh
python -m llm_kbc.evidence_corpus \
  --input /absolute/path/to/reviewed_external_documents.jsonl \
  --output /absolute/path/to/existing/corpus_directory/corpus_v1.jsonl
```

This writes `corpus_v1.jsonl` and `corpus_v1.jsonl.manifest.json`. The output parent
must already exist. Existing output files or symlinks are refused. Inputs are
read only and checked again before publication for ordinary concurrent changes.
Each artifact uses a closed, flushed same-directory temporary file and an
exclusive hard link; unsupported filesystems fail without an overwrite fallback.
Prefer local/HPC storage supporting hard links rather than Google Drive FUSE.
The manifest is published last and is the completion marker. There is no atomic
transaction across both files: interruption or a publication error may leave an
orphan corpus without a manifest. Consumers must require both files and verify
the corpus hash before use. Parent-directory crash durability is not guaranteed.

Reusable CPU-only functions:

- `validate_document(mapping)` validates/derives one `EvidenceDocument`.
- `serialize_document(document)` revalidates and returns canonical JSONL bytes.
- `load_documents(path)` validates the complete file and duplicates, returning
  documents in input order; it performs no publication.
- `ingest_corpus(input_path, output_path)` publishes both artifacts and returns
  the manifest dictionary.

The implementation materializes the corpus and manifest in memory. Streaming,
indexing, chunk IDs, a manifest-aware consumer and incremental updates are out
of scope. A changed snapshot must use new output paths.

## Research-integrity and temporal contract

Only independent external source text is eligible. Benchmark gold answers,
evaluation reports, model predictions, raw generations, benchmark-derived answer
lists, submissions, leaderboards and gold-derived mirrors are **not evidence**.
Training gold is also excluded: permission to use it for demonstrations does not
make it an independent evidence source. No benchmark loader or evaluator is
imported, and no gold labels are consulted to curate candidate text.

Enforcement rejects benchmark schemas (including `SubjectEntity`, `Relation`,
`ObjectEntities`, `gold`, `predictions`, `metrics`), all unknown document fields,
non-external source categories and identifiable benchmark/thesis URLs. URL guards
recognize LM-KBC, KAMEL, KnowledgeNet, RAVEC, this thesis's named repository and
the AKBC website. Path guards reject `train.jsonl`, `val.jsonl`, `test.jsonl`,
`results`/`experiments` ancestor directories, and named LM-KBC/dataset-year
directories, following resolved symlink targets. Stage independently reviewed
external text outside those excluded locations (e.g. a separate `data/evidence/`
directory), and use the same location policy for emitted corpora. The existing
Git ignore rules exclude files under `data/`; archive corpus snapshots and
manifests separately. These are conservative safeguards, not a comprehensive
origin detector.

Renamed, copied, hard-linked or mislabeled prose and unknown mirror URLs cannot
be reliably identified automatically. Declaring `other_external` does not establish
eligibility. Before ingestion, the researcher must review provenance and exclude
benchmark-derived content; retain an acquisition/source inventory and document
the review. Do not use validation answers, errors or evaluation scores to select
evidence text or manufacture answer lists. Future query-driven acquisition must
use permitted input identities/relation definitions and a frozen acquisition
protocol, not held-out gold. Pin the same approved corpus for controlled methods.

`retrieved_at` means **when the document was obtained**, not when a claim became
true or when the document was published/revised. `revision_id` is opaque; a
revision's factual date cannot be inferred from its spelling. For the research
plan's LM-KBC 2026 target date of 1 July 2026, a later retrieval is neither automatic
evidence of that past world state nor automatically invalid. Review the actual
revision/publication dates and claim scope separately; flag unresolved temporal
ambiguity. This minimal schema does not assert publication dates, fact-validity
intervals, completeness or source trustworthiness.

Confirm applicable licenses, redistribution rights, required attribution and
snapshot permissions before acquisition/publication. This module does not grant
rights or inspect licenses. Direct Wikidata/KB answer lookup is a different
experimental condition from independent textual verification, even if its export
can be represented here. Treat all document content as untrusted data in future
model prompts; the corpus layer does not execute or obey source instructions.
No RAVEC novelty or performance claim follows from this infrastructure.
