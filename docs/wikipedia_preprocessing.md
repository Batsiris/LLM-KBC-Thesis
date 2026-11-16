# RAG-03: offline wikitext preprocessing and evidence passages

This module derives searchable source text from a frozen RAG-02 snapshot. It
performs no retrieval, ranking, embedding, model inference, template downloading,
answer extraction, factual correction or benchmark evaluation. Cleaning rules
are general syntax rules, not relation-specific rules or gold-informed choices.
Original corpora, acquisition metadata and API caches are read only.

## Installation and offline command

One optional, pinned dependency is justified by nested MediaWiki syntax:
[`mwparserfromhell==0.7.2`](https://pypi.org/project/mwparserfromhell/0.7.2/)
(MIT, Python 3.9+, compatible with this project's Python 3.11+). It supplies an
AST instead of a custom regex grammar. The collector/evaluator/inference extras
are unchanged. Install the parser before moving to an offline environment:

```sh
python -m pip install -e ".[dev,evaluation,preprocessing]"
```

No network is used by preprocessing itself. For the locally archived pilot:

```sh
python -m llm_kbc.wikipedia_preprocess \
  --corpus data/evidence/wiki_pilot/smoke_corpus_v2.jsonl \
  --cache-dir data/evidence/wiki_pilot/cache_smoke_v2 \
  --output data/evidence/wiki_pilot/passages_new_run.jsonl \
  --chunk-words 200 --overlap-words 40
```

Choose a new output name for each run; its parent directory must already exist.
The default sidecar inputs are `<corpus>.manifest.json` and
`<corpus>.collection.json`; `--manifest` and `--collection` relocate them explicitly.
`--cache-dir` is optional, but recommended: if omitted, the output states that
cache provenance was **not verified**, rather than claiming a full cache audit.
Supplied but missing/invalid sidecars or caches fail before output publication.

## Input integrity checks

Before cleaning, the pipeline reuses RAG-01's strict document validation and
duplicate detection. It verifies corpus hashes/counts and the manifest's ordered
document metadata, then checks complete RAG-02 status, representation, coverage,
unique acquisition identities, manifest/corpus hash links, pinned source URLs,
revision IDs/timestamps, cutoff checks, license declarations and history links.
It does not accept an incomplete collection, a current extract labeled as source
wikitext, or a sidecar from a different corpus.

When caches are supplied, their hashes and request keys, response-body hashes,
and reconstructed RAG-02 documents are checked. Page IDs, revision timestamps,
cutoff status and attribution must agree with the API response as well. Cache
validation calls only RAG-02's pure parsing functions, never its network client.
Inputs are checked again before publication for ordinary concurrent changes.
Hashes establish consistency/integrity, not authenticity against collusive edits;
retain independently archived checksums and acquisition notes.

## Versioned source transformations

Policy: `wikipedia-preprocessing-v1`, using the pinned parser with
`skip_style_tags=True`. Parser round-trip preservation is checked before offsets
are computed. This is **source-only extraction, not MediaWiki rendering**.
The parser documents [limitations](https://mwparserfromhell.readthedocs.io/en/latest/limitations.html)
around transclusions, crossing syntax, localized namespaces and link trails.

- Wiki links use explicit display text or the literal unpiped title, recursively
  handling nested display markup. Adjacent text/link trails are not separated
  artificially (`[[land border]]s` becomes `land borders`). Labeled external links
  retain their label; unlabeled external links retain their URL.
- Bold/italic apostrophe syntax and recognized formatting wrappers are removed.
  HTML entities are decoded; Unicode spelling and combining forms are not
  normalized. Whitespace is collapsed when passages are assembled. Inserted and
  deleted/struck text remains explicitly qualified; superscripts/subscripts use
  `^{...}`/`_{...}` source notation rather than invented characters or values.
- References, comments, category links, recognized maintenance/hatnote templates
  and citation templates are omitted from passage text **with raw-source loss
  diagnostics**. References/notes can contain facts: their omission is a limitation,
  not a claim that their contents are irrelevant or false.
- Infoboxes with literal, unique named fields preserve every nonempty parameter
  as a literal `field: value`. Field names, value spelling, numeric punctuation
  and ordering are retained; aliases/implicit field references are not resolved.
  Duplicate/positional/nonliteral fields are not guessed. This is not proof of
  template defaults, field semantics or truth.
- Explicit ordered `hlist`/`unbulleted list`/`ubl`/`plainlist`/`flatlist` parameters
  without named options are joined in source order. Known two-argument language
  wrappers retain the supplied text, with a diagnostic for unrendered options.
- A narrow `convert`/`cvt` rule retains a literal numeric value and a recognized
  **source unit token** (`28748 sqkm`, for example). It does no arithmetic,
  rounding, range interpretation or template expansion. Only two positional
  arguments and listed display options are supported. Range/multi-output/unknown
  forms are unprocessed, with all source arguments preserved in diagnostics.
- Simple tables retain captions, unique column headers and rows. A matching row
  becomes `Year: 2001; Recipient: Person A; Amount: 1,234.50`; headerless rows keep
  their ordered cells without invented column names. Nested tables, spanning
  cells, mismatched/ambiguous headers and unsupported layouts are not flattened
  into guessed associations.
- Unknown templates, media links, extension tags, unresolved arguments and
  unsupported tables produce an `[UNPROCESSED ...]` marker. Their complete raw
  source, hash, reason and enclosing location are retained in diagnostics. An
  unknown template's argument is not assumed to be its rendered answer.
- Malformed syntax recovered by the parser as plain text is retained verbatim
  and flagged when known residual delimiters are detected. This is a diagnostic
  heuristic, not a proof that every syntactic error is detected. Unsupported raw
  syntax can therefore remain in a passage; inspect diagnostics before indexing.

There is no current Wikipedia lookup, transclusion expansion, numeric
normalization, inferred recipient/listing fact or benchmark-answer substitution.
Important values may remain only in raw diagnostics when their source structure
is unsupported. Do not treat passages as a complete lossless rendering of the
article or silently index diagnostics as if they were cleaned factual evidence.
The original wikitext remains the authority for manual inspection.

## Sections, chunks and source traceability

Top-level headings define a stack of original heading levels. Equal/shallower
headings close prior sections; deeper headings extend `section_path`. Lead text
has an empty path. Repeated heading names remain distinct through `section_index`.
Paths/levels are stored on every passage; heading locations, including empty
sections with no passages, are also retained in diagnostics. Heading text is
metadata rather than an extra prefix counted in each passage's word budget.

Defaults are **200 Unicode non-whitespace runs per passage**, **40-word overlap**,
step 160, with a final shorter window. Windows stop at section boundaries and
never mix articles. No model tokenizer is used. Word counts are not LLM token
counts. There is no sentence/table-row boundary optimization: long sentences or
rows can cross windows. Overlap reduces context loss but cannot guarantee that
every statement/row fits wholly in one passage.

`source_spans` are zero-based, half-open **Python Unicode character offsets** into
the parent document's original wikitext, with one-based inclusive line numbers.
They enclose top-level AST nodes contributing to the window. Nested infobox
fields/table cells may therefore point to the whole enclosing structure; multiple
overlapping passages can share a large source block. These are **not UTF-8 byte
offsets and not exact alignments of cleaned characters to source substrings**.
Whitespace, links, entity decoding and structural extraction change the text.
Use parent ID, section index and enclosing blocks to inspect the frozen source;
do not present these locations as exact extracted quotation spans.

## Output artifacts

For `passages_new_run.jsonl`, outputs are:

1. `passages_new_run.jsonl`: deterministic passage JSONL.
2. `passages_new_run.jsonl.diagnostics.jsonl`: one diagnostic record per document.
3. `passages_new_run.jsonl.manifest.json`: completion marker, published last.

Every passage includes:

- Stable `passage_id`, parent `doc_id`, article title and source URL.
- Revision ID/timestamp, original retrieval timestamp and language.
- Section path/index/heading level and `[word_start, word_end)` in cleaned section
  word coordinates, distinct from original-source offsets.
- Cleaned `text`, its exact UTF-8 SHA256, configuration and conservative source
  blocks/alignment declaration.

ID policy: `passage-` plus SHA256 of RAG-01 canonical JSON bytes containing parent
ID, section index, word window bounds, cleaned content hash and configuration.
Order is canonical corpus document order, then source section order, then window
start. No timestamp, absolute filesystem path, random value or global output
ordinal enters IDs. Duplicate textual passages in different locations stay
distinct. The same snapshot/configuration/software produces byte-identical output
at different output paths; changing chunk settings intentionally changes IDs.

The manifest records schema `evidence-passages-v1`, completion status, document
and passage counts, all input fingerprints (including supplied caches), parser
version, preprocessing policy/code SHA256, word/overlap configuration, passage and
diagnostic file hashes, and parent licenses/history attribution. Diagnostic events
contain category, action, reason, raw source/hash and enclosing source block;
summary counts distinguish unprocessed/verbatim events from deliberate omissions.

No existing output or symlink is overwritten. Validation/conversion completes
before publication; RAG-01's exclusive hard-link writer publishes each artifact.
This is not an atomic transaction across files. A crash/publication failure can
leave orphan outputs without a completion manifest. Consumers must require the
manifest and verify both hashes. Use a filesystem supporting hard links; archive
all outputs with the original snapshot and source Git revision. The pipeline
currently materializes documents/passages/diagnostics in memory.

## Validation and remaining limitations

Synthetic tests cover nested links, templates, infoboxes, numeric string spelling,
simple/ambiguous tables, Unicode, malformed source, sections, fixed overlap,
coarse source locations, snapshot/cache consistency, stable IDs, byte-identical
replay and no-overwrite behavior. No benchmark gold or evaluation outcomes enter
these tests or transformations.

The archived Albania/Andorra smoke snapshot can be processed offline with the
command above. Inspect the lead border sentences against their original source;
this is a readability/provenance check, **not benchmark evaluation or evidence of
answer completeness**.

Local offline acceptance was executed on Albania revision `1360594420` and Andorra
revision `1361153783`, with default 200/40 windows and API-cache validation. It
produced **181 passages** (107 Albania, 74 Andorra), with **234 unprocessed/verbatim
diagnostic events** (136 Albania, 98 Andorra). Both lead border statements remain
understandable after link cleaning. A replay at a second output path was
byte-identical for passages, diagnostics and manifest. SHA256 checks confirmed
the original corpus, manifest, collection report and two API cache files were
unchanged. Final verified derived output is
`data/evidence/wiki_pilot/rag03_passages_verified.jsonl`, with passage-file SHA256
`0da7b9f2a263c0792959043ca41e596f5cf41e36ad1112932ca0d4333b868f41`.
These are preprocessing checks, not new inference or experimental scores.

Many article structures remain unprocessed; review the
diagnostics before a future retrieval experiment. English namespace/template
conventions are only a limited allowlist, not multilingual template support.
Historical revision time and retrieval time retain their RAG-02 meaning and do
not prove a statement's factual validity. Source text and instructions remain
untrusted data; no source instructions are executed.
