# RAG-02: deterministic BM25 retrieval candidates

This Standard RAG extension is an independent retrieval-assisted research
condition, not an official closed-book challenge submission. It performs no
inference, evaluation, model loading, network activity or factual correction.
The existing acquisition/preprocessing modules remain unchanged. Their historical
RAG-02/RAG-03 names refer to earlier infrastructure tasks.

## Inputs and validation

Supply **one** `evidence-passages-v1` JSONL, its manifest and diagnostic sidecar,
and the original source corpus with its corpus manifest and collection report.
Do not concatenate preprocessing variants or replay files. The retriever reuses
`wikipedia_preprocess.load_snapshot()` to reject incomplete acquisition, mismatched
source hashes/counts/provenance, and invalid revision/cutoff metadata. It checks
the passage/diagnostic hashes, schema and completion state, source fingerprints,
document provenance, passage counts, unique IDs, canonical order, text hashes,
derived IDs, window settings, section metadata and coarse source spans.

`--cache-dir` additionally revalidates the original API caches offline, without
refreshing them, including when preprocessing originally ran without cache
verification. Core corpus/manifest/collection fingerprints are always required
and matched. Every archived fingerprint is retained: fingerprints for freshly
checked files must match exactly, including byte counts. Additional freshly
verified caches need not have appeared in the old preprocessing manifest.
Unmatched or mismatching archived fingerprints are rejected, not dropped.

The output distinguishes `source_inputs` (the unchanged archived preprocessing
fingerprints) from `inputs` (files checked during this run, including any freshly
verified caches). `archived_cache_validation` preserves the old preprocessing
status; `cache_revalidated` reports whether this run actually requested and
completed cache verification. Missing, corrupted, or source-inconsistent caches
fail before publication when verification is requested. Without `--cache-dir`,
`cache_revalidated` is false even if preprocessing previously reported `verified`.
Archived cache hashes are checked against collection metadata, but cache bytes
and their recorded byte counts cannot be freshly verified without those files.
Malformed preprocessing `cache_validation` types/values produce a controlled
validation error and CLI exit code 2. No parser installation is needed for retrieval.
Hash agreement proves integrity relative to the supplied manifests, not source
authenticity, factual correctness or independence. Keep trusted archive hashes.

The identity input is UTF-8 JSONL containing nonempty `SubjectEntity` and one of
the six supported `Relation` values. Only these two fields are projected; all
other fields, including `ObjectEntities`, are ignored and excluded from the
identity fingerprint. Duplicate identities after NFKC, casefold and collapsed
subject whitespace are rejected. Original subject spelling and input order are
preserved in queries/results. Invalid JSON/UTF-8, blank records, duplicate JSON
keys and empty identity files fail. Use a subject/relation-only file in practice;
do not supply validation gold, predictions or test artifacts during development.

## Fixed retrieval policies

BM25 uses `k1=1.2`, `b=0.75`, default `top_k=5`, with natural-log positive IDF:

```text
idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(q,d) = sum over distinct query tokens t:
    idf(t) * tf(t,d) * (k1 + 1)
    / (tf(t,d) + k1 * (1 - b + b * len(d) / avgdl))
```

`N` includes every validated passage, including passages with zero lexical tokens;
average length uses the same population. Only passage `text` is indexed, with
no title boost, subject restriction, stemming, stopword removal or synonym
expansion. Repeated query tokens count once (binary query-term frequency).
Contributions are accumulated in sorted token order. Results sort by descending
score, then ascending full passage ID. Zero-score passages are never returned.
An empty index, punctuation-only query or no token overlap yields `no_hits` and
an empty hit list. There is no fallback source or arbitrary tie-filled result.

Tokenization policy `nfkc-casefold-unicode-alphanumeric-v1` applies Unicode NFKC,
then casefold, then extracts maximal Unicode letter/digit runs using Python
`[^\W_]+`. Punctuation, underscores and remaining combining marks delimit tokens.
Diacritics are not deliberately removed; Unicode compatibility characters and
case variants can merge. Numbers are **not parsed**: `104.4` becomes `104`, `4`;
`12,345` becomes `12`, `345`; `-418` becomes `418`; `1e3` stays `1e3`. These are
lexical tokens, not equivalent numeric values. Source text is never rewritten.
Python and Unicode database versions are recorded.

Query policy `subject-relation-lexical-v1` uses the exact original subject:

| Relation | Query |
|---|---|
| `awardWonBy` | `{subject} award winners recipients` |
| `companyTradesAtStockExchange` | `{subject} stock exchange listing` |
| `countryLandBordersCountry` | `{subject} land borders neighbouring countries` |
| `hasArea` | `{subject} total area` |
| `hasCapacity` | `{subject} capacity` |
| `personHasCityOfDeath` | `{subject} place of death` |

No gold labels, answer aliases, model outputs or evaluator results enter queries,
ranking, passage filtering or parameter selection.

## CLI and artifacts

Use the existing Python environment; there are no new dependencies. For a small
offline pilot, create a **new** directory and an `identities.jsonl` with:

```jsonl
{"SubjectEntity":"Albania","Relation":"countryLandBordersCountry"}
{"SubjectEntity":"Andorra","Relation":"hasArea"}
```

```sh
python -m llm_kbc.bm25_retrieval \
  --passages data/evidence/wiki_pilot/rag03_passages_verified.jsonl \
  --corpus data/evidence/wiki_pilot/smoke_corpus_v2.jsonl \
  --cache-dir data/evidence/wiki_pilot/cache_smoke_v2 \
  --identities results/rag02_bm25_smoke/identities.jsonl \
  --output results/rag02_bm25_smoke/retrieval.jsonl
```

The parent output directory must exist and the output names must be new. Optional
`--manifest`, `--corpus-manifest` and `--collection` relocate the corresponding
input sidecars. Diagnostics remain `<passages>.diagnostics.jsonl`.

Each output row has `input_index` (1-based), `SubjectEntity`, `Relation`, `query`,
`status` (`retrieved`/`no_hits`) and `hits`. Each hit contains its 1-based `rank`,
unrounded floating-point `score`, every original passage field (including full
text, ID/hash, parent ID, section, timestamps, revision URL and coarse spans),
and source category, page ID, license, history URL and parent content hash.
These are ranked candidates, **not final model-ready contexts**.

`<output>.manifest.json` records schema `bm25-retrieval-v1`, completion status,
input artifact hashes/sizes, upstream source fingerprints, canonical identity
fingerprint, retrieval settings, policy versions, code hash, Python/Unicode
versions, archived and current cache verification status, record/passage/no-hit counts, and output
hash. Source/identity ordering is retained; manifests contain no absolute paths,
wall-clock build timestamps or randomized IDs. Identical inputs/configuration and
software produce byte-identical output and manifest at fresh paths. Floating-point
results are not promised bitwise identical across arbitrary runtimes/platforms.

All validation and retrieval finishes before publication, and inputs are checked
again for ordinary concurrent changes. Existing files/symlinks are refused. The
existing exclusive hard-link publisher writes JSONL first and manifest last.
There is no multi-file transaction: interruption can leave an orphan JSONL, which
must not be accepted without a matching completion manifest. Filesystem hard-link
support is required; no overwrite fallback is used. Archive both artifacts and
their source snapshot outside temporary scratch storage. `results/` is ignored
by Git. Existing baselines, datasets and archived evidence are never rewritten.

## Pilot interpretation and temporal limitations

The verified snapshot contains 181 passages from two historical articles,
Albania revision `1360594420` (2026-06-22T13:28:29Z) and Andorra revision
`1361153783` (2026-06-26T00:54:43Z). Collection used the inclusive cutoff
`2026-07-01T00:00:00Z`; API acquisition timestamps are in November 2026. The midnight
cutoff is a specific acquisition choice, not evidence that the benchmark defines
that exact instant. Revision time is not factual validity time, and acquisition
time is not when a claim became true. The collector refuses current-revision
fallbacks and current redirect mappings in historical runs.

The executed two-query pilot returned five positive-score candidates per query.
The leading candidates were Albania/Demographics (9.698603557733705) for the border
query and Andorra/Geography/Environment (7.616469384237019) for the area query.
Other ranked hits belong to the other article: subject relevance and factual
support are not guaranteed by lexical BM25. This pilot is an integrity and ranking
smoke check, **not full-benchmark retrieval effectiveness or answer completeness**.

Full ranked text/provenance is stored locally in
`results/rag02_bm25_smoke/retrieval_v1.jsonl`; a second fresh run
`retrieval_v1_replay.jsonl` produced byte-identical JSONL and manifest. Both
revalidated the API caches, and all 19 existing pilot/archive files retained their
original hashes. Retrieval JSONL SHA256:
`e5d0498ff3f19f2ec27b8949b23edac5eeb9cc30b123f1c6e9ec67958e463466`.
Manifest SHA256:
`f5b0dec582cc0d01eba2b1e76883b1b159ca952a6c17321edc4a5ac430d75d94`.
These local artifacts are ignored by Git and require separate archiving.

| Query | Rank | Passage ID prefix | Score | Article / section |
|---|---:|---|---:|---|
| Albania borders | 1 | `f9e2c772053d` | 9.698603557733705 | Albania / Demographics |
| Albania borders | 2 | `c62a1338a5b6` | 6.013009848192683 | Albania / lead |
| Albania borders | 3 | `1350b5c8e41c` | 5.968292164340277 | Andorra / Economy / Transport |
| Albania borders | 4 | `0bf4545c4b73` | 4.738208217098295 | Andorra / Geography / Environment |
| Albania borders | 5 | `c519a000e625` | 4.006427767480596 | Albania / History / Antiquity |
| Andorra area | 1 | `0bf4545c4b73` | 7.616469384237019 | Andorra / Geography / Environment |
| Andorra area | 2 | `cca08457975d` | 6.927088951686819 | Albania / Geography |
| Andorra area | 3 | `0276d55e6595` | 5.762867486803902 | Andorra / Economy |
| Andorra area | 4 | `e8259043940d` | 5.1048073091516315 | Albania / Economy / Energy |
| Andorra area | 5 | `da9044e7703b` | 4.655584862519979 | Andorra / lead |

Prefixes omit `passage-`; full IDs, text, original timestamps, license and source
spans are in the JSONL. Albania hits use
`https://en.wikipedia.org/w/index.php?oldid=1360594420`; Andorra hits use
`https://en.wikipedia.org/w/index.php?oldid=1361153783`.

### RAG-02.2 validation replay

The targeted cache-validation fixes were replayed at fresh paths
`results/rag02_bm25_smoke/retrieval_rag022.jsonl` and
`retrieval_rag022_replay.jsonl`. Both used the same verified source snapshot,
two identities, default top five, and offline API-cache revalidation. Their
retrieval records are byte-identical to the original `retrieval_v1.jsonl`,
including all rankings, scores, text and provenance. Updated manifests are
byte-identical across the two executions.

The retrieval SHA256 remains
`e5d0498ff3f19f2ec27b8949b23edac5eeb9cc30b123f1c6e9ec67958e463466`.
The updated manifest SHA256 is
`c6bf87e03f519b679c2c90057da2fc56eb0be542f838e673919e75765145e916`.
Relative to the earlier v1 manifest, only `code_sha256` and the newly explicit
`archived_cache_validation: verified` field differ. `cache_revalidated` remains
true. A changed implementation hash changes manifest bytes and their fingerprint;
it does not imply changed retrieval scores or evidence. Earlier artifacts remain
unchanged, and no full validation retrieval was performed.

The source preprocessing manifest reports 234 unprocessed/retained-verbatim
events. Unsupported markers and overlapping windows remain part of the frozen
passage texts; this task does not silently filter them or treat them as verified
facts. Original spans are coarse enclosing nodes, not exact quotation alignment.
Memory use scales with the in-memory corpus, postings and retrieved text; this is
not a full-Wikipedia indexing system. BM25 scores are not confidence probabilities.

Next: jointly preflight ranked candidates with all four pinned native tokenizers,
pack the same ordered evidence for every model under a fixed input budget, and
freeze that shared context artifact before implementing separate RAG inference.
