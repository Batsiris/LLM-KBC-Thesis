# RAG-04.4.2: query-driven historical corpus assembly

`llm_kbc.rag_corpus_assembly` joins validated Wikipedia discovery to the existing
historical collector and offline preprocessor. It does not discover new articles,
rank passages, load gold objects, generate answers or run a model. Discovery may
contain arbitrary datasets/relations: assembly reads candidate page IDs and
preserves the original query records without interpreting their task schema.
The current acquisition adapter is English Wikipedia, not a language-agnostic
collector. The completed Aristotle discovery pilots are researcher-reported;
this implementation has not collected their 82-article inventory.

## Interface

The Python entry point is `assemble(discovery_path=..., discovery_cache_dir=...,
output_dir=..., ...)`; `python -m llm_kbc.rag_corpus_assembly --help` lists flags.
Use the existing Python 3.11+ environment and optional
`mwparserfromhell==0.7.2` preprocessing dependency. No new dependency is added.
Fresh preprocessing requires that dependency; validated passage reuse does not
import the parser. An empty article inventory also needs no parser.

Example **offline** assembly from a completed discovery snapshot:

```sh
python -B -m llm_kbc.rag_corpus_assembly \
  --discovery /path/to/pilot/discovery_5.jsonl \
  --discovery-cache-dir /path/to/pilot/discovery_cache \
  --output-dir /scratch/project/new_assembly \
  --store-dir /scratch/project/wikipedia_articles \
  --revision-cache-dir /scratch/project/revision_cache \
  --cutoff 2026-07-01T00:00:00Z --offline-only --no-clobber
```

The output directory must not exist and its parent must exist. Default store is
`<output parent>/wikipedia_articles`; default revision cache is `<store>/cache`.
Discovery and revision caches are separate configurable paths, so new revision
responses need not be written into a frozen discovery archive or a full home
directory. Offline is the default; missing revision responses fail explicitly.
`--offline-only` and `--allow-network` are mutually exclusive. No-clobber is always
enforced, including without the optional `--no-clobber` flag.

Future live collection requires explicit authorization and `--allow-network
--user-agent 'Project/Version (real contact)'`. It uses the existing collector's
serial requests, request delay (default one second), timeout, maxlag handling and
no application-level retries. The planned 82-article pilot permits at most 82
collector fetch calls from a fresh revision cache (one per selected article;
cache hits require no fetch call). A fetch call is not necessarily one physical
HTTP exchange: Python's HTTP transport may automatically follow redirects before
the collector checks the final URL. Redirects can therefore add physical HTTP
exchanges even though the application never retries or paginates. This is not a
strict 82-exchange transport budget. No live collection was performed during
implementation.

Cutoff defaults to the validated discovery artifact. An explicit differing cutoff
fails unless `--revalidate-cutoff` is supplied. That option partitions collection
by the new cutoff and verifies/replays revision requests using the new `rvstart`;
it cannot reuse a differently bounded collection just because its revision happens
to be old enough. It does **not** revalidate the historical search index, entity
identity or factual validity. Discovery still describes the current index at its
original acquisition times. No current-content fallback or redirect substitution
is added. The original discovery cutoff and the new collection cutoff are recorded.

## Reuse rules

Each store slot is keyed by SHA256 of page ID, exact canonical cutoff, English
edition and collector policy. A passage filename has a 24-hex-character locator
derived from the full preprocessing configuration. The shorter locator avoids
Windows path-length problems; full configuration, content hashes and passage IDs
are still verified. A locator collision fails closed rather than selecting the
wrong configuration. Slots must not be symlinked directories.

On every reuse, including offline runs:

1. Validate corpus, corpus manifest and complete collection report with
   `wikipedia_preprocess.load_snapshot(..., cache_dir=...)`.
2. Verify exactly one requested page, cutoff, edition, request provenance, source
   revision/timestamp, attribution and original raw revision API response.
3. Validate passages, manifest, diagnostics and parent snapshot with the shared
   `bm25_retrieval.load_passages()`. No BM25 scoring is invoked or altered.
4. Require supported collector, preprocessing and parser policy versions and exact
   chunk settings. All recorded fingerprints remain checked. Archived implementation
   hashes are provenance, not an assumption that all files were produced by current code.

Missing collection slots invoke the existing collector with exactly one page ID
and `max_articles=1`, avoiding its default ten-article batch cap. Existing complete
collections with no passage output get preprocessed once. Different chunk settings
use a new passage filename and retain the original outputs. Malformed, corrupted,
unsupported artifact versions or partially published artifacts fail; they are never silently
refreshed, overwritten or treated as ordinary cache misses.

Existing archives can be imported explicitly, including multi-document corpora:

```sh
python -B -m llm_kbc.rag_corpus_assembly \
  --discovery /path/to/discovery.jsonl --discovery-cache-dir /path/to/discovery_cache \
  --output-dir /scratch/project/new_imported_assembly \
  --store-dir /scratch/project/wikipedia_articles \
  --reuse-snapshot /archive/corpus.jsonl /archive/passages.jsonl /archive/revision_cache
```

Repeat `--reuse-snapshot CORPUS PASSAGES CACHE` for distinct archives. Use `-` for
PASSAGES to import only collected documents. All supplied archives are validated;
ambiguous duplicate page IDs across them fail. A valid archive with different
chunk settings supplies documents but gets fresh passages. Unsupported or corrupt
passages still fail validation; omit them explicitly with `-` to request independent
preprocessing of an otherwise verified corpus. Cutoff/edition mismatches fail.
Selected documents, their metadata and compatible passages are copied into the
store without changing originals. Referenced raw cache envelopes are copied only
when absent; conflicting immutable envelopes fail. Original snapshot/passage
fingerprints and preprocessing code provenance are retained in imported metadata.

## Outputs and validation

Every successful output directory contains:

| Artifact | Purpose |
|---|---|
| `query_articles.jsonl` | Original validated discovery records in original order; all per-query candidate associations and diagnostics retained. V1 and v2 schemas remain distinguishable. |
| `query_availability.jsonl` | One ordered availability record per original query, including queries with no evidence. |
| `articles.jsonl` | Global first-encounter page-ID inventory, compatible with the existing collector. |
| `corpus.jsonl` and `.manifest.json` | `evidence-corpus-v1` with an explicit assembly-only duplicate-content policy; canonical document-ID order, original wikitext and provenance. Only collected sources appear here. |
| `corpus.jsonl.collection.json` | Derived acquisition index linking verified source snapshots; not a claim that assembly made fresh API calls. |
| `passages.jsonl`, `.manifest.json`, `.diagnostics.jsonl` | Standard `evidence-passages-v1` composed from validated per-document outputs, with original passage IDs, source locations and information-loss diagnostics. |
| `manifest.json` | `rag-corpus-assembly-v2` completion marker, published last. |

The final manifest records discovery fingerprints/version, both cutoffs, query
and unresolved counts, article/document/passage counts, ordered store references,
revision metadata and attribution, source/passages fingerprints, reused/new counts,
configuration, availability/outcome counts, network permission and implementation hashes. Paths and new
wall-clock timestamps do not enter it. Aggregate passage manifests identify the
assembly code as their writer; per-source artifacts retain their original code
provenance. They do not pretend that reused text was newly cleaned.

`newly_collected_articles` means a new collected snapshot was materialized by the
collector, potentially entirely from cached API responses. It is **not** a count
of network downloads. Imported or existing snapshots count as reused. Similarly,
new preprocessing counts actual cleaner invocations; copying compatible passages
counts as reuse. Thus cold and warm run manifests necessarily differ in operational
counts, while their corpus/passage artifacts and fingerprints remain identical.
Terminal evidence recording/reuse has separate `newly_recorded_terminal_articles`
and `reused_terminal_articles` counters; these are not collected documents.
Repeated runs with identical frozen inputs and the same reuse plan produce
byte-identical assembly manifests at fresh output paths.

Consumers should call:

```python
from llm_kbc.rag_corpus_assembly import load_assembly

passages, manifest = load_assembly(
    "/scratch/project/new_assembly",
    discovery_path="/path/to/pilot/discovery_5.jsonl",
    discovery_cache_dir="/path/to/pilot/discovery_cache",
    store_dir="/scratch/project/wikipedia_articles",
    revision_cache_dir="/scratch/project/revision_cache",
)
```

This requires the completion marker, revalidates discovery and all referenced
store/cache artifacts, verifies the composed acquisition index and fingerprints,
and compares aggregate documents/passages/diagnostics and query associations to
their verified sources. All producer-defined aggregate corpus, collection and
passage manifest fields are independently reconstructed, including empty outputs.
Recomputing dependent hashes cannot authorize contradictory policy, chunking,
count, source-reference or artifact-role claims. The aggregate passage writer hash
must equal the **recorded** assembly implementation hash; archived supported
implementations do not have to match the currently installed source hash.

For an assembly corpus, lower-level Python consumers must explicitly request
`load_passages(..., allow_duplicate_content=True)` (or `load_snapshot` with that
option). This requires the supported assembly marker and duplicate policy.
The standalone BM25 CLI retains its original strict defaults; use `load_assembly`
and its validated passages for these new corpus snapshots. No retrieval scope or
query-local BM25 statistics are implemented here.
Future dense/hybrid retrievers should use the same frozen
snapshot when comparing retrieval algorithms; no such retriever is implemented here.

## Versioned outcomes and completion

The outcome policy is `historical-evidence-availability-v1`. Every selected
unique page has exactly one outcome:

| Outcome | Reason / evidence | Behavior |
|---|---|---|
| `usable` | Valid historical document with at least one passage | Collect/reuse, preprocess/reuse and include source and passages. |
| `zero_passages` | `no_usable_passages` | Keep the valid source and diagnostics; no passage eligibility. |
| `unavailable` | `missing_page` or `no_eligible_revision` | Record validated terminal evidence and continue. |
| `excluded` | `unsupported_source`, `empty_source`, or `ambiguous_or_directive_source` | Record the explicit source-policy exclusion and continue. |

The collector's standalone behavior is unchanged: incomplete per-page collection
still publishes no complete corpus. The new opt-in `article_outcome()` validates
the cached formatversion-2 response and raises a typed `TerminalArticleOutcome`
only for supported terminal conditions. No error-message matching is used.
Missing flags must be boolean `true`; invalid/contradictory metadata, malformed
revisions, absent required fields, incorrect identity/language, newer revisions
or current redirect substitutions fail. Missing/empty revision lists can establish
no eligible revision only when the accompanying page metadata is valid and there
is no unresolved continuation. A `continue` object, when present, must contain
exactly `continue` and `rvcontinue`, both nonblank UTF-8 strings without ASCII
control characters. Tokens are opaque. An empty revision result with continuation
is an incomplete execution, never terminal unavailability. A valid returned
historical revision may carry valid continuation: the collector requests only the
single newest revision at/before the cutoff. No pagination or retries are added.
Missing content is a validation failure; an explicitly present empty string is an
exclusion. Hidden/incomplete source payloads are not guessed into empty content.

A terminal store slot contains `requests.jsonl`, the original incomplete
`corpus.jsonl.collection.json` and `outcome.json`. The latter includes the outcome
policy, page ID, reason, cutoff, exact API request, title, revision metadata when
available, attribution, retrieval time, original cache key/envelope fingerprint
and raw-response fingerprint. It contains no invented document or passages.
On every read/reuse, raw cache bytes and request provenance are validated again,
classification is repeated, and the outcome file is compared exactly. The native
collector's partial report metadata is independently reconstructed by replaying
its existing document producer against the verified response and request. Field
presence, types and values must match, including page ID/title, revision ID/time
and cutoff status when produced. Missing-page reports have no page/revision
metadata; revision-less reports have page metadata but no revision metadata.
Missing native fields cannot be substituted with invented `null` values. The
separate `outcome.json` schema explicitly uses `null` revision ID/time when no
revision exists. Diagnostic error wording is retained, not used for classification
or required to match regenerated wording.

The completion reader enforces:

```text
selected_article_count = article_count
selected_article_count = collected_article_count
                       + terminal_unavailable_articles + policy_excluded_articles
collected_article_count = document_count = usable_articles + zero_passage_articles
collected_article_count = reused_collected_articles + newly_collected_articles
terminal_unavailable_articles + policy_excluded_articles
    = newly_recorded_terminal_articles + reused_terminal_articles
```

New/reused preprocessing counts cover collected articles, including zero-passage
sources. Every selected page remains in `articles.jsonl` and every association
remains in `query_articles.jsonl`. An assembly containing only terminal outcomes
can complete with an empty corpus and no parser invocation. Missing offline
caches, transient HTTP/rate limits, malformed responses, corruption or inconsistent
provenance remain incomplete executions, never terminal absence. No completed
assembly manifest is published for them.

## Query-level availability

Each `query_availability.jsonl` record has:

- `input_index` and `query_record_sha256`, binding the complete original discovery
  record (identity, task/query provenance and associations) without using gold.
- `candidate_page_ids` in original order.
- `available_article_ids` (pages with usable passages) and `unavailable_article_ids`
  (terminal outcomes or collected sources with zero passages), preserving order.
- `eligible_document_ids`, `eligible_passage_count` and ordered `article_outcomes`
  (`page_id`, `outcome`, `reason`). Zero-passage documents remain in the corpus
  but not the query's eligible-document list.
- `status`: `unresolved_discovery`, `resolved_no_usable_evidence`,
  `partial_candidate_availability`, or `full_candidate_availability`.

Full availability means every selected candidate has passages, not that those
passages answer the query or are factually correct. The root
`query_availability_counts` counts these four statuses. `unresolved_query_count`
retains its original discovery-only meaning. No query is dropped, and no final
generation/abstention decision is made here.

## Source-preserving duplicate-content policy

`preserve-distinct-sources-identical-text-assembly-v1` applies only to explicitly
marked assembled corpora. Distinct page/revision identities retain separate
documents, source metadata and passage IDs even when their source text is identical.
There is no semantic deduplication, content rewriting or fuzzy matching. Duplicate
document IDs still fail. Query-specific page-to-document eligibility is preserved.
The normal corpus loader/ingester still rejects duplicate text by default; the
preprocessor's cleaner and BM25 mathematics are unchanged. Repeated text may affect
retrieval statistics, which must be disclosed and held constant across retrievers.

## Failure recovery and research limitations

Processing failures leave `failure.json` with the current page, completed prefix
and error; preflight errors need not create an output directory. Collector reports,
successful store slots and raw caches remain available. A valid collection can be
reused after interruption before preprocessing. Partial/corrupt collection or
passage slots are refused; retain them and use separately named stores/outputs
after investigation. Do not delete failed archives to disguise missing coverage.

For HTTP/rate limits, first respect `Retry-After` and inspect the retained collector
report (`network_details`) and assembly failure report. A missing offline cache is
an execution requirement, not a missing article. A fresh assembly output may reuse
fully validated store slots. If a slot is incomplete, use a **new store** and import
verified successful slots using repeated `--reuse-snapshot CORPUS PASSAGES CACHE`;
retain the previous store unchanged. Successful raw cache entries can be shared
with this recovery run. A complete collection interrupted before any preprocessing
output can be reused; a partially published passage triplet is refused.

Malformed cached responses require a separately named revision cache snapshot as
well as fresh store/output paths. Preserve the original bad cache and diagnostics;
never overwrite an entry or refresh it silently. Terminal outcome files are reused
only after offline evidence revalidation. Validated assembled corpus/passages can
also supply explicitly imported documents; terminal records are not documents and
are re-established from their original revision cache in a new store.

Individual component manifests may exist after a failed assembly; only the final
assembly completion marker authorizes the combined snapshot. Publication uses
existing exclusive hard links and requires filesystem support. It is not a
multi-file transaction or a cross-process store lock; concurrent writers can
fail safely and require an explicit new invocation.

Archive discovery files/caches, the article store, revision caches, assembly and
source code together outside temporary scratch. Hashes establish consistency,
not authenticity, licensing permission, relevance or factual truth. Consistently
rewriting all provenance and its unsigned root cannot be detected as a forgery;
recorded code hashes and acquisition/reuse history need a trusted external archive.
Inputs and outputs are held in
memory and repeatedly verified, suitable for pilots rather than a Wikipedia dump.
Historical page availability, unsupported wikitext and current-index temporal
leakage remain limitations of the existing acquisition/preprocessing policies.
No benchmark effectiveness, external-dataset generalization or RAVEC improvement
is established by this infrastructure.

This supersedes the uncommitted assembly-v1 draft. Unsupported assembly versions
are rejected; archived discovery v1/v2 and standalone collector/preprocessing
artifacts keep their existing schemas and defaults. No archived artifact is migrated
or overwritten. Profile the 82-article pilot before scaling: archive import scans
passage lists per document, validation rereads raw caches, per-article plus aggregate
text/passages consume duplicate storage, and concurrent store writers have no lock.
