# RAG-03.1: Wikipedia article discovery

`llm_kbc.wikipedia_discovery` discovers **article candidates**, not object answers,
factual evidence or verified entity links. It is an independent retrieval-assisted
research condition, not an official closed-book submission. No model, embeddings,
article text, benchmark answers, evaluation results or snippets enter selection.
The collector, preprocessor, BM25, baselines and evaluator remain unchanged.

## Fixed policy and temporal scope

Policy `wikipedia-article-discovery-v1` uses BM25's existing identity loader and
`subject-relation-lexical-v1` query templates. All other input fields, including
`ObjectEntities`, are ignored and excluded from the canonical identity hash.
Original spelling and input order are preserved; normalized duplicate identities
are rejected. Prefer a subject/relation-only JSONL, with no gold fields.

For each input:

1. Attempt an exact subject-title lookup against English Wikipedia. Inspect
   `info|pageprops`, including current normalization, redirects and disambiguation.
   Invalid direct-title syntax is diagnosed and skipped; the unchanged lexical
   query is still searched. Title normalization is allowed and preserved in cache,
   but is never interpreted as semantic entity resolution.
2. Always search the existing subject-relation query, retaining the first **ten**
   API results. No continuation, query rewriting, retries or extra searches are
   performed. Snippets are disabled (`srprop=""`) and are never used, even if an
   unexpected server response contains one.
3. Fetch current `info|pageprops` metadata for the unique search page IDs, without
   following redirects. Metadata response order does not replace search ranking.
4. Select up to **three unique page IDs**: an eligible direct candidate first,
   then eligible search results in original rank order. Deduplicate by page ID,
   retaining original search ranks, including gaps. No semantic reranking,
   subject filter, factual correction or answer-based decision is performed.

Only main-namespace pages with valid titles/positive IDs are eligible. Missing,
invalid, disambiguation, redirect and identifiable benchmark/thesis pages are
excluded with diagnostics. A direct lookup that follows a current redirect is
**not** prioritized or accepted as a historical mapping. Its target may enter
independently through search, but only as an explicitly unverified candidate.
Current title changes between search and metadata are recorded, not hidden.
Duplicate search IDs are diagnosed; duplicate IDs or missing coverage in the
metadata response are malformed responses and abort the run.

Page metadata is validated before exclusion. In formatversion 2, present
`missing`, `invalid` and `redirect` flags must be the boolean `true`; false,
numbers, strings, null, containers and conflicting flags are rejected. Existing
pages (including redirects/disambiguation) require a positive page ID, nonempty
title and integer namespace. A missing title lookup requires title and namespace;
a missing page-ID lookup may legitimately contain only its positive `pageid` and
`missing: true`. Invalid-title outcomes require a title and nonempty
`invalidreason`; namespace is optional. Any provided ID, namespace, title or
page-properties object is still validated. Disambiguation properties must contain
string values. Malformed metadata aborts discovery, never becomes `unresolved`.

Every selected candidate has `entity_identity_verified: false` and
`historical_mapping_verified: false`. A title match does not establish that the
page represents the intended subject. Disambiguation does not trigger automatic
selection of its links. Multiple possible interpretations remain candidates;
zero candidates yields `unresolved`, never a fabricated entity link.

The subsequent historical content cutoff remains **2026-07-01T00:00:00Z**.
Discovery itself queries a **current search index at API acquisition time**.
It is not historical search, and neither current ranking nor current page IDs
establish what search would have returned before the cutoff. Page creation,
renaming, merging or redirects after the cutoff can make a discovered article
unavailable or unsuitable historically. Current metadata eligibility also does
not prove historical page eligibility.

Each cached response retains its acquisition timestamp; candidates retain search
or direct discovery time and metadata-check time. Cached requests may have
different dates: this is not a globally atomic Wikipedia snapshot. Offline replay
does not invent a new discovery timestamp. Actual revision IDs, revision
timestamps, attribution and historical wikitext are obtained **later**, using the
unchanged revision collector. Revision time and retrieval time are not dates when
facts became true. No current extracts/snippets are evidence.

## API, cache and failure policy

The only endpoint is `https://en.wikipedia.org/w/api.php`:

- Common parameters: `action=query`, JSON `formatversion=2`, `maxlag=5`.
- Direct: `titles=<subject>`, `redirects=1`, `prop=info|pageprops`,
  `ppprop=disambiguation`.
- Search: `list=search`, `srsearch=<query>`, `srnamespace=0`, `srlimit=10`,
  `srprop=""`, `srinfo=""`.
- Search metadata: `pageids=<rank-ordered unique IDs>`, `prop=info|pageprops`,
  `ppprop=disambiguation`; no redirect following or revisions/content request.

Transport and caching reuse `wikipedia_collect._snapshot()` and its existing
`wikipedia-revision-source-v1` cache envelope. That envelope version identifies
the reused transport format; a discovery cache is not a historical article.
SHA256 of canonical endpoint/parameters determines the cache key; envelopes
retain raw API response bytes as UTF-8 text, response hashes, original timestamps,
headers and User-Agent. Request identity, envelope and response hash checks apply
on every replay. A corrupted cache is rejected, never silently refreshed.

Live misses require `--allow-network`, an informative project/version/contact
User-Agent, and a finite delay of at least one second. Requests are serial with
30-second timeout, maxlag protection and **no automatic retries**. HTTP errors,
rate limits, API warnings/errors and malformed responses stop further discovery.
Missing pages and empty searches are valid discovery outcomes, not transport
failures. No HTTP or API failure is converted into a successful empty result.

Successful and ordinary malformed response bodies are retained by the existing
cache mechanism. API maxlag/ratelimited responses are not reusable success caches;
their original envelopes are retained in the failure report's `network_details`.
HTTP failures expose status/Retry-After where available; the existing transport
does not retain their response bodies. Do not claim an absent HTTP body was cached.
After a rate limit, respect Retry-After and inspect the report before any new live
run. Completed runs are replayable offline; an incomplete live run may still have
cache misses and is not a completed snapshot.

Recovery is an explicit new invocation, not an automatic retry:

- **Transient HTTP failures or API maxlag/rate limits:** inspect the failure
  report, respect Retry-After/server waiting instructions, then use a **new output
  path with the same cache directory**. Successful prefix responses remain frozen
  and are reused without network access or delay; only missing requests are made.
  Failure reports and old output names remain untouched.
- **Malformed cached JSON, API errors/warnings or page metadata:** a new output
  filename alone will replay the same failure. Preserve the failed cache and
  report, investigate the cause, then use a **separately named cache snapshot and
  new output path** for any authorized fresh acquisition. Do not delete, edit,
  overwrite or silently refresh cached entries. A fresh snapshot may have new
  acquisition timestamps/rankings; it is a separate acquisition, not identical
  replay. The same applies to corrupted cache entries: reject and preserve them.

API parameter choices follow the existing collector and MediaWiki interfaces:
[Query](https://www.mediawiki.org/wiki/API:Query),
[Search](https://www.mediawiki.org/wiki/API:Search),
[Info](https://www.mediawiki.org/wiki/API:Info),
[Pageprops](https://www.mediawiki.org/wiki/API:Pageprops), and
[Etiquette](https://www.mediawiki.org/wiki/API:Etiquette).
This implementation was verified with mocked APIs only; no live request or article
download was performed. Real response compatibility remains a pilot check.

## Artifacts and integrity

Given `--output discovery.jsonl`, a successful run writes:

| Artifact | Contents |
|---|---|
| `discovery.jsonl` | One record per input: 1-based `input_index`, subject, relation, query, status, candidates, diagnostics and API response references. |
| `discovery.jsonl.articles.jsonl` | Globally unique `{"page_id": positive_integer}` rows in first-selected order, directly compatible with the existing collector. |
| `discovery.jsonl.manifest.json` | Required completion marker: policy/query versions, endpoint, fixed limits, temporal policy/cutoff, identity fingerprint, counts, output/selection hashes, implementation hash and request settings. |

Each candidate records page ID, current title, acquisition method (`direct_title`
or `search`), original search rank (null for direct), discovery/metadata timestamps,
current page URL and both unverified-identity flags. Each API reference records
stage, exact endpoint/parameters, request-derived cache key, envelope SHA256,
response SHA256, timestamp and received/failed status. Original search ranking,
normalization and redirect chains remain inspectable in the cached responses.
The query-to-page associations must be archived alongside the globally deduplicated
selection; collection deduplication must not erase query coverage.

All inputs remain represented, including `unresolved` records. If no query yields
any article, the selection is empty and `article_count=0`; the existing collector
rejects an empty list, so stop rather than invoking collection in that case.
The selection JSONL alone does **not** encode or enforce the historical cutoff.
Pass the manifest's exact cutoff explicitly to the collector.

`load_discovery(output_path, cache_dir=...)` requires all three artifacts,
checks schema/completion/hashes/counts/ordered identities and deterministically
replays their caches offline. It rejects missing/orphaned, corrupted or
selection-inconsistent artifacts. Integrity against supplied manifests is not
cryptographic proof of authenticity; retain independently trusted archive hashes.

On an ordinary request/validation failure, `<output>.failure.json` retains all
identities, successful intermediate cache references, the failed stage/error,
and subsequent `not_attempted` records. No complete selection is published.
An interrupted run is similarly reported where possible. Preflight errors occur
before requests and need not produce a report. Failure-report publication is
best-effort if storage itself fails. No failure artifact is a completion marker.

The existing exclusive hard-link publisher prevents overwrites. Outputs and
sidecars must all be fresh; the parent must exist. The manifest is published last,
after full input coverage and offline cache replay checks. This is not a
multi-file transaction: a crash/publication failure can leave orphan JSONL files.
Require the validated completion marker; never reuse an orphan output name.
Use local/HPC filesystems supporting hard links, not unverified Drive FUSE mounts.
Place artifacts/caches under a separate `data/evidence/` directory, outside
benchmark/results/experiment locations, so the collector's source guards accept
the selection. Existing Git ignore rules require separate external archiving.

## Proposed 18-training-instance pilot (NOT executed)

Freeze the first three identities per relation in original train-file order,
preserving their overall file order. Do not inspect or rank by answer labels,
cardinality, correctness, model predictions or evaluation results. Do not reuse
the few-shot demonstration selector, which has a different gold-aware purpose.
This is a small engineering/source-quality pilot, not benchmark evaluation.

The following preparation and live commands are **proposed only**. Use a new,
existing-parent directory and replace the contact before explicitly authorizing
the live discovery. No training/validation discovery was executed for this task.

Preparation verifies the reviewed training SHA256
`ad37cd30d1ff4b9f1ef2579b25e64093b202c40da11e8c412e13386f1e5d332e`
before selection/publication. It reuses the identity-only loader, requires all six
relations with three instances each, checks the source remains unchanged, and
publishes only subject/relation JSONL. Hashing all source bytes establishes file
provenance; no gold values enter selection. The CLI always uses this fixed hash;
the Python function's explicit expected-hash override exists for synthetic tests.
Preparation prints the source and ordered-identity SHA256 hashes; archive these
with the identities and later discovery artifacts. It performs no API requests.

```sh
# Proposed preparation only; do not execute until the pilot is authorized.
mkdir data/evidence/discovery_pilot_v1
python -m llm_kbc.wikipedia_discovery \
  --prepare-pilot data/lm_kbc_2026/train.jsonl \
  --output data/evidence/discovery_pilot_v1/identities.jsonl

# Future live metadata discovery ONLY after authorization; no article text.
python -m llm_kbc.wikipedia_discovery \
  --identities data/evidence/discovery_pilot_v1/identities.jsonl \
  --output data/evidence/discovery_pilot_v1/discovery.jsonl \
  --cache-dir data/evidence/discovery_pilot_v1/cache \
  --request-delay 1 --allow-network \
  --user-agent 'LLMKBC-Thesis/0.1 (https://YOUR-REAL-CONTACT-PAGE)'

# Future offline replay, using fresh output names and the frozen cache.
python -m llm_kbc.wikipedia_discovery \
  --identities data/evidence/discovery_pilot_v1/identities.jsonl \
  --output data/evidence/discovery_pilot_v1/replay.jsonl \
  --cache-dir data/evidence/discovery_pilot_v1/cache
```

Require byte-identical discovery, article-selection and manifest files on replay.
Inspect diagnostics, redirect/disambiguation rates, candidate associations and
current source identity without consulting gold answers. At most 54 unique
articles can be selected for 18 inputs; inspect the actual `article_count` before
collection. Shared requests can reduce the number of live API calls.
Discovery makes at most 54 live GETs (direct + search + one metadata batch per
identity), with a delay before every live miss. Empty searches and shared cache
keys reduce this number; warm offline replay makes zero requests and incurs no
transport delay. This is separate from later historical article requests.

Before collection, validate completion and coverage against the frozen input
identities, rather than relying only on the artifact's internal identity hash:

```sh
python - <<'PY'
from pathlib import Path
from llm_kbc import wikipedia_discovery as discovery
from llm_kbc import bm25_retrieval as retrieval
root = Path("data/evidence/discovery_pilot_v1")
identities = retrieval.load_identities(root / "identities.jsonl")
records, articles, manifest = discovery.load_discovery(
    root / "discovery.jsonl", cache_dir=root / "cache")
if len(records) != 18 or manifest["input_identities_sha256"] != discovery._identity_hash(identities):
    raise SystemExit("Pilot coverage mismatch")
print("Articles:", len(articles), "Unresolved:", manifest["unresolved_count"])
PY
```

Historical collection is a **separate, later authorization**, after inspecting
and validating the discovery artifacts. A proposed command is:

```sh
python -m llm_kbc.wikipedia_collect \
  --input data/evidence/discovery_pilot_v1/discovery.jsonl.articles.jsonl \
  --output data/evidence/discovery_pilot_v1/historical_corpus.jsonl \
  --cache-dir data/evidence/discovery_pilot_v1/cache \
  --cutoff 2026-07-01T00:00:00Z --max-articles 54 \
  --allow-network --user-agent 'LLMKBC-Thesis/0.1 (https://YOUR-REAL-CONTACT-PAGE)'
```

The maximum 54 prevents the collector's default ten-article cap from silently
reducing the pilot selection. Require `status=complete` and collected count equal
to discovery `article_count`. Historical failures must remain failures; never
substitute current content or repair identities using benchmark answers.

Remaining limitations: current-index temporal leakage, incomplete search coverage,
unverified entity identity, current-only disambiguation eligibility, page moves,
rate limits, and in-memory artifact handling. No corpus-scale performance, final
retrieval effectiveness, model-ready contexts or RAVEC controller is established.
