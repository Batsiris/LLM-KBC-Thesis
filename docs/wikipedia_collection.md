# RAG-02: reproducible Wikipedia revision collection

This is evidence acquisition for future Standard RAG, RAVEC and set-level
revision experiments. It accepts an explicitly reviewed article list and reuses
[RAG-01](rag_corpus.md) for evidence validation, IDs, hashes, serialization,
duplicates and corpus manifests. No benchmark reader, evaluator, model, GPU,
embeddings, retrieval ranking, link following or answer extraction is involved.
No live pilot was run while implementing this module. Unit tests use synthetic
mocked responses; documentation browsing is separate from evidence acquisition.

## Input and selection

Supply UTF-8 JSONL, with **exactly one** of these fields per record:

```json
{"title":"<independently chosen Wikipedia article title>"}
{"page_id":123456}
```

These are schema placeholders, not an executed article list. IDs are positive
integers in the explicitly selected Wikipedia language edition, not Wikidata
Q-identifiers. Titles are preserved, must have no surrounding whitespace,
controls, section fragment or API multi-title separator (`|`). Unknown fields,
benchmark answer payloads, invalid IDs, repeated exact requests and empty lists
fail before acquisition. Independently record how page identities were resolved;
the collector performs no search or automatic disambiguation. Input file names
and locations follow RAG-01's benchmark/artifact exclusion policy.

The complete input is validated before selecting the first `--max-articles`
records in file order (default **10**). The report distinguishes total input
records from the selected prefix. A successful pilot is complete for that prefix,
not necessarily the entire list. Changing the cap is an explicit configuration
change; do not extrapolate pilot evidence coverage to all 475 evaluation queries.

Only subject names or independently resolved subject-page IDs may determine this
list. Never use gold objects, correct-answer lists, prediction correctness,
evaluation reports, leaderboard pages or benchmark-derived mirrors for discovery.
The program cannot authenticate the origin of a manually supplied list. The
researcher must retain/review the identity-only roster and resolution procedure.
RAG-01's source restrictions still apply, and identifiable benchmark/thesis
article titles are also rejected even when requested by page ID.

## Official API and collection policy

The only endpoint is the selected Wikipedia edition's official Action API:
`https://<language>.wikipedia.org/w/api.php`. There is no user-supplied endpoint,
search fallback or alternative source. One serial GET per selected article uses:

- `action=query`, JSON `formatversion=2`, `maxlag=5`.
- `prop=info|revisions|pageprops`, `ppprop=disambiguation`, `redirects=1`.
- `rvprop=ids|timestamp|content|contentmodel`, `rvslots=main`, `rvlimit=1`,
  `rvdir=older`; optional `rvstart=<explicit UTC cutoff>`.
- `meta=siteinfo`, `siprop=rightsinfo` for the site's licensing declaration.

The [official revisions API](https://www.mediawiki.org/wiki/API:Revisions)
documents source content and timestamp-based enumeration. The collector checks
that the returned revision timestamp is **at or before** the supplied cutoff.
Without a cutoff, it requests the latest available revision. Cutoffs require a
known timezone and whole-second precision; offsets are converted to UTC without
rounding. No revision is replaced with today's content when history is missing.
API continuation of older revisions is intentionally ignored: exactly one
revision is requested, rather than a complete history.

Text is the exact, unexpanded main-slot **wikitext**, including source templates,
tables, comments, links and references. It is not plain rendered prose. No
HTML stripping, template expansion, infobox interpretation, summary endpoint or
TextExtracts request is made. This avoids presenting current extracts or current
template expansions as an exact historical article snapshot. Linked/template
source pages are not downloaded. Future text conversion/chunking must be a
separately versioned, frozen offline preprocessing step; none is implemented here.

Live access requires `--allow-network` plus a real project/version User-Agent
with contact information, following Wikimedia's
[User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).
Requests are serial, delayed at least one second before each call, with a
30-second timeout and a 20 MiB response-body ceiling. There are **no automatic
retries**. On a transport/HTTP failure or API `maxlag`/`ratelimited` response,
the run stops making live requests and marks the remaining selection unattempted.
HTTP status and `Retry-After`, where returned, are retained. Wait at least the
server-specified interval and inspect the failure before explicitly rerunning.
This stop policy avoids aggressive retry loops; see
[API etiquette](https://www.mediawiki.org/wiki/API:Etiquette) and
[maxlag guidance](https://www.mediawiki.org/wiki/Manual:Maxlag_parameter).
Run a single collector for the pilot; the delay is per process, not a global
throttle shared across simultaneous jobs.

## Redirects, ambiguity and completeness

The response must identify one accessible, main-namespace article and one
available wikitext revision. Missing/invalid pages, hidden/empty content,
unsupported content formats, API warnings, malformed responses and unexpected
page identity substitutions fail explicitly. No search result or alternative
article is silently selected.

For current collection, the API's
[normalization/redirect mappings](https://www.mediawiki.org/wiki/API:Query#Resolving_redirects)
are retained with original requests and final page identities. Redirect targets
with section fragments are still full-page acquisitions, not section-only text.
Circular/unresolved redirects fail. **Cutoff runs reject any returned redirect
mapping**: this is today's routing and does not establish the historical target.
Resolve the historical page identity independently and revise the explicit list
with a documented reason; do not remove the cutoff or substitute current text.
Page titles, namespace, language and page properties are metadata returned at
acquisition, not a reconstructed historical page-move database.

[Disambiguation page properties](https://www.mediawiki.org/wiki/Extension:Disambiguator#With_API)
are checked by presence of the `disambiguation` key (even an empty-string value).
The source magic word `__DISAMBIG__` and sources beginning with `#` are also
rejected conservatively as potential disambiguation/redirect/directive content.
The latter may reject a legitimate numbered-list article. Page properties reflect
current state; these checks are **not a complete classifier for historical or
unmarked disambiguation pages and localized source templates**. Manual review of
pilot revisions and subject identity remains necessary. No candidate article is
chosen from the alternatives on a disambiguation page.

Two requests resolving to the same revision/text fail RAG-01 duplicate validation;
neither is silently dropped. Individual retrieval failures can be inspected
together, but **no corpus is published unless every selected request succeeds**
and the full corpus passes RAG-01 validation.

## Frozen cache and output artifacts

Each query has an immutable cache key: SHA256 of RAG-01 canonical JSON bytes for
`{"endpoint": ..., "params": ...}`. The cache file `<key>.json` contains the
request, raw UTF-8 API response body and its SHA256, original retrieval timestamp,
User-Agent and selected HTTP headers. Cached request identity, schema, timestamp
and body hash are checked before use. Corrupt/malformed caches are never refreshed
silently. A fresh cache directory is an explicit new acquisition snapshot.

Offline is the default. Cache misses fail without contacting Wikipedia. With live
access enabled, existing entries are still reused; only misses contact the API.
HTTP-200 missing-page and malformed responses are retained for reproducible
diagnosis when the body is valid UTF-8. Invalid UTF-8/oversized responses stop
the run as transport failures rather than being truncated into cache entries.
Transient API load/rate-limit responses are retained in the failure
report rather than installed as reusable cache entries. Successful earlier
responses survive interruptions. Cached bytes/hashes are integrity records, not
proof of authenticity against deliberate rewriting.

For output `corpus_v1.jsonl`, a successful run produces:

1. `corpus_v1.jsonl`: ordinary RAG-01 `EvidenceDocument` records. Source is
   `encyclopedia`, revision IDs are strings, and the source URL is pinned with
   `oldid=<revision_id>`. Text and content hashes are unchanged source wikitext.
2. `corpus_v1.jsonl.manifest.json`: the unchanged RAG-01 manifest.
3. `corpus_v1.jsonl.collection.json`: acquisition report, including original-list
   hash, selected prefix, query policy, collector source hash, cache-file hashes,
   page IDs, article titles, source/revision IDs, revision timestamps, original
   retrieval timestamps, languages, redirects/normalizations, cutoff checks,
   history attribution links and API licensing declarations, joined by `doc_id`.

A failed collection produces an `incomplete` (or `interrupted`) acquisition
report, with per-request errors/unattempted statuses, and keeps cached responses.
It does not publish an intentionally partial corpus. CLI exits are 0 for a complete
selection, 1 for an incomplete/interrupted collection, and 2 for invalid setup.
Use new output paths for every attempt; all final artifacts refuse overwrites.

The acquisition report is published last. A usable Wikipedia snapshot requires
all three artifacts, `status=complete`, matching corpus/manifest hashes, and the
referenced cache archive. Publication is atomic per file using RAG-01's hard-link
writer, not a transaction across files. Filesystem errors or interruption during
publication may leave orphan/incomplete artifacts; do not treat these as a
complete collection. Prefer local/HPC filesystems supporting hard links over
Google Drive FUSE. Store corpora/caches under an appropriate ignored external-data
directory, outside benchmark, `results` and `experiments` paths.

Replaying the same list, selection/configuration, implementation and immutable
cache produces byte-identical corpus, manifest and collection report at new output
paths. Replay retains **original acquisition timestamps**, not replay time.
Fresh live acquisition is not byte-reproducible: even unchanged article text may
have new HTTP metadata and retrieval times. Archive the list, cache, all three
outputs, source commit and resolution notes together. Do not rely on live APIs
to reproduce an experiment after freezing its corpus.

## Attribution, licensing and time

The API's [site-wide rights information](https://www.mediawiki.org/wiki/API:Siteinfo#Rightsinfo)
is preserved verbatim, rather than hardcoding a license inferred from language
or a modern website footer. It is the site's declaration at acquisition,
**not a per-article/per-revision rights audit**. Preserve the original article
revision and history links for attribution, and check article-specific notices,
third-party inclusions and redistribution requirements before sharing snapshots.
The [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)
describe licensing/attribution obligations. This implementation copies no API
example code and downloads no images/media or external linked sources.

`revision_timestamp` says when Wikipedia recorded that revision, and
`retrieved_at` says when this response was acquired. Neither establishes when
every contained fact was true. `cutoff_satisfied=true` checks only revision time,
not factual accuracy, historical validity, completeness or absence of benchmark
leakage. An older source may be wrong or discuss earlier/later events. Deleted,
suppressed, moved and newly created pages can make historical collection fail.

## Proposed pilot — not executed

Prepare an identity-only roster with subject and relation, excluding answer
fields. Before any download, freeze a deterministic selection: sort exact subject
names by UTF-8 bytes within each relation and select two distinct subject identities
for each of the following six relation types (nominally **12 articles**, cap 20).
Resolve only these subjects to Wikipedia titles/page IDs without consulting gold
objects, predictions or evaluation results. Record unresolved/overlapping page
identities before acquisition; use one declared page per unique subject and log
shared coverage rather than submitting duplicate pages.

| Relation | Subject pages to review; no object-driven discovery |
|---|---|
| `countryLandBordersCountry` | Two country/territory subject articles; scope of land-border statements. |
| `personHasCityOfDeath` | Two person subject articles; identity, biography and temporal ambiguity. |
| `hasCapacity` | Two venue subject articles; infobox/table wikitext and capacity configuration. |
| `hasArea` | Two geographic subject articles; scope and units of reported area. |
| `awardWonBy` | Two award subject articles; source length, tables and unexpanded winner templates. |
| `companyTradesAtStockExchange` | Two issuer subject articles; identity, listing/delisting date scope. |

This pilot checks acquisition/provenance and text usability, **not answer recall
or benchmark performance**. Do not change its list based on gold coverage. Report
missing/disambiguation/redirect/history failures; any later list correction is
a new documented input snapshot. Apply the same frozen subject-only acquisition
protocol and corpus to Standard RAG/RAVEC/learned-revision comparisons.

Future pilot command, only after explicit authorization and list review:

```sh
cd ~/LLM-KBC-Thesis
mkdir -p data/evidence/wiki_pilot
: "${WIKIMEDIA_USER_AGENT:?Set a real project/version (contact URL/email/wiki user)}"
python -m llm_kbc.wikipedia_collect \
  --input /absolute/path/to/reviewed_subject_articles.jsonl \
  --output data/evidence/wiki_pilot/corpus_v1.jsonl \
  --cache-dir data/evidence/wiki_pilot/cache_v1 \
  --language en --max-articles 20 --cutoff 2026-07-01T00:00:00Z \
  --allow-network --user-agent "$WIKIMEDIA_USER_AGENT"
```

The shown whole-second UTC cutoff is a **proposed operational boundary**, not a
verified organizer specification of the time within 1 July. Approve/freeze that
boundary before acquisition; there is no implicit cutoff default. For a 10-article
smoke subset use `--max-articles 10` and new output paths. No command above was run.

Offline replay of the same completed 20-cap selection, with no network flag:

```sh
python -m llm_kbc.wikipedia_collect \
  --input /absolute/path/to/reviewed_subject_articles.jsonl \
  --output data/evidence/wiki_pilot/corpus_v1_replay.jsonl \
  --cache-dir data/evidence/wiki_pilot/cache_v1 \
  --language en --max-articles 20 --cutoff 2026-07-01T00:00:00Z
```

No new runtime dependency is required. Run mocked tests with
`python -m pytest tests/test_wikipedia_collect.py -q`. Wikitext-to-text processing,
template dependency snapshots, robust historical disambiguation, streaming/full
Wikipedia dumps and large-scale collection policies remain out of scope.
