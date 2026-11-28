# RAG-04.5: query-specific BM25 over dynamic evidence

`llm_kbc.rag_retrieval` is a CPU-only, offline integration layer. It reads a
completed historical assembly, not an arbitrary passage file. It does not discover
articles, acquire evidence, read gold labels, generate answers or pack contexts.
Existing standalone BM25, discovery, assembly and baseline behavior is unchanged.

## Interface

```sh
python -B -m llm_kbc.rag_retrieval \
  --assembly /archive/completed_assembly \
  --discovery /archive/discovery_5.jsonl \
  --discovery-cache-dir /archive/discovery_cache \
  --store-dir /archive/article_store \
  --revision-cache-dir /archive/revision_cache \
  --output /scratch/project/new_retrieval.jsonl --top-k 5
```

The output parent must exist. Neither the JSONL nor its `.manifest.json` sidecar
may already exist, including dangling symlinks. Store/cache defaults match the
assembly reader, but explicit paths are recommended for relocated archives.
No network option exists. No dependency is added; validation uses the existing
readers and does not import the wikitext parser.

Python APIs:

- `load_evidence_pools(assembly_dir=..., discovery_path=...,
  discovery_cache_dir=..., store_dir=..., revision_cache_dir=...)` returns ordered
  `EvidencePool` instances and upstream provenance after `load_assembly()`.
- `rank_pool(pool, top_k=5)` delegates exclusively to `BM25Index`.
- `retrieve(..., output_path=..., top_k=5)` validates, ranks, revalidates upstream
  evidence and publishes a completion marker last.
- `load_retrieval(output_path, assembly_dir=..., discovery_path=...,
  discovery_cache_dir=..., store_dir=..., revision_cache_dir=...)` validates the
  artifact against upstream evidence and recomputes exact ranked hits and scores.

`EvidencePool` is the reusable eligibility boundary for later dense/hybrid
experiments. It contains query provenance, upstream availability, passages sorted
by ID, document attribution and an evidence-pool fingerprint. Consumers must not
mutate its nested dictionaries. No alternate retriever is implemented here.

## Eligibility, queries and statistics

The complete upstream bundle is validated before constructing pools. Each pool
contains exactly the passages whose document IDs occur in that query's validated
`eligible_document_ids`. Its size must equal `eligible_passage_count`. Distinct
sources with identical text remain distinct, with their own page/revision IDs.
Shared articles can appear in several pools without another acquisition.

Discovery v2 query inputs are passed through `rag_query.build_query()`, and the
complete resulting provenance must match the archived record and search text.
Discovery v1 identities receive the explicit `lm-kbc-2026` namespace; all six
legacy query texts must match exactly. Newly derived generic fingerprints for v1
are distinguished from the preserved original `query_record_sha256`. Namespaces,
descriptions, original subject spelling and query order survive the integration.
No second relation-template registry is introduced.

Every pool gets its own BM25 index. N, average tokenized passage length, document
frequencies and IDF come only from that pool. Unrelated query/article additions
cannot affect an existing query's pool or scores. The unchanged engine uses:

- `k1=1.2`, `b=0.75`, positive IDF `log(1+(N-df+0.5)/(df+0.5))`.
- Text only; binary query-term frequency; deterministic sorted query terms.
- NFKC, casefold, maximal Unicode alphanumeric runs. Punctuation/underscores and
  remaining combining marks delimit tokens. Decimal punctuation splits numbers;
  no numeric normalization, stemming or stopwords are added.
- Descending score, then ascending passage ID; only positive hits, up to `top_k`.

`top_k` is a positive integer, default 5. The existing engine does **not** support
configurable k1/b: they are recorded and validated as fixed constants. This stage
does not extend the engine or silently accept other values. There is no global
statistics, subject filtering, field weighting or reranking option.

## Artifact contract and publication

Each `rag-retrieval-v1` record stores input index, full generic query provenance,
query text, original query-record fingerprint, upstream availability status,
eligible document IDs/count, pool SHA256, retrieval status and ordered hits. Hits
contain the full passage record, rank, unrounded score, page ID, license,
attribution/history URL and parent content hash. Revision IDs/times, retrieval time,
section paths and source spans remain those of the validated historical passages.

Retrieval statuses are separate from upstream availability:

| Retrieval status | Meaning |
|---|---|
| `unresolved_discovery` | No candidate article was resolved. |
| `no_usable_evidence` | Candidates existed but no eligible passage exists. |
| `no_hits` | A nonempty eligible pool has no positive lexical match. |
| `retrieved` | At least one positive hit. |

All queries remain present; the first three outcomes have empty hit lists. These
are retrieval diagnostics, not model abstention or scoring predictions.

Pool hashes use canonical UTF-8 JSON (sorted keys, compact separators, LF,
unescaped Unicode, no nonfinite numbers): `{policy, documents, passages}`.
Documents are sorted by document ID and passages by passage ID. Full source
attribution/text are covered. Query identity, ranking configuration and unrelated
corpus contents do not enter the pool hash, so competing retrievers can verify
identical evidence eligibility. Each record separately preserves query identity.

The completion sidecar records schema/policy versions, explicit query-local
statistics, BM25 configuration, query/tokenization policies, Python/Unicode
versions, assembly and discovery fingerprints, ordered query/pool fingerprints,
outcome/availability counts, output bytes/hash and implementation hashes. Paths,
wall time and RSS are excluded. Upstream fingerprints include the assembly
completion marker and all aggregate artifacts plus discovery records/inventory/
manifest. The assembly reader verifies their links through the article store and
original API caches. Acquisition/reuse histories therefore remain auditable.

The writer validates upstream inputs again before publication. Exclusive hard-link
publication prevents replacement. This is not a multi-file transaction or store
lock: interruption may leave an orphan JSONL. Consumers must call the reader,
which rejects missing/inconsistent manifests. Preserve orphans and use a fresh
output path. Invalid upstream input produces no retrieval output.

The reader reconstructs the complete expected records and manifest, including
scores, rankings, provenance, configurations and counts; simply recomputing output
hashes does not authorize inconsistent records. Supported archived implementation
hashes need not equal today's source hashes. Hashes are unsigned consistency
checks, not authenticity: consistently forged acquisition history/code claims
require a trusted external archive to detect. Recorded Python versions are
descriptive; Unicode/tokenization policy must match the reader. Exact floating
point replay across different math-library/platform versions is not guaranteed;
a score discrepancy fails verification rather than being rounded away.

## Proposed Aristotle pilot (after audit and commit; not executed)

The researcher reports an 18-query/82-article/2,430-passage assembly. These artifacts
were not downloaded or verified locally. Set the three actual assembly locations;
the completed assembly/store/cache paths have not been supplied in this checkout.
Do not reuse or overwrite earlier pilot outputs.

```sh
set -eu
cd "$HOME/LLM-KBC-Thesis"
source /scratch/m/mbatsir/llm-kbc/venv-benchmark-modern/bin/activate
export ASSEMBLY_DIR=/scratch/m/mbatsir/llm-kbc/REPLACE_WITH_COMPLETED_ASSEMBLY
export ARTICLE_STORE=/scratch/m/mbatsir/llm-kbc/REPLACE_WITH_ARTICLE_STORE
export REVISION_CACHE=/scratch/m/mbatsir/llm-kbc/REPLACE_WITH_REVISION_CACHE
export DISCOVERY=/scratch/m/mbatsir/llm-kbc/discovery_pilot_v2_5_audit_01/discovery_5.jsonl
export DISCOVERY_CACHE="$HOME/LLM-KBC-Thesis/data/evidence/discovery_pilot_v1/cache"
export RETRIEVAL_RUN=/scratch/m/mbatsir/llm-kbc/rag_retrieval_pilot_01
test -f "$ASSEMBLY_DIR/manifest.json"
mkdir "$RETRIEVAL_RUN"
/usr/bin/time -v -o "$RETRIEVAL_RUN/runtime.txt" \
  python -B -m llm_kbc.rag_retrieval \
    --assembly "$ASSEMBLY_DIR" --discovery "$DISCOVERY" \
    --discovery-cache-dir "$DISCOVERY_CACHE" --store-dir "$ARTICLE_STORE" \
    --revision-cache-dir "$REVISION_CACHE" --top-k 5 \
    --output "$RETRIEVAL_RUN/retrieval.jsonl"
python -B - <<'PY'
import os
from llm_kbc.rag_retrieval import load_retrieval
rows, manifest = load_retrieval(os.environ['RETRIEVAL_RUN'] + '/retrieval.jsonl',
    assembly_dir=os.environ['ASSEMBLY_DIR'], discovery_path=os.environ['DISCOVERY'],
    discovery_cache_dir=os.environ['DISCOVERY_CACHE'], store_dir=os.environ['ARTICLE_STORE'],
    revision_cache_dir=os.environ['REVISION_CACHE'])
assert len(rows) == 18
print(manifest['outcome_counts'], manifest['output'])
PY
du -sh "$RETRIEVAL_RUN"
```

Before treating this as the reported pilot, independently check the supplied
archive fingerprints: corpus `65a37c8c9ff3478e897f341837f102e5e16f39b2cd79509d26fcc1768f61f297`,
passages `ed58034603adebd0130560e60831031776f7052641b177db4e6c9c073ceb2794`,
availability `df1808a8a33675d31b89ac3dee17ea7e620e3d425f5271a61a2c5db039495d20`.
Re-run at a separate output path with the same arguments and compare both JSONL
and manifest with `cmp`. Keep profiling outputs separate and archive inputs,
caches, stores, code and retrieval artifacts outside temporary scratch.

## Limitations

All passages and query pools are held in memory; eligible subsets are reindexed
per query. Full upstream validation is intentionally repeated before publication
and on read. Profile pilot runtime/RSS before scaling. There is no persistent
index, concurrency control or filesystem durability guarantee. Historical content
cutoff does not make current-index article discovery historical. No retrieval
effectiveness, multi-model benefit or external-dataset generalization is established
by synthetic correctness tests. Final tokenizer-aware shared context packing and
retrieval-quality evaluation remain separate work.
