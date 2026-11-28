"""Offline query-local retrieval over a fully validated historical assembly."""

import argparse
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import bm25_retrieval as bm25
from . import evidence_corpus as corpus
from . import rag_corpus_assembly as assembly
from . import rag_query
from . import wikipedia_discovery as discovery
from .wikipedia_collect import _json_object


VERSION = "rag-retrieval-v1"
POLICY = "query-local-bm25-v1"
ELIGIBILITY_POLICY = "validated-query-eligible-documents-v1"
STATUSES = ("unresolved_discovery", "no_usable_evidence", "no_hits", "retrieved")


class RetrievalError(ValueError):
    """Invalid, incomplete or inconsistent evidence/retrieval artifacts."""


@dataclass(frozen=True)
class EvidencePool:
    """Retriever-independent eligibility snapshot, ordered by passage ID.

    Nested records retain the validated upstream schema. Consumers must not
    mutate them. Pool fingerprints cover full passages and source attribution,
    independently of query text, ranking algorithm and downstream model.
    """

    input_index: int
    query_provenance: dict
    availability: dict
    passages: tuple[dict, ...]
    documents: tuple[dict, ...]
    evidence_pool_sha256: str


def _configuration(top_k):
    if type(top_k) is not int or top_k < 1:
        raise RetrievalError("top_k must be a positive integer")
    # The unchanged engine supports fixed k1/b only. Never imply tunable values.
    return {"top_k": top_k, "k1": bm25.K1, "b": bm25.B,
        "idf": "log(1+(N-df+0.5)/(df+0.5))", "query_term_frequency": "binary",
        "indexed_field": "text", "tie_break": "passage_id_ascending"}


def _fingerprints(paths):
    return {key: {"sha256": corpus._sha256(data), "bytes": len(data)}
        for key, path in paths.items() for data in [corpus._input_bytes(path)]}


def _query(record, legacy):
    if legacy:
        query = rag_query.QueryInput(record["SubjectEntity"], record["Relation"],
            relation_namespace=rag_query.LM_KBC_2026_NAMESPACE)
    else:
        query = rag_query.validate_query_input({key: record[key] for key in (
            "subject", "relation_id", "relation_namespace", "relation_description")})
    provenance = rag_query.build_query(query)
    if legacy:
        consistent = record["query"] == provenance["query"]
    else:
        consistent = (corpus._json_bytes(record["query_provenance"]) == corpus._json_bytes(provenance)
            and record["search_expression"] == provenance["query"])
    if not consistent:
        raise RetrievalError("Discovery query differs from the supported generic query policy")
    return provenance


def load_evidence_pools(*, assembly_dir, discovery_path, discovery_cache_dir,
                        store_dir=None, revision_cache_dir=None):
    """Return (ordered pools, provenance) after complete offline assembly validation.

    This is the eligibility boundary for subsequent retrievers as well as BM25.
    No standalone passage/availability artifact can bypass load_assembly().
    """
    root, source = Path(assembly_dir), Path(discovery_path)
    paths = {"assembly/" + name: root / name for name in (*assembly.OUTPUT_FILES, "manifest.json")}
    paths.update({"discovery/" + name: Path(str(source) + suffix) for name, suffix in (
        ("records", ""), ("articles", ".articles.jsonl"), ("manifest", ".manifest.json"))})
    before = _fingerprints(paths)
    passages, manifest = assembly.load_assembly(root, discovery_path=source,
        discovery_cache_dir=discovery_cache_dir, store_dir=store_dir, revision_cache_dir=revision_cache_dir)
    query_data = corpus._input_bytes(root / "query_articles.jsonl")
    availability_data = corpus._input_bytes(root / "query_availability.jsonl")
    for name, data in (("query_articles.jsonl", query_data), ("query_availability.jsonl", availability_data)):
        if before["assembly/" + name] != {"sha256": corpus._sha256(data), "bytes": len(data)}:
            raise RetrievalError("Assembly inputs changed while constructing evidence pools")
    records = bm25._rows(query_data, "validated assembly queries")
    availability = bm25._rows(availability_data, "validated assembly availability")
    if len(records) != len(availability) or len(records) != manifest["query_count"]:
        raise RetrievalError("Incomplete query availability coverage")
    by_document = defaultdict(list)
    for passage in passages:
        by_document[passage["doc_id"]].append(passage)
    parents = {entry["document"]["doc_id"]: {"doc_id": entry["document"]["doc_id"],
        "page_id": entry["page_id"], "license": entry["license"],
        "attribution_history_url": entry["attribution_history_url"],
        "parent_content_sha256": entry["document"]["content_sha256"]}
        for entry in manifest["articles"] if "document" in entry}
    pools = []
    for ordinal, (record, available) in enumerate(zip(records, availability, strict=True), 1):
        if (record["input_index"] != ordinal or available["input_index"] != ordinal
                or available["query_record_sha256"] != corpus._sha256(corpus._json_bytes(record))):
            raise RetrievalError("Query order/identity provenance mismatch")
        documents = available["eligible_document_ids"]
        if len(set(documents)) != len(documents) or any(doc not in parents for doc in documents):
            raise RetrievalError("Invalid eligible document identities")
        subset = tuple(sorted((p for doc in documents for p in by_document[doc]), key=lambda p: p["passage_id"]))
        if len(subset) != available["eligible_passage_count"]:
            raise RetrievalError("Eligible passage count differs from the validated query evidence pool")
        provenance = _query(record, manifest["discovery_policy"] == discovery.POLICY_VERSION)
        attribution = tuple(parents[doc] for doc in sorted(documents))
        fingerprint = corpus._sha256(corpus._json_bytes({"policy": ELIGIBILITY_POLICY,
            "documents": attribution, "passages": subset}))
        pools.append(EvidencePool(ordinal, provenance, available, subset, attribution, fingerprint))
    if before != _fingerprints(paths):
        raise RetrievalError("Upstream artifacts changed during validation")
    return pools, {"files": before, "assembly_policy": manifest["schema_version"],
        "historical_cutoff": manifest["configuration"]["cutoff"],
        "discovery_policy": manifest["discovery_policy"], "search_index_scope": manifest["search_index_scope"]}


def rank_pool(pool: EvidencePool, *, top_k=5):
    """Rank only this pool using the existing BM25 engine; never global statistics."""
    _configuration(top_k)
    parents = {doc["doc_id"]: doc for doc in pool.documents}
    hits = []
    for rank, (passage, score) in enumerate(bm25.BM25Index(list(pool.passages)).search(
            pool.query_provenance["query"], top_k), 1):
        hits.append({**passage, **parents[passage["doc_id"]], "rank": rank, "score": score,
            "source": "encyclopedia"})
    availability = pool.availability
    status = ("unresolved_discovery" if availability["status"] == "unresolved_discovery"
        else "no_usable_evidence" if not pool.passages else "retrieved" if hits else "no_hits")
    return {"schema_version": VERSION, "input_index": pool.input_index,
        "query_provenance": pool.query_provenance, "query": pool.query_provenance["query"],
        "query_record_sha256": availability["query_record_sha256"],
        "upstream_availability_status": availability["status"],
        "eligible_document_ids": availability["eligible_document_ids"],
        "eligible_passage_count": len(pool.passages), "evidence_pool_sha256": pool.evidence_pool_sha256,
        "status": status, "hits": hits}


def _implementations():
    return {Path(module.__file__).name: corpus._sha256(Path(module.__file__).read_bytes())
        for module in (bm25, rag_query, assembly, corpus)} | {
            Path(__file__).name: corpus._sha256(Path(__file__).read_bytes())}


def _manifest(records, upstream, data, configuration, implementations, python_version):
    counts = Counter(record["status"] for record in records)
    return {"schema_version": VERSION, "status": "complete", "retrieval_policy": POLICY,
        "artifact_role": "ranked_candidates_not_packed_contexts", "eligibility_policy": ELIGIBILITY_POLICY,
        "statistics_scope": "query_eligible_passages_only", "configuration": configuration,
        "query_policy_version": rag_query.QUERY_POLICY_VERSION,
        "tokenization_version": bm25.TOKENIZATION_VERSION, "unicode_version": unicodedata.unidata_version,
        "python_version": python_version, "inputs": upstream,
        "ordered_queries": [{"input_index": record["input_index"],
            **{key: record["query_provenance"][key] for key in (
                "identity_sha256", "task_definition_sha256", "query_sha256")},
            **{key: record[key] for key in ("query_record_sha256", "evidence_pool_sha256")}} for record in records],
        "query_count": len(records), "record_count": len(records),
        "outcome_counts": {key: counts[key] for key in STATUSES},
        "availability_counts": dict(sorted(Counter(r["upstream_availability_status"] for r in records).items())),
        "implementation_sha256": implementations,
        "output": {"sha256": corpus._sha256(data), "bytes": len(data)}}


def retrieve(*, assembly_dir, discovery_path, discovery_cache_dir, output_path,
             store_dir=None, revision_cache_dir=None, top_k=5):
    """Validate, rank and revalidate inputs; publish the completion marker last."""
    configuration = _configuration(top_k)
    output = Path(output_path)
    targets = (output, Path(str(output) + ".manifest.json"))
    for target in targets:
        corpus._check_location(target)
        if target.exists() or target.is_symlink():
            raise RetrievalError(f"Refusing to overwrite {target}")
    if not output.parent.is_dir():
        raise RetrievalError("Output parent directory must exist")
    options = dict(assembly_dir=assembly_dir, discovery_path=discovery_path,
        discovery_cache_dir=discovery_cache_dir, store_dir=store_dir, revision_cache_dir=revision_cache_dir)
    pools, upstream = load_evidence_pools(**options)
    records = [rank_pool(pool, top_k=top_k) for pool in pools]
    data = b"".join(corpus._json_bytes(record) for record in records)
    manifest = _manifest(records, upstream, data, configuration, _implementations(), sys.version.split()[0])
    checked, inputs = load_evidence_pools(**options)
    if inputs != upstream or [p.evidence_pool_sha256 for p in checked] != [p.evidence_pool_sha256 for p in pools]:
        raise RetrievalError("Evidence changed before publication")
    corpus._publish(targets[0], data)
    corpus._publish(targets[1], corpus._json_bytes(manifest))
    return manifest


def load_retrieval(output_path, *, assembly_dir, discovery_path, discovery_cache_dir,
                   store_dir=None, revision_cache_dir=None):
    """Revalidate upstream evidence, eligibility, exact rankings and full scores.

    A JSONL without its complete matching manifest is an orphan, never a valid
    retrieval. Archived implementation hashes are provenance, not authentication.
    """
    path = Path(output_path)
    manifest_data = corpus._input_bytes(Path(str(path) + ".manifest.json"))
    manifest = _json_object(manifest_data, "Retrieval completion manifest")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise RetrievalError("Invalid retrieval configuration")
    expected_config = _configuration(configuration.get("top_k"))
    implementations = manifest.get("implementation_sha256")
    python_version = manifest.get("python_version")
    if (not isinstance(implementations, dict) or set(implementations) != set(_implementations())
            or any(not bm25._hash(v) for v in implementations.values())
            or not isinstance(python_version, str) or re.fullmatch(r"\d+\.\d+\.\d+", python_version) is None):
        raise RetrievalError("Invalid implementation/environment provenance")
    pools, upstream = load_evidence_pools(assembly_dir=assembly_dir, discovery_path=discovery_path,
        discovery_cache_dir=discovery_cache_dir, store_dir=store_dir, revision_cache_dir=revision_cache_dir)
    records = [rank_pool(pool, top_k=expected_config["top_k"]) for pool in pools]
    data = b"".join(corpus._json_bytes(record) for record in records)
    expected = _manifest(records, upstream, data, expected_config, implementations, python_version)
    if (corpus._input_bytes(path) != data or manifest_data != corpus._json_bytes(expected)):
        raise RetrievalError("Incomplete, unsupported or inconsistent retrieval artifacts/rankings")
    return records, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", required=True, type=Path)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--discovery-cache-dir", required=True, type=Path)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--revision-cache-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = retrieve(assembly_dir=args.assembly, discovery_path=args.discovery,
            discovery_cache_dir=args.discovery_cache_dir, store_dir=args.store_dir,
            revision_cache_dir=args.revision_cache_dir, output_path=args.output, top_k=args.top_k)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Query-local retrieval failed: {error}\n")
    print(f"Queries: {result['query_count']}; outcomes: {result['outcome_counts']}; SHA256: {result['output']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
