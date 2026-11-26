"""Assemble verified historical articles and passages; offline unless authorized.

Discovery, acquisition, cleaning and passage validation remain owned by their
existing modules. The shared store is immutable; only a final assembly manifest
marks the complete query-driven snapshot. No retrieval or gold-label loading.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

from . import bm25_retrieval as validation
from . import evidence_corpus as corpus
from . import wikipedia_collect as collection
from . import wikipedia_discovery as discovery
from . import wikipedia_discovery_v2 as discovery_v2
from . import wikipedia_preprocess as preprocessing


VERSION = "rag-corpus-assembly-v2"
OUTCOME_POLICY = "historical-evidence-availability-v1"
OUTPUT_FILES = ("query_articles.jsonl", "query_availability.jsonl", "articles.jsonl", "corpus.jsonl",
                "corpus.jsonl.manifest.json", "corpus.jsonl.collection.json",
                "passages.jsonl", "passages.jsonl.manifest.json", "passages.jsonl.diagnostics.jsonl")


class AssemblyError(ValueError):
    """Incomplete or incompatible evidence must not become a completed assembly."""


def _fingerprints(paths):
    return {key: {"sha256": corpus._sha256(data), "bytes": len(data)}
            for key, path in paths.items() for data in [corpus._input_bytes(Path(path))]}


def _code_hashes():
    return {Path(module.__file__).name: corpus._sha256(Path(module.__file__).read_bytes())
            for module in (discovery, discovery_v2, collection, preprocessing, validation, corpus)} | {
                Path(__file__).name: corpus._sha256(Path(__file__).read_bytes())}


def _discovery(path, cache):
    path = Path(path)
    paths = {name: Path(str(path) + suffix) for name, suffix in (
        ("discovery", ""), ("articles", ".articles.jsonl"), ("manifest", ".manifest.json"))}
    before = _fingerprints(paths)
    records, articles, manifest = discovery.load_discovery(path, cache_dir=cache)
    if _fingerprints(paths) != before:
        raise AssemblyError("Discovery changed during validation")
    return records, articles, manifest, before


def _configuration(manifest, cutoff, revalidate_cutoff, chunk_words, overlap_words):
    if type(revalidate_cutoff) is not bool:
        raise AssemblyError("revalidate_cutoff must be boolean")
    original = (manifest["historical_revision_cutoff"] if manifest["schema_version"] == discovery.POLICY_VERSION
                else manifest["configuration"]["historical_cutoff"])
    cutoff = original if cutoff is None else cutoff
    discovery_v2.DiscoveryConfig(historical_cutoff=cutoff)
    if cutoff != original and not revalidate_cutoff:
        raise AssemblyError("Cutoff differs from discovery; explicit --revalidate-cutoff is required")
    preprocessing._configuration(chunk_words, overlap_words)
    return {"cutoff": cutoff, "revalidate_cutoff": revalidate_cutoff,
            "chunk_words": chunk_words, "overlap_words": overlap_words}


def _store_paths(store, page_id, config):
    key = corpus._sha256(corpus._json_bytes({"page_id": page_id, "cutoff": config["cutoff"],
        "language": "en", "collector_policy": collection.POLICY_VERSION}))
    pp_key = corpus._sha256(corpus._json_bytes(preprocessing._configuration(
        config["chunk_words"], config["overlap_words"])))
    directory = store / key
    # Short locator avoids Windows path limits. Full configuration and content
    # fingerprints are verified on every read; a locator collision fails closed.
    return key, directory, directory / "corpus.jsonl", directory / f"p-{pp_key[:24]}.jsonl"


def _snapshot(path, cache, cutoff, *, allow_duplicate_content=False):
    documents, metadata, fingerprints = preprocessing.load_snapshot(path, cache_dir=cache,
        allow_duplicate_content=allow_duplicate_content)
    report = collection._json_object(corpus._input_bytes(Path(str(path) + ".collection.json")), "Collection report")
    if report.get("cutoff") != cutoff or report.get("endpoint") != discovery.ENDPOINT:
        raise AssemblyError("Historical snapshot cutoff/edition mismatch")
    if allow_duplicate_content:
        data = corpus._input_bytes(path)
        if corpus._input_bytes(Path(str(path) + ".manifest.json")) != corpus._json_bytes(
                _corpus_manifest(documents, data, assembled=True)):
            raise AssemblyError("Imported assembly corpus manifest semantics mismatch")
    else:
        manifest = collection._json_object(corpus._input_bytes(Path(str(path) + ".manifest.json")), "Source corpus manifest")
        for key, value in (("id_policy", "sha256-canonical-source-url-language-revision-content-v1"),
                ("duplicate_policy", "reject-identical-doc-id-or-text-sha256"),
                ("ingestion_policy", "external-text-only-no-benchmark-artifacts-v1")):
            if manifest.get(key) != value:
                raise AssemblyError("Unsupported source corpus policy")
    return documents, metadata, fingerprints


def _passages(path, source, cache, config=None, *, allow_duplicate_content=False):
    rows, manifest, _, fingerprints = validation.load_passages(path, source, cache_dir=cache,
        allow_duplicate_content=allow_duplicate_content)
    if config is not None and (manifest["configuration"]["chunk_words"] != config["chunk_words"]
            or manifest["configuration"]["overlap_words"] != config["overlap_words"]):
        raise AssemblyError("Preprocessing configuration mismatch")
    diagnostics = validation._rows(corpus._input_bytes(Path(str(path) + ".diagnostics.jsonl")), "Diagnostics")
    return rows, diagnostics, fingerprints


def _imports(snapshots, config):
    """Explicit archive triples: (corpus, passages or None/'-', original cache)."""
    result = {}
    for source, passages, cache in snapshots:
        source, cache = Path(source), Path(cache)
        source_manifest = collection._json_object(corpus._input_bytes(Path(str(source) + ".manifest.json")), "Reuse corpus manifest")
        assembled = source_manifest.get("duplicate_policy") == corpus.ASSEMBLY_DUPLICATE_POLICY
        documents, metadata, fingerprints = _snapshot(source, cache, config["cutoff"], allow_duplicate_content=assembled)
        rows, diagnostics = None, None
        provenance = {"snapshot_inputs": fingerprints}
        if passages is not None and str(passages) != "-":
            rows, diagnostics, pp_inputs = _passages(Path(passages), source, cache, allow_duplicate_content=assembled)
            pp_manifest = collection._json_object(corpus._input_bytes(Path(str(passages) + ".manifest.json")), "Reuse passage manifest")
            provenance.update(passage_inputs=pp_inputs, preprocessing_code_sha256=pp_manifest["code_sha256"])
            if any(pp_manifest["configuration"][k] != config[k] for k in ("chunk_words", "overlap_words")):
                rows, diagnostics = None, None  # Valid archive; generate the requested new configuration.
        for document in documents:
            record = metadata[document.doc_id]
            page_id = record["page_id"]
            if page_id in result:
                raise AssemblyError(f"Ambiguous reuse sources for page ID {page_id}; supply one archive")
            result[page_id] = (document, record, provenance, cache,
                None if rows is None else [r for r in rows if r["doc_id"] == document.doc_id],
                None if diagnostics is None else next(d for d in diagnostics if d["doc_id"] == document.doc_id))
    return result


def _collection_report(documents, metadata, cutoff, source_fingerprints, corpus_hash, manifest_hash):
    records = [{**metadata[d.doc_id], "input_line": index} for index, d in enumerate(documents, 1)]
    return {"policy_version": collection.POLICY_VERSION, "status": "complete",
        "artifact_role": "assembled_index_of_verified_acquisitions", "assembly_policy": VERSION,
        "endpoint": discovery.ENDPOINT, "cutoff": cutoff, "text_representation": "unexpanded-mediawiki-wikitext",
        "input_records": len(records), "selected_records": len(records), "collected_records": len(records),
        "input_sha256": corpus._sha256(b"".join(corpus._json_bytes(r["request"]) for r in records)),
        "corpus_sha256": corpus_hash, "manifest_sha256": manifest_hash,
        "records": records, "source_snapshots": source_fingerprints}


def _corpus_manifest(documents, data, *, assembled):
    result = {"schema_version": corpus.SCHEMA_VERSION, "document_count": len(documents),
        "input_sha256": corpus._sha256(data), "corpus_sha256": corpus._sha256(data),
        "id_policy": "sha256-canonical-source-url-language-revision-content-v1",
        "duplicate_policy": (corpus.ASSEMBLY_DUPLICATE_POLICY if assembled else "reject-identical-doc-id-or-text-sha256"),
        "ingestion_policy": "external-text-only-no-benchmark-artifacts-v1",
        "documents": [{k: v for k, v in asdict(d).items() if k != "text"} for d in documents]}
    if assembled:
        result.update(artifact_role="assembled_historical_corpus", assembly_policy=VERSION)
    return result


def _write_snapshot(directory, documents, metadata, cutoff, source_fingerprints, *, assembled=False):
    """Compose validated schemas; duplicate text is allowed only in the aggregate."""
    documents = sorted(documents, key=lambda d: d.doc_id)
    path = directory / "corpus.jsonl"
    data = b"".join(corpus.serialize_document(d) for d in documents)
    corpus._documents(data, str(path), allow_duplicate_content=assembled)
    manifest = _corpus_manifest(documents, data, assembled=assembled)
    corpus._publish(path, data)
    corpus._publish(Path(str(path) + ".manifest.json"), corpus._json_bytes(manifest))
    report = _collection_report(documents, metadata, cutoff, source_fingerprints, manifest["corpus_sha256"],
        corpus._sha256(corpus._input_bytes(Path(str(path) + ".manifest.json"))))
    corpus._publish(Path(str(path) + ".collection.json"), corpus._json_bytes(report))


def _passage_manifest(documents, metadata, inputs, config, rows, diagnostics, writer_hash):
    data = b"".join(corpus._json_bytes(row) for row in rows)
    diagnostic_data = b"".join(corpus._json_bytes(row) for row in diagnostics)
    return {"schema_version": "evidence-passages-v1", "status": "complete",
        "preprocessing_version": preprocessing.VERSION, "mwparserfromhell": preprocessing.PARSER_VERSION,
        "artifact_role": "assembled_index_of_verified_passages", "assembly_policy": VERSION,
        "code_sha256": writer_hash,
        "configuration": {"chunk_words": config["chunk_words"], "overlap_words": config["overlap_words"],
                          "word_definition": "Unicode non-whitespace runs", "section_local": True},
        "document_count": len(documents), "passage_count": len(rows), "inputs": inputs,
        "passages_sha256": corpus._sha256(data), "diagnostics_sha256": corpus._sha256(diagnostic_data),
        "cache_validation": "verified", "unprocessed_events": sum(d["unprocessed_events"] for d in diagnostics),
        "document_provenance": [{"doc_id": d.doc_id, "page_id": metadata[d.doc_id]["page_id"],
            "license": metadata[d.doc_id]["license"], "attribution_history_url": metadata[d.doc_id]["attribution_history_url"],
            "content_sha256": d.content_sha256} for d in documents]}


def _write_passages(path, source, cache, config, rows, diagnostics, *, assembled=False):
    """Compose validated per-document outputs; never rerun the cleaner here."""
    documents, metadata, inputs = preprocessing.load_snapshot(source, cache_dir=cache,
        allow_duplicate_content=assembled)
    manifest = _passage_manifest(documents, metadata, inputs, config, rows, diagnostics,
        corpus._sha256(Path(__file__).read_bytes()))
    corpus._publish(path, b"".join(corpus._json_bytes(row) for row in rows))
    corpus._publish(Path(str(path) + ".diagnostics.jsonl"), b"".join(corpus._json_bytes(row) for row in diagnostics))
    corpus._publish(Path(str(path) + ".manifest.json"), corpus._json_bytes(manifest))


def _stored_snapshot(source, cache, page_id, config):
    documents, metadata, _ = _snapshot(source, cache, config["cutoff"])
    if len(documents) != 1 or metadata[documents[0].doc_id]["page_id"] != page_id:
        raise AssemblyError("Stored article page identity/coverage mismatch")
    document = documents[0]
    record = metadata[document.doc_id]
    requests = corpus._input_bytes(source.parent / "requests.jsonl")
    report = collection._json_object(corpus._input_bytes(Path(str(source) + ".collection.json")), "Collection report")
    if (collection._requests(requests) != [record["request"]]
            or report.get("input_records") != 1 or report.get("input_sha256") != corpus._sha256(requests)):
        raise AssemblyError("Stored article request provenance mismatch")
    snapshot, reference = collection._snapshot(discovery.ENDPOINT, collection._params(record["request"], config["cutoff"]),
        cache, allow_network=False, user_agent=None, delay=1.0)
    if reference != record["cache"] or collection.article_outcome(snapshot, record["request"], "en", config["cutoff"]) != document:
        raise AssemblyError("Stored article outcome/provenance mismatch")
    return document, record


def _article(source, passages, cache, page_id, config):
    document, record = _stored_snapshot(source, cache, page_id, config)
    rows, diagnostics, _ = _passages(passages, source, cache, config)
    paths = {"requests.jsonl": source.parent / "requests.jsonl",
             **{p.name: p for p in (source, Path(str(source) + ".manifest.json"), Path(str(source) + ".collection.json"),
                  passages, Path(str(passages) + ".manifest.json"), Path(str(passages) + ".diagnostics.jsonl"))}}
    return document, record, rows, diagnostics[0], _fingerprints(paths)


def _entry(page_id, key, passages, bundle, collection_action, preprocessing_action):
    document, metadata, rows, diagnostic, fingerprints = bundle
    return {"page_id": page_id, "store_key": key, "passages_file": passages.name,
        "outcome": "usable" if rows else "zero_passages", "reason": None if rows else "no_usable_passages",
        "collection_action": collection_action, "preprocessing_action": preprocessing_action,
        "document": {k: v for k, v in asdict(document).items() if k != "text"},
        "revision_timestamp": metadata["revision_timestamp"], "cutoff_satisfied": metadata["cutoff_satisfied"],
        "license": metadata["license"], "attribution_history_url": metadata["attribution_history_url"],
        "passage_count": len(rows), "unprocessed_events": diagnostic["unprocessed_events"], "artifacts": fingerprints}


def _terminal_evidence(source, cache, page_id, config):
    """Reclassify immutable raw evidence; report text is never a decision input."""
    request_data = corpus._input_bytes(source.parent / "requests.jsonl")
    requests = collection._requests(request_data)
    if requests != [{"page_id": page_id}]:
        raise AssemblyError("Terminal outcome request identity mismatch")
    snapshot, reference = collection._snapshot(discovery.ENDPOINT, collection._params(requests[0], config["cutoff"]),
        cache, allow_network=False, user_agent=None, delay=1.0)
    report = collection._json_object(corpus._input_bytes(Path(str(source) + ".collection.json")), "Terminal collection report")
    records = report.get("records")
    if (report.get("policy_version") != collection.POLICY_VERSION or report.get("status") != "incomplete"
            or report.get("endpoint") != discovery.ENDPOINT or report.get("cutoff") != config["cutoff"]
            or report.get("text_representation") != "unexpanded-mediawiki-wikitext"
            or report.get("input_sha256") != corpus._sha256(request_data)
            or any(type(report.get(k)) is not int or report[k] != v for k, v in (
                ("input_records", 1), ("selected_records", 1), ("collected_records", 0), ("max_articles", 1)))
            or not validation._hash(report.get("source_sha256"))
            or not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict)):
        raise AssemblyError("Invalid terminal collection report")
    policy = report.get("request_policy")
    if not isinstance(policy, dict):
        raise AssemblyError("Invalid terminal request policy")
    delay = policy.get("minimum_delay_seconds")
    try:
        discovery_v2._network_options(False, None, delay)
    except OverflowError as error:
        raise AssemblyError("Invalid terminal request delay") from error
    if corpus._json_bytes(policy) != corpus._json_bytes({"serial": True, "minimum_delay_seconds": delay,
            "maxlag": 5, "timeout_seconds": 30, "automatic_retries": 0}):
        raise AssemblyError("Invalid terminal request policy")
    record = records[0]
    if (record.get("request") != requests[0] or record.get("status") != "failed"
            or type(record.get("input_line")) is not int or record["input_line"] != 1 or record.get("cache") != reference
            or record.get("retrieved_at") != snapshot["retrieved_at"]
            or record.get("user_agent") != snapshot["user_agent"] or "network_details" in record):
        raise AssemblyError("Terminal collection/cache provenance mismatch")
    try:
        collection.article_outcome(snapshot, requests[0], "en", config["cutoff"])
    except collection.TerminalArticleOutcome as outcome:
        # Reconstruct the native producer's partial metadata at its failure
        # boundary. Missing pages have no page/revision metadata; revision-less
        # pages have no revision metadata. Neither case permits invented nulls.
        expected_metadata = {}
        try:
            collection._document(snapshot, requests[0], "en", config["cutoff"], expected_metadata)
        except (collection.CollectionError, corpus.CorpusError):
            pass
        else:
            raise AssemblyError("Terminal report contradicts successful native collection")
        report_metadata = {k: v for k, v in record.items() if k not in {
            "input_line", "request", "status", "cache", "retrieved_at", "user_agent", "error"}}
        if (corpus._json_bytes(report_metadata) != corpus._json_bytes(expected_metadata)
                or not isinstance(record.get("error"), str) or not record["error"].strip()):
            raise AssemblyError("Terminal collection report metadata differs from verified raw evidence")
        payload = collection._json_object(snapshot["response_body"], "Verified terminal response")["query"]
        page = payload["pages"][0]
        revision = next(iter(page.get("revisions", [])), None)
        return {"schema_version": OUTCOME_POLICY, "page_id": page_id, "cutoff": config["cutoff"],
            "endpoint": discovery.ENDPOINT, "outcome": outcome.status, "reason": outcome.reason,
            "request": snapshot["request"], "retrieved_at": snapshot["retrieved_at"],
            "title": page.get("title"), "license": payload["rightsinfo"],
            "revision_id": str(revision["revid"]) if revision else None,
            "revision_timestamp": revision["timestamp"] if revision else None,
            "cache": reference, "response_sha256": snapshot["response_sha256"]}
    raise AssemblyError("Incomplete collection is not verified terminal unavailability")


def _terminal_entry(source, cache, page_id, config, key, action):
    evidence = _terminal_evidence(source, cache, page_id, config)
    outcome_path = source.parent / "outcome.json"
    if corpus._input_bytes(outcome_path) != corpus._json_bytes(evidence):
        raise AssemblyError("Terminal outcome differs from verified raw evidence")
    if any(p.exists() or p.is_symlink() for p in (source, Path(str(source) + ".manifest.json"))):
        raise AssemblyError("Terminal outcome conflicts with a collected document")
    return {"page_id": page_id, "store_key": key, "outcome": evidence["outcome"], "reason": evidence["reason"],
        "collection_action": action, "preprocessing_action": "not_applicable", "passage_count": 0,
        "outcome_evidence": evidence, "artifacts": _fingerprints({p.name: p for p in (
            source.parent / "requests.jsonl", Path(str(source) + ".collection.json"), outcome_path)})}


def _ensure_article(store, cache, page_id, config, imports, options):
    key, directory, source, passages = _store_paths(store, page_id, config)
    collection_action, preprocessing_action = "reused", "reused"
    imported = imports.get(page_id)
    if directory.is_symlink():
        raise AssemblyError("Article-store slots must not be symlinked directories")
    if not directory.exists():
        directory.mkdir()
        request = {"page_id": page_id} if imported is None else imported[1]["request"]
        corpus._publish(directory / "requests.jsonl", corpus._json_bytes(request))
        if imported is None:
            report = collection.collect_wikipedia(input_path=directory / "requests.jsonl", output_path=source,
                cache_dir=cache, language="en", cutoff=config["cutoff"], max_articles=1, **options)
            if report["status"] == "interrupted":
                raise KeyboardInterrupt
            if report["status"] != "complete":
                # Replay the validated cache offline. Transient/cache/integrity
                # failures cannot reach terminal publication.
                try:
                    evidence = _terminal_evidence(source, cache, page_id, config)
                except (ValueError, OSError) as error:
                    raise AssemblyError(f"Page {page_id}: incomplete collection; inspect {source}.collection.json: {error}") from error
                corpus._publish(directory / "outcome.json", corpus._json_bytes(evidence))
                return None, _terminal_entry(source, cache, page_id, config, key, "new")
            collection_action = "new"
        else:
            document, metadata, fingerprints, original_cache, _, _ = imported
            reference = metadata["cache"]
            data = corpus._input_bytes(original_cache / (reference["key"] + ".json"))
            if corpus._sha256(data) != reference["sha256"]:
                raise AssemblyError("Reuse cache changed before import")
            target = cache / (reference["key"] + ".json")
            if target.exists() or target.is_symlink():
                if corpus._input_bytes(target) != data:
                    raise AssemblyError("Conflicting immutable revision cache snapshots")
            else:
                corpus._publish(target, data)
            _write_snapshot(directory, [document], {document.doc_id: metadata}, config["cutoff"], [fingerprints])
    if (directory / "outcome.json").exists() or (directory / "outcome.json").is_symlink():
        if imported is not None:
            raise AssemblyError("Archive conflicts with an immutable terminal article outcome")
        return None, _terminal_entry(source, cache, page_id, config, key, "reused")
    # Validate even an interrupted/corrupted slot before generating anything else.
    document, _ = _stored_snapshot(source, cache, page_id, config)
    if imported is not None and imported[0] != document:
        raise AssemblyError("Reuse source conflicts with the immutable stored article")
    pp_files = [passages, Path(str(passages) + ".manifest.json"), Path(str(passages) + ".diagnostics.jsonl")]
    if any(p.exists() or p.is_symlink() for p in pp_files):
        if not all(p.is_file() for p in pp_files):
            raise AssemblyError("Incomplete stored preprocessing artifacts")
    elif imported is not None and imported[4] is not None:
        _write_passages(passages, source, cache, config, imported[4], [imported[5]])
    else:
        preprocessing.preprocess_corpus(corpus_path=source, output_path=passages, cache_dir=cache,
            chunk_words=config["chunk_words"], overlap_words=config["overlap_words"])
        preprocessing_action = "new"
    bundle = _article(source, passages, cache, page_id, config)
    return bundle, _entry(page_id, key, passages, bundle, collection_action, preprocessing_action)


def _availability(records, entries):
    """Availability means usable passages, not merely a collected source."""
    by_page = {e["page_id"]: e for e in entries}
    result = []
    for record in records:
        ids = [c["page_id"] for c in record["candidates"]]
        available = [i for i in ids if by_page[i]["outcome"] == "usable"]
        unavailable = [i for i in ids if i not in available]
        status = ("unresolved_discovery" if not ids else "resolved_no_usable_evidence" if not available
                  else "partial_candidate_availability" if unavailable else "full_candidate_availability")
        result.append({"input_index": record["input_index"], "query_record_sha256": corpus._sha256(corpus._json_bytes(record)),
            "status": status, "candidate_page_ids": ids, "available_article_ids": available,
            "unavailable_article_ids": unavailable,
            "eligible_document_ids": [by_page[i]["document"]["doc_id"] for i in available],
            "eligible_passage_count": sum(by_page[i]["passage_count"] for i in available),
            "article_outcomes": [{"page_id": i, "outcome": by_page[i]["outcome"], "reason": by_page[i]["reason"]} for i in ids]})
    return result


def _manifest(output, discovery_manifest, discovery_inputs, config, entries, allow_network, implementations, availability):
    collected = [e for e in entries if e["outcome"] in ("usable", "zero_passages")]
    return {"schema_version": VERSION, "status": "complete", "configuration": config,
        "outcome_policy": OUTCOME_POLICY, "duplicate_content_policy": corpus.ASSEMBLY_DUPLICATE_POLICY,
        "discovery_policy": discovery_manifest["schema_version"], "discovery_inputs": discovery_inputs,
        "discovery_cutoff": (discovery_manifest.get("historical_revision_cutoff")
                             or discovery_manifest["configuration"]["historical_cutoff"]),
        "search_index_scope": discovery_manifest["search_index_scope"],
        "query_count": discovery_manifest["record_count"], "unresolved_query_count": discovery_manifest["unresolved_count"],
        "article_count": len(entries), "selected_article_count": len(entries),
        "document_count": len(collected), "collected_article_count": len(collected),
        "terminal_unavailable_articles": sum(e["outcome"] == "unavailable" for e in entries),
        "policy_excluded_articles": sum(e["outcome"] == "excluded" for e in entries),
        "zero_passage_articles": sum(e["outcome"] == "zero_passages" for e in entries),
        "usable_articles": sum(e["outcome"] == "usable" for e in entries),
        "query_availability_counts": {status: sum(r["status"] == status for r in availability) for status in (
            "unresolved_discovery", "resolved_no_usable_evidence", "partial_candidate_availability", "full_candidate_availability")},
        "passage_count": sum(e["passage_count"] for e in entries),
        "reused_collected_articles": sum(e["collection_action"] == "reused" for e in collected),
        "newly_collected_articles": sum(e["collection_action"] == "new" for e in collected),
        "newly_recorded_terminal_articles": sum(e["collection_action"] == "new" for e in entries if e["outcome"] in ("unavailable", "excluded")),
        "reused_terminal_articles": sum(e["collection_action"] == "reused" for e in entries if e["outcome"] in ("unavailable", "excluded")),
        "reused_preprocessed_articles": sum(e["preprocessing_action"] == "reused" for e in entries),
        "newly_preprocessed_articles": sum(e["preprocessing_action"] == "new" for e in entries),
        "allow_network": allow_network, "collector_policy": collection.POLICY_VERSION,
        "preprocessing_policy": preprocessing.VERSION, "parser_version": preprocessing.PARSER_VERSION,
        "articles": entries, "outputs": _fingerprints({name: output / name for name in OUTPUT_FILES}),
        "implementation_sha256": implementations}


def assemble(*, discovery_path, discovery_cache_dir, output_dir, cutoff=None, store_dir=None,
             revision_cache_dir=None, reuse_snapshots=(), chunk_words=200, overlap_words=40,
             revalidate_cutoff=False, allow_network=False, user_agent=None, request_delay=1.0):
    """Build a fresh assembly; reuse immutable per-article slots and explicit archives."""
    output = Path(output_dir)
    store = Path(store_dir) if store_dir is not None else output.parent / "wikipedia_articles"
    cache = Path(revision_cache_dir) if revision_cache_dir is not None else store / "cache"
    options = discovery_v2._network_options(allow_network, user_agent, request_delay)
    for path in (output, store, cache):
        corpus._check_location(path)
    if output.exists() or output.is_symlink():
        raise AssemblyError("Output directory must be new; no clobber is always enforced")
    if not output.parent.is_dir():
        raise AssemblyError("Output parent directory must exist")
    records, inventory, discovery_manifest, discovery_inputs = _discovery(discovery_path, discovery_cache_dir)
    config = _configuration(discovery_manifest, cutoff, revalidate_cutoff, chunk_words, overlap_words)
    imports = _imports(reuse_snapshots, config)
    output.mkdir()
    entries, bundles = [], []
    current_page = None
    try:
        store.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        for article in inventory:
            current_page = article["page_id"]
            bundle, entry = _ensure_article(store, cache, current_page, config, imports, options)
            if bundle is not None:
                bundles.append(bundle)
            entries.append(entry)
        documents = sorted((b[0] for b in bundles), key=lambda d: d.doc_id)
        by_id = {b[0].doc_id: b for b in bundles}
        corpus._publish(output / "query_articles.jsonl", b"".join(corpus._json_bytes(r) for r in records))
        corpus._publish(output / "articles.jsonl", b"".join(corpus._json_bytes(r) for r in inventory))
        availability = _availability(records, entries)
        corpus._publish(output / "query_availability.jsonl", b"".join(corpus._json_bytes(r) for r in availability))
        _write_snapshot(output, documents, {b[0].doc_id: b[1] for b in bundles}, config["cutoff"],
                        [e["artifacts"] for e in entries], assembled=True)
        _write_passages(output / "passages.jsonl", output / "corpus.jsonl", cache, config,
            [r for d in documents for r in by_id[d.doc_id][2]], [by_id[d.doc_id][3] for d in documents], assembled=True)
        # Validate the composed standard artifact and all referenced store files again.
        validation.load_passages(output / "passages.jsonl", output / "corpus.jsonl", cache_dir=cache, allow_duplicate_content=True)
        if _discovery(discovery_path, discovery_cache_dir)[3] != discovery_inputs:
            raise AssemblyError("Discovery changed during assembly")
        for entry in entries:
            key, directory, source, passages = _store_paths(store, entry["page_id"], config)
            if entry["outcome"] in ("unavailable", "excluded"):
                expected = _terminal_entry(source, cache, entry["page_id"], config, key, entry["collection_action"])
            else:
                checked = _article(source, passages, cache, entry["page_id"], config)
                expected = _entry(entry["page_id"], key, passages, checked, entry["collection_action"], entry["preprocessing_action"])
            if expected != entry:
                raise AssemblyError("Stored article changed during assembly")
        manifest = _manifest(output, discovery_manifest, discovery_inputs, config, entries, allow_network, _code_hashes(), availability)
        corpus._publish(output / "manifest.json", corpus._json_bytes(manifest))
        return manifest
    except (ValueError, OSError, KeyboardInterrupt) as error:
        corpus._publish(output / "failure.json", corpus._json_bytes({"schema_version": VERSION,
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "incomplete",
            "page_id": current_page, "error": str(error) or type(error).__name__, "configuration": config,
            "discovery_inputs": discovery_inputs, "completed_articles": entries}))
        raise


def load_assembly(output_dir, *, discovery_path, discovery_cache_dir, store_dir=None, revision_cache_dir=None):
    """Require the completion marker and verify aggregate-to-store-to-cache links."""
    output = Path(output_dir)
    store = Path(store_dir) if store_dir is not None else output.parent / "wikipedia_articles"
    cache = Path(revision_cache_dir) if revision_cache_dir is not None else store / "cache"
    data = corpus._input_bytes(output / "manifest.json")
    manifest = collection._json_object(data, "Assembly manifest")
    if manifest.get("schema_version") != VERSION or manifest.get("status") != "complete":
        raise AssemblyError("Unsupported or incomplete assembly version")
    implementations = manifest.get("implementation_sha256")
    if (not isinstance(implementations, dict) or set(implementations) != set(_code_hashes())
            or any(not validation._hash(v) for v in implementations.values()) or type(manifest.get("allow_network")) is not bool):
        raise AssemblyError("Invalid assembly implementation/execution provenance")
    records, inventory, original, inputs = _discovery(discovery_path, discovery_cache_dir)
    config = manifest.get("configuration")
    if not isinstance(config, dict) or set(config) != {"cutoff", "revalidate_cutoff", "chunk_words", "overlap_words"}:
        raise AssemblyError("Invalid assembly configuration")
    if corpus._json_bytes(config) != corpus._json_bytes(_configuration(original, **config)):
        raise AssemblyError("Assembly configuration must record the explicit resolved cutoff")
    entries = manifest.get("articles")
    if not isinstance(entries, list) or len(entries) != len(inventory):
        raise AssemblyError("Assembly article coverage mismatch")
    bundles = []
    for article, entry in zip(inventory, entries, strict=True):
        if (not isinstance(entry, dict) or entry.get("collection_action") not in ("new", "reused")
                or entry.get("preprocessing_action") not in ("new", "reused", "not_applicable")):
            raise AssemblyError("Invalid article action/provenance")
        key, directory, source, passages = _store_paths(store, article["page_id"], config)
        if directory.is_symlink():
            raise AssemblyError("Article-store slots must not be symlinked directories")
        if entry.get("outcome") in ("unavailable", "excluded"):
            expected = _terminal_entry(source, cache, article["page_id"], config, key, entry["collection_action"])
        else:
            if entry["preprocessing_action"] == "not_applicable":
                raise AssemblyError("Collected article must record preprocessing")
            bundle = _article(source, passages, cache, article["page_id"], config)
            expected = _entry(article["page_id"], key, passages, bundle, entry["collection_action"], entry["preprocessing_action"])
            bundles.append(bundle)
        if corpus._json_bytes(entry) != corpus._json_bytes(expected):
            raise AssemblyError("Assembly store reference/provenance mismatch")
    rows, _, _, _ = validation.load_passages(output / "passages.jsonl", output / "corpus.jsonl", cache_dir=cache,
        allow_duplicate_content=True)
    ordered = sorted(bundles, key=lambda b: b[0].doc_id)
    corpus_data = corpus._input_bytes(output / "corpus.jsonl")
    corpus_manifest_data = corpus._input_bytes(output / "corpus.jsonl.manifest.json")
    corpus_manifest = collection._json_object(corpus_manifest_data, "Assembled corpus manifest")
    if corpus_manifest.get("input_sha256") != corpus._sha256(corpus_data):
        raise AssemblyError("Assembled canonical corpus input fingerprint mismatch")
    expected_corpus = _corpus_manifest([b[0] for b in ordered], corpus_data, assembled=True)
    if corpus_manifest_data != corpus._json_bytes(expected_corpus):
        raise AssemblyError("Assembled corpus manifest semantics mismatch")
    expected_collection = _collection_report([b[0] for b in ordered], {b[0].doc_id: b[1] for b in ordered},
        config["cutoff"], [entry["artifacts"] for entry in entries], corpus._sha256(corpus_data), corpus._sha256(corpus_manifest_data))
    if corpus._input_bytes(output / "corpus.jsonl.collection.json") != corpus._json_bytes(expected_collection):
        raise AssemblyError("Assembled acquisition index differs from verified source snapshots")
    documents, metadata, source_inputs = preprocessing.load_snapshot(output / "corpus.jsonl", cache_dir=cache,
        allow_duplicate_content=True)
    expected_passages = _passage_manifest(documents, metadata, source_inputs, config,
        [r for b in ordered for r in b[2]], [b[3] for b in ordered], implementations[Path(__file__).name])
    if corpus._input_bytes(output / "passages.jsonl.manifest.json") != corpus._json_bytes(expected_passages):
        raise AssemblyError("Assembled passage manifest semantics/writer mismatch")
    availability = _availability(records, entries)
    for filename, expected in (
        ("query_articles.jsonl", b"".join(corpus._json_bytes(r) for r in records)),
        ("query_availability.jsonl", b"".join(corpus._json_bytes(r) for r in availability)),
        ("articles.jsonl", b"".join(corpus._json_bytes(r) for r in inventory)),
        ("corpus.jsonl", b"".join(corpus.serialize_document(b[0]) for b in ordered)),
        ("passages.jsonl", b"".join(corpus._json_bytes(r) for b in ordered for r in b[2])),
        ("passages.jsonl.diagnostics.jsonl", b"".join(corpus._json_bytes(b[3]) for b in ordered))):
        if corpus._input_bytes(output / filename) != expected:
            raise AssemblyError(f"Assembly content/association mismatch: {filename}")
    if data != corpus._json_bytes(_manifest(output, original, inputs, config, entries, manifest["allow_network"], implementations, availability)):
        raise AssemblyError("Incomplete, incompatible or fingerprint-mismatched assembly")
    return rows, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--discovery-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--store-dir", type=Path, help="Default: output parent / wikipedia_articles")
    parser.add_argument("--revision-cache-dir", type=Path, help="Default: store / cache")
    parser.add_argument("--cutoff", help="Default: discovery cutoff; canonical whole-second UTC")
    parser.add_argument("--revalidate-cutoff", action="store_true", help="Explicitly verify revisions at a different cutoff; not historical search")
    parser.add_argument("--reuse-snapshot", action="append", nargs=3, default=[], metavar=("CORPUS", "PASSAGES_OR_DASH", "CACHE"))
    parser.add_argument("--chunk-words", type=int, default=200)
    parser.add_argument("--overlap-words", type=int, default=40)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline-only", action="store_true", help="Default; cache misses fail")
    modes.add_argument("--allow-network", action="store_true", help="Explicitly permit revision API cache misses")
    parser.add_argument("--no-clobber", action="store_true", default=True, help="Always enforced; cannot be disabled")
    parser.add_argument("--user-agent")
    parser.add_argument("--request-delay", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        result = assemble(discovery_path=args.discovery, discovery_cache_dir=args.discovery_cache_dir,
            output_dir=args.output_dir, cutoff=args.cutoff, store_dir=args.store_dir, revision_cache_dir=args.revision_cache_dir,
            reuse_snapshots=args.reuse_snapshot, chunk_words=args.chunk_words, overlap_words=args.overlap_words,
            revalidate_cutoff=args.revalidate_cutoff, allow_network=args.allow_network,
            user_agent=args.user_agent, request_delay=args.request_delay)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Corpus assembly failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Corpus assembly interrupted; no completed assembly published\n")
    print(f"Queries: {result['query_count']}; articles: {result['article_count']}; passages: {result['passage_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
