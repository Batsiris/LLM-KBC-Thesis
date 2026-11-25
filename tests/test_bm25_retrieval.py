"""Synthetic, CPU-only retrieval checks; no benchmark labels or live sources."""

import json
import math
from dataclasses import asdict

import pytest

from llm_kbc import bm25_retrieval as retrieval
from llm_kbc import evidence_corpus as corpus
from llm_kbc import wikipedia_collect as acquisition
from llm_kbc import wikipedia_preprocess as prep


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Retrieval must never access the network")
    monkeypatch.setattr(acquisition, "urlopen", forbidden)


@pytest.fixture
def artifacts(tmp_path):
    """Build the existing archive schemas directly, without requiring a parser."""
    documents = [corpus.validate_document({"title": f"Synthetic {i}", "text": text,
        "source": "encyclopedia", "source_url": f"https://en.wikipedia.org/w/index.php?oldid={10+i}",
        "revision_id": str(10+i), "retrieved_at": "2026-11-16T10:00:00Z", "language": "en"})
        for i, text in enumerate(("Alpha land borders Beta", "Gamma total area 104.4"))]
    supplied = tmp_path / "supplied.jsonl"
    supplied.write_bytes(b"".join(corpus.serialize_document(d) for d in documents))
    source = tmp_path / "corpus.jsonl"
    corpus.ingest_corpus(supplied, source)
    documents = corpus.load_documents(source)
    license_info = {"text": "Synthetic license", "url": "https://example.org/license"}
    acquisition_records, provenance, passages, diagnostics = [], [], [], []
    config = prep._configuration(200, 40)
    for i, document in enumerate(documents):
        record = {**asdict(document), "input_line": i+1, "status": "collected", "request": {"title": document.title},
                  "revision_timestamp": "2026-06-30T12:00:00Z", "cutoff_satisfied": True,
                  "page_id": i+1, "license": license_info,
                  "attribution_history_url": "https://en.wikipedia.org/w/index.php?action=history"}
        record.pop("text")
        acquisition_records.append(record)
        provenance.append({k: record[k] for k in ("doc_id", "page_id", "license", "attribution_history_url", "content_sha256")})
        identity = {"doc_id": document.doc_id, "section_index": 0, "word_start": 0,
                    "word_end": len(document.text.split()), "content_sha256": document.content_sha256, "configuration": config}
        passages.append({**identity, "passage_id": "passage-" + corpus._sha256(corpus._json_bytes(identity)),
            "article_title": document.title, "text": document.text, "source_url": document.source_url,
            "revision_id": document.revision_id, "revision_timestamp": record["revision_timestamp"],
            "retrieved_at": document.retrieved_at, "language": "en", "section_path": [], "heading_level": 0,
            "alignment": "enclosing_top_level_nodes_not_exact_cleaned_spans",
            "source_spans": [{"start": 0, "end": len(document.text), "start_line": 1, "end_line": 1}]})
        diagnostics.append({"doc_id": document.doc_id, "article_title": document.title, "events": [],
                            "event_counts": {}, "unprocessed_events": 0,
                            "sections": [{"section_path": [], "heading_level": 0, "heading_span": None}]})
    collection = {"policy_version": acquisition.POLICY_VERSION, "status": "complete",
        "text_representation": "unexpanded-mediawiki-wikitext", "endpoint": "https://en.wikipedia.org/w/api.php",
        "cutoff": "2026-07-01T00:00:00Z", "selected_records": 2, "collected_records": 2, "input_records": 2,
        "corpus_sha256": corpus._sha256(source.read_bytes()),
        "manifest_sha256": corpus._sha256((tmp_path / "corpus.jsonl.manifest.json").read_bytes()),
        "records": acquisition_records}
    (tmp_path / "corpus.jsonl.collection.json").write_bytes(corpus._json_bytes(collection))
    path = tmp_path / "passages.jsonl"
    path.write_bytes(b"".join(corpus._json_bytes(p) for p in passages))
    (tmp_path / "passages.jsonl.diagnostics.jsonl").write_bytes(b"".join(corpus._json_bytes(d) for d in diagnostics))
    fingerprints = prep.load_snapshot(source)[2]
    manifest = {"schema_version": "evidence-passages-v1", "status": "complete", "preprocessing_version": prep.VERSION,
        "mwparserfromhell": prep.PARSER_VERSION, "code_sha256": "a" * 64, "cache_validation": "not_supplied_not_verified",
        "document_count": 2, "passage_count": len(passages), "unprocessed_events": 0,
        "configuration": {"chunk_words": 200, "overlap_words": 40, "section_local": True,
                          "word_definition": "Unicode non-whitespace runs"},
        "inputs": fingerprints, "document_provenance": provenance,
        "passages_sha256": corpus._sha256(path.read_bytes()),
        "diagnostics_sha256": corpus._sha256((tmp_path / "passages.jsonl.diagnostics.jsonl").read_bytes())}
    (tmp_path / "passages.jsonl.manifest.json").write_bytes(corpus._json_bytes(manifest))
    inputs = tmp_path / "identities.jsonl"
    inputs.write_bytes(b"".join(corpus._json_bytes(row) for row in [
        {"SubjectEntity": "Alpha", "Relation": "countryLandBordersCountry"},
        {"SubjectEntity": "Gamma", "Relation": "hasArea"},
        {"SubjectEntity": "NoSuchToken", "Relation": "hasCapacity"}]))
    return {"passages_path": path, "corpus_path": source, "identities_path": inputs,
            "output_path": tmp_path / "retrieval.jsonl"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rewrite_passages(artifacts, mutate):
    path = artifacts["passages_path"]
    rows = read_rows(path)
    mutate(rows)
    path.write_bytes(b"".join(corpus._json_bytes(row) for row in rows))
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    manifest.update(passage_count=len(rows), passages_sha256=corpus._sha256(path.read_bytes()))
    manifest_path.write_bytes(corpus._json_bytes(manifest))


@pytest.fixture
def cached_artifacts(artifacts):
    """Original unverified preprocessing plus fully valid, offline API caches."""
    source = artifacts["corpus_path"]
    collection_path = source.with_name(source.name + ".collection.json")
    collection = json.loads(collection_path.read_bytes())
    documents = {d.doc_id: d for d in corpus.load_documents(source)}
    cache_dir = source.parent / "cache"
    cache_dir.mkdir()
    for record in collection["records"]:
        document = documents[record["doc_id"]]
        request = {"endpoint": collection["endpoint"],
                   "params": acquisition._params(record["request"], collection["cutoff"])}
        key = corpus._sha256(corpus._json_bytes(request))
        payload = {"query": {"rightsinfo": record["license"], "pages": [{
            "title": document.title, "pageid": record["page_id"], "ns": 0, "pagelanguage": document.language,
            "revisions": [{"revid": int(document.revision_id), "timestamp": record["revision_timestamp"],
                "slots": {"main": {"contentmodel": "wikitext", "contentformat": "text/x-wiki", "content": document.text}}}]}]}}
        body = corpus._json_bytes(payload)
        snapshot = {"schema_version": acquisition.POLICY_VERSION, "request": request,
                    "response_body": body.decode("utf-8"), "response_sha256": corpus._sha256(body),
                    "retrieved_at": document.retrieved_at}
        metadata = {}
        assert acquisition._document(snapshot, record["request"], "en", collection["cutoff"], metadata) == document
        record.update(metadata)
        cache_path = cache_dir / (key + ".json")
        cache_path.write_bytes(corpus._json_bytes(snapshot))
        record["cache"] = {"key": key, "sha256": corpus._sha256(cache_path.read_bytes())}
    collection_path.write_bytes(corpus._json_bytes(collection))
    manifest_path = artifacts["passages_path"].with_name("passages.jsonl.manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    manifest["inputs"] = prep.load_snapshot(source)[2]
    for parent in manifest["document_provenance"]:
        record = next(r for r in collection["records"] if r["doc_id"] == parent["doc_id"])
        parent["attribution_history_url"] = record["attribution_history_url"]
    manifest_path.write_bytes(corpus._json_bytes(manifest))
    return artifacts, cache_dir


def record_cache_audit(artifacts, cache_dir):
    path = artifacts["passages_path"].with_name("passages.jsonl.manifest.json")
    manifest = json.loads(path.read_bytes())
    manifest.update(inputs=prep.load_snapshot(artifacts["corpus_path"], cache_dir=cache_dir)[2], cache_validation="verified")
    path.write_bytes(corpus._json_bytes(manifest))


def test_bm25_hand_calculation_and_query_frequency():
    rows = [{"passage_id": str(i), "text": text} for i, text in enumerate(("apple apple pear", "apple", "pear banana"))]
    index = retrieval.BM25Index(rows)
    results = index.search("apple")
    idf = math.log(1 + 1.5 / 2.5)
    expected = {"0": idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2)),
                "1": idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 / 2))}
    assert [p["passage_id"] for p, _ in results] == ["1", "0"]
    assert {p["passage_id"]: score for p, score in results} == pytest.approx(expected)
    assert index.search("apple apple") == results


def test_ties_top_k_and_input_order():
    a, b = {"passage_id": "a", "text": "same"}, {"passage_id": "b", "text": "same"}
    assert retrieval.BM25Index([b, a]).search("same") == retrieval.BM25Index([a, b]).search("same")
    assert retrieval.BM25Index([b, a]).search("same", 1)[0][0] == a


def test_unicode_and_numerical_tokenization():
    assert retrieval.tokenize("CAFÉ Cafe\u0301 ＡＢＣ Straße 東京_Βήτα 104.4 -12,345 1e3") == [
        "café", "café", "abc", "strasse", "東京", "βήτα", "104", "4", "12", "345", "1e3"]
    assert retrieval.tokenize("a-b/c's") == ["a", "b", "c", "s"]


@pytest.mark.parametrize("relation,suffix", [
    ("awardWonBy", "award winners recipients"), ("companyTradesAtStockExchange", "stock exchange listing"),
    ("countryLandBordersCountry", "land borders neighbouring countries"), ("hasArea", "total area"),
    ("hasCapacity", "capacity"), ("personHasCityOfDeath", "place of death")])
def test_exact_queries(relation, suffix):
    assert retrieval.build_query("Café {subject}", relation) == "Café {subject} " + suffix


def test_no_hit_and_empty_index():
    for rows in ([], [{"passage_id": "x", "text": "text"}], [{"passage_id": "x", "text": "---"}]):
        index = retrieval.BM25Index(rows)
        assert index.search("unknown") == []
        assert index.search("---") == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(top_k):
    with pytest.raises(retrieval.RetrievalError, match="positive integer"):
        retrieval.BM25Index([]).search("x", top_k)


def test_complete_artifacts_determinism_and_provenance(artifacts):
    paths = [artifacts["corpus_path"], artifacts["passages_path"]]
    before = {p: p.read_bytes() for p in paths}
    result = retrieval.retrieve(**artifacts)
    rows = read_rows(artifacts["output_path"])
    assert result["record_count"] == result["input_count"] == 3
    assert result["retrieved_count"] == 2 and result["no_hit_count"] == 1
    assert [r["input_index"] for r in rows] == [1, 2, 3]
    assert [(r["SubjectEntity"], r["Relation"]) for r in rows] == [
        (r["SubjectEntity"], r["Relation"]) for r in read_rows(artifacts["identities_path"])]
    assert rows[2]["status"] == "no_hits" and rows[2]["hits"] == []
    hit = rows[0]["hits"][0]
    assert hit["text"] == "Alpha land borders Beta" and hit["score"] > 0
    assert hit["revision_id"] and hit["source_spans"] and hit["license"]["text"] == "Synthetic license"
    assert hit["revision_timestamp"] != hit["retrieved_at"]
    assert hit["source"] == "encyclopedia" and hit["page_id"] > 0
    assert result["output_sha256"] == corpus._sha256(artifacts["output_path"].read_bytes())
    replay = artifacts["output_path"].with_name("replay.jsonl")
    assert retrieval.retrieve(**{**artifacts, "output_path": replay}) == result
    for suffix in ("", ".manifest.json"):
        assert artifacts["output_path"].with_name(artifacts["output_path"].name + suffix).read_bytes() == replay.with_name(replay.name + suffix).read_bytes()
    assert all(p.read_bytes() == data for p, data in before.items())


def test_gold_independence(artifacts):
    result = retrieval.retrieve(**artifacts)
    path = artifacts["identities_path"]
    identities = read_rows(path)
    for row in identities:
        row["ObjectEntities"] = {"arbitrary": ["NOT EVIDENCE", None]}
    path.write_bytes(b"".join(corpus._json_bytes(r) for r in identities))
    new_output = artifacts["output_path"].with_name("changed_gold.jsonl")
    assert retrieval.retrieve(**{**artifacts, "output_path": new_output}) == result
    assert new_output.read_bytes() == artifacts["output_path"].read_bytes()


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.append(rows[0]),
    lambda rows: rows[0].update(text="tampered"),
    lambda rows: rows[0].update(revision_id="999"),
    lambda rows: rows[0].update(source_spans=[]),
    lambda rows: rows[0].update(section_path="not a list"),
    lambda rows: rows[0].update(word_start=True),
    lambda rows: rows[0].update(doc_id="unknown"),
    lambda rows: rows[0].update(doc_id=[]),
    lambda rows: rows[0].update(passage_id=[]),
    lambda rows: rows[0].update(unexpected="not in the passage schema"),
    lambda rows: rows[0].update(section_index=99),
    lambda rows: rows.reverse(),
])
def test_invalid_passages_fail_before_publication(artifacts, mutation):
    rewrite_passages(artifacts, mutation)
    with pytest.raises(retrieval.RetrievalError):
        retrieval.retrieve(**artifacts)
    assert not artifacts["output_path"].exists()
    assert not artifacts["output_path"].with_name("retrieval.jsonl.manifest.json").exists()


@pytest.mark.parametrize("field,value", [("schema_version", "future"), ("status", "incomplete"),
    ("inputs", {}), ("document_provenance", []), ("document_count", True), ("passage_count", 99),
    ("passages_sha256", "0"*64), ("diagnostics_sha256", "0"*64), ("configuration", []),
    ("preprocessing_version", "unknown"), ("unprocessed_events", -1)])
def test_bad_manifest(artifacts, field, value):
    path = artifacts["passages_path"].with_name("passages.jsonl.manifest.json")
    manifest = json.loads(path.read_bytes())
    manifest[field] = value
    path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises((retrieval.RetrievalError, prep.PreprocessingError)):
        retrieval.retrieve(**artifacts)
    assert not artifacts["output_path"].exists()


@pytest.mark.parametrize("suffix", [".manifest.json", ".diagnostics.jsonl"])
def test_missing_passage_sidecars(artifacts, suffix):
    artifacts["passages_path"].with_name("passages.jsonl" + suffix).unlink()
    with pytest.raises(corpus.CorpusError, match="could not read"):
        retrieval.retrieve(**artifacts)
    assert not artifacts["output_path"].exists()


def test_incomplete_source_rejected(artifacts):
    path = artifacts["corpus_path"].with_name("corpus.jsonl.collection.json")
    collection = json.loads(path.read_bytes())
    collection["status"] = "incomplete"
    path.write_bytes(corpus._json_bytes(collection))
    with pytest.raises(prep.PreprocessingError, match="incomplete"):
        retrieval.retrieve(**artifacts)


@pytest.mark.parametrize("artifact", ["corpus.jsonl", "passages.jsonl", "passages.jsonl.diagnostics.jsonl"])
def test_changed_content_fingerprints_rejected(artifacts, artifact):
    path = artifacts["passages_path"].with_name(artifact)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        retrieval.retrieve(**artifacts)
    assert not artifacts["output_path"].exists()


def test_missing_output_parent(artifacts):
    with pytest.raises(retrieval.RetrievalError, match="parent"):
        retrieval.retrieve(**{**artifacts, "output_path": artifacts["output_path"].parent / "absent" / "out.jsonl"})


def test_input_change_detected_before_publication(artifacts, monkeypatch):
    original = retrieval.BM25Index.search
    def changed(self, query, top_k=5):
        artifacts["identities_path"].write_bytes(corpus._json_bytes({"SubjectEntity": "Changed", "Relation": "hasArea"}))
        return original(self, query, top_k)
    monkeypatch.setattr(retrieval.BM25Index, "search", changed)
    with pytest.raises(retrieval.RetrievalError, match="changed"):
        retrieval.retrieve(**artifacts)
    assert not artifacts["output_path"].exists()


@pytest.mark.parametrize("content", [b'{"SubjectEntity":"A"', b'[]\n', b'\xff', b'\n', b'',
    b'{"SubjectEntity":"A","Relation":"unknown"}\n', b'{"Relation":"hasArea"}\n',
    b'{"SubjectEntity":"A","SubjectEntity":"B","Relation":"hasArea"}\n'])
def test_invalid_identity_inputs(artifacts, content):
    artifacts["identities_path"].write_bytes(content)
    with pytest.raises((ValueError, corpus.CorpusError)):
        retrieval.retrieve(**artifacts)
    assert not artifacts["output_path"].exists()


def test_duplicate_normalized_identities(artifacts):
    path = artifacts["identities_path"]
    path.write_bytes(corpus._json_bytes({"SubjectEntity": " Alpha ", "Relation": "hasArea"}) +
                     corpus._json_bytes({"SubjectEntity": "ＡLPHA", "Relation": "hasArea"}))
    with pytest.raises(retrieval.RetrievalError, match="duplicate normalized"):
        retrieval.retrieve(**artifacts)


@pytest.mark.parametrize("suffix", ["", ".manifest.json"])
def test_no_overwrite(artifacts, suffix):
    path = artifacts["output_path"].with_name("retrieval.jsonl" + suffix)
    path.write_bytes(b"existing artifact")
    with pytest.raises(retrieval.RetrievalError, match="overwrite"):
        retrieval.retrieve(**artifacts)
    assert path.read_bytes() == b"existing artifact"
    assert not artifacts["output_path"].with_name("retrieval.jsonl" + ("" if suffix else ".manifest.json")).exists()


def test_publication_failure_has_no_completion_marker(artifacts, monkeypatch):
    original = corpus._publish
    def fail_manifest(path, data):
        if str(path).endswith(".manifest.json"):
            raise OSError("simulated interruption")
        original(path, data)
    monkeypatch.setattr(corpus, "_publish", fail_manifest)
    with pytest.raises(OSError, match="interruption"):
        retrieval.retrieve(**artifacts)
    assert artifacts["output_path"].exists()
    assert not artifacts["output_path"].with_name("retrieval.jsonl.manifest.json").exists()


def test_cli(artifacts, capsys):
    args = ["--passages", str(artifacts["passages_path"]), "--corpus", str(artifacts["corpus_path"]),
            "--identities", str(artifacts["identities_path"]), "--output", str(artifacts["output_path"])]
    assert retrieval.main(args) == 0
    assert "Records: 3; no hits: 1" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        retrieval.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("value", [[], {}, ["verified"], {"status": "verified"}, None, True, False, 0, 1, "invalid"])
def test_malformed_cache_validation_and_controlled_cli(artifacts, capsys, value):
    path = artifacts["passages_path"].with_name("passages.jsonl.manifest.json")
    manifest = json.loads(path.read_bytes())
    manifest["cache_validation"] = value
    path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises(retrieval.RetrievalError, match="preprocessing provenance"):
        retrieval.retrieve(**artifacts)
    with pytest.raises(SystemExit) as error:
        retrieval.main(["--passages", str(artifacts["passages_path"]), "--corpus", str(artifacts["corpus_path"]),
            "--identities", str(artifacts["identities_path"]), "--output", str(artifacts["output_path"])])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "Retrieval failed: Invalid preprocessing provenance" in stderr
    assert "Traceback" not in stderr and "TypeError" not in stderr
    assert not artifacts["output_path"].exists()
    assert not artifacts["output_path"].with_name("retrieval.jsonl.manifest.json").exists()


@pytest.mark.parametrize("archived_verified", [False, True])
@pytest.mark.parametrize("request_cache", [False, True])
def test_archived_and_fresh_cache_verification(cached_artifacts, archived_verified, request_cache):
    artifacts, cache_dir = cached_artifacts
    if archived_verified:
        record_cache_audit(artifacts, cache_dir)
    manifest_path = artifacts["passages_path"].with_name("passages.jsonl.manifest.json")
    archived = json.loads(manifest_path.read_bytes())
    cache_args = {"cache_dir": cache_dir} if request_cache else {}
    result = retrieval.retrieve(**artifacts, **cache_args)
    assert result["source_inputs"] == archived["inputs"]
    assert result["archived_cache_validation"] == ("verified" if archived_verified else "not_supplied_not_verified")
    assert result["cache_revalidated"] is request_cache
    fresh_keys = {k for k in result["inputs"] if k.startswith("cache:")}
    assert len(fresh_keys) == (2 if request_cache else 0)
    if request_cache:
        fresh = prep.load_snapshot(artifacts["corpus_path"], cache_dir=cache_dir)[2]
        for key in fresh_keys:
            assert result["inputs"][key] == fresh[key]
        assert bool(fresh_keys - archived["inputs"].keys()) is not archived_verified
    # Repeated executions at new paths retain both records and manifests exactly.
    replay = artifacts["output_path"].with_name("cache_replay.jsonl")
    assert retrieval.retrieve(**{**artifacts, "output_path": replay}, **cache_args) == result
    for suffix in ("", ".manifest.json"):
        assert replay.with_name(replay.name + suffix).read_bytes() == artifacts["output_path"].with_name("retrieval.jsonl" + suffix).read_bytes()
    # Cache verification has no effect on retrieval records, rankings or scores.
    other = artifacts["output_path"].with_name("other_verification.jsonl")
    retrieval.retrieve(**{**artifacts, "output_path": other}, **({} if request_cache else {"cache_dir": cache_dir}))
    assert other.read_bytes() == replay.read_bytes()


@pytest.mark.parametrize("archived_verified", [False, True])
@pytest.mark.parametrize("damage", ["missing", "corrupted", "inconsistent"])
def test_requested_cache_failures_are_closed(cached_artifacts, archived_verified, damage):
    artifacts, cache_dir = cached_artifacts
    if archived_verified:
        record_cache_audit(artifacts, cache_dir)
    cache_path = next(cache_dir.iterdir())
    if damage == "missing":
        cache_path.unlink()
    elif damage == "corrupted":
        cache_path.write_bytes(cache_path.read_bytes() + b" ")
    else:
        # Internally rehashed API content must still agree with the frozen corpus.
        snapshot = json.loads(cache_path.read_bytes())
        payload = json.loads(snapshot["response_body"])
        payload["query"]["pages"][0]["revisions"][0]["revid"] += 100
        body = corpus._json_bytes(payload)
        snapshot.update(response_body=body.decode("utf-8"), response_sha256=corpus._sha256(body))
        cache_path.write_bytes(corpus._json_bytes(snapshot))
        collection_path = artifacts["corpus_path"].with_name("corpus.jsonl.collection.json")
        collection = json.loads(collection_path.read_bytes())
        for record in collection["records"]:
            if record["cache"]["key"] == cache_path.stem:
                record["cache"]["sha256"] = corpus._sha256(cache_path.read_bytes())
        collection_path.write_bytes(corpus._json_bytes(collection))
    with pytest.raises(ValueError):
        retrieval.retrieve(**artifacts, cache_dir=cache_dir)
    assert not artifacts["output_path"].exists()
    assert not artifacts["output_path"].with_name("retrieval.jsonl.manifest.json").exists()


@pytest.mark.parametrize("field", ["corpus", "manifest", "collection", "cache_hash", "cache_bytes", "unknown_cache"])
def test_archived_fingerprint_checks_preserved(cached_artifacts, field):
    artifacts, cache_dir = cached_artifacts
    record_cache_audit(artifacts, cache_dir)
    path = artifacts["passages_path"].with_name("passages.jsonl.manifest.json")
    manifest = json.loads(path.read_bytes())
    cache_key = next(k for k in manifest["inputs"] if k.startswith("cache:"))
    if field == "unknown_cache":
        manifest["inputs"]["cache:" + "f" * 64] = {"sha256": "a" * 64, "bytes": 1}
    elif field == "cache_bytes":
        manifest["inputs"][cache_key]["bytes"] += 1
    else:
        manifest["inputs"][cache_key if field == "cache_hash" else field]["sha256"] = "0" * 64
    path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises(retrieval.RetrievalError, match="fingerprint"):
        retrieval.retrieve(**artifacts, cache_dir=cache_dir)
    assert not artifacts["output_path"].exists()


def test_archived_cache_hash_bound_to_collection_without_fresh_cache(cached_artifacts):
    artifacts, cache_dir = cached_artifacts
    record_cache_audit(artifacts, cache_dir)
    path = artifacts["passages_path"].with_name("passages.jsonl.manifest.json")
    manifest = json.loads(path.read_bytes())
    key = next(k for k in manifest["inputs"] if k.startswith("cache:"))
    manifest["inputs"][key]["sha256"] = "0" * 64
    path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises(retrieval.RetrievalError, match="fingerprint"):
        retrieval.retrieve(**artifacts)
    assert not artifacts["output_path"].exists()
