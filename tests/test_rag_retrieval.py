"""Query-local retrieval using synthetic frozen API responses; no real network."""

import copy
import math
from dataclasses import replace

import pytest

from llm_kbc import bm25_retrieval as bm25
from llm_kbc import evidence_corpus as corpus
from llm_kbc import rag_corpus_assembly as assembly
from llm_kbc import rag_query
from llm_kbc import rag_retrieval as retrieval
from llm_kbc import wikipedia_collect as collection
from llm_kbc import wikipedia_discovery_v2 as discovery_v2
from test_rag_corpus_assembly import arguments, change_cached_pages, environment


def make(env, *, version="v2", subjects=("Alpha", "Beta", "Alpha"), name="discovery"):
    args = arguments(env, env["discover"](version, subjects, name), name + "_assembly")
    env["seed"]((1, 2, 3))
    def text(page):
        page["revisions"][0]["slots"]["main"]["content"] = (
            f'{page["title"]} apple apple.\n== History ==\n{page["title"]} history.')
    change_cached_pages(env, text, page_ids=(1, 2, 3))
    assembly.assemble(**args)
    return {"assembly_dir": args["output_dir"], "discovery_path": args["discovery_path"],
        "discovery_cache_dir": args["discovery_cache_dir"], "store_dir": args["store_dir"],
        "revision_cache_dir": args["revision_cache_dir"], "output_path": env["root"] / (name + "_retrieval.jsonl")}


def read(options):
    return retrieval.load_retrieval(**options)


def pool_options(options):
    return {key: value for key, value in options.items() if key != "output_path"}


@pytest.fixture
def prepared(environment, monkeypatch):
    options = make(environment)
    monkeypatch.setattr(collection, "_fetch", lambda *a, **k: pytest.fail("Offline retrieval attempted acquisition"))
    monkeypatch.setattr(bm25, "load_identities", lambda *a, **k: pytest.fail("Retrieval loaded a benchmark dataset"))
    return options


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_valid_assembly_integration_generic_and_legacy(environment, version):
    options = make(environment, version=version)
    manifest = retrieval.retrieve(**options)
    records, checked = read(options)
    assert checked == manifest and len(records) == 3
    assert manifest["statistics_scope"] == "query_eligible_passages_only"
    assert manifest["configuration"]["k1"] == 1.2 and manifest["configuration"]["b"] == 0.75
    pools, _ = retrieval.load_evidence_pools(**pool_options(options))
    for record, pool in zip(records, pools, strict=True):
        assert record["query_provenance"] == rag_query.build_query({key: record["query_provenance"][key]
            for key in ("subject", "relation_id", "relation_description", "relation_namespace")})
        assert record["eligible_passage_count"] == len(pool.passages) == 2
        assert record["status"] == "retrieved" and len(record["hits"]) == 2
        assert all(hit["doc_id"] in record["eligible_document_ids"] for hit in record["hits"])
        assert all(hit["revision_id"] and hit["page_id"] and hit["license"] for hit in record["hits"])
    if version == "v1":
        assert records[0]["query"] == bm25.build_query("Alpha", "hasArea")
        assert records[0]["query_provenance"]["relation_namespace"] == "lm-kbc-2026"
    else:
        assert records[0]["query"] == "Alpha independent description"
        assert records[0]["query_provenance"]["relation_namespace"] == "synthetic"


def test_shared_articles_and_exact_pool_fingerprints(prepared):
    pools, _ = retrieval.load_evidence_pools(**pool_options(prepared))
    assert pools[0].passages == pools[2].passages
    assert pools[0].evidence_pool_sha256 == pools[2].evidence_pool_sha256
    assert pools[0].query_provenance["identity_sha256"] != pools[2].query_provenance["identity_sha256"]
    assert pools[0].evidence_pool_sha256 != pools[1].evidence_pool_sha256
    expected = corpus._sha256(corpus._json_bytes({"policy": retrieval.ELIGIBILITY_POLICY,
        "documents": pools[0].documents, "passages": pools[0].passages}))
    assert expected == pools[0].evidence_pool_sha256


def synthetic_pool(passages, subject="apple", relation="apple"):
    documents = tuple({"doc_id": doc, "page_id": index, "license": {},
        "attribution_history_url": "https://example.org/history", "parent_content_sha256": "a" * 64}
        for index, doc in enumerate(sorted({p["doc_id"] for p in passages}), 1))
    availability = {"status": "full_candidate_availability", "eligible_document_ids": [d["doc_id"] for d in documents],
        "query_record_sha256": "b" * 64}
    return retrieval.EvidencePool(1, rag_query.build_query(rag_query.QueryInput(subject, relation)),
        availability, tuple(passages), documents, "c" * 64)


def test_independent_reference_bm25_statistics_and_binary_query_terms():
    pool = synthetic_pool([{"passage_id": "p1", "doc_id": "d1", "text": "apple apple pear"},
        {"passage_id": "p2", "doc_id": "d2", "text": "pear"}])
    record = retrieval.rank_pool(pool)
    expected = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5)) * (2 * (1.2 + 1)) / (
        2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 2))
    assert len(record["hits"]) == 1 and record["hits"][0]["score"] == pytest.approx(expected, rel=1e-15)
    # The query repeats 'apple'; the engine treats its term frequency as binary.
    assert record["hits"][0]["score"] == retrieval.rank_pool(
        replace(pool, query_provenance=rag_query.build_query(rag_query.QueryInput("apple", "missing"))))["hits"][0]["score"]


def test_positive_only_ties_and_top_k():
    pool = synthetic_pool([{"passage_id": "z", "doc_id": "d1", "text": "apple"},
        {"passage_id": "a", "doc_id": "d2", "text": "apple"},
        {"passage_id": "empty-tokenized", "doc_id": "d3", "text": "!!!"}])
    hits = retrieval.rank_pool(pool, top_k=10)["hits"]
    assert [h["passage_id"] for h in hits] == ["a", "z"]
    assert hits[0]["score"] == hits[1]["score"] > 0
    assert [h["passage_id"] for h in retrieval.rank_pool(pool, top_k=1)["hits"]] == ["a"]


@pytest.mark.parametrize("relation", list(bm25.QUERIES))
def test_all_six_legacy_queries_keep_exact_text(relation):
    subject = "  Caf\u00e9 \u6771\u4eac  "
    record = {"SubjectEntity": subject, "Relation": relation, "query": bm25.build_query(subject, relation)}
    provenance = retrieval._query(record, True)
    assert provenance["query"] == record["query"]
    assert provenance["subject"] == subject and provenance["relation_namespace"] == "lm-kbc-2026"


@pytest.mark.parametrize("damage", ["archived_provenance", "submitted_query"])
def test_generic_query_consistency_is_checked(damage):
    query = rag_query.build_query(rag_query.QueryInput("Unseen subject", "P123", "orbital period", "astronomy"))
    record = {key: query[key] for key in ("subject", "relation_id", "relation_description", "relation_namespace")}
    record.update(query_provenance=query, search_expression=query["query"])
    if damage == "archived_provenance":
        record["query_provenance"]["query_sha256"] = "0" * 64
    else:
        record["search_expression"] += " unsupported expansion"
    with pytest.raises(retrieval.RetrievalError, match="Discovery query differs"):
        retrieval._query(record, False)


def test_unrelated_query_and_document_invariance(environment):
    first = make(environment, subjects=("Alpha", "Beta", "Alpha"), name="first")
    retrieval.retrieve(**first)
    original, _ = read(first)
    expanded = make(environment, subjects=("Alpha", "Beta", "Alpha", "Gamma"), name="expanded")
    retrieval.retrieve(**expanded)
    larger, _ = read(expanded)
    assert len(larger) == 4 and larger[:3] == original
    before, _ = retrieval.load_evidence_pools(**pool_options(first))
    after, _ = retrieval.load_evidence_pools(**pool_options(expanded))
    assert before[0].passages == after[0].passages


def test_identical_text_preserves_distinct_sources(environment):
    args = arguments(environment)
    environment["seed"]()
    change_cached_pages(environment, lambda p: p["revisions"][0]["slots"]["main"].update(
        content="Alpha Beta identical text."), page_ids=(1, 2))
    assembly.assemble(**args)
    pools, _ = retrieval.load_evidence_pools(assembly_dir=args["output_dir"],
        **{k: v for k, v in args.items() if k != "output_dir"})
    assert pools[0].passages[0]["text"] == pools[1].passages[0]["text"]
    assert pools[0].passages[0]["passage_id"] != pools[1].passages[0]["passage_id"]
    assert pools[0].passages[0]["doc_id"] != pools[1].passages[0]["doc_id"]
    assert retrieval.rank_pool(pools[0])["hits"][0]["page_id"] == 1
    assert retrieval.rank_pool(pools[1])["hits"][0]["page_id"] == 2


def test_unresolved_unavailable_and_no_hit_queries_are_preserved(environment):
    args = arguments(environment, environment["discover"](subjects=("Alpha", "Beta", "Missing")))
    environment["seed"]()
    change_cached_pages(environment, lambda p: p["revisions"][0]["slots"]["main"].update(content="Only unrelated words."), page_ids=(1,))
    change_cached_pages(environment, lambda p: p.pop("revisions"), page_ids=(2,))
    assembly.assemble(**args)
    options = {"assembly_dir": args.pop("output_dir"), **args, "output_path": environment["root"] / "empty_hits.jsonl"}
    manifest = retrieval.retrieve(**options)
    records, _ = read(options)
    assert [r["input_index"] for r in records] == [1, 2, 3]
    assert [r["status"] for r in records] == ["no_hits", "no_usable_evidence", "unresolved_discovery"]
    assert [r["upstream_availability_status"] for r in records] == [
        "full_candidate_availability", "resolved_no_usable_evidence", "unresolved_discovery"]
    assert all(r["hits"] == [] for r in records)
    assert manifest["outcome_counts"] == {"retrieved": 0, "no_hits": 1, "no_usable_evidence": 1, "unresolved_discovery": 1}


@pytest.mark.parametrize("source", ["zero_passages", "excluded"])
def test_resolved_query_with_no_eligible_passages(environment, source):
    args = arguments(environment, environment["discover"](subjects=("Alpha",)))
    environment["seed"]((1,))
    change_cached_pages(environment, lambda p: p["revisions"][0]["slots"]["main"].update(
        content="<ref>Reference only</ref>" if source == "zero_passages" else ""), page_ids=(1,))
    assembly.assemble(**args)
    options = {"assembly_dir": args.pop("output_dir"), **args, "output_path": environment["root"] / "no_evidence.jsonl"}
    retrieval.retrieve(**options)
    records, _ = read(options)
    assert len(records) == 1 and records[0]["status"] == "no_usable_evidence"
    assert records[0]["eligible_passage_count"] == 0 and records[0]["eligible_document_ids"] == []
    assert records[0]["hits"] == []


@pytest.mark.parametrize("top_k", [None, False, True, 0, -1, 1.5, "5"])
def test_invalid_configuration_fails_before_io(tmp_path, top_k):
    target = tmp_path / "retrieval.jsonl"
    with pytest.raises(retrieval.RetrievalError, match="top_k"):
        retrieval.retrieve(assembly_dir="missing", discovery_path="missing", discovery_cache_dir="missing",
            output_path=target, top_k=top_k)
    assert not target.exists() and not list(tmp_path.iterdir())


def rehash_output(options, records, manifest):
    data = b"".join(corpus._json_bytes(r) for r in records)
    manifest["output"] = {"sha256": corpus._sha256(data), "bytes": len(data)}
    options["output_path"].write_bytes(data)
    path = options["output_path"].with_name(options["output_path"].name + ".manifest.json")
    path.write_bytes(corpus._json_bytes(manifest))


@pytest.mark.parametrize("damage", ["identity", "query", "pool", "document", "count", "score", "rank", "passage",
    "ordering", "missing_query", "duplicate_query", "statistics", "k1", "b", "input_hash", "outcome_count",
    "policy", "schema", "unicode", "implementation"])
def test_rehashed_tampering_is_rejected(prepared, damage):
    retrieval.retrieve(**prepared)
    records, manifest = read(prepared)
    first = records[0]
    if damage == "identity": first["query_provenance"]["identity_sha256"] = "0" * 64
    elif damage == "query": first["query"] = "another query"
    elif damage == "pool": first["evidence_pool_sha256"] = "0" * 64
    elif damage == "document": first["eligible_document_ids"] = []
    elif damage == "count": first["eligible_passage_count"] += 1
    elif damage == "score": first["hits"][0]["score"] += 0.1
    elif damage == "rank": first["hits"][0]["rank"] = 9
    elif damage == "passage": first["hits"][0]["passage_id"] = first["hits"][1]["passage_id"]
    elif damage == "ordering": first["hits"].reverse()
    elif damage == "missing_query": records.pop()
    elif damage == "duplicate_query": records[1] = copy.deepcopy(first)
    elif damage == "statistics": manifest["statistics_scope"] = "global"
    elif damage in {"k1", "b"}: manifest["configuration"][damage] += 0.1
    elif damage == "input_hash": manifest["inputs"]["files"]["assembly/manifest.json"]["sha256"] = "0" * 64
    elif damage == "outcome_count": manifest["outcome_counts"]["retrieved"] += 1
    elif damage == "policy": manifest["retrieval_policy"] = "unsupported"
    elif damage == "schema": manifest["schema_version"] = "unsupported"
    elif damage == "unicode": manifest["unicode_version"] = "unsupported"
    else: manifest["implementation_sha256"]["rag_retrieval.py"] = "invalid"
    rehash_output(prepared, records, manifest)
    with pytest.raises(ValueError):
        read(prepared)


@pytest.mark.parametrize("name", ["manifest.json", "query_articles.jsonl", "query_availability.jsonl", "passages.jsonl",
    "corpus.jsonl.collection.json"])
def test_corrupted_upstream_cannot_publish(prepared, name):
    (prepared["assembly_dir"] / name).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        retrieval.retrieve(**prepared)
    assert not prepared["output_path"].exists()
    assert not prepared["output_path"].with_suffix(".jsonl.manifest.json").exists()


def test_archived_implementation_hash_is_not_falsely_required_to_match_current(prepared):
    retrieval.retrieve(**prepared)
    records, manifest = read(prepared)
    manifest["implementation_sha256"]["rag_retrieval.py"] = "f" * 64
    rehash_output(prepared, records, manifest)
    assert read(prepared)[1] == manifest


def test_offline_replay_is_byte_identical_and_does_not_mutate_upstream(prepared):
    paths = list(prepared["assembly_dir"].rglob("*")) + list(prepared["store_dir"].rglob("*"))
    before = {p: p.read_bytes() for p in paths if p.is_file()}
    first = retrieval.retrieve(**prepared)
    other = {**prepared, "output_path": prepared["output_path"].with_name("replay.jsonl")}
    assert retrieval.retrieve(**other) == first
    assert read(other)[0] == read(prepared)[0]
    assert other["output_path"].read_bytes() == prepared["output_path"].read_bytes()
    assert other["output_path"].with_suffix(".jsonl.manifest.json").read_bytes() == prepared["output_path"].with_suffix(".jsonl.manifest.json").read_bytes()
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("existing", ["records", "manifest"])
def test_no_clobber(prepared, existing):
    path = prepared["output_path"] if existing == "records" else prepared["output_path"].with_suffix(".jsonl.manifest.json")
    path.write_bytes(b"keep")
    with pytest.raises(ValueError, match="overwrite"):
        retrieval.retrieve(**prepared)
    assert path.read_bytes() == b"keep"


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_interrupted_publication_leaves_rejected_orphan(prepared, monkeypatch, failure):
    publish = corpus._publish
    marker = prepared["output_path"].with_suffix(".jsonl.manifest.json")
    def interrupted(path, data):
        if path == marker:
            raise failure("synthetic publication failure")
        return publish(path, data)
    monkeypatch.setattr(corpus, "_publish", interrupted)
    with pytest.raises(failure):
        retrieval.retrieve(**prepared)
    assert prepared["output_path"].exists() and not marker.exists()
    with pytest.raises(corpus.CorpusError, match="could not read evidence input"):
        read(prepared)
    with pytest.raises(ValueError, match="overwrite"):
        retrieval.retrieve(**prepared)


def test_evidence_change_before_publication_is_rejected(prepared, monkeypatch):
    original = retrieval.load_evidence_pools
    calls = 0
    def changed(**options):
        nonlocal calls
        pools, provenance = original(**options)
        calls += 1
        if calls == 2:
            provenance["historical_cutoff"] = "different"
        return pools, provenance
    monkeypatch.setattr(retrieval, "load_evidence_pools", changed)
    with pytest.raises(ValueError, match="Evidence changed"):
        retrieval.retrieve(**prepared)
    assert not prepared["output_path"].exists()


def test_generic_projection_never_inspects_unused_gold_values():
    class Poison:
        def __str__(self): raise AssertionError("Gold accessed")
        def __bool__(self): raise AssertionError("Gold accessed")
        def __iter__(self): raise AssertionError("Gold accessed")
    row = {"subject": "Unseen star", "relation_id": "P-unknown", "relation_namespace": "astronomy",
        "relation_description": "spectral type", "ObjectEntities": Poison()}
    projected = discovery_v2.project_input(row, "generic")
    record = {"subject": projected.subject, "relation_id": projected.relation_id,
        "relation_namespace": projected.relation_namespace, "relation_description": projected.relation_description,
        "query_provenance": rag_query.build_query(projected), "search_expression": "Unseen star spectral type"}
    assert retrieval._query(record, False)["query"] == "Unseen star spectral type"


def test_cli_help_and_controlled_validation_failure(tmp_path, capsys):
    with pytest.raises(SystemExit) as result:
        retrieval.main(["--help"])
    assert result.value.code == 0
    args = ["--assembly", "missing", "--discovery", "missing", "--discovery-cache-dir", "missing",
        "--output", str(tmp_path / "out.jsonl"), "--top-k", "0"]
    with pytest.raises(SystemExit) as result:
        retrieval.main(args)
    assert result.value.code == 2 and "top_k" in capsys.readouterr().err
