"""Synthetic historical snapshots only; never live requests or benchmark labels."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm_kbc import evidence_corpus as corpus
from llm_kbc import rag_corpus_assembly as assembly
from llm_kbc import wikipedia_collect as collection
from llm_kbc import wikipedia_discovery as discovery
from llm_kbc import wikipedia_discovery_v2 as discovery_v2
from llm_kbc import wikipedia_preprocess as preprocessing


CUTOFF = "2026-07-01T00:00:00Z"
UA = "SyntheticAssembly/1.0 (https://example.org/contact)"


def write(path, rows):
    path.write_bytes(b"".join(corpus._json_bytes(row) for row in rows))
    return path


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "urlopen", lambda *a, **k: pytest.fail("Real network forbidden"))
    calls = []
    titles = {1: "Alpha", 2: "Beta", 3: "Gamma"}

    def fetch(endpoint, params, user_agent, delay):
        calls.append(copy.deepcopy(params))
        if "rvprop" in params:
            page_id = int(params["pageids"]) if "pageids" in params else next(i for i, t in titles.items() if t == params["titles"])
            payload = {"query": {"rightsinfo": {"text": "Synthetic license", "url": "https://example.org/license"},
                "pages": [{"pageid": page_id, "title": titles[page_id], "ns": 0, "pagelanguage": "en",
                    "revisions": [{"revid": 100 + page_id, "timestamp": "2026-06-01T00:00:00Z",
                        "slots": {"main": {"contentmodel": "wikitext", "contentformat": "text/x-wiki",
                            "content": f"{titles[page_id]} has [[borders|land borders]].\n== History ==\nSynthetic history {page_id}."}}}]}]}}
        elif "titles" in params:
            title = params["titles"]
            pages = [{"title": title, "ns": 0, "missing": True}] if title == "Missing" else [
                {"title": title, "ns": 0, "pageid": next(i for i, t in titles.items() if t == title)}]
            payload = {"query": {"pages": pages}}
        else:
            assert params["list"] == "search"
            payload = {"query": {"search": []}}
        body = corpus._json_bytes(payload)
        return {"schema_version": collection.POLICY_VERSION, "request": {"endpoint": endpoint, "params": params},
            "retrieved_at": "2026-11-01T00:00:00Z", "user_agent": user_agent, "headers": {},
            "response_body": body.decode("utf-8"), "response_sha256": corpus._sha256(body)}

    monkeypatch.setattr(collection, "_fetch", fetch)
    discovery_cache = tmp_path / "discovery_cache"
    revision_cache = tmp_path / "revision_cache"
    revision_cache.mkdir()

    def discover(version="v2", subjects=("Alpha", "Beta", "Alpha"), name="discovery"):
        rows = ([{"SubjectEntity": s, "Relation": r} for s, r in zip(subjects, ("hasArea", "hasCapacity", "awardWonBy"))]
                if version == "v1" else [{"subject": s, "relation_id": f"R{i}", "relation_namespace": "synthetic",
                    "relation_description": "independent description"} for i, s in enumerate(subjects)])
        inputs = write(tmp_path / (name + "_inputs.jsonl"), rows)
        output = tmp_path / (name + ".jsonl")
        function = discovery.discover if version == "v1" else discovery_v2.discover
        function(identities_path=inputs, output_path=output, cache_dir=discovery_cache, allow_network=True, user_agent=UA)
        return output

    def seed(page_ids=(1, 2), cutoff=CUTOFF):
        for page_id in page_ids:
            collection._snapshot(discovery.ENDPOINT, collection._params({"page_id": page_id}, cutoff), revision_cache,
                allow_network=True, user_agent=UA, delay=1.0)

    return {"root": tmp_path, "discovery_cache": discovery_cache, "revision_cache": revision_cache,
            "discover": discover, "seed": seed, "calls": calls}


def arguments(env, discovery_path=None, name="assembled"):
    return {"discovery_path": discovery_path or env["discover"](), "discovery_cache_dir": env["discovery_cache"],
        "output_dir": env["root"] / name, "store_dir": env["root"] / "store", "revision_cache_dir": env["revision_cache"]}


def complete(args):
    rows, manifest = assembly.load_assembly(**args)
    assert len(rows) == manifest["passage_count"]
    return manifest


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_valid_discovery_coverage_and_native_retrieval_artifacts(environment, version):
    env = environment
    args = arguments(env, env["discover"](version))
    env["seed"]()
    count = len(env["calls"])
    result = assembly.assemble(**args)
    assert complete(args) == result
    assert len(env["calls"]) == count  # No HTTP, including no new _fetch calls.
    assert result["query_count"] == 3 and result["article_count"] == 2
    assert result["newly_collected_articles"] == result["newly_preprocessed_articles"] == 2
    assert result["reused_collected_articles"] == result["reused_preprocessed_articles"] == 0
    assert (args["output_dir"] / "query_articles.jsonl").read_bytes() == args["discovery_path"].read_bytes()
    assert (args["output_dir"] / "articles.jsonl").read_bytes() == Path(str(args["discovery_path"]) + ".articles.jsonl").read_bytes()
    assert [a["page_id"] for a in result["articles"]] == [1, 2]
    assert all(a["cutoff_satisfied"] and a["document"]["revision_id"] in {"101", "102"} for a in result["articles"])


def test_warm_reuse_avoids_collection_and_cleaning_and_is_deterministic(environment, monkeypatch):
    args = arguments(environment)
    environment["seed"]()
    cold = assembly.assemble(**args)
    monkeypatch.setattr(collection, "collect_wikipedia", lambda **k: pytest.fail("Unnecessary recollection"))
    monkeypatch.setattr(preprocessing, "_parser", lambda: pytest.fail("Unnecessary reparsing"))
    warm_args = {**args, "output_dir": environment["root"] / "warm"}
    warm = assembly.assemble(**warm_args)
    replay_args = {**args, "output_dir": environment["root"] / "replay"}
    assert assembly.assemble(**replay_args) == warm
    assert complete(warm_args) == warm
    assert warm["reused_collected_articles"] == warm["reused_preprocessed_articles"] == 2
    assert warm["newly_collected_articles"] == warm["newly_preprocessed_articles"] == 0
    assert warm["outputs"] == cold["outputs"]
    for name in (*assembly.OUTPUT_FILES, "manifest.json"):
        assert (warm_args["output_dir"] / name).read_bytes() == (replay_args["output_dir"] / name).read_bytes()


def test_mixed_warm_and_new_articles(environment):
    env = environment
    args = arguments(env, env["discover"](subjects=("Alpha",), name="first"))
    env["seed"]((1, 2, 3))
    assembly.assemble(**args)
    expanded = arguments(env, env["discover"](subjects=("Alpha", "Beta", "Gamma"), name="expanded"), "expanded_output")
    result = assembly.assemble(**expanded)
    assert complete(expanded) == result
    assert result["reused_collected_articles"] == result["reused_preprocessed_articles"] == 1
    assert result["newly_collected_articles"] == result["newly_preprocessed_articles"] == 2


def archive(env, pages=(1, 2), with_passages=True):
    root = env["root"] / "archive"
    root.mkdir()
    requests = write(root / "requests.jsonl", [{"page_id": page_id} for page_id in pages])
    source = root / "source.jsonl"
    collection.collect_wikipedia(input_path=requests, output_path=source, cache_dir=env["revision_cache"],
        allow_network=True, user_agent=UA, cutoff=CUTOFF, max_articles=len(pages))
    passages = root / "passages.jsonl"
    if with_passages:
        preprocessing.preprocess_corpus(corpus_path=source, output_path=passages, cache_dir=env["revision_cache"])
    return source, passages if with_passages else "-", env["revision_cache"]


@pytest.mark.parametrize("with_passages", [False, True])
def test_import_existing_batch_archive_without_modifying_original(environment, monkeypatch, with_passages):
    env = environment
    args = arguments(env)
    source = archive(env, with_passages=with_passages)
    args["revision_cache_dir"] = env["root"] / "import_cache"
    frozen = {p: p.read_bytes() for p in source[0].parent.iterdir()}
    monkeypatch.setattr(collection, "collect_wikipedia", lambda **k: pytest.fail("Archive recollected"))
    if with_passages:
        monkeypatch.setattr(preprocessing, "_parser", lambda: pytest.fail("Archive reprocessed"))
    result = assembly.assemble(**args, reuse_snapshots=[source])
    assert complete(args) == result
    assert result["reused_collected_articles"] == 2
    assert result["reused_preprocessed_articles"] == (2 if with_passages else 0)
    assert result["newly_preprocessed_articles"] == (0 if with_passages else 2)
    assert all(p.read_bytes() == data for p, data in frozen.items())


def test_mixed_import_and_missing_collection(environment):
    env = environment
    args = arguments(env)
    source = archive(env, pages=(1,))
    env["seed"]((2,))
    result = assembly.assemble(**args, reuse_snapshots=[source])
    assert complete(args) == result
    assert result["reused_collected_articles"] == result["newly_collected_articles"] == 1
    assert result["reused_preprocessed_articles"] == result["newly_preprocessed_articles"] == 1


def test_missing_revision_offline_retains_failure_and_prefix(environment):
    env = environment
    args = arguments(env)
    env["seed"]((1,))
    before = len(env["calls"])
    with pytest.raises(ValueError, match="incomplete collection"):
        assembly.assemble(**args)
    assert len(env["calls"]) == before
    assert not (args["output_dir"] / "manifest.json").exists()
    failure = json.loads((args["output_dir"] / "failure.json").read_bytes())
    assert failure["page_id"] == 2 and len(failure["completed_articles"]) == 1
    with pytest.raises(ValueError):
        complete(args)
    with pytest.raises(ValueError):
        assembly.assemble(**{**args, "output_dir": env["root"] / "retry"})
    assert not (env["root"] / "retry" / "manifest.json").exists()


def test_explicit_network_mode_uses_existing_collector_with_exact_cutoff(environment):
    env = environment
    args = arguments(env)
    before = len(env["calls"])
    result = assembly.assemble(**args, allow_network=True, user_agent=UA)
    assert complete(args) == result and len(env["calls"]) == before + 2
    assert all(p["rvstart"] == CUTOFF and p["rvdir"] == "older" for p in env["calls"][before:])


@pytest.mark.parametrize("kwargs", [{"cutoff": "2025-01-01T00:00:00Z"}, {"cutoff": "nonsense"},
    {"chunk_words": 0}, {"overlap_words": 200}, {"allow_network": True}, {"allow_network": 1}])
def test_invalid_configuration_fails_before_output(environment, kwargs):
    args = arguments(environment)
    with pytest.raises(ValueError):
        assembly.assemble(**args, **kwargs)
    assert not args["output_dir"].exists()


def test_explicit_cutoff_revalidation_uses_different_cache_and_store_identity(environment):
    env = environment
    args = arguments(env)
    env["seed"]()
    original = assembly.assemble(**args)
    changed = {**args, "output_dir": env["root"] / "changed"}
    cutoff = "2026-06-30T00:00:00Z"
    with pytest.raises(ValueError):
        assembly.assemble(**changed, cutoff=cutoff, revalidate_cutoff=True)
    env["seed"](cutoff=cutoff)
    changed["output_dir"] = env["root"] / "revalidated"
    changed["store_dir"] = env["root"] / "fresh_store"
    result = assembly.assemble(**changed, cutoff=cutoff, revalidate_cutoff=True)
    assert complete(changed) == result
    assert result["discovery_cutoff"] == CUTOFF and result["configuration"]["cutoff"] == cutoff
    assert {a["store_key"] for a in result["articles"]}.isdisjoint(a["store_key"] for a in original["articles"])


@pytest.mark.parametrize("damage", ["discovery", "discovery_cache"])
def test_bad_discovery_fails_before_output(environment, damage):
    env = environment
    args = arguments(env)
    target = args["discovery_path"] if damage == "discovery" else next(env["discovery_cache"].iterdir())
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        assembly.assemble(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("damage", ["corpus", "collection", "passages", "diagnostics", "pp_manifest", "revision_cache", "requests"])
def test_corrupt_store_is_never_refreshed_or_silently_reprocessed(environment, monkeypatch, damage):
    env = environment
    args = arguments(env)
    env["seed"]()
    result = assembly.assemble(**args)
    entry = result["articles"][0]
    slot = args["store_dir"] / entry["store_key"]
    targets = {"corpus": slot / "corpus.jsonl", "collection": slot / "corpus.jsonl.collection.json",
        "passages": slot / entry["passages_file"], "diagnostics": slot / (entry["passages_file"] + ".diagnostics.jsonl"),
        "pp_manifest": slot / (entry["passages_file"] + ".manifest.json"),
        "revision_cache": next(env["revision_cache"].iterdir()), "requests": slot / "requests.jsonl"}
    targets[damage].write_bytes(b"damaged")
    monkeypatch.setattr(collection, "collect_wikipedia", lambda **k: pytest.fail("Corrupt artifact recollected"))
    monkeypatch.setattr(preprocessing, "preprocess_corpus", lambda **k: pytest.fail("Corrupt artifact reprocessed"))
    retry = {**args, "output_dir": env["root"] / "new_output"}
    with pytest.raises(ValueError):
        assembly.assemble(**retry)
    assert targets[damage].read_bytes() == b"damaged"
    assert not (retry["output_dir"] / "manifest.json").exists()


def test_new_chunk_configuration_reuses_collection_only(environment):
    env = environment
    args = arguments(env)
    env["seed"]()
    first = assembly.assemble(**args)
    changed = {**args, "output_dir": env["root"] / "small_chunks"}
    second = assembly.assemble(**changed, chunk_words=3, overlap_words=1)
    assert complete(changed) == second
    assert second["reused_collected_articles"] == second["newly_preprocessed_articles"] == 2
    assert second["articles"][0]["passages_file"] != first["articles"][0]["passages_file"]


def test_incompatible_or_ambiguous_reuse_source_is_rejected(environment):
    env = environment
    args = arguments(env)
    source = archive(env)
    with pytest.raises(ValueError, match="Ambiguous"):
        assembly.assemble(**args, reuse_snapshots=[source, source])
    with pytest.raises(ValueError, match="cutoff/edition mismatch"):
        assembly.assemble(**args, reuse_snapshots=[source], cutoff="2026-06-30T00:00:00Z", revalidate_cutoff=True)
    assert not args["output_dir"].exists()


def test_valid_import_with_other_chunk_settings_regenerates_passages_only(environment):
    env = environment
    args = arguments(env)
    source = archive(env)
    result = assembly.assemble(**args, reuse_snapshots=[source], chunk_words=10, overlap_words=2)
    assert complete(args) == result
    assert result["reused_collected_articles"] == result["newly_preprocessed_articles"] == 2
    assert result["reused_preprocessed_articles"] == result["newly_collected_articles"] == 0


@pytest.mark.parametrize("field", ["article_count", "reused_collected_articles", "schema_version", "outputs", "articles"])
def test_completion_manifest_tampering_rejected(environment, field):
    args = arguments(environment)
    environment["seed"]()
    assembly.assemble(**args)
    path = args["output_dir"] / "manifest.json"
    value = json.loads(path.read_bytes())
    value[field] = [] if field in {"outputs", "articles"} else "incorrect"
    path.write_bytes(corpus._json_bytes(value))
    with pytest.raises(ValueError):
        complete(args)


@pytest.mark.parametrize("filename", ["query_articles.jsonl", "articles.jsonl", "corpus.jsonl", "passages.jsonl"])
def test_rehashed_output_tampering_still_rejected(environment, filename):
    args = arguments(environment)
    environment["seed"]()
    assembly.assemble(**args)
    output = args["output_dir"] / filename
    output.write_bytes(output.read_bytes() + b"\n")
    path = args["output_dir"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["outputs"][filename] = {"sha256": corpus._sha256(output.read_bytes()), "bytes": output.stat().st_size}
    path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises(ValueError):
        complete(args)


def test_no_clobber_and_completion_publication_failure(environment, monkeypatch):
    env = environment
    args = arguments(env)
    args["output_dir"].mkdir()
    with pytest.raises(ValueError, match="new"):
        assembly.assemble(**args)
    args["output_dir"] = env["root"] / "fresh"
    env["seed"]()
    original = corpus._publish
    def fail(path, data):
        if Path(path) == args["output_dir"] / "manifest.json":
            raise OSError("Synthetic completion failure")
        original(path, data)
    monkeypatch.setattr(corpus, "_publish", fail)
    with pytest.raises(OSError):
        assembly.assemble(**args)
    assert (args["output_dir"] / "failure.json").exists()
    with pytest.raises(ValueError):
        complete(args)


def test_interruption_retains_diagnostics_without_completion(environment, monkeypatch):
    args = arguments(environment)
    monkeypatch.setattr(collection, "collect_wikipedia", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        assembly.assemble(**args)
    assert json.loads((args["output_dir"] / "failure.json").read_bytes())["status"] == "interrupted"
    assert not (args["output_dir"] / "manifest.json").exists()


def test_unresolved_empty_inventory_is_valid_and_never_loads_parser(environment, monkeypatch):
    env = environment
    args = arguments(env, env["discover"](subjects=("Missing",)))
    monkeypatch.setattr(preprocessing, "_parser", lambda: pytest.fail("No articles to preprocess"))
    result = assembly.assemble(**args)
    assert complete(args) == result
    assert result["article_count"] == result["passage_count"] == 0
    assert result["unresolved_query_count"] == 1


def test_cli_default_offline_and_controlled_errors(environment, capsys):
    env = environment
    args = arguments(env)
    cli = ["--discovery", str(args["discovery_path"]), "--discovery-cache-dir", str(env["discovery_cache"]),
        "--output-dir", str(args["output_dir"]), "--revision-cache-dir", str(env["revision_cache"])]
    with pytest.raises(SystemExit) as error:
        assembly.main(cli)
    assert error.value.code == 2 and "Traceback" not in capsys.readouterr().err
    with pytest.raises(SystemExit) as help_result:
        assembly.main(["--help"])
    assert help_result.value.code == 0


def test_resume_valid_collection_after_preprocessing_interruption(environment, monkeypatch):
    env = environment
    args = arguments(env)
    env["seed"]()
    original = preprocessing.preprocess_corpus
    monkeypatch.setattr(preprocessing, "preprocess_corpus", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        assembly.assemble(**args)
    monkeypatch.setattr(preprocessing, "preprocess_corpus", original)
    retry = {**args, "output_dir": env["root"] / "resumed"}
    result = assembly.assemble(**retry)
    assert complete(retry) == result
    assert result["reused_collected_articles"] == result["newly_collected_articles"] == 1
    assert result["newly_preprocessed_articles"] == 2


def test_import_cache_conflict_is_not_overwritten(environment):
    env = environment
    args = arguments(env)
    source = archive(env)
    cache = env["root"] / "different_snapshot"
    cache.mkdir()
    for old in env["revision_cache"].iterdir():
        envelope = json.loads(old.read_bytes())
        envelope["retrieved_at"] = "2026-11-02T00:00:00Z"
        (cache / old.name).write_bytes(corpus._json_bytes(envelope))
    frozen = {p: p.read_bytes() for p in cache.iterdir()}
    with pytest.raises(ValueError, match="Conflicting immutable"):
        assembly.assemble(**{**args, "revision_cache_dir": cache}, reuse_snapshots=[source])
    assert all(p.read_bytes() == data for p, data in frozen.items())
    assert not (args["output_dir"] / "manifest.json").exists()


@pytest.mark.parametrize("damage", ["source_links", "canonical_input"])
def test_rehashed_acquisition_provenance_tampering_rejected(environment, damage):
    args = arguments(environment)
    environment["seed"]()
    assembly.assemble(**args)
    output = args["output_dir"]
    if damage == "source_links":
        path = output / "corpus.jsonl.collection.json"
        value = json.loads(path.read_bytes())
        value["source_snapshots"] = []
        path.write_bytes(corpus._json_bytes(value))
    else:
        path = output / "corpus.jsonl.manifest.json"
        value = json.loads(path.read_bytes())
        value["input_sha256"] = "0" * 64
        path.write_bytes(corpus._json_bytes(value))
        report_path = output / "corpus.jsonl.collection.json"
        report = json.loads(report_path.read_bytes())
        report["manifest_sha256"] = corpus._sha256(path.read_bytes())
        report_path.write_bytes(corpus._json_bytes(report))
    pp_path = output / "passages.jsonl.manifest.json"
    pp_manifest = json.loads(pp_path.read_bytes())
    for key, name in (("manifest", "corpus.jsonl.manifest.json"), ("collection", "corpus.jsonl.collection.json")):
        data = (output / name).read_bytes()
        pp_manifest["inputs"][key] = {"sha256": corpus._sha256(data), "bytes": len(data)}
    pp_path.write_bytes(corpus._json_bytes(pp_manifest))
    path = output / "manifest.json"
    manifest = json.loads(path.read_bytes())
    for name in assembly.OUTPUT_FILES:
        data = (output / name).read_bytes()
        manifest["outputs"][name] = {"sha256": corpus._sha256(data), "bytes": len(data)}
    path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises(ValueError, match="source snapshots|input fingerprint"):
        complete(args)


def test_cli_success_and_mutually_exclusive_modes(environment, capsys):
    env = environment
    args = arguments(env)
    env["seed"]()
    cli = ["--discovery", str(args["discovery_path"]), "--discovery-cache-dir", str(env["discovery_cache"]),
        "--output-dir", str(args["output_dir"]), "--store-dir", str(args["store_dir"]),
        "--revision-cache-dir", str(env["revision_cache"]), "--offline-only", "--no-clobber"]
    with pytest.raises(SystemExit) as error:
        assembly.main([*cli, "--allow-network"])
    assert error.value.code == 2
    assert assembly.main(cli) == 0 and "Queries: 3; articles: 2" in capsys.readouterr().out
    complete(args)


def test_empty_assembly_still_requires_explicit_resolved_cutoff(environment):
    env = environment
    args = arguments(env, env["discover"](subjects=("Missing",)))
    assembly.assemble(**args)
    path = args["output_dir"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["configuration"]["cutoff"] = None
    path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises(ValueError, match="explicit resolved cutoff"):
        complete(args)


def change_cached_pages(env, transform, page_ids=(2,)):
    """Prepare synthetic server responses before use, never alter real archives."""
    for path in env["revision_cache"].iterdir():
        envelope = json.loads(path.read_bytes())
        payload = json.loads(envelope["response_body"])
        page = payload["query"]["pages"][0]
        if page["pageid"] in page_ids:
            transform(page)
            body = corpus._json_bytes(payload)
            envelope.update(response_body=body.decode("utf-8"), response_sha256=corpus._sha256(body))
            path.write_bytes(corpus._json_bytes(envelope))


def rehash_aggregate(output, store=None):
    """Update every dependent hash, so tests exercise semantics, not stale hashes."""
    report_path = output / "corpus.jsonl.collection.json"
    report = json.loads(report_path.read_bytes())
    if store is not None:
        path = output / "manifest.json"
        manifest = json.loads(path.read_bytes())
        for entry in manifest["articles"]:
            entry["artifacts"] = assembly._fingerprints({
                name: store / entry["store_key"] / name for name in entry["artifacts"]})
        report["source_snapshots"] = [entry["artifacts"] for entry in manifest["articles"]]
        path.write_bytes(corpus._json_bytes(manifest))
    report["manifest_sha256"] = corpus._sha256((output / "corpus.jsonl.manifest.json").read_bytes())
    report_path.write_bytes(corpus._json_bytes(report))
    passage_path = output / "passages.jsonl.manifest.json"
    passages = json.loads(passage_path.read_bytes())
    for key, name in (("manifest", "corpus.jsonl.manifest.json"), ("collection", "corpus.jsonl.collection.json")):
        data = (output / name).read_bytes()
        passages["inputs"][key] = {"sha256": corpus._sha256(data), "bytes": len(data)}
    passage_path.write_bytes(corpus._json_bytes(passages))
    path = output / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["outputs"] = assembly._fingerprints({name: output / name for name in assembly.OUTPUT_FILES})
    path.write_bytes(corpus._json_bytes(manifest))


@pytest.mark.parametrize("corpus_kind", ["ordinary", "empty", "zero_passages"])
@pytest.mark.parametrize("artifact,field,value", [
    ("corpus.jsonl.manifest.json", "artifact_role", "unrelated"),
    ("corpus.jsonl.manifest.json", "assembly_policy", "unsupported-v99"),
    ("corpus.jsonl.manifest.json", "ingestion_policy", "unsupported-v99"),
    ("corpus.jsonl.manifest.json", "id_policy", "unsupported-v99"),
    ("corpus.jsonl.manifest.json", "duplicate_policy", "unsupported-v99"),
    ("passages.jsonl.manifest.json", "artifact_role", "unrelated"),
    ("passages.jsonl.manifest.json", "assembly_policy", "unsupported-v99"),
    ("passages.jsonl.manifest.json", "code_sha256", "0" * 64),
    ("passages.jsonl.manifest.json", "preprocessing_version", "unsupported-v99"),
    ("passages.jsonl.manifest.json", "configuration", {"chunk_words": 999, "overlap_words": 40,
        "word_definition": "Unicode non-whitespace runs", "section_local": True}),
    ("passages.jsonl.manifest.json", "configuration", {"chunk_words": 200, "overlap_words": 0,
        "word_definition": "Unicode non-whitespace runs", "section_local": True}),
    ("passages.jsonl.manifest.json", "passage_count", 999),
])
def test_aggregate_semantics_even_with_recomputed_hashes(environment, corpus_kind, artifact, field, value):
    env = environment
    source = env["discover"](subjects=("Missing",)) if corpus_kind == "empty" else env["discover"]()
    args = arguments(env, source)
    env["seed"]()
    if corpus_kind == "zero_passages":
        change_cached_pages(env, lambda p: p["revisions"][0]["slots"]["main"].update(
            content="<ref>Reference only</ref><!-- maintenance -->"), page_ids=(1, 2))
    assembly.assemble(**args)
    path = args["output_dir"] / artifact
    data = json.loads(path.read_bytes())
    data[field] = value
    path.write_bytes(corpus._json_bytes(data))
    rehash_aggregate(args["output_dir"])
    with pytest.raises(ValueError):
        complete(args)


def test_recorded_assembly_writer_hash_supports_archived_implementations(environment):
    args = arguments(environment)
    environment["seed"]()
    assembly.assemble(**args)
    path = args["output_dir"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["implementation_sha256"]["rag_corpus_assembly.py"] = "a" * 64
    path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises(ValueError, match="writer"):
        complete(args)
    pp_path = args["output_dir"] / "passages.jsonl.manifest.json"
    pp_manifest = json.loads(pp_path.read_bytes())
    pp_manifest["code_sha256"] = "a" * 64
    pp_path.write_bytes(corpus._json_bytes(pp_manifest))
    rehash_aggregate(args["output_dir"])
    assert complete(args)["implementation_sha256"]["rag_corpus_assembly.py"] == "a" * 64


TERMINAL_CASES = [
    (lambda p: p.pop("revisions"), "unavailable", "no_eligible_revision"),
    (lambda p: p.update(revisions=[]), "unavailable", "no_eligible_revision"),
    (lambda p: (p.pop("revisions"), p.update(missing=True)), "unavailable", "missing_page"),
    (lambda p: p["revisions"][0]["slots"]["main"].update(contentmodel="json"), "excluded", "unsupported_source"),
    (lambda p: p["revisions"][0]["slots"]["main"].update(content="  "), "excluded", "empty_source"),
    (lambda p: p.update(pageprops={"disambiguation": ""}), "excluded", "ambiguous_or_directive_source"),
]


@pytest.mark.parametrize("transform,status,reason", TERMINAL_CASES)
def test_verified_terminal_outcomes_continue_and_replay_offline(environment, monkeypatch, transform, status, reason):
    env = environment
    args = arguments(env)
    env["seed"]()
    change_cached_pages(env, transform)
    frozen_cache = {p.name: p.read_bytes() for p in env["revision_cache"].iterdir()}
    monkeypatch.setattr(collection, "_fetch", lambda *a, **k: pytest.fail("Offline means no HTTP"))
    cold = assembly.assemble(**args)
    assert complete(args) == cold
    assert cold["selected_article_count"] == 2 and cold["collected_article_count"] == 1
    assert cold["terminal_unavailable_articles"] == (status == "unavailable")
    assert cold["policy_excluded_articles"] == (status == "excluded")
    entry = cold["articles"][1]
    assert entry["outcome"] == status and entry["reason"] == reason
    assert entry["outcome_evidence"]["cache"]["sha256"] in {
        corpus._sha256(data) for data in frozen_cache.values()}
    assert cold["newly_recorded_terminal_articles"] == 1
    warm_args = {**args, "output_dir": env["root"] / "warm"}
    warm = assembly.assemble(**warm_args)
    assert complete(warm_args) == warm
    assert warm["reused_terminal_articles"] == 1
    assert cold["outputs"] == warm["outputs"]
    repeat = {**args, "output_dir": env["root"] / "repeat"}
    assert assembly.assemble(**repeat) == warm
    assert (repeat["output_dir"] / "manifest.json").read_bytes() == (warm_args["output_dir"] / "manifest.json").read_bytes()
    assert all(p.read_bytes() == frozen_cache[p.name] for p in env["revision_cache"].iterdir())


@pytest.mark.parametrize("transform", [
    lambda p: (p.pop("revisions"), p.update(missing=False)),
    lambda p: (p.pop("revisions"), p.update(missing="")),
    lambda p: (p.pop("revisions"), p.update(missing={})),
    lambda p: p.update(invalid=False),
    lambda p: p.update(redirect=""),
    lambda p: p.update(revisions=None),
    lambda p: p.update(revisions={}),
    lambda p: p.update(pageprops={"disambiguation": False}),
    lambda p: p.update(pageprops={"disambiguation": {}}),
    lambda p: p.update(pageprops={"disambiguation": None}),
    lambda p: p.update(missing=True),  # Contradicts existing revisions.
    lambda p: p.pop("pagelanguage"),
    lambda p: p.update(pagelanguage="el"),
    lambda p: p.update(pageid=42),
    lambda p: p["revisions"][0].pop("revid"),
    lambda p: p["revisions"][0].update(timestamp="2027-01-01T00:00:00Z"),
    lambda p: p["revisions"][0]["slots"]["main"].pop("content"),
    lambda p: p["revisions"][0]["slots"]["main"].update(content=None),
    lambda p: p["revisions"][0]["slots"]["main"].update(contentmodel=""),
])
def test_malformed_or_inconsistent_outcomes_are_never_terminal(environment, transform):
    env = environment
    args = arguments(env)
    env["seed"]()
    change_cached_pages(env, transform)
    with pytest.raises(ValueError):
        assembly.assemble(**args)
    assert not (args["output_dir"] / "manifest.json").exists()
    assert (args["output_dir"] / "failure.json").exists()
    config = assembly._configuration(json.loads(Path(str(args["discovery_path"]) + ".manifest.json").read_bytes()),
        None, False, 200, 40)
    _, slot, _, _ = assembly._store_paths(args["store_dir"], 2, config)
    assert not (slot / "outcome.json").exists()


def test_all_articles_unavailable_can_complete_without_a_parser(environment, monkeypatch):
    env = environment
    args = arguments(env)
    env["seed"]()
    change_cached_pages(env, lambda p: p.pop("revisions"), page_ids=(1, 2))
    monkeypatch.setattr(preprocessing, "_parser", lambda: pytest.fail("No document to preprocess"))
    result = assembly.assemble(**args)
    assert complete(args) == result
    assert result["article_count"] == result["terminal_unavailable_articles"] == 2
    assert result["document_count"] == result["passage_count"] == 0
    assert result["query_availability_counts"]["resolved_no_usable_evidence"] == 3


@pytest.mark.parametrize("identical_source", [True, False])
def test_duplicate_text_preserves_both_sources_and_eligibility(environment, identical_source):
    env = environment
    args = arguments(env)
    env["seed"]()
    change_cached_pages(env, lambda p: p["revisions"][0]["slots"]["main"].update(
        content=("" if identical_source else f'<!-- source {p["pageid"]} -->') + "Same clean text."), page_ids=(1, 2))
    result = assembly.assemble(**args)
    rows, checked = assembly.load_assembly(**args)
    assert checked == result and len(rows) == 2
    assert len({r["passage_id"] for r in rows}) == len({r["doc_id"] for r in rows}) == 2
    assert len({r["text"] for r in rows}) == 1
    availability = [json.loads(line) for line in (args["output_dir"] / "query_availability.jsonl").read_text().splitlines()]
    assert [r["candidate_page_ids"] for r in availability] == [[1], [2], [1]]
    assert availability[0]["eligible_document_ids"] == availability[2]["eligible_document_ids"]
    assert availability[0]["eligible_document_ids"] != availability[1]["eligible_document_ids"]
    if identical_source:
        with pytest.raises(corpus.CorpusError, match="duplicate content"):
            corpus.load_documents(args["output_dir"] / "corpus.jsonl")
        with pytest.raises(corpus.CorpusError, match="duplicate content"):
            corpus.ingest_corpus(args["output_dir"] / "corpus.jsonl", env["root"] / "standalone.jsonl")


def test_query_availability_partial_unresolved_and_zero_passages(environment, monkeypatch):
    env = environment
    original = collection._fetch
    def search_with_beta(endpoint, params, user_agent, delay):
        response = original(endpoint, params, user_agent, delay)
        if params.get("list") == "search" and not params["srsearch"].startswith("Missing"):
            body = {"query": {"search": [{"pageid": 2, "title": "Beta", "ns": 0}]}}
        elif "pageids" in params and "rvprop" not in params:
            body = {"query": {"pages": [{"pageid": 2, "title": "Beta", "ns": 0}]}}
        else:
            return response
        data = corpus._json_bytes(body)
        return {**response, "response_body": data.decode(), "response_sha256": corpus._sha256(data)}
    # The fixture's original fetch supports direct/revision/search, not batched metadata.
    def fetch(endpoint, params, user_agent, delay):
        if "pageids" in params and "rvprop" not in params:
            template = original(endpoint, {**params, "list": "search"}, user_agent, delay)
            body = corpus._json_bytes({"query": {"pages": [{"pageid": 2, "title": "Beta", "ns": 0}]}})
            return {**template, "request": {"endpoint": endpoint, "params": params},
                "response_body": body.decode(), "response_sha256": corpus._sha256(body)}
        return search_with_beta(endpoint, params, user_agent, delay)
    monkeypatch.setattr(collection, "_fetch", fetch)
    args = arguments(env, env["discover"](subjects=("Alpha", "Beta", "Missing")))
    env["seed"]()
    change_cached_pages(env, lambda p: p["revisions"][0]["slots"]["main"].update(content="<ref>Only citation</ref>"))
    result = assembly.assemble(**args)
    assert complete(args) == result
    assert result["zero_passage_articles"] == 1 and result["usable_articles"] == 1
    rows = [json.loads(line) for line in (args["output_dir"] / "query_availability.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["partial_candidate_availability", "resolved_no_usable_evidence", "unresolved_discovery"]
    assert rows[0]["candidate_page_ids"] == [1, 2] and rows[0]["available_article_ids"] == [1]
    assert rows[0]["unavailable_article_ids"] == [2] and rows[0]["eligible_passage_count"] > 0


@pytest.mark.parametrize("damage", ["terminal_reason", "terminal_evidence", "availability_association", "availability_count",
    "terminal_unavailable_articles", "zero_passage_articles", "usable_articles", "selected_article_count", "outcome_policy"])
def test_outcome_and_availability_tampering_fails_closed(environment, damage):
    env = environment
    args = arguments(env)
    env["seed"]()
    change_cached_pages(env, lambda p: p.pop("revisions"))
    result = assembly.assemble(**args)
    path = args["output_dir"] / "manifest.json"
    if damage.startswith("terminal_") and damage in ("terminal_reason", "terminal_evidence"):
        if damage == "terminal_reason":
            result["articles"][1]["reason"] = "missing_page"
        else:
            slot = args["store_dir"] / result["articles"][1]["store_key"] / "outcome.json"
            data = json.loads(slot.read_bytes()); data["reason"] = "missing_page"
            slot.write_bytes(corpus._json_bytes(data))
    elif damage.startswith("availability_"):
        target = args["output_dir"] / "query_availability.jsonl"
        data = [json.loads(line) for line in target.read_text().splitlines()]
        if damage == "availability_count":
            data[0]["eligible_passage_count"] += 1
        else:
            data[0]["candidate_page_ids"] = [2]
        write(target, data)
    else:
        result[damage] = "unsupported" if damage == "outcome_policy" else 999
    path.write_bytes(corpus._json_bytes(result))
    rehash_aggregate(args["output_dir"])
    with pytest.raises(ValueError):
        complete(args)


def test_transient_failure_retains_details_and_recovers_with_fresh_store(environment, monkeypatch):
    env = environment
    args = arguments(env)
    env["seed"]((1,))
    original = collection._fetch
    def unavailable(*a, **k):
        raise collection.NetworkFailure("Synthetic HTTP 429", details={"http_status": 429, "retry_after": "60"})
    monkeypatch.setattr(collection, "_fetch", unavailable)
    with pytest.raises(ValueError, match="incomplete collection"):
        assembly.assemble(**args, allow_network=True, user_agent=UA)
    assert not (args["output_dir"] / "manifest.json").exists()
    reports = [json.loads(p.read_bytes()) for p in args["store_dir"].glob("*/corpus.jsonl.collection.json")]
    failed = next(r for r in reports if r["status"] == "incomplete")
    assert failed["records"][0]["network_details"]["retry_after"] == "60"
    failure = json.loads((args["output_dir"] / "failure.json").read_bytes())
    first = failure["completed_articles"][0]
    slot = args["store_dir"] / first["store_key"]
    archive_source = (slot / "corpus.jsonl", slot / first["passages_file"], env["revision_cache"])
    frozen = {p: p.read_bytes() for p in args["store_dir"].rglob("*") if p.is_file()}
    monkeypatch.setattr(collection, "_fetch", original)
    retry = {**args, "output_dir": env["root"] / "recovered", "store_dir": env["root"] / "fresh_store"}
    result = assembly.assemble(**retry, allow_network=True, user_agent=UA, reuse_snapshots=[archive_source])
    assert complete(retry) == result
    assert result["reused_collected_articles"] == result["newly_collected_articles"] == 1
    assert all(p.read_bytes() == data for p, data in frozen.items())


def test_duplicate_assembly_archive_can_be_imported_with_both_sources(environment, monkeypatch):
    env = environment
    args = arguments(env)
    env["seed"]()
    change_cached_pages(env, lambda p: p["revisions"][0]["slots"]["main"].update(content="Identical source."), page_ids=(1, 2))
    assembly.assemble(**args)
    imported = {**args, "store_dir": env["root"] / "new_store", "revision_cache_dir": env["root"] / "new_cache",
        "output_dir": env["root"] / "new_output"}
    monkeypatch.setattr(collection, "collect_wikipedia", lambda **k: pytest.fail("Archive recollected"))
    monkeypatch.setattr(preprocessing, "_parser", lambda: pytest.fail("Archive reprocessed"))
    result = assembly.assemble(**imported, reuse_snapshots=[(args["output_dir"] / "corpus.jsonl",
        args["output_dir"] / "passages.jsonl", env["revision_cache"])])
    assert complete(imported) == result
    assert result["reused_collected_articles"] == result["reused_preprocessed_articles"] == 2
    assert (imported["output_dir"] / "corpus.jsonl").read_bytes() == (args["output_dir"] / "corpus.jsonl").read_bytes()
    assert (imported["output_dir"] / "passages.jsonl").read_bytes() == (args["output_dir"] / "passages.jsonl").read_bytes()


def test_duplicate_opt_in_cannot_be_applied_to_an_unmarked_native_snapshot(environment):
    env = environment
    source, _, cache = archive(env)
    with pytest.raises(ValueError, match="opt-in"):
        preprocessing.load_snapshot(source, cache_dir=cache, allow_duplicate_content=True)


@pytest.mark.parametrize("corruption", ["cache", "cache_missing", "outcome", "collection"])
def test_terminal_store_corruption_is_never_refreshed(environment, monkeypatch, corruption):
    env = environment
    args = arguments(env)
    env["seed"]()
    change_cached_pages(env, lambda p: p.pop("revisions"))
    result = assembly.assemble(**args)
    entry = result["articles"][1]
    slot = args["store_dir"] / entry["store_key"]
    cache = env["revision_cache"] / (entry["outcome_evidence"]["cache"]["key"] + ".json")
    target = {"cache": cache, "cache_missing": cache, "outcome": slot / "outcome.json",
        "collection": slot / "corpus.jsonl.collection.json"}[corruption]
    if corruption == "cache_missing":
        target.unlink()
    else:
        target.write_bytes(b"corrupt")
    monkeypatch.setattr(collection, "_fetch", lambda *a, **k: pytest.fail("Corruption must not trigger HTTP"))
    with pytest.raises(ValueError):
        complete(args)
    fresh = {**args, "output_dir": env["root"] / "retry"}
    with pytest.raises(ValueError):
        assembly.assemble(**fresh)
    assert not (fresh["output_dir"] / "manifest.json").exists()


@pytest.mark.parametrize("field,value", [("text_representation", "current-extract"),
    ("max_articles", True), ("request_policy", {}), ("request_policy", {"minimum_delay_seconds": 0}),
    ("request_policy", {"minimum_delay_seconds": 10 ** 400})])
def test_terminal_collection_policy_metadata_is_validated(environment, field, value):
    env = environment
    args = arguments(env)
    env["seed"]()
    change_cached_pages(env, lambda p: p.pop("revisions"))
    result = assembly.assemble(**args)
    report = args["store_dir"] / result["articles"][1]["store_key"] / "corpus.jsonl.collection.json"
    data = json.loads(report.read_bytes()); data[field] = value
    report.write_bytes(corpus._json_bytes(data))
    with pytest.raises(ValueError):
        complete(args)


@pytest.mark.parametrize("field", ["id_policy", "duplicate_policy", "ingestion_policy"])
def test_unsupported_native_source_manifest_policies_fail_before_output(environment, field):
    env = environment
    args = arguments(env)
    source, _, cache = archive(env, with_passages=False)
    path = Path(str(source) + ".manifest.json")
    value = json.loads(path.read_bytes()); value[field] = "unsupported-v99"
    path.write_bytes(corpus._json_bytes(value))
    report_path = Path(str(source) + ".collection.json")
    report = json.loads(report_path.read_bytes()); report["manifest_sha256"] = corpus._sha256(path.read_bytes())
    report_path.write_bytes(corpus._json_bytes(report))
    with pytest.raises(ValueError, match="Unsupported source corpus policy"):
        assembly.assemble(**args, reuse_snapshots=[(source, "-", cache)])
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("field,value", [("page_id", 999), ("revision_id", "999"),
    ("revision_timestamp", "2027-01-01T00:00:00Z"), ("cutoff_satisfied", False), ("title", "Other article"),
    ("page_id", True), ("revision_id", None), ("revision_timestamp", None), ("cutoff_satisfied", 1),
    ("title", None)])
def test_terminal_report_metadata_rejected_with_all_dependent_hashes_recomputed(environment, field, value):
    args = arguments(environment)
    environment["seed"]()
    change_cached_pages(environment, lambda p: p["revisions"][0]["slots"]["main"].update(content=""))
    result = assembly.assemble(**args)
    report_path = args["store_dir"] / result["articles"][1]["store_key"] / "corpus.jsonl.collection.json"
    report = json.loads(report_path.read_bytes())
    report["records"][0][field] = value
    report_path.write_bytes(corpus._json_bytes(report))
    rehash_aggregate(args["output_dir"], args["store_dir"])
    with pytest.raises(assembly.AssemblyError, match="report metadata differs"):
        complete(args)
    warm = {**args, "output_dir": environment["root"] / "warm"}
    with pytest.raises(assembly.AssemblyError, match="report metadata differs"):
        assembly.assemble(**warm)
    assert not (warm["output_dir"] / "manifest.json").exists()


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("field", ["page_id", "title", "revision_id", "revision_timestamp", "cutoff_satisfied"])
def test_terminal_report_requires_outcome_specific_metadata_presence(environment, missing, field):
    args = arguments(environment)
    environment["seed"]()
    def absent(page):
        page.pop("revisions")
        if missing:
            page["missing"] = True
    change_cached_pages(environment, absent)
    result = assembly.assemble(**args)
    assert complete(args) == result
    evidence = result["articles"][1]["outcome_evidence"]
    assert evidence["revision_id"] is evidence["revision_timestamp"] is None
    path = args["store_dir"] / result["articles"][1]["store_key"] / "corpus.jsonl.collection.json"
    data = json.loads(path.read_bytes())
    record = data["records"][0]
    if field in record:
        del record[field]  # Native metadata cannot be omitted when produced.
    else:
        record[field] = None  # Absent native metadata cannot be fabricated as null.
    path.write_bytes(corpus._json_bytes(data))
    rehash_aggregate(args["output_dir"], args["store_dir"])
    with pytest.raises(assembly.AssemblyError, match="report metadata differs"):
        complete(args)


CONTINUATION = {"continue": "||", "rvcontinue": "20260601000000|102"}


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("revision_form", ["absent", "empty"])
def test_empty_continuation_never_publishes_completion_live_or_offline(environment, monkeypatch, cached, revision_form):
    env = environment
    args = arguments(env)
    original = collection._fetch
    def incomplete(endpoint, params, user_agent, delay):
        response = original(endpoint, params, user_agent, delay)
        if "rvprop" in params and params.get("pageids") == "2":
            body = json.loads(response["response_body"])
            page = body["query"]["pages"][0]
            if revision_form == "absent":
                page.pop("revisions")
            else:
                page["revisions"] = []
            body["continue"] = CONTINUATION
            data = corpus._json_bytes(body)
            response.update(response_body=data.decode(), response_sha256=corpus._sha256(data))
        return response
    monkeypatch.setattr(collection, "_fetch", incomplete)
    if cached:
        env["seed"]()
        monkeypatch.setattr(collection, "_fetch", lambda *a, **k: pytest.fail("Offline replay made HTTP request"))
    with pytest.raises(assembly.AssemblyError, match="unresolved continuation"):
        assembly.assemble(**args, allow_network=not cached, user_agent=UA if not cached else None)
    assert not (args["output_dir"] / "manifest.json").exists()
    assert (args["output_dir"] / "failure.json").exists()
    reports = [json.loads(p.read_bytes()) for p in args["store_dir"].glob("*/corpus.jsonl.collection.json")]
    failed = next(r for r in reports if r["status"] == "incomplete")
    assert failed["records"][0]["request"] == {"page_id": 2}
    assert not list(args["store_dir"].glob("*/outcome.json"))
    frozen = {p: p.read_bytes() for p in env["revision_cache"].iterdir()}
    calls = len(env["calls"])
    monkeypatch.setattr(collection, "_fetch", lambda *a, **k: pytest.fail("Offline recovery made HTTP request"))
    replay = {**args, "output_dir": env["root"] / "replay", "store_dir": env["root"] / "fresh_store"}
    with pytest.raises(assembly.AssemblyError, match="unresolved continuation"):
        assembly.assemble(**replay)
    assert not (replay["output_dir"] / "manifest.json").exists()
    assert len(env["calls"]) == calls
    assert all(p.read_bytes() == data for p, data in frozen.items())


@pytest.mark.parametrize("continuation", [None, {}, {"continue": "||"},
    {"continue": "||", "rvcontinue": False}, {"continue": "||", "rvcontinue": " "}])
def test_malformed_cached_continuation_cannot_publish_completion(environment, monkeypatch, continuation):
    env = environment
    args = arguments(env)
    env["seed"]()
    for path in env["revision_cache"].iterdir():
        envelope = json.loads(path.read_bytes())
        body = json.loads(envelope["response_body"])
        page = body["query"]["pages"][0]
        if page["pageid"] != 2:
            continue
        page["revisions"] = []
        body["continue"] = continuation
        raw = corpus._json_bytes(body)
        envelope.update(response_body=raw.decode(), response_sha256=corpus._sha256(raw))
        path.write_bytes(corpus._json_bytes(envelope))
    frozen = {p: p.read_bytes() for p in env["revision_cache"].iterdir()}
    monkeypatch.setattr(collection, "_fetch", lambda *a, **k: pytest.fail("Malformed cache must not trigger HTTP"))
    with pytest.raises(assembly.AssemblyError, match="Malformed revision continuation"):
        assembly.assemble(**args)
    assert not (args["output_dir"] / "manifest.json").exists()
    assert not list(args["store_dir"].glob("*/outcome.json"))
    assert all(p.read_bytes() == data for p, data in frozen.items())


def test_valid_revision_continuation_preserves_offline_cold_warm_reproducibility(environment, monkeypatch):
    env = environment
    args = arguments(env)
    env["seed"]()
    for path in env["revision_cache"].iterdir():
        envelope = json.loads(path.read_bytes())
        body = json.loads(envelope["response_body"])
        body["continue"] = CONTINUATION
        raw = corpus._json_bytes(body)
        envelope.update(response_body=raw.decode(), response_sha256=corpus._sha256(raw))
        path.write_bytes(corpus._json_bytes(envelope))
    monkeypatch.setattr(collection, "_fetch", lambda *a, **k: pytest.fail("Offline means no HTTP"))
    cold = assembly.assemble(**args)
    assert complete(args) == cold
    warm_args = {**args, "output_dir": env["root"] / "warm"}
    warm = assembly.assemble(**warm_args)
    assert complete(warm_args) == warm
    assert warm["outputs"] == cold["outputs"]
    assert warm["reused_collected_articles"] == 2
    replay_args = {**args, "output_dir": env["root"] / "replay"}
    assert assembly.assemble(**replay_args) == warm
    assert (warm_args["output_dir"] / "manifest.json").read_bytes() == (replay_args["output_dir"] / "manifest.json").read_bytes()
