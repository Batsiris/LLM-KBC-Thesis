"""Metadata-only mocked discovery; no benchmark files, live HTTP or article downloads."""

import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from llm_kbc import bm25_retrieval as retrieval
from llm_kbc import evidence_corpus as corpus
from llm_kbc import wikipedia_collect as acquisition
from llm_kbc import wikipedia_discovery as discovery


STAMP = "2026-11-25T10:00:00Z"
UA = "SyntheticDiscovery/1.0 (https://example.org/contact)"


def page(page_id, title, **extra):
    return {"pageid": page_id, "title": title, "ns": 0, **extra}


def standard_responses(subject="Alpha"):
    return [
        {"query": {"pages": [page(1, subject)]}},
        {"query": {"search": [page(2, "Beta"), page(3, "Gamma"), page(4, "Delta")]}},
        # Metadata order deliberately differs from search ranking.
        {"query": {"pages": [page(4, "Delta"), page(3, "Gamma"), page(2, "Beta")]}},
    ]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No actual network access is permitted")
    monkeypatch.setattr(acquisition, "urlopen", forbidden)
    monkeypatch.setattr(acquisition.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(acquisition, "_utc_now", lambda: STAMP)


@pytest.fixture
def kwargs(tmp_path):
    identities = tmp_path / "identities.jsonl"
    identities.write_bytes(corpus._json_bytes({"SubjectEntity": "Alpha", "Relation": "hasArea"}))
    return {"identities_path": identities, "output_path": tmp_path / "discovery.jsonl", "cache_dir": tmp_path / "cache"}


def mock_api(monkeypatch, values):
    responses, calls = iter(values), []
    def open_mock(request, timeout):
        assert timeout == 30
        calls.append(request)
        item = next(responses)
        if isinstance(item, BaseException):
            raise item
        stream = io.BytesIO(item if isinstance(item, bytes) else corpus._json_bytes(item))
        stream.headers = {"Content-Type": "application/json", "Date": STAMP}
        stream.geturl = lambda: request.full_url
        return stream
    monkeypatch.setattr(acquisition, "urlopen", open_mock)
    return calls


def live(kwargs):
    return discovery.discover(**kwargs, allow_network=True, user_agent=UA)


def rows(path):
    return retrieval._rows(Path(path).read_bytes(), str(path))


def assert_no_completion(kwargs):
    assert not Path(str(kwargs["output_path"]) + ".manifest.json").exists()


@pytest.mark.parametrize("relation", list(retrieval.QUERIES))
def test_all_relations_direct_priority_rank_limit_and_collector_compatibility(kwargs, monkeypatch, relation):
    kwargs["identities_path"].write_bytes(corpus._json_bytes({"SubjectEntity": "Alpha", "Relation": relation}))
    calls = mock_api(monkeypatch, standard_responses())
    manifest = live(kwargs)
    records, articles, checked = discovery.load_discovery(kwargs["output_path"], cache_dir=kwargs["cache_dir"])
    record = records[0]
    assert len(calls) == 3 and checked == manifest
    assert record["query"] == retrieval.build_query("Alpha", relation)
    assert record["status"] == "candidates_unverified"
    assert [p["page_id"] for p in record["candidates"]] == [1, 2, 3]
    assert [p["search_rank"] for p in record["candidates"]] == [None, 1, 2]
    assert [p["acquisition_method"] for p in record["candidates"]] == ["direct_title", "search", "search"]
    assert all(p["discovery_timestamp"] == p["metadata_timestamp"] == STAMP for p in record["candidates"])
    assert not any(p["entity_identity_verified"] or p["historical_mapping_verified"] for p in record["candidates"])
    selection = Path(str(kwargs["output_path"]) + ".articles.jsonl").read_bytes()
    assert acquisition._requests(selection) == articles == [{"page_id": i} for i in (1, 2, 3)]
    assert manifest["historical_revision_cutoff"] == discovery.HISTORICAL_CUTOFF
    assert manifest["search_index_scope"] == discovery.SEARCH_SCOPE
    for reference in record["api_responses"]:
        params = reference["request"]["params"]
        assert "revisions" not in params.get("prop", "") and "extracts" not in params.get("prop", "")
        assert not any(k.startswith("rv") for k in params)
        assert reference["request"]["endpoint"] == discovery.ENDPOINT
        assert reference["discovery_timestamp"] == STAMP
        assert reference["status"] == "received" and len(reference["cache"]["sha256"]) == 64
    assert record["api_responses"][1]["request"]["params"]["srprop"] == ""


def test_duplicates_and_global_selection_order(kwargs, monkeypatch):
    kwargs["identities_path"].write_bytes(b"".join(corpus._json_bytes({"SubjectEntity": "Alpha", "Relation": relation})
        for relation in ("hasArea", "hasCapacity")))
    responses = standard_responses()
    responses[1]["query"]["search"] = [page(1, "Alpha"), page(2, "Beta"), page(2, "Beta"), page(3, "Gamma")]
    responses[2]["query"]["pages"] = [page(3, "Gamma"), page(1, "Alpha"), page(2, "Beta")]
    # Direct and metadata responses are reused from cache across relations.
    calls = mock_api(monkeypatch, [*responses, responses[1]])
    result = live(kwargs)
    records, articles, _ = discovery.load_discovery(kwargs["output_path"], cache_dir=kwargs["cache_dir"])
    assert len(calls) == 4 and result["record_count"] == 2 and result["article_count"] == 3
    assert [r["input_index"] for r in records] == [1, 2]
    assert articles == [{"page_id": i} for i in (1, 2, 3)]
    assert [p["search_rank"] for p in records[0]["candidates"]] == [None, 2, 4]
    assert sum(d["reason"] == "duplicate_page_id" for d in records[0]["diagnostics"]) == 2


@pytest.mark.parametrize("kind", ["redirect", "disambiguation", "missing", "unresolved_redirect"])
def test_direct_exclusions_and_unverified_search_candidates(kwargs, monkeypatch, kind):
    responses = standard_responses()
    if kind == "redirect":
        responses[0]["query"].update(pages=[page(1, "Other")], redirects=[{"from": "Alpha", "to": "Other"}])
    elif kind == "disambiguation":
        responses[0]["query"]["pages"][0]["pageprops"] = {"disambiguation": ""}
    elif kind == "missing":
        responses[0]["query"]["pages"] = [{"ns": 0, "title": "Alpha", "missing": True}]
    else:
        responses[0]["query"]["pages"][0]["redirect"] = True
    mock_api(monkeypatch, responses)
    live(kwargs)
    record = rows(kwargs["output_path"])[0]
    assert [p["page_id"] for p in record["candidates"]] == [2, 3, 4]
    assert all(p["acquisition_method"] == "search" and not p["historical_mapping_verified"] for p in record["candidates"])
    assert record["diagnostics"]


def test_normalization_is_recorded_not_semantic_identity(kwargs, monkeypatch):
    responses = [{"query": {"normalized": [{"from": "Alpha", "to": "ALPHA"}], "pages": [page(1, "ALPHA")]}},
                 {"query": {"search": []}}]
    mock_api(monkeypatch, responses)
    live(kwargs)
    record = rows(kwargs["output_path"])[0]
    assert record["candidates"][0]["title"] == "ALPHA"
    assert not record["candidates"][0]["entity_identity_verified"]
    cache_ref = record["api_responses"][0]["cache"]
    snapshot = json.loads((kwargs["cache_dir"] / (cache_ref["key"] + ".json")).read_bytes())
    assert json.loads(snapshot["response_body"])["query"]["normalized"][0]["from"] == "Alpha"


def test_search_metadata_exclusions_and_renames(kwargs, monkeypatch):
    responses = standard_responses()
    responses[0]["query"]["pages"][0]["missing"] = True
    responses[2]["query"]["pages"] = [page(2, "Beta", pageprops={"disambiguation": ""}),
        page(3, "Gamma", redirect=True), page(4, "Delta renamed")]
    mock_api(monkeypatch, responses)
    live(kwargs)
    record = rows(kwargs["output_path"])[0]
    assert [p["page_id"] for p in record["candidates"]] == [4]
    assert record["candidates"][0]["search_rank"] == 3
    assert {d["reason"] for d in record["diagnostics"]} >= {
        "disambiguation_page", "unresolved_current_redirect", "title_changed_since_search"}


def test_empty_search_and_unresolved_coverage(kwargs, monkeypatch):
    mock_api(monkeypatch, [{"query": {"pages": [{"ns": 0, "title": "Alpha", "missing": True}]}}, {"query": {"search": []}}])
    manifest = live(kwargs)
    records, articles, _ = discovery.load_discovery(kwargs["output_path"], cache_dir=kwargs["cache_dir"])
    assert manifest["record_count"] == manifest["unresolved_count"] == 1 and manifest["article_count"] == 0
    assert records[0]["status"] == "unresolved" and records[0]["candidates"] == articles == []
    assert Path(str(kwargs["output_path"]) + ".articles.jsonl").read_bytes() == b""


def test_snippets_do_not_affect_selection_or_enter_artifacts(kwargs, monkeypatch):
    responses = standard_responses()
    responses[1]["query"]["search"][0]["snippet"] = "UNTRUSTED SNIPPET MUST NOT BE EVIDENCE"
    mock_api(monkeypatch, responses)
    live(kwargs)
    assert b"UNTRUSTED SNIPPET" not in kwargs["output_path"].read_bytes()
    assert b"UNTRUSTED SNIPPET" in b"".join(p.read_bytes() for p in kwargs["cache_dir"].iterdir())


def test_gold_independence_and_byte_identical_offline_replay(kwargs, monkeypatch):
    calls = mock_api(monkeypatch, standard_responses())
    first = live(kwargs)
    before = {p: p.read_bytes() for p in kwargs["cache_dir"].iterdir()}
    kwargs["identities_path"].write_bytes(corpus._json_bytes({"SubjectEntity": "Alpha", "Relation": "hasArea",
        "ObjectEntities": {"arbitrary": ["NOT EVIDENCE", None]}}))
    replay_path = kwargs["output_path"].with_name("replay.jsonl")
    replay = discovery.discover(**{**kwargs, "output_path": replay_path})
    assert first == replay and len(calls) == 3
    for suffix in ("", ".articles.jsonl", ".manifest.json"):
        assert Path(str(kwargs["output_path"]) + suffix).read_bytes() == Path(str(replay_path) + suffix).read_bytes()
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("mutation", [
    lambda r: r[0].update(query=None),
    lambda r: r[0]["query"].update(pages=[]),
    lambda r: r[0]["query"]["pages"][0].update(title="Unexplained"),
    lambda r: r[0]["query"].update(normalized=[{}]),
    lambda r: r[1]["query"].update(search=None),
    lambda r: r[1]["query"]["search"][0].update(pageid=True),
    lambda r: r[1]["query"]["search"][0].update(ns=1),
    lambda r: r[2]["query"].update(pages=[]),
    lambda r: r[2]["query"]["pages"].append(r[2]["query"]["pages"][0]),
    lambda r: r[2]["query"].update(redirects=[{"from": "Beta", "to": "Other"}]),
    lambda r: r[2]["query"]["pages"][0].update(pageprops=[]),
])
def test_malformed_api_responses_fail_closed(kwargs, monkeypatch, mutation):
    responses = standard_responses()
    mutation(responses)
    mock_api(monkeypatch, responses)
    with pytest.raises(ValueError):
        live(kwargs)
    assert_no_completion(kwargs)
    assert not Path(str(kwargs["output_path"]) + ".articles.jsonl").exists()
    failure = json.loads(Path(str(kwargs["output_path"]) + ".failure.json").read_bytes())
    assert failure["status"] == "incomplete" and failure["records"][0]["status"] == "failed"


def test_invalid_json_cached_for_diagnosis_not_selection(kwargs, monkeypatch):
    mock_api(monkeypatch, [b"not JSON"])
    with pytest.raises(acquisition.CollectionError, match="malformed JSON"):
        live(kwargs)
    assert_no_completion(kwargs)
    assert len(list(kwargs["cache_dir"].iterdir())) == 1
    replay_path = kwargs["output_path"].with_name("invalid_replay.jsonl")
    with pytest.raises(acquisition.CollectionError):
        discovery.discover(**{**kwargs, "output_path": replay_path})


@pytest.mark.parametrize("error", [HTTPError("https://example.org", 429, "rate limit", {"Retry-After": "60"}, None),
    HTTPError("https://example.org", 503, "unavailable", {}, None), URLError("offline")])
def test_http_errors_stop_and_preserve_complete_identity_coverage(kwargs, monkeypatch, error):
    kwargs["identities_path"].write_bytes(b"".join(corpus._json_bytes({"SubjectEntity": s, "Relation": "hasArea"}) for s in ("Alpha", "Beta")))
    calls = mock_api(monkeypatch, [error])
    with pytest.raises(acquisition.NetworkFailure):
        live(kwargs)
    assert len(calls) == 1
    report = json.loads(Path(str(kwargs["output_path"]) + ".failure.json").read_bytes())
    assert [r["status"] for r in report["records"]] == ["failed", "not_attempted"]
    assert report["input_count"] == 2
    if isinstance(error, HTTPError):
        details = report["records"][0]["api_responses"][0]["network_details"]
        assert details["http_status"] == error.code
    assert_no_completion(kwargs)


@pytest.mark.parametrize("code", ["maxlag", "ratelimited", "badvalue"])
def test_api_errors_and_original_throttle_response_preserved(kwargs, monkeypatch, code):
    mock_api(monkeypatch, [{"error": {"code": code, "info": "Synthetic API error"}}])
    with pytest.raises(ValueError):
        live(kwargs)
    report = json.loads(Path(str(kwargs["output_path"]) + ".failure.json").read_bytes())
    request = report["records"][0]["api_responses"][0]
    if code in {"maxlag", "ratelimited"}:
        assert code in request["network_details"]["response_body"]
    assert_no_completion(kwargs)


@pytest.mark.parametrize("damage", ["response", "request", "missing"])
def test_cache_corruption_and_missing_cache_rejected(kwargs, monkeypatch, damage):
    live_kwargs = dict(kwargs)
    mock_api(monkeypatch, standard_responses())
    live(live_kwargs)
    cache = next(kwargs["cache_dir"].iterdir())
    if damage == "missing":
        cache.unlink()
    else:
        envelope = json.loads(cache.read_bytes())
        if damage == "response":
            envelope["response_body"] += " "
        else:
            envelope["request"]["endpoint"] = "https://example.org"
        cache.write_bytes(corpus._json_bytes(envelope))
    with pytest.raises(ValueError):
        discovery.load_discovery(kwargs["output_path"], cache_dir=kwargs["cache_dir"])
    with pytest.raises(ValueError):
        discovery.discover(**{**kwargs, "output_path": kwargs["output_path"].with_name("bad_replay.jsonl")})


@pytest.mark.parametrize("suffix", ["", ".articles.jsonl", ".manifest.json", ".failure.json"])
def test_no_overwrite_before_any_request(kwargs, monkeypatch, suffix):
    path = Path(str(kwargs["output_path"]) + suffix)
    path.write_bytes(b"original")
    with pytest.raises(discovery.DiscoveryError, match="overwrite"):
        live(kwargs)
    assert path.read_bytes() == b"original" and not kwargs["cache_dir"].exists()


@pytest.mark.parametrize("suffix", ["", ".articles.jsonl", ".manifest.json"])
def test_consumer_rejects_missing_or_corrupt_artifacts(kwargs, monkeypatch, suffix):
    mock_api(monkeypatch, standard_responses())
    live(kwargs)
    path = Path(str(kwargs["output_path"]) + suffix)
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError):
        discovery.load_discovery(kwargs["output_path"], cache_dir=kwargs["cache_dir"])
    path.write_bytes(original + b"garbage")
    with pytest.raises(ValueError):
        discovery.load_discovery(kwargs["output_path"], cache_dir=kwargs["cache_dir"])


@pytest.mark.parametrize("failure", ["manifest", "interrupt"])
def test_safe_publication_on_interruption_or_write_failure(kwargs, monkeypatch, failure):
    responses = standard_responses()
    if failure == "interrupt":
        mock_api(monkeypatch, [responses[0], KeyboardInterrupt()])
        with pytest.raises(KeyboardInterrupt):
            live(kwargs)
        assert not kwargs["output_path"].exists()
    else:
        mock_api(monkeypatch, responses)
        original = corpus._publish
        def fail_manifest(path, data):
            if str(path).endswith(".manifest.json"):
                raise OSError("simulated publication failure")
            return original(path, data)
        monkeypatch.setattr(corpus, "_publish", fail_manifest)
        with pytest.raises(OSError, match="publication failure"):
            live(kwargs)
    assert_no_completion(kwargs)


def test_collector_fetches_only_historical_revisions_after_discovery(kwargs, monkeypatch):
    mock_api(monkeypatch, standard_responses())
    live(kwargs)
    records, selected, _ = discovery.load_discovery(kwargs["output_path"], cache_dir=kwargs["cache_dir"])
    titles = {p["page_id"]: p["title"] for p in records[0]["candidates"]}
    payloads = [{"query": {"rightsinfo": {"text": "Synthetic license", "url": "https://example.org/license"},
        "pages": [page(row["page_id"], titles[row["page_id"]], pagelanguage="en", revisions=[{
            "revid": 100 + row["page_id"], "timestamp": "2026-06-30T12:00:00Z",
            "slots": {"main": {"contentmodel": "wikitext", "contentformat": "text/x-wiki",
                                "content": "Synthetic historical text " + str(row["page_id"])}}}])]}} for row in selected]
    calls = mock_api(monkeypatch, payloads)
    output = kwargs["output_path"].with_name("collected.jsonl")
    report = acquisition.collect_wikipedia(input_path=Path(str(kwargs["output_path"]) + ".articles.jsonl"),
        output_path=output, cache_dir=kwargs["cache_dir"], cutoff=discovery.HISTORICAL_CUTOFF,
        max_articles=3, allow_network=True, user_agent=UA)
    assert report["status"] == "complete" and len(calls) == 3
    assert all(r["cutoff_satisfied"] for r in report["records"])
    assert all(r["revision_timestamp"] != r["retrieved_at"] for r in report["records"])
    assert len(corpus.load_documents(output)) == 3


def test_cli_offline_default_and_controlled_errors(kwargs, monkeypatch, capsys):
    mock_api(monkeypatch, standard_responses())
    live(kwargs)
    output = kwargs["output_path"].with_name("cli_replay.jsonl")
    args = ["--identities", str(kwargs["identities_path"]), "--output", str(output), "--cache-dir", str(kwargs["cache_dir"])]
    assert discovery.main(args) == 0
    assert "Records: 1; selected articles: 3; unresolved: 0" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        discovery.main(args)
    assert error.value.code == 2 and "Traceback" not in capsys.readouterr().err


def test_offline_miss_never_enables_network(kwargs):
    kwargs["cache_dir"].mkdir()
    with pytest.raises(acquisition.CollectionError, match="Offline cache miss"):
        discovery.discover(**kwargs)
    assert_no_completion(kwargs)


def test_unicode_subject_and_invalid_direct_title_handling(kwargs, monkeypatch):
    subject = "Δοκιμή | café"
    kwargs["identities_path"].write_bytes(corpus._json_bytes({"SubjectEntity": subject, "Relation": "hasArea"}))
    calls = mock_api(monkeypatch, [{"query": {"search": [page(2, "Café Δοκιμή")]}},
                                  {"query": {"pages": [page(2, "Café Δοκιμή")]}}])
    live(kwargs)
    record = rows(kwargs["output_path"])[0]
    assert len(calls) == 2
    assert record["query"] == subject + " total area" and record["SubjectEntity"] == subject
    assert record["diagnostics"][0]["reason"] == "invalid_direct_title"
    assert record["candidates"][0]["title"] == "Café Δοκιμή"


def test_redirect_target_can_only_enter_as_unverified_search_candidate(kwargs, monkeypatch):
    responses = [{"query": {"pages": [page(1, "Other")], "redirects": [{"from": "Alpha", "to": "Other"}]}},
                 {"query": {"search": [page(1, "Other")]}}, {"query": {"pages": [page(1, "Other")]}}]
    mock_api(monkeypatch, responses)
    live(kwargs)
    record = rows(kwargs["output_path"])[0]
    assert record["candidates"][0]["acquisition_method"] == "search"
    assert record["candidates"][0]["historical_mapping_verified"] is False
    assert record["diagnostics"][0]["reason"] == "current_redirect_not_historically_verified"


def test_benchmark_article_and_missing_metadata_excluded(kwargs, monkeypatch):
    responses = standard_responses()
    responses[2]["query"]["pages"] = [page(2, "LM-KBC benchmark gold"), {"pageid": 3, "missing": True}, page(4, "Delta")]
    mock_api(monkeypatch, responses)
    live(kwargs)
    record = rows(kwargs["output_path"])[0]
    assert [c["page_id"] for c in record["candidates"]] == [1, 4]
    assert {d["reason"] for d in record["diagnostics"]} >= {"excluded_or_invalid_article_title", "missing_or_invalid_page"}


@pytest.mark.parametrize("field,value", [("status", "incomplete"), ("record_count", 99),
    ("article_count", True), ("input_identities_sha256", "0"*64), ("request_policy", []),
    ("code_sha256", None), ("historical_revision_cutoff", "2027-01-01T00:00:00Z")])
def test_consumer_rejects_invalid_completion_manifest(kwargs, monkeypatch, field, value):
    mock_api(monkeypatch, standard_responses())
    live(kwargs)
    path = Path(str(kwargs["output_path"]) + ".manifest.json")
    manifest = json.loads(path.read_bytes())
    manifest[field] = value
    path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises(discovery.DiscoveryError):
        discovery.load_discovery(kwargs["output_path"], cache_dir=kwargs["cache_dir"])


def test_consumer_rejects_rehashed_selection_tampering(kwargs, monkeypatch):
    mock_api(monkeypatch, standard_responses())
    live(kwargs)
    data = rows(kwargs["output_path"])
    data[0]["candidates"][0]["page_id"] = 999
    kwargs["output_path"].write_bytes(b"".join(corpus._json_bytes(row) for row in data))
    path = Path(str(kwargs["output_path"]) + ".manifest.json")
    manifest = json.loads(path.read_bytes())
    manifest["output_sha256"] = corpus._sha256(kwargs["output_path"].read_bytes())
    path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises(discovery.DiscoveryError, match="selection mismatch"):
        discovery.load_discovery(kwargs["output_path"], cache_dir=kwargs["cache_dir"])


@pytest.mark.parametrize("configuration", [{"request_delay": 0}, {"request_delay": float("nan")},
    {"request_delay": True}, {"allow_network": "yes"}, {"allow_network": True},
    {"allow_network": True, "user_agent": "anonymous"}])
def test_invalid_configuration_before_requests(kwargs, configuration):
    with pytest.raises(discovery.DiscoveryError):
        discovery.discover(**kwargs, **configuration)
    assert not kwargs["cache_dir"].exists() and not kwargs["output_path"].exists()


@pytest.mark.parametrize("identities", [[], [{"SubjectEntity": "A", "Relation": "unknown"}],
    [{"SubjectEntity": "A", "Relation": "hasArea"}, {"SubjectEntity": " a ", "Relation": "hasArea"}]])
def test_invalid_identities_fail_before_network_or_output(kwargs, identities):
    kwargs["identities_path"].write_bytes(b"".join(corpus._json_bytes(row) for row in identities))
    with pytest.raises(ValueError):
        live(kwargs)
    assert not kwargs["cache_dir"].exists() and not kwargs["output_path"].exists()


def test_input_mutation_aborts_before_final_publication(kwargs, monkeypatch):
    mock_api(monkeypatch, standard_responses())
    original = discovery._discover_record
    def changed(*args, **options):
        original(*args, **options)
        kwargs["identities_path"].write_bytes(corpus._json_bytes({"SubjectEntity": "Changed", "Relation": "hasArea"}))
    monkeypatch.setattr(discovery, "_discover_record", changed)
    with pytest.raises(discovery.DiscoveryError, match="identities changed"):
        live(kwargs)
    assert_no_completion(kwargs)
    assert not kwargs["output_path"].exists()


@pytest.mark.parametrize("flag", ["missing", "invalid", "redirect"])
@pytest.mark.parametrize("value", [False, None, 0, 1, "", "true", [], {}])
@pytest.mark.parametrize("stage", ["direct", "search_metadata"])
def test_malformed_page_flags_fail_closed(kwargs, monkeypatch, flag, value, stage):
    responses = standard_responses()
    target = responses[0 if stage == "direct" else 2]["query"]["pages"][0]
    target[flag] = value
    mock_api(monkeypatch, responses)
    with pytest.raises(discovery.DiscoveryError, match="page flags"):
        live(kwargs)
    assert_no_completion(kwargs)
    assert not kwargs["output_path"].exists()
    failure = json.loads(Path(str(kwargs["output_path"]) + ".failure.json").read_bytes())
    assert failure["records"][0]["status"] == "failed"


@pytest.mark.parametrize("bad_page", [
    {"missing": True}, {"ns": 0, "missing": True}, {"title": "Alpha", "missing": True},
    {"title": "Alpha", "ns": False, "missing": True},
    {"title": "Alpha", "ns": 0, "pageid": False, "missing": True},
    {"invalid": True}, {"title": "Alpha", "invalid": True},
    {"title": "Alpha", "invalid": True, "invalidreason": ""},
    {"title": "Alpha", "invalid": True, "invalidreason": []},
    page(1, "Alpha", missing=True, redirect=True),
    page(1, "Alpha", missing=True, invalid=True),
    {"title": "Alpha", "ns": 0, "redirect": True},
    {"pageid": 1, "ns": 0}, {"pageid": 1, "title": "Alpha"},
    page(1, "Alpha", pageprops=None), page(1, "Alpha", pageprops={"disambiguation": False}),
])
def test_incomplete_metadata_has_controlled_cli_failure(kwargs, monkeypatch, capsys, bad_page):
    mock_api(monkeypatch, [{"query": {"pages": [bad_page]}}])
    with pytest.raises(SystemExit) as error:
        discovery.main(["--identities", str(kwargs["identities_path"]), "--output", str(kwargs["output_path"]),
            "--cache-dir", str(kwargs["cache_dir"]), "--allow-network", "--user-agent", UA])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "Wikipedia discovery failed:" in stderr and "Traceback" not in stderr
    assert_no_completion(kwargs)
    assert not kwargs["output_path"].exists()


@pytest.mark.parametrize("direct", [
    {"ns": 0, "title": "Alpha", "missing": True},
    {"title": "Alpha", "invalid": True, "invalidreason": "Synthetic invalid title"},
    page(1, "Alpha", redirect=True), page(1, "Alpha", pageprops={"disambiguation": ""}),
])
def test_legitimate_excluded_page_schemas_preserve_search_selection(kwargs, monkeypatch, direct):
    responses = standard_responses()
    responses[0]["query"]["pages"] = [direct]
    mock_api(monkeypatch, responses)
    live(kwargs)
    record = rows(kwargs["output_path"])[0]
    assert [c["page_id"] for c in record["candidates"]] == [2, 3, 4]


def test_transport_headers_timeout_delay_and_no_offline_transport(kwargs, monkeypatch):
    delays = []
    monkeypatch.setattr(acquisition.time, "sleep", delays.append)
    calls = mock_api(monkeypatch, standard_responses())
    first = discovery.discover(**kwargs, allow_network=True, user_agent=UA, request_delay=1.75)
    assert delays == [1.75] * 3 and len(calls) == 3
    assert all(request.get_header("User-agent") == UA for request in calls)
    assert all(request.get_header("Accept") == "application/json" for request in calls)
    cached = {p: p.read_bytes() for p in kwargs["cache_dir"].iterdir()}
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **k: pytest.fail("Offline replay attempted network"))
    replay_path = kwargs["output_path"].with_name("transport_replay.jsonl")
    replay = discovery.discover(**{**kwargs, "output_path": replay_path}, request_delay=1.75)
    discovery.load_discovery(replay_path, cache_dir=kwargs["cache_dir"])
    assert replay == first and delays == [1.75] * 3
    for suffix in ("", ".articles.jsonl", ".manifest.json"):
        assert Path(str(kwargs["output_path"]) + suffix).read_bytes() == Path(str(replay_path) + suffix).read_bytes()
    assert all(p.read_bytes() == content for p, content in cached.items())


@pytest.mark.parametrize("kind", ["http429", "maxlag", "ratelimited"])
def test_recovery_reuses_successful_prefix_without_overwriting_failure(kwargs, monkeypatch, kind):
    responses = standard_responses()
    error = (HTTPError("https://example.org", 429, "rate limit", {"Retry-After": "60"}, None)
             if kind == "http429" else {"error": {"code": kind, "info": "Wait before a new run"}})
    delays = []
    monkeypatch.setattr(acquisition.time, "sleep", delays.append)
    calls = mock_api(monkeypatch, [responses[0], error])
    with pytest.raises(acquisition.NetworkFailure):
        live(kwargs)
    assert len(calls) == 2 and len(delays) == 2  # No automatic retry.
    prefix = {p: p.read_bytes() for p in kwargs["cache_dir"].iterdir()}
    assert len(prefix) == 1
    failure_path = Path(str(kwargs["output_path"]) + ".failure.json")
    failure_bytes = failure_path.read_bytes()
    if kind == "http429":
        report = json.loads(failure_bytes)
        assert report["records"][0]["api_responses"][-1]["network_details"]["retry_after"] == "60"
    # Simulate a later, operator-authorized run after the server's waiting period.
    calls = mock_api(monkeypatch, responses[1:])
    recovered = kwargs["output_path"].with_name("recovered.jsonl")
    result = live({**kwargs, "output_path": recovered})
    assert result["article_count"] == 3 and len(calls) == 2 and len(delays) == 4
    assert all(p.read_bytes() == content for p, content in prefix.items())
    assert failure_path.read_bytes() == failure_bytes
    discovery.load_discovery(recovered, cache_dir=kwargs["cache_dir"])
    assert len(calls) == 2 and len(delays) == 4


def test_malformed_cache_recovery_uses_a_separate_snapshot(kwargs, monkeypatch):
    mock_api(monkeypatch, [b"not JSON"])
    with pytest.raises(acquisition.CollectionError):
        live(kwargs)
    failed_cache = {p: p.read_bytes() for p in kwargs["cache_dir"].iterdir()}
    failure_path = Path(str(kwargs["output_path"]) + ".failure.json")
    failure_bytes = failure_path.read_bytes()
    calls = mock_api(monkeypatch, standard_responses())
    fresh = {**kwargs, "output_path": kwargs["output_path"].with_name("fresh.jsonl"),
             "cache_dir": kwargs["cache_dir"].with_name("fresh_cache")}
    assert live(fresh)["article_count"] == 3 and len(calls) == 3
    assert all(p.read_bytes() == content for p, content in failed_cache.items())
    assert failure_path.read_bytes() == failure_bytes
    discovery.load_discovery(fresh["output_path"], cache_dir=fresh["cache_dir"])


def pilot_fixture(tmp_path):
    # Interleave relations, include a fourth row per relation, and vary unused gold.
    records = [{"SubjectEntity": f"Subject {i} {relation}", "Relation": relation,
                "ObjectEntities": {"ignored": i}}
               for i in range(4) for relation in reversed(retrieval.QUERIES)]
    train = tmp_path / "synthetic_training.jsonl"
    data = b"".join(corpus._json_bytes(row) for row in records)
    train.write_bytes(data)
    return train, records, corpus._sha256(data)


def test_pilot_selection_order_gold_independence_and_determinism(tmp_path):
    train, records, fingerprint = pilot_fixture(tmp_path)
    first, second = tmp_path / "pilot.jsonl", tmp_path / "replay.jsonl"
    manifest = discovery.prepare_pilot(train, first, expected_train_sha256=fingerprint)
    repeated = discovery.prepare_pilot(train, second, expected_train_sha256=fingerprint)
    expected = [{k: row[k] for k in ("SubjectEntity", "Relation")} for row in records[:18]]
    assert rows(first) == expected and first.read_bytes() == second.read_bytes()
    assert manifest == repeated == {"train_sha256": fingerprint, "record_count": 18,
                                   "input_identities_sha256": corpus._sha256(first.read_bytes())}
    for row in records:
        row["ObjectEntities"] = ["CHANGED, NOT USED"]
    train.write_bytes(b"".join(corpus._json_bytes(row) for row in records))
    changed = tmp_path / "changed.jsonl"
    discovery.prepare_pilot(train, changed, expected_train_sha256=corpus._sha256(train.read_bytes()))
    assert changed.read_bytes() == first.read_bytes()
    with pytest.raises(discovery.DiscoveryError, match="overwrite"):
        discovery.prepare_pilot(train, first, expected_train_sha256=fingerprint)
    assert rows(first) == expected


@pytest.mark.parametrize("fingerprint", ["0" * 64, "", "not-a-hash", "A" * 64, None])
def test_pilot_fingerprint_mismatch_or_malformed_fails_without_output(tmp_path, fingerprint):
    train, _, _ = pilot_fixture(tmp_path)
    output = tmp_path / "pilot.jsonl"
    with pytest.raises(discovery.DiscoveryError, match="SHA256"):
        discovery.prepare_pilot(train, output, expected_train_sha256=fingerprint)
    assert not output.exists()


def test_pilot_cli_pins_production_hash_and_mismatch_exits_cleanly(tmp_path, capsys):
    train, _, _ = pilot_fixture(tmp_path)
    output = tmp_path / "pilot.jsonl"
    assert discovery.PILOT_TRAIN_SHA256 == "ad37cd30d1ff4b9f1ef2579b25e64093b202c40da11e8c412e13386f1e5d332e"
    with pytest.raises(SystemExit) as error:
        discovery.main(["--prepare-pilot", str(train), "--output", str(output)])
    assert error.value.code == 2 and not output.exists()
    stderr = capsys.readouterr().err
    assert "Training snapshot SHA256 mismatch" in stderr and "Traceback" not in stderr


@pytest.mark.parametrize("missing_relation", [False, True])
def test_pilot_requires_all_relations_and_three_per_relation(tmp_path, missing_relation):
    train, records, _ = pilot_fixture(tmp_path)
    selected = ([row for row in records if row["Relation"] != next(iter(retrieval.QUERIES))]
                if missing_relation else records[:17])
    train.write_bytes(b"".join(corpus._json_bytes(row) for row in selected))
    output = tmp_path / "pilot.jsonl"
    with pytest.raises(discovery.DiscoveryError, match="three training identities"):
        discovery.prepare_pilot(train, output, expected_train_sha256=corpus._sha256(train.read_bytes()))
    assert not output.exists()


def test_pilot_rejects_source_mutation_before_publication(tmp_path, monkeypatch):
    train, _, fingerprint = pilot_fixture(tmp_path)
    output = tmp_path / "pilot.jsonl"
    original = retrieval.load_identities
    def changed(path):
        identities = original(path)
        train.write_bytes(train.read_bytes() + b"\n")
        return identities
    monkeypatch.setattr(retrieval, "load_identities", changed)
    with pytest.raises(discovery.DiscoveryError, match="changed during pilot"):
        discovery.prepare_pilot(train, output, expected_train_sha256=fingerprint)
    assert not output.exists()
