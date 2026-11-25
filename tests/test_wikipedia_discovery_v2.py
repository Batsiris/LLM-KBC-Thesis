"""Synthetic v2 discovery and archived-v1 compatibility; never live API access."""

import io
import json
import subprocess
import types
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from llm_kbc import evidence_corpus as corpus, rag_query
from llm_kbc import wikipedia_collect as acquisition
from llm_kbc import wikipedia_discovery as v1
from llm_kbc import wikipedia_discovery_v2 as v2


STAMP = "2026-11-25T10:00:00Z"
UA = "SyntheticDiscovery/1.0 (https://example.org/contact)"


def page(index, title, **extra):
    return {"pageid": index, "title": title, "ns": 0, **extra}


def responses(subject="Alpha", count=10):
    hits = [page(i, f"Article {i}") for i in range(2, count + 2)]
    return [{"query": {"pages": [page(1, subject)]}}, {"query": {"search": hits}},
            {"query": {"pages": list(reversed(hits))}}]


def write(path, rows):
    path.write_bytes(b"".join(corpus._json_bytes(row) for row in rows))
    return path


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **k: pytest.fail("Actual network access forbidden"))
    monkeypatch.setattr(acquisition.time, "sleep", lambda delay: None)
    monkeypatch.setattr(acquisition, "_utc_now", lambda: STAMP)


@pytest.fixture
def paths(tmp_path):
    return {"identities_path": write(tmp_path / "inputs.jsonl", [{"subject": "Alpha", "relation_id": "R"}]),
            "output_path": tmp_path / "discovery.jsonl", "cache_dir": tmp_path / "cache"}


def mock_api(monkeypatch, payloads):
    iterator, calls = iter(payloads), []
    def fetch(request, timeout):
        calls.append(parse_qs(urlsplit(request.full_url).query, keep_blank_values=True))
        assert timeout == 30 and request.get_header("User-agent") == UA
        payload = next(iterator)
        if isinstance(payload, BaseException):
            raise payload
        stream = io.BytesIO(payload if isinstance(payload, bytes) else corpus._json_bytes(payload))
        stream.headers = {"Content-Type": "application/json"}
        stream.geturl = lambda: request.full_url
        return stream
    monkeypatch.setattr(acquisition, "urlopen", fetch)
    return calls


def run(paths, **kwargs):
    return v2.discover(**paths, allow_network=True, user_agent=UA, **kwargs)


def read(paths):
    return v1.load_discovery(paths["output_path"], cache_dir=paths["cache_dir"])


def no_completion(paths):
    assert not Path(str(paths["output_path"]) + ".manifest.json").exists()


@pytest.mark.parametrize("limit", [3, 5, 10])
def test_v1_cache_expansion_prefix_and_no_http(paths, monkeypatch, limit):
    write(paths["identities_path"], [{"SubjectEntity": "Alpha", "Relation": "hasArea"}])
    calls = mock_api(monkeypatch, responses())
    original = paths["output_path"].with_name("v1.jsonl")
    v1.discover(**{**paths, "output_path": original}, allow_network=True, user_agent=UA)
    old_records, _, old_manifest = v1.load_discovery(original, cache_dir=paths["cache_dir"])
    archived = {p: p.read_bytes() for p in [original, Path(str(original) + ".articles.jsonl"),
                Path(str(original) + ".manifest.json"), *paths["cache_dir"].iterdir()]}
    config = v2.DiscoveryConfig(candidate_limit=limit)
    v2.discover(**paths, input_format="lm-kbc-2026", configuration=config)
    records, articles, manifest = read(paths)
    assert len(calls) == 3 and len(records[0]["candidates"]) == limit
    assert records[0]["candidates"][:3] == old_records[0]["candidates"]
    assert articles == [{"page_id": i} for i in range(1, limit + 1)]
    assert manifest["configuration"]["candidate_limit"] == limit
    assert old_manifest["candidate_limit"] == 3
    assert all(p.read_bytes() == data for p, data in archived.items())


@pytest.mark.parametrize("relation", list(v1.retrieval.QUERIES))
@pytest.mark.parametrize("subject", ["Νέο 東京 e\u0301", "  New   subject  "])
def test_legacy_projection_and_exact_requests(paths, monkeypatch, relation, subject):
    write(paths["identities_path"], [{"SubjectEntity": subject, "Relation": relation, "ObjectEntities": []}])
    payloads = responses(subject)
    if subject != subject.strip():
        payloads = payloads[1:]
    calls = mock_api(monkeypatch, payloads)
    run(paths, input_format="lm-kbc-2026")
    records, _, _ = read(paths)
    record = records[0]
    assert record["relation_namespace"] == "lm-kbc-2026"
    assert record["search_expression"] == v1.retrieval.build_query(subject, relation)
    search = next(call for call in calls if "srsearch" in call)
    assert search == {"action": ["query"], "format": ["json"], "formatversion": ["2"], "maxlag": ["5"],
                      "list": ["search"], "srsearch": [record["search_expression"]], "srnamespace": ["0"],
                      "srlimit": ["10"], "srprop": [""], "srinfo": [""]}
    assert record["query_provenance"]["method"] == "lm_kbc_2026_template"


@pytest.mark.parametrize("relation,description,namespace,method,expression", [
    ("livesIn", "city of residence", "external", "relation_description", "Alpha city of residence"),
    ("livesIn", None, None, "literal_relation_fallback", "Alpha livesIn"),
    ("P999999", None, "external", "literal_relation_fallback", "Alpha P999999"),
    ("hasArea", "district", "external", "relation_description", "Alpha district"),
    ("hasArea", None, None, "literal_relation_fallback", "Alpha hasArea"),
])
def test_generic_relations(paths, monkeypatch, relation, description, namespace, method, expression):
    write(paths["identities_path"], [{"subject": "Alpha", "relation_id": relation,
          "relation_description": description, "relation_namespace": namespace}])
    mock_api(monkeypatch, responses())
    run(paths)
    records, _, manifest = read(paths)
    assert records[0]["search_expression"] == expression
    assert records[0]["query_provenance"]["method"] == method
    assert manifest["ordered_inputs"][0]["relation_namespace"] == namespace


class Poison:
    def __repr__(self):
        raise AssertionError("Gold value inspected")
    def __str__(self):
        raise AssertionError("Gold value inspected")
    def __deepcopy__(self, memo):
        raise AssertionError("Gold value inspected")


@pytest.mark.parametrize("format", ["generic", "lm-kbc-2026"])
def test_poison_labels_are_not_inspected(format):
    row = ({"subject": "Alpha", "relation_id": "new"} if format == "generic"
           else {"SubjectEntity": "Alpha", "Relation": "hasArea"})
    assert v2.project_input({**row, "ObjectEntities": Poison(), "gold": Poison()}, format) == v2.project_input(row, format)


@pytest.mark.parametrize("format", ["generic", "lm-kbc-2026"])
def test_gold_independent_byte_identical_replay(paths, monkeypatch, format):
    row = ({"subject": "Alpha", "relation_id": "new"} if format == "generic"
           else {"SubjectEntity": "Alpha", "Relation": "hasArea"})
    write(paths["identities_path"], [{**row, "ObjectEntities": []}])
    calls = mock_api(monkeypatch, responses())
    first = run(paths, input_format=format)
    cache = {p: p.read_bytes() for p in paths["cache_dir"].iterdir()}
    write(paths["identities_path"], [{**row, "ObjectEntities": {"changed": [None, 123]}}])
    replay = paths["output_path"].with_name("replay.jsonl")
    assert v2.discover(**{**paths, "output_path": replay}, input_format=format) == first
    for suffix in ("", ".articles.jsonl", ".manifest.json"):
        assert Path(str(paths["output_path"]) + suffix).read_bytes() == Path(str(replay) + suffix).read_bytes()
    assert len(calls) == 3 and all(p.read_bytes() == data for p, data in cache.items())


@pytest.mark.parametrize("values", [
    {"candidate_limit": True}, {"candidate_limit": 0}, {"candidate_limit": 11}, {"candidate_limit": 5.0},
    {"candidate_limit": "5"}, {"search_limit": False}, {"search_limit": 0}, {"search_limit": 11},
    {"search_limit": 4}, {"historical_cutoff": None}, {"historical_cutoff": "2026-02-30T00:00:00Z"},
    {"historical_cutoff": "2026-07-01T00:00:00+00:00"}, {"historical_cutoff": "2026-07-01T00:00:00.001Z"},
])
def test_invalid_configuration(values):
    with pytest.raises(ValueError):
        v2.DiscoveryConfig(**values)


@pytest.mark.parametrize("field", ["subject", "relation_id", "relation_description", "relation_namespace"])
@pytest.mark.parametrize("char", ["\t", "\n", "\x01", "\x1b", "\x7f", "\u200b", "\ufeff", "\u202e", "\u2028", "\u2029"])
def test_backend_rejects_controls_before_requests(paths, field, char):
    write(paths["identities_path"], [{"subject": "Alpha", "relation_id": "R", field: "a" + char + "b"}])
    with pytest.raises(v2.DiscoveryError, match="unsupported Wikipedia input"):
        run(paths)
    assert not paths["cache_dir"].exists() and not paths["output_path"].exists()


@pytest.mark.parametrize("subject", ["فارسی\u200cنام", "👩\u200d🔬", "東京 café e\u0301", "A\u00a0B"])
def test_meaningful_unicode_is_preserved(paths, monkeypatch, subject):
    write(paths["identities_path"], [{"subject": subject, "relation_id": "R"}])
    mock_api(monkeypatch, responses(subject))
    run(paths)
    assert read(paths)[0][0]["search_expression"] == subject + " R"


@pytest.mark.parametrize("subject", ["\u200d", "\u0301"])
def test_invisible_or_combining_only_inputs_rejected(paths, subject):
    write(paths["identities_path"], [{"subject": subject, "relation_id": "R"}])
    with pytest.raises(v2.DiscoveryError, match="visible text"):
        run(paths)


def test_verbatim_search_operators_and_title_ineligible_diagnostic(paths, monkeypatch):
    subject = 'Alpha | Beta #section OR "quoted"'
    write(paths["identities_path"], [{"subject": subject, "relation_id": "R"}])
    calls = mock_api(monkeypatch, responses()[1:])
    run(paths)
    record = read(paths)[0][0]
    assert len(calls) == 2 and calls[0]["srsearch"] == [subject + " R"]
    assert record["diagnostics"][0]["reason"] == "invalid_direct_title"


@pytest.mark.parametrize("second,error", [
    ({"subject": "Alpha", "relation_id": "R"}, "exact duplicate"),
    ({"subject": " ＡLPHA  ", "relation_id": "R"}, "normalized duplicate"),
    ({"subject": "Alpha", "relation_id": "R", "relation_description": "other"}, "conflicting relation descriptions"),
    ({"subject": " alpha ", "relation_id": "R", "relation_description": "other"}, "conflicting relation descriptions"),
])
def test_duplicate_and_conflicting_definitions_fail_before_cache(paths, second, error):
    write(paths["identities_path"], [{"subject": "Alpha", "relation_id": "R"}, second])
    with pytest.raises(v2.DiscoveryError, match=error):
        run(paths)
    assert not paths["cache_dir"].exists()


def test_namespaces_and_distinct_tasks_share_raw_cache_without_identity_conflation(paths, monkeypatch):
    write(paths["identities_path"], [{"subject": "Alpha", "relation_id": relation,
          "relation_namespace": ns, "relation_description": "same query words"}
          for relation, ns in [("R", "first"), ("R", "second"), ("other", "second")]])
    calls = mock_api(monkeypatch, responses())
    run(paths)
    records, articles, manifest = read(paths)
    assert len(calls) == 3 and len(articles) == 5 and manifest["unique_request_count"] == 3
    assert len({r["query_provenance"]["identity_sha256"] for r in records}) == 3
    assert len({r["query_provenance"]["query_sha256"] for r in records}) == 3
    assert all([c["page_id"] for c in r["candidates"]] == [1, 2, 3, 4, 5] for r in records)


def test_shortfalls_redirects_disambiguation_and_unresolved(paths, monkeypatch):
    payloads = responses(count=3)
    payloads[0]["query"].update(redirects=[{"from": "Alpha", "to": "Other"}], pages=[page(1, "Other")])
    payloads[2]["query"]["pages"] = [page(2, "Article 2", pageprops={"disambiguation": ""}),
                                       page(3, "Article 3", redirect=True), {"pageid": 4, "missing": True}]
    mock_api(monkeypatch, payloads)
    manifest = run(paths)
    record = read(paths)[0][0]
    assert record["status"] == "unresolved" and record["candidates"] == []
    assert manifest["shortfall_count"] == manifest["unresolved_count"] == 1
    assert record["diagnostics"][-1] == {"stage": "selection", "reason": "candidate_shortfall", "requested": 5, "selected": 0}


def test_candidate_and_search_duplicates_preserve_rank(paths, monkeypatch):
    payloads = responses(count=3)
    payloads[1]["query"]["search"] = [page(1, "Alpha"), page(2, "Article 2"), page(2, "Article 2"), page(3, "Article 3")]
    payloads[2]["query"]["pages"] = [page(3, "Article 3"), page(2, "Article 2"), page(1, "Alpha")]
    mock_api(monkeypatch, payloads)
    run(paths)
    record = read(paths)[0][0]
    assert [c["page_id"] for c in record["candidates"]] == [1, 2, 3]
    assert [c["search_rank"] for c in record["candidates"]] == [None, 2, 4]


def test_changed_search_parameters_are_controlled_offline_misses(paths, monkeypatch):
    calls = mock_api(monkeypatch, responses())
    run(paths)
    changed = {**paths, "output_path": paths["output_path"].with_name("new_search.jsonl")}
    with pytest.raises(ValueError, match="Offline cache miss"):
        v2.discover(**changed, configuration=v2.DiscoveryConfig(search_limit=9))
    assert len(calls) == 3
    no_completion(changed)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "unknown"), ("discovery_policy_version", "unknown"), ("query_policy_version", "unknown"),
    ("backend_validation_policy", "unknown"), ("search_expression_policy", "unknown"), ("unicode_version", "unknown"),
    ("record_count", True), ("input_format", []), ("configuration", []), ("implementation_sha256", {}),
    ("request_policy", []), ("output_sha256", "0" * 64), ("ordered_inputs", []),
])
def test_manifest_tampering_rejected(paths, monkeypatch, field, value):
    mock_api(monkeypatch, responses())
    run(paths)
    manifest = Path(str(paths["output_path"]) + ".manifest.json")
    payload = json.loads(manifest.read_bytes())
    payload[field] = value
    manifest.write_bytes(corpus._json_bytes(payload))
    with pytest.raises(ValueError):
        read(paths)


@pytest.mark.parametrize("field,value", [("search_expression", "changed"), ("relation_description", "changed"),
    ("query_provenance", {}), ("candidates", []), ("api_responses", []), ("diagnostics", [])])
def test_rehashed_record_tampering_rejected(paths, monkeypatch, field, value):
    mock_api(monkeypatch, responses(count=2))
    run(paths)
    record = json.loads(paths["output_path"].read_bytes())
    record[field] = value
    paths["output_path"].write_bytes(corpus._json_bytes(record))
    manifest = Path(str(paths["output_path"]) + ".manifest.json")
    payload = json.loads(manifest.read_bytes())
    payload["output_sha256"] = corpus._sha256(paths["output_path"].read_bytes())
    manifest.write_bytes(corpus._json_bytes(payload))
    with pytest.raises(ValueError):
        read(paths)


@pytest.mark.parametrize("damage", ["missing", "response", "request"])
def test_cache_damage_rejected(paths, monkeypatch, damage):
    mock_api(monkeypatch, responses())
    run(paths)
    cache = next(paths["cache_dir"].iterdir())
    if damage == "missing":
        cache.unlink()
    else:
        payload = json.loads(cache.read_bytes())
        if damage == "response":
            payload["response_body"] += " "
        else:
            payload["request"]["endpoint"] = "https://example.org"
        cache.write_bytes(corpus._json_bytes(payload))
    with pytest.raises(ValueError):
        read(paths)


@pytest.mark.parametrize("suffix", ["", ".articles.jsonl", ".manifest.json", ".failure.json"])
def test_no_clobber_before_requests(paths, suffix):
    target = Path(str(paths["output_path"]) + suffix)
    target.write_bytes(b"original")
    with pytest.raises(v2.DiscoveryError, match="overwrite"):
        run(paths)
    assert target.read_bytes() == b"original" and not paths["cache_dir"].exists()


@pytest.mark.parametrize("failure", ["interrupt", "manifest"])
def test_failure_publication_never_completes(paths, monkeypatch, failure):
    payloads = responses()
    mock_api(monkeypatch, [payloads[0], KeyboardInterrupt()] if failure == "interrupt" else payloads)
    original = corpus._publish
    def publish(path, content):
        if str(path).endswith(".manifest.json"):
            raise OSError("Synthetic publication failure")
        original(path, content)
    if failure == "manifest":
        monkeypatch.setattr(corpus, "_publish", publish)
    with pytest.raises((KeyboardInterrupt, OSError)):
        run(paths)
    no_completion(paths)
    with pytest.raises(ValueError):
        read(paths)


@pytest.mark.parametrize("failure", ["rate_limit", "malformed"])
def test_failure_recovery_preserves_old_artifacts(paths, monkeypatch, failure):
    payloads = responses()
    error = HTTPError("https://example.org", 429, "wait", {"Retry-After": "60"}, None) if failure == "rate_limit" else b"not JSON"
    mock_api(monkeypatch, [payloads[0], error])
    with pytest.raises(ValueError):
        run(paths)
    preserved = {p: p.read_bytes() for p in paths["cache_dir"].iterdir()}
    report_path = Path(str(paths["output_path"]) + ".failure.json")
    report = report_path.read_bytes()
    recovered = {**paths, "output_path": paths["output_path"].with_name("recovered.jsonl")}
    if failure == "malformed":
        recovered["cache_dir"] = paths["cache_dir"].with_name("fresh_cache")
        calls = mock_api(monkeypatch, payloads)
    else:
        # Simulate a later explicit invocation after respecting Retry-After.
        calls = mock_api(monkeypatch, payloads[1:])
    assert run(recovered)["article_count"] == 5
    assert len(calls) == (3 if failure == "malformed" else 2)
    assert report_path.read_bytes() == report and all(p.read_bytes() == data for p, data in preserved.items())


def test_original_commit_v1_artifact_is_still_verifiable(paths, monkeypatch, tmp_path):
    source = subprocess.run(["git", "show", "bd5202fb8f95345aeff4a18b7ec672767e3c0c2c:src/llm_kbc/wikipedia_discovery.py"],
                            capture_output=True, check=True, timeout=15).stdout
    source_path = tmp_path / "archived_v1.py"
    source_path.write_bytes(source)
    archived = types.ModuleType("llm_kbc._archived_discovery_v1")
    archived.__package__, archived.__file__ = "llm_kbc", str(source_path)
    exec(compile(source, str(source_path), "exec"), archived.__dict__)
    write(paths["identities_path"], [{"SubjectEntity": "Alpha", "Relation": "hasArea"}])
    mock_api(monkeypatch, responses())
    manifest = archived.discover(**paths, allow_network=True, user_agent=UA)
    assert manifest["code_sha256"] == corpus._sha256(source)
    records, _, validated = read(paths)
    assert validated == manifest and len(records[0]["candidates"]) == 3


def test_cli_v2_and_legacy_flag_rejection(paths, monkeypatch, capsys):
    mock_api(monkeypatch, responses())
    args = ["--identities", str(paths["identities_path"]), "--output", str(paths["output_path"]),
            "--cache-dir", str(paths["cache_dir"]), "--allow-network", "--user-agent", UA]
    with pytest.raises(SystemExit) as error:
        v1.main([*args, "--candidate-limit", "5"])
    assert error.value.code == 2
    assert v1.main([*args, "--policy-version", "v2"]) == 0
    assert "selected articles: 5" in capsys.readouterr().out
    invalid = paths["output_path"].with_name("bad.jsonl")
    with pytest.raises(SystemExit) as error:
        v1.main(["--identities", str(paths["identities_path"]), "--output", str(invalid),
                 "--cache-dir", str(paths["cache_dir"]), "--policy-version", "v2", "--candidate-limit", "0"])
    assert error.value.code == 2 and "Traceback" not in capsys.readouterr().err


def test_shorter_search_limit_and_custom_cutoff(paths, monkeypatch):
    calls = mock_api(monkeypatch, responses(count=3))
    config = v2.DiscoveryConfig(candidate_limit=3, search_limit=3, historical_cutoff="2025-01-01T00:00:00Z")
    run(paths, configuration=config)
    records, _, manifest = read(paths)
    assert len(records[0]["candidates"]) == 3
    assert calls[1]["srlimit"] == ["3"] and all("rvstart" not in call for call in calls)
    assert manifest["configuration"]["historical_cutoff"] == "2025-01-01T00:00:00Z"


@pytest.mark.parametrize("content", [b"", b"[]\n", b"{broken}\n", b'{"subject":"Alpha"}\n',
    b'{"subject":"Alpha","subject":"Beta","relation_id":"R"}\n'])
def test_invalid_input_never_publishes(paths, content):
    paths["identities_path"].write_bytes(content)
    with pytest.raises(ValueError):
        run(paths)
    no_completion(paths)
    assert not paths["cache_dir"].exists()


def test_lm_adapter_remains_strict(paths):
    write(paths["identities_path"], [{"SubjectEntity": "Alpha", "Relation": "unknown"}])
    with pytest.raises(ValueError):
        run(paths, input_format="lm-kbc-2026")
    assert not paths["cache_dir"].exists()


@pytest.mark.parametrize("damage", ["orphan", "articles", "configuration", "incomplete_coverage"])
def test_completion_and_inventory_integrity(paths, monkeypatch, damage):
    mock_api(monkeypatch, responses())
    run(paths)
    manifest_path = Path(str(paths["output_path"]) + ".manifest.json")
    if damage == "orphan":
        manifest_path.unlink()
    elif damage == "articles":
        Path(str(paths["output_path"]) + ".articles.jsonl").write_bytes(b'{"page_id":999}\n')
    else:
        manifest = json.loads(manifest_path.read_bytes())
        if damage == "configuration":
            manifest["configuration"]["candidate_limit"] = 3
        else:
            manifest["input_count"] = 2
        manifest_path.write_bytes(corpus._json_bytes(manifest))
    with pytest.raises(ValueError):
        read(paths)


def test_backend_cli_failure_is_controlled(paths, capsys):
    write(paths["identities_path"], [{"subject": "bad\u200btitle", "relation_id": "R"}])
    with pytest.raises(SystemExit) as error:
        v1.main(["--policy-version", "v2", "--identities", str(paths["identities_path"]),
                 "--output", str(paths["output_path"]), "--cache-dir", str(paths["cache_dir"])])
    message = capsys.readouterr().err
    assert error.value.code == 2 and "unsupported Wikipedia input" in message and "Traceback" not in message
    no_completion(paths)
