"""Synthetic Wikimedia API responses; tests never access a network."""

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from llm_kbc import evidence_corpus as corpus
from llm_kbc import wikipedia_collect as wiki


UA = "ThesisEvidenceBot/0.1 (https://example.org/contact)"
STAMP = "2026-11-16T10:00:00Z"


def payload(title="Synthetic article", page_id=1, revision=10, timestamp="2026-06-30T12:00:00Z", text="Synthetic article text."):
    return {"batchcomplete": True, "query": {
        "rightsinfo": {"url": "https://creativecommons.org/licenses/by-sa/4.0/", "text": "CC BY-SA 4.0"},
        "pages": [{"pageid": page_id, "ns": 0, "title": title, "pagelanguage": "en",
                   "revisions": [{"revid": revision, "parentid": revision - 1, "timestamp": timestamp,
                                  "slots": {"main": {"contentmodel": "wikitext", "contentformat": "text/x-wiki", "content": text}}}]}],
    }}


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not access the network")
    monkeypatch.setattr(wiki, "urlopen", forbidden)
    monkeypatch.setattr(wiki.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(wiki, "_utc_now", lambda: STAMP)


def inputs(tmp_path, requests=None):
    input_path = tmp_path / "article_list.jsonl"
    values = [{"title": "Synthetic article"}] if requests is None else requests
    input_path.write_bytes(b"".join(corpus._json_bytes(value) for value in values))
    return {"input_path": input_path, "output_path": tmp_path / "corpus.jsonl", "cache_dir": tmp_path / "cache"}


def mock_api(monkeypatch, values):
    responses = iter(values)
    calls = []

    def open_mock(request, timeout):
        calls.append(request)
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        raw = item if isinstance(item, bytes) else corpus._json_bytes(item)
        stream = io.BytesIO(raw)
        stream.headers = {"Content-Type": "application/json", "Date": STAMP}
        stream.geturl = lambda: request.full_url
        return stream

    monkeypatch.setattr(wiki, "urlopen", open_mock)
    return calls


def live(kwargs):
    return wiki.collect_wikipedia(**kwargs, allow_network=True, user_agent=UA)


def test_metadata_unicode_and_rag01_integration(tmp_path, monkeypatch):
    title = "\u0391\u03b8\u03ae\u03bd\u03b1"
    text = "  {{Synthetic|value=1}}\n\u0394\u03bf\u03ba\u03b9\u03bc\u03ae caf\u00e9 \u6771\u4eac  "
    kwargs = inputs(tmp_path, [{"title": title}])
    calls = mock_api(monkeypatch, [payload(title, page_id=42, revision=101, text=text)])
    report = live(kwargs)
    assert report["status"] == "complete"
    assert len(calls) == 1 and calls[0].get_header("User-agent") == UA
    assert "maxlag=5" in calls[0].full_url and "rvslots=main" in calls[0].full_url
    document = corpus.load_documents(kwargs["output_path"])[0]
    assert document.text == text and document.title == title
    assert document.source == "encyclopedia" and document.revision_id == "101"
    assert document.retrieved_at == STAMP
    entry = report["records"][0]
    assert entry["page_id"] == 42 and entry["revision_timestamp"] == "2026-06-30T12:00:00Z"
    assert entry["cutoff_satisfied"] is None
    assert "oldid=101" in document.source_url
    assert "action=history" in entry["attribution_history_url"]
    assert entry["license"]["text"] == "CC BY-SA 4.0"
    assert document.content_sha256 == corpus._sha256(text.encode("utf-8"))
    manifest = json.loads((tmp_path / "corpus.jsonl.manifest.json").read_bytes())
    assert manifest["document_count"] == 1 and manifest["corpus_sha256"] == report["corpus_sha256"]


def test_cache_replay_byte_identical_and_no_live_access(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    calls = mock_api(monkeypatch, [payload()])
    report = live(kwargs)
    kwargs["output_path"] = tmp_path / "replay.jsonl"
    second = wiki.collect_wikipedia(**kwargs)
    assert len(calls) == 1 and second == report
    for suffix in ("", ".manifest.json", ".collection.json"):
        assert (tmp_path / ("corpus.jsonl" + suffix)).read_bytes() == (tmp_path / ("replay.jsonl" + suffix)).read_bytes()


def test_current_redirect_and_normalization_are_recorded(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path, [{"title": "original_title"}])
    value = payload("Target article", page_id=7)
    value["query"].update(normalized=[{"from": "original_title", "to": "Original title"}],
                          redirects=[{"from": "Original title", "to": "Target article"}])
    mock_api(monkeypatch, [value])
    report = live(kwargs)
    assert report["status"] == "complete"
    assert report["records"][0]["redirects"][0]["to"] == "Target article"


@pytest.mark.parametrize("timestamp,complete", [("2026-07-01T00:00:00Z", True),
                                               ("2026-06-30T00:00:00Z", True),
                                               ("2026-07-01T00:00:01Z", False)])
def test_historical_cutoff_checked_inclusively(tmp_path, monkeypatch, timestamp, complete):
    kwargs = inputs(tmp_path)
    calls = mock_api(monkeypatch, [payload(timestamp=timestamp)])
    report = wiki.collect_wikipedia(**kwargs, allow_network=True, user_agent=UA, cutoff="2026-07-01T03:00:00+03:00")
    assert "rvstart=2026-07-01T00%3A00%3A00Z" in calls[0].full_url
    assert report["records"][0]["cutoff_satisfied"] is complete
    assert (report["status"] == "complete") is complete
    assert kwargs["output_path"].exists() is complete


def test_historical_redirect_has_no_silent_fallback(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path, [{"title": "Old alias"}])
    value = payload("Target article")
    value["query"]["redirects"] = [{"from": "Old alias", "to": "Target article"}]
    calls = mock_api(monkeypatch, [value])
    report = wiki.collect_wikipedia(**kwargs, allow_network=True, user_agent=UA, cutoff="2026-07-01T00:00:00Z")
    assert report["status"] == "incomplete" and len(calls) == 1
    assert "redirect mapping" in report["records"][0]["error"]
    assert not kwargs["output_path"].exists()


@pytest.mark.parametrize("mutation,expected", [
    (lambda p: p["query"]["pages"][0].update(missing=True), "Missing"),
    (lambda p: p["query"]["pages"][0].update(pageprops={"disambiguation": ""}), "Disambiguation"),
    (lambda p: p["query"]["pages"][0].update(revisions=[]), "No unique available revision"),
    (lambda p: p["query"]["pages"][0].update(redirect=True), "Unresolved redirect"),
    (lambda p: p["query"]["pages"][0].update(ns=4), "main-namespace"),
    (lambda p: p["query"]["pages"][0].update(title="Unexplained target"), "identity substitution"),
    (lambda p: p["query"]["pages"][0]["revisions"][0]["slots"]["main"].update(content=""), "Empty"),
    (lambda p: p["query"]["pages"][0]["revisions"][0]["slots"]["main"].update(content="#REDIRECT [[Target]]"), "Possible redirect"),
    (lambda p: p["query"]["pages"][0]["revisions"][0]["slots"]["main"].update(content="__DISAMBIG__"), "disambiguation"),
    (lambda p: p["query"]["pages"][0]["revisions"][0].pop("slots"), "incomplete API"),
    (lambda p: p["query"]["pages"][0]["revisions"][0].update(revid=True), "positive integer"),
    (lambda p: p["query"].pop("rightsinfo"), "incomplete API"),
    (lambda p: p.update(warnings={"query": "truncated result"}), "warning"),
])
def test_explicit_retrieval_failures(tmp_path, monkeypatch, mutation, expected):
    kwargs = inputs(tmp_path)
    value = payload()
    mutation(value)
    mock_api(monkeypatch, [value])
    report = live(kwargs)
    assert report["status"] == "incomplete"
    assert expected in report["records"][0]["error"]
    assert not kwargs["output_path"].exists()
    assert not (tmp_path / "corpus.jsonl.manifest.json").exists()


def test_independently_supplied_page_id(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path, [{"page_id": 12}])
    calls = mock_api(monkeypatch, [payload(page_id=12)])
    assert live(kwargs)["status"] == "complete"
    assert "pageids=12" in calls[0].full_url


@pytest.mark.parametrize("error", [URLError("synthetic timeout"),
    HTTPError("https://en.wikipedia.org/w/api.php", 429, "Too many", {"Retry-After": "60"}, None),
    HTTPError("https://en.wikipedia.org/w/api.php", 503, "Unavailable", {"Retry-After": "5"}, None)])
def test_network_failures_stop_remaining_live_requests(tmp_path, monkeypatch, error):
    kwargs = inputs(tmp_path, [{"title": "First"}, {"title": "Second"}])
    calls = mock_api(monkeypatch, [error])
    report = live(kwargs)
    assert len(calls) == 1 and report["status"] == "incomplete"
    assert report["records"][1]["status"] == "not_attempted"
    assert not kwargs["output_path"].exists()
    if isinstance(error, HTTPError):
        assert report["records"][0]["network_details"]["retry_after"] in {"5", "60"}


@pytest.mark.parametrize("code", ["maxlag", "ratelimited"])
def test_api_rate_limits_are_not_reused_as_valid_cache(tmp_path, monkeypatch, code):
    kwargs = inputs(tmp_path, [{"title": "First"}, {"title": "Second"}])
    calls = mock_api(monkeypatch, [{"error": {"code": code, "info": "Synthetic busy server"}}])
    report = live(kwargs)
    assert len(calls) == 1 and report["records"][1]["status"] == "not_attempted"
    assert not list(kwargs["cache_dir"].iterdir())
    assert report["records"][0]["network_details"]["response_body"]


def test_bad_api_body_is_cached_for_offline_diagnosis(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    calls = mock_api(monkeypatch, [b"{malformed"])
    report = live(kwargs)
    assert report["status"] == "incomplete" and "malformed JSON" in report["records"][0]["error"]
    kwargs["output_path"] = tmp_path / "replay.jsonl"
    assert wiki.collect_wikipedia(**kwargs) == report
    assert len(calls) == 1


def test_corrupt_cache_fails_without_refresh(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    calls = mock_api(monkeypatch, [payload()])
    live(kwargs)
    cache = next(kwargs["cache_dir"].iterdir())
    value = json.loads(cache.read_bytes())
    value["response_sha256"] = "0" * 64
    cache.write_bytes(corpus._json_bytes(value))
    kwargs["output_path"] = tmp_path / "new.jsonl"
    report = live(kwargs)
    assert report["status"] == "incomplete" and "SHA256 mismatch" in report["records"][0]["error"]
    assert len(calls) == 1


def test_missing_cache_never_implies_network_permission(tmp_path):
    kwargs = inputs(tmp_path)
    kwargs["cache_dir"].mkdir()
    report = wiki.collect_wikipedia(**kwargs)
    assert report["status"] == "incomplete" and "Offline cache miss" in report["records"][0]["error"]


def test_cutoff_is_part_of_cache_identity(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    calls = mock_api(monkeypatch, [payload()])
    live(kwargs)
    kwargs["output_path"] = tmp_path / "historical.jsonl"
    report = wiki.collect_wikipedia(**kwargs, cutoff="2026-07-01T00:00:00Z")
    assert report["status"] == "incomplete" and len(calls) == 1
    assert "Offline cache miss" in report["records"][0]["error"]


def test_interruption_keeps_earlier_cache_but_publishes_no_corpus(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path, [{"title": "First"}, {"title": "Second"}, {"title": "Third"}])
    calls = []

    def interrupted(request, timeout):
        calls.append(request)
        if len(calls) > 1:
            raise KeyboardInterrupt
        stream = io.BytesIO(corpus._json_bytes(payload("First")))
        stream.headers = {}
        stream.geturl = lambda: request.full_url
        return stream

    monkeypatch.setattr(wiki, "urlopen", interrupted)
    report = live(kwargs)
    assert report["status"] == "interrupted"
    assert [r["status"] for r in report["records"]] == ["collected", "interrupted", "not_attempted"]
    assert len(list(kwargs["cache_dir"].iterdir())) == 1
    assert not kwargs["output_path"].exists()


def test_cli_offline_default(tmp_path, capsys):
    kwargs = inputs(tmp_path)
    kwargs["cache_dir"].mkdir()
    args = ["--input", str(kwargs["input_path"]), "--output", str(kwargs["output_path"]),
            "--cache-dir", str(kwargs["cache_dir"])]
    assert wiki.main(args) == 1
    assert "Status: incomplete" in capsys.readouterr().out


def test_page_id_cannot_hide_benchmark_article_title(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path, [{"page_id": 1}])
    mock_api(monkeypatch, [payload("LM-KBC benchmark")])
    report = live(kwargs)
    assert report["status"] == "incomplete"
    assert "benchmark/thesis article titles" in report["records"][0]["error"]


def test_selection_cap_and_no_overwrite(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path, [{"title": "First"}, {"title": "Second"}])
    calls = mock_api(monkeypatch, [payload("First")])
    report = wiki.collect_wikipedia(**kwargs, max_articles=1, allow_network=True, user_agent=UA)
    assert report["status"] == "complete" and report["selected_records"] == 1 and report["input_records"] == 2
    with pytest.raises(wiki.CollectionError, match="overwrite"):
        live(kwargs)
    assert len(calls) == 1


def test_two_requests_resolving_same_article_fail_rag01_duplicate_validation(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path, [{"title": "Synthetic article"}, {"page_id": 1}])
    mock_api(monkeypatch, [payload(), payload()])
    report = live(kwargs)
    assert report["status"] == "incomplete" and "duplicate" in report["error"]
    assert not kwargs["output_path"].exists()


@pytest.mark.parametrize("requests", [[], [{"SubjectEntity": "A", "ObjectEntities": ["answer"]}],
    [{"title": "A", "page_id": 1}], [{"page_id": True}], [{"page_id": -1}], [{"title": "A|B"}],
    [{"title": "A"}, {"title": "A"}], [{"title": "LM-KBC benchmark"}]])
def test_invalid_or_benchmark_lists_rejected_before_collection(tmp_path, requests):
    kwargs = inputs(tmp_path, requests)
    with pytest.raises(wiki.CollectionError):
        wiki.collect_wikipedia(**kwargs)
    assert not kwargs["cache_dir"].exists()


@pytest.mark.parametrize("changes", [{"max_articles": 0}, {"request_delay": 0}, {"request_delay": float("nan")},
    {"allow_network": True}, {"allow_network": True, "user_agent": "Python-urllib"},
    {"language": "https://example.org"}, {"cutoff": "2026-07-01"}, {"cutoff": "2026-07-01T00:00:00.1Z"}])
def test_invalid_configuration(tmp_path, changes):
    with pytest.raises(wiki.CollectionError):
        wiki.collect_wikipedia(**inputs(tmp_path), **changes)
