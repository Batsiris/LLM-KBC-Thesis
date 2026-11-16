"""Source-only synthetic fixtures; no gold labels, retrieval or live requests."""

import json
import pytest

from llm_kbc import evidence_corpus as corpus
from llm_kbc import wikipedia_collect as acquisition
from llm_kbc import wikipedia_preprocess as prep


def document(text, title="Synthetic article", revision="10"):
    return corpus.validate_document({"title": title, "text": text, "source": "encyclopedia",
        "source_url": f"https://en.wikipedia.org/w/index.php?oldid={revision}", "revision_id": revision,
        "retrieved_at": "2026-11-16T10:00:00Z", "language": "en"})


def passages(text, **kwargs):
    return prep.make_passages(document(text), {"revision_timestamp": "2026-06-30T12:00:00Z"}, **kwargs)


def text_of(rows):
    return "\n".join(row["text"] for row in rows)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preprocessing must never access a network")
    monkeypatch.setattr(acquisition, "urlopen", forbidden)


def snapshot(tmp_path, monkeypatch, texts=None):
    texts = ["Synthetic article with [[visible|display]] text."] if texts is None else texts
    titles = [f"Synthetic article {i}" for i in range(len(texts))]
    input_path = tmp_path / "article_list.jsonl"
    input_path.write_bytes(b"".join(corpus._json_bytes({"title": title}) for title in titles))
    by_title = dict(zip(titles, texts))

    def fetch(endpoint, params, user_agent, delay):
        title = params["titles"]
        i = titles.index(title) + 1
        payload = {"query": {"rightsinfo": {"text": "Synthetic license", "url": "https://example.org/license"},
            "pages": [{"title": title, "pageid": i, "ns": 0, "pagelanguage": "en",
                "revisions": [{"revid": 10 + i, "timestamp": "2026-06-30T12:00:00Z",
                    "slots": {"main": {"contentmodel": "wikitext", "contentformat": "text/x-wiki", "content": by_title[title]}}}]}]}}
        body = corpus._json_bytes(payload)
        return {"schema_version": acquisition.POLICY_VERSION, "request": {"endpoint": endpoint, "params": params},
                "response_body": body.decode("utf-8"), "response_sha256": corpus._sha256(body),
                "retrieved_at": "2026-11-16T10:00:00Z", "user_agent": user_agent, "headers": {}}

    monkeypatch.setattr(acquisition, "_fetch", fetch)
    path, cache = tmp_path / "corpus.jsonl", tmp_path / "cache"
    result = acquisition.collect_wikipedia(input_path=input_path, output_path=path, cache_dir=cache,
        allow_network=True, user_agent="SyntheticBot/1.0 (https://example.org)", cutoff="2026-07-01T00:00:00Z")
    assert result["status"] == "complete"
    return path, cache


def test_links_nested_display_trails_and_unicode():
    rows, diagnostics = passages("[[Target|display [[Inner|label]]]] and [[land border]]s. caf\u00e9 \u6771\u4eac &amp; e\u0301.")
    assert text_of(rows) == "display label and land borders. caf\u00e9 \u6771\u4eac & e\u0301."
    assert diagnostics["unprocessed_events"] == 0


def test_infobox_preserves_values_without_alias_or_numeric_normalization():
    text = "{{Infobox venue|name=[[Synthetic venue|Venue]]|capacity=12,345|area=1.2300|listed={{hlist|[[Exchange A]]|[[Exchange B]]}}}}"
    rows, _ = passages(text)
    cleaned = text_of(rows)
    for fragment in ("capacity: 12,345", "area: 1.2300", "name: Venue", "listed: Exchange A; Exchange B"):
        assert fragment in cleaned


def test_quantity_uses_explicit_source_number_unit_only():
    rows, diagnostics = passages("Area {{convert|28748|sqkm|abbr=on}}; range {{convert|10|to|20|km}}.")
    assert "28748 sqkm" in text_of(rows)
    assert "[UNPROCESSED template]" in text_of(rows)
    assert diagnostics["unprocessed_events"] == 1
    assert any("{{convert|10|to|20|km}}" == event["raw_wikitext"] for event in diagnostics["events"])


def test_unknown_nested_templates_are_not_guessed():
    text = "{{Infobox award|recipient={{Unknown|[[Person A]]|1999}}}}"
    rows, diagnostics = passages(text)
    assert "recipient: [UNPROCESSED template]" in text_of(rows)
    event = next(e for e in diagnostics["events"] if e["action"] == "unprocessed")
    assert event["raw_wikitext"] == "{{Unknown|[[Person A]]|1999}}"
    assert "Person A" not in text_of(rows)  # Raw arguments do not establish display semantics.
    assert event["source_block"]["start"] == 0 and event["source_block"]["end"] == len(text)


def test_references_comments_and_maintenance_are_auditable():
    source = "{{Short description|Synthetic}}Alpha<ref>{{cite web|title=Source 123}}</ref>beta<!-- hidden 456 -->."
    rows, diagnostics = passages(source)
    assert text_of(rows) == "Alpha beta ."
    assert {e["kind"] for e in diagnostics["events"]} == {"citation_or_maintenance_template", "reference", "comment"}
    assert any("123" in e["raw_wikitext"] for e in diagnostics["events"])


def test_tables_preserve_headers_numeric_values_and_recipients():
    source = '{| class="wikitable"\n|+ Synthetic awards\n! Year !! Recipient !! Amount\n|-\n| 2001 || [[Person|Person A]] || 1,234.50\n|-\n| 2002 || Person B || 44.80\n|}'
    rows, diagnostics = passages(source)
    text = text_of(rows)
    assert "Table caption: Synthetic awards" in text
    assert "Year: 2001; Recipient: Person A; Amount: 1,234.50" in text
    assert "Year: 2002; Recipient: Person B; Amount: 44.80" in text
    assert not diagnostics["unprocessed_events"]


@pytest.mark.parametrize("source", [
    '{|\n! A !! B\n|-\n| colspan="2" | 123\n|}',
    '{|\n! A !! B\n|-\n| 123\n|}',
    '{|\n| A || {|\n| 123\n|}\n|}',
])
def test_ambiguous_tables_keep_raw_diagnostics(source):
    rows, diagnostics = passages(source)
    assert "[UNPROCESSED table]" in text_of(rows)
    assert any(event["kind"] == "table" and "123" in event["raw_wikitext"] for event in diagnostics["events"])


def test_section_hierarchy_and_no_cross_section_windows():
    source = "Lead.\n== Geography ==\nOuter text.\n=== Borders ===\nBorder text.\n== Economy ==\nListing text."
    rows, diagnostics = passages(source, chunk_words=200, overlap_words=0)
    assert [r["section_path"] for r in rows] == [[], ["Geography"], ["Geography", "Borders"], ["Economy"]]
    assert [r["heading_level"] for r in rows] == [0, 2, 3, 2]
    assert [r["text"] for r in rows] == ["Lead.", "Outer text.", "Border text.", "Listing text."]
    assert diagnostics["sections"][2]["heading_span"]["start_line"] == 4


def test_fixed_overlap_order_ids_and_coarse_source_spans():
    source = "zero one two three four five six seven eight nine"
    rows, _ = passages(source, chunk_words=4, overlap_words=1)
    assert [r["text"] for r in rows] == ["zero one two three", "three four five six", "six seven eight nine"]
    assert [(r["word_start"], r["word_end"]) for r in rows] == [(0, 4), (3, 7), (6, 10)]
    assert rows == passages(source, chunk_words=4, overlap_words=1)[0]
    assert len({r["passage_id"] for r in rows}) == 3
    assert rows[0]["source_spans"] == rows[-1]["source_spans"]  # Same enclosing Text node, not exact substring offsets.
    for row in rows:
        assert row["source_spans"][0]["end"] == len(source)
        assert row["content_sha256"] == corpus._sha256(row["text"].encode("utf-8"))
        assert row["doc_id"] == document(source).doc_id
        assert row["revision_id"] == "10" and "oldid=10" in row["source_url"]


@pytest.mark.parametrize("source", ["Prose {{broken|123", "Prose [[broken link", "Prose <ref>unclosed citation 123"])
def test_malformed_wikitext_is_flagged_and_retained(source):
    rows, diagnostics = passages(source)
    assert "Prose" in text_of(rows)
    assert any(e["action"] == "retained_verbatim" for e in diagnostics["events"])
    assert any(e["raw_wikitext"] in source for e in diagnostics["events"])


def test_deleted_values_are_not_silently_promoted_to_positive_claims():
    rows, _ = passages("Capacity <del>500</del> <ins>600</ins> m<sup>2</sup>.")
    assert "[deleted: 500]" in text_of(rows) and "[inserted: 600]" in text_of(rows)
    assert "m^{2}" in text_of(rows)


@pytest.mark.parametrize("chunk,overlap", [(0, 0), (3, 3), (3, -1), (True, 0)])
def test_bad_chunk_config(chunk, overlap):
    with pytest.raises(prep.PreprocessingError):
        passages("Text", chunk_words=chunk, overlap_words=overlap)


def test_snapshot_and_byte_identical_replay(tmp_path, monkeypatch):
    path, cache = snapshot(tmp_path, monkeypatch, ["Lead [[text]].\n== Data ==\n{{Infobox venue|capacity=12,345}}"])
    original_files = [path, tmp_path / "corpus.jsonl.manifest.json", tmp_path / "corpus.jsonl.collection.json", *cache.iterdir()]
    before = {p: p.read_bytes() for p in original_files}
    first, second = tmp_path / "passages.jsonl", tmp_path / "replay.jsonl"
    manifest = prep.preprocess_corpus(corpus_path=path, output_path=first, cache_dir=cache)
    assert manifest == prep.preprocess_corpus(corpus_path=path, output_path=second, cache_dir=cache)
    for suffix in ("", ".manifest.json", ".diagnostics.jsonl"):
        assert (tmp_path / (first.name + suffix)).read_bytes() == (tmp_path / (second.name + suffix)).read_bytes()
    assert all(p.read_bytes() == data for p, data in before.items())
    assert manifest["cache_validation"] == "verified"
    assert manifest["inputs"]["corpus"]["sha256"] == corpus._sha256(path.read_bytes())
    assert manifest["passages_sha256"] == corpus._sha256(first.read_bytes())
    assert manifest["document_provenance"][0]["license"]["text"] == "Synthetic license"
    with pytest.raises(prep.PreprocessingError, match="overwrite"):
        prep.preprocess_corpus(corpus_path=path, output_path=first)


@pytest.mark.parametrize("field,value", [("status", "incomplete"), ("manifest_sha256", "0" * 64),
                                       ("selected_records", 999), ("text_representation", "current-extract"),
                                       ("endpoint", None), ("input_records", -1)])
def test_bad_collection_fails_before_publication(tmp_path, monkeypatch, field, value):
    path, _ = snapshot(tmp_path, monkeypatch)
    collection_path = tmp_path / "corpus.jsonl.collection.json"
    data = json.loads(collection_path.read_bytes())
    data[field] = value
    collection_path.write_bytes(corpus._json_bytes(data))
    output = tmp_path / "out.jsonl"
    with pytest.raises(prep.PreprocessingError):
        prep.preprocess_corpus(corpus_path=path, output_path=output)
    assert not output.exists()


def test_provenance_mismatch_and_missing_metadata_fail(tmp_path, monkeypatch):
    path, _ = snapshot(tmp_path, monkeypatch)
    collection_path = tmp_path / "corpus.jsonl.collection.json"
    data = json.loads(collection_path.read_bytes())
    data["records"][0]["revision_id"] = "999"
    collection_path.write_bytes(corpus._json_bytes(data))
    with pytest.raises(prep.PreprocessingError, match="revision_id mismatch"):
        prep.load_snapshot(path)
    collection_path.unlink()
    with pytest.raises(corpus.CorpusError, match="could not read"):
        prep.load_snapshot(path)


def test_tampered_corpus_manifest_and_cache_rejected(tmp_path, monkeypatch):
    path, cache = snapshot(tmp_path, monkeypatch)
    cache_file = next(cache.iterdir())
    original = cache_file.read_bytes()
    cache_file.write_bytes(original + b" ")
    with pytest.raises(prep.PreprocessingError, match="cache hash"):
        prep.load_snapshot(path, cache_dir=cache)
    cache_file.write_bytes(original)
    path.write_bytes(path.read_bytes().replace(b"Synthetic article with", b"Changed article with"))
    with pytest.raises((prep.PreprocessingError, corpus.CorpusError)):
        prep.load_snapshot(path)


def test_revision_metadata_must_agree_with_supplied_api_cache(tmp_path, monkeypatch):
    path, cache = snapshot(tmp_path, monkeypatch)
    collection_path = tmp_path / "corpus.jsonl.collection.json"
    data = json.loads(collection_path.read_bytes())
    data["records"][0]["revision_timestamp"] = "2026-06-29T00:00:00Z"
    collection_path.write_bytes(corpus._json_bytes(data))
    with pytest.raises(prep.PreprocessingError, match="provenance differs"):
        prep.load_snapshot(path, cache_dir=cache)


def test_duplicate_infobox_parameters_are_not_resolved_by_guessing():
    rows, diagnostic = passages("{{Infobox venue|capacity=100|capacity=200}}")
    assert "[UNPROCESSED template]" in text_of(rows)
    assert any(event["reason"] == "duplicate parameters" for event in diagnostic["events"])


def test_repeated_heading_paths_still_have_unique_passage_ids():
    rows, _ = passages("== Same ==\nSame text\n== Same ==\nSame text")
    assert rows[0]["section_path"] == rows[1]["section_path"]
    assert rows[0]["passage_id"] != rows[1]["passage_id"]


def test_cli(tmp_path, monkeypatch, capsys):
    path, _ = snapshot(tmp_path, monkeypatch)
    assert prep.main(["--corpus", str(path), "--output", str(tmp_path / "out.jsonl")]) == 0
    assert "passages:" in capsys.readouterr().out
