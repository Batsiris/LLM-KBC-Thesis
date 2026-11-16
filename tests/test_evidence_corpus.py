"""Synthetic offline corpus fixtures; no evidence acquisition or factual claims."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from llm_kbc import evidence_corpus as corpus


def record(**changes):
    return {
        "title": "Synthetic source",
        "text": "Synthetic fixture text.",
        "source": "other_external",
        "source_url": "https://example.org/document",
        "retrieved_at": "2026-06-30T12:30:00+03:00",
        "revision_id": "source-revision-1",
        "language": "en",
        **changes,
    }


def write(path, rows):
    path.write_bytes(b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows))
    return path


def test_document_validation_and_hashes():
    original = record()
    document = corpus.validate_document(original)
    assert isinstance(document, corpus.EvidenceDocument)
    assert document.content_sha256 == hashlib.sha256(original["text"].encode("utf-8")).hexdigest()
    identity = {key: original[key] for key in ("source", "source_url", "revision_id", "language")}
    identity["content_sha256"] = document.content_sha256
    canonical = (json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    assert document.doc_id == "doc-" + hashlib.sha256(canonical).hexdigest()
    assert corpus.validate_document(asdict(document)) == document
    assert document.retrieved_at == original["retrieved_at"]
    assert document.revision_id == original["revision_id"]
    assert corpus.validate_document(record(revision_id=None)).revision_id is None
    without_revision = record()
    del without_revision["revision_id"]
    assert corpus.validate_document(without_revision).revision_id is None


def test_ids_are_retrieval_time_independent_but_source_revision_sensitive():
    original = corpus.validate_document(record())
    later = corpus.validate_document(record(retrieved_at="2026-08-01T14:00:00Z", title="Retitled"))
    assert original.doc_id == later.doc_id
    assert original != later
    for changes in ({"revision_id": "revision-2"}, {"text": "Different synthetic text"},
                    {"source_url": "https://example.org/another"}, {"language": "el"},
                    {"source": "encyclopedia"}):
        assert corpus.validate_document(record(**changes)).doc_id != original.doc_id


def test_unicode_and_exact_text_round_trip(tmp_path):
    text = "  \u0391\u03b8\u03ae\u03bd\u03b1 \u2014 caf\u00e9 e\u0301 \u6771\u4eac \U0001f4da\r\n\tline\u2028separator\u2029  "
    value = record(title="\u0394\u03bf\u03ba\u03b9\u03bc\u03ae", text=text, language="el-GR")
    document = corpus.validate_document(value)
    encoded = corpus.serialize_document(document)
    assert "\u0391\u03b8\u03ae\u03bd\u03b1".encode("utf-8") in encoded
    assert json.loads(encoded)["text"] == text
    assert document.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = tmp_path / "external.jsonl"
    path.write_bytes(encoded)
    assert corpus.load_documents(path) == [document]
    assert corpus.serialize_document(corpus.load_documents(path)[0]) == encoded


@pytest.mark.parametrize("field", ["title", "text", "source", "source_url", "retrieved_at", "language"])
def test_required_fields(tmp_path, field):
    value = record()
    del value[field]
    path = write(tmp_path / "external.jsonl", [value])
    with pytest.raises(corpus.CorpusError, match=rf"external.jsonl:1:.*{field}"):
        corpus.load_documents(path)


@pytest.mark.parametrize("changes", [
    {"title": " "}, {"text": ""}, {"text": 12}, {"text": ["text"]}, {"text": "nul\x00"},
    {"text": "bad\ud800"}, {"title": None}, {"revision_id": ""}, {"revision_id": 123},
    {"source": "benchmark_gold"}, {"source": "model_predictions"}, {"source": "evaluation"},
    {"source": "benchmark_answer_list"}, {"language": "en_US"}, {"language": "english text"},
    {"retrieved_at": "2026-02-30T00:00:00Z"}, {"retrieved_at": "2026-07-01"},
    {"retrieved_at": "2026-07-01T01:02:03"}, {"retrieved_at": "2026-07-01T01:02:03-00:00"},
    {"retrieved_at": "2026-07-01T01:02:03+03:99"},
    {"source_url": "file:///private/document"}, {"source_url": "not a URL"},
    {"source_url": "https://name:password@example.org"}, {"source_url": "https://example.org:bad"},
    {"source_url": "https://example.org/path with space"}, {"source_url": "https://example.org\\path"},
    {"doc_id": "invented"}, {"content_sha256": "0" * 64}, {"unknown": "metadata"},
])
def test_invalid_fields_rejected(changes):
    with pytest.raises(corpus.CorpusError):
        corpus.validate_document(record(**changes))


@pytest.mark.parametrize("url", [
    "https://github.com/lm-kbc/dataset2026/blob/main/data/val.jsonl",
    "https://raw.githubusercontent.com/lm-kbc/dataset2026/main/val.jsonl",
    "https://huggingface.co/datasets/lm-kbc/dataset2026",
    "https://lm-kbc.github.io/challenge2026/",
    "https://www.akbc.ws/2026/leaderboard.pdf",
    "https://example.org/benchmark/lm_kbc_2026/answers",
    "https://github.com/example/LLM-KBC-Thesis/results/report.json",
])
def test_known_benchmark_and_experiment_urls_excluded(url):
    with pytest.raises(corpus.CorpusError, match="source URLs are excluded"):
        corpus.validate_document(record(source_url=url))


@pytest.mark.parametrize("value", [
    {"SubjectEntity": "Synthetic", "Relation": "hasArea", "ObjectEntities": [["1"]]},
    {"SubjectEntity": "Synthetic", "Relation": "hasArea", "ObjectEntities": ["1"]},
    {"metrics": {"macro-f1": 0.1}}, record(ObjectEntities=[]), record(gold=["answer"]),
    record(predictions=["answer"]), [], None,
])
def test_benchmark_schemas_not_ingested(value):
    with pytest.raises(corpus.CorpusError):
        corpus.validate_document(value)


@pytest.mark.parametrize("name", ["val.jsonl", "train.jsonl", "test.jsonl", "results/external.jsonl",
                                  "experiments/external.jsonl", "data/lm_kbc_2026/external.jsonl"])
def test_benchmark_input_locations_excluded(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, [record()])
    with pytest.raises(corpus.CorpusError, match="locations are excluded"):
        corpus.load_documents(path)


def test_output_location_is_readable_by_the_same_corpus_policy(tmp_path):
    path = write(tmp_path / "external.jsonl", [record()])
    excluded = tmp_path / "results"
    excluded.mkdir()
    with pytest.raises(corpus.CorpusError, match="locations are excluded"):
        corpus.ingest_corpus(path, excluded / "corpus.jsonl")
    assert not list(excluded.iterdir())


@pytest.mark.parametrize("changes,label", [({}, "doc_id"),
    ({"retrieved_at": "2026-08-01T12:00:00Z", "title": "Other title"}, "doc_id"),
    ({"source_url": "https://example.net/copied", "source": "news"}, "content")])
def test_duplicates_fail_without_silent_deduplication(tmp_path, changes, label):
    path = write(tmp_path / "external.jsonl", [record(), record(**changes)])
    with pytest.raises(corpus.CorpusError, match=rf":2: duplicate {label}.*first at line 1"):
        corpus.load_documents(path)


@pytest.mark.parametrize("content", [b"\n", b" \n", b"{}\n\n", b"[1]\n", b"{broken}\n", b"\xff",
    b'{"title":"one","title":"two"}\n', b'{"text":NaN}\n', b'{"text":Infinity}\n'])
def test_invalid_jsonl(tmp_path, content):
    path = tmp_path / "external.jsonl"
    path.write_bytes(content)
    with pytest.raises(corpus.CorpusError, match="external.jsonl"):
        corpus.load_documents(path)


def test_manifest_and_outputs_are_byte_identical_across_paths(tmp_path):
    original = write(tmp_path / "external.jsonl", [record(), record(text="Another fixture", source_url="https://example.org/second")])
    relocated = tmp_path / "relocated.jsonl"
    relocated.write_bytes(original.read_bytes())
    before = original.read_bytes()
    first, second = tmp_path / "corpus_a.jsonl", tmp_path / "corpus_b.jsonl"
    manifest = corpus.ingest_corpus(original, first)
    assert corpus.ingest_corpus(relocated, second) == manifest
    assert first.read_bytes() == second.read_bytes()
    first_manifest = tmp_path / "corpus_a.jsonl.manifest.json"
    assert first_manifest.read_bytes() == (tmp_path / "corpus_b.jsonl.manifest.json").read_bytes()
    assert original.read_bytes() == before
    assert manifest["schema_version"] == corpus.SCHEMA_VERSION
    assert manifest["document_count"] == 2
    assert manifest["corpus_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert manifest["input_sha256"] == hashlib.sha256(before).hexdigest()
    rows = [json.loads(line) for line in first.read_bytes().splitlines()]
    assert [r["doc_id"] for r in rows] == sorted(r["doc_id"] for r in rows)
    assert manifest["documents"] == [{key: value for key, value in row.items() if key != "text"} for row in rows]
    assert all(item["retrieved_at"] == record()["retrieved_at"] for item in manifest["documents"])
    assert corpus.load_documents(first)  # Emitted corpus is accepted unchanged.


def test_empty_corpus(tmp_path):
    path = write(tmp_path / "external.jsonl", [])
    assert corpus.load_documents(path) == []
    output = tmp_path / "empty_corpus.jsonl"
    manifest = corpus.ingest_corpus(path, output)
    assert output.read_bytes() == b""
    assert manifest["document_count"] == 0 and manifest["documents"] == []
    assert manifest["corpus_sha256"] == hashlib.sha256(b"").hexdigest()


def test_no_outputs_on_validation_error_and_no_overwrites(tmp_path):
    path = write(tmp_path / "external.jsonl", [record(), record()])
    output = tmp_path / "corpus.jsonl"
    manifest_path = tmp_path / "corpus.jsonl.manifest.json"
    with pytest.raises(corpus.CorpusError):
        corpus.ingest_corpus(path, output)
    assert not output.exists() and not manifest_path.exists()
    write(path, [record()])
    corpus.ingest_corpus(path, output)
    before = output.read_bytes(), manifest_path.read_bytes()
    with pytest.raises(corpus.CorpusError, match="overwrite"):
        corpus.ingest_corpus(path, output)
    assert before == (output.read_bytes(), manifest_path.read_bytes())
    output.unlink()
    with pytest.raises(corpus.CorpusError, match="overwrite"):
        corpus.ingest_corpus(path, output)
    assert not output.exists()


def test_manually_constructed_invalid_documents_cannot_be_serialized():
    document = corpus.validate_document(record())
    with pytest.raises(corpus.CorpusError, match="content_sha256"):
        corpus.serialize_document(replace(document, text="Changed text"))


def test_cli_and_clear_missing_file_error(tmp_path, capsys):
    path = tmp_path / "external.jsonl"
    output = tmp_path / "corpus.jsonl"
    args = ["--input", str(path), "--output", str(output)]
    with pytest.raises(SystemExit) as error:
        corpus.main(args)
    assert error.value.code == 2
    assert "could not read evidence input" in capsys.readouterr().err
    write(path, [record()])
    assert corpus.main(args) == 0
    assert "Documents: 1" in capsys.readouterr().out
