"""Deterministic, offline ingestion of externally supplied evidence text.

No network, benchmark reader, prediction loader, or model dependency is used.
Provenance restrictions reject identifiable benchmark artifacts; they cannot
certify the origin, independence, licensing, or factual accuracy of arbitrary text.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit


SCHEMA_VERSION = "evidence-corpus-v1"
EXTERNAL_SOURCES = frozenset({
    "encyclopedia", "official_website", "news", "research_publication", "other_external",
})
_REQUIRED = {"title", "text", "source", "source_url", "retrieved_at", "language"}
_OPTIONAL = {"doc_id", "content_sha256", "revision_id"}
_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)
_LANGUAGE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
_BENCHMARK_MARKER = re.compile(
    r"(?<![a-z0-9])(?:lm[-_]?kbc|llm[-_]kbc[-_]thesis|ravec|kamel|knowledgenet)(?![a-z0-9])"
)


class CorpusError(ValueError):
    """An input is invalid or cannot support a provenance-preserving corpus."""


@dataclass(frozen=True)
class EvidenceDocument:
    """One source document/revision; strings are preserved without normalization.

    Construct validated instances with validate_document() or load_documents().
    source declares an external-source category, not a credibility score.
    retrieved_at records acquisition time, never the validity date of a fact.
    """

    doc_id: str
    title: str
    text: str
    source: str
    source_url: str
    retrieved_at: str
    language: str
    content_sha256: str
    revision_id: str | None = None


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _string(value: object, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise CorpusError(f"{location}: {field} must be a nonempty string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CorpusError(f"{location}: {field} contains an invalid Unicode surrogate") from error
    return value


def validate_document(value: object, *, location: str = "document") -> EvidenceDocument:
    """Validate or derive IDs/hashes; never repair, strip, or normalize content.

    Inputs may omit derived doc_id/content_sha256. Supplied values must match.
    Unknown fields fail, including benchmark answers and prediction/evaluation
    payloads. revision_id is an opaque source identifier, not a parsed date.
    """
    if not isinstance(value, dict):
        raise CorpusError(f"{location}: expected an evidence JSON object")
    if any(not isinstance(key, str) for key in value):
        raise CorpusError(f"{location}: field names must be strings")
    if set(value) & {"SubjectEntity", "Relation", "ObjectEntities", "predictions", "gold", "metrics"}:
        raise CorpusError(f"{location}: benchmark gold, predictions and evaluation records are not evidence")
    missing, unknown = _REQUIRED - value.keys(), value.keys() - _REQUIRED - _OPTIONAL
    if missing:
        raise CorpusError(f"{location}: missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise CorpusError(f"{location}: unknown fields: {', '.join(sorted(unknown))}")
    fields = {field: _string(value[field], field, location) for field in sorted(_REQUIRED)}
    if fields["source"] not in EXTERNAL_SOURCES:
        raise CorpusError(f"{location}: source must be an external category: {', '.join(sorted(EXTERNAL_SOURCES))}")
    if not _LANGUAGE.fullmatch(fields["language"]):
        raise CorpusError(f"{location}: language must be a hyphen-separated language tag, e.g. en or el-GR")
    timestamp = fields["retrieved_at"]
    try:
        if not _TIMESTAMP.fullmatch(timestamp) or timestamp.endswith("-00:00"):
            raise ValueError("require a known explicit timezone")
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise CorpusError(f"{location}: retrieved_at must be an RFC3339 timestamp with a known timezone") from error
    url = fields["source_url"]
    try:
        parsed = urlsplit(url)
        valid_url = (parsed.scheme in {"http", "https"} and parsed.hostname
                     and parsed.username is None and parsed.password is None
                     and parsed.port != 0)
    except ValueError as error:
        raise CorpusError(f"{location}: invalid source_url") from error
    if not valid_url or any(char.isspace() or ord(char) <= 32 or ord(char) == 127 or char in '\\<>"{}|^`' for char in url):
        raise CorpusError(f"{location}: source_url must be an absolute HTTP(S) URL without credentials, invalid characters or whitespace")
    decoded_url = unquote(url).casefold()
    if _BENCHMARK_MARKER.search(decoded_url) or parsed.hostname in {"akbc.ws", "www.akbc.ws"}:
        raise CorpusError(f"{location}: benchmark/RAVEC source URLs are excluded from evidence")
    revision = value.get("revision_id")
    if revision is not None:
        revision = _string(revision, "revision_id", location)
    content_hash = _sha256(fields["text"].encode("utf-8"))
    identity = {key: fields[key] for key in ("source", "source_url", "language")}
    identity.update(revision_id=revision, content_sha256=content_hash)
    doc_id = "doc-" + _sha256(_json_bytes(identity))
    for key, actual in (("content_sha256", content_hash), ("doc_id", doc_id)):
        if key in value and value[key] != actual:
            raise CorpusError(f"{location}: {key} does not match the derived value")
    return EvidenceDocument(**fields, revision_id=revision, doc_id=doc_id, content_sha256=content_hash)


def serialize_document(document: EvidenceDocument) -> bytes:
    """Return canonical UTF-8 JSONL, revalidating manually constructed instances."""
    validated = validate_document(asdict(document))
    return _json_bytes(asdict(validated))


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CorpusError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str):
    raise CorpusError(f"nonstandard JSON constant: {value}")


def _documents(content: bytes, location: str) -> list[EvidenceDocument]:
    documents, ids, hashes = [], {}, {}
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CorpusError(f"{location}: evidence JSONL must be UTF-8") from error
    # Split only on LF: Unicode line separators inside JSON strings are content.
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    for index, line in enumerate(lines, 1):
        where = f"{location}:{index}"
        try:
            value = json.loads(line, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
        except (json.JSONDecodeError, CorpusError) as error:
            raise CorpusError(f"{where}: invalid JSON: {error}") from error
        document = validate_document(value, location=where)
        for key, seen, label in ((document.doc_id, ids, "doc_id"),
                                 (document.content_sha256, hashes, "content")):
            if key in seen:
                raise CorpusError(f"{where}: duplicate {label} (first at line {seen[key]}); curate explicitly, never silently deduplicate")
            seen[key] = index
        documents.append(document)
    return documents


def _check_location(path: Path) -> None:
    # Defense in depth for known local benchmark/artifact locations, including
    # symlinks. Schema/provenance checks are also required; names alone prove nothing.
    parts = {part.casefold() for part in path.resolve().parts}
    if (parts & {"results", "experiments"} or
        any(re.fullmatch(r"(?:lm[-_]kbc[-_]?|dataset)20[0-9]{2}", part) for part in parts) or
        path.resolve().name in {"train.jsonl", "val.jsonl", "test.jsonl"}):
        raise CorpusError(f"{path}: benchmark/evaluation/experiment locations are excluded")


def _input_bytes(path: Path) -> bytes:
    _check_location(path)
    try:
        return path.read_bytes()
    except OSError as error:
        raise CorpusError(f"{path}: could not read evidence input: {error.strerror}") from error


def load_documents(path: str | Path) -> list[EvidenceDocument]:
    """Read/validate a complete external-document snapshot in file order."""
    path = Path(path)
    return _documents(_input_bytes(path), str(path))


def _publish(path: Path, content: bytes) -> None:
    """Atomic, no-clobber publication on filesystems supporting hard links."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ingest_corpus(input_path: str | Path, output_path: str | Path) -> dict:
    """Validate all inputs, then publish corpus and manifest without overwriting.

    Manifest is published last and is required to recognize a complete corpus.
    No wall-clock time or filesystem paths enter the reproducible manifest.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    manifest_path = Path(str(output_path) + ".manifest.json")
    for path in (output_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise CorpusError(f"Refusing to overwrite {path}")
        _check_location(path)
    if not output_path.parent.is_dir():
        raise CorpusError(f"Output directory must already exist: {output_path.parent}")
    content = _input_bytes(input_path)
    documents = sorted(_documents(content, str(input_path)), key=lambda document: document.doc_id)
    corpus = b"".join(serialize_document(document) for document in documents)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "document_count": len(documents),
        "input_sha256": _sha256(content),
        "corpus_sha256": _sha256(corpus),
        "id_policy": "sha256-canonical-source-url-language-revision-content-v1",
        "duplicate_policy": "reject-identical-doc-id-or-text-sha256",
        "ingestion_policy": "external-text-only-no-benchmark-artifacts-v1",
        "documents": [{key: value for key, value in asdict(document).items() if key != "text"}
                      for document in documents],
    }
    manifest_bytes = _json_bytes(manifest)
    if _input_bytes(input_path) != content:
        raise CorpusError(f"{input_path}: input changed during ingestion")
    _publish(output_path, corpus)
    _publish(manifest_path, manifest_bytes)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Offline external-document JSONL")
    parser.add_argument("--output", required=True, type=Path, help="New corpus JSONL; parent must exist")
    args = parser.parse_args(argv)
    try:
        manifest = ingest_corpus(args.input, args.output)
    except (CorpusError, OSError) as error:
        parser.exit(2, f"Corpus ingestion failed: {error}\n")
    print(f"Documents: {manifest['document_count']}; corpus SHA256: {manifest['corpus_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
