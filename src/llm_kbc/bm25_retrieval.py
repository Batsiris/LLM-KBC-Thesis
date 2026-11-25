"""Offline BM25 over one validated Wikipedia passage snapshot; no model or gold use."""

import argparse
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from . import evidence_corpus as corpus
from . import wikipedia_preprocess as preprocessing
from .wikipedia_collect import _json_object


VERSION = "bm25-retrieval-v1"
QUERY_VERSION = "subject-relation-lexical-v1"
TOKENIZATION_VERSION = "nfkc-casefold-unicode-alphanumeric-v1"
QUERIES = {
    "awardWonBy": "{subject} award winners recipients",
    "companyTradesAtStockExchange": "{subject} stock exchange listing",
    "countryLandBordersCountry": "{subject} land borders neighbouring countries",
    "hasArea": "{subject} total area",
    "hasCapacity": "{subject} capacity",
    "personHasCityOfDeath": "{subject} place of death",
}
K1 = 1.2
B = 0.75


class RetrievalError(ValueError):
    """Invalid inputs or unsafe publication prevent a complete retrieval artifact."""


def tokenize(text: str) -> list[str]:
    """NFKC then casefold; maximal Unicode letter/digit runs, no numeric parsing.

    Punctuation, underscores and remaining combining marks delimit tokens.
    Decimal/grouping punctuation splits numbers; signs are discarded. No stemming,
    stopwords, aliases, transliteration, or unit conversion is performed.
    """
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def build_query(subject: str, relation: str) -> str:
    """Preserve the subject spelling; relation expansions contain no answer labels."""
    corpus._string(subject, "SubjectEntity", "query")
    if not isinstance(relation, str) or relation not in QUERIES:
        raise RetrievalError(f"Unsupported relation: {relation!r}")
    return QUERIES[relation].format(subject=subject)


def _rows(data: bytes, location: str) -> list[dict]:
    try:
        lines = data.decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise RetrievalError(f"{location}: JSONL must be UTF-8") from error
    if lines[-1] == "":
        lines.pop()
    return [_json_object(line, f"{location}:{index}") for index, line in enumerate(lines, 1)]


def load_identities(path: str | Path) -> list[dict]:
    """Project identities only; never access, validate or fingerprint gold fields."""
    rows, seen = [], set()
    for index, row in enumerate(_rows(Path(path).read_bytes(), str(path)), 1):
        subject, relation = row.get("SubjectEntity"), row.get("Relation")
        build_query(subject, relation)
        identity = (" ".join(unicodedata.normalize("NFKC", subject).casefold().split()), relation)
        if identity in seen:
            raise RetrievalError(f"{path}:{index}: duplicate normalized subject-relation identity")
        seen.add(identity)
        rows.append({"SubjectEntity": subject, "Relation": relation})
    if not rows:
        raise RetrievalError(f"{path}: no input identities")
    return rows


def _hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_passages(passages_path: str | Path, corpus_path: str | Path, *,
                  manifest_path=None, corpus_manifest=None, collection_path=None, cache_dir=None):
    """Validate the existing passage schema and its linked, complete source snapshot.

    Diagnostics are required at <passages>.diagnostics.jsonl. Optional original API
    caches are reverified when supplied; archived 'verified' is not a fresh audit.
    No wikitext parser, template expansion or network access is invoked.
    """
    path = Path(passages_path)
    manifest_path = Path(manifest_path or str(path) + ".manifest.json")
    diagnostics_path = Path(str(path) + ".diagnostics.jsonl")
    source_path = Path(corpus_path)
    paths = {"passages": path, "passage_manifest": manifest_path, "diagnostics": diagnostics_path,
             "corpus": source_path, "manifest": Path(corpus_manifest or str(source_path) + ".manifest.json"),
             "collection": Path(collection_path or str(source_path) + ".collection.json")}
    original = {key: corpus._input_bytes(p) for key, p in paths.items()}
    documents, acquisition, fingerprints = preprocessing.load_snapshot(
        source_path, corpus_manifest, collection_path, cache_dir)
    for key in fingerprints:
        if key.startswith("cache:"):
            paths[key] = Path(cache_dir) / (key.removeprefix("cache:") + ".json")
            original[key] = corpus._input_bytes(paths[key])
    manifest = _json_object(original["passage_manifest"], str(manifest_path))
    config = manifest.get("configuration", {})
    if (manifest.get("schema_version") != "evidence-passages-v1" or manifest.get("status") != "complete"
            or manifest.get("preprocessing_version") != preprocessing.VERSION
            or manifest.get("mwparserfromhell") != preprocessing.PARSER_VERSION
            or not isinstance(config, dict)):
        raise RetrievalError("Incomplete or incompatible passage manifest")
    passage_config = preprocessing._configuration(config.get("chunk_words"), config.get("overlap_words"))
    if config != {"chunk_words": passage_config["chunk_words"], "overlap_words": passage_config["overlap_words"],
                  "word_definition": "Unicode non-whitespace runs", "section_local": True}:
        raise RetrievalError("Incompatible passage configuration")
    recorded = manifest.get("inputs")
    if not isinstance(recorded, dict) or not {"corpus", "manifest", "collection"} <= recorded.keys():
        raise RetrievalError("Passage manifest source fingerprints mismatch")
    cache_references = {}
    for record in acquisition.values():
        cache = record.get("cache")
        if isinstance(cache, dict) and _hash(cache.get("key")) and _hash(cache.get("sha256")):
            cache_references["cache:" + cache["key"]] = cache["sha256"]
    for key, value in recorded.items():
        if (not isinstance(value, dict) or not _hash(value.get("sha256"))
                or type(value.get("bytes")) is not int or value["bytes"] < 0):
            raise RetrievalError("Invalid source fingerprint")
        if key in fingerprints:
            if value != fingerprints[key]:
                raise RetrievalError(f"Passage manifest source fingerprints mismatch: {key}")
        elif cache_dir is None and key in cache_references:
            # Bind archived cache hashes to the collection metadata. Without cache
            # bytes, their actual content and byte counts are not freshly verified.
            if value["sha256"] != cache_references[key]:
                raise RetrievalError(f"Passage manifest source fingerprints mismatch: {key}")
        else:
            raise RetrievalError(f"Unmatched archived source fingerprint: {key}")
    # Fresh cache verification may add fingerprints absent from an older passage
    # manifest; load_snapshot has checked their requests, bytes and source content.
    for key in ("passages", "diagnostics"):
        if manifest.get(key + "_sha256") != corpus._sha256(original[key]):
            raise RetrievalError(f"{key} fingerprint mismatch")
    cache_validation = manifest.get("cache_validation")
    if (not _hash(manifest.get("code_sha256")) or not isinstance(cache_validation, str)
            or cache_validation not in {"verified", "not_supplied_not_verified"}):
        raise RetrievalError("Invalid preprocessing provenance")
    by_id = {d.doc_id: d for d in documents}
    provenance = [{"doc_id": d.doc_id, "page_id": acquisition[d.doc_id]["page_id"],
                   "license": acquisition[d.doc_id]["license"],
                   "attribution_history_url": acquisition[d.doc_id]["attribution_history_url"],
                   "content_sha256": d.content_sha256} for d in documents]
    if type(manifest.get("document_count")) is not int or manifest["document_count"] != len(documents) or manifest.get("document_provenance") != provenance:
        raise RetrievalError("Passage document count/provenance mismatch")
    diagnostics = _rows(original["diagnostics"], str(diagnostics_path))
    if ([d.get("doc_id") for d in diagnostics] != list(by_id)
            or any(type(d.get("unprocessed_events")) is not int or d["unprocessed_events"] < 0 for d in diagnostics)
            or type(manifest.get("unprocessed_events")) is not int
            or manifest["unprocessed_events"] != sum(d["unprocessed_events"] for d in diagnostics)):
        raise RetrievalError("Diagnostic document coverage/count mismatch")
    rows = _rows(original["passages"], str(path))
    if type(manifest.get("passage_count")) is not int or manifest["passage_count"] != len(rows):
        raise RetrievalError("Passage count mismatch")
    seen, previous = set(), None
    document_order = {key: index for index, key in enumerate(by_id)}
    sections = {d["doc_id"]: d.get("sections") for d in diagnostics}
    fields = {"passage_id", "doc_id", "section_index", "word_start", "word_end", "configuration",
              "content_sha256", "article_title", "source_url", "revision_id", "revision_timestamp",
              "retrieved_at", "language", "section_path", "heading_level", "text", "source_spans", "alignment"}
    for index, row in enumerate(rows, 1):
        where = f"{path}:{index}"
        if set(row) != fields or not isinstance(row.get("passage_id"), str) or not isinstance(row.get("doc_id"), str):
            raise RetrievalError(f"{where}: invalid passage fields/identities")
        if row["passage_id"] in seen:
            raise RetrievalError(f"{where}: duplicate passage identity")
        doc = by_id.get(row.get("doc_id"))
        if doc is None:
            raise RetrievalError(f"{where}: unknown parent document")
        text = corpus._string(row.get("text"), "text", where)
        for key in ("section_index", "heading_level", "word_start", "word_end"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise RetrievalError(f"{where}: invalid {key}")
        if (not 0 <= row["heading_level"] <= 6 or row["word_end"] <= row["word_start"]
                or row["word_end"] - row["word_start"] != len(re.findall(r"\S+", text))
                or row["word_end"] - row["word_start"] > config["chunk_words"]
                or row["word_start"] % (config["chunk_words"] - config["overlap_words"])):
            raise RetrievalError(f"{where}: invalid passage word window")
        if (not isinstance(row.get("section_path"), list)
                or any(not isinstance(h, str) or not h.strip() for h in row["section_path"])
                or row.get("configuration") != passage_config
                or row.get("alignment") != "enclosing_top_level_nodes_not_exact_cleaned_spans"):
            raise RetrievalError(f"{where}: invalid section/configuration/alignment")
        section_records = sections[doc.doc_id]
        if (not isinstance(section_records, list) or row["section_index"] >= len(section_records)
                or not isinstance(section_records[row["section_index"]], dict)
                or any(section_records[row["section_index"]].get(key) != row[key] for key in ("section_path", "heading_level"))):
            raise RetrievalError(f"{where}: section does not match diagnostics")
        for key, value in {"article_title": doc.title, "source_url": doc.source_url, "revision_id": doc.revision_id,
                           "retrieved_at": doc.retrieved_at, "language": doc.language,
                           "revision_timestamp": acquisition[doc.doc_id]["revision_timestamp"]}.items():
            if row.get(key) != value:
                raise RetrievalError(f"{where}: {key} provenance mismatch")
        spans = row.get("source_spans")
        if not isinstance(spans, list) or not spans:
            raise RetrievalError(f"{where}: missing source spans")
        for span in spans:
            if (not isinstance(span, dict) or any(type(span.get(k)) is not int for k in ("start", "end", "start_line", "end_line"))
                    or not 0 <= span["start"] <= span["end"] <= len(doc.text)
                    or not 1 <= span["start_line"] <= span["end_line"] <= doc.text.count("\n") + 1):
                raise RetrievalError(f"{where}: invalid source span")
        digest = corpus._sha256(text.encode("utf-8"))
        identity = {k: row[k] for k in ("doc_id", "section_index", "word_start", "word_end", "configuration")}
        identity["content_sha256"] = digest
        if row.get("content_sha256") != digest or row.get("passage_id") != "passage-" + corpus._sha256(corpus._json_bytes(identity)):
            raise RetrievalError(f"{where}: passage content/identity fingerprint mismatch")
        order = (document_order[doc.doc_id], row["section_index"], row["word_start"])
        if previous is not None and order <= previous:
            raise RetrievalError(f"{where}: passage order is not canonical")
        seen.add(row["passage_id"])
        previous = order
    # Bind the separately validated source reads to these exact archived bytes.
    if any(fingerprints[k] != {"sha256": corpus._sha256(original[k]), "bytes": len(original[k])} for k in fingerprints):
        raise RetrievalError("Source snapshot changed during validation")
    return rows, manifest, paths, {k: corpus._sha256(v) for k, v in original.items()}


class BM25Index:
    """Text-only BM25, binary query-term frequency, no field weighting or index files."""

    def __init__(self, passages: list[dict]):
        self.passages = sorted(passages, key=lambda row: row["passage_id"])
        if len({r["passage_id"] for r in passages}) != len(passages):
            raise RetrievalError("Duplicate passage identity")
        self.lengths, self.postings = [], defaultdict(list)
        for index, row in enumerate(self.passages):
            terms = tokenize(row["text"])
            self.lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self.postings[term].append((index, count))
        self.average_length = sum(self.lengths) / len(passages) if passages else 0.0

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        if type(top_k) is not int or top_k < 1:
            raise RetrievalError("top_k must be a positive integer")
        scores = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (len(self.passages) - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                denominator = frequency + K1 * (1 - B + B * self.lengths[index] / self.average_length)
                scores[index] += idf * frequency * (K1 + 1) / denominator
        ranked = sorted(scores, key=lambda i: (-scores[i], self.passages[i]["passage_id"]))
        return [(self.passages[i], scores[i]) for i in ranked[:top_k] if scores[i] > 0]


def retrieve(*, passages_path, corpus_path, identities_path, output_path, top_k=5,
             manifest_path=None, corpus_manifest=None, collection_path=None, cache_dir=None) -> dict:
    """Validate everything before publication; output manifest is the completion marker."""
    if type(top_k) is not int or top_k < 1:
        raise RetrievalError("top_k must be a positive integer")
    output = Path(output_path)
    targets = [output, Path(str(output) + ".manifest.json")]
    for target in targets:
        if target.exists() or target.is_symlink():
            raise RetrievalError(f"Refusing to overwrite {target}")
    if not output.parent.is_dir():
        raise RetrievalError("Output parent directory must already exist")
    identities = load_identities(identities_path)
    passages, manifest, paths, hashes = load_passages(passages_path, corpus_path,
        manifest_path=manifest_path, corpus_manifest=corpus_manifest, collection_path=collection_path, cache_dir=cache_dir)
    index = BM25Index(passages)
    provenance = {d["doc_id"]: d for d in manifest["document_provenance"]}
    records = []
    for ordinal, identity in enumerate(identities, 1):
        query = build_query(identity["SubjectEntity"], identity["Relation"])
        hits = []
        for rank, (passage, score) in enumerate(index.search(query, top_k), 1):
            parent = provenance[passage["doc_id"]]
            hits.append({**passage, "rank": rank, "score": score, "source": "encyclopedia",
                         "page_id": parent["page_id"], "license": parent["license"],
                         "attribution_history_url": parent["attribution_history_url"],
                         "parent_content_sha256": parent["content_sha256"]})
        records.append({"input_index": ordinal, **identity, "query": query,
                        "status": "retrieved" if hits else "no_hits", "hits": hits})
    data = b"".join(corpus._json_bytes(row) for row in records)
    result = {"schema_version": VERSION, "status": "complete", "artifact_role": "ranked_candidates_not_packed_contexts",
              "inputs": {k: {"sha256": hashes[k], "bytes": paths[k].stat().st_size} for k in hashes},
              "source_inputs": manifest["inputs"], "archived_cache_validation": manifest["cache_validation"],
              "cache_revalidated": cache_dir is not None,
              "input_identities_sha256": corpus._sha256(b"".join(corpus._json_bytes(row) for row in identities)),
              "configuration": {"k1": K1, "b": B, "top_k": top_k, "idf": "log(1+(N-df+0.5)/(df+0.5))",
                                "query_term_frequency": "binary", "indexed_field": "text", "tie_break": "passage_id_ascending"},
              "query_policy_version": QUERY_VERSION, "tokenization_version": TOKENIZATION_VERSION,
              "unicode_version": unicodedata.unidata_version, "python_version": sys.version.split()[0],
              "code_sha256": corpus._sha256(Path(__file__).read_bytes()),
              "input_count": len(identities), "record_count": len(records), "passage_count": len(passages),
              "retrieved_count": sum(bool(r["hits"]) for r in records), "no_hit_count": sum(not r["hits"] for r in records),
              "output_sha256": corpus._sha256(data)}
    if load_identities(identities_path) != identities or any(corpus._sha256(p.read_bytes()) != hashes[k] for k, p in paths.items()):
        raise RetrievalError("Inputs changed before publication")
    corpus._publish(targets[0], data)
    corpus._publish(targets[1], corpus._json_bytes(result))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", required=True, type=Path, help="One evidence-passages-v1 JSONL")
    parser.add_argument("--corpus", required=True, type=Path, help="Original source corpus; sidecars required")
    parser.add_argument("--identities", required=True, type=Path, help="SubjectEntity/Relation JSONL; gold fields ignored")
    parser.add_argument("--output", required=True, type=Path, help="New retrieval JSONL; parent must exist")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--manifest", type=Path, help="Relocated passage manifest")
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--cache-dir", type=Path, help="Optional archived API cache revalidation; never downloads")
    args = parser.parse_args(argv)
    try:
        result = retrieve(passages_path=args.passages, corpus_path=args.corpus, identities_path=args.identities,
            output_path=args.output, top_k=args.top_k, manifest_path=args.manifest,
            corpus_manifest=args.corpus_manifest, collection_path=args.collection, cache_dir=args.cache_dir)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Retrieval failed: {error}\n")
    print(f"Records: {result['record_count']}; no hits: {result['no_hit_count']}; SHA256: {result['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
