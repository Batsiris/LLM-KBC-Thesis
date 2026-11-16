"""Offline, conservative wikitext extraction and section-local word windows.

This is source extraction, not MediaWiki rendering. Unsupported content is
retained verbatim in diagnostics. Source spans enclose top-level syntax nodes;
they are not character alignments between cleaned passages and wikitext.
"""

import argparse
import bisect
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import evidence_corpus as corpus
from . import wikipedia_collect as acquisition


VERSION = "wikipedia-preprocessing-v1"
PARSER_VERSION = "0.7.2"
_OMIT_TEMPLATES = {
    "short description", "about", "other uses", "hatnote group", "pp-move", "pp-sock",
    "use dmy dates", "use mdy dates", "use british english", "use american english",
    "main", "see also", "reflist", "refbegin", "refend", "sfn", "sfnp", "harvnb", "harv",
}
_UNITS = {"km", "m", "cm", "mm", "mi", "ft", "in", "sqkm", "km2", "m2", "sqmi", "ha", "acre"}
_NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


class PreprocessingError(ValueError):
    """Invalid snapshot, configuration, or unsupported parser version."""


def _parser():
    try:
        import mwparserfromhell
    except ModuleNotFoundError as error:
        raise PreprocessingError('Install the optional extra: python -m pip install -e ".[preprocessing]"') from error
    if mwparserfromhell.__version__ != PARSER_VERSION:
        raise PreprocessingError(f"Require mwparserfromhell=={PARSER_VERSION} for reproducible parsing")
    return mwparserfromhell


def load_snapshot(corpus_path, manifest_path=None, collection_path=None, cache_dir=None):
    """Check all three artifacts and optionally their immutable API caches.

    Returns documents, acquisition metadata by document ID and input fingerprints.
    Paths do not enter fingerprints, so relocation does not change derived output.
    """
    path = Path(corpus_path)
    paths = {"corpus": path, "manifest": Path(manifest_path or str(path) + ".manifest.json"),
             "collection": Path(collection_path or str(path) + ".collection.json")}
    content = {key: corpus._input_bytes(value) for key, value in paths.items()}
    manifest = acquisition._json_object(content["manifest"], str(paths["manifest"]))
    collection = acquisition._json_object(content["collection"], str(paths["collection"]))
    documents = corpus._documents(content["corpus"], str(path))
    expected = [{k: v for k, v in asdict(document).items() if k != "text"} for document in documents]
    count = len(documents)
    if (manifest.get("schema_version") != corpus.SCHEMA_VERSION
            or type(manifest.get("document_count")) is not int or manifest["document_count"] != count
            or manifest.get("corpus_sha256") != corpus._sha256(content["corpus"])
            or manifest.get("documents") != expected):
        raise PreprocessingError("Corpus manifest schema/count/hash/document provenance mismatch")
    if [d.doc_id for d in documents] != sorted(d.doc_id for d in documents):
        raise PreprocessingError("Corpus must retain canonical document order")
    if (collection.get("policy_version") != acquisition.POLICY_VERSION or collection.get("status") != "complete"
            or collection.get("text_representation") != "unexpanded-mediawiki-wikitext"
            or collection.get("corpus_sha256") != manifest["corpus_sha256"]
            or collection.get("manifest_sha256") != corpus._sha256(content["manifest"])):
        raise PreprocessingError("Acquisition metadata is incomplete or does not match the snapshot")
    for field in ("selected_records", "collected_records"):
        if type(collection.get(field)) is not int or collection[field] != count:
            raise PreprocessingError(f"Acquisition {field} coverage mismatch")
    if type(collection.get("input_records")) is not int or collection["input_records"] < count:
        raise PreprocessingError("Invalid acquisition input record count")
    records = collection.get("records")
    if not isinstance(records, list) or len(records) != count:
        raise PreprocessingError("Acquisition record coverage mismatch")
    endpoint = collection.get("endpoint", "")
    match = re.fullmatch(r"https://([a-z-]+)\.wikipedia\.org/w/api\.php", endpoint) if isinstance(endpoint, str) else None
    if not match:
        raise PreprocessingError("Acquisition endpoint is not a supported Wikipedia edition")
    by_id = {}
    fingerprints = {key: {"sha256": corpus._sha256(data), "bytes": len(data)} for key, data in content.items()}
    for index, record in enumerate(records, 1):
        if (not isinstance(record, dict) or record.get("status") != "collected"
                or type(record.get("input_line")) is not int or record["input_line"] != index
                or not isinstance(record.get("doc_id"), str) or record["doc_id"] in by_id):
            raise PreprocessingError("Invalid/duplicate acquisition record or input order")
        by_id[record["doc_id"]] = record
        acquisition._requests(corpus._json_bytes(record.get("request")))
    if set(by_id) != {document.doc_id for document in documents}:
        raise PreprocessingError("Acquisition document identities do not cover the corpus")
    for document in documents:
        record = by_id[document.doc_id]
        for field in ("title", "source_url", "revision_id", "retrieved_at", "language", "content_sha256"):
            if record.get(field) != getattr(document, field):
                raise PreprocessingError(f"{document.title}: acquisition {field} mismatch")
        if (document.source != "encyclopedia" or not document.revision_id or not document.revision_id.isascii()
                or not document.revision_id.isdigit() or int(document.revision_id) <= 0
                or document.source_url != f"https://{match[1]}.wikipedia.org/w/index.php?oldid={document.revision_id}"):
            raise PreprocessingError("Require pinned Wikipedia revision source provenance")
        acquisition._positive_id(record.get("page_id"), "page ID")
        instant = acquisition._timestamp(record.get("revision_timestamp"))
        cutoff = collection.get("cutoff")
        satisfies = instant <= acquisition._timestamp(cutoff) if cutoff is not None else None
        if record.get("cutoff_satisfied") is not satisfies or satisfies is False:
            raise PreprocessingError("Acquisition cutoff metadata is inconsistent")
        rights = record.get("license")
        if not isinstance(rights, dict) or any(not isinstance(rights.get(key), str) or not rights[key].strip() for key in ("text", "url")):
            raise PreprocessingError("Missing source license attribution")
        if not isinstance(record.get("attribution_history_url"), str) or not record["attribution_history_url"].startswith(f"https://{match[1]}.wikipedia.org/"):
            raise PreprocessingError("Missing Wikipedia history attribution")
        if cache_dir is not None:
            cache = record.get("cache", {})
            if not isinstance(cache, dict):
                raise PreprocessingError("Invalid cache reference")
            key = cache.get("key")
            if not isinstance(key, str) or not re.fullmatch("[0-9a-f]{64}", key):
                raise PreprocessingError("Invalid cache reference")
            cache_path = Path(cache_dir) / (key + ".json")
            data = corpus._input_bytes(cache_path)
            if corpus._sha256(data) != cache.get("sha256"):
                raise PreprocessingError("Archived cache hash mismatch")
            snapshot = acquisition._json_object(data, str(cache_path))
            request = {"endpoint": endpoint, "params": acquisition._params(record["request"], cutoff)}
            if (snapshot.get("schema_version") != acquisition.POLICY_VERSION or snapshot.get("request") != request
                    or corpus._sha256(corpus._json_bytes(request)) != key
                    or not isinstance(snapshot.get("response_body"), str)
                    or corpus._sha256(snapshot["response_body"].encode("utf-8")) != snapshot.get("response_sha256")):
                raise PreprocessingError("Invalid archived API response/request")
            recovered_metadata = {}
            recovered = acquisition._document(snapshot, record["request"], match[1], cutoff, recovered_metadata)
            if recovered != document or any(record.get(field) != recovered_metadata.get(field) for field in (
                    "page_id", "revision_timestamp", "cutoff_satisfied", "license", "attribution_history_url")):
                raise PreprocessingError("Archived API content/provenance differs from corpus or collection metadata")
            fingerprints["cache:" + key] = {"sha256": corpus._sha256(data), "bytes": len(data)}
    return documents, by_id, fingerprints


class _Cleaner:
    """Small source-only AST visitor; unknown semantics never use strip_code()."""

    def __init__(self, parser, source):
        self.parser, self.source = parser, source
        self.events = []
        self.span = {"start": 0, "end": len(source)}
        self.newlines = [i for i, char in enumerate(source) if char == "\n"]

    def location(self):
        return {**self.span, "start_line": bisect.bisect_left(self.newlines, self.span["start"]) + 1,
                "end_line": bisect.bisect_left(self.newlines, max(self.span["start"], self.span["end"] - 1)) + 1}

    def log(self, kind, raw, action="unprocessed", reason=None):
        self.events.append({"kind": kind, "action": action, "source_block": self.location(),
                            "raw_wikitext": raw, "raw_sha256": corpus._sha256(raw.encode("utf-8")), "reason": reason})

    def unknown(self, kind, node, reason=None):
        self.log(kind, str(node), reason=reason)
        return f" [UNPROCESSED {kind}] "

    def code(self, code):
        return "".join(self.node(node) for node in code.nodes)

    def template(self, node):
        name = " ".join(str(node.name).replace("_", " ").strip().casefold().split())
        params = list(node.params)
        names = [str(p.name).strip() for p in params]
        if len(set(names)) != len(names):
            return self.unknown("template", node, "duplicate parameters")
        if name in _OMIT_TEMPLATES or name.startswith("cite "):
            self.log("citation_or_maintenance_template", str(node), "omitted")
            return " "
        if name == "infobox" or name.startswith("infobox "):
            if any(not p.showkey or not re.fullmatch(r"[A-Za-z0-9_ -]+", str(p.name).strip()) for p in params):
                return self.unknown("infobox", node, "nonliteral/positional field names")
            return "\n" + str(node.name).strip() + "\n" + "\n".join(
                f"{str(p.name).strip()}: {self.code(p.value).strip()}" for p in params if str(p.value).strip()) + "\n"
        positional = {str(p.name).strip(): p.value for p in params if str(p.name).strip().isdigit()}
        options = set(names) - positional.keys()
        if name in {"hlist", "unbulleted list", "ubl", "plainlist", "flatlist"} and not options:
            if list(positional) != [str(i) for i in range(1, len(positional) + 1)]:
                return self.unknown("template", node, "nonsequential list arguments")
            return "; ".join(self.code(value).strip() for value in positional.values())
        if name in {"lang", "native name", "native phrase"} and set(positional) == {"1", "2"} and options <= {"italic", "paren", "label"}:
            self.log("source_language_wrapper", str(node), "explicit_text_only", "language code/display options not rendered")
            return self.code(positional["2"])
        if name in {"convert", "cvt"} and set(positional) == {"1", "2"} and options <= {"abbr", "disp", "adj", "lk"}:
            value, unit = (str(positional[k]).strip() for k in ("1", "2"))
            if _NUMBER.fullmatch(value) and unit in _UNITS:
                self.log("source_quantity", str(node), "source_value_unit_only", "no conversion/rounding/template expansion")
                return value + " " + unit
        return self.unknown("template", node, "no frozen source-only rule")

    def table(self, node):
        tags = node.contents.filter_tags(recursive=True)
        if str(node).count("{|") > 1 or any(str(tag.tag).casefold() == "table" or any(str(a.name).casefold() in {"rowspan", "colspan"} for a in tag.attributes) for tag in tags):
            return self.unknown("table", node, "nested table or spanning cells")
        rows, pending, captions = [], [], []
        for child in node.contents.nodes:
            if isinstance(child, self.parser.nodes.Text) and not str(child).strip():
                continue
            if not isinstance(child, self.parser.nodes.Tag):
                return self.unknown("table", node, "unsupported table-level content")
            tag = str(child.tag).casefold()
            if tag in {"th", "td"}:
                if str(child).lstrip().startswith("|+"):
                    captions.append(str(child.contents).lstrip()[1:].strip())
                else:
                    pending.append(child)
            elif tag == "tr":
                if pending:
                    rows.append(pending)
                    pending = []
                cells = []
                for cell in child.contents.nodes:
                    if isinstance(cell, self.parser.nodes.Text) and not str(cell).strip():
                        continue
                    if not isinstance(cell, self.parser.nodes.Tag) or str(cell.tag).casefold() not in {"th", "td"}:
                        return self.unknown("table", node, "unsupported row structure")
                    cells.append(cell)
                if cells:
                    rows.append(cells)
            else:
                return self.unknown("table", node, "unsupported table element")
        if pending:
            rows.append(pending)
        header = rows.pop(0) if rows and all(str(cell.tag).casefold() == "th" for cell in rows[0]) else []
        if any((header and len(row) != len(header)) or all(str(c.tag).casefold() == "th" for c in row) for row in rows):
            return self.unknown("table", node, "ambiguous headers/column count")
        labels = [self.code(cell.contents).strip() for cell in header]
        if header and (any(not label for label in labels) or len(set(labels)) != len(labels)):
            return self.unknown("table", node, "empty/duplicate column labels")
        output = ["Table caption: " + self.code(self.parser.parse(caption, skip_style_tags=True)) for caption in captions]
        if labels:
            output.append("Table columns: " + " | ".join(labels))
        for row in rows:
            values = [self.code(cell.contents).strip() for cell in row]
            output.append("; ".join(f"{label}: {value}" for label, value in zip(labels, values)) if labels else "Table row: " + " | ".join(values))
        return "\n" + "\n".join(output) + "\n"

    def node(self, node):
        n = self.parser.nodes
        if isinstance(node, n.Text):
            text = str(node)
            if any(marker in text for marker in ("{{", "}}", "[[", "]]", "{|", "|}", "<ref", "<!--")):
                self.log("unparsed_markup", text, "retained_verbatim", "parser recovered as plain text; not claimed rendered")
            return re.sub(r"'{2,5}", "", re.sub(r"(?m)^[*#;:]+[ \t]*", "", text))
        if isinstance(node, n.Wikilink):
            title = str(node.title)
            if title.casefold().startswith(("file:", "image:")):
                return self.unknown("media_link", node, "caption/options not interpreted")
            if title.casefold().startswith("category:"):
                self.log("category_link", str(node), "omitted")
                return " "
            return self.code(node.text if node.text is not None else node.title)
        if isinstance(node, n.ExternalLink):
            return self.code(node.title) if node.title is not None else str(node.url)
        if isinstance(node, n.Template):
            return self.template(node)
        if isinstance(node, n.Comment):
            self.log("comment", str(node), "omitted")
            return " "
        if isinstance(node, n.HTMLEntity):
            try:
                value = node.normalize()
                if "\x00" in value:
                    raise ValueError("NUL entity")
                value.encode("utf-8")
                return value
            except (ValueError, OverflowError, KeyError):
                return self.unknown("html_entity", node)
        if isinstance(node, n.Tag):
            tag = str(node.tag).casefold()
            if tag in {"ref", "references"}:
                self.log("reference", str(node), "omitted", "may contain factual notes; retained in diagnostics")
                return " "
            if tag == "table":
                return self.table(node)
            if tag in {"nowiki", "code", "pre"}:
                return str(node.contents)
            if tag in {"del", "s", "strike", "ins"}:
                return f" [{'inserted' if tag == 'ins' else 'deleted'}: {self.code(node.contents)}] "
            if tag in {"sup", "sub"}:
                return ("^{" if tag == "sup" else "_{") + self.code(node.contents) + "}"
            if tag in {"br", "hr"}:
                return "\n"
            if tag in {"b", "i", "em", "strong", "span", "small", "big", "u", "abbr"}:
                return self.code(node.contents)
            if tag in {"div", "p", "center", "blockquote", "li", "ul", "ol", "dl", "dt", "dd"}:
                return "\n" + self.code(node.contents) + "\n"
            return self.unknown("tag", node, "unsupported extension/HTML semantics")
        return self.unknown("syntax", node, "unresolved argument/other syntax")


def clean_sections(source):
    """Return sections and loss diagnostics, with enclosing raw-source blocks."""
    parser = _parser()
    parsed = parser.parse(source, skip_style_tags=True)
    if str(parsed) != source:
        raise PreprocessingError("Parser did not preserve source round-trip")
    cleaner = _Cleaner(parser, source)
    sections, stack, offset = [], [], 0
    section = {"section_path": [], "heading_level": 0, "heading_span": None, "parts": []}
    sections.append(section)
    for node in parsed.nodes:
        end = offset + len(str(node))
        cleaner.span = {"start": offset, "end": end}
        if isinstance(node, parser.nodes.Heading):
            title = " ".join(cleaner.code(node.title).split())
            while stack and stack[-1][0] >= node.level:
                stack.pop()
            stack.append((node.level, title))
            section = {"section_path": [value for _, value in stack], "heading_level": node.level,
                       "heading_span": cleaner.location(), "parts": []}
            sections.append(section)
        else:
            rendered = cleaner.node(node)
            section["parts"].append((rendered, cleaner.location()))
        offset = end
    for section in sections:
        section["text"] = "".join(text for text, _ in section["parts"])
    return sections, cleaner.events


def _configuration(chunk_words, overlap_words):
    if (type(chunk_words) is not int or type(overlap_words) is not int or chunk_words < 1
            or not 0 <= overlap_words < chunk_words):
        raise PreprocessingError("Require chunk_words > 0 and 0 <= overlap_words < chunk_words")
    return {"version": VERSION, "parser": PARSER_VERSION, "chunk_words": chunk_words, "overlap_words": overlap_words}


def make_passages(document, metadata, *, chunk_words=200, overlap_words=40):
    """Fixed whitespace-word windows; no window crosses a section boundary."""
    config = _configuration(chunk_words, overlap_words)
    sections, events = clean_sections(document.text)
    passages = []
    for section_index, section in enumerate(sections):
        words = list(re.finditer(r"\S+", section["text"]))
        boundaries, cursor = [], 0
        for text, span in section["parts"]:
            boundaries.append((cursor, cursor + len(text), span))
            cursor += len(text)
        for start in range(0, len(words), chunk_words - overlap_words):
            end = min(start + chunk_words, len(words))
            text = " ".join(match.group() for match in words[start:end])
            spans = [span for low, high, span in boundaries if low < words[end - 1].end() and high > words[start].start()]
            digest = corpus._sha256(text.encode("utf-8"))
            identity = {"doc_id": document.doc_id, "section_index": section_index, "word_start": start,
                        "word_end": end, "content_sha256": digest, "configuration": config}
            passages.append({**identity, "passage_id": "passage-" + corpus._sha256(corpus._json_bytes(identity)),
                             "article_title": document.title, "source_url": document.source_url,
                             "revision_id": document.revision_id, "revision_timestamp": metadata["revision_timestamp"],
                             "retrieved_at": document.retrieved_at, "language": document.language,
                             "section_path": section["section_path"], "heading_level": section["heading_level"],
                             "text": text, "source_spans": spans, "alignment": "enclosing_top_level_nodes_not_exact_cleaned_spans"})
            if end == len(words):
                break
    diagnostics = {"doc_id": document.doc_id, "article_title": document.title, "events": events,
                   "event_counts": dict(sorted(Counter(event["kind"] for event in events).items())),
                   "unprocessed_events": sum(event["action"] in {"unprocessed", "retained_verbatim"} for event in events),
                   "sections": [{key: value for key, value in section.items() if key not in {"text", "parts"}} for section in sections]}
    return passages, diagnostics


def preprocess_corpus(*, corpus_path, output_path, manifest_path=None, collection_path=None,
                      cache_dir=None, chunk_words=200, overlap_words=40):
    _configuration(chunk_words, overlap_words)
    _parser()
    output = Path(output_path)
    targets = [output, Path(str(output) + ".diagnostics.jsonl"), Path(str(output) + ".manifest.json")]
    for path in targets:
        corpus._check_location(path)
        if path.exists() or path.is_symlink():
            raise PreprocessingError(f"Refusing to overwrite {path}")
    if not output.parent.is_dir():
        raise PreprocessingError("Output parent directory must already exist")
    documents, metadata, fingerprints = load_snapshot(corpus_path, manifest_path, collection_path, cache_dir)
    passages, diagnostics = [], []
    for document in documents:
        rows, diagnostic = make_passages(document, metadata[document.doc_id], chunk_words=chunk_words, overlap_words=overlap_words)
        passages.extend(rows)
        diagnostics.append(diagnostic)
    data = b"".join(corpus._json_bytes(row) for row in passages)
    diagnostic_data = b"".join(corpus._json_bytes(row) for row in diagnostics)
    # Verify ordinary concurrent changes, including supplied caches, before publication.
    if load_snapshot(corpus_path, manifest_path, collection_path, cache_dir)[2] != fingerprints:
        raise PreprocessingError("Input snapshot changed during preprocessing")
    manifest = {"schema_version": "evidence-passages-v1", "status": "complete", "preprocessing_version": VERSION,
                "document_count": len(documents), "passage_count": len(passages), "inputs": fingerprints,
                "configuration": {"chunk_words": chunk_words, "overlap_words": overlap_words,
                                  "word_definition": "Unicode non-whitespace runs", "section_local": True},
                "code_sha256": corpus._sha256(Path(__file__).read_bytes()), "mwparserfromhell": PARSER_VERSION,
                "passages_sha256": corpus._sha256(data), "diagnostics_sha256": corpus._sha256(diagnostic_data),
                "cache_validation": "verified" if cache_dir is not None else "not_supplied_not_verified",
                "unprocessed_events": sum(row["unprocessed_events"] for row in diagnostics),
                "document_provenance": [{"doc_id": d.doc_id, "page_id": metadata[d.doc_id]["page_id"],
                                         "license": metadata[d.doc_id]["license"],
                                         "attribution_history_url": metadata[d.doc_id]["attribution_history_url"],
                                         "content_sha256": d.content_sha256} for d in documents]}
    corpus._publish(targets[0], data)
    corpus._publish(targets[1], diagnostic_data)
    corpus._publish(targets[2], corpus._json_bytes(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Default: <corpus>.manifest.json")
    parser.add_argument("--collection", type=Path, help="Default: <corpus>.collection.json")
    parser.add_argument("--cache-dir", type=Path, help="Optional read-only verification of referenced API caches")
    parser.add_argument("--chunk-words", type=int, default=200)
    parser.add_argument("--overlap-words", type=int, default=40)
    args = parser.parse_args(argv)
    try:
        result = preprocess_corpus(corpus_path=args.corpus, output_path=args.output, manifest_path=args.manifest,
                                   collection_path=args.collection, cache_dir=args.cache_dir,
                                   chunk_words=args.chunk_words, overlap_words=args.overlap_words)
    except (PreprocessingError, corpus.CorpusError, acquisition.CollectionError, OSError, UnicodeError) as error:
        parser.exit(2, f"Offline preprocessing failed: {error}\n")
    print(f"Documents: {result['document_count']}; passages: {result['passage_count']}; unprocessed events: {result['unprocessed_events']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
