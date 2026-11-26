"""Acquire fixed Wikipedia revision source text; replay snapshots offline by default.

Text is unexpanded wikitext, not a current extract or rendered historical HTML.
RAG-01 owns document validation, IDs, hashing, serialization and corpus manifests.
Additional acquisition metadata stays in a separate collection report/cache.
"""

import argparse
import json
import math
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from . import evidence_corpus as corpus


POLICY_VERSION = "wikipedia-revision-source-v1"
MAX_RESPONSE_BYTES = 20 * 1024 * 1024


class CollectionError(ValueError):
    """Invalid inputs, cached responses, or unavailable/ambiguous article content."""


class NetworkFailure(CollectionError):
    """Stop further live requests after a network/server/rate-limit failure."""

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details or {}


class TerminalArticleOutcome(CollectionError):
    """Validated evidence absence/exclusion; never a transport or integrity failure.

    Only article_outcome() raises this. The standalone collector retains its
    existing all-or-nothing behavior and does not use this opt-in classifier.
    """

    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def article_outcome(snapshot, request, language, cutoff):
    """Classify a validated revision response, without guessing from error text.

    Callers must first validate the immutable envelope using _snapshot(). Missing
    fields are errors, except absent revisions (MediaWiki's no-revision outcome).
    Current redirects are not accepted as historical identity mappings.
    """
    payload = _json_object(snapshot["response_body"], "API response")
    if "error" in payload or "warnings" in payload:
        raise CollectionError("API errors/warnings are not terminal evidence outcomes")
    continuation = payload.get("continue")
    if "continue" in payload:
        # This request paginates revisions only. Tokens are opaque, not dates or
        # evidence of absence. Do not follow them or change standalone collection.
        if (not isinstance(continuation, dict) or set(continuation) != {"continue", "rvcontinue"}
                or any(not isinstance(v, str) or not v.strip()
                    or any(ord(c) < 32 or ord(c) == 127 for c in v) for v in continuation.values())):
            raise CollectionError("Malformed revision continuation metadata")
        try:
            for token in continuation.values():
                token.encode("utf-8")
        except UnicodeError as error:
            raise CollectionError("Malformed revision continuation token") from error
    try:
        query = payload["query"]
        pages = query["pages"]
        if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
            raise CollectionError("Require exactly one page outcome")
        page = pages[0]
        for flag in ("missing", "invalid", "redirect"):
            if flag in page and page[flag] is not True:
                raise CollectionError(f"Malformed formatversion-2 {flag} flag")
        if "invalid" in page:
            raise CollectionError("Invalid page request is not verified evidence absence")
        for key in ("normalized", "redirects"):
            chain = query.get(key, [])
            if not isinstance(chain, list) or any(not isinstance(m, dict)
                    or any(not isinstance(m.get(k), str) or not m[k].strip() for k in ("from", "to")) for m in chain):
                raise CollectionError("Malformed page mapping")
        if query.get("redirects") or "redirect" in page:
            raise CollectionError("Historical identity unresolved; no current redirect substitution")
        rights = query["rightsinfo"]
        if not isinstance(rights, dict) or any(not isinstance(rights.get(k), str)
                or not rights[k].strip() for k in ("text", "url")):
            raise CollectionError("Missing site-wide license attribution")
        if "page_id" in request:
            if _positive_id(page["pageid"], "page ID") != request["page_id"]:
                raise CollectionError("Page outcome identity mismatch")
        else:
            title = request["title"]
            for mapping in query.get("normalized", []):
                if title == mapping["from"]:
                    title = mapping["to"]
            if page.get("title") != title:
                raise CollectionError("Page outcome title mismatch")
        if "missing" in page:
            if continuation is not None:
                raise CollectionError("Missing page with unresolved revision continuation is incomplete")
            if ("revisions" in page or "invalid" in page or "pageprops" in page
                    or ("ns" in page and (type(page["ns"]) is not int or page["ns"] != 0))):
                raise CollectionError("Contradictory missing-page metadata")
            if "title" in page:
                corpus._string(page["title"], "title", "Missing page")
            if "title" in request and "ns" not in page:
                raise CollectionError("Missing title outcome requires a namespace")
            raise TerminalArticleOutcome("unavailable", "missing_page")
        _positive_id(page["pageid"], "page ID")
        title = corpus._string(page["title"], "title", "API")
        if corpus._BENCHMARK_MARKER.search(title.casefold()):
            raise CollectionError("Benchmark/thesis titles are excluded")
        if type(page["ns"]) is not int or page["ns"] != 0 or page["pagelanguage"] != language:
            raise CollectionError("Invalid article namespace/language")
        properties = page.get("pageprops", {})
        if not isinstance(properties, dict) or any(not isinstance(v, str) for v in properties.values()):
            raise CollectionError("Malformed page properties")
        revisions = page.get("revisions", [])
        if not isinstance(revisions, list) or len(revisions) > 1:
            raise CollectionError("Malformed revision result")
        if not revisions:
            if continuation is not None:
                raise CollectionError("Empty revision result with unresolved continuation is incomplete")
            raise TerminalArticleOutcome("unavailable", "no_eligible_revision")
        revision = revisions[0]
        _positive_id(revision["revid"], "revision ID")
        if cutoff is not None and _timestamp(revision["timestamp"]) > _timestamp(cutoff):
            raise CollectionError("Returned revision is newer than cutoff")
        _timestamp(revision["timestamp"])
        slot = revision["slots"]["main"]
        if not isinstance(slot, dict) or any(not isinstance(slot.get(k), str)
                or not slot[k].strip() for k in ("contentmodel", "contentformat")):
            raise CollectionError("Incomplete revision content metadata")
        text = slot["content"]
        if not isinstance(text, str):
            raise CollectionError("Revision source must be a string")
        text.encode("utf-8")
        if "\x00" in text:
            raise CollectionError("Invalid NUL in revision source")
        if slot["contentmodel"] != "wikitext" or slot["contentformat"] != "text/x-wiki":
            raise TerminalArticleOutcome("excluded", "unsupported_source")
        if not text.strip():
            raise TerminalArticleOutcome("excluded", "empty_source")
        if "disambiguation" in properties or "__DISAMBIG__" in text or text.lstrip().startswith("#"):
            raise TerminalArticleOutcome("excluded", "ambiguous_or_directive_source")
    except (KeyError, TypeError, AttributeError, IndexError, UnicodeError) as error:
        raise CollectionError(f"Malformed or incomplete article outcome: {error}") from error
    return _document(snapshot, request, language, cutoff, {})


def _json_object(content: str | bytes, location: str) -> dict:
    try:
        value = json.loads(content, object_pairs_hook=corpus._unique_keys,
                           parse_constant=corpus._invalid_constant)
    except (ValueError, UnicodeError) as error:
        raise CollectionError(f"{location}: malformed JSON: {error}") from error
    if not isinstance(value, dict):
        raise CollectionError(f"{location}: expected a JSON object")
    return value


def _timestamp(value: str) -> datetime:
    if (not isinstance(value, str) or not corpus._TIMESTAMP.fullmatch(value)
            or value.endswith("-00:00")):
        raise CollectionError(f"Invalid timestamp with known timezone: {value!r}")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CollectionError(f"Invalid timestamp: {value!r}") from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _requests(content: bytes) -> list[dict]:
    try:
        lines = content.decode("utf-8").split("\n")
    except UnicodeError as error:
        raise CollectionError("Article list must be UTF-8") from error
    if lines[-1] == "":
        lines.pop()
    requests, seen = [], set()
    for index, line in enumerate(lines, 1):
        value = _json_object(line, f"Article list:{index}")
        if set(value) not in ({"title"}, {"page_id"}):
            raise CollectionError(f"Article list:{index}: supply only title or page_id; benchmark answer fields are prohibited")
        key, item = next(iter(value.items()))
        if key == "title":
            if (not isinstance(item, str) or not item.strip() or item != item.strip()
                    or any(ord(c) < 32 or c in "|#" for c in item)):
                raise CollectionError(f"Article list:{index}: invalid single article title")
            try:
                item.encode("utf-8")
            except UnicodeError as error:
                raise CollectionError(f"Article list:{index}: invalid Unicode title") from error
            if corpus._BENCHMARK_MARKER.search(item.casefold()):
                raise CollectionError(f"Article list:{index}: identifiable benchmark/thesis titles are excluded")
        elif type(item) is not int or item <= 0:
            raise CollectionError(f"Article list:{index}: page_id must be a positive integer")
        if (key, item) in seen:
            raise CollectionError(f"Article list:{index}: duplicate requested identity")
        seen.add((key, item))
        requests.append(value)
    if not requests:
        raise CollectionError("Article list must not be empty")
    return requests


def _params(request: dict, cutoff: str | None) -> dict:
    params = {
        "action": "query", "format": "json", "formatversion": "2", "maxlag": "5",
        "prop": "info|revisions|pageprops", "ppprop": "disambiguation", "redirects": "1",
        "rvprop": "ids|timestamp|content|contentmodel", "rvslots": "main",
        "rvlimit": "1", "rvdir": "older", "meta": "siteinfo", "siprop": "rightsinfo",
    }
    params["titles" if "title" in request else "pageids"] = str(next(iter(request.values())))
    if cutoff is not None:
        params["rvstart"] = cutoff
    return params


def _fetch(endpoint: str, params: dict, user_agent: str, delay: float) -> dict:
    """One serial GET; stop on failures rather than retrying against server load."""
    time.sleep(delay)
    url = endpoint + "?" + urlencode(sorted(params.items()))
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            if response.geturl() != url:
                raise NetworkFailure("HTTP redirect refused; no cross-endpoint source substitution")
            body = response.read(MAX_RESPONSE_BYTES + 1)
            headers = {key: response.headers.get(key) for key in (
                "Content-Type", "Date", "ETag", "Last-Modified", "Retry-After") if response.headers.get(key) is not None}
            if len(body) > MAX_RESPONSE_BYTES:
                raise NetworkFailure("API response exceeded the 20 MiB limit; no truncated content accepted")
    except HTTPError as error:
        details = {"http_status": error.code, "retry_after": (error.headers or {}).get("Retry-After")}
        error.close()
        raise NetworkFailure(f"HTTP {error.code}; stop collection and respect Retry-After before another live run", details=details) from error
    except (URLError, OSError) as error:
        raise NetworkFailure(f"Network failure: {error}") from error
    try:
        text = body.decode("utf-8")
    except UnicodeError as error:
        raise NetworkFailure("API returned invalid UTF-8") from error
    return {"schema_version": POLICY_VERSION, "request": {"endpoint": endpoint, "params": params},
            "retrieved_at": _utc_now(), "user_agent": user_agent, "headers": headers,
            "response_body": text, "response_sha256": corpus._sha256(body)}


def _snapshot(endpoint, params, cache_dir, *, allow_network, user_agent, delay):
    request = {"endpoint": endpoint, "params": params}
    key = corpus._sha256(corpus._json_bytes(request))
    path = cache_dir / f"{key}.json"
    if path.exists() or path.is_symlink():
        cache_bytes = path.read_bytes()
        snapshot = _json_object(cache_bytes, str(path))
    elif not allow_network:
        raise CollectionError(f"Offline cache miss: {key}; no live request made")
    else:
        snapshot = _fetch(endpoint, params, user_agent, delay)
        try:
            payload = _json_object(snapshot["response_body"], "API response")
        except CollectionError:
            payload = {}  # Cache malformed bodies for reproducible diagnosis, never as evidence.
        error = payload.get("error")
        if isinstance(error, dict) and error.get("code") in {"maxlag", "ratelimited"}:
            raise NetworkFailure(f"API {error['code']}; stop collection before further requests", details=snapshot)
        cache_bytes = corpus._json_bytes(snapshot)
        corpus._publish(path, cache_bytes)
    if (snapshot.get("schema_version") != POLICY_VERSION or snapshot.get("request") != request
            or not isinstance(snapshot.get("response_body"), str)
            or not isinstance(snapshot.get("headers"), dict)
            or not isinstance(snapshot.get("user_agent"), str) or not snapshot["user_agent"].strip()):
        raise CollectionError(f"{path}: invalid cache envelope/request identity")
    _timestamp(snapshot.get("retrieved_at"))
    try:
        response_hash = corpus._sha256(snapshot["response_body"].encode("utf-8"))
    except UnicodeError as error:
        raise CollectionError(f"{path}: cached response has invalid Unicode") from error
    if response_hash != snapshot.get("response_sha256"):
        raise CollectionError(f"{path}: response SHA256 mismatch; no automatic cache refresh")
    return snapshot, {"key": key, "sha256": corpus._sha256(cache_bytes)}


def _positive_id(value, field):
    if type(value) is not int or value <= 0:
        raise CollectionError(f"Malformed API {field}: require a positive integer")
    return value


def _document(snapshot, request, language, cutoff, details):
    payload = _json_object(snapshot["response_body"], "API response")
    if "error" in payload or "warnings" in payload:
        raise CollectionError(f"API error/warning: {json.dumps(payload.get('error', payload.get('warnings')), ensure_ascii=False)}")
    try:
        query = payload["query"]
        pages = query["pages"]
        if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
            raise CollectionError("API must return exactly one page")
        page = pages[0]
        if "missing" in page or "invalid" in page:
            raise CollectionError("Missing/invalid Wikipedia page; no alternative article selected")
        page_id = _positive_id(page["pageid"], "page ID")
        title = corpus._string(page["title"], "title", "API")
        if corpus._BENCHMARK_MARKER.search(title.casefold()):
            raise CollectionError("Identifiable benchmark/thesis article titles are excluded")
        if type(page["ns"]) is not int or page["ns"] != 0:
            raise CollectionError("Only main-namespace articles are eligible")
        mappings = {key: query.get(key, []) for key in ("normalized", "redirects")}
        for chain in mappings.values():
            if not isinstance(chain, list) or any(not isinstance(item, dict) or not isinstance(item.get("from"), str)
                                                 or not isinstance(item.get("to"), str) for item in chain):
                raise CollectionError("Malformed normalization/redirect metadata")
        details.update(page_id=page_id, title=title, **mappings)
        if "title" in request:
            resolved = request["title"]
            for chain in mappings.values():
                for mapping in chain:
                    if resolved == mapping["from"]:
                        resolved = mapping["to"]
            if resolved != title:
                raise CollectionError("API returned an unexplained article identity substitution")
        elif not mappings["redirects"] and page_id != request["page_id"]:
            raise CollectionError("API returned another page ID without a redirect")
        revisions = page.get("revisions", [])
        if not isinstance(revisions, list) or len(revisions) != 1:
            raise CollectionError("No unique available revision at/before cutoff; no current-revision fallback")
        revision = revisions[0]
        revid = _positive_id(revision["revid"], "revision ID")
        timestamp = revision["timestamp"]
        revision_time = _timestamp(timestamp)
        satisfies = revision_time <= _timestamp(cutoff) if cutoff is not None else None
        details.update(revision_id=str(revid), revision_timestamp=timestamp, cutoff_satisfied=satisfies)
        if satisfies is False:
            raise CollectionError("Returned revision is newer than the explicit cutoff")
        if cutoff is not None and mappings["redirects"]:
            raise CollectionError("Historical run refuses today's redirect mapping; independently resolve the historical page identity")
        if "redirect" in page:
            raise CollectionError("Unresolved redirect page")
        properties = page.get("pageprops", {})
        if not isinstance(properties, dict):
            raise CollectionError("Malformed page properties")
        details["disambiguation_check"] = "current-pageprops-plus-source-magic-word; not a complete historical classifier"
        if "disambiguation" in properties:
            raise CollectionError("Disambiguation page; no target chosen automatically")
        slot = revision["slots"]["main"]
        if slot.get("contentmodel") != "wikitext" or slot.get("contentformat") != "text/x-wiki":
            raise CollectionError("Unsupported or missing revision content model/format")
        text = slot["content"]
        if not isinstance(text, str) or not text.strip():
            raise CollectionError("Empty, hidden, or unavailable revision source")
        if text.lstrip().startswith("#") or "__DISAMBIG__" in text:
            raise CollectionError("Possible redirect/directive/disambiguation source; manual resolution required")
        rights = query["rightsinfo"]
        if not isinstance(rights, dict) or not all(isinstance(rights.get(k), str) and rights[k].strip() for k in ("url", "text")):
            raise CollectionError("Missing site-wide license attribution")
        source_url = f"https://{language}.wikipedia.org/w/index.php?oldid={revid}"
        document = corpus.validate_document({
            "title": title, "text": text, "source": "encyclopedia", "source_url": source_url,
            "revision_id": str(revid), "retrieved_at": snapshot["retrieved_at"],
            "language": page["pagelanguage"],
        }, location="Wikipedia revision")
        details.update(doc_id=document.doc_id, source_url=source_url, language=document.language,
                       content_sha256=document.content_sha256, license=rights,
                       license_scope="site-wide at acquisition; not a per-revision rights audit",
                       attribution_history_url=f"https://{language}.wikipedia.org/w/index.php?title={quote(title, safe='')}&action=history")
        return document
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise CollectionError(f"Malformed or incomplete API response: {error}") from error


def collect_wikipedia(*, input_path, output_path, cache_dir, language="en", cutoff=None,
                      max_articles=10, allow_network=False, user_agent=None, request_delay=1.0):
    """Collect a fixed prefix or replay it offline; incomplete runs publish no corpus.

    A collection report is always retained for ordinary per-page/network failures.
    Successful responses are immutable cache checkpoints. A fresh live request is
    not byte-reproducible; replaying the same cache and selection is reproducible.
    """
    if type(max_articles) is not int or max_articles < 1:
        raise CollectionError("max_articles must be a positive integer")
    if not isinstance(request_delay, (int, float)) or not math.isfinite(request_delay) or request_delay < 1:
        raise CollectionError("request_delay must be finite and at least one second")
    if not isinstance(language, str) or not re.fullmatch(r"[a-z]{2,8}(?:-[a-z0-9]{1,8})*", language):
        raise CollectionError("language must be a lowercase Wikipedia language-edition code")
    if cutoff is not None:
        instant = _timestamp(cutoff)
        if instant.microsecond:
            raise CollectionError("cutoff must use whole-second precision; no rounding or truncation")
        cutoff = instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if allow_network and (not isinstance(user_agent, str) or not user_agent.isascii()
                          or any(ord(c) < 32 for c in user_agent)
                          or not re.match(r"[A-Za-z][A-Za-z0-9_.-]*/[^ ]+ \(.+\)", user_agent)
                          or not any(contact in user_agent for contact in ("@", "https://", "http://", "User:"))):
        raise CollectionError("Live access requires an informative project/version User-Agent with real contact information")
    input_path, output_path, cache_dir = map(Path, (input_path, output_path, cache_dir))
    report_path = Path(str(output_path) + ".collection.json")
    for path in (output_path, Path(str(output_path) + ".manifest.json"), report_path):
        corpus._check_location(path)
        if path.exists() or path.is_symlink():
            raise CollectionError(f"Refusing to overwrite {path}")
    if not output_path.parent.is_dir():
        raise CollectionError("Output parent directory must already exist")
    corpus._check_location(cache_dir)
    content = corpus._input_bytes(input_path)
    requests = _requests(content)
    selected = requests[:max_articles]
    if not allow_network and not cache_dir.is_dir():
        raise CollectionError("Offline cache directory does not exist; no live request made")
    cache_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    report = {"policy_version": POLICY_VERSION, "status": "incomplete", "input_sha256": corpus._sha256(content),
              "input_records": len(requests), "selected_records": len(selected), "max_articles": max_articles,
              "endpoint": endpoint, "cutoff": cutoff, "text_representation": "unexpanded-mediawiki-wikitext",
              "request_policy": {"serial": True, "minimum_delay_seconds": request_delay,
                                 "maxlag": 5, "timeout_seconds": 30, "automatic_retries": 0},
              "source_sha256": corpus._sha256(Path(__file__).read_bytes()), "records": []}
    documents, stopped = [], False
    for index, request in enumerate(selected, 1):
        record = {"input_line": index, "request": request, "status": "not_attempted"}
        report["records"].append(record)
        if stopped:
            continue
        try:
            snapshot, cache = _snapshot(endpoint, _params(request, cutoff), cache_dir,
                                        allow_network=allow_network, user_agent=user_agent, delay=request_delay)
            record.update(cache=cache, retrieved_at=snapshot["retrieved_at"], user_agent=snapshot["user_agent"])
            documents.append(_document(snapshot, request, language, cutoff, record))
            record["status"] = "collected"
        except (CollectionError, corpus.CorpusError, OSError) as error:
            record.update(status="failed", error=str(error))
            if isinstance(error, NetworkFailure):
                record["network_details"] = error.details
                stopped = True
        except KeyboardInterrupt:
            record.update(status="interrupted", error="KeyboardInterrupt")
            report["status"] = "interrupted"
            stopped = True
    report["collected_records"] = len(documents)
    if len(documents) == len(selected) and report["status"] != "interrupted":
        temporary = None
        try:
            if corpus._input_bytes(input_path) != content:
                raise CollectionError("Article input changed during collection")
            with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".jsonl", delete=False) as stream:
                temporary = Path(stream.name)
                for document in documents:
                    stream.write(corpus.serialize_document(document))
            manifest = corpus.ingest_corpus(temporary, output_path)
            report.update(status="complete", corpus_sha256=manifest["corpus_sha256"],
                          manifest_sha256=corpus._sha256(Path(str(output_path) + ".manifest.json").read_bytes()))
        except (CollectionError, corpus.CorpusError, OSError) as error:
            report["error"] = str(error)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    corpus._publish(report_path, corpus._json_bytes(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Explicit title/page_id JSONL; no answer labels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--cutoff", help="Inclusive revision cutoff with explicit timezone")
    parser.add_argument("--max-articles", type=int, default=10)
    parser.add_argument("--allow-network", action="store_true", help="Explicitly enable live cache misses; default offline")
    parser.add_argument("--user-agent", help="Project/version (real contact URL/email/wiki user)")
    parser.add_argument("--request-delay", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        report = collect_wikipedia(input_path=args.input, output_path=args.output, cache_dir=args.cache_dir,
                                   language=args.language, cutoff=args.cutoff, max_articles=args.max_articles,
                                   allow_network=args.allow_network, user_agent=args.user_agent, request_delay=args.request_delay)
    except (CollectionError, corpus.CorpusError, OSError) as error:
        parser.exit(2, f"Wikipedia collection failed: {error}\n")
    print(f"Status: {report['status']}; collected {report['collected_records']}/{report['selected_records']}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
