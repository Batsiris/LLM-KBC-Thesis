"""Wikipedia metadata discovery, never evidence text or historical entity resolution.

Offline by default. Reuse the collector's immutable HTTP cache and no-clobber
publisher, and BM25's gold-independent identities and versioned lexical queries.
"""

import argparse
import math
import re
from pathlib import Path

from . import bm25_retrieval as retrieval
from . import evidence_corpus as corpus
from . import wikipedia_collect as acquisition


POLICY_VERSION = "wikipedia-article-discovery-v1"
ENDPOINT = "https://en.wikipedia.org/w/api.php"
HISTORICAL_CUTOFF = "2026-07-01T00:00:00Z"
SEARCH_LIMIT = 10
CANDIDATE_LIMIT = 3
SEARCH_SCOPE = "current-at-api-acquisition-not-historical"
TIMESTAMP_POLICY = "cached-api-acquisition-times-no-new-replay-time"
PILOT_TRAIN_SHA256 = "ad37cd30d1ff4b9f1ef2579b25e64093b202c40da11e8c412e13386f1e5d332e"


class DiscoveryError(ValueError):
    """Invalid or incomplete discovery cannot publish a completed selection."""


def _params(**kwargs):
    return {"action": "query", "format": "json", "formatversion": "2", "maxlag": "5", **kwargs}


def _request(record, stage, params, cache_dir, *, allow_network, user_agent, request_delay):
    """Preserve each request/reference before parsing its immutable response."""
    request = {"endpoint": ENDPOINT, "params": params}
    reference = {"stage": stage, "request": request, "status": "pending",
                 "cache": {"key": corpus._sha256(corpus._json_bytes(request))}}
    record["api_responses"].append(reference)
    try:
        snapshot, cache = acquisition._snapshot(ENDPOINT, params, cache_dir,
            allow_network=allow_network, user_agent=user_agent, delay=request_delay)
        reference.update(cache=cache, discovery_timestamp=snapshot["retrieved_at"],
                         response_sha256=snapshot["response_sha256"], status="received")
        payload = acquisition._json_object(snapshot["response_body"], f"Discovery {stage}")
        if "error" in payload or "warnings" in payload:
            raise DiscoveryError(f"Discovery API error/warning: {payload.get('error', payload.get('warnings'))}")
        if not isinstance(payload.get("query"), dict):
            raise DiscoveryError(f"Discovery {stage}: missing query object")
        return payload["query"], snapshot["retrieved_at"]
    except (ValueError, OSError, KeyboardInterrupt) as error:
        reference.update(status="failed", error=str(error) or type(error).__name__)
        if isinstance(error, acquisition.NetworkFailure):
            # HTTP failures have status/Retry-After only; API throttling may also
            # supply its original response envelope, which _snapshot does not cache.
            reference["network_details"] = error.details
        raise


def _mappings(query, key):
    rows = query.get(key, [])
    if (not isinstance(rows, list) or any(not isinstance(row, dict)
            or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("from", "to")) for row in rows)):
        raise DiscoveryError(f"Malformed {key} mappings")
    return rows


def _eligible(page, diagnostics, stage):
    if not isinstance(page, dict):
        raise DiscoveryError(f"{stage}: page must be an object")
    # formatversion=2 encodes present page flags as true, never false, 1 or "".
    flags = [flag for flag in ("missing", "invalid", "redirect") if flag in page]
    if any(page[flag] is not True for flag in flags) or len(flags) > 1:
        raise DiscoveryError(f"{stage}: malformed or conflicting page flags")
    props = page.get("pageprops", {})
    if not isinstance(props, dict) or ("disambiguation" in props and not isinstance(props["disambiguation"], str)):
        raise DiscoveryError(f"{stage}: invalid page properties")
    if "pageid" in page:
        acquisition._positive_id(page["pageid"], "page ID")
    if "ns" in page and type(page["ns"]) is not int:
        raise DiscoveryError(f"{stage}: invalid namespace")
    if "title" in page:
        corpus._string(page["title"], "title", stage)
    if "invalid" in page:
        corpus._string(page.get("title"), "title", stage)
        corpus._string(page.get("invalidreason"), "invalidreason", stage)
    elif "missing" in page and stage == "search_metadata":
        # A requested page ID deleted since search has no known title/namespace.
        acquisition._positive_id(page.get("pageid"), "missing page ID")
    else:
        corpus._string(page.get("title"), "title", stage)
        if type(page.get("ns")) is not int:
            raise DiscoveryError(f"{stage}: missing or invalid namespace")
    if "missing" in page or "invalid" in page:
        diagnostics.append({"stage": stage, "reason": "missing_or_invalid_page", "title": page.get("title")})
        return False
    acquisition._positive_id(page.get("pageid"), "page ID")
    title = corpus._string(page.get("title"), "title", stage)
    if page["ns"] != 0:
        reason = "non_main_namespace"
    elif "redirect" in page:
        reason = "unresolved_current_redirect"
    elif "disambiguation" in page.get("pageprops", {}):
        reason = "disambiguation_page"
    else:
        try:
            acquisition._requests(corpus._json_bytes({"title": title}))
        except acquisition.CollectionError:
            reason = "excluded_or_invalid_article_title"
        else:
            return True
    diagnostics.append({"stage": stage, "reason": reason, "page_id": page["pageid"], "title": title})
    return False


def _candidate(page, method, rank, timestamp, metadata_timestamp):
    return {"page_id": page["pageid"], "title": page["title"], "acquisition_method": method,
            "search_rank": rank, "discovery_timestamp": timestamp, "metadata_timestamp": metadata_timestamp,
            "source_url": f"https://en.wikipedia.org/?curid={page['pageid']}",
            "entity_identity_verified": False, "historical_mapping_verified": False}


def _discover_record(record, cache_dir, *, candidate_limit=CANDIDATE_LIMIT, search_limit=SEARCH_LIMIT, **options):
    diagnostics = record["diagnostics"]
    subject = record["SubjectEntity"]
    selected = record["candidates"]
    try:
        acquisition._requests(corpus._json_bytes({"title": subject}))
    except acquisition.CollectionError as error:
        diagnostics.append({"stage": "direct", "reason": "invalid_direct_title", "detail": str(error)})
    else:
        direct, stamp = _request(record, "direct", _params(titles=subject, redirects="1",
            prop="info|pageprops", ppprop="disambiguation"), cache_dir, **options)
        pages = direct.get("pages")
        if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
            raise DiscoveryError("Direct lookup must return exactly one page object")
        normalized, redirects = _mappings(direct, "normalized"), _mappings(direct, "redirects")
        eligible = _eligible(pages[0], diagnostics, "direct")
        if redirects:
            diagnostics.append({"stage": "direct", "reason": "current_redirect_not_historically_verified",
                                "normalized": normalized, "redirects": redirects})
        else:
            page = pages[0]
            if eligible:
                title = subject
                for mapping in normalized:
                    if mapping["from"] == title:
                        title = mapping["to"]
                if title != page["title"]:
                    raise DiscoveryError("Direct lookup returned an unexplained title substitution")
                selected.append(_candidate(page, "direct_title", None, stamp, stamp))
    search, stamp = _request(record, "search", _params(list="search", srsearch=record["query"],
        srnamespace="0", srlimit=str(search_limit), srprop="", srinfo=""), cache_dir, **options)
    matches = search.get("search")
    if not isinstance(matches, list) or len(matches) > search_limit:
        raise DiscoveryError("Malformed or excessive search results")
    ranked, seen = [], set()
    for rank, match in enumerate(matches, 1):
        if not isinstance(match, dict):
            raise DiscoveryError("Search result must be an object")
        page_id = acquisition._positive_id(match.get("pageid"), "search page ID")
        corpus._string(match.get("title"), "title", "search")
        if type(match.get("ns")) is not int or match["ns"] != 0:
            raise DiscoveryError("Search result is outside requested main namespace")
        if page_id in seen:
            diagnostics.append({"stage": "search", "reason": "duplicate_page_id", "page_id": page_id, "search_rank": rank})
            continue
        seen.add(page_id)
        ranked.append((rank, match))
    if not ranked:
        diagnostics.append({"stage": "search", "reason": "empty_search_results"})
    else:
        metadata, metadata_stamp = _request(record, "search_metadata", _params(
            pageids="|".join(str(match["pageid"]) for _, match in ranked),
            prop="info|pageprops", ppprop="disambiguation"), cache_dir, **options)
        if _mappings(metadata, "redirects"):
            raise DiscoveryError("Search metadata unexpectedly resolved current redirects")
        pages = metadata.get("pages")
        if not isinstance(pages, list) or any(not isinstance(page, dict) for page in pages):
            raise DiscoveryError("Malformed search page metadata")
        by_id = {}
        for page in pages:
            page_id = acquisition._positive_id(page.get("pageid"), "metadata page ID")
            if page_id in by_id:
                raise DiscoveryError("Duplicate page ID in metadata response")
            by_id[page_id] = page
        if set(by_id) != seen:
            raise DiscoveryError("Search metadata does not cover requested page IDs exactly")
        for rank, match in ranked:
            page = by_id[match["pageid"]]
            if not _eligible(page, diagnostics, "search_metadata"):
                continue
            if page["title"] != match["title"]:
                diagnostics.append({"stage": "search_metadata", "reason": "title_changed_since_search",
                                    "page_id": page["pageid"], "search_title": match["title"], "metadata_title": page["title"]})
            if page["pageid"] in {candidate["page_id"] for candidate in selected}:
                diagnostics.append({"stage": "selection", "reason": "duplicate_page_id", "page_id": page["pageid"], "search_rank": rank})
            elif len(selected) < candidate_limit:
                selected.append(_candidate(page, "search", rank, stamp, metadata_stamp))
    record["status"] = "candidates_unverified" if selected else "unresolved"


def _records(identities):
    return [{"input_index": i, **identity, "query": retrieval.build_query(identity["SubjectEntity"], identity["Relation"]),
             "status": "not_attempted", "candidates": [], "diagnostics": [], "api_responses": []}
            for i, identity in enumerate(identities, 1)]


def _articles(records):
    seen, articles = set(), []
    for record in records:
        for candidate in record["candidates"]:
            if candidate["page_id"] not in seen:
                articles.append({"page_id": candidate["page_id"]})
                seen.add(candidate["page_id"])
    return articles


def _identity_hash(identities):
    return corpus._sha256(b"".join(corpus._json_bytes(row) for row in identities))


def prepare_pilot(train_path, output_path, *, expected_train_sha256=PILOT_TRAIN_SHA256):
    """Freeze first three identities per relation; hash override is for synthetic tests.

    The CLI always pins the reviewed training snapshot. Gold fields are neither
    inspected nor selected; hashing the source bytes only establishes provenance.
    """
    train, output = Path(train_path), Path(output_path)
    corpus._check_location(output)
    if output.exists() or output.is_symlink():
        raise DiscoveryError(f"Refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise DiscoveryError("Output parent directory must already exist")
    if not retrieval._hash(expected_train_sha256):
        raise DiscoveryError("Expected training SHA256 must be 64 lowercase hexadecimal characters")
    source = train.read_bytes()
    if corpus._sha256(source) != expected_train_sha256:
        raise DiscoveryError("Training snapshot SHA256 mismatch; no pilot published")
    identities = retrieval.load_identities(train)
    counts, selected = dict.fromkeys(retrieval.QUERIES, 0), []
    for identity in identities:
        relation = identity["Relation"]
        if counts[relation] < 3:
            selected.append(identity)
            counts[relation] += 1
    if len(selected) != 18 or any(count != 3 for count in counts.values()):
        raise DiscoveryError("Pilot requires three training identities for each of six relations")
    if train.read_bytes() != source:
        raise DiscoveryError("Training input changed during pilot preparation")
    corpus._publish(output, b"".join(corpus._json_bytes(row) for row in selected))
    return {"train_sha256": expected_train_sha256, "input_identities_sha256": _identity_hash(selected),
            "record_count": len(selected)}


def discover(*, identities_path, output_path, cache_dir, allow_network=False, user_agent=None, request_delay=1.0):
    """Discover metadata only. Failed runs retain a report, never a final selection."""
    if type(allow_network) is not bool or type(request_delay) not in (int, float) or not math.isfinite(request_delay) or request_delay < 1:
        raise DiscoveryError("Network permission must be boolean; request delay must be finite and >= 1 second")
    if allow_network and (not isinstance(user_agent, str) or not user_agent.isascii()
            or any(ord(c) < 32 for c in user_agent) or not re.match(r"[A-Za-z][A-Za-z0-9_.-]*/[^ ]+ \(.+\)", user_agent)
            or not any(contact in user_agent for contact in ("@", "https://", "http://", "User:"))):
        raise DiscoveryError("Live discovery requires an informative project/version User-Agent with real contact information")
    output, cache_dir = Path(output_path), Path(cache_dir)
    articles_path, manifest_path = Path(str(output) + ".articles.jsonl"), Path(str(output) + ".manifest.json")
    failure_path = Path(str(output) + ".failure.json")
    for path in (output, articles_path, manifest_path, failure_path, cache_dir):
        corpus._check_location(path)
        if path != cache_dir and (path.exists() or path.is_symlink()):
            raise DiscoveryError(f"Refusing to overwrite {path}")
    if not output.parent.is_dir():
        raise DiscoveryError("Output parent directory must already exist")
    identities = retrieval.load_identities(identities_path)
    if not allow_network and not cache_dir.is_dir():
        raise DiscoveryError("Offline cache directory does not exist; no live request made")
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = _records(identities)
    try:
        for record in records:
            _discover_record(record, cache_dir, allow_network=allow_network, user_agent=user_agent, request_delay=request_delay)
        if retrieval.load_identities(identities_path) != identities:
            raise DiscoveryError("Input identities changed during discovery")
        articles = _articles(records)
        data = b"".join(corpus._json_bytes(row) for row in records)
        article_data = b"".join(corpus._json_bytes(row) for row in articles)
        if articles:
            acquisition._requests(article_data)
        # Replay every used cache before publication, checking identity coverage,
        # response bytes, selection and ordering without any further live access.
        replay = _records(identities)
        for record in replay:
            _discover_record(record, cache_dir, allow_network=False, user_agent=None, request_delay=request_delay)
        if replay != records:
            raise DiscoveryError("API caches changed before publication")
        manifest = {"schema_version": POLICY_VERSION, "status": "complete", "endpoint": ENDPOINT,
                    "query_policy_version": retrieval.QUERY_VERSION, "search_limit": SEARCH_LIMIT,
                    "candidate_limit": CANDIDATE_LIMIT, "historical_revision_cutoff": HISTORICAL_CUTOFF,
                    "search_index_scope": SEARCH_SCOPE, "timestamp_policy": TIMESTAMP_POLICY,
                    "input_identities_sha256": _identity_hash(identities), "input_count": len(identities),
                    "record_count": len(records), "article_count": len(articles),
                    "unresolved_count": sum(record["status"] == "unresolved" for record in records),
                    "output_sha256": corpus._sha256(data), "articles_sha256": corpus._sha256(article_data),
                    "code_sha256": corpus._sha256(Path(__file__).read_bytes()),
                    "request_policy": {"serial": True, "minimum_delay_seconds": request_delay, "automatic_retries": 0,
                                       "maxlag": 5, "timeout_seconds": 30}}
        corpus._publish(output, data)
        corpus._publish(articles_path, article_data)
        corpus._publish(manifest_path, corpus._json_bytes(manifest))
        return manifest
    except (ValueError, OSError, KeyboardInterrupt) as error:
        for record in records:
            if record["status"] == "not_attempted" and record["api_responses"]:
                record.update(status="failed", error=str(error) or type(error).__name__)
        report = {"schema_version": POLICY_VERSION, "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "incomplete",
                  "input_identities_sha256": _identity_hash(identities), "input_count": len(identities),
                  "error": str(error) or type(error).__name__, "records": records}
        corpus._publish(failure_path, corpus._json_bytes(report))
        raise


def load_discovery(output_path, *, cache_dir):
    """Require a complete, integrity-checked selection and replay its caches offline."""
    output = Path(output_path)
    data = corpus._input_bytes(output)
    article_data = corpus._input_bytes(Path(str(output) + ".articles.jsonl"))
    manifest = acquisition._json_object(corpus._input_bytes(Path(str(output) + ".manifest.json")), "Discovery manifest")
    if manifest.get("schema_version") == "wikipedia-article-discovery-v2":
        from .wikipedia_discovery_v2 import load_discovery as load_v2
        return load_v2(output, cache_dir=cache_dir)
    if (manifest.get("schema_version") != POLICY_VERSION or manifest.get("status") != "complete"
            or manifest.get("output_sha256") != corpus._sha256(data)
            or manifest.get("articles_sha256") != corpus._sha256(article_data)
            or manifest.get("query_policy_version") != retrieval.QUERY_VERSION
            or manifest.get("historical_revision_cutoff") != HISTORICAL_CUTOFF
            or manifest.get("endpoint") != ENDPOINT
            or manifest.get("search_index_scope") != SEARCH_SCOPE or manifest.get("timestamp_policy") != TIMESTAMP_POLICY
            or not retrieval._hash(manifest.get("code_sha256"))
            or manifest.get("search_limit") != SEARCH_LIMIT or manifest.get("candidate_limit") != CANDIDATE_LIMIT):
        raise DiscoveryError("Incomplete, incompatible or fingerprint-mismatched discovery artifacts")
    policy = manifest.get("request_policy")
    if (not isinstance(policy, dict) or type(policy.get("minimum_delay_seconds")) not in (int, float)
            or not math.isfinite(policy["minimum_delay_seconds"]) or policy["minimum_delay_seconds"] < 1
            or policy != {"serial": True, "minimum_delay_seconds": policy["minimum_delay_seconds"],
                          "automatic_retries": 0, "maxlag": 5, "timeout_seconds": 30}):
        raise DiscoveryError("Invalid discovery request policy")
    identities = retrieval.load_identities(output)
    expected = _records(identities)
    for record in expected:
        _discover_record(record, Path(cache_dir), allow_network=False, user_agent=None, request_delay=1.0)
    articles = _articles(expected)
    if (data != b"".join(corpus._json_bytes(row) for row in expected)
            or article_data != b"".join(corpus._json_bytes(row) for row in articles)
            or manifest.get("input_identities_sha256") != _identity_hash(identities)
            or any(type(manifest.get(key)) is not int or manifest[key] != value for key, value in {
                "input_count": len(identities), "record_count": len(expected), "article_count": len(articles),
                "unresolved_count": sum(record["status"] == "unresolved" for record in expected)}.items())):
        raise DiscoveryError("Discovery coverage, cache provenance or selection mismatch")
    return expected, articles, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--identities", type=Path, help="JSONL identities: v1 SubjectEntity/Relation; v2 uses --input-format")
    inputs.add_argument("--prepare-pilot", type=Path, metavar="TRAIN_JSONL", help="Freeze 18 identities from the SHA256-pinned training split; no network")
    parser.add_argument("--output", required=True, type=Path, help="New discovery JSONL, outside benchmark/results paths")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-network", action="store_true", help="Explicitly permit live metadata cache misses")
    parser.add_argument("--user-agent", help="Project/version (real contact URL/email/wiki user)")
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--policy-version", choices=("v1", "v2"), default="v1", help="Default v1; opt into generic v2")
    parser.add_argument("--input-format", choices=("generic", "lm-kbc-2026"), help="V2 only; default generic")
    parser.add_argument("--candidate-limit", type=int, help="V2 only; default 5")
    parser.add_argument("--search-limit", type=int, help="V2 only; default 10")
    parser.add_argument("--cutoff", help="V2 only; canonical whole-second UTC; default 2026-07-01T00:00:00Z")
    args = parser.parse_args(argv)
    if (args.prepare_pilot is not None or args.policy_version == "v1") and (
            args.input_format is not None or args.candidate_limit is not None or args.search_limit is not None
            or args.cutoff is not None or (args.prepare_pilot is not None and args.policy_version != "v1")):
        parser.error("Generic inputs/configuration require discovery --policy-version v2; pilot preparation remains v1")
    if args.prepare_pilot is not None and (args.cache_dir is not None or args.allow_network or args.user_agent):
        parser.error("Pilot preparation does not accept cache or live-network options")
    if args.prepare_pilot is None and args.cache_dir is None:
        parser.error("Discovery requires --cache-dir")
    try:
        if args.prepare_pilot is not None:
            result = prepare_pilot(args.prepare_pilot, args.output)
            print(f"Pilot records: {result['record_count']}; training SHA256: {result['train_sha256']}; "
                  f"identity SHA256: {result['input_identities_sha256']}")
            return 0
        options = {"allow_network": args.allow_network, "user_agent": args.user_agent, "request_delay": args.request_delay}
        if args.policy_version == "v2":
            from . import wikipedia_discovery_v2 as v2
            config = v2.DiscoveryConfig(
                candidate_limit=5 if args.candidate_limit is None else args.candidate_limit,
                search_limit=10 if args.search_limit is None else args.search_limit,
                historical_cutoff=HISTORICAL_CUTOFF if args.cutoff is None else args.cutoff)
            result = v2.discover(identities_path=args.identities, output_path=args.output, cache_dir=args.cache_dir,
                                 input_format=args.input_format or "generic", configuration=config, **options)
        else:
            result = discover(identities_path=args.identities, output_path=args.output, cache_dir=args.cache_dir, **options)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Wikipedia discovery failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Wikipedia discovery interrupted; no complete selection published\n")
    print(f"Records: {result['record_count']}; selected articles: {result['article_count']}; unresolved: {result['unresolved_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
