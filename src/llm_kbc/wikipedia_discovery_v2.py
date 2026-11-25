"""Generic Wikipedia discovery v2; reuse v1 transport, eligibility and selection.

No corpus assembly or generation. Offline by default; v1 artifacts stay separate.
"""

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from . import evidence_corpus as corpus
from . import rag_query
from . import wikipedia_collect as acquisition
from . import wikipedia_discovery as v1


POLICY_VERSION = "wikipedia-article-discovery-v2"
BACKEND_POLICY = "wikipedia-unicode-controls-and-format-v1"
SEARCH_POLICY = "mediawiki-verbatim-search-v1"
INPUT_FORMATS = ("generic", "lm-kbc-2026")
DiscoveryError = v1.DiscoveryError


@dataclass(frozen=True)
class DiscoveryConfig:
    """Bounded one-search/one-metadata-batch acquisition; no continuation."""

    candidate_limit: int = 5
    search_limit: int = 10
    historical_cutoff: str = v1.HISTORICAL_CUTOFF

    def __post_init__(self):
        if (type(self.candidate_limit) is not int or type(self.search_limit) is not int
                or not 1 <= self.candidate_limit <= self.search_limit <= 10):
            raise DiscoveryError("Require integer limits: 1 <= candidate_limit <= search_limit <= 10")
        instant = acquisition._timestamp(self.historical_cutoff)
        if (not self.historical_cutoff.endswith("Z") or instant.microsecond
                or self.historical_cutoff != instant.isoformat(timespec="seconds").replace("+00:00", "Z")):
            raise DiscoveryError("Historical cutoff must be canonical whole-second UTC: YYYY-MM-DDTHH:MM:SSZ")


def project_input(row: Mapping, input_format: str = "generic") -> rag_query.QueryInput:
    """Read permitted keys only, without inspecting unused label/metadata values."""
    if not isinstance(input_format, str) or input_format not in INPUT_FORMATS:
        raise DiscoveryError("Input format must be generic or lm-kbc-2026")
    if not isinstance(row, Mapping):
        raise DiscoveryError("Discovery input must be an object")
    if input_format == "lm-kbc-2026":
        subject, relation = row.get("SubjectEntity"), row.get("Relation")
        # Deliberately strict benchmark adapter; the generic branch has no whitelist.
        v1.retrieval.build_query(subject, relation)
        value = rag_query.QueryInput(subject, relation, relation_namespace=rag_query.LM_KBC_2026_NAMESPACE)
    else:
        value = rag_query.validate_query_input({key: row.get(key) for key in (
            "subject", "relation_id", "relation_namespace", "relation_description")})
    return rag_query.validate_query_input(value)


def _backend_validate(query):
    for field, text in asdict(query).items():
        if text is None:
            continue
        for char in text:
            category = unicodedata.category(char)
            if category in {"Cc", "Zl", "Zp"} or (category == "Cf" and char not in "\u200c\u200d"):
                raise DiscoveryError(f"{field}: unsupported Wikipedia input U+{ord(char):04X} under {BACKEND_POLICY}")
        if not any(not char.isspace() and unicodedata.category(char) not in {"Cf", "Mn", "Me"} for char in text):
            raise DiscoveryError(f"{field}: Wikipedia input must contain visible text")


def _validate_batch(queries):
    seen = {}
    for index, query in enumerate(queries, 1):
        rag_query.validate_query_input(query)
        _backend_validate(query)
        key = (" ".join(unicodedata.normalize("NFKC", query.subject).casefold().split()),
               query.relation_namespace, query.relation_id)
        if key in seen:
            first_index, first = seen[key]
            if first.relation_description != query.relation_description:
                reason = "conflicting relation descriptions"
            else:
                reason = "exact duplicate identity" if first.subject == query.subject else "normalized duplicate identity"
            raise DiscoveryError(f"Input {index}: {reason} (first at input {first_index})")
        seen[key] = index, query
    if not queries:
        raise DiscoveryError("No input queries")


def load_inputs(path, input_format="generic"):
    """Strict JSONL syntax, explicit projection, and separate batch duplicate policy."""
    try:
        lines = Path(path).read_bytes().decode("utf-8").split("\n")
    except UnicodeError as error:
        raise DiscoveryError("Discovery inputs must be UTF-8") from error
    if lines[-1] == "":
        lines.pop()
    queries = []
    for index, line in enumerate(lines, 1):
        try:
            queries.append(project_input(acquisition._json_object(line, f"{path}:{index}"), input_format))
        except ValueError as error:
            raise DiscoveryError(f"{path}:{index}: {error}") from error
    _validate_batch(queries)
    return queries


def _records(queries):
    records = []
    for index, query in enumerate(queries, 1):
        provenance = rag_query.build_query(query)
        records.append({"input_index": index, **asdict(query), "query_provenance": provenance,
                        "search_expression": provenance["query"], "status": "not_attempted",
                        "candidates": [], "diagnostics": [], "api_responses": []})
    return records


def _process(record, cache_dir, configuration, **options):
    # Adapt to the shared selection engine; no LM-KBC fields enter v2 artifacts.
    engine = {"SubjectEntity": record["subject"], "query": record["search_expression"],
              **{key: record[key] for key in ("status", "candidates", "diagnostics", "api_responses")}}
    try:
        v1._discover_record(engine, cache_dir, candidate_limit=configuration.candidate_limit,
                            search_limit=configuration.search_limit, **options)
    except (ValueError, OSError, KeyboardInterrupt) as error:
        record.update(status="failed", error=str(error) or type(error).__name__)
        raise
    record["status"] = engine["status"]
    if len(record["candidates"]) < configuration.candidate_limit:
        record["diagnostics"].append({"stage": "selection", "reason": "candidate_shortfall",
                                      "requested": configuration.candidate_limit, "selected": len(record["candidates"])})


def _implementation_hashes():
    return {Path(module.__file__).name: corpus._sha256(Path(module.__file__).read_bytes())
            for module in (v1, rag_query, acquisition, corpus, v1.retrieval)} | {
                Path(__file__).name: corpus._sha256(Path(__file__).read_bytes())}


def _ordered_inputs(records):
    return [{"input_index": record["input_index"], **{key: record[key] for key in (
        "subject", "relation_id", "relation_description", "relation_namespace")},
        **{key: record["query_provenance"][key] for key in (
            "identity_sha256", "task_definition_sha256", "query_sha256")}} for record in records]


def _manifest(records, articles, configuration, input_format, delay, implementations):
    inputs = _ordered_inputs(records)
    return {
        "schema_version": POLICY_VERSION, "discovery_policy_version": POLICY_VERSION, "status": "complete",
        "endpoint": v1.ENDPOINT, "query_policy_version": rag_query.QUERY_POLICY_VERSION,
        "backend_validation_policy": BACKEND_POLICY, "search_expression_policy": SEARCH_POLICY,
        "unicode_version": unicodedata.unidata_version, "configuration": asdict(configuration), "input_format": input_format,
        "search_index_scope": v1.SEARCH_SCOPE, "timestamp_policy": v1.TIMESTAMP_POLICY,
        "ordered_inputs": inputs, "input_queries_sha256": corpus._sha256(corpus._json_bytes(inputs)),
        "input_count": len(records), "record_count": len(records), "article_count": len(articles),
        "unresolved_count": sum(r["status"] == "unresolved" for r in records),
        "shortfall_count": sum(len(r["candidates"]) < configuration.candidate_limit for r in records),
        "unique_request_count": len({ref["cache"]["key"] for r in records for ref in r["api_responses"]}),
        "output_sha256": corpus._sha256(b"".join(corpus._json_bytes(r) for r in records)),
        "articles_sha256": corpus._sha256(b"".join(corpus._json_bytes(r) for r in articles)),
        "implementation_sha256": implementations,
        "request_policy": {"serial": True, "minimum_delay_seconds": delay, "automatic_retries": 0,
                           "maxlag": 5, "timeout_seconds": 30},
    }


def _network_options(allow_network, user_agent, delay):
    if type(allow_network) is not bool or type(delay) not in (int, float) or not math.isfinite(delay) or delay < 1:
        raise DiscoveryError("Network permission must be boolean; request delay must be finite and >= 1 second")
    if allow_network and (not isinstance(user_agent, str) or not user_agent.isascii()
            or any(ord(c) < 32 for c in user_agent) or not re.match(r"[A-Za-z][A-Za-z0-9_.-]*/[^ ]+ \(.+\)", user_agent)
            or not any(contact in user_agent for contact in ("@", "https://", "http://", "User:"))):
        raise DiscoveryError("Live discovery requires an informative project/version User-Agent with real contact information")
    return {"allow_network": allow_network, "user_agent": user_agent, "request_delay": delay}


def discover(*, identities_path, output_path, cache_dir, input_format="generic", configuration=None,
             allow_network=False, user_agent=None, request_delay=1.0):
    configuration = DiscoveryConfig() if configuration is None else configuration
    if type(configuration) is not DiscoveryConfig:
        raise DiscoveryError("Expected DiscoveryConfig")
    options = _network_options(allow_network, user_agent, request_delay)
    output, cache = Path(output_path), Path(cache_dir)
    articles_path, manifest_path = Path(str(output) + ".articles.jsonl"), Path(str(output) + ".manifest.json")
    failure_path = Path(str(output) + ".failure.json")
    for path in (output, articles_path, manifest_path, failure_path, cache):
        corpus._check_location(path)
        if path != cache and (path.exists() or path.is_symlink()):
            raise DiscoveryError(f"Refusing to overwrite {path}")
    if not output.parent.is_dir():
        raise DiscoveryError("Output parent directory must already exist")
    queries = load_inputs(identities_path, input_format)
    if not allow_network and not cache.is_dir():
        raise DiscoveryError("Offline cache directory does not exist; no live request made")
    cache.mkdir(parents=True, exist_ok=True)
    records = _records(queries)
    try:
        for record in records:
            _process(record, cache, configuration, **options)
        if load_inputs(identities_path, input_format) != queries:
            raise DiscoveryError("Input queries changed during discovery")
        replay = _records(queries)
        for record in replay:
            _process(record, cache, configuration, allow_network=False, user_agent=None, request_delay=request_delay)
        if replay != records:
            raise DiscoveryError("API caches changed before publication")
        articles = v1._articles(records)
        if articles:
            acquisition._requests(b"".join(corpus._json_bytes(row) for row in articles))
        manifest = _manifest(records, articles, configuration, input_format, request_delay, _implementation_hashes())
        corpus._publish(output, b"".join(corpus._json_bytes(row) for row in records))
        corpus._publish(articles_path, b"".join(corpus._json_bytes(row) for row in articles))
        corpus._publish(manifest_path, corpus._json_bytes(manifest))
        return manifest
    except (ValueError, OSError, KeyboardInterrupt) as error:
        report = {"schema_version": POLICY_VERSION, "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "incomplete",
                  "configuration": asdict(configuration), "input_format": input_format,
                  "ordered_inputs": _ordered_inputs(records), "error": str(error) or type(error).__name__, "records": records}
        corpus._publish(failure_path, corpus._json_bytes(report))
        raise


def load_discovery(output_path, *, cache_dir):
    """Validate v2 only; the v1 module's public reader dispatches both versions."""
    output = Path(output_path)
    data = corpus._input_bytes(output)
    article_data = corpus._input_bytes(Path(str(output) + ".articles.jsonl"))
    manifest_data = corpus._input_bytes(Path(str(output) + ".manifest.json"))
    manifest = acquisition._json_object(manifest_data, "Discovery v2 manifest")
    config = manifest.get("configuration")
    if not isinstance(config, dict) or set(config) != {"candidate_limit", "search_limit", "historical_cutoff"}:
        raise DiscoveryError("Invalid v2 configuration")
    configuration = DiscoveryConfig(**config)
    implementations = manifest.get("implementation_sha256")
    if (not isinstance(implementations, dict) or set(implementations) != set(_implementation_hashes())
            or any(not v1.retrieval._hash(value) for value in implementations.values())):
        raise DiscoveryError("Invalid implementation fingerprints")
    policy = manifest.get("request_policy")
    if not isinstance(policy, dict):
        raise DiscoveryError("Invalid request policy")
    delay = policy.get("minimum_delay_seconds")
    _network_options(False, None, delay)
    input_format = manifest.get("input_format")
    queries = load_inputs(output, "generic")
    records = _records(queries)
    for record in records:
        _process(record, Path(cache_dir), configuration, allow_network=False, user_agent=None, request_delay=delay)
    articles = v1._articles(records)
    expected = _manifest(records, articles, configuration, input_format, delay, implementations)
    if (input_format not in INPUT_FORMATS or (input_format == "lm-kbc-2026" and any(
            q.relation_namespace != rag_query.LM_KBC_2026_NAMESPACE or q.relation_description is not None
            or q.relation_id not in v1.retrieval.QUERIES for q in queries))
            or manifest_data != corpus._json_bytes(expected)
            or data != b"".join(corpus._json_bytes(r) for r in records)
            or article_data != b"".join(corpus._json_bytes(r) for r in articles)):
        raise DiscoveryError("Incomplete, incompatible or fingerprint/coverage/provenance-mismatched v2 artifacts")
    return records, articles, manifest
