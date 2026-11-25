"""Standalone, gold-free query construction; no discovery or retriever imports."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType


QUERY_POLICY_VERSION = "rag-query-v1"
LM_KBC_2026_NAMESPACE = "lm-kbc-2026"
# Deliberately isolated from the unchanged v1 retriever. Parity tests guard drift.
_LM_KBC_TEMPLATES = MappingProxyType({
    "awardWonBy": "{subject} award winners recipients",
    "companyTradesAtStockExchange": "{subject} stock exchange listing",
    "countryLandBordersCountry": "{subject} land borders neighbouring countries",
    "hasArea": "{subject} total area",
    "hasCapacity": "{subject} capacity",
    "personHasCityOfDeath": "{subject} place of death",
})


class QueryError(ValueError):
    """Invalid query inputs; no inferred semantics or silent input repairs."""


@dataclass(frozen=True)
class QueryInput:
    """An identity and optional explicit task definition, never object labels.

    Use validate_query_input() for mappings. Construction alone does not validate
    Python runtime types; build_query() always validates again before use.
    """

    subject: str
    relation_id: str
    relation_description: str | None = None
    relation_namespace: str | None = None


def validate_query_input(value: object) -> QueryInput:
    """Validate independently of query policy; reject unknown fields, including gold.

    Preserve exact strings. Optional fields may be absent or None, but not blank.
    Dataset adapters must explicitly project permitted fields before calling.
    """
    if type(value) is QueryInput:
        query = value
    elif isinstance(value, Mapping):
        required = {"subject", "relation_id"}
        allowed = required | {"relation_description", "relation_namespace"}
        if any(not isinstance(key, str) for key in value):
            raise QueryError("Query field names must be strings")
        if required - value.keys():
            raise QueryError("Query requires subject and relation_id")
        if value.keys() - allowed:
            raise QueryError("Unknown query fields; project identity/task fields only, never gold labels")
        query = QueryInput(**value)
    else:
        raise QueryError("Expected QueryInput or a mapping of query fields")
    for field in ("subject", "relation_id", "relation_description", "relation_namespace"):
        text = getattr(query, field)
        if text is None and field in {"relation_description", "relation_namespace"}:
            continue
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise QueryError(f"{field} must be a nonempty string without NUL")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise QueryError(f"{field} contains invalid Unicode") from error
    return query


def _fingerprint(kind: str, payload: dict) -> str:
    data = {"fingerprint_type": kind, **payload}
    encoded = (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_query(value: object) -> dict[str, str | bool | None]:
    """Return query text and deterministic provenance under the fixed v1 policy.

    Namespace and identifier matching is exact. A literal fallback never claims
    semantic resolution, including when its identifier happens to be readable.
    """
    query = validate_query_input(value)
    fields = asdict(query)
    template_id = None
    if query.relation_namespace == LM_KBC_2026_NAMESPACE and query.relation_id in _LM_KBC_TEMPLATES:
        method = "lm_kbc_2026_template"
        template = _LM_KBC_TEMPLATES[query.relation_id]
        template_id = f"{LM_KBC_2026_NAMESPACE}:{query.relation_id}"
    elif query.relation_description is not None:
        method, template = "relation_description", "{subject} {relation_description}"
    else:
        method, template = "literal_relation_fallback", "{subject} {relation_id}"
    identity = {key: fields[key] for key in ("subject", "relation_id", "relation_namespace")}
    task = {key: fields[key] for key in ("relation_id", "relation_namespace", "relation_description")}
    provenance = {
        **fields, "query_policy_version": QUERY_POLICY_VERSION, "method": method,
        "template_id": template_id, "template": template,
        "limited_fallback": method == "literal_relation_fallback", "query": template.format(**fields),
        "identity_sha256": _fingerprint("rag-query-identity-v1", identity),
        "task_definition_sha256": _fingerprint("rag-query-task-definition-v1", task),
    }
    provenance["query_sha256"] = _fingerprint("rag-query-provenance-v1", provenance)
    return provenance
