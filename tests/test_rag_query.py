"""Synthetic query policies and provenance; no dataset, network or model access."""

import copy
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest

from llm_kbc import bm25_retrieval as legacy
from llm_kbc import rag_query as query


@pytest.mark.parametrize("relation", list(legacy.QUERIES))
@pytest.mark.parametrize("subject", ["Unseen fictional subject", "Νέο θέμα 東京 café {subject}"])
def test_all_legacy_queries_are_exactly_preserved(relation, subject):
    result = query.build_query(query.QueryInput(subject, relation, relation_namespace=query.LM_KBC_2026_NAMESPACE))
    assert result["query"] == legacy.build_query(subject, relation)
    assert result["method"] == "lm_kbc_2026_template"
    assert result["template_id"] == f"lm-kbc-2026:{relation}"
    assert result["limited_fallback"] is False


def test_unseen_relation_description_and_literal_fallback():
    described = query.QueryInput("Fictional Observatory", "locatedIn", "city where the observatory is located", "synthetic-kbc")
    result = query.build_query(described)
    assert result["query"] == "Fictional Observatory city where the observatory is located"
    assert result["method"] == "relation_description" and result["template_id"] is None
    assert result["limited_fallback"] is False
    literal = query.build_query(replace(described, relation_description=None))
    assert literal["query"] == "Fictional Observatory locatedIn"
    assert literal["method"] == "literal_relation_fallback" and literal["limited_fallback"] is True


def test_opaque_identifier_has_explicit_limited_fallback():
    result = query.build_query(query.QueryInput("Fictional Observatory", "P999999"))
    assert result["query"] == "Fictional Observatory P999999"
    assert result["method"] == "literal_relation_fallback"
    assert result["limited_fallback"] is True
    assert result["relation_description"] is result["relation_namespace"] is None


@pytest.mark.parametrize("namespace", [None, "another-dataset", "LM-KBC-2026", "lm-kbc-2026 "])
def test_colliding_relation_identifiers_do_not_inherit_lm_kbc_semantics(namespace):
    original = query.QueryInput("Synthetic object", "hasArea", relation_namespace=namespace)
    result = query.build_query(original)
    assert result["query"] == "Synthetic object hasArea" and result["limited_fallback"] is True
    described = query.build_query(replace(original, relation_description="administrative district"))
    assert described["query"] == "Synthetic object administrative district"
    assert described["method"] == "relation_description"


def test_unknown_relation_in_lm_namespace_still_supports_generic_policy():
    result = query.build_query(query.QueryInput("Synthetic object", "unseenPredicate", "explicit relation words", "lm-kbc-2026"))
    assert result["method"] == "relation_description"
    assert result["query"] == "Synthetic object explicit relation words"


def test_legacy_template_has_precedence_but_description_is_preserved():
    result = query.build_query(query.QueryInput("Synthetic object", "hasArea", "Provided metadata", "lm-kbc-2026"))
    assert result["query"] == "Synthetic object total area"
    assert result["relation_description"] == "Provided metadata"


@pytest.mark.parametrize("value", [None, [], "subject", 3, {}, {"subject": "A"},
    {"relation_id": "r"}, {"subject": "A", "relation_id": "r", 1: "bad field"},
    {"subject": "A", "relation_id": "r", "unexpected": "value"}])
def test_invalid_input_shape(value):
    with pytest.raises(query.QueryError):
        query.validate_query_input(value)


@pytest.mark.parametrize("field", ["subject", "relation_id", "relation_description", "relation_namespace"])
@pytest.mark.parametrize("value", ["", " \t\n", 1, False, [], {}, "bad\x00value", "bad\ud800"])
def test_invalid_fields_in_mappings_and_direct_instances(field, value):
    fields = {"subject": "Synthetic subject", "relation_id": "syntheticRelation", field: value}
    with pytest.raises(query.QueryError, match=field):
        query.validate_query_input(fields)
    with pytest.raises(query.QueryError, match=field):
        query.build_query(query.QueryInput(**fields))


@pytest.mark.parametrize("field", ["subject", "relation_id"])
def test_required_fields_cannot_be_none(field):
    with pytest.raises(query.QueryError, match=field):
        query.build_query({"subject": "A", "relation_id": "r", field: None})


def test_validation_does_not_require_a_known_relation():
    value = {"subject": "New subject", "relation_id": "neverSeenBefore"}
    assert query.validate_query_input(value) == query.QueryInput("New subject", "neverSeenBefore")


def test_absent_and_null_optional_fields_have_identical_provenance():
    minimal = {"subject": "A", "relation_id": "r"}
    explicit = {**minimal, "relation_description": None, "relation_namespace": None}
    assert query.build_query(minimal) == query.build_query(explicit)


def test_canonical_utf8_null_and_lf_fingerprint_contract():
    # Fixed digests checked against literal canonical UTF-8 payloads, including LF.
    result = query.build_query(query.QueryInput("Café", "R"))
    assert result["identity_sha256"] == "1371c87c9aa61af83f26c0866238f8b5ceb62102ef19367507f41383e6d5b388"
    assert result["task_definition_sha256"] == "5844fb187e54bc461c7c309c0478b5b98b84a8b7720962f224230b64cb53541b"
    assert result["query_sha256"] == "82f51c0e2dc433c33a8c2c27067f69d39bad9048ee789df72cda52d04390c595"


def test_fingerprint_separation():
    original = query.QueryInput("Synthetic A", "r", "description one", "synthetic")
    base = query.build_query(original)
    changed_description = query.build_query(replace(original, relation_description="description two"))
    assert base["identity_sha256"] == changed_description["identity_sha256"]
    assert base["task_definition_sha256"] != changed_description["task_definition_sha256"]
    assert base["query_sha256"] != changed_description["query_sha256"]
    changed_subject = query.build_query(replace(original, subject="Synthetic B"))
    assert base["task_definition_sha256"] == changed_subject["task_definition_sha256"]
    assert base["identity_sha256"] != changed_subject["identity_sha256"]
    assert base["query_sha256"] != changed_subject["query_sha256"]
    changed_namespace = query.build_query(replace(original, relation_namespace="other"))
    assert base["query"] == changed_namespace["query"]
    for field in ("identity_sha256", "task_definition_sha256", "query_sha256"):
        assert base[field] != changed_namespace[field]
    changed_relation = query.build_query(replace(original, relation_id="anotherRelation"))
    assert base["query"] == changed_relation["query"]
    assert base["identity_sha256"] != changed_relation["identity_sha256"]
    assert base["task_definition_sha256"] != changed_relation["task_definition_sha256"]
    assert base["query_sha256"] != changed_relation["query_sha256"]


def test_ignored_legacy_description_changes_provenance_even_when_query_text_is_unchanged():
    original = query.QueryInput("Synthetic A", "hasArea", relation_namespace="lm-kbc-2026")
    before = query.build_query(original)
    after = query.build_query(replace(original, relation_description="An explicit task description"))
    assert before["query"] == after["query"]
    assert before["identity_sha256"] == after["identity_sha256"]
    assert before["task_definition_sha256"] != after["task_definition_sha256"]
    assert before["query_sha256"] != after["query_sha256"]


def test_exact_unicode_and_whitespace_preservation_without_normalization():
    original = query.QueryInput("  Δοκιμή 東京 café e\u0301\n", "σχέση", "  description\twords ", "δοκιμή")
    result = query.build_query(original)
    assert result["query"] == original.subject + " " + original.relation_description
    assert {field: result[field] for field in asdict(original)} == asdict(original)
    composed = query.build_query(replace(original, subject=original.subject.replace("e\u0301", "é")))
    assert result["identity_sha256"] != composed["identity_sha256"]


def test_determinism_mapping_order_no_mutation_and_frozen_input():
    fields = {"subject": "Synthetic A", "relation_id": "r", "relation_description": "a description", "relation_namespace": "synthetic"}
    before = copy.deepcopy(fields)
    frozen = query.validate_query_input(fields)
    result = query.build_query(fields)
    assert result == query.build_query(dict(reversed(list(fields.items())))) == query.build_query(frozen)
    assert fields == before
    with pytest.raises(FrozenInstanceError):
        frozen.subject = "changed"
    result["subject"] = "changed output"
    assert query.build_query(frozen)["subject"] == "Synthetic A"


@pytest.mark.parametrize("gold", [[], [["label"]], {"arbitrary": None}, False])
def test_gold_is_rejected_at_generic_boundary_and_not_used_by_explicit_projection(gold):
    source = {"subject": "Synthetic A", "relation_id": "unseenRelation", "ObjectEntities": gold}
    with pytest.raises(query.QueryError, match="Unknown query fields"):
        query.validate_query_input(source)
    projected = {key: source[key] for key in ("subject", "relation_id")}
    assert query.build_query(projected) == query.build_query(query.QueryInput("Synthetic A", "unseenRelation"))


def test_builds_without_site_packages_network_or_existing_pipeline_imports():
    source = Path(__file__).resolve().parents[1] / "src"
    script = '''
import builtins, json, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'huggingface_hub', 'requests', 'socket', 'urllib'}:
        raise AssertionError('Forbidden dependency: ' + name)
    if any(part in name for part in ('bm25_retrieval', 'wikipedia_collect', 'evidence_corpus')):
        raise AssertionError('Pipeline dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from llm_kbc.rag_query import QueryInput, build_query
print(json.dumps(build_query(QueryInput('Synthetic A', 'opaque123')), sort_keys=True))
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(source)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == query.build_query(query.QueryInput("Synthetic A", "opaque123"))
