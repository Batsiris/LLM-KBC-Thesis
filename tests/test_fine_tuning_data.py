"""Training preparation and identity-only audits; no models or external services."""

import json
import os
import subprocess
import sys

import pytest

from llm_kbc import fine_tuning_data as data, zero_shot
from llm_kbc.dataset_inspector import DatasetError


def record(subject="Subject", relation="hasArea", objects=None):
    return {
        "SubjectEntity": subject, "Relation": relation,
        "ObjectEntities": [["1.2300", "other alias"]] if objects is None else objects,
    }


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_conversations_use_all_relations_and_unchanged_zero_shot_messages(tmp_path):
    rows = [record(f"Subject {i}", relation) for i, relation in enumerate(zero_shot.QUESTIONS)]
    train = tmp_path / "train.jsonl"
    write_rows(train, rows)
    before = train.read_bytes()
    examples = data.prepare_training_data(tmp_path)
    assert len(examples) == len(rows)
    for row, example in zip(rows, examples, strict=True):
        assert example == {
            "SubjectEntity": row["SubjectEntity"], "Relation": row["Relation"],
            "messages": [
                *zero_shot.build_messages(row["SubjectEntity"], row["Relation"]),
                {"role": "assistant", "content": '["1.2300"]'},
            ],
        }
        assert [message["role"] for message in example["messages"]] == ["system", "user", "assistant"]
    assert train.read_bytes() == before


def test_first_aliases_unicode_punctuation_repeated_labels_and_order_are_preserved(tmp_path):
    subject = '  Åland "quoted" 東京  '
    labels = ["Éva O’Neil", "東京", "  Montréal  ", 'Name "quoted"\\suffix', "Éva O’Neil"]
    write_rows(tmp_path / "train.jsonl", [record(subject, "awardWonBy", [[label, "UNUSED_ALIAS"] for label in labels])])
    example = data.prepare_training_data(tmp_path)[0]
    assert example["SubjectEntity"] == subject
    assert example["messages"][:2] == zero_shot.build_messages(subject, "awardWonBy")
    target = example["messages"][-1]["content"]
    assert target == json.dumps(labels, ensure_ascii=False)
    assert json.loads(target) == labels
    assert "東京" in target and "Éva" in target and "UNUSED_ALIAS" not in target


@pytest.mark.parametrize("relation", ["hasArea", "hasCapacity"])
def test_numerical_strings_are_not_parsed_or_normalized(tmp_path, relation):
    labels = ["1.2300", "-0.0", "4.2E-7", "1,000", "9007199254740993", "0.12345678901234567890123456789"]
    write_rows(tmp_path / "train.jsonl", [record(relation=relation, objects=[[value, "unused"] for value in labels])])
    target = data.prepare_training_data(tmp_path)[0]["messages"][-1]["content"]
    assert target == json.dumps(labels, ensure_ascii=False)


def test_empty_objects_are_explicit_empty_array_targets(tmp_path):
    write_rows(tmp_path / "train.jsonl", [record(objects=[])])
    assert data.prepare_training_data(tmp_path)[0]["messages"][-1] == {"role": "assistant", "content": "[]"}


def test_large_answers_are_not_capped_or_truncated(tmp_path):
    labels = [f"Recipient {i}" for i in range(100)]
    write_rows(tmp_path / "train.jsonl", [record(relation="awardWonBy", objects=[[label] for label in labels])])
    assert json.loads(data.prepare_training_data(tmp_path)[0]["messages"][-1]["content"]) == labels


def test_deterministic_train_order_without_validation_or_test_reads(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    rows = [record("Zulu"), record("Alpha"), record("Middle")]
    write_rows(train, rows)
    for split in ("val", "test"):
        (tmp_path / f"{split}.jsonl").write_text("not JSON", encoding="utf-8")
    original_open = type(train).open

    def guarded_open(path, *args, **kwargs):
        if path.name in ("val.jsonl", "test.jsonl"):
            raise AssertionError("Preparation must not open validation/test files")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(type(train), "open", guarded_open)
    first = data.prepare_training_data(tmp_path)
    assert first == data.prepare_training_data(tmp_path)
    assert [row["SubjectEntity"] for row in first] == ["Zulu", "Alpha", "Middle"]
    write_rows(train, list(reversed(rows)))
    assert data.prepare_training_data(tmp_path) == list(reversed(first))


@pytest.mark.parametrize("duplicate", ["Alpha Beta", " ALPHA   BETA ", "Ａｌｐｈａ　Ｂｅｔａ", "alpha\tbeta"])
def test_normalized_duplicate_training_pairs_fail_with_line_context(tmp_path, duplicate):
    write_rows(tmp_path / "train.jsonl", [record("Alpha Beta"), record(duplicate)])
    with pytest.raises(DatasetError, match=r"train\.jsonl:2: duplicate normalized training identity.*line 1"):
        data.prepare_training_data(tmp_path)


def test_same_subject_in_different_relations_is_not_a_duplicate(tmp_path):
    write_rows(tmp_path / "train.jsonl", [record("Alpha"), record(" ALPHA ", "hasCapacity")])
    assert len(data.prepare_training_data(tmp_path)) == 2


@pytest.mark.parametrize("objects", [
    None, "label", {}, ["flat"], [[]], [[1]], [[True]], [[None]], [[{}]], [[["nested"]]],
    [["valid", ""]], [["valid", "  "]], [["valid", False]],
])
def test_invalid_alias_groups_and_all_aliases_are_validated(tmp_path, objects):
    row = record()
    row["ObjectEntities"] = objects
    write_rows(tmp_path / "train.jsonl", [row])
    with pytest.raises(DatasetError, match=r"train\.jsonl:1:.*ObjectEntities"):
        data.prepare_training_data(tmp_path)


@pytest.mark.parametrize("field", ["SubjectEntity", "Relation", "ObjectEntities"])
def test_missing_training_fields_fail(tmp_path, field):
    row = record()
    del row[field]
    write_rows(tmp_path / "train.jsonl", [row])
    with pytest.raises(DatasetError, match=f"missing required field.*{field}"):
        data.prepare_training_data(tmp_path)


@pytest.mark.parametrize("field,value", [("SubjectEntity", 1), ("SubjectEntity", " "), ("Relation", None), ("Relation", "")])
def test_invalid_training_identity_fields_fail(tmp_path, field, value):
    row = record()
    row[field] = value
    write_rows(tmp_path / "train.jsonl", [row])
    with pytest.raises(DatasetError, match=field):
        data.prepare_training_data(tmp_path)


@pytest.mark.parametrize("text", ['{malformed}\n', '\n', '[]\n', 'null\n', '{"SubjectEntity": "truncated"'])
def test_malformed_training_input_fails_without_returning_partial_data(tmp_path, text):
    train = tmp_path / "train.jsonl"
    write_rows(train, [record()])
    with train.open("a", encoding="utf-8") as stream:
        stream.write(text)
    with pytest.raises(DatasetError, match=r"train\.jsonl:2:"):
        data.prepare_training_data(tmp_path)


def test_missing_empty_and_non_utf8_training_files_fail(tmp_path):
    train = tmp_path / "train.jsonl"
    with pytest.raises(DatasetError, match="file not found"):
        data.prepare_training_data(tmp_path)
    train.write_bytes(b"")
    with pytest.raises(DatasetError, match="must not be empty"):
        data.prepare_training_data(tmp_path)
    train.write_bytes(b"\xff\n")
    with pytest.raises(DatasetError, match="UTF-8"):
        data.prepare_training_data(tmp_path)


def test_unsupported_training_relation_fails_explicitly(tmp_path):
    write_rows(tmp_path / "train.jsonl", [record(relation="unsupported")])
    with pytest.raises(DatasetError, match=r"train\.jsonl:1: unsupported relation"):
        data.prepare_training_data(tmp_path)


@pytest.mark.parametrize("split", ["val", "test"])
def test_training_link_to_other_split_is_rejected(tmp_path, split):
    other = tmp_path / f"{split}.jsonl"
    write_rows(other, [record()])
    os.link(other, tmp_path / "train.jsonl")
    with pytest.raises(DatasetError, match="not a validation/test link"):
        data.prepare_training_data(tmp_path)


def test_default_directory_is_the_official_local_training_location(tmp_path, monkeypatch):
    directory = tmp_path / "data" / "lm_kbc_2026"
    directory.mkdir(parents=True)
    write_rows(directory / "train.jsonl", [record()])
    monkeypatch.chdir(tmp_path)
    assert data.prepare_training_data()[0]["SubjectEntity"] == "Subject"


def test_overlap_audit_reports_pairs_and_subject_only_matches_in_file_order(tmp_path):
    write_rows(tmp_path / "train.jsonl", [record("Zulu"), record("Alpha"), record("ALPHA", "hasCapacity")])
    validation = [
        {"SubjectEntity": " Ａｌｐｈａ ", "Relation": "hasArea"},
        {"SubjectEntity": "zULU", "Relation": "awardWonBy", "ObjectEntities": {"unused": "secret"}},
        {"SubjectEntity": "Unseen", "Relation": "hasArea"},
    ]
    write_rows(tmp_path / "val.jsonl", validation)
    expected_pair = {
        "train_line": 2, "validation_line": 1,
        "train_subject": "Alpha", "train_relation": "hasArea",
        "validation_subject": " Ａｌｐｈａ ", "validation_relation": "hasArea",
    }
    expected_subjects = [
        {**expected_pair, "train_line": 3, "train_subject": "ALPHA", "train_relation": "hasCapacity"},
        {"train_line": 1, "validation_line": 2, "train_subject": "Zulu", "train_relation": "hasArea",
         "validation_subject": "zULU", "validation_relation": "awardWonBy"},
    ]
    before = data.prepare_training_data(tmp_path)
    report = data.audit_validation_overlap(tmp_path)
    assert report == {
        "training_records": 3, "validation_records": 3,
        "pair_overlaps": [expected_pair], "subject_only_overlaps": expected_subjects,
        "subject_matching": "NFKC, casefold, collapsed whitespace; no alias resolution",
    }
    assert data.audit_validation_overlap(tmp_path) == report
    assert data.prepare_training_data(tmp_path) == before


def test_audit_never_accesses_validation_gold_or_test_file(tmp_path, monkeypatch):
    write_rows(tmp_path / "train.jsonl", [record()])
    validation = tmp_path / "val.jsonl"
    write_rows(validation, [{"SubjectEntity": "Query", "Relation": "hasArea", "ObjectEntities": "SECRET"}])
    (tmp_path / "test.jsonl").write_text("not JSON", encoding="utf-8")
    original_loads = data.json.loads

    class IdentityOnly(dict):
        def get(self, key, *args):
            assert key != "ObjectEntities", "Validation labels must not be accessed"
            return super().get(key, *args)

        def __getitem__(self, key):
            assert key != "ObjectEntities", "Validation labels must not be accessed"
            return super().__getitem__(key)

    def guarded_loads(text, *args, **kwargs):
        row = original_loads(text, *args, **kwargs)
        return IdentityOnly(row) if row.get("SubjectEntity") == "Query" else row

    monkeypatch.setattr(data.json, "loads", guarded_loads)
    report = data.audit_validation_overlap(tmp_path)
    assert report["pair_overlaps"] == report["subject_only_overlaps"] == []
    assert "SECRET" not in repr(report)
    monkeypatch.setattr(data.json, "loads", original_loads)
    write_rows(validation, [{"SubjectEntity": "Query", "Relation": "hasArea"}])
    assert data.audit_validation_overlap(tmp_path) == report


@pytest.mark.parametrize("text", [
    '{bad}\n', '\n', '[]\n', 'null\n', '{}\n',
    '{"SubjectEntity": 1, "Relation": "hasArea"}\n',
    '{"SubjectEntity": "Query", "Relation": "unsupported"}\n',
])
def test_audit_rejects_invalid_validation_identities_with_line_context(tmp_path, text):
    write_rows(tmp_path / "train.jsonl", [record()])
    (tmp_path / "val.jsonl").write_text(text, encoding="utf-8")
    with pytest.raises(DatasetError, match=r"val\.jsonl:1:"):
        data.audit_validation_overlap(tmp_path)


def test_audit_validates_training_duplicates_too(tmp_path):
    write_rows(tmp_path / "train.jsonl", [record(), record(" SUBJECT ")])
    with pytest.raises(DatasetError, match="duplicate normalized training identity"):
        data.audit_validation_overlap(tmp_path)


def test_audit_handles_missing_non_utf8_and_empty_validation(tmp_path):
    write_rows(tmp_path / "train.jsonl", [record()])
    validation = tmp_path / "val.jsonl"
    with pytest.raises(DatasetError, match="validation input file not found"):
        data.audit_validation_overlap(tmp_path)
    validation.write_bytes(b"\xff\n")
    with pytest.raises(DatasetError, match="UTF-8"):
        data.audit_validation_overlap(tmp_path)
    validation.write_bytes(b"")
    report = data.audit_validation_overlap(tmp_path)
    assert report["validation_records"] == 0
    assert report["pair_overlaps"] == report["subject_only_overlaps"] == []


def test_module_import_does_not_load_gpu_or_training_dependencies():
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; import llm_kbc.fine_tuning_data; "
        "assert not {'torch', 'transformers', 'peft', 'trl', 'bitsandbytes'} & sys.modules.keys()",
    ], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
