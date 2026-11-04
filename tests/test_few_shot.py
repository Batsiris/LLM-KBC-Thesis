"""Training-only prompt preparation checks; no inference, downloads, or scoring."""

import copy
import hashlib
import json
import random

import pytest

from llm_kbc import few_shot as fs, zero_shot
from llm_kbc.dataset_inspector import DatasetError
from llm_kbc.prediction_io import validate_prediction


def record(subject, relation, count):
    return {
        "SubjectEntity": subject, "Relation": relation,
        "ObjectEntities": [[f"Object {i}", "ALIAS_NOT_A_CANONICAL_LABEL"] for i in range(count)],
    }


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def training(tmp_path):
    rows = []
    for relation in zero_shot.QUESTIONS:
        if relation in fs.NUMERIC_RELATIONS:
            for i, value in enumerate(("1000", "0.1", "1.00", "100", "50")):
                row = record(f"{relation}-{i}", relation, 1)
                row["ObjectEntities"] = [[value, "NUMERIC_ALIAS_NOT_USED"]]
                rows.append(row)
        else:
            counts = (9, 19, 27, 29, 47) if relation == "awardWonBy" else (0, 1, 2, 4, 1)
            rows.extend(record(f"{relation}-{i}", relation, count) for i, count in enumerate(counts))
    write_rows(tmp_path / "train.jsonl", rows)
    return tmp_path, rows


def test_selection_is_deterministic_independent_of_file_order(training):
    directory, rows = training
    expected = fs.select_demonstrations(directory)
    random.Random(73).shuffle(rows)
    write_rows(directory / "train.jsonl", rows)
    assert fs.select_demonstrations(directory) == expected
    assert list(expected) == list(zero_shot.QUESTIONS)
    assert all(len(demos) == 3 for demos in expected.values())
    for demos in expected.values():
        for demo in demos:
            assert validate_prediction(demo) == demo


def test_cardinality_strata_numeric_ranks_and_canonical_labels(training):
    directory, _ = training
    selected = fs.select_demonstrations(directory)
    assert [len(row["ObjectEntities"]) for row in selected["countryLandBordersCountry"]] == [0, 1, 4]
    assert [len(row["ObjectEntities"]) for row in selected["awardWonBy"]] == [9, 19, 29]
    for relation in fs.NUMERIC_RELATIONS:
        assert [row["ObjectEntities"] for row in selected[relation]] == [["0.1"], ["50"], ["1000"]]
    assert "ALIAS_NOT_A_CANONICAL_LABEL" not in json.dumps(selected)
    assert "NUMERIC_ALIAS_NOT_USED" not in json.dumps(selected)


def test_two_cardinality_buckets_prefer_nonempty_filler(training):
    directory, rows = training
    for row in rows:
        if row["Relation"] == "personHasCityOfDeath" and row["ObjectEntities"]:
            row["ObjectEntities"] = [["Paris", "alias"]]
    write_rows(directory / "train.jsonl", rows)
    demos = fs.select_demonstrations(directory)["personHasCityOfDeath"]
    assert [len(row["ObjectEntities"]) for row in demos] == [0, 1, 1]
    assert [row["SubjectEntity"] for row in demos[1:]] == ["personHasCityOfDeath-1", "personHasCityOfDeath-2"]


def test_exhausted_cardinality_bucket_uses_nearest_remaining_example(training):
    directory, rows = training
    relation = "personHasCityOfDeath"
    rows = [row for row in rows if row["Relation"] != relation]
    rows += [record("Empty A", relation, 0), record("Empty B", relation, 0), record("Nonempty", relation, 1)]
    write_rows(directory / "train.jsonl", rows)
    demos = fs.select_demonstrations(directory)[relation]
    assert [row["SubjectEntity"] for row in demos] == ["Empty A", "Nonempty", "Empty B"]


def test_subjects_and_numeric_spelling_are_preserved(training):
    directory, rows = training
    rows = [row for row in rows if row["Relation"] != "hasArea"]
    subject = '  Åland "quoted"  '
    for name, value in ((subject, "-0.0"), ("Middle", "1.2300"), ("Largest", "1e3")):
        row = record(name, "hasArea", 1)
        row["ObjectEntities"] = [[value, "ignored alias"]]
        rows.append(row)
    write_rows(directory / "train.jsonl", rows)
    demos = fs.select_demonstrations(directory)
    assert [row["ObjectEntities"] for row in demos["hasArea"]] == [["-0.0"], ["1.2300"], ["1e3"]]
    assert demos["hasArea"][0]["SubjectEntity"] == subject
    messages = fs.build_messages("Query", "hasArea", demos)
    assert messages[1] == zero_shot.build_messages(subject, "hasArea")[1]


def test_numeric_empty_example_and_equal_value_ties(training):
    directory, rows = training
    numeric = [row for row in rows if row["Relation"] == "hasArea"]
    numeric[0]["ObjectEntities"] = []
    numeric[1]["ObjectEntities"] = [["1e2"]]
    numeric[2]["ObjectEntities"] = [["100.00"]]
    write_rows(directory / "train.jsonl", rows)
    demos = fs.select_demonstrations(directory)["hasArea"]
    assert [row["ObjectEntities"] for row in demos] == [[], ["50"], ["100"]]


def test_caps_exclude_whole_answers_without_truncation(training):
    directory, rows = training
    row = record("Byte-heavy", "awardWonBy", 1)
    row["ObjectEntities"] = [["Å" * 600]]
    rows.append(row)
    write_rows(directory / "train.jsonl", rows)
    selected = fs.select_demonstrations(directory)["awardWonBy"]
    assert [len(row["ObjectEntities"]) for row in selected] == [9, 19, 29]
    assert all(len(json.dumps(row["ObjectEntities"], ensure_ascii=False).encode("utf-8")) <= fs.MAX_DEMO_ANSWER_BYTES for row in selected)


@pytest.mark.parametrize("problem", ["missing_relation", "too_few", "duplicate", "unknown_relation", "invalid_number", "multiple_numbers"])
def test_invalid_training_fails_safely(training, problem):
    directory, rows = training
    if problem == "missing_relation":
        rows = [row for row in rows if row["Relation"] != "awardWonBy"]
    elif problem == "too_few":
        rows = [row for row in rows if row["Relation"] != "awardWonBy" or row["SubjectEntity"].endswith(("-0", "-1"))]
    elif problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif problem == "unknown_relation":
        rows.append(record("Unknown", "unsupported", 1))
    else:
        row = next(row for row in rows if row["Relation"] == "hasArea")
        row["ObjectEntities"] = [["NaN"]] if problem == "invalid_number" else [["1"], ["2"]]
    write_rows(directory / "train.jsonl", rows)
    with pytest.raises(fs.FewShotError):
        fs.select_demonstrations(directory)


def test_selection_never_reads_validation_or_test_labels(training):
    directory, _ = training
    expected = fs.select_demonstrations(directory)
    (directory / "val.jsonl").write_text("not JSON", encoding="utf-8")
    (directory / "test.jsonl").write_text("not JSON either", encoding="utf-8")
    assert fs.select_demonstrations(directory) == expected


def test_training_hardlink_to_validation_is_rejected(training):
    directory, _ = training
    (directory / "val.jsonl").hardlink_to(directory / "train.jsonl")
    with pytest.raises(fs.FewShotError, match="validation/test link"):
        fs.select_demonstrations(directory)


def test_prompts_use_same_system_questions_and_three_flat_answers(training):
    directory, _ = training
    demos = fs.select_demonstrations(directory)
    for relation in zero_shot.QUESTIONS:
        messages = fs.build_messages('New "subject"', relation, demos)
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
        assert messages[0] == zero_shot.build_messages('New "subject"', relation)[0]
        assert messages[-1] == zero_shot.build_messages('New "subject"', relation)[1]
        for index, demo in enumerate(demos[relation]):
            assert messages[1 + 2 * index] == zero_shot.build_messages(demo["SubjectEntity"], relation)[1]
            assert zero_shot.parse_response(messages[2 + 2 * index]["content"]) == demo["ObjectEntities"]


@pytest.mark.parametrize("variant", ["exact", "case_space", "compatibility"])
def test_pair_leakage_is_rejected_without_reselection(training, variant):
    directory, _ = training
    demos = fs.select_demonstrations(directory)
    example = demos["hasArea"][0]
    subject = example["SubjectEntity"]
    if variant == "case_space":
        subject = "  " + subject.upper() + "  "
    elif variant == "compatibility":
        subject = "".join(chr(ord(char) + 0xFEE0) if "!" <= char <= "~" else char for char in subject)
    with pytest.raises(fs.FewShotError, match="leakage"):
        fs.build_messages(subject, "hasArea", demos)


def test_cross_relation_subject_overlap_is_reported(training):
    directory, _ = training
    demos = fs.select_demonstrations(directory)
    subject = demos["awardWonBy"][0]["SubjectEntity"]
    report = fs.check_leakage(demos, [(subject, "hasArea")])
    assert report["subject_only_overlaps"] == [{
        "demonstration_subject": subject, "demonstration_relation": "awardWonBy",
        "input_subject": subject, "input_relation": "hasArea",
    }]
    assert len(fs.build_messages(subject, "hasArea", demos)) == 8


@pytest.mark.parametrize("problem", ["count", "relation", "schema", "duplicates", "missing_group"])
def test_invalid_demonstration_manifests_are_rejected(training, problem):
    directory, _ = training
    demos = fs.select_demonstrations(directory)
    if problem == "count": demos["hasArea"].pop()
    elif problem == "relation": demos["hasArea"][0]["Relation"] = "hasCapacity"
    elif problem == "schema": demos["hasArea"][0]["ObjectEntities"] = [["nested"]]
    elif problem == "duplicates": demos["hasArea"][1] = demos["hasArea"][0]
    else: del demos["hasArea"]
    with pytest.raises((fs.FewShotError, DatasetError)):
        fs.build_messages("Query", "hasArea", demos)


class FakeTokenizer:
    def get_chat_template(self):
        return "fake-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "truncation": False}
        assert len(messages) == 8
        return list(range(37))


def test_context_limit_includes_generation_budget_and_never_truncates(training):
    directory, _ = training
    messages = fs.build_messages("Query", "awardWonBy", fs.select_demonstrations(directory))
    assert fs.check_context(messages, FakeTokenizer(), context_limit=549) == 37
    with pytest.raises(fs.FewShotError, match="context window"):
        fs.check_context(messages, FakeTokenizer(), context_limit=548)
    tokenizer = FakeTokenizer()
    tokenizer.model_max_length = 548
    with pytest.raises(fs.FewShotError, match="context window"):
        fs.check_context(messages, tokenizer, context_limit=32768)


def test_preparation_ignores_evaluation_labels_and_preserves_inputs(training):
    directory, rows = training
    inputs = directory / "val.jsonl"
    query = record("Query", "hasArea", 1)
    query["ObjectEntities"] = "EVALUATION_GOLD_MUST_NOT_BE_USED"
    write_rows(inputs, [query])
    original = (directory / "train.jsonl").read_bytes(), inputs.read_bytes()
    prepared = fs.prepare_prompts(directory, inputs)
    assert prepared["prompts"][0]["input_tokens"] is None
    assert prepared["tokenizer_checked"] is False
    assert prepared["prompt_version"] == "lm-kbc-2026-few-shot-v1"
    assert prepared["max_new_tokens"] == 512
    assert prepared["train_sha256"] == hashlib.sha256(original[0]).hexdigest()
    assert (directory / "train.jsonl").read_bytes() == original[0]
    assert inputs.read_bytes() == original[1]
    assert "EVALUATION_GOLD_MUST_NOT_BE_USED" not in json.dumps(prepared)
    query["ObjectEntities"] = {"different": "invalid gold shape is still unused"}
    write_rows(inputs, [query])
    changed = fs.prepare_prompts(directory, inputs, tokenizer=FakeTokenizer())
    assert changed["demonstrations"] == prepared["demonstrations"]
    assert changed["prompts"][0]["messages"] == prepared["prompts"][0]["messages"]
    assert changed["prompts"][0]["input_tokens"] == 37


@pytest.mark.parametrize("queries", [[], [record("Query", "hasArea", 0)] * 2, [record("Query", "unsupported", 0)]])
def test_empty_duplicate_or_unsupported_inputs_fail(training, queries):
    directory, _ = training
    path = directory / "val.jsonl"
    write_rows(path, queries)
    with pytest.raises(fs.FewShotError):
        fs.prepare_prompts(directory, path)


@pytest.mark.parametrize("text", ['{broken}\n', '[]\n', '{"SubjectEntity": 1, "Relation": "hasArea"}\n'])
def test_malformed_input_keys_have_line_context(training, text):
    directory, _ = training
    path = directory / "val.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DatasetError, match=":1:"):
        fs.prepare_prompts(directory, path)
