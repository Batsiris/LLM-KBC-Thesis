"""Training-only selection and prompt regression checks; no inference or downloads."""

import copy
import hashlib
import json
import random

import pytest

from llm_kbc import few_shot as v1, few_shot_v2 as fs, zero_shot
from llm_kbc.dataset_inspector import DatasetError
from llm_kbc.prediction_io import validate_prediction


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def training(tmp_path):
    rows = []
    for relation in zero_shot.QUESTIONS:
        counts = (0, 3, 9, 19, 27, 29, 47) if relation == "awardWonBy" else (0, 1, 2, 4, 1, 3, 2)
        for i, count in enumerate(counts):
            objects = [[f"Object {j}", "IGNORED_ALIAS"] for j in range(count)]
            if relation in v1.NUMERIC_RELATIONS:
                value = (None, "1000", "0.1000", "1.00", "100", "50", "2.00")[i]
                objects = [] if value is None else [[value, "IGNORED_ALIAS"]]
            rows.append({"SubjectEntity": f"{relation}-{i}", "Relation": relation, "ObjectEntities": objects})
    write_rows(tmp_path / "train.jsonl", rows)
    return tmp_path, rows


@pytest.mark.parametrize("shots", [1, 5])
def test_selection_deterministic_independent_of_train_order_and_complete(training, shots):
    directory, rows = training
    expected = fs.select_demonstrations(directory, shots=shots)
    random.Random(73).shuffle(rows)
    write_rows(directory / "train.jsonl", rows)
    assert fs.select_demonstrations(directory, shots=shots) == expected
    assert list(expected) == list(zero_shot.QUESTIONS)
    for relation, demos in expected.items():
        assert len(demos) == shots
        assert len({(v1._subject_key(r["SubjectEntity"]), r["Relation"]) for r in demos}) == shots
        assert all(validate_prediction(row) == row and row["Relation"] == relation for row in demos)
    assert "IGNORED_ALIAS" not in json.dumps(expected)


def test_one_shot_nonempty_lower_medians_preserve_numeric_strings(training):
    selected = fs.select_demonstrations(training[0], shots=1)
    assert selected["countryLandBordersCountry"][0]["SubjectEntity"] == "countryLandBordersCountry-2"
    assert len(selected["awardWonBy"][0]["ObjectEntities"]) == 19
    for relation in v1.NUMERIC_RELATIONS:
        assert selected[relation][0]["ObjectEntities"] == ["2.00"]


def test_one_shot_all_empty_fallback_and_minimum_pool_size(training):
    directory, rows = training
    for row in rows:
        row["ObjectEntities"] = []
    write_rows(directory / "train.jsonl", rows)
    selected = fs.select_demonstrations(directory, shots=1)
    assert all(demos[0]["SubjectEntity"].endswith("-0") and demos[0]["ObjectEntities"] == [] for demos in selected.values())
    write_rows(directory / "train.jsonl", rows[::7])
    assert fs.select_demonstrations(directory, shots=1) == selected


def test_five_shot_exact_v1_prefix_and_quartile_additions(training):
    directory, _ = training
    baseline = v1.select_demonstrations(directory)
    selected = fs.select_demonstrations(directory, shots=5)
    for relation in zero_shot.QUESTIONS:
        assert selected[relation][:3] == baseline[relation]
        old = v1.build_messages('Query "quoted"', relation, baseline)
        new = fs.build_messages('Query "quoted"', relation, selected, shots=5)
        assert new[:7] == old[:7]  # System and all three demonstration turn pairs.
        assert new[-1] == old[-1]
    assert [r["SubjectEntity"] for r in selected["countryLandBordersCountry"][3:]] == [
        "countryLandBordersCountry-4", "countryLandBordersCountry-6",
    ]
    for relation in v1.NUMERIC_RELATIONS:
        assert [r["ObjectEntities"] for r in selected[relation][3:]] == [["1.00"], ["50"]]


def test_numeric_equal_value_subject_ties_and_empty_fallback(training):
    directory, rows = training
    for row in rows:
        if row["Relation"] == "hasArea":
            row["ObjectEntities"] = [["1.00"]] if row["SubjectEntity"].endswith(("-1", "-2")) else []
    write_rows(directory / "train.jsonl", rows)
    selected = fs.select_demonstrations(directory, shots=5)["hasArea"]
    assert [r["SubjectEntity"] for r in selected] == [f"hasArea-{i}" for i in (0, 1, 2, 3, 4)]


@pytest.mark.parametrize("shots", [1, 5])
def test_caps_exclude_whole_answers_without_truncation(training, shots):
    directory, rows = training
    rows += [
        {"SubjectEntity": "Byte-heavy", "Relation": "awardWonBy", "ObjectEntities": [["\u00c5" * 600]]},
    ]
    write_rows(directory / "train.jsonl", rows)
    selected = fs.select_demonstrations(directory, shots=shots)
    assert all(r["SubjectEntity"] not in ("Byte-heavy", "awardWonBy-6") for r in selected["awardWonBy"])
    assert all(len(r["ObjectEntities"]) <= 32 and len(json.dumps(r["ObjectEntities"], ensure_ascii=False).encode()) <= 1024
               for demos in selected.values() for r in demos)


@pytest.mark.parametrize("shots", [1, 5])
@pytest.mark.parametrize("problem", ["insufficient", "duplicate", "relation", "numeric", "multiple_numeric", "aliases", "malformed"])
def test_invalid_training_rejected(training, shots, problem):
    directory, rows = training
    if problem == "insufficient":
        rows = [r for r in rows if r["Relation"] != "hasArea"]
    elif problem == "duplicate":
        row = copy.deepcopy(rows[0])
        row["SubjectEntity"] = "  " + row["SubjectEntity"].upper() + "  "
        rows.append(row)
    elif problem == "relation":
        rows[0]["Relation"] = "unsupported"
    elif problem in ("numeric", "multiple_numeric"):
        row = next(r for r in rows if r["Relation"] == "hasArea")
        row["ObjectEntities"] = [["Infinity"]] if problem == "numeric" else [["1"], ["2"]]
    elif problem == "aliases":
        rows[0]["ObjectEntities"] = [[]]
    write_rows(directory / "train.jsonl", rows)
    if problem == "malformed":
        (directory / "train.jsonl").write_text("{bad}\n")
    with pytest.raises((fs.FewShotError, DatasetError)):
        fs.select_demonstrations(directory, shots=shots)


def test_five_requires_five_eligible_examples(training):
    directory, rows = training
    rows = [r for r in rows if r["Relation"] != "awardWonBy" or r["SubjectEntity"].endswith(("-0", "-1", "-2", "-3"))]
    write_rows(directory / "train.jsonl", rows)
    with pytest.raises(fs.FewShotError, match="fewer than 5"):
        fs.select_demonstrations(directory, shots=5)


@pytest.mark.parametrize("shots", [1, 5])
def test_selection_ignores_evaluation_files_and_rejects_training_link(training, shots):
    directory, _ = training
    expected = fs.select_demonstrations(directory, shots=shots)
    (directory / "val.jsonl").write_text("NOT JSON")
    (directory / "test.jsonl").write_text("NOT JSON")
    assert fs.select_demonstrations(directory, shots=shots) == expected
    (directory / "val.jsonl").unlink()
    (directory / "val.jsonl").hardlink_to(directory / "train.jsonl")
    with pytest.raises(fs.FewShotError, match="validation/test link"):
        fs.select_demonstrations(directory, shots=shots)


@pytest.mark.parametrize("shots", [1, 5])
def test_prompt_roles_schema_unicode_and_unchanged_questions(training, shots):
    directory, rows = training
    for row in rows:
        if row["Relation"] not in v1.NUMERIC_RELATIONS:
            row["ObjectEntities"] = [["\u00c5land", 'Name "quoted"'] for _ in row["ObjectEntities"]]
    write_rows(directory / "train.jsonl", rows)
    demos = fs.select_demonstrations(directory, shots=shots)
    for relation in zero_shot.QUESTIONS:
        messages = fs.build_messages('New "subject"', relation, demos, shots=shots)
        query = zero_shot.build_messages('New "subject"', relation)
        assert [m["role"] for m in messages] == ["system"] + ["user", "assistant"] * shots + ["user"]
        assert messages[0] == query[0] and messages[-1] == query[1]
        for i, demo in enumerate(demos[relation]):
            assert messages[1 + i * 2] == zero_shot.build_messages(demo["SubjectEntity"], relation)[1]
            assert zero_shot.parse_response(messages[2 + i * 2]["content"]) == demo["ObjectEntities"]


@pytest.mark.parametrize("shots", [1, 5])
@pytest.mark.parametrize("variant", ["case_space", "NFKC"])
def test_pair_leakage_rejected_without_reselection(training, shots, variant):
    demos = fs.select_demonstrations(training[0], shots=shots)
    original = copy.deepcopy(demos)
    subject = demos["hasArea"][-1]["SubjectEntity"]
    subject = "  " + subject.upper() + "  " if variant == "case_space" else "".join(
        chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in subject
    )
    with pytest.raises(fs.FewShotError, match="leakage"):
        fs.build_messages(subject, "hasArea", demos, shots=shots)
    assert demos == original


@pytest.mark.parametrize("shots", [1, 5])
def test_subject_only_overlap_and_duplicate_evaluation_ids(training, shots):
    demos = fs.select_demonstrations(training[0], shots=shots)
    subject = demos["awardWonBy"][0]["SubjectEntity"]
    report = fs.check_leakage(demos, [(subject, "hasArea")], shots=shots)
    assert report["subject_only_overlaps"] == [{
        "demonstration_subject": subject, "demonstration_relation": "awardWonBy",
        "input_subject": subject, "input_relation": "hasArea",
    }]
    with pytest.raises(fs.FewShotError, match="unique"):
        fs.check_leakage(demos, [("Query", "hasArea"), ("  QUERY ", "hasArea")], shots=shots)


@pytest.mark.parametrize("problem", ["count", "duplicate", "group", "relation", "schema"])
def test_invalid_demo_manifest(training, problem):
    demos = fs.select_demonstrations(training[0], shots=5)
    if problem == "count": demos["hasArea"].pop()
    elif problem == "duplicate": demos["hasArea"][4] = demos["hasArea"][0]
    elif problem == "group": del demos["hasArea"]
    elif problem == "relation": demos["hasArea"][0]["Relation"] = "hasCapacity"
    else: demos["hasArea"][0]["ObjectEntities"] = [["nested"]]
    with pytest.raises((fs.FewShotError, DatasetError)):
        fs.check_leakage(demos, [], shots=5)


class FakeTokenizer:
    model_max_length = 32768
    name_or_path = zero_shot.DEFAULT_MODEL

    def get_chat_template(self):
        return "fake-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "truncation": False}
        assert len(messages) in (4, 12)
        return list(range(37))


@pytest.mark.parametrize("shots", [1, 5])
def test_preparation_labels_unused_hashes_and_exact_context(training, shots):
    directory, _ = training
    inputs = directory / "val.jsonl"
    write_rows(inputs, [{"SubjectEntity": "Query", "Relation": "hasArea", "ObjectEntities": {"SECRET_GOLD": "invalid"}}])
    original = inputs.read_bytes(), (directory / "train.jsonl").read_bytes()
    prepared = fs.prepare_prompts(directory, inputs, shots=shots, tokenizer=FakeTokenizer(), context_limit=549)
    assert prepared["shots"] == shots and prepared["tokenizer_checked"]
    assert prepared["prompts"][0]["input_tokens"] == 37
    assert prepared["prompt_version"] == fs.PROMPT_VERSION
    assert prepared["selection_version"] == fs.SELECTION_VERSION
    assert prepared["train_sha256"] == hashlib.sha256(original[1]).hexdigest()
    assert prepared["inputs_sha256"] == hashlib.sha256(original[0]).hexdigest()
    assert prepared["chat_template_sha256"] == hashlib.sha256(b"fake-template").hexdigest()
    assert "SECRET_GOLD" not in json.dumps(prepared)
    assert original == (inputs.read_bytes(), (directory / "train.jsonl").read_bytes())
    with pytest.raises(fs.FewShotError, match="context window"):
        fs.prepare_prompts(directory, inputs, shots=shots, tokenizer=FakeTokenizer(), context_limit=548)
    assert fs.prepare_prompts(directory, inputs, shots=shots)["prompts"][0]["input_tokens"] is None


@pytest.mark.parametrize("text", ["", "{bad}\n", "[]\n", '{"SubjectEntity":1,"Relation":"hasArea"}\n',
                                     '{"SubjectEntity":"Query","Relation":"unknown"}\n'])
def test_invalid_evaluation_inputs(training, text):
    directory, _ = training
    path = directory / "val.jsonl"
    path.write_text(text)
    with pytest.raises((DatasetError, fs.FewShotError)):
        fs.prepare_prompts(directory, path, shots=1)


@pytest.mark.parametrize("shots", [0, 3, 6, True, "1", None])
def test_only_one_or_five_shots(training, shots):
    with pytest.raises(fs.FewShotError, match="1 or 5"):
        fs.select_demonstrations(training[0], shots=shots)
