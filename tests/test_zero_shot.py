"""Zero-shot pipeline tests using fake generation; no weights, network, or GPU."""

import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_kbc import zero_shot as zs
from llm_kbc.dataset_inspector import DatasetError
from llm_kbc.prediction_io import load_predictions, validate_prediction


def write_dataset(tmp_path, relations=("hasArea", "awardWonBy")):
    path = tmp_path / "val.jsonl"
    rows = [
        {"SubjectEntity": f"Subject {index}", "Relation": relation,
         "ObjectEntities": [["SECRET_GOLD_ANSWER"]]}
        for index, relation in enumerate(relations)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def fake_generator(monkeypatch, responses):
    """Replace the one dependency-loading boundary with a deterministic fake."""
    calls = []
    responses = iter(responses)

    def load(model_id, revision, max_new_tokens):
        calls.append((model_id, revision, max_new_tokens))

        def generate(messages):
            calls.append(messages)
            response = next(responses)
            if isinstance(response, BaseException):
                raise response
            return response

        return generate, {
            "model_id": model_id, "requested_revision": revision,
            "resolved_revision": "a" * 40,
            "generation": {"do_sample": False, "max_new_tokens": max_new_tokens},
            "versions": {"torch": "fake", "transformers": "fake"},
        }

    monkeypatch.setattr(zs, "_load_generator", load)
    return calls


@pytest.mark.parametrize("relation", zs.QUESTIONS)
def test_prompt_covers_all_six_relations_without_gold_or_examples(relation):
    messages = zs.build_messages('A "quoted" subject', relation)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert json.dumps('A "quoted" subject') in messages[1]["content"]
    assert "JSON array of strings" in messages[0]["content"]
    assert "SECRET_GOLD_ANSWER" not in json.dumps(messages)
    assert "example" not in json.dumps(messages).lower()


def test_short_v2_question_templates():
    assert zs.PROMPT_VERSION == "lm-kbc-2026-zero-shot-v2"
    assert zs.QUESTIONS == {
        "countryLandBordersCountry": "Which countries share a land border with {subject}?",
        "personHasCityOfDeath": "In which city did {subject} die?",
        "hasCapacity": "What is the spectator capacity of {subject}?",
        "awardWonBy": "Who has received the award {subject}?",
        "companyTradesAtStockExchange": "On which stock exchanges is {subject} listed?",
        "hasArea": "What is the total area of {subject} in square kilometres?",
    }


@pytest.mark.parametrize(("text", "expected"), [
    (' ["Éva", "123"] \n', ["Éva", "123"]),
    ('```json\n["x"]\n```', ["x"]),
    ('```\n[]\n```', []),
    ('["x", "x", "", " "]', ["x", "x", "", " "]),
])
def test_parse_valid_responses_without_normalization(text, expected):
    assert zs.parse_response(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ('[104.4]', ["104.4"]),
    ('[418]', ["418"]),
    ('[44.8]', ["44.8"]),
    ('["name", 104.4, "418", 418, 44.8]', ["name", "104.4", "418", "418", "44.8"]),
    ('[-12, 0, -0, -0.0, 1.2300, 1e3, 4.2E-7]', ["-12", "0", "-0", "-0.0", "1.2300", "1e3", "4.2E-7"]),
    ('[9007199254740993, 0.12345678901234567890123456789]', ["9007199254740993", "0.12345678901234567890123456789"]),
    ('```json\n[104.4, "name"]\n```', ["104.4", "name"]),
    ('["NaN", "Infinity"]', ["NaN", "Infinity"]),
])
def test_numeric_tokens_become_strings_without_value_or_order_changes(text, expected):
    assert zs.parse_response(text) == expected


@pytest.mark.parametrize("text", [
    "", "I do not know", 'Here is the answer: ["x"]', '["x"] trailing text',
    '["x"] ["y"]', '["truncated"', '{"objects":["x"]}', '[["x"]]',
    "[true]", "[false]", "[null]", "null", "[NaN]", "[Infinity]", "[-Infinity]",
    '[1, true]', '[1, null]', '[1, [2]]', '[1, {"value": 2}]', '[1, NaN]',
    '[1, Infinity]', '[1, -Infinity]', '```json\n[false]\n```',
    '[104.4,]', '[01]', '[+1]', '[.4]', '[1.]', '[1e]', 'Answer: [104.4]',
    '24522\u27e9',
])
def test_parse_failure_never_becomes_empty(text):
    with pytest.raises(zs.InferenceError):
        zs.parse_response(text)


def test_successful_run_is_complete_validated_and_reproducible(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)
    before = dataset.read_bytes()
    output = tmp_path / "results" / "predictions.jsonl"
    calls = fake_generator(monkeypatch, ['["123"]', '```json\n[]\n```'])
    metadata = zs.run_inference(dataset, output, max_new_tokens=64)
    assert list(load_predictions(output)) == [
        {"SubjectEntity": "Subject 0", "Relation": "hasArea", "ObjectEntities": ["123"]},
        {"SubjectEntity": "Subject 1", "Relation": "awardWonBy", "ObjectEntities": []},
    ]
    assert calls[0] == (zs.DEFAULT_MODEL, zs.DEFAULT_REVISION, 64)
    assert "SECRET_GOLD_ANSWER" not in json.dumps(calls)
    assert dataset.read_bytes() == before
    assert metadata["status"] == "complete"
    assert metadata["valid_records"] == metadata["processed_records"] == 2
    assert metadata["parse_failures"] == 0
    assert metadata["total_inputs"] == 2
    assert metadata["format_validity_rate"] == 1.0
    assert metadata["parse_failure_policy"] == "parse-failure-as-abstention-v1"
    assert metadata["numeric_normalized_responses"] == 0
    assert metadata["prompt_version"] == zs.PROMPT_VERSION
    assert metadata["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert metadata["dataset"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert metadata == json.loads(Path(str(output) + ".metadata.json").read_text())
    assert Path(str(output) + ".failures.jsonl").read_text() == ""


def test_smoke_limit_preserves_order_and_custom_revision(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path, ("hasArea",) * 6)
    calls = fake_generator(monkeypatch, ["[]"] * 5)
    output = tmp_path / "smoke.jsonl"
    metadata = zs.run_inference(
        dataset, output, model_id="other/model", revision="pinned-branch", max_examples=5,
    )
    assert calls[0] == ("other/model", "pinned-branch", 512)
    assert [row["SubjectEntity"] for row in load_predictions(output)] == [f"Subject {i}" for i in range(5)]
    assert metadata["selection"] == {
        "strategy": "first_records_in_file_order", "max_examples": 5, "records": 5,
    }


def test_estonia_failure_abstains_with_complete_ordered_coverage_and_raw_log(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path, ("hasArea",) * 5)
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    rows[1]["SubjectEntity"] = "Estonia"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    raw_path = Path(str(output) + ".raw.jsonl")
    responses = ['  [104.4]\n', '24522\u27e9', '```json\n[418]\n```', '[]', '[44.8]']
    calls = []

    def generate(messages):
        # Previous responses must be flushed while generation is still running.
        logged = [json.loads(line) for line in raw_path.read_text().splitlines()]
        assert [row["response"] for row in logged] == responses[:len(calls)]
        assert not output.exists()
        calls.append(messages)
        return responses[len(calls) - 1]

    monkeypatch.setattr(zs, "_load_generator", lambda *args: (generate, {}))
    metadata = zs.run_inference(dataset, output)
    predictions = list(load_predictions(output))
    assert [(row["SubjectEntity"], row["Relation"]) for row in predictions] == [
        (row["SubjectEntity"], row["Relation"]) for row in rows
    ]
    assert [row["ObjectEntities"] for row in predictions] == [["104.4"], [], ["418"], [], ["44.8"]]
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert [row["response"] for row in raw] == responses
    assert [row["input_line"] for row in raw] == [1, 2, 3, 4, 5]
    assert [(row["SubjectEntity"], row["Relation"]) for row in raw] == [
        (row["SubjectEntity"], row["Relation"]) for row in rows
    ]
    assert [row["parsing_status"] for row in raw] == ["valid", "parse_failed", "valid", "valid", "valid"]
    assert all(row["error"] is None for row in raw if row["parsing_status"] == "valid")
    assert raw[1]["error"].startswith("Response is not a complete JSON array")
    failures = [json.loads(line) for line in Path(str(output) + ".failures.jsonl").read_text().splitlines()]
    assert len(failures) == 1
    assert failures[0] == {key: value for key, value in raw[1].items() if key != "parsing_status"}
    assert metadata["status"] == "complete"
    assert metadata["total_inputs"] == metadata["processed_records"] == 5
    assert metadata["valid_records"] == 4
    assert metadata["parse_failures"] == 1
    assert metadata["format_validity_rate"] == 0.8
    assert metadata["numeric_normalized_responses"] == 3
    assert metadata == json.loads(Path(str(output) + ".metadata.json").read_text())


def test_normalization_metadata_counts_responses_not_elements(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path, ("hasArea",) * 4)
    output = tmp_path / "predictions.jsonl"
    fake_generator(monkeypatch, ['[104.4, 418, "44.8"]', '[44.8]', '["418"]', '[]'])
    metadata = zs.run_inference(dataset, output)
    assert [row["ObjectEntities"] for row in load_predictions(output)] == [
        ["104.4", "418", "44.8"], ["44.8"], ["418"], [],
    ]
    assert metadata["numeric_normalized_responses"] == 2
    assert metadata["valid_records"] == 4
    assert metadata["parse_failures"] == 0
    assert json.loads(Path(str(output) + ".metadata.json").read_text())["numeric_normalized_responses"] == 2


def test_normalization_count_is_preserved_when_another_response_fails(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    raw_failure = '[44.8, false]'
    fake_generator(monkeypatch, ['[104.4]', raw_failure])
    zs.run_inference(dataset, output)
    metadata = json.loads(Path(str(output) + ".metadata.json").read_text())
    assert metadata["numeric_normalized_responses"] == 1
    assert metadata["valid_records"] == metadata["parse_failures"] == 1
    assert json.loads(Path(str(output) + ".failures.jsonl").read_text())["response"] == raw_failure
    assert [row["ObjectEntities"] for row in load_predictions(output)] == [["104.4"], []]


@pytest.mark.parametrize("first_response", ["[]", '24522\u27e9'])
@pytest.mark.parametrize("error", [RuntimeError("fake out of memory"), StopIteration("incomplete"), KeyboardInterrupt()])
def test_interrupted_generation_preserves_raw_log_without_publishing(tmp_path, monkeypatch, first_response, error):
    dataset = write_dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    fake_generator(monkeypatch, [first_response, error])
    with pytest.raises(type(error)):
        zs.run_inference(dataset, output)
    assert not output.exists()
    metadata = json.loads(Path(str(output) + ".metadata.json").read_text())
    assert metadata["status"] == ("interrupted" if isinstance(error, KeyboardInterrupt) else "runtime_error")
    assert metadata["processed_records"] == 1
    assert metadata["total_inputs"] == 2
    assert metadata["format_validity_rate"] == (1.0 if first_response == "[]" else 0.0)
    raw = [json.loads(line) for line in Path(str(output) + ".raw.jsonl").read_text().splitlines()]
    assert len(raw) == 1
    assert raw[0]["response"] == first_response


def test_unexpected_parser_interruption_still_preserves_raw_response(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    fake_generator(monkeypatch, ['[104.4]'])

    def interrupt(response):
        raise RuntimeError("unexpected parser failure")

    monkeypatch.setattr(zs, "_parse_response", interrupt)
    with pytest.raises(RuntimeError, match="unexpected parser failure"):
        zs.run_inference(dataset, output)
    assert not output.exists()
    raw = json.loads(Path(str(output) + ".raw.jsonl").read_text())
    assert raw["response"] == '[104.4]'
    assert raw["parsing_status"] == "interrupted"
    assert raw["error"] == "unexpected parser failure"


@pytest.mark.parametrize("corruption", ["schema", "identity", "duplicate"])
def test_final_validation_rejects_invalid_predictions_before_publication(tmp_path, monkeypatch, corruption):
    dataset = write_dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    fake_generator(monkeypatch, ['[104.4]', '[418]'])
    original_validate = zs.validate_prediction

    def corrupt(row, *, location):
        validated = original_validate(row, location=location)
        if location == "generated record 2":
            if corruption == "schema":
                validated["ObjectEntities"] = [["nested"]]
            elif corruption == "identity":
                validated["SubjectEntity"] = "wrong subject"
            else:
                validated.update(SubjectEntity="Subject 0", Relation="hasArea")
        return validated

    monkeypatch.setattr(zs, "validate_prediction", corrupt)
    with pytest.raises((DatasetError, zs.InferenceError)):
        zs.run_inference(dataset, output)
    assert not output.exists()


@pytest.mark.parametrize("gold_objects, expected_f1", [([], 1.0), ([["104.4"]], 0.0)])
def test_malformed_generation_abstention_uses_official_empty_case_scoring(tmp_path, monkeypatch, gold_objects, expected_f1):
    from llm_kbc.evaluation.lm_kbc_2026 import LMKBC2026Evaluator

    dataset = write_dataset(tmp_path, ("hasArea",))
    row = json.loads(dataset.read_text())
    row["ObjectEntities"] = gold_objects
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    fake_generator(monkeypatch, ['24522\u27e9'])
    metadata = zs.run_inference(dataset, output)
    assert metadata["valid_records"] == 0
    assert metadata["parse_failures"] == 1
    assert metadata["format_validity_rate"] == 0.0
    result = LMKBC2026Evaluator().evaluate(data_dir=tmp_path, split="val", predictions_path=output)
    assert result.overall["macro-p"] == result.overall["micro-p"] == 1.0
    assert result.overall["macro-r"] == result.overall["micro-r"] == expected_f1
    assert result.overall["macro-f1"] == result.overall["micro-f1"] == expected_f1


def test_failed_publication_leaves_no_partial_output_or_temporary_file(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path, ("hasArea",))
    output = tmp_path / "predictions.jsonl"
    fake_generator(monkeypatch, ['["1"]'])

    def fail_publication(source, destination):
        assert not output.exists()
        assert len(list(load_predictions(source))) == 1
        raise OSError("fake publish failure")

    monkeypatch.setattr(zs.os, "replace", fail_publication)
    with pytest.raises(OSError, match="fake publish failure"):
        zs.run_inference(dataset, output)
    assert not output.exists()
    assert {path.name for path in tmp_path.iterdir()} == {
        "val.jsonl", "predictions.jsonl.metadata.json", "predictions.jsonl.failures.jsonl",
        "predictions.jsonl.raw.jsonl",
    }
    metadata = json.loads(Path(str(output) + ".metadata.json").read_text())
    assert metadata["status"] == "runtime_error"


def test_duplicate_input_pairs_are_rejected_before_loading_model(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path, ("hasArea", "hasArea"))
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    rows[1]["SubjectEntity"] = rows[0]["SubjectEntity"]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    calls = fake_generator(monkeypatch, [])
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(zs.InferenceError, match="duplicate subject-relation pairs"):
        zs.run_inference(dataset, output)
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize("suffix", ["", ".metadata.json", ".failures.jsonl", ".raw.jsonl"])
def test_existing_artifacts_are_never_overwritten(tmp_path, monkeypatch, suffix):
    dataset = write_dataset(tmp_path)
    calls = fake_generator(monkeypatch, [])
    output = tmp_path / "predictions.jsonl"
    protected = Path(str(output) + suffix)
    protected.write_text("keep this", encoding="utf-8")
    with pytest.raises(zs.InferenceError, match="Refusing to overwrite"):
        zs.run_inference(dataset, output)
    assert protected.read_text() == "keep this"
    assert calls == []


@pytest.mark.parametrize("kwargs", [{"max_examples": 0}, {"max_examples": -1}, {"max_new_tokens": 0}])
def test_invalid_limits_fail_before_loading_a_model(tmp_path, monkeypatch, kwargs):
    calls = fake_generator(monkeypatch, [])
    with pytest.raises(zs.InferenceError, match="positive"):
        zs.run_inference(tmp_path / "missing", tmp_path / "output", **kwargs)
    assert calls == []


def test_invalid_inputs_fail_before_loading_a_model(tmp_path, monkeypatch):
    calls = fake_generator(monkeypatch, [])
    dataset = tmp_path / "missing.jsonl"
    with pytest.raises(DatasetError, match="file not found"):
        zs.run_inference(dataset, tmp_path / "output")
    dataset.write_text("", encoding="utf-8")
    with pytest.raises(zs.InferenceError, match="no records"):
        zs.run_inference(dataset, tmp_path / "output")
    dataset = write_dataset(tmp_path, ("unknownRelation",))
    with pytest.raises(zs.InferenceError, match="Unsupported"):
        zs.run_inference(dataset, tmp_path / "output")
    assert calls == []


def test_in_memory_prediction_validation_uses_existing_contract():
    with pytest.raises(DatasetError, match="flat list of strings"):
        validate_prediction({"SubjectEntity": "A", "Relation": "hasArea", "ObjectEntities": [["123"]]})


def fake_huggingface_runtime(monkeypatch):
    """Exercise real loading/generation wiring with lightweight Python stand-ins."""
    events = {}
    tensor = SimpleNamespace(shape=(1, 3))

    class Batch(dict):
        def to(self, device):
            events["input_device"] = device
            return self

    class Output:
        def __getitem__(self, key):
            assert key == (0, slice(3, None))  # Decode completion, not the prompt.
            return [11, 12]

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            events["chat"] = (messages, kwargs)
            return Batch(input_ids=tensor)

        def decode(self, tokens, **kwargs):
            assert tokens == [11, 12]
            assert kwargs == {"skip_special_tokens": True}
            return '["generated only"]'

        def get_chat_template(self):
            return "fake-chat-template"

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[1, 2])
        config = SimpleNamespace(max_position_embeddings=1024)

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            events["eval"] = True

        def generate(self, **kwargs):
            events["generate"] = kwargs
            return Output()

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs

        def to_dict(self):
            return self.values

    def load_model(name, **kwargs):
        events["model"] = (name, kwargs)
        return Model()

    def load_tokenizer(name, **kwargs):
        events["tokenizer"] = (name, kwargs)
        return Tokenizer()

    def model_info(name, **kwargs):
        events["hub"] = (name, kwargs)
        return SimpleNamespace(sha="b" * 40)

    torch = SimpleNamespace(
        __version__="fake-torch", float16="fake-fp16",
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fake T4"),
        manual_seed=lambda seed: events.update(seed=seed),
        use_deterministic_algorithms=lambda flag: events.update(deterministic=flag),
        inference_mode=nullcontext, version=SimpleNamespace(cuda="fake CUDA"),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 123),
            cuda=SimpleNamespace(matmul=SimpleNamespace()),
        ),
    )
    transformers = SimpleNamespace(
        __version__="fake-transformers", GenerationConfig=Config,
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(__version__="fake-hub", model_info=model_info))
    monkeypatch.setattr(zs, "version", lambda name: "fake-" + name)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "will-be-overridden")
    return events, torch


def test_actual_generation_wiring_pins_revision_and_greedy_settings(monkeypatch):
    events, torch = fake_huggingface_runtime(monkeypatch)
    generate, metadata = zs._load_generator(zs.DEFAULT_MODEL, "branch", 64)
    assert generate(zs.build_messages("A", "hasArea")) == '["generated only"]'
    assert events["hub"][1]["revision"] == "branch"
    assert events["model"][1]["revision"] == events["tokenizer"][1]["revision"] == "b" * 40
    assert events["model"][1]["torch_dtype"] == torch.float16
    assert events["model"][1]["trust_remote_code"] is False
    assert events["model"][1]["attn_implementation"] == "eager"
    assert events["eval"] and events["deterministic"]
    assert events["seed"] == 0
    assert events["input_device"] == "cuda"
    config = events["generate"]["generation_config"].to_dict()
    assert config["do_sample"] is False
    assert config["num_beams"] == config["num_return_sequences"] == 1
    assert config["max_new_tokens"] == 64
    assert metadata["resolved_revision"] == "b" * 40
    assert metadata["generation"] == config
    assert metadata["versions"]["transformers"] == "fake-transformers"
    assert metadata["cublas_workspace_config"] == ":4096:8"
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_no_gpu_fails_before_hub_access(monkeypatch):
    events, torch = fake_huggingface_runtime(monkeypatch)
    torch.cuda.is_available = lambda: False
    with pytest.raises(zs.InferenceError, match="CUDA GPU"):
        zs._load_generator(zs.DEFAULT_MODEL, zs.DEFAULT_REVISION, 64)
    assert "hub" not in events


def test_cli_help_needs_no_inference_dependencies():
    completed = subprocess.run(
        [sys.executable, "-m", "llm_kbc.zero_shot", "--help"],
        text=True, capture_output=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    for flag in ("--dataset", "--output", "--model", "--revision", "--max-examples", "--max-new-tokens"):
        assert flag in completed.stdout


def test_cli_smoke_arguments_and_success_with_logged_abstention(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path, ("hasArea",))
    output = tmp_path / "predictions.jsonl"
    calls = fake_generator(monkeypatch, ["not JSON"])
    assert zs.main([
        "--dataset", str(dataset), "--output", str(output), "--max-examples", "1", "--max-new-tokens", "32",
    ]) == 0
    assert calls[0][-1] == 32
    assert list(load_predictions(output))[0]["ObjectEntities"] == []
    assert json.loads(Path(str(output) + ".failures.jsonl").read_text())["response"] == "not JSON"
