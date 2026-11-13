"""CPU-only contracts for the shared inference runner and new v2 preflight."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from llm_kbc import few_shot, few_shot_v2, few_shot_v2_inference as inference, zero_shot
from llm_kbc.dataset_inspector import DatasetError
from llm_kbc.prediction_io import load_predictions


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FakeTokenizer:
    name_or_path = zero_shot.DEFAULT_MODEL
    model_max_length = 131072

    def get_chat_template(self):
        return "fake-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "truncation": False}
        assert len(messages) in (4, 12)
        return list(range(32768 if "TOO_LONG" in messages[-1]["content"] else 37))


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    training, inputs = [], []
    for relation in zero_shot.QUESTIONS:
        for i in range(5):
            objects = [[str(i + 1), "UNUSED_ALIAS"]] if relation in few_shot.NUMERIC_RELATIONS else [
                [f"Canonical {j}", "UNUSED_ALIAS"] for j in range(i)
            ]
            training.append({"SubjectEntity": f"Training {relation} {i}", "Relation": relation, "ObjectEntities": objects})
        inputs.append({"SubjectEntity": f"Query {relation}", "Relation": relation, "ObjectEntities": "SECRET_GOLD"})
    train, dataset = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_rows(train, training)
    write_rows(dataset, inputs)
    monkeypatch.setattr(inference, "TRAIN_SHA256", hashlib.sha256(train.read_bytes()).hexdigest())
    monkeypatch.setattr(inference, "_load_tokenizer", FakeTokenizer)
    monkeypatch.setattr(inference, "_source_provenance", lambda: {
        "commit": "a" * 40, "working_tree_dirty": True, "files_sha256": {},
    })
    return tmp_path, dataset, tmp_path / "new_outputs" / "predictions.jsonl", inputs


def fake_generator(monkeypatch, responses, **runtime_changes):
    calls, iterator = [], iter(responses)

    def load(*args):
        calls.append(args)

        def generate(messages):
            calls.append(messages)
            response = next(iterator)
            if isinstance(response, BaseException):
                raise response
            return response

        return generate, {
            "model_id": zero_shot.DEFAULT_MODEL, "requested_revision": zero_shot.DEFAULT_REVISION,
            "resolved_revision": zero_shot.DEFAULT_REVISION, "dtype": "float16",
            "attention_implementation": "eager", "batch_size": 1,
            "chat_template_sha256": hashlib.sha256(b"fake-template").hexdigest(),
            "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 512},
            "versions": {"torch": "fake", "transformers": "fake"}, **runtime_changes,
        }

    monkeypatch.setattr(zero_shot, "_load_generator", load)
    return calls


@pytest.mark.parametrize("shots", [1, 5])
def test_complete_run_schema_parser_artifacts_and_reproducibility(setup_run, monkeypatch, shots):
    directory, dataset, output, inputs = setup_run
    original = (dataset.read_bytes(), (directory / "train.jsonl").read_bytes())
    responses = ['["answer"]', '[]', '[104.400, 1e3]', '24522\u27e9', '[null]', '```json\n["last"]\n```']
    calls = fake_generator(monkeypatch, responses)
    metadata = inference.run_inference(directory, dataset, output, shots=shots)
    assert calls[0] == (zero_shot.DEFAULT_MODEL, zero_shot.DEFAULT_REVISION, 512)
    prepared = few_shot_v2.prepare_prompts(directory, dataset, shots=shots, tokenizer=FakeTokenizer())
    assert calls[1:] == [p["messages"] for p in prepared["prompts"]]
    assert "SECRET_GOLD" not in json.dumps(calls) and "UNUSED_ALIAS" not in json.dumps(calls)
    predictions = list(load_predictions(output))
    assert [(r["SubjectEntity"], r["Relation"]) for r in predictions] == [(r["SubjectEntity"], r["Relation"]) for r in inputs]
    assert [r["ObjectEntities"] for r in predictions] == [["answer"], [], ["104.400", "1e3"], [], [], ["last"]]
    raw = [json.loads(line) for line in Path(str(output) + ".raw.jsonl").read_text().splitlines()]
    failures = [json.loads(line) for line in Path(str(output) + ".failures.jsonl").read_text().splitlines()]
    assert [r["response"] for r in raw] == responses
    assert [r["input_line"] for r in raw] == list(range(1, 7))
    assert [r["response"] for r in failures] == responses[3:5]
    assert metadata["status"] == "complete" and metadata["shots"] == shots
    assert metadata["total_inputs"] == metadata["processed_records"] == 6
    assert metadata["valid_records"] == 4 and metadata["parse_failures"] == 2
    assert metadata["numeric_normalized_responses"] == 1 and metadata["format_validity_rate"] == 4 / 6
    assert metadata["parse_failure_policy"] == "parse-failure-as-abstention-v1"
    assert metadata["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert metadata == json.loads(Path(str(output) + ".metadata.json").read_text())
    manifest_path = Path(str(output) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert metadata["manifest"]["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for key in ("shots", "demonstrations", "train_sha256", "inputs_sha256", "chat_template_sha256", "tokenizer_versions"):
        assert manifest[key] == prepared[key]
    assert manifest["prompt_version"] == metadata["prompt_version"] == few_shot_v2.PROMPT_VERSION
    assert manifest["selection_policy"]["version"] == few_shot_v2.SELECTION_VERSION
    assert manifest["runtime"]["dtype"] == "float16" and manifest["generation"]["max_new_tokens"] == 512
    assert manifest["tokenizer_checked"] and manifest["source"]["commit"] == "a" * 40
    assert manifest["source"]["files_sha256"]["few_shot_v2.py"] == hashlib.sha256(Path(few_shot_v2.__file__).read_bytes()).hexdigest()
    assert original == (dataset.read_bytes(), (directory / "train.jsonl").read_bytes())


@pytest.mark.parametrize("shots", [1, 5])
def test_smoke_preflights_all_inputs_but_generates_prefix(setup_run, monkeypatch, shots):
    directory, dataset, output, _ = setup_run
    calls = fake_generator(monkeypatch, ["[]"] * 2)
    metadata = inference.run_inference(directory, dataset, output, shots=shots, max_examples=2)
    assert len(calls) == 3 and len(list(load_predictions(output))) == 2
    assert metadata["selection"] == {
        "strategy": "first_records_in_file_order", "max_examples": 2, "records": 2, "preflight_records": 6,
    }
    assert len(json.loads(Path(str(output) + ".manifest.json").read_text())["prompts"]) == 6


@pytest.mark.parametrize("shots", [1, 5])
@pytest.mark.parametrize("problem", ["leakage", "duplicate", "context", "malformed", "unsupported", "hash"])
def test_preflight_failure_before_gpu_or_directory_creation(setup_run, monkeypatch, shots, problem):
    directory, dataset, output, inputs = setup_run
    if problem == "leakage":
        inputs[-1]["SubjectEntity"] = few_shot_v2.select_demonstrations(directory, shots=shots)[inputs[-1]["Relation"]][-1]["SubjectEntity"].upper()
    elif problem == "duplicate": inputs[-1] = inputs[0]
    elif problem == "context": inputs[-1]["SubjectEntity"] = "TOO_LONG"
    elif problem == "unsupported": inputs[-1]["Relation"] = "unsupported"
    elif problem == "hash": monkeypatch.setattr(inference, "TRAIN_SHA256", "0" * 64)
    write_rows(dataset, inputs)
    if problem == "malformed":
        with dataset.open("a") as stream:
            stream.write("{bad}\n")

    def forbidden(*args):
        pytest.fail("GPU loader must not be called before complete preflight")

    monkeypatch.setattr(zero_shot, "_load_generator", forbidden)
    with pytest.raises((few_shot_v2.FewShotError, DatasetError, zero_shot.InferenceError)):
        inference.run_inference(directory, dataset, output, shots=shots, max_examples=1)
    assert not output.parent.exists()


@pytest.mark.parametrize("suffix", ["", ".manifest.json", ".metadata.json", ".raw.jsonl", ".failures.jsonl"])
def test_no_overwrite_before_tokenizer_load(setup_run, monkeypatch, suffix):
    directory, dataset, output, _ = setup_run
    output.parent.mkdir()
    artifact = Path(str(output) + suffix)
    artifact.write_bytes(b"PRESERVE")
    monkeypatch.setattr(inference, "_load_tokenizer", lambda: pytest.fail("Must refuse before loading"))
    with pytest.raises(zero_shot.InferenceError, match="overwrite"):
        inference.run_inference(directory, dataset, output, shots=5)
    assert artifact.read_bytes() == b"PRESERVE"
    assert list(output.parent.iterdir()) == [artifact]


@pytest.mark.parametrize("error", [RuntimeError("OOM"), KeyboardInterrupt()])
def test_interrupted_run_retains_diagnostics_no_final_predictions(setup_run, monkeypatch, error):
    directory, dataset, output, _ = setup_run
    fake_generator(monkeypatch, ["24522\u27e9", error])
    with pytest.raises(type(error)):
        inference.run_inference(directory, dataset, output, shots=5)
    assert not output.exists()
    metadata = json.loads(Path(str(output) + ".metadata.json").read_text())
    assert metadata["processed_records"] == 1 and metadata["parse_failures"] == 1
    assert metadata["status"] == ("runtime_error" if isinstance(error, Exception) else "interrupted")
    assert json.loads(Path(str(output) + ".raw.jsonl").read_text())["response"] == "24522\u27e9"
    assert Path(str(output) + ".manifest.json").exists()
    assert not list(output.parent.glob("*.tmp"))


@pytest.mark.parametrize("field,value", [("resolved_revision", "wrong"), ("chat_template_sha256", "wrong")])
def test_runtime_identity_mismatch_no_artifacts(setup_run, monkeypatch, field, value):
    directory, dataset, output, _ = setup_run
    fake_generator(monkeypatch, [], **{field: value})
    with pytest.raises(zero_shot.InferenceError, match="differs"):
        inference.run_inference(directory, dataset, output, shots=1)
    assert not output.parent.exists()


@pytest.mark.parametrize("path_name", ["train.jsonl", "val.jsonl"])
def test_input_mutation_during_model_loading_prevents_publication(setup_run, monkeypatch, path_name):
    directory, dataset, output, _ = setup_run
    fake_generator(monkeypatch, ["[]"])
    original_loader = zero_shot._load_generator

    def load(*args):
        result = original_loader(*args)
        with (directory / path_name).open("a") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(zero_shot, "_load_generator", load)
    with pytest.raises(zero_shot.InferenceError, match="changed"):
        inference.run_inference(directory, dataset, output, shots=5)
    assert not output.parent.exists()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limits(setup_run, limit):
    directory, dataset, output, _ = setup_run
    with pytest.raises(zero_shot.InferenceError, match="positive integer"):
        inference.run_inference(directory, dataset, output, shots=1, max_examples=limit)


def test_help_and_invalid_three_shot_cli_are_cpu_only():
    result = subprocess.run([sys.executable, "-m", "llm_kbc.few_shot_v2_inference", "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--shots {1,5}" in result.stdout and "--max-examples" in result.stdout
    assert "--max-new-tokens" not in result.stdout and "--revision" not in result.stdout
    with pytest.raises(SystemExit) as error:
        inference.main(["--shots", "3", "--data-dir", ".", "--dataset", "val.jsonl"])
    assert error.value.code == 2


@pytest.mark.parametrize("shots", [1, 5])
def test_cli_default_outputs_distinct_and_fixed_settings(setup_run, monkeypatch, shots):
    directory, dataset, _, _ = setup_run
    monkeypatch.chdir(directory)
    calls = fake_generator(monkeypatch, ["[]"])
    assert inference.main(["--shots", str(shots), "--data-dir", str(directory), "--dataset", str(dataset), "--max-examples", "1"]) == 0
    assert Path(f"experiments/few_shot_v2/{shots}_shot.jsonl").exists()
    assert calls[0] == (zero_shot.DEFAULT_MODEL, zero_shot.DEFAULT_REVISION, 512)
