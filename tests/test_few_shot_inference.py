"""Few-shot inference contracts with CPU-only fakes; no downloads or scoring."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_kbc import few_shot, few_shot_inference as inference, zero_shot
from llm_kbc.dataset_inspector import DatasetError
from llm_kbc.prediction_io import load_predictions


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FakeTokenizer:
    name_or_path = zero_shot.DEFAULT_MODEL
    model_max_length = 131072

    def get_chat_template(self):
        return "fake-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "truncation": False}
        assert len(messages) == 8
        return list(range(37))


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    training, inputs = [], []
    for relation in zero_shot.QUESTIONS:
        for index in range(3):
            objects = [[str(index + 1), "UNUSED_ALIAS"]] if relation in few_shot.NUMERIC_RELATIONS else [
                [f"Canonical {j}", "UNUSED_ALIAS"] for j in range(index)
            ]
            training.append({
                "SubjectEntity": f"Training {relation} {index}", "Relation": relation,
                "ObjectEntities": objects,
            })
        inputs.append({
            "SubjectEntity": f"Query {relation}", "Relation": relation,
            "ObjectEntities": "SECRET_EVALUATION_GOLD_NOT_FOR_PROMPTS",
        })
    train, dataset = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_rows(train, training)
    write_rows(dataset, inputs)
    monkeypatch.setattr(inference, "TRAIN_SHA256", hashlib.sha256(train.read_bytes()).hexdigest())
    monkeypatch.setattr(inference, "_load_tokenizer", FakeTokenizer)
    monkeypatch.setattr(inference, "_source_provenance", lambda: {
        "commit": "a" * 40, "working_tree_dirty": False, "files_sha256": {"example.py": "b" * 64},
    })
    return tmp_path, dataset, tmp_path / "out" / "few_shot.jsonl", inputs


def fake_generator(monkeypatch, responses):
    responses, calls = iter(responses), []

    def load(*args):
        calls.append(args)

        def generate(messages):
            calls.append(messages)
            response = next(responses)
            if isinstance(response, BaseException):
                raise response
            return response

        return generate, {
            "model_id": zero_shot.DEFAULT_MODEL, "requested_revision": zero_shot.DEFAULT_REVISION,
            "resolved_revision": zero_shot.DEFAULT_REVISION,
            "chat_template_sha256": hashlib.sha256(b"fake-template").hexdigest(),
            "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 512},
            "dtype": "float16", "attention_implementation": "eager", "batch_size": 1,
            "versions": {"torch": "fake", "transformers": "fake"},
        }

    monkeypatch.setattr(zero_shot, "_load_generator", load)
    return calls


def test_complete_run_reuses_committed_prompts_and_audited_artifacts(setup_run, monkeypatch):
    directory, dataset, output, inputs = setup_run
    original = dataset.read_bytes(), (directory / "train.jsonl").read_bytes()
    responses = ['["answer"]', '[]', '[104.400, 1e3]', '24522\u27e9', '[null]', '```json\n["last"]\n```']
    calls = fake_generator(monkeypatch, responses)
    metadata = inference.run_inference(directory, dataset, output)
    assert calls[0] == (zero_shot.DEFAULT_MODEL, zero_shot.DEFAULT_REVISION, 512)
    prepared = few_shot.prepare_prompts(directory, dataset, tokenizer=FakeTokenizer())
    assert calls[1:] == [row["messages"] for row in prepared["prompts"]]
    assert "SECRET_EVALUATION_GOLD" not in json.dumps(calls)
    assert "UNUSED_ALIAS" not in json.dumps(calls)
    predictions = list(load_predictions(output))
    assert [(row["SubjectEntity"], row["Relation"]) for row in predictions] == [
        (row["SubjectEntity"], row["Relation"]) for row in inputs
    ]
    assert [row["ObjectEntities"] for row in predictions] == [["answer"], [], ["104.400", "1e3"], [], [], ["last"]]
    raw = read_rows(Path(str(output) + ".raw.jsonl"))
    assert [row["response"] for row in raw] == responses
    assert [row["input_line"] for row in raw] == list(range(1, 7))
    assert [row["parsing_status"] for row in raw] == ["valid"] * 3 + ["parse_failed"] * 2 + ["valid"]
    assert [row["response"] for row in read_rows(Path(str(output) + ".failures.jsonl"))] == responses[3:5]
    assert metadata["status"] == "complete"
    assert metadata["total_inputs"] == metadata["processed_records"] == 6
    assert metadata["valid_records"] == 4 and metadata["parse_failures"] == 2
    assert metadata["numeric_normalized_responses"] == 1
    assert metadata["format_validity_rate"] == 4 / 6
    assert metadata["parse_failure_policy"] == "parse-failure-as-abstention-v1"
    assert metadata["prompt_version"] == few_shot.PROMPT_VERSION
    assert metadata["base_prompt_version"] == "lm-kbc-2026-zero-shot-v2"
    assert metadata["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert metadata == json.loads(Path(str(output) + ".metadata.json").read_text())
    manifest_path = Path(str(output) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert metadata["manifest"]["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest["train_sha256"] == inference.TRAIN_SHA256
    assert manifest["inputs_sha256"] == hashlib.sha256(original[0]).hexdigest()
    assert manifest["demonstrations"] == few_shot.select_demonstrations(directory)
    assert manifest["selection_policy"]["version"] == few_shot.SELECTION_VERSION
    assert manifest["source"]["commit"] == "a" * 40
    assert manifest["max_new_tokens"] == 512 and manifest["tokenizer_checked"]
    assert manifest["context_limit"] == 32768
    assert manifest["tokenizer_name_or_path"] == zero_shot.DEFAULT_MODEL
    assert manifest["generation"] == metadata["generation"]
    assert manifest["runtime"]["dtype"] == "float16"
    assert manifest["runtime"]["attention_implementation"] == "eager"
    assert manifest["tokenizer_versions"].keys() == {"transformers", "tokenizers", "jinja2"}
    assert all(row["input_tokens"] == 37 for row in manifest["prompts"])
    assert (dataset.read_bytes(), (directory / "train.jsonl").read_bytes()) == original


def test_smoke_checks_all_inputs_but_generates_only_prefix(setup_run, monkeypatch):
    directory, dataset, output, _ = setup_run
    calls = fake_generator(monkeypatch, ["[]"] * 2)
    metadata = inference.run_inference(directory, dataset, output, max_examples=2)
    assert len(calls) == 3
    assert len(list(load_predictions(output))) == 2
    assert metadata["selection"] == {
        "strategy": "first_records_in_file_order", "max_examples": 2, "records": 2, "preflight_records": 6,
    }
    assert len(json.loads(Path(str(output) + ".manifest.json").read_text())["prompts"]) == 6


@pytest.mark.parametrize("problem", ["leakage", "duplicate", "context", "malformed", "unsupported"])
def test_preflight_checks_beyond_smoke_prefix_before_model_load(setup_run, monkeypatch, problem):
    directory, dataset, output, inputs = setup_run
    if problem == "leakage":
        demo = few_shot.select_demonstrations(directory)[inputs[-1]["Relation"]][0]
        inputs[-1]["SubjectEntity"] = "  " + demo["SubjectEntity"].upper() + "  "
    elif problem == "duplicate":
        inputs[-1] = inputs[0]
    elif problem == "unsupported":
        inputs[-1]["Relation"] = "unknown"
    elif problem == "context":
        original = FakeTokenizer.apply_chat_template

        def long_prompt(self, messages, **kwargs):
            return list(range(32768)) if "hasArea" in messages[-1]["content"] else original(self, messages, **kwargs)

        monkeypatch.setattr(FakeTokenizer, "apply_chat_template", long_prompt)
    write_rows(dataset, inputs)
    if problem == "malformed":
        with dataset.open("a", encoding="utf-8") as stream:
            stream.write("{bad json}\n")
    calls = fake_generator(monkeypatch, [])
    with pytest.raises((few_shot.FewShotError, DatasetError)):
        inference.run_inference(directory, dataset, output, max_examples=1)
    assert calls == [] and not output.parent.exists()


def test_subject_only_overlap_is_reported_without_reselection(setup_run, monkeypatch):
    directory, dataset, output, inputs = setup_run
    expected = few_shot.select_demonstrations(directory)
    inputs[0]["SubjectEntity"] = expected["awardWonBy"][0]["SubjectEntity"]
    write_rows(dataset, inputs)
    fake_generator(monkeypatch, ["[]"] * 6)
    inference.run_inference(directory, dataset, output)
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert manifest["demonstrations"] == expected
    assert len(manifest["leakage"]["subject_only_overlaps"]) == 1


@pytest.mark.parametrize("error", [RuntimeError("fake OOM"), StopIteration("incomplete"), KeyboardInterrupt()])
def test_interruption_retains_manifest_and_flushed_raw_without_predictions(setup_run, monkeypatch, error):
    directory, dataset, output, _ = setup_run
    fake_generator(monkeypatch, ['24522\u27e9', error])
    with pytest.raises(type(error)):
        inference.run_inference(directory, dataset, output)
    assert not output.exists()
    assert Path(str(output) + ".manifest.json").exists()
    assert read_rows(Path(str(output) + ".raw.jsonl"))[0]["response"] == '24522\u27e9'
    metadata = json.loads(Path(str(output) + ".metadata.json").read_text())
    assert metadata["processed_records"] == metadata["parse_failures"] == 1
    assert metadata["status"] == ("interrupted" if isinstance(error, KeyboardInterrupt) else "runtime_error")


@pytest.mark.parametrize("suffix", ["", ".manifest.json", ".metadata.json", ".failures.jsonl", ".raw.jsonl"])
def test_no_artifact_is_overwritten(setup_run, monkeypatch, suffix):
    directory, dataset, output, _ = setup_run
    output.parent.mkdir()
    protected = Path(str(output) + suffix)
    protected.write_bytes(b"keep exactly")
    calls = fake_generator(monkeypatch, [])
    with pytest.raises(zero_shot.InferenceError, match="Refusing to overwrite"):
        inference.run_inference(directory, dataset, output)
    assert calls == []
    assert protected.read_bytes() == b"keep exactly"
    assert list(output.parent.iterdir()) == [protected]


def test_training_snapshot_is_fixed(setup_run, monkeypatch):
    directory, dataset, output, _ = setup_run
    with (directory / "train.jsonl").open("a") as stream:
        stream.write("\n")
    calls = fake_generator(monkeypatch, [])
    with pytest.raises(few_shot.FewShotError, match="Training SHA256"):
        inference.run_inference(directory, dataset, output)
    assert calls == [] and not output.parent.exists()


@pytest.mark.parametrize("mismatch", ["resolved_revision", "chat_template_sha256"])
def test_runtime_must_match_preflight_before_generation(setup_run, monkeypatch, mismatch):
    directory, dataset, output, _ = setup_run
    calls = fake_generator(monkeypatch, [])
    load = zero_shot._load_generator

    def wrong_runtime(*args):
        generate, runtime = load(*args)
        runtime[mismatch] = "wrong"
        return generate, runtime

    monkeypatch.setattr(zero_shot, "_load_generator", wrong_runtime)
    with pytest.raises(zero_shot.InferenceError, match="differs"):
        inference.run_inference(directory, dataset, output)
    assert len(calls) == 1 and not output.parent.exists()


def test_changed_inputs_during_model_loading_fail_before_generation(setup_run, monkeypatch):
    directory, dataset, output, _ = setup_run
    calls = fake_generator(monkeypatch, [])
    load = zero_shot._load_generator

    def change_input(*args):
        result = load(*args)
        dataset.write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(zero_shot, "_load_generator", change_input)
    with pytest.raises(zero_shot.InferenceError, match="changed before generation"):
        inference.run_inference(directory, dataset, output)
    assert len(calls) == 1 and not output.parent.exists()


def test_invalid_final_prediction_cannot_be_published(setup_run, monkeypatch):
    directory, dataset, output, _ = setup_run
    fake_generator(monkeypatch, ["[]"] * 6)
    validate = zero_shot.validate_prediction

    def corrupt(row, *, location):
        result = validate(row, location=location)
        if location == "generated record 6":
            result["SubjectEntity"] = "wrong identity"
        return result

    monkeypatch.setattr(zero_shot, "validate_prediction", corrupt)
    with pytest.raises(zero_shot.InferenceError, match="order or unique identities"):
        inference.run_inference(directory, dataset, output)
    assert not output.exists()
    assert json.loads(Path(str(output) + ".metadata.json").read_text())["status"] == "runtime_error"


def test_tokenizer_loading_uses_exact_pinned_revision(monkeypatch):
    calls = []
    tokenizer = FakeTokenizer()

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return tokenizer

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load),
    ))
    assert inference._load_tokenizer() is tokenizer
    assert calls == [((zero_shot.DEFAULT_MODEL,), {
        "revision": zero_shot.DEFAULT_REVISION, "trust_remote_code": False,
    })]


def test_source_provenance_records_commit_dirty_state_and_source_hashes(monkeypatch):
    calls = []

    def git(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="c" * 40 if args[-1] == "HEAD" else " M README.md\n")

    monkeypatch.setattr(inference.subprocess, "run", git)
    provenance = inference._source_provenance()
    assert provenance["commit"] == "c" * 40 and provenance["working_tree_dirty"]
    assert provenance["files_sha256"]["few_shot_inference.py"] == hashlib.sha256(
        Path(inference.__file__).read_bytes()
    ).hexdigest()
    assert len(calls) == 2


def test_missing_git_provenance_fails_safely(monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(inference.subprocess, "run", fail)
    with pytest.raises(zero_shot.InferenceError, match="Git checkout"):
        inference._source_provenance()


def test_cli_help_is_cpu_only_and_generation_settings_are_not_overridable():
    result = subprocess.run(
        [sys.executable, "-m", "llm_kbc.few_shot_inference", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--data-dir" in result.stdout and "--max-examples" in result.stdout
    assert "--max-new-tokens" not in result.stdout and "--revision" not in result.stdout


def test_cli_smoke_uses_fixed_configuration(setup_run, monkeypatch):
    directory, dataset, output, _ = setup_run
    calls = fake_generator(monkeypatch, ["[]"])
    assert inference.main([
        "--data-dir", str(directory), "--dataset", str(dataset), "--output", str(output), "--max-examples", "1",
    ]) == 0
    assert calls[0] == (zero_shot.DEFAULT_MODEL, zero_shot.DEFAULT_REVISION, 512)
