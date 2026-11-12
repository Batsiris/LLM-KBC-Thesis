"""QLoRA inference with CPU fakes: no model weights, network, training, or GPU."""

import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_kbc import qlora_inference as inference, zero_shot
from llm_kbc.dataset_inspector import DatasetError
from llm_kbc.prediction_io import load_predictions
from test_zero_shot import fake_huggingface_runtime


@pytest.fixture
def adapter(tmp_path):
    directory = tmp_path / "adapter"
    directory.mkdir()
    config = {
        "base_model_name_or_path": zero_shot.DEFAULT_MODEL, "revision": zero_shot.DEFAULT_REVISION,
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": True,
        "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj"], "bias": "none",
    }
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"synthetic CPU test adapter; not real weights")
    return directory


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "val.jsonl"
    rows = [
        {"SubjectEntity": "Estonia", "Relation": "hasArea", "ObjectEntities": [["GOLD_MUST_NOT_ENTER_PROMPTS"]]},
        {"SubjectEntity": "Award", "Relation": "awardWonBy", "ObjectEntities": {"unused": "gold"}},
        {"SubjectEntity": "Person", "Relation": "personHasCityOfDeath"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fake_generator(monkeypatch, responses):
    calls, responses = [], iter(responses)

    def load(adapter_dir, cache_dir, max_new_tokens):
        calls.append((adapter_dir, cache_dir, max_new_tokens))

        def generate(messages):
            calls.append(messages)
            response = next(responses)
            if isinstance(response, BaseException):
                raise response
            return response

        return generate, {
            "model_id": zero_shot.DEFAULT_MODEL, "requested_revision": zero_shot.DEFAULT_REVISION,
            "resolved_revision": zero_shot.DEFAULT_REVISION,
            "generation": {"do_sample": False, "max_new_tokens": max_new_tokens},
            "dtype": "float16", "seed": 0, "gpu": "fake GPU", "cuda": "fake CUDA",
            "versions": {"torch": "fake", "peft": "0.17.1"},
        }

    monkeypatch.setattr(inference, "_load_generator", load)
    return calls


def test_complete_run_reuses_zero_shot_prompts_parser_and_reproducibility_metadata(dataset, adapter, tmp_path, monkeypatch):
    output = tmp_path / "out" / "predictions.jsonl"
    original = {path: path.read_bytes() for path in [dataset, *adapter.iterdir()]}
    responses = ['[104.400, 1e3]', '24522\u27e9', '["Paris"]']
    calls = fake_generator(monkeypatch, responses)
    parsed = []
    parse = zero_shot._parse_response

    def spy(response):
        parsed.append(response)
        return parse(response)

    monkeypatch.setattr(zero_shot, "_parse_response", spy)
    metadata = inference.run_inference(dataset, output, adapter_dir=adapter)
    assert parsed == responses
    assert calls[0] == (adapter.resolve(), None, 512)
    rows = read_rows(dataset)
    assert calls[1:] == [zero_shot.build_messages(row["SubjectEntity"], row["Relation"]) for row in rows]
    assert "GOLD_MUST_NOT_ENTER_PROMPTS" not in json.dumps(calls[1:])
    predictions = list(load_predictions(output))
    assert [row["ObjectEntities"] for row in predictions] == [["104.400", "1e3"], [], ["Paris"]]
    assert [(row["SubjectEntity"], row["Relation"]) for row in predictions] == [
        (row["SubjectEntity"], row["Relation"]) for row in rows
    ]
    assert all(set(row) == {"SubjectEntity", "Relation", "ObjectEntities"} for row in predictions)
    assert metadata == json.loads(Path(str(output) + ".metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["total_inputs"] == metadata["processed_records"] == 3
    assert metadata["parse_failures"] == metadata["numeric_normalized_responses"] == 1
    assert metadata["valid_records"] == 2 and metadata["format_validity_rate"] == 2 / 3
    assert metadata["parse_failure_policy"] == "parse-failure-as-abstention-v1"
    assert metadata["dataset"]["sha256"] == hashlib.sha256(original[dataset]).hexdigest()
    assert metadata["adapter"] == inference._read_adapter(adapter.resolve())
    assert metadata["adapter"]["weights_sha256"] == hashlib.sha256(original[adapter / "adapter_model.safetensors"]).hexdigest()
    assert metadata["adapter"]["config"] == json.loads(original[adapter / "adapter_config.json"])
    assert metadata["adapter"]["path"] == str(adapter.resolve()) and metadata["adapter"]["merged"] is False
    assert metadata["prompt_version"] == "lm-kbc-2026-zero-shot-v2"
    assert metadata["prompt_sha256"] == hashlib.sha256(json.dumps(
        [zero_shot.SYSTEM_PROMPT, zero_shot.QUESTIONS], sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    assert metadata["resolved_revision"] == zero_shot.DEFAULT_REVISION
    assert metadata["generation"] == {"do_sample": False, "max_new_tokens": 512}
    assert metadata["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    raw = read_rows(Path(str(output) + ".raw.jsonl"))
    assert [row["response"] for row in raw] == responses
    assert [row["input_line"] for row in raw] == [1, 2, 3]
    assert [row["parsing_status"] for row in raw] == ["valid", "parse_failed", "valid"]
    assert read_rows(Path(str(output) + ".failures.jsonl"))[0]["response"] == responses[1]
    assert {path: path.read_bytes() for path in original} == original


def test_smoke_selection_preserves_order_and_configured_limits(dataset, adapter, tmp_path, monkeypatch):
    calls = fake_generator(monkeypatch, ["[]", "[]"])
    output = tmp_path / "smoke.jsonl"
    cache = tmp_path / "model-cache"
    metadata = inference.run_inference(dataset, output, adapter_dir=adapter, cache_dir=cache, max_examples=2, max_new_tokens=64)
    assert calls[0] == (adapter.resolve(), cache.resolve(), 64)
    assert [row["SubjectEntity"] for row in load_predictions(output)] == ["Estonia", "Award"]
    assert metadata["selection"] == {"strategy": "first_records_in_file_order", "max_examples": 2, "records": 2}


@pytest.mark.parametrize("field,value", [
    ("base_model_name_or_path", "other/model"), ("base_model_name_or_path", None),
    ("revision", "main"), ("revision", "b" * 40), ("revision", None),
    ("peft_type", "PREFIX_TUNING"), ("task_type", "SEQ_CLS"),
])
def test_incompatible_adapters_fail_before_gpu_and_outputs(dataset, adapter, tmp_path, monkeypatch, field, value):
    path = adapter / "adapter_config.json"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))
    calls = fake_generator(monkeypatch, [])
    output = tmp_path / "out" / "predictions.jsonl"
    with pytest.raises(zero_shot.InferenceError, match="Incompatible adapter"):
        inference.run_inference(dataset, output, adapter_dir=adapter)
    assert calls == [] and not output.parent.exists()


@pytest.mark.parametrize("problem", ["missing_config", "missing_weights", "bad_json", "non_object"])
def test_incomplete_or_malformed_adapter_is_rejected(dataset, adapter, tmp_path, monkeypatch, problem):
    config = adapter / "adapter_config.json"
    if problem == "missing_config": config.unlink()
    elif problem == "missing_weights": (adapter / "adapter_model.safetensors").unlink()
    elif problem == "bad_json": config.write_text("{bad JSON}")
    else: config.write_text("[]")
    calls = fake_generator(monkeypatch, [])
    with pytest.raises(zero_shot.InferenceError):
        inference.run_inference(dataset, tmp_path / "predictions.jsonl", adapter_dir=adapter)
    assert calls == []


@pytest.mark.parametrize("suffix", ["", ".metadata.json", ".failures.jsonl", ".raw.jsonl"])
def test_existing_results_and_sidecars_are_not_overwritten(dataset, adapter, tmp_path, monkeypatch, suffix):
    output = tmp_path / "predictions.jsonl"
    protected = Path(str(output) + suffix)
    protected.write_bytes(b"preserve exactly")
    calls = fake_generator(monkeypatch, [])
    with pytest.raises(zero_shot.InferenceError, match="Refusing to overwrite"):
        inference.run_inference(dataset, output, adapter_dir=adapter)
    assert calls == [] and protected.read_bytes() == b"preserve exactly"


def test_adapter_directory_is_read_only_for_output_and_cache(dataset, adapter, tmp_path, monkeypatch):
    fake_generator(monkeypatch, [])
    for output, cache in ((adapter / "new.jsonl", None), (tmp_path / "new.jsonl", adapter / "cache")):
        with pytest.raises(zero_shot.InferenceError, match="outside the adapter"):
            inference.run_inference(dataset, output, adapter_dir=adapter, cache_dir=cache)


@pytest.mark.parametrize("error", [RuntimeError("fake OOM"), KeyboardInterrupt()])
def test_interruption_keeps_flushed_raw_and_failure_logs_without_predictions(dataset, adapter, tmp_path, monkeypatch, error):
    output = tmp_path / "predictions.jsonl"
    calls = []

    def generate(messages):
        if calls:
            assert read_rows(Path(str(output) + ".raw.jsonl"))[0]["response"] == '24522\u27e9'
            assert not output.exists()
            raise error
        calls.append(messages)
        return '24522\u27e9'

    monkeypatch.setattr(inference, "_load_generator", lambda *args: (generate, {}))
    with pytest.raises(type(error)):
        inference.run_inference(dataset, output, adapter_dir=adapter)
    assert not output.exists()
    metadata = json.loads(Path(str(output) + ".metadata.json").read_text())
    assert metadata["status"] == ("interrupted" if isinstance(error, KeyboardInterrupt) else "runtime_error")
    assert metadata["processed_records"] == metadata["parse_failures"] == 1
    assert read_rows(Path(str(output) + ".failures.jsonl"))[0]["response"] == '24522\u27e9'


def test_final_prediction_validation_is_reused(dataset, adapter, tmp_path, monkeypatch):
    fake_generator(monkeypatch, ["[]"] * 3)
    original = zero_shot.validate_prediction

    def corrupt(row, *, location):
        validated = original(row, location=location)
        if location == "generated record 2":
            validated["SubjectEntity"] = "wrong identity"
        return validated

    monkeypatch.setattr(zero_shot, "validate_prediction", corrupt)
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(zero_shot.InferenceError, match="order or unique identities"):
        inference.run_inference(dataset, output, adapter_dir=adapter)
    assert not output.exists()


@pytest.mark.parametrize("what", ["adapter", "dataset"])
def test_changed_artifacts_during_model_loading_are_rejected(dataset, adapter, tmp_path, monkeypatch, what):
    def load(*args):
        path = adapter / "adapter_model.safetensors" if what == "adapter" else dataset
        path.write_bytes(path.read_bytes() + b"\n")
        return lambda messages: pytest.fail("Generation must not start"), {}

    monkeypatch.setattr(inference, "_load_generator", load)
    output = tmp_path / "out" / "predictions.jsonl"
    with pytest.raises(zero_shot.InferenceError, match="files changed"):
        inference.run_inference(dataset, output, adapter_dir=adapter)
    assert not output.parent.exists()


@pytest.mark.parametrize("text", ["", "\n", "{bad}\n", "[]\n", '{}\n',
                                      '{"SubjectEntity": "A", "Relation": "unknown"}\n'])
def test_invalid_inputs_are_rejected_before_loading(dataset, adapter, tmp_path, monkeypatch, text):
    dataset.write_text(text)
    calls = fake_generator(monkeypatch, [])
    with pytest.raises((DatasetError, zero_shot.InferenceError)):
        inference.run_inference(dataset, tmp_path / "predictions.jsonl", adapter_dir=adapter)
    assert calls == []


def test_duplicate_identities_fail_before_loading(dataset, adapter, tmp_path, monkeypatch):
    row = dataset.read_text().splitlines()[0]
    dataset.write_text(row + "\n" + row + "\n")
    calls = fake_generator(monkeypatch, [])
    with pytest.raises(zero_shot.InferenceError, match="duplicate"):
        inference.run_inference(dataset, tmp_path / "predictions.jsonl", adapter_dir=adapter)
    assert calls == []


@pytest.mark.parametrize("kwargs", [{"max_examples": 0}, {"max_new_tokens": 0}, {"max_examples": True}])
def test_invalid_limits_are_rejected(tmp_path, kwargs):
    with pytest.raises(zero_shot.InferenceError, match="positive integers"):
        inference.run_inference(tmp_path / "missing", tmp_path / "output", adapter_dir=tmp_path / "adapter", **kwargs)


@pytest.fixture
def runtime(monkeypatch):
    events, torch = fake_huggingface_runtime(monkeypatch)
    tf = sys.modules["transformers"]
    original_load = tf.AutoModelForCausalLM.from_pretrained

    def load_base(*args, **kwargs):
        model = original_load(*args, **kwargs)
        model.config._commit_hash = zero_shot.DEFAULT_REVISION
        return model

    tf.AutoModelForCausalLM.from_pretrained = load_base

    class Adapted:
        def __init__(self, base):
            self.base, self.device = base, base.device

        def to(self, **kwargs):
            events["adapter_to"] = kwargs
            return self

        def requires_grad_(self, flag):
            events["requires_grad"] = flag

        def eval(self):
            events["adapter_eval"] = True
            self.base.eval()

        def generate(self, **kwargs):
            events["adapter_generate"] = True
            return self.base.generate(**kwargs)

    def attach(base, path, **kwargs):
        events["attach"] = (base, path, kwargs)
        return Adapted(base)

    peft = SimpleNamespace(__version__="0.17.1", PeftModel=SimpleNamespace(from_pretrained=attach))
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr(inference, "version", lambda name: "fake-" + name)
    return events, torch, peft


def test_actual_loader_wiring_matches_zero_shot_settings_with_frozen_unmerged_fp16_adapter(adapter, tmp_path, runtime):
    events, torch, _ = runtime
    cache = tmp_path / "cache"
    generate, metadata = inference._load_generator(adapter, cache, 64)
    assert generate(zero_shot.build_messages("Query", "hasArea")) == '["generated only"]'
    assert events["model"][0] == events["tokenizer"][0] == zero_shot.DEFAULT_MODEL
    assert events["model"][1]["revision"] == events["tokenizer"][1]["revision"] == zero_shot.DEFAULT_REVISION
    assert events["model"][1]["cache_dir"] == events["tokenizer"][1]["cache_dir"] == str(cache)
    assert events["model"][1]["torch_dtype"] == torch.float16 and events["model"][1]["attn_implementation"] == "eager"
    assert "quantization_config" not in events["model"][1] and "load_in_4bit" not in events["model"][1]
    assert events["attach"][1] == str(adapter)
    assert events["attach"][2] == {"is_trainable": False, "autocast_adapter_dtype": False,
                                      "local_files_only": True, "ignore_mismatched_sizes": False}
    assert events["adapter_to"] == {"dtype": torch.float16}
    assert events["requires_grad"] is False and events["adapter_eval"] and events["adapter_generate"]
    assert events["seed"] == 0 and events["deterministic"]
    assert torch.backends.cudnn.benchmark is False and torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert metadata["resolved_revision"] == zero_shot.DEFAULT_REVISION
    assert metadata["versions"]["peft"] == "0.17.1" and metadata["gpu"] == "fake T4"
    _, baseline = zero_shot._load_generator(zero_shot.DEFAULT_MODEL, zero_shot.DEFAULT_REVISION, 64)
    assert metadata["generation"] == baseline["generation"]
    assert metadata["chat_template_sha256"] == baseline["chat_template_sha256"]


def test_gpu_required_before_model_loading(adapter, runtime):
    events, torch, _ = runtime
    torch.cuda.is_available = lambda: False
    with pytest.raises(zero_shot.InferenceError, match="CUDA GPU"):
        inference._load_generator(adapter, None, 512)
    assert "model" not in events


def test_missing_adapter_tensors_are_fatal(adapter, runtime):
    _, _, peft = runtime

    def missing(*args, **kwargs):
        warnings.warn("Found missing adapter keys while loading the checkpoint: ['lora_A'].")

    peft.PeftModel.from_pretrained = missing
    with pytest.raises(zero_shot.InferenceError, match="Incomplete adapter weights"):
        inference._load_generator(adapter, None, 512)


@pytest.mark.parametrize("problem", ["revision", "quantized"])
def test_loaded_base_is_verified_before_adapter_attachment(adapter, runtime, problem):
    events, _, _ = runtime
    auto_model = sys.modules["transformers"].AutoModelForCausalLM
    original = auto_model.from_pretrained

    def incompatible(*args, **kwargs):
        base = original(*args, **kwargs)
        if problem == "revision":
            base.config._commit_hash = "b" * 40
        else:
            base.is_loaded_in_4bit = True
        return base

    auto_model.from_pretrained = incompatible
    with pytest.raises(zero_shot.InferenceError):
        inference._load_generator(adapter, None, 512)
    assert "attach" not in events


def test_adapter_shape_errors_are_not_ignored(adapter, runtime):
    _, _, peft = runtime

    def incompatible(*args, **kwargs):
        assert kwargs["ignore_mismatched_sizes"] is False
        raise RuntimeError("synthetic adapter tensor shape mismatch")

    peft.PeftModel.from_pretrained = incompatible
    with pytest.raises(RuntimeError, match="tensor shape mismatch"):
        inference._load_generator(adapter, None, 512)


def test_context_overflow_does_not_generate(adapter, runtime):
    events, _, _ = runtime
    generate, _ = inference._load_generator(adapter, None, 1024)
    with pytest.raises(zero_shot.InferenceError, match="context window"):
        generate(zero_shot.build_messages("Query", "hasArea"))
    assert "generate" not in events


def test_cli_help_imports_no_gpu_dependencies():
    result = subprocess.run([sys.executable, "-m", "llm_kbc.qlora_inference", "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    for name in ("--dataset", "--output", "--adapter-dir", "--cache-dir", "--max-examples", "--max-new-tokens"):
        assert name in result.stdout
    result = subprocess.run([sys.executable, "-c", "import sys; import llm_kbc.qlora_inference; "
                             "assert not {'torch', 'transformers', 'peft'} & sys.modules.keys()"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_cli_smoke_passes_all_arguments(dataset, adapter, tmp_path, monkeypatch):
    calls = fake_generator(monkeypatch, ["[]"])
    output = tmp_path / "predictions.jsonl"
    cache = tmp_path / "cache"
    assert inference.main([
        "--dataset", str(dataset), "--output", str(output), "--adapter-dir", str(adapter),
        "--cache-dir", str(cache), "--max-examples", "1", "--max-new-tokens", "32",
    ]) == 0
    assert calls[0] == (adapter.resolve(), cache.resolve(), 32)
    assert len(list(load_predictions(output))) == 1
