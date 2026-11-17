"""CPU-only benchmark contract checks; all Hub/model operations are stand-ins."""

import json
import hashlib
import subprocess
import sys
import warnings
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_kbc import benchmark as bm, few_shot, few_shot_v2, zero_shot
from llm_kbc.prediction_io import load_predictions


SHA = "a" * 40


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def data(tmp_path, monkeypatch):
    train, inputs = [], []
    for relation in zero_shot.QUESTIONS:
        for index in range(7):
            objects = [[str(index * 100), "unused alias"]] if relation in few_shot.NUMERIC_RELATIONS else [
                [f"Object {value}", "unused alias"] for value in range(index)
            ]
            train.append({"SubjectEntity": f"Train {relation} {index}", "Relation": relation, "ObjectEntities": objects})
        # Deliberately invalid gold: inference must only access identities.
        inputs.append({"SubjectEntity": f"Input {relation}", "Relation": relation, "ObjectEntities": "SECRET GOLD"})
    write_jsonl(tmp_path / "train.jsonl", train)
    write_jsonl(tmp_path / "inputs.jsonl", inputs)
    monkeypatch.setattr(bm, "TRAIN_SHA256", bm._sha256(tmp_path / "train.jsonl"))
    return tmp_path, tmp_path / "inputs.jsonl"


class Tokenizer:
    model_max_length = 4096
    special_tokens_map = {"eos_token": "<end>"}
    pad_token_id = None
    eos_token_id = 19

    def get_chat_template(self):
        return "native-template"

    def apply_chat_template(self, messages, **kwargs):
        if kwargs["tokenize"] is False:
            return "\n".join(row["content"] for row in messages)
        assert kwargs["truncation"] is False
        if kwargs.get("return_tensors"):
            class Batch(dict):
                def to(self, device):
                    assert device == "cuda"
                    return self
            class Tensor:
                shape = (1, 20)

                def __getitem__(self, index):
                    assert index == 0
                    return SimpleNamespace(tolist=lambda: list(range(20)))

            return Batch(input_ids=Tensor(), attention_mask="mask")
        return list(range(20))

    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        assert tokens == [7, 8]
        return '["result"]'


@pytest.fixture
def cpu_runtime(monkeypatch):
    events = {}

    def config(model, **kwargs):
        events["config"] = (model, kwargs)
        return SimpleNamespace(_commit_hash=SHA, max_position_embeddings=4096, is_encoder_decoder=False)

    def tokenizer(model, **kwargs):
        events["tokenizer"] = (model, kwargs)
        return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=config), AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
        AutoModelForCausalLM=SimpleNamespace(),
    ))
    monkeypatch.setattr(bm, "_source_provenance", lambda: {"commit": "b" * 40, "working_tree_dirty": True, "files_sha256": {}})

    def loader(config, tokenizer, preflight, cache_dir):
        events["loaded"] = as_tuple = (config, preflight, cache_dir)
        assert as_tuple[0].revision == SHA
        responses = iter(events.get("responses", ['[104.40, "name"]'] * 6))

        def generate(messages):
            events.setdefault("messages", []).append(messages)
            response = next(responses)
            if isinstance(response, BaseException):
                raise response
            return response

        return generate, {**preflight, "model_id": config.model, "requested_revision": config.revision,
                          "generation": {"do_sample": False, "max_new_tokens": config.max_new_tokens},
                          "versions": {"transformers": "fake"}, "model_load_seconds": 1.25}

    monkeypatch.setattr(bm, "_load_generator", loader)
    return events


@pytest.mark.parametrize("shots", (0, 1, 3, 5))
@pytest.mark.parametrize("model", ("org/model-a", "org/model-b", "org/model-c", "org/model-d"))
def test_all_conditions_and_models_preserve_exact_prompts(data, tmp_path, monkeypatch, shots, model):
    directory, dataset = data
    config = bm.RunConfig(model, SHA, shots)
    monkeypatch.setattr(bm, "_tokenizer_preflight", lambda *a: pytest.fail("dry-run loaded a tokenizer"))
    first = bm.run(dataset, directory, tmp_path / "runs", "dry", config, dry_run=True)
    assert first == bm.run(dataset, directory, tmp_path / "runs", "dry", config, dry_run=True)
    assert not (tmp_path / "runs").exists()
    assert not first["tokenizer_checked"]
    assert len(first["prompts"]) == 6
    if shots == 3:
        expected = few_shot.prepare_prompts(directory, dataset)
    elif shots:
        expected = few_shot_v2.prepare_prompts(directory, dataset, shots=shots)
    else:
        expected = None
    for index, row in enumerate(first["prompts"]):
        assert len(row["messages"]) == 2 + shots * 2
        messages = expected["prompts"][index]["messages"] if expected else zero_shot.build_messages(row["SubjectEntity"], row["Relation"])
        assert row["messages"] == messages
        assert "SECRET GOLD" not in json.dumps(row)
        assert "unused alias" not in json.dumps(row)
    if expected:
        assert first["demonstrations"] == expected["demonstrations"]
        assert first["selection_version"] == expected["selection_version"]


@pytest.mark.parametrize("changes", [{"revision": "main"}, {"revision": None}, {"shots": 2}, {"seed": -1}, {"seed": True},
                                    {"max_new_tokens": 0}, {"dtype": "auto"}, {"context_limit": 0}, {"model": ""}])
def test_invalid_configuration(changes):
    with pytest.raises(bm.Error):
        replace(bm.RunConfig("org/model", SHA, 0), **changes)


def test_unique_safe_paths(tmp_path):
    configs = [bm.RunConfig(model, revision, shots) for model in ("a/b", "a_b")
               for revision in (SHA, "b" * 40) for shots in (0, 1, 3, 5)]
    paths = {bm.output_path(tmp_path, config, run) for config in configs for run in ("smoke", "full")}
    assert len(paths) == 32
    for run in ("..", "a/b", "a\\b", "C:", ""):
        with pytest.raises(bm.Error):
            bm.output_path(tmp_path, configs[0], run)
    target = bm.output_path(tmp_path, configs[0], "existing")
    target.parent.mkdir(parents=True)
    with pytest.raises(bm.Error, match="reuse"):
        bm.output_path(tmp_path, configs[0], "existing")


@pytest.mark.parametrize("shots", (0, 1, 3, 5))
def test_audited_run_metadata_failure_policy_and_no_overwrite(data, tmp_path, cpu_runtime, shots):
    directory, dataset = data
    cpu_runtime["responses"] = ['[104.40, "name"]', "24522⟩", "[]", "[]", "[]", "[]"]
    config = bm.RunConfig("independent/instruction-model", SHA, shots, seed=17)
    metadata = bm.run(dataset, directory, tmp_path / "runs", "complete", config)
    output = Path(metadata["output"])
    predictions = list(load_predictions(output))
    assert predictions[0]["ObjectEntities"] == ["104.40", "name"]
    assert predictions[1]["ObjectEntities"] == []
    assert metadata["status"] == "complete"
    assert metadata["valid_records"] == 5 and metadata["parse_failures"] == 1
    assert metadata["numeric_normalized_responses"] == 1
    assert metadata["format_validity_rate"] == 5 / 6
    assert metadata["inference_runtime_seconds"] >= 0
    assert metadata["resolved_revision"] == SHA
    assert metadata["dataset"]["sha256"] == bm._sha256(dataset)
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert manifest["configuration"] == {"model": config.model, "revision": SHA, "shots": shots, "seed": 17,
                                         "max_new_tokens": 512, "dtype": "float16", "context_limit": None,
                                         "quantization": "none", "thinking": "default", "determinism": "strict"}
    assert manifest["runtime"]["preflight_records"] == 6
    assert manifest["tokenizer_checked"]
    assert manifest["runtime"]["chat_template"] == "native-template"
    assert manifest["prompts_sha256"] == metadata["prompt_sha256"]
    assert metadata["manifest"]["sha256"] == bm._sha256(Path(str(output) + ".manifest.json"))
    raw = [json.loads(line) for line in Path(str(output) + ".raw.jsonl").read_text().splitlines()]
    assert len(raw) == 6 and raw[1]["response"] == "24522⟩"
    files = {file: file.read_bytes() for file in output.parent.iterdir()}
    with pytest.raises(bm.Error, match="reuse"):
        bm.run(dataset, directory, tmp_path / "runs", "complete", config)
    assert all(file.read_bytes() == content for file, content in files.items())


@pytest.mark.parametrize("error", [RuntimeError("OOM"), KeyboardInterrupt()])
def test_interruption_never_publishes_partial_output(data, tmp_path, cpu_runtime, error):
    directory, dataset = data
    cpu_runtime["responses"] = ["24522⟩", error]
    config = bm.RunConfig("org/model", SHA, 3)
    output = bm.output_path(tmp_path / "runs", config, "interrupted")
    with pytest.raises(type(error)):
        bm.run(dataset, directory, tmp_path / "runs", "interrupted", config)
    assert not output.exists()
    meta = json.loads(Path(str(output) + ".metadata.json").read_text())
    assert meta["status"] in ("interrupted", "runtime_error")
    assert meta["processed_records"] == meta["parse_failures"] == 1
    assert len(Path(str(output) + ".raw.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("problem", ("leakage", "duplicate", "unsupported", "training_hash", "malformed"))
def test_complete_file_checked_before_tokenizer_even_for_smoke(data, tmp_path, monkeypatch, problem):
    directory, dataset = data
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if problem == "leakage":
        rows[-1] = few_shot.select_demonstrations(directory)["hasArea"][0]
    elif problem == "duplicate":
        rows[-1] = {**rows[0], "SubjectEntity": "  " + rows[0]["SubjectEntity"].upper()}
    elif problem == "unsupported":
        rows[-1]["Relation"] = "unsupported"
    elif problem == "training_hash":
        monkeypatch.setattr(bm, "TRAIN_SHA256", "0" * 64)
    write_jsonl(dataset, rows)
    if problem == "malformed":
        dataset.write_text(dataset.read_text() + "{broken\n")
    monkeypatch.setattr(bm, "_tokenizer_preflight", lambda *a: pytest.fail("loaded before validation"))
    with pytest.raises((bm.Error, few_shot.FewShotError)):
        bm.run(dataset, directory, tmp_path / "runs", "bad", bm.RunConfig("org/model", SHA, 3), max_examples=1)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("problem", ("overflow", "missing_context", "revision", "encoder", "omitted_system", "template_error"))
def test_native_template_and_model_specific_context_failures(data, tmp_path, cpu_runtime, monkeypatch, problem):
    directory, dataset = data
    fake = sys.modules["transformers"]
    attributes = {"_commit_hash": SHA, "max_position_embeddings": 4096, "is_encoder_decoder": False}
    if problem == "revision":
        attributes["_commit_hash"] = "c" * 40
    if problem == "encoder":
        attributes["is_encoder_decoder"] = True
    if problem in ("overflow", "missing_context"):
        attributes["max_position_embeddings"] = 25 if problem == "overflow" else None
        monkeypatch.setattr(Tokenizer, "model_max_length", 10**30)
    if problem == "omitted_system":
        original = Tokenizer.apply_chat_template
        monkeypatch.setattr(Tokenizer, "apply_chat_template", lambda self, messages, **kw: original(self, messages[1:], **kw))
    if problem == "template_error":
        def fail(*args, **kwargs):
            raise ValueError("roles unsupported")
        monkeypatch.setattr(Tokenizer, "apply_chat_template", fail)
    fake.AutoConfig.from_pretrained = lambda *a, **kw: SimpleNamespace(**attributes)
    with pytest.raises(bm.Error):
        bm.run(dataset, directory, tmp_path / "runs", "bad", bm.RunConfig("org/model", SHA, 1))
    assert "loaded" not in cpu_runtime
    assert not (tmp_path / "runs").exists()


def test_smoke_preflights_all_records(data, tmp_path, cpu_runtime):
    directory, dataset = data
    meta = bm.run(dataset, directory, tmp_path / "runs", "smoke", bm.RunConfig("org/model", SHA, 5), max_examples=2)
    assert meta["total_inputs"] == 2 and meta["preflight_records"] == 6
    assert len(cpu_runtime["messages"]) == 2


@pytest.mark.parametrize("index", range(4))
def test_native_model_specific_templates_and_contexts(data, cpu_runtime, monkeypatch, index):
    directory, dataset = data
    config = bm.RunConfig(f"organization/model-{index}", SHA, 3)
    prepared = bm.prepare(dataset, directory, config)
    fake = sys.modules["transformers"]
    context = 1024 * (index + 1)
    # Different config field names; never fall back to a Qwen context constant.
    fields = ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length")
    fake.AutoConfig.from_pretrained = lambda model, **kw: SimpleNamespace(_commit_hash=SHA, **{fields[index]: context})
    monkeypatch.setattr(Tokenizer, "get_chat_template", lambda self: f"native-model-{index}")
    _, metadata = bm._tokenizer_preflight(config, prepared["prompts"], None)
    assert metadata["context_limit"] == context
    assert metadata["chat_template"] == f"native-model-{index}"
    assert cpu_runtime["tokenizer"][0] == config.model
    assert cpu_runtime["tokenizer"][1]["revision"] == SHA


@pytest.mark.parametrize("limit,allowed", [(4096, True), (1000, True), (8000, False), (10, False)])
def test_explicit_context_ceiling(data, cpu_runtime, limit, allowed):
    directory, dataset = data
    config = bm.RunConfig("org/model", SHA, 0, context_limit=limit)
    prepared = bm.prepare(dataset, directory, config)
    if allowed:
        _, metadata = bm._tokenizer_preflight(config, prepared["prompts"], None)
        assert metadata["context_limit"] == limit
    else:
        with pytest.raises(bm.Error):
            bm._tokenizer_preflight(config, prepared["prompts"], None)


def test_dataset_mutation_during_loading_never_publishes(data, tmp_path, cpu_runtime, monkeypatch):
    directory, dataset = data
    loader = bm._load_generator

    def changed(*args):
        result = loader(*args)
        dataset.write_bytes(dataset.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(bm, "_load_generator", changed)
    with pytest.raises(bm.Error, match="changed before generation"):
        bm.run(dataset, directory, tmp_path / "runs", "changed", bm.RunConfig("org/model", SHA, 0))
    assert not (tmp_path / "runs").exists()


@pytest.fixture
def torch_runtime(monkeypatch):
    events = {}

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[11, 12])

        def to(self, device):
            events["to"] = device
            self.device = device
            return self

        def eval(self):
            events["eval"] = True

        def generate(self, **kwargs):
            events["generate"] = kwargs
            class Output:
                def __getitem__(self, key):
                    assert key == (0, slice(20, None))
                    return [7, 8]
            return Output()

    class Generation:
        def __init__(self, **kwargs):
            self.values = kwargs

        def to_dict(self):
            return self.values

    def load(model, **kwargs):
        events["load"] = (model, kwargs)
        instance = Model()
        instance.device = "cuda"
        events["instance"] = instance
        return instance

    torch = SimpleNamespace(
        float16="fp16", bfloat16="bf16", manual_seed=lambda seed: events.update(seed=seed),
        use_deterministic_algorithms=lambda flag, **kw: events.update(deterministic=flag, deterministic_options=kw), inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                             synchronize=lambda: None, get_device_name=lambda _: "mock GPU",
                             get_device_properties=lambda _: SimpleNamespace(total_memory=40 * 1024**3),
                             get_device_capability=lambda _: (8, 0), mem_get_info=lambda _: (39 * 1024**3, 40 * 1024**3),
                             reset_peak_memory_stats=lambda _: events.update(memory_reset=True),
                             max_memory_allocated=lambda _: 123456, max_memory_reserved=lambda _: 234567),
        version=SimpleNamespace(cuda="mock CUDA"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 1), cuda=SimpleNamespace(matmul=SimpleNamespace())),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load), GenerationConfig=Generation))
    monkeypatch.setattr(bm, "_versions", lambda: {"python": "mock", "transformers": "4.57.1"})
    return events, torch


@pytest.mark.parametrize("model,pad,eos,dtype", [("a/model", None, 19, "float16"),
                                               ("b/model", 0, 7, "bfloat16"),
                                               ("c/model", None, None, "float16"),
                                               ("d/model", 3, 4, "float16")])
def test_real_loader_wiring_is_model_agnostic(monkeypatch, torch_runtime, model, pad, eos, dtype):
    events, torch = torch_runtime
    config = bm.RunConfig(model, SHA, 5, seed=23, max_new_tokens=512, dtype=dtype)
    tokenizer = Tokenizer()
    tokenizer.pad_token_id, tokenizer.eos_token_id = pad, eos
    preflight = {"resolved_revision": SHA, "context_limit": 4096,
                 "chat_template_configuration": {"tokenize": True, "add_generation_prompt": True, "truncation": False}}
    generate, metadata = bm._load_generator(config, tokenizer, preflight, None)
    assert generate(zero_shot.build_messages("A", "hasArea")) == '["result"]'
    assert events["load"][0] == model
    assert events["load"][1] == {"revision": SHA, "cache_dir": None, "trust_remote_code": False,
                                 "torch_dtype": getattr(torch, dtype), "attn_implementation": "eager", "use_safetensors": True}
    assert events["seed"] == 23 and events["deterministic"] and events["eval"]
    assert metadata["generation"] == {"do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                                       "max_new_tokens": 512, "use_cache": True, "eos_token_id": [11, 12],
                                       "pad_token_id": pad if pad is not None else (eos if eos is not None else 11)}
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert metadata["versions"]["transformers"] == "4.57.1"


@pytest.mark.parametrize("problem", ("no_cuda", "no_bf16"))
def test_hardware_failure_before_weights(torch_runtime, problem):
    events, torch = torch_runtime
    if problem == "no_cuda":
        torch.cuda.is_available = lambda: False
    else:
        torch.cuda.is_bf16_supported = lambda: False
    with pytest.raises(bm.Error):
        bm._load_generator(bm.RunConfig("org/model", SHA, 0, dtype="bfloat16"), Tokenizer(), {}, None)
    assert "load" not in events


def test_report_uses_official_metrics_and_checks_integrity(data, tmp_path, cpu_runtime):
    from llm_kbc.evaluation.lm_kbc_2026 import LMKBC2026Evaluator

    directory, dataset = data
    gold = [{**row, "ObjectEntities": []} for row in [json.loads(line) for line in dataset.read_text().splitlines()]]
    dataset = directory / "val.jsonl"
    write_jsonl(dataset, gold)
    cpu_runtime["responses"] = ["24522⟩", "[]", "[]", "[]", "[]", "[]"]
    meta = bm.run(dataset, directory, tmp_path / "runs", "report", bm.RunConfig("org/model", SHA, 0))
    predictions = Path(meta["output"])
    result = bm.report(directory, "val", predictions)
    direct = LMKBC2026Evaluator().evaluate(data_dir=directory, split="val", predictions_path=predictions)
    assert result["evaluation"]["overall"] == direct.overall
    assert result["evaluation"]["per_relation"] == direct.per_relation
    assert result["format_validity_rate"] == 5 / 6
    assert result["evaluation"]["overall"]["macro-f1"] == 1.0  # Empty gold + abstention: official behavior.
    assert result["inference_runtime_seconds"] == meta["inference_runtime_seconds"]
    with pytest.raises(bm.Error, match="unlabeled"):
        bm.report(directory, "test", predictions)
    raw_path = Path(str(predictions) + ".raw.jsonl")
    raw_path.write_text(raw_path.read_text().replace("24522", "12345"))
    # Both malformed strings still mean abstention; changing to valid data must fail.
    raw_path.write_text(raw_path.read_text().replace("12345\\u27e9", "[]"))
    with pytest.raises(bm.Error):
        bm.report(directory, "val", predictions)


def test_report_rejects_partial_smoke(data, tmp_path, cpu_runtime):
    directory, dataset = data
    val = directory / "val.jsonl"
    val.write_bytes(dataset.read_bytes())
    meta = bm.run(val, directory, tmp_path / "runs", "smoke", bm.RunConfig("org/model", SHA, 0), max_examples=2)
    with pytest.raises(bm.Error, match="complete scoring split"):
        bm.report(directory, "val", Path(meta["output"]))


def test_cli_help_and_dry_run_do_not_import_heavy_dependencies(data, tmp_path):
    directory, dataset = data
    script = """
import sys
sys.modules['torch'] = None
sys.modules['transformers'] = None
sys.modules['huggingface_hub'] = None
from llm_kbc.benchmark import main
raise SystemExit(main(sys.argv[1:]))
"""
    for arguments in (["--help"], ["run", "--help"], ["report", "--help"],
                      ["run", "--model", "org/model", "--revision", SHA, "--shots", "0", "--dataset", str(dataset),
                       "--run-id", "dry", "--output-root", str(tmp_path / "runs"), "--dry-run"]):
        completed = subprocess.run([sys.executable, "-B", "-c", script, *arguments], capture_output=True, text=True, timeout=15)
        assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "runs").exists()


def install_quantization_standins(monkeypatch):
    class BitsConfig:
        def __init__(self, **kwargs):
            self.values = kwargs

        def to_dict(self):
            return self.values

    monkeypatch.setitem(sys.modules, "accelerate", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace())
    monkeypatch.setattr(sys.modules["transformers"], "BitsAndBytesConfig", BitsConfig, raising=False)


@pytest.mark.parametrize("model", bm.TARGETS)
@pytest.mark.parametrize("quantization", ("none", "nf4"))
def test_target_architecture_loading_and_quantization(torch_runtime, monkeypatch, model, quantization):
    events, torch = torch_runtime
    fake = sys.modules["transformers"]
    model_type, class_name, _ = bm.TARGETS[model]
    setattr(fake, class_name, fake.AutoModelForCausalLM)
    # Use a larger synthetic GPU for the unquantized wiring test; 40 GB rejection is tested separately.
    torch.cuda.mem_get_info = lambda _: (100 * 1024**3, 100 * 1024**3)
    install_quantization_standins(monkeypatch)
    config = bm.RunConfig(model, SHA, 3, dtype="bfloat16", quantization=quantization)
    preflight = {"model_type": model_type, "context_limit": 4096,
                 "chat_template_configuration": {"tokenize": True, "add_generation_prompt": True, "truncation": False}}
    generate, metadata = bm._load_generator(config, Tokenizer(), preflight, None)
    assert generate(zero_shot.build_messages("A", "hasArea")) == '["result"]'
    assert metadata["loading_class"] == class_name
    assert events["load"][1]["revision"] == SHA
    assert metadata["generation"]["eos_token_id"] == [11, 12]
    assert metadata["gpu_total_memory_bytes"] == 40 * 1024**3
    assert metadata["gpu_compute_capability"] == [8, 0]
    assert metadata["gpu_memory"] == {"peak_allocated_bytes": 123456, "peak_reserved_bytes": 234567}
    assert events["memory_reset"]
    if quantization == "nf4":
        expected = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": "bf16"}
        assert metadata["quantization"] == expected
        assert events["load"][1]["quantization_config"].to_dict() == expected
        assert events["load"][1]["device_map"] == {"": 0}
        assert "to" not in events  # Moving an already dispatched quantized model is forbidden.
    else:
        assert metadata["quantization"] is None and events["to"] == "cuda"


@pytest.mark.parametrize("model", list(bm.TARGETS)[1:])
def test_40gb_rejects_impossible_unquantized_targets_before_weights(torch_runtime, model):
    events, _ = torch_runtime
    model_type, name, _ = bm.TARGETS[model]
    fake = sys.modules["transformers"]
    setattr(fake, name, fake.AutoModelForCausalLM)
    with pytest.raises(bm.Error, match="weights alone exceed"):
        bm._load_generator(bm.RunConfig(model, SHA, 0, dtype="bfloat16"), Tokenizer(), {"model_type": model_type}, None)
    assert "load" not in events


@pytest.mark.parametrize("model", ("Qwen/Qwen3.5-9B", "google/gemma-4-31B-it"))
def test_legacy_environment_fails_before_config_download(data, cpu_runtime, model):
    directory, dataset = data
    config = bm.RunConfig(model, SHA, 0)
    prompts = bm.prepare(dataset, directory, config)["prompts"]
    with pytest.raises(bm.Error, match="isolated benchmark-modern"):
        bm._tokenizer_preflight(config, prompts, None)
    assert "config" not in cpu_runtime and "tokenizer" not in cpu_runtime


@pytest.mark.parametrize("model", ("Qwen/Qwen3.5-9B", "google/gemma-4-31B-it"))
@pytest.mark.parametrize("thinking", ("default", "enabled", "disabled"))
def test_modern_text_only_native_chat_and_thinking(data, cpu_runtime, monkeypatch, model, thinking):
    directory, dataset = data
    fake = sys.modules["transformers"]
    model_type, name, _ = bm.TARGETS[model]
    setattr(fake, name, SimpleNamespace())
    fake.AutoConfig.from_pretrained = lambda *a, **kw: SimpleNamespace(
        _commit_hash=SHA, model_type=model_type, architectures=[name],
        get_text_config=lambda: SimpleNamespace(max_position_embeddings=262144))
    monkeypatch.setattr(Tokenizer, "get_chat_template", lambda self: "native enable_thinking template")
    chats = []
    apply = Tokenizer.apply_chat_template

    def capture(self, messages, **kwargs):
        chats.append(kwargs)
        return apply(self, messages, **kwargs)

    monkeypatch.setattr(Tokenizer, "apply_chat_template", capture)
    config = bm.RunConfig(model, SHA, 5, thinking=thinking)
    prompts = bm.prepare(dataset, directory, config)["prompts"]
    _, preflight = bm._tokenizer_preflight(config, prompts, None)
    assert preflight["model_type"] == model_type and preflight["loading_class"] == name
    assert preflight["vision_weights_loaded"] and preflight["text_only_inputs"]
    assert preflight["architectures"] == [name]
    assert preflight["chat_backend"] == "native-jinja"
    assert len(preflight["prompt_token_hashes"]) == len(prompts)
    assert all(row["rendered_chat"] and row["input_ids_sha256"] for row in prompts)
    if thinking == "default":
        assert all("enable_thinking" not in kw for kw in chats)
    else:
        assert all(kw["enable_thinking"] == (thinking == "enabled") for kw in chats)


@pytest.mark.parametrize("legacy_api", (True, False))
def test_mistral_native_tekken_chat_without_jinja(data, cpu_runtime, monkeypatch, tmp_path, legacy_api):
    directory, dataset = data
    model = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    fake = sys.modules["transformers"]
    setattr(fake, "Mistral3For" "ConditionalGeneration", SimpleNamespace())
    fake.AutoConfig.from_pretrained = lambda *a, **kw: SimpleNamespace(
        _commit_hash=SHA, model_type="mistral3", architectures=["Mistral3For" "ConditionalGeneration"],
        get_text_config=lambda: SimpleNamespace(max_position_embeddings=131072))
    tekken = tmp_path / "tekken.json"
    tekken.write_text("fixture tokenizer", encoding="utf-8")
    events = {}

    def download(repo, filename, **kwargs):
        events["download"] = (repo, filename, kwargs)
        return str(tekken)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))

    class NativeTokenizer:
        # Real 4.57.1 MistralCommonTokenizer has no get_chat_template/special_tokens_map.
        model_max_length = 10**30
        eos_token_id = 2
        pad_token_id = -1

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert Path(path) == tekken.parent
            assert kwargs == {"local_files_only": True}
            return cls()

        def apply_chat_template(self, messages, *, tokenize=True, truncation=False, return_dict=False):
            assert not truncation and not return_dict
            events.setdefault("messages", []).append(messages)
            return list(range(20)) if tokenize else "\n".join(row["content"] for row in messages)

    if not legacy_api:
        legacy_apply = NativeTokenizer.apply_chat_template

        def modern(self, messages, *, add_generation_prompt=False, **kwargs):
            assert add_generation_prompt
            return legacy_apply(self, messages, **kwargs)

        NativeTokenizer.apply_chat_template = modern
    setattr(fake, "MistralCommonTokenizer" if legacy_api else "MistralCommonBackend", NativeTokenizer)
    config = bm.RunConfig(model, SHA, 3)
    prompts = bm.prepare(dataset, directory, config)["prompts"]
    _, preflight = bm._tokenizer_preflight(config, prompts, None)
    assert events["download"] == (model, "tekken.json", {"revision": SHA, "cache_dir": None})
    assert preflight["chat_backend"] == "mistral-common"
    assert preflight["chat_template"] is preflight["chat_template_sha256"] is None
    assert preflight["tokenizer_file_sha256"] == bm._sha256(tekken)
    assert preflight["special_tokens_map"] is None
    assert preflight["context_limit"] == 131072
    assert ("add_generation_prompt" in preflight["chat_template_configuration"]) != legacy_api
    with pytest.raises(bm.Error, match="no enable_thinking switch"):
        bm._tokenizer_preflight(replace(config, thinking="disabled"), prompts, None)


def test_missing_quantization_dependency_never_loads_weights(torch_runtime, monkeypatch):
    events, _ = torch_runtime
    monkeypatch.setitem(sys.modules, "accelerate", None)
    with pytest.raises(ImportError):
        bm._load_generator(bm.RunConfig("org/model", SHA, 0, quantization="nf4"), Tokenizer(), {}, None)
    assert "load" not in events


@pytest.mark.parametrize("determinism", ("strict", "warn"))
def test_determinism_warnings_and_peak_memory_survive_generation_errors(torch_runtime, determinism):
    events, torch = torch_runtime
    config = bm.RunConfig("org/model", SHA, 0, determinism=determinism)
    generate, metadata = bm._load_generator(config, Tokenizer(), {
        "context_limit": 4096, "chat_template_configuration": {"tokenize": True, "truncation": False}}, None)
    assert events["deterministic_options"] == ({"warn_only": True} if determinism == "warn" else {})

    def failure(**kwargs):
        warnings.warn("synthetic nondeterministic kernel warning", RuntimeWarning)
        raise RuntimeError("synthetic GPU OOM")

    events["instance"].generate = failure
    torch.cuda.max_memory_allocated = lambda _: 987654
    with pytest.raises(RuntimeError, match="synthetic GPU OOM"):
        generate(zero_shot.build_messages("A", "hasArea"))
    assert metadata["gpu_memory"]["peak_allocated_bytes"] == 987654
    assert metadata["runtime_warnings"] == [{"category": "RuntimeWarning", "message": "synthetic nondeterministic kernel warning"}]
    assert "do not guarantee" in metadata["determinism_caveat"]


@pytest.mark.parametrize("field,value", [("thinking", "auto"), ("quantization", "int8"), ("determinism", "off")])
def test_invalid_loading_configuration(field, value):
    with pytest.raises(bm.Error):
        replace(bm.RunConfig("org/model", SHA, 0), **{field: value})


def test_new_cli_options_forwarded_without_inference(data, cpu_runtime, tmp_path, monkeypatch, capsys):
    _, dataset = data
    seen = {}

    def fake_run(*args, **kwargs):
        seen["config"] = args[4]
        return {"status": "dry_run"}

    monkeypatch.setattr(bm, "run", fake_run)
    assert bm.main(["run", "--model", "org/model", "--revision", SHA, "--shots", "3", "--dataset", str(dataset),
                    "--run-id", "nf4", "--dtype", "bfloat16", "--quantization", "nf4", "--thinking", "disabled",
                    "--determinism", "warn", "--dry-run"]) == 0
    assert seen["config"].quantization == "nf4" and seen["config"].thinking == "disabled"
    assert seen["config"].determinism == "warn" and seen["config"].max_new_tokens == 512


@pytest.mark.parametrize("matching", (True, False))
def test_generation_checks_actual_token_ids_against_preflight(torch_runtime, matching):
    events, _ = torch_runtime
    messages = zero_shot.build_messages("A", "hasArea")
    token_hash = hashlib.sha256(json.dumps(list(range(20))).encode("utf-8")).hexdigest()
    preflight = {"context_limit": 4096,
                 "chat_template_configuration": {"tokenize": True, "add_generation_prompt": True, "truncation": False},
                 "prompt_token_hashes": {bm._chat_hash(messages): token_hash if matching else "0" * 64}}
    generate, _ = bm._load_generator(bm.RunConfig("org/model", SHA, 0), Tokenizer(), preflight, None)
    if matching:
        assert generate(messages) == '["result"]'
    else:
        with pytest.raises(bm.Error, match="token IDs changed"):
            generate(messages)
        assert "generate" not in events


def test_modern_dependency_group_is_explicit_and_leaves_baseline_pins():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    groups = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert "transformers==4.57.1" in groups["inference"]
    assert "transformers==4.57.1" in groups["fine-tuning"]
    assert "huggingface-hub>=0.34,<1" in groups["inference"]
    assert "transformers==5.5.0" in groups["benchmark-modern"]
    assert "torch==2.8.0" in groups["benchmark-modern"]
    assert "accelerate==1.10.1" in groups["benchmark-modern"]
    assert "bitsandbytes==0.48.1" in groups["benchmark-modern"]


def test_missing_dependency_message_does_not_recommend_downgrading(data, cpu_runtime, tmp_path, monkeypatch):
    directory, dataset = data
    monkeypatch.setattr(bm, "_tokenizer_preflight", lambda *a: (_ for _ in ()).throw(ImportError("missing dependency")))
    with pytest.raises(bm.Error, match="isolated benchmark-modern"):
        bm.run(dataset, directory, tmp_path / "runs", "missing", bm.RunConfig("Qwen/Qwen3.5-9B", SHA, 0))
    assert not (tmp_path / "runs").exists()


def test_public_mistral_identifier_and_loading_configuration_are_exact():
    model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    assert bm.TARGETS[model_id] == ("mistral3", "Mistral3For" "ConditionalGeneration", 24_000_000_000)
    assert bm.CONDITIONAL_CLASSES["mistral3"] == "Mistral3For" "ConditionalGeneration"
