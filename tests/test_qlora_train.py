"""QLoRA wiring and safety with CPU stand-ins; never load weights or train."""

import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_kbc import qlora_train as train, zero_shot
from llm_kbc.dataset_inspector import DatasetError
from llm_kbc.fine_tuning_tokens import collate_batch, tokenize_training_data


@pytest.mark.parametrize("values", [
    {"rank": 0}, {"rank": True}, {"alpha": -1}, {"micro_batch_size": 1.5},
    {"gradient_accumulation_steps": 0}, {"max_length": None}, {"max_length": 0},
    {"learning_rate": 0}, {"learning_rate": float("nan")}, {"epochs": -1},
    {"dropout": 1}, {"dropout": -0.1}, {"weight_decay": -1}, {"warmup_ratio": 1.1},
    {"max_grad_norm": 0}, {"seed": -1}, {"seed": 2**32}, {"seed": True},
    {"gradient_checkpointing": "true"}, {"optimizer": "paged_adamw_8bit"},
    {"max_examples": 0}, {"max_examples": False}, {"max_steps": 0}, {"max_steps": 1.5},
])
def test_invalid_configuration_fails(values):
    with pytest.raises(train.TrainingError):
        train.TrainingConfig(**values)


def test_initial_and_explicit_smoke_configurations():
    config = train.TrainingConfig()
    assert (config.rank, config.alpha, config.dropout, config.learning_rate) == (16, 32, 0.05, 2e-4)
    assert (config.epochs, config.micro_batch_size, config.gradient_accumulation_steps) == (3, 1, 16)
    assert (config.max_length, config.seed, config.optimizer) == (2048, 42, "adamw_torch")
    assert config.gradient_checkpointing and config.max_examples is None and config.max_steps == -1
    smoke = train.TrainingConfig.smoke()
    assert (smoke.max_examples, smoke.max_steps, smoke.epochs, smoke.gradient_accumulation_steps) == (8, 2, 1, 4)
    assert train.TrainingConfig(**json.loads(json.dumps(asdict(config)))) == config


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    rows = [{"SubjectEntity": f"Subject {index}", "Relation": "hasArea",
             "ObjectEntities": [] if index == 0 else [["1.2300", "unused alias"]]} for index in range(16)]
    (directory / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (directory / "val.jsonl").write_text("VALIDATION_LABELS_MUST_NOT_BE_READ", encoding="utf-8")
    (directory / "test.jsonl").write_text("TEST_LABELS_MUST_NOT_BE_READ", encoding="utf-8")
    return directory


class Tokenizer:
    pad_token_id = 0

    def get_chat_template(self):
        return "unit-test-template"

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        prefix = [1, 2, 3]
        ids = prefix if add_generation_prompt else prefix + [ord(char) for char in messages[-1]["content"]] + [9, 10]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class FakeTensor:
    def __init__(self, values, dtype, device):
        self.values, self.dtype, self.device = copy.deepcopy(values), dtype, device


@pytest.fixture
def runtime(monkeypatch):
    events = {}
    torch = SimpleNamespace(
        long="long", bfloat16="bf16", tensor=lambda values, **kwargs: FakeTensor(values, **kwargs),
        version=SimpleNamespace(cuda="12.6", hip=None),
        use_deterministic_algorithms=lambda flag: events.update(deterministic=flag),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 9000), cuda=SimpleNamespace(matmul=SimpleNamespace())),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda: True,
                             get_device_capability=lambda _: (8, 0), get_device_name=lambda _: "fake A100",
                             get_device_properties=lambda _: SimpleNamespace(total_memory=40 * 1024**3)),
        utils=SimpleNamespace(data=SimpleNamespace(SequentialSampler=lambda rows: list(range(len(rows))))),
    )

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(_commit_hash=zero_shot.DEFAULT_REVISION, max_position_embeddings=32768, use_cache=True)
            self.is_loaded_in_4bit = True

        def named_parameters(self):
            return [("base.weight", SimpleNamespace(requires_grad=False)),
                    ("layer.q_proj.lora_A.default.weight", SimpleNamespace(requires_grad=True))]

        def get_nb_trainable_parameters(self):
            return 128, 1024

        def save_pretrained(self, path, **kwargs):
            events["save_adapter"] = (path, kwargs)
            (path / "adapter_config.json").write_text(json.dumps({"revision": zero_shot.DEFAULT_REVISION}))
            (path / "adapter_model.safetensors").write_bytes(b"synthetic unit-test adapter")

    model = Model()

    def load_model(*args, **kwargs):
        events["load_model"] = args, kwargs
        return model

    def load_tokenizer(*args, **kwargs):
        events["load_tokenizer"] = args, kwargs
        return Tokenizer()

    def prepare(model, **kwargs):
        events["prepare"] = kwargs
        return model

    def attach(model, config):
        events["lora"] = config
        return model

    class Arguments:
        def __init__(self, **kwargs):
            self.values = kwargs

        def to_dict(self):
            return self.values

    class Trainer:
        def __init__(self, **kwargs):
            events["trainer"] = self
            self.options = kwargs
            self.train_dataset = kwargs["train_dataset"]
            self.state = SimpleNamespace(global_step=2, epoch=1.0, log_history=[{"loss": 0.75, "step": 2}])

        def train(self, *, resume_from_checkpoint):
            events["resume"] = resume_from_checkpoint
            events["sampler"] = self._get_train_sampler(self.train_dataset)
            events["batch"] = self.options["data_collator"](self.train_dataset[:2])
            return SimpleNamespace(metrics={"train_loss": 0.75})

    tf = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        BitsAndBytesConfig=lambda **kwargs: kwargs, TrainingArguments=Arguments, Trainer=Trainer,
        set_seed=lambda seed: events.update(seed=seed),
    )
    peft = SimpleNamespace(prepare_model_for_kbit_training=prepare, LoraConfig=lambda **kwargs: kwargs,
                           TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"), get_peft_model=attach)
    fake = SimpleNamespace(torch=torch, transformers=tf, peft=peft, check_runtime=train._load_runtime)
    monkeypatch.setattr(train, "_load_runtime", lambda: fake)
    monkeypatch.setattr(train, "_source_provenance", lambda: {"commit": "a" * 40, "working_tree_dirty": True, "files_sha256": {}})
    monkeypatch.setattr(train, "distributions", lambda: [SimpleNamespace(metadata={"Name": name}, version=value)
                                                        for name, value in train.REQUIRED_VERSIONS.items()])
    return fake, model, events


def test_selection_is_training_only_deterministic_prefix_and_preserves_order(dataset, monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name not in ("val.jsonl", "test.jsonl")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    selected = train.select_training_data(dataset, train.TrainingConfig.smoke())
    assert [row["SubjectEntity"] for row in selected] == [f"Subject {index}" for index in range(8)]
    assert train.select_training_data(dataset, train.TrainingConfig.smoke()) == selected
    assert len(train.select_training_data(dataset, train.TrainingConfig())) == 16


def test_complete_training_file_is_validated_beyond_smoke_selection(dataset):
    with (dataset / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{bad JSON}\n")
    with pytest.raises(DatasetError, match="malformed JSON"):
        train.select_training_data(dataset, train.TrainingConfig.smoke())


def test_tensor_boundary_preserves_masking_labels_trailing_newline_and_padding(dataset, runtime):
    fake, _, _ = runtime
    tokenizer = Tokenizer()
    features = tokenize_training_data(train.select_training_data(dataset, train.TrainingConfig.smoke()), tokenizer, max_length=2048)
    before = copy.deepcopy(features)
    expected = collate_batch(features, tokenizer)
    actual = train.tensor_collator(features, tokenizer=tokenizer, torch=fake.torch)
    assert features == before
    for key in expected:
        assert actual[key].values == expected[key]
        assert actual[key].dtype == fake.torch.long and actual[key].device == "cpu"
    assert features[0]["labels"] == [-100, -100, -100, ord("["), ord("]"), 9, 10]


def test_mocked_end_to_end_pinned_qlora_and_metadata(dataset, tmp_path, runtime):
    fake, model, events = runtime
    output, checkpoints, cache, temporary = (tmp_path / name for name in ("run", "checkpoints", "cache", "tmp"))
    before = {path.name: path.read_bytes() for path in dataset.iterdir()}
    record = train.run_training(dataset, output, config=train.TrainingConfig.smoke(), checkpoint_dir=checkpoints,
                                cache_dir=cache, tmp_dir=temporary)
    assert record == json.loads((output / "metadata.json").read_text())
    assert record["status"] == "complete" and record["training_config"] == asdict(train.TrainingConfig.smoke())
    assert record["effective_batch_size"] == 4 and record["selection"]["records"] == 8
    assert record["dataset"]["sha256"] == hashlib.sha256(before["train.jsonl"]).hexdigest()
    assert record["dataset"]["expected_sha256"] is None  # Synthetic data does not inherit a production pin.
    assert record["model_revision"] == zero_shot.DEFAULT_REVISION and record["source"]["commit"] == "a" * 40
    assert record["training_results"]["global_step"] == 2 and record["training_results"]["metrics"]["train_loss"] == 0.75
    assert record["environment"]["packages"] == train.REQUIRED_VERSIONS
    assert record["tokenization"]["chat_template_sha256"] == hashlib.sha256(b"unit-test-template").hexdigest()
    assert events["load_model"][0] == events["load_tokenizer"][0] == (zero_shot.DEFAULT_MODEL,)
    kwargs = events["load_model"][1]
    assert kwargs["revision"] == events["load_tokenizer"][1]["revision"] == zero_shot.DEFAULT_REVISION
    assert kwargs["torch_dtype"] == fake.torch.bfloat16 and kwargs["attn_implementation"] == "eager"
    assert kwargs["device_map"] == {"": 0} and not kwargs["trust_remote_code"] and kwargs["use_safetensors"]
    assert kwargs["quantization_config"] == {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": fake.torch.bfloat16,
    }
    assert events["prepare"] == {"use_gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False}}
    assert events["lora"]["r"] == 16 and events["lora"]["lora_alpha"] == 32
    assert events["lora"]["target_modules"] == list(train.TARGET_MODULES)
    assert events["lora"]["bias"] == "none" and events["lora"]["revision"] == zero_shot.DEFAULT_REVISION
    assert events["seed"] == 42 and events["deterministic"] and events["resume"] is False
    assert events["sampler"] == list(range(8))
    args = events["trainer"].options["args"].to_dict()
    assert args["bf16"] and not args["fp16"] and not args["tf32"] and args["full_determinism"]
    assert args["eval_strategy"] == "no" and not args["do_eval"] and args["report_to"] == []
    assert args["optim"] == "adamw_torch" and args["label_smoothing_factor"] == 0
    assert not args["auto_find_batch_size"] and args["dataloader_num_workers"] == 0
    assert args["remove_unused_columns"] is False and args["label_names"] == ["labels"]
    assert "eval_dataset" not in events["trainer"].options
    assert model.config.use_cache is False
    assert events["save_adapter"][1] == {"safe_serialization": True, "save_embedding_layers": False}
    assert (output / "adapter").is_dir() and not (output / "adapter.incomplete").exists()
    for name, digest in record["adapter_sha256"].items():
        assert _hash(output / "adapter" / name) == digest
    assert {path.name: path.read_bytes() for path in dataset.iterdir()} == before


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("uppercase", [False, True])
def test_matching_expected_training_hash_accepts_synthetic_data_and_records_pin(dataset, tmp_path, runtime, uppercase):
    expected = _hash(dataset / "train.jsonl")
    supplied = expected.upper() if uppercase else expected
    record = train.run_training(
        dataset, tmp_path / "run", config=train.TrainingConfig.smoke(), expected_train_sha256=supplied,
    )
    assert record["dataset"]["expected_sha256"] == record["dataset"]["sha256"] == expected
    assert record["status"] == "complete"


def test_mismatching_expected_hash_fails_before_gpu_or_directory_creation(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "_load_runtime", lambda: pytest.fail("GPU dependencies must not initialize"))
    root = tmp_path / "artifacts"
    with pytest.raises(train.TrainingError, match="Training dataset SHA256 mismatch"):
        train.run_training(
            dataset, root / "run", checkpoint_dir=root / "checkpoints", cache_dir=root / "cache",
            tmp_dir=root / "tmp", expected_train_sha256="0" * 64,
        )
    assert not root.exists()


@pytest.mark.parametrize("expected", ["", "a" * 63, "a" * 65, "g" * 64, " " * 64,
                                        " " + "a" * 64, "a" * 64 + "\n", 123, True])
def test_malformed_expected_hash_is_rejected_before_file_reads_or_gpu(tmp_path, monkeypatch, expected):
    monkeypatch.setattr(train, "_load_runtime", lambda: pytest.fail("GPU dependencies must not initialize"))
    with pytest.raises(train.TrainingError, match="exactly 64 hexadecimal characters"):
        train.run_training(tmp_path / "missing-data", tmp_path / "run", expected_train_sha256=expected)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("where", ["output", "checkpoint"])
def test_existing_even_empty_paths_are_refused_before_runtime(dataset, tmp_path, monkeypatch, where):
    output, checkpoint = tmp_path / "run", tmp_path / "checkpoints"
    protected = output if where == "output" else checkpoint
    protected.mkdir()
    monkeypatch.setattr(train, "_load_runtime", lambda: pytest.fail("runtime should not load"))
    with pytest.raises(train.TrainingError, match="Refusing existing experiment"):
        train.run_training(dataset, output, checkpoint_dir=checkpoint)
    assert list(protected.iterdir()) == []


def test_unsafe_and_overlapping_paths_are_rejected(dataset, tmp_path):
    output = tmp_path / "run"
    for checkpoint, cache, temporary in ((output, None, None), (None, output / "cache", None),
                                          (None, None, output / "tmp")):
        with pytest.raises(train.TrainingError):
            train._paths(dataset, output, checkpoint, cache, temporary)
    with pytest.raises(train.TrainingError, match="data/source"):
        train._paths(dataset, dataset / "new", None, None, None)


@pytest.mark.parametrize("error", [RuntimeError("synthetic OOM"), KeyboardInterrupt()])
def test_failure_keeps_metadata_and_checkpoints_without_final_adapter(dataset, tmp_path, runtime, monkeypatch, error):
    fake, _, _ = runtime

    def fail(self, **kwargs):
        raise error

    monkeypatch.setattr(fake.transformers.Trainer, "train", fail)
    output = tmp_path / "run"
    with pytest.raises(type(error)):
        train.run_training(dataset, output, config=train.TrainingConfig.smoke())
    record = json.loads((output / "metadata.json").read_text())
    assert record["status"] == ("interrupted" if isinstance(error, KeyboardInterrupt) else "runtime_error")
    assert not (output / "adapter").exists() and (output / "checkpoints").is_dir()
    with pytest.raises(train.TrainingError, match="existing"):
        train.run_training(dataset, output)


@pytest.mark.parametrize("problem", ["revision", "quantization", "context", "base_trainable", "missing_adapter", "nonfinite"])
def test_invalid_model_or_results_cannot_publish_complete_adapter(dataset, tmp_path, runtime, monkeypatch, problem):
    fake, model, _ = runtime
    if problem == "revision": model.config._commit_hash = "b" * 40
    elif problem == "quantization": model.is_loaded_in_4bit = False
    elif problem == "context": model.config.max_position_embeddings = 1024
    elif problem == "base_trainable":
        model.named_parameters = lambda: [("base.weight", SimpleNamespace(requires_grad=True))]
    elif problem == "missing_adapter": model.save_pretrained = lambda *args, **kwargs: None
    else:
        monkeypatch.setattr(fake.transformers.Trainer, "train", lambda *args, **kwargs: SimpleNamespace(metrics={"train_loss": float("nan")}))
    output = tmp_path / "run"
    with pytest.raises((train.TrainingError, ValueError)):
        train.run_training(dataset, output, config=train.TrainingConfig.smoke())
    assert not (output / "adapter").exists()
    assert json.loads((output / "metadata.json").read_text())["status"] == "runtime_error"


def test_temporary_environment_is_restored_even_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", "original")
    with pytest.raises(RuntimeError):
        with train._temporary_files(tmp_path / "scratch"):
            assert os.environ["TMPDIR"] == str(tmp_path / "scratch")
            raise RuntimeError("test")
    assert os.environ["TMPDIR"] == "original"


@pytest.mark.parametrize("problem", ["version", "no_cuda", "multi_gpu", "bf16", "old_gpu", "world_size"])
def test_runtime_requires_reviewed_versions_and_single_native_bf16_gpu(runtime, monkeypatch, problem):
    fake, _, _ = runtime
    # Exercise the real dependency checker with CPU stand-ins for its imports.
    for name, module in (("torch", fake.torch), ("transformers", fake.transformers), ("peft", fake.peft),
                         ("accelerate", SimpleNamespace()), ("bitsandbytes", SimpleNamespace())):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(train, "version", lambda name: "0.0.0" if problem == "version" else train.REQUIRED_VERSIONS[name])
    monkeypatch.setenv("WORLD_SIZE", "2" if problem == "world_size" else "1")
    if problem == "no_cuda": fake.torch.cuda.is_available = lambda: False
    elif problem == "multi_gpu": fake.torch.cuda.device_count = lambda: 2
    elif problem == "bf16": fake.torch.cuda.is_bf16_supported = lambda: False
    elif problem == "old_gpu": fake.torch.cuda.get_device_capability = lambda _: (7, 5)
    with pytest.raises(train.TrainingError):
        fake.check_runtime()


def test_dependency_group_pins_and_existing_groups_unchanged():
    root = Path(__file__).resolve().parents[1]
    groups = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    for name, value in train.REQUIRED_VERSIONS.items():
        assert f"{name}=={value}" in groups["fine-tuning"]
    assert groups["dev"] == ["pytest>=8,<9"]
    assert groups["evaluation"] == ["pandas==2.2.3"]
    assert groups["inference"] == ["torch>=2.2,<3", "transformers==4.57.1", "huggingface-hub>=0.34,<1"]


def test_generated_training_arguments_match_installed_pinned_api(dataset, tmp_path, runtime):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.1":
        pytest.skip("Signature check requires the pinned Transformers version")
    parameters = inspect.signature(transformers.TrainingArguments).parameters
    train.run_training(dataset, tmp_path / "run", config=train.TrainingConfig.smoke())
    supplied = runtime[2]["trainer"].options["args"].to_dict()
    assert set(supplied).issubset(parameters)


def test_missing_gpu_dependency_has_helpful_error(runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(train.TrainingError, match="GPU dependencies unavailable"):
        runtime[0].check_runtime()


def test_runtime_accepts_cuda_wheel_suffix_without_changing_required_version(runtime, monkeypatch):
    fake, _, _ = runtime
    for name, module in (("torch", fake.torch), ("transformers", fake.transformers), ("peft", fake.peft),
                         ("accelerate", SimpleNamespace()), ("bitsandbytes", SimpleNamespace())):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(train, "version", lambda name: train.REQUIRED_VERSIONS[name] + ("+cu126" if name == "torch" else ""))
    assert fake.check_runtime().torch is fake.torch


def test_metadata_is_reserved_exclusively_then_updated_without_nan(tmp_path):
    path = tmp_path / "metadata.json"
    train._write_metadata(path, {"status": "running"}, initial=True)
    with pytest.raises(FileExistsError):
        train._write_metadata(path, {"status": "other"}, initial=True)
    assert json.loads(path.read_text()) == {"status": "running"}
    train._write_metadata(path, {"status": "complete"})
    assert json.loads(path.read_text()) == {"status": "complete"}
    with pytest.raises(ValueError):
        train._write_metadata(path, {"loss": float("nan")})
    assert json.loads(path.read_text()) == {"status": "complete"}
    assert list(tmp_path.iterdir()) == [path]


def test_source_provenance_includes_dirty_state_and_all_training_sources(monkeypatch):
    def git(args, **kwargs):
        return SimpleNamespace(stdout="c" * 40 if args[-1] == "HEAD" else "?? src/llm_kbc/qlora_train.py\n")

    monkeypatch.setattr(train.subprocess, "run", git)
    record = train._source_provenance()
    assert record["commit"] == "c" * 40 and record["working_tree_dirty"]
    assert len(record["files_sha256"]) == 8
    assert all(len(digest) == 64 for digest in record["files_sha256"].values())


def test_source_provenance_requires_git(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(train.subprocess, "run", unavailable)
    with pytest.raises(train.TrainingError, match="Git checkout"):
        train._source_provenance()


def test_cli_help_requires_no_gpu_dependencies():
    result = subprocess.run([sys.executable, "-m", "llm_kbc.qlora_train", "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--smoke" in result.stdout and "--cache-dir" in result.stdout and "--checkpoint-dir" in result.stdout
    assert "--expected-train-sha256" in result.stdout


def test_cli_forwards_explicit_production_hash_without_changing_smoke_config(tmp_path, monkeypatch):
    expected = "ad37cd30d1ff4b9f1ef2579b25e64093b202c40da11e8c412e13386f1e5d332e"
    calls = []
    monkeypatch.setattr(train, "run_training", lambda *args, **kwargs: calls.append(kwargs))
    assert train.main([
        "--output-dir", str(tmp_path / "run"), "--smoke", "--expected-train-sha256", expected,
    ]) == 0
    assert calls[0]["expected_train_sha256"] == expected
    assert calls[0]["config"] == train.TrainingConfig.smoke()


@pytest.mark.parametrize("expected", ["0" * 64, "malformed"])
def test_cli_hash_failures_never_initialize_gpu_or_create_outputs(dataset, tmp_path, monkeypatch, expected):
    monkeypatch.setattr(train, "_load_runtime", lambda: pytest.fail("GPU dependencies must not initialize"))
    with pytest.raises(SystemExit) as error:
        train.main([
            "--data-dir", str(dataset), "--output-dir", str(tmp_path / "run"),
            "--expected-train-sha256", expected,
        ])
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_cli_reads_explicit_configuration_without_running_training(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"rank": 8, "seed": 7, "epochs": 1.0}))
    calls = []
    monkeypatch.setattr(train, "run_training", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert train.main(["--output-dir", str(tmp_path / "run"), "--config", str(path)]) == 0
    config = calls[0][1]["config"]
    assert (config.rank, config.seed, config.epochs, config.max_length) == (8, 7, 1, 2048)
    with pytest.raises(SystemExit):
        train.main(["--output-dir", str(tmp_path / "run"), "--config", str(path), "--smoke"])


@pytest.mark.parametrize("text", ['{broken}', '[]', '{"unknown": 1}', '{"rank": 0}'])
def test_invalid_cli_configuration_never_starts_training(tmp_path, monkeypatch, text):
    path = tmp_path / "config.json"
    path.write_text(text)
    monkeypatch.setattr(train, "run_training", lambda *args, **kwargs: pytest.fail("Training must not start"))
    with pytest.raises(SystemExit) as error:
        train.main(["--output-dir", str(tmp_path / "run"), "--config", str(path)])
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_import_does_not_load_gpu_dependencies():
    result = subprocess.run([sys.executable, "-c", "import sys; import llm_kbc.qlora_train; "
                             "assert not {'torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes'} & sys.modules.keys()"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
