"""Single-GPU, training-only QLoRA v1; initial hyperparameters, not optimized ones.

The existing tokenizer masks the complete system/user/assistant-header prefix.
Its JSON answer, end-of-turn, and template-provided trailing newline remain
supervised. No labels are rebuilt, shifted, packed, or truncated here. Trainer
and Qwen's causal-LM loss perform the standard causal shift. No evaluation runs.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import distributions, version
from pathlib import Path
from types import SimpleNamespace

from . import fine_tuning_data, fine_tuning_tokens, zero_shot
from .dataset_inspector import DatasetError


REQUIRED_VERSIONS = {
    "torch": "2.8.0", "transformers": "4.57.1", "peft": "0.17.1",
    "accelerate": "1.10.1", "bitsandbytes": "0.48.1",
}
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class TrainingError(ValueError):
    """Configuration, environment, or artifacts prevent an auditable training run."""


@dataclass(frozen=True)
class TrainingConfig:
    """Nominal effective batch = micro batch * accumulation on exactly one GPU."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: float = 3.0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 2048
    gradient_checkpointing: bool = True
    optimizer: str = "adamw_torch"
    seed: int = 42
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_examples: int | None = None
    max_steps: int = -1

    def __post_init__(self):
        for name in ("rank", "alpha", "micro_batch_size", "gradient_accumulation_steps", "max_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise TrainingError(f"{name} must be a positive integer")
        for name in ("dropout", "learning_rate", "epochs", "weight_decay", "warmup_ratio", "max_grad_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise TrainingError(f"{name} must be a finite number")
        if not 0 <= self.dropout < 1 or not 0 <= self.warmup_ratio <= 1:
            raise TrainingError("dropout must be in [0,1); warmup_ratio must be in [0,1]")
        if min(self.learning_rate, self.epochs, self.max_grad_norm) <= 0 or self.weight_decay < 0:
            raise TrainingError("Learning rate, epochs, and max_grad_norm must be positive; weight_decay nonnegative")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise TrainingError("seed must be an integer in [0, 2**32)")
        if type(self.gradient_checkpointing) is not bool or self.optimizer != "adamw_torch":
            raise TrainingError("Require boolean gradient_checkpointing and optimizer='adamw_torch'")
        if self.max_examples is not None and (type(self.max_examples) is not int or self.max_examples < 1):
            raise TrainingError("max_examples must be a positive integer or null")
        if type(self.max_steps) is not int or (self.max_steps != -1 and self.max_steps < 1):
            raise TrainingError("max_steps must be -1 (epochs) or a positive integer")

    @classmethod
    def smoke(cls):
        """First eight records, two optimizer steps, nominal batch four; no tuning."""
        return cls(epochs=1.0, gradient_accumulation_steps=4, max_examples=8, max_steps=2)


def select_training_data(data_dir: Path, config: TrainingConfig) -> list[dict]:
    """Validate the complete train file, then select a deterministic ordered prefix."""
    return fine_tuning_data.prepare_training_data(data_dir)[:config.max_examples]


def tensor_collator(features, *, tokenizer, torch):
    """Preserve existing padding and loss labels; convert only at Trainer's boundary."""
    return {
        key: torch.tensor(values, dtype=torch.long, device="cpu")
        for key, values in fine_tuning_tokens.collate_batch(features, tokenizer).items()
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True,
                               capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise TrainingError("A Git checkout is required to record source provenance") from error
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise TrainingError("Could not establish source Git commit")
    paths = [Path(__file__), Path(fine_tuning_data.__file__), Path(fine_tuning_tokens.__file__),
             Path(zero_shot.__file__), Path(__file__).with_name("prediction_io.py"),
             Path(__file__).with_name("dataset_inspector.py"), Path(__file__).with_name("few_shot.py"),
             root / "pyproject.toml"]
    return {"commit": commit, "working_tree_dirty": bool(dirty),
            "files_sha256": {str(path.relative_to(root)): _sha256(path) for path in paths}}


def _load_runtime():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Before torch/CUDA initialization.
    try:
        import torch
        import transformers
        import peft
        import accelerate
        import bitsandbytes
    except (ImportError, OSError) as error:
        raise TrainingError('GPU dependencies unavailable; use the reviewed ".[fine-tuning]" environment') from error
    for package, expected in REQUIRED_VERSIONS.items():
        actual = version(package)
        if actual.split("+")[0] != expected:
            raise TrainingError(f"Require {package}=={expected}; found {actual}")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise TrainingError("Exactly one visible CUDA GPU and one process are required")
    if not torch.cuda.is_available() or not torch.version.cuda or getattr(torch.version, "hip", None):
        raise TrainingError("NVIDIA CUDA is required; CPU/ROCm training is unsupported")
    if torch.cuda.get_device_capability(0)[0] < 8 or not torch.cuda.is_bf16_supported():
        raise TrainingError("Native BF16 CUDA hardware (Ampere or newer) is required; no FP16 fallback")
    return SimpleNamespace(torch=torch, transformers=transformers, peft=peft)


def _paths(data_dir, output_dir, checkpoint_dir, cache_dir, tmp_dir):
    """Require fresh run/checkpoint directories; protect source, data, and caches."""
    output = Path(output_dir).expanduser()
    checkpoints = Path(checkpoint_dir).expanduser() if checkpoint_dir is not None else output / "checkpoints"
    for path in (output, checkpoints):
        if path.exists() or path.is_symlink():
            raise TrainingError(f"Refusing existing experiment path: {path}")
    output, checkpoints = output.resolve(), checkpoints.resolve()
    cache = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    temporary = Path(tmp_dir).expanduser().resolve() if tmp_dir is not None else None
    if output.is_relative_to(checkpoints):
        raise TrainingError("Checkpoint directory must not equal or contain the output directory")
    root = Path(__file__).resolve().parents[2]
    protected = [Path(data_dir).resolve(), *(root / name for name in ("data", "src", "tests", "docs", "configs", ".git"))]
    for path in (output, checkpoints, cache, temporary):
        if path is not None and any(path.is_relative_to(base) for base in protected):
            raise TrainingError(f"Refusing writes inside data/source directories: {path}")
    for path in (cache, temporary):
        if path is not None and any(path.is_relative_to(run) or run.is_relative_to(path) for run in (output, checkpoints)):
            raise TrainingError("Cache/tmp directories must be separate from experiment directories")
    return output, checkpoints, cache, temporary


@contextmanager
def _temporary_files(directory):
    """Scope library temporary-file settings; retain scratch files for diagnosis."""
    names = ("TMPDIR", "TEMP", "TMP")
    previous, previous_temp = {name: os.environ.get(name) for name in names}, tempfile.tempdir
    try:
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            os.environ.update({name: str(directory) for name in names})
            tempfile.tempdir = str(directory)
        yield
    finally:
        tempfile.tempdir = previous_temp
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _write_metadata(path: Path, record: dict, *, initial=False):
    text = json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if initial:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
        return
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        os.replace(temporary, path)  # Only this run's exclusively reserved metadata.
    finally:
        temporary.unlink(missing_ok=True)


def run_training(
    data_dir: Path, output_dir: Path, *, config: TrainingConfig = TrainingConfig(),
    cache_dir: Path | None = None, checkpoint_dir: Path | None = None, tmp_dir: Path | None = None,
    expected_train_sha256: str | None = None,
) -> dict:
    """Train from scratch into fresh directories, saving adapters only on success.

    Existing (even empty) run/checkpoint directories are refused. No auto-resume,
    Hub publication, validation, or base-model saving/merging. Interrupted runs
    retain metadata/checkpoints and possibly adapter.incomplete. Only a fully
    saved adapter is renamed to adapter; metadata status is authoritative.
    SIGKILL/power loss can leave status='running', even after adapter publication.
    An optional expected training hash is checked before GPU dependencies or
    directory creation. No production dataset hash is imposed by default.
    """
    if not isinstance(config, TrainingConfig):
        raise TrainingError("config must be a validated TrainingConfig")
    if expected_train_sha256 is not None:
        if (
            not isinstance(expected_train_sha256, str) or len(expected_train_sha256) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in expected_train_sha256)
        ):
            raise TrainingError("expected_train_sha256 must contain exactly 64 hexadecimal characters")
        expected_train_sha256 = expected_train_sha256.lower()
    output, checkpoints, cache, temporary = _paths(data_dir, output_dir, checkpoint_dir, cache_dir, tmp_dir)
    train = Path(data_dir) / "train.jsonl"
    train_hash = _sha256(train)
    if expected_train_sha256 is not None and train_hash != expected_train_sha256:
        raise TrainingError(
            f"Training dataset SHA256 mismatch for {train}: expected {expected_train_sha256}, actual {train_hash}"
        )
    examples = select_training_data(data_dir, config)
    source = _source_provenance()
    runtime = _load_runtime()
    torch, tf, peft = runtime.torch, runtime.transformers, runtime.peft
    record = {
        "experiment": "lm-kbc-2026-qlora-v1", "status": "running", "source": source,
        "model_id": zero_shot.DEFAULT_MODEL, "model_revision": zero_shot.DEFAULT_REVISION,
        "prompt_version": zero_shot.PROMPT_VERSION, "training_config": asdict(config),
        "effective_batch_size": config.micro_batch_size * config.gradient_accumulation_steps,
        "partial_final_batch": "retained; nominal effective batch may be smaller on the last update",
        "dataset": {"path": str(train.resolve()), "sha256": train_hash, "expected_sha256": expected_train_sha256},
        "selection": {"strategy": "first_records_in_file_order", "records": len(examples),
                      "identities": [{key: example[key] for key in ("SubjectEntity", "Relation")} for example in examples]},
        "loss_masking": "assistant-json-plus-end-of-turn-and-template-trailing-newline; prefix/padding=-100",
        "data_order": "SequentialSampler, original file order repeated each epoch; workers=0; no packing",
        "quantization": {"bits": 4, "type": "nf4", "double_quantization": True, "compute_dtype": "bfloat16"},
        "lora_target_modules": list(TARGET_MODULES), "attention_implementation": "eager",
        "determinism": {"seed": config.seed, "algorithms": True, "tf32": False,
                        "cublas_workspace_config": ":4096:8", "cross_hardware_bitwise_reproducibility": False},
        "paths": {"output": str(output), "checkpoints": str(checkpoints),
                  "cache": str(cache) if cache else None, "tmp": str(temporary) if temporary else None},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "gpu": torch.cuda.get_device_name(0),
                        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in distributions()))},
    }
    output.mkdir(parents=True, exist_ok=False)  # Atomic reservation before loading weights.
    metadata_path = output / "metadata.json"
    _write_metadata(metadata_path, record, initial=True)
    try:
        checkpoints.mkdir(parents=True, exist_ok=False)
        with _temporary_files(temporary):
            tf.set_seed(config.seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            tokenizer = tf.AutoTokenizer.from_pretrained(
                zero_shot.DEFAULT_MODEL, revision=zero_shot.DEFAULT_REVISION,
                cache_dir=str(cache) if cache else None, trust_remote_code=False,
            )
            features = fine_tuning_tokens.tokenize_training_data(examples, tokenizer, max_length=config.max_length)
            # Preflight collator/pad token before downloading weights.
            fine_tuning_tokens.collate_batch(features[:1], tokenizer)
            if _sha256(train) != train_hash:
                raise TrainingError("Training file changed during preparation")
            record["tokenization"] = {
                "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode("utf-8")).hexdigest(),
                "min_tokens": min(len(row["input_ids"]) for row in features),
                "max_tokens": max(len(row["input_ids"]) for row in features),
                "supervised_tokens": sum(label != -100 for row in features for label in row["labels"]),
                "features_sha256": hashlib.sha256(json.dumps(features, sort_keys=True).encode("utf-8")).hexdigest(),
            }
            quantization = tf.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = tf.AutoModelForCausalLM.from_pretrained(
                zero_shot.DEFAULT_MODEL, revision=zero_shot.DEFAULT_REVISION,
                cache_dir=str(cache) if cache else None, quantization_config=quantization,
                torch_dtype=torch.bfloat16, attn_implementation="eager", device_map={"": 0},
                trust_remote_code=False, use_safetensors=True,
            )
            if model.config._commit_hash != zero_shot.DEFAULT_REVISION:
                raise TrainingError("Loaded base model revision differs from the pinned revision")
            if not getattr(model, "is_loaded_in_4bit", False):
                raise TrainingError("Base model was not loaded in 4-bit mode")
            if config.max_length > model.config.max_position_embeddings:
                raise TrainingError("max_length exceeds the pinned model context window")
            model.config.use_cache = False
            model = peft.prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            lora = peft.LoraConfig(
                task_type=peft.TaskType.CAUSAL_LM, r=config.rank, lora_alpha=config.alpha,
                lora_dropout=config.dropout, target_modules=list(TARGET_MODULES), bias="none",
                inference_mode=False, revision=zero_shot.DEFAULT_REVISION,
            )
            model = peft.get_peft_model(model, lora)
            trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            if not trainable or any(".lora_" not in name for name in trainable):
                raise TrainingError("Only LoRA adapter parameters may be trainable")
            record["trainable_parameters"], record["total_parameters"] = model.get_nb_trainable_parameters()
            args = tf.TrainingArguments(
                output_dir=str(checkpoints), overwrite_output_dir=False,
                num_train_epochs=config.epochs, max_steps=config.max_steps,
                per_device_train_batch_size=config.micro_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate, optim=config.optimizer,
                adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                weight_decay=config.weight_decay, max_grad_norm=config.max_grad_norm,
                lr_scheduler_type="linear", warmup_ratio=config.warmup_ratio,
                bf16=True, fp16=False, tf32=False, full_determinism=True,
                seed=config.seed, data_seed=config.seed,
                gradient_checkpointing=config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                eval_strategy="no", do_eval=False, label_names=["labels"], label_smoothing_factor=0.0,
                save_strategy="epoch", save_total_limit=None, save_safetensors=True,
                logging_strategy="steps", logging_steps=1, logging_nan_inf_filter=False,
                report_to=[], push_to_hub=False, remove_unused_columns=False,
                dataloader_num_workers=0, dataloader_drop_last=False, dataloader_pin_memory=False,
                group_by_length=False, auto_find_batch_size=False,
            )

            class OrderedTrainer(tf.Trainer):
                # Transformers 4.57.1 accepts an optional dataset argument here.
                def _get_train_sampler(self, train_dataset=None):
                    return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

            trainer = OrderedTrainer(
                model=model, args=args, train_dataset=features, processing_class=tokenizer,
                data_collator=partial(tensor_collator, tokenizer=tokenizer, torch=torch),
            )
            record["training_arguments"] = args.to_dict()
            _write_metadata(metadata_path, record)
            result = trainer.train(resume_from_checkpoint=False)
            results = {"metrics": result.metrics, "global_step": trainer.state.global_step,
                       "epoch": trainer.state.epoch, "log_history": trainer.state.log_history}
            json.dumps(results, allow_nan=False)  # Do not publish non-finite results as a successful run.
            if trainer.state.global_step < 1:
                raise TrainingError("Trainer completed without an optimizer step")
            record["training_results"] = results
            pending, adapter = output / "adapter.incomplete", output / "adapter"
            pending.mkdir(exist_ok=False)
            model.save_pretrained(pending, safe_serialization=True, save_embedding_layers=False)
            expected = ("adapter_config.json", "adapter_model.safetensors")
            if any(not (pending / name).is_file() or (pending / name).is_symlink() for name in expected):
                raise TrainingError("Saved adapter is incomplete")
            record["adapter_sha256"] = {name: _sha256(pending / name) for name in expected}
            if adapter.exists() or adapter.is_symlink():
                raise TrainingError("Refusing to overwrite adapter output")
            pending.rename(adapter)
            record["status"] = "complete"
    except BaseException as error:
        record["status"] = "runtime_error" if isinstance(error, Exception) else "interrupted"
        record["error"] = str(error) or type(error).__name__
        raise
    finally:
        _write_metadata(metadata_path, record)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=fine_tuning_data.DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True, help="New experiment directory; must not exist")
    parser.add_argument("--checkpoint-dir", type=Path, help="New checkpoint directory; default: output-dir/checkpoints")
    parser.add_argument("--cache-dir", type=Path, help="Shared Hugging Face cache; default: Hugging Face environment")
    parser.add_argument("--tmp-dir", type=Path, help="Library temporary-file root; default: system environment")
    parser.add_argument("--expected-train-sha256", help="Optional SHA256 pin for the complete train.jsonl file")
    parser.add_argument("--config", type=Path, help="JSON object with explicit TrainingConfig overrides")
    parser.add_argument("--smoke", action="store_true", help="First 8 train records, 2 steps, accumulation 4, 1 epoch")
    args = parser.parse_args(argv)
    try:
        if args.smoke and args.config:
            raise TrainingError("Use --smoke or --config, not both; no implicit hyperparameter overrides")
        values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(values, dict):
            raise TrainingError("Configuration must be a JSON object")
        config = TrainingConfig.smoke() if args.smoke else TrainingConfig(**values)
        run_training(args.data_dir, args.output_dir, config=config, cache_dir=args.cache_dir,
                     checkpoint_dir=args.checkpoint_dir, tmp_dir=args.tmp_dir,
                     expected_train_sha256=args.expected_train_sha256)
    except (DatasetError, TrainingError, OSError, RuntimeError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Completed adapter: {args.output_dir / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
