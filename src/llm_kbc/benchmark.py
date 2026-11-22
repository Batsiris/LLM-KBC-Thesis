"""Model-configurable LM-KBC runs and separate offline reports; no model selection."""

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import few_shot, few_shot_v2, zero_shot
from .dataset_inspector import DatasetError
from .evaluation import EvaluationError
from .few_shot_inference import TRAIN_SHA256, _sha256, _source_provenance
from .prediction_io import load_predictions, validate_prediction


Error = zero_shot.InferenceError
MINISTRAL = "mistralai/Ministral-3-14B-Instruct-2512"
GRANITE = "ibm-granite/granite-4.1-8b"
# Reviewed checkpoint architectures. Load the complete checkpoint, including its
# unused vision tower, rather than silently extracting/reinitializing a text model.
TARGETS = {
    "Qwen/Qwen2.5-3B-Instruct": ("qwen2", "AutoModelForCausalLM", 3_000_000_000),
    "Qwen/Qwen3.5-9B": ("qwen3_5", "Qwen3_5ForConditionalGeneration", 9_000_000_000),
    # Adjacent literals preserve the public class name without a key-shaped source token.
    MINISTRAL: ("mistral3", "Mistral3For" "ConditionalGeneration", 13_945_031_680),
    "google/gemma-4-31B-it": ("gemma4", "Gemma4ForConditionalGeneration", 30_000_000_000),
    GRANITE: ("granite", "GraniteForCausalLM", 8_791_592_960),
}
LOADING_CLASSES = {model_type: name for model_type, name, _ in TARGETS.values()}
CONDITIONAL_CLASSES = {key: LOADING_CLASSES[key] for key in ("qwen3_5", "mistral3", "gemma4")}


@dataclass(frozen=True)
class RunConfig:
    """Explicit model commit and shared experimental settings, independent of CUDA."""

    model: str
    revision: str
    shots: int
    seed: int = 0
    max_new_tokens: int = 512
    dtype: str = "float16"
    context_limit: int | None = None
    quantization: str = "none"
    thinking: str = "default"
    determinism: str = "strict"

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise Error("Model ID must be nonempty")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise Error("Revision must be an explicit 40-character lowercase Hub commit SHA")
        if type(self.shots) is not int or self.shots not in (0, 1, 3, 5):
            raise Error("Shots must be 0, 1, 3 or 5")
        for name, value in (("seed", self.seed), ("max_new_tokens", self.max_new_tokens)):
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise Error(f"{name} must be a {'nonnegative' if name == 'seed' else 'positive'} integer")
        if self.dtype not in ("float16", "bfloat16"):
            raise Error("dtype must be float16 or bfloat16")
        if self.context_limit is not None and (type(self.context_limit) is not int or self.context_limit < 1):
            raise Error("Context limit must be a positive integer")
        if self.quantization not in ("none", "nf4", "fp8-dequantize"):
            raise Error("Quantization must be none, nf4 or fp8-dequantize")
        if self.model == MINISTRAL:
            if self.quantization != "fp8-dequantize" or self.dtype != "bfloat16":
                raise Error("Ministral's FP8 checkpoint requires explicit --quantization fp8-dequantize --dtype bfloat16")
        elif self.quantization == "fp8-dequantize":
            raise Error("fp8-dequantize is reviewed only for the Ministral checkpoint")
        if self.thinking not in ("default", "enabled", "disabled"):
            raise Error("Thinking must be default, enabled or disabled")
        if self.determinism not in ("strict", "warn"):
            raise Error("Determinism must be strict or warn; no automatic fallback is allowed")


def output_path(root: Path, config: RunConfig, run_id: str) -> Path:
    """Isolate models (including revision), conditions and runs; never reuse a run directory."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise Error("Run ID must contain only letters, digits, underscores and hyphens")
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", config.model)[:60]
    identity = hashlib.sha256(f"{config.model}@{config.revision}".encode()).hexdigest()[:16]
    directory = Path(root) / f"{slug}-{identity}" / f"{config.shots}-shot" / run_id
    if directory.exists() or directory.is_symlink():
        raise Error(f"Refusing to reuse existing run directory: {directory}")
    # Existing ancestor symlinks could redirect a new run into archived outputs.
    if any(parent.is_symlink() for parent in (directory, *directory.absolute().parents)):
        raise Error("Output path must not traverse symlinks")
    return directory / "predictions.jsonl"


def _inputs(path: Path) -> list[tuple[str, str]]:
    """Read only identities; ObjectEntities is deliberately never accessed."""
    pairs, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise Error(f"{path}:{index}: malformed input JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise Error(f"{path}:{index}: input must be an object")
            keys = validate_prediction({
                "SubjectEntity": row.get("SubjectEntity"), "Relation": row.get("Relation"), "ObjectEntities": [],
            }, location=f"{path}:{index}")
            pair = (keys["SubjectEntity"], keys["Relation"])
            key = (few_shot._subject_key(pair[0]), pair[1])
            if pair[1] not in zero_shot.QUESTIONS or key in seen:
                raise Error(f"{path}:{index}: unsupported relation or duplicate normalized input identity")
            seen.add(key)
            pairs.append(pair)
    if not pairs:
        raise Error("Dataset has no inputs")
    return pairs


def prepare(dataset: Path, data_dir: Path, config: RunConfig) -> dict:
    """Use the committed selectors/builders verbatim, with no model-dependent reselection."""
    fingerprint = _sha256(Path(dataset))
    pairs = _inputs(dataset)
    demonstrations, leakage, train_hash = {}, {}, None
    if config.shots:
        train_hash = _sha256(Path(data_dir) / "train.jsonl")
        if train_hash != TRAIN_SHA256:
            raise Error("Training SHA256 differs from the reviewed demonstration snapshot")
        if config.shots == 3:
            demonstrations = few_shot.select_demonstrations(data_dir)
            leakage = few_shot.check_leakage(demonstrations, pairs)
            chats = [few_shot.build_messages(*pair, demonstrations) for pair in pairs]
            prompt_version, policy = few_shot.PROMPT_VERSION, few_shot.SELECTION_VERSION
        else:
            demonstrations = few_shot_v2.select_demonstrations(data_dir, shots=config.shots)
            leakage = few_shot_v2.check_leakage(demonstrations, pairs, shots=config.shots)
            chats = [few_shot_v2.build_messages(*pair, demonstrations, shots=config.shots) for pair in pairs]
            prompt_version, policy = few_shot_v2.PROMPT_VERSION, few_shot_v2.SELECTION_VERSION
        if _sha256(Path(data_dir) / "train.jsonl") != train_hash:
            raise Error("Training file changed during preparation")
    else:
        chats = [zero_shot.build_messages(*pair) for pair in pairs]
        prompt_version, policy = zero_shot.PROMPT_VERSION, "no-demonstrations"
    if _sha256(Path(dataset)) != fingerprint:
        raise Error("Dataset changed during preparation")
    return {
        "configuration": asdict(config), "prompt_version": prompt_version, "selection_version": policy,
        "base_prompt_version": zero_shot.PROMPT_VERSION, "train_sha256": train_hash,
        "dataset": {"path": str(Path(dataset).resolve()), "sha256": fingerprint},
        "demonstrations": demonstrations, "leakage": leakage,
        "prompts": [{"SubjectEntity": s, "Relation": r, "messages": chat}
                    for (s, r), chat in zip(pairs, chats)],
        "prompts_sha256": hashlib.sha256(json.dumps(chats, ensure_ascii=False).encode("utf-8")).hexdigest(),
    }


def _versions() -> dict:
    result = {"python": platform.python_version()}
    for name in ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "numpy", "jinja2",
                 "accelerate", "bitsandbytes", "mistral-common", "torchvision", "Pillow",
                 "flash-linear-attention", "causal-conv1d", "triton"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _loading_class(transformers, model_type: str):
    name = LOADING_CLASSES.get(model_type, "AutoModelForCausalLM")
    loader = getattr(transformers, name, None)
    if loader is None:
        raise Error(f"Installed Transformers lacks {name}; use the documented isolated benchmark-modern environment")
    return loader, name


def _chat_hash(messages) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _tokenizer_preflight(config: RunConfig, prompts: list, cache_dir: Path | None):
    """Native templates only. Reject unsupported roles/context rather than rewriting prompts."""
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    kwargs = {"revision": config.revision, "trust_remote_code": False, "cache_dir": cache_dir}
    if config.model in TARGETS:
        _loading_class(transformers, TARGETS[config.model][0])  # Before an unknown AutoConfig or weights download.
    model_config = AutoConfig.from_pretrained(config.model, **kwargs)
    resolved = getattr(model_config, "_commit_hash", None)
    if resolved != config.revision:
        raise Error("Could not verify the requested model commit from the loaded configuration")
    if getattr(model_config, "is_encoder_decoder", False):
        raise Error("Benchmark requires a causal language model")
    model_type = getattr(model_config, "model_type", "")
    if config.model in TARGETS and model_type != TARGETS[config.model][0]:
        raise Error("Checkpoint architecture disagrees with the reviewed model ID")
    _, loading_class = _loading_class(transformers, model_type)
    text_config = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
    checkpoint_quantization = None
    if config.model == MINISTRAL:
        checkpoint_quantization = getattr(model_config, "quantization_config", None)
        if (not isinstance(checkpoint_quantization, dict)
                or checkpoint_quantization.get("quant_method") != "fp8"
                or checkpoint_quantization.get("activation_scheme") != "static"
                or checkpoint_quantization.get("weight_block_size", "absent") is not None
                or getattr(text_config, "model_type", None) != "ministral3"):
            raise Error("Ministral checkpoint differs from the reviewed static, per-tensor FP8 configuration")
        if not hasattr(transformers, "FineGrainedFP8Config"):
            raise Error("Explicit FP8 dequantization requires the isolated benchmark-modern environment")
    chat_kwargs = {"add_generation_prompt": True}
    tokenizer_file = None
    if model_type == "mistral3":
        if config.thinking != "default":
            raise Error("This Mistral tokenizer has no enable_thinking switch; use --thinking default")
        from huggingface_hub import hf_hub_download

        tokenizer_file = Path(hf_hub_download(config.model, "tekken.json", revision=config.revision, cache_dir=cache_dir))
        native = getattr(transformers, "MistralCommonBackend", None) or getattr(transformers, "MistralCommonTokenizer", None)
        if native is None:
            raise Error("Install the documented mistral-common tokenizer dependencies")
        tokenizer = native.from_pretrained(str(tokenizer_file.parent), local_files_only=True)
        # 4.57.1 rejects this argument; 5.5 accepts it as a documented no-op.
        if "add_generation_prompt" not in inspect.signature(tokenizer.apply_chat_template).parameters:
            chat_kwargs = {}
        template = None  # Native mistral-common encoding is not a Jinja template.
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model, **kwargs)
        template = tokenizer.get_chat_template()
        if config.thinking != "default":
            if model_type not in ("qwen3_5", "gemma4") or "enable_thinking" not in template:
                raise Error("This checkpoint has no reviewed enable_thinking template switch")
            chat_kwargs["enable_thinking"] = config.thinking == "enabled"
    limits = [getattr(text_config, key, None) for key in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length")]
    limits.append(getattr(tokenizer, "model_max_length", None))
    limits = [value for value in limits if type(value) is int and 0 < value < 10**9]
    if config.context_limit is not None:
        if limits and config.context_limit > min(limits):
            raise Error("Explicit context limit exceeds the declared model/tokenizer limit")
        limits.append(config.context_limit)
    if not limits:
        raise Error("Unknown context window; supply a verified --context-limit")
    limit = min(limits)
    counts = []
    for row in prompts:
        try:
            rendered = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, **chat_kwargs,
            )
            if not isinstance(rendered, str) or any(message["content"] not in rendered for message in row["messages"]):
                raise Error("Template omitted or changed message content (including the system instruction)")
            tokens = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, truncation=False, return_dict=False, **chat_kwargs,
            )
        except Exception as error:
            raise Error(f"Native chat template cannot represent the unchanged messages: {error}") from error
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
            raise Error("Tokenizer must return nonempty flat token IDs")
        if len(tokens) + config.max_new_tokens > limit:
            raise Error(f"Context overflow for {row['SubjectEntity']!r}, {row['Relation']}")
        row["input_tokens"] = len(tokens)
        # Retain the actual serialization (including native thinking headers).
        row["rendered_chat"] = rendered
        row["input_ids_sha256"] = hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()
        counts.append(len(tokens))
    preflight = {
        "resolved_revision": resolved, "context_limit": limit,
        "model_type": model_type, "architectures": getattr(model_config, "architectures", None),
        "loading_class": loading_class, "text_only_inputs": True, "vision_weights_loaded": model_type in CONDITIONAL_CLASSES,
        "chat_template": template,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest() if template is not None else None,
        "chat_backend": "mistral-common" if tokenizer_file else "native-jinja",
        "tokenizer_file_sha256": _sha256(tokenizer_file) if tokenizer_file else None,
        "chat_template_configuration": {"tokenize": True, "truncation": False, **chat_kwargs},
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        "special_token_ids": {name: getattr(tokenizer, name, None) for name in ("bos_token_id", "eos_token_id", "pad_token_id")},
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_model_max_length": tokenizer.model_max_length,
        "prompt_token_hashes": {_chat_hash(row["messages"]): row["input_ids_sha256"] for row in prompts},
        "preflight_records": len(counts), "input_tokens_min": min(counts), "input_tokens_max": max(counts),
    }
    if checkpoint_quantization is not None:
        preflight["checkpoint_quantization"] = checkpoint_quantization
    return tokenizer, preflight


def _load_generator(config: RunConfig, tokenizer, preflight: dict, cache_dir: Path | None):
    """Explicit single-GPU precision/quantization; model-native EOS, no remote code."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import transformers
    from transformers import GenerationConfig

    if not torch.cuda.is_available():
        raise Error("A CUDA GPU is required for benchmark inference")
    if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise Error("bfloat16 is unsupported on this GPU; dtype will not be changed automatically")
    torch.manual_seed(config.seed)
    if config.determinism == "strict":
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    loader, loading_class = _loading_class(transformers, preflight.get("model_type", ""))
    properties = torch.cuda.get_device_properties(0)
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if config.model in TARGETS and config.quantization in ("none", "fp8-dequantize"):
        weight_floor = TARGETS[config.model][2] * 2  # Both allowed dtypes occupy two bytes.
        if weight_floor >= free_bytes:
            raise Error("Nominal 16-bit weights alone exceed free GPU memory; this configuration cannot fit")
    load_kwargs = {
        "revision": config.revision, "cache_dir": cache_dir, "trust_remote_code": False,
        "torch_dtype": getattr(torch, config.dtype), "attn_implementation": "eager", "use_safetensors": True,
    }
    quantization = None
    if config.quantization == "nf4":
        # Require these packages; never install, offload, shard or change dtype automatically.
        import accelerate  # noqa: F401
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig

        quantized = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, config.dtype),
        )
        quantization = quantized.to_dict()
        load_kwargs.update(quantization_config=quantized, device_map={"": 0})
    elif config.quantization == "fp8-dequantize":
        from transformers import FineGrainedFP8Config

        source = preflight.get("checkpoint_quantization")
        if not isinstance(source, dict) or source.get("quant_method") != "fp8":
            raise Error("Missing reviewed FP8 checkpoint configuration from tokenizer preflight")
        # Preserve the checkpoint's per-tensor scaling and static activation scheme.
        # The config class defaults to blockwise scaling, which is inappropriate here.
        quantized = FineGrainedFP8Config(**{**source, "dequantize": True})
        quantization = quantized.to_dict()
        load_kwargs.update(quantization_config=quantized, device_map={"": 0})
    runtime_warnings, memory = [], {}

    def update_memory():
        memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(0))

    def save_warnings(records):
        for record in records:
            entry = {"category": record.category.__name__, "message": str(record.message)}
            if entry not in runtime_warnings:
                runtime_warnings.append(entry)

    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            model = loader.from_pretrained(config.model, **load_kwargs)
            if config.quantization == "none":
                model = model.to("cuda")
        finally:
            update_memory()
            save_warnings(captured)
    model.eval()
    eos = model.generation_config.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None or pad < 0:
        pad = tokenizer.eos_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, list) and eos else eos
    if pad is None or eos is None:
        raise Error("Model/tokenizer must declare usable EOS and padding token IDs")
    generation = GenerationConfig(
        do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=config.max_new_tokens,
        use_cache=True, eos_token_id=eos, pad_token_id=pad,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started

    def generate(messages):
        batch = tokenizer.apply_chat_template(
            messages, **preflight["chat_template_configuration"], return_tensors="pt", return_dict=True,
        ).to(model.device)
        length = batch["input_ids"].shape[1]
        if length + config.max_new_tokens > preflight["context_limit"]:
            raise Error("Prompt no longer fits the preflight context window")
        if preflight.get("prompt_token_hashes"):
            actual = hashlib.sha256(json.dumps(batch["input_ids"][0].tolist()).encode("utf-8")).hexdigest()
            if actual != preflight["prompt_token_hashes"].get(_chat_hash(messages)):
                raise Error("Templated token IDs changed after preflight; no generation performed")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                with torch.inference_mode():
                    output = model.generate(**batch, generation_config=generation)
                torch.cuda.synchronize()
                return tokenizer.decode(output[0, length:], skip_special_tokens=True)
            finally:
                update_memory()
                save_warnings(captured)

    return generate, {
        "model_id": config.model, "requested_revision": config.revision, **preflight,
        "generation": generation.to_dict(), "dtype": config.dtype, "seed": config.seed,
        "attention_implementation": "eager", "batch_size": 1, "deterministic_algorithms": True,
        "loading_class": loading_class, "quantization": quantization,
        "model_implementation_module": type(model).__module__,
        "hybrid_fast_path_available": getattr(sys.modules.get(type(model).__module__), "is_fast_path_available", None),
        "thinking": config.thinking, "determinism": config.determinism,
        "determinism_caveat": "Greedy decoding and PyTorch checks do not guarantee determinism of third-party quantization/hybrid kernels or across hardware/versions",
        "runtime_warnings": runtime_warnings, "gpu_memory": memory,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "tf32": False, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "gpu_total_memory_bytes": properties.total_memory, "gpu_free_before_load_bytes": free_bytes,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "cudnn": torch.backends.cudnn.version(), "versions": _versions(), "model_load_seconds": load_seconds,
    }


def run(dataset: Path, data_dir: Path, root: Path, run_id: str, config: RunConfig, *,
        dry_run: bool = False, max_examples: int | None = None, cache_dir: Path | None = None,
        expected_records: int | None = None) -> dict:
    """Preflight the whole file, optionally generate a prefix; reserve a fresh run directory."""
    if max_examples is not None and (type(max_examples) is not int or max_examples < 1):
        raise Error("max_examples must be a positive integer")
    if expected_records is not None and (type(expected_records) is not int or expected_records < 1):
        raise Error("expected_records must be a positive integer")
    output = output_path(root, config, run_id)
    prepared = prepare(dataset, data_dir, config)
    if expected_records is not None:
        if len(prepared["prompts"]) != expected_records:
            raise Error(f"Expected {expected_records} input records, found {len(prepared['prompts'])}")
        prepared["expected_records"] = expected_records
    prompts = prepared["prompts"][:max_examples]
    summary = {
        **prepared, "output": str(output.resolve()), "run_id": run_id,
        "selection": {"strategy": "first_records_in_file_order", "max_examples": max_examples,
                      "records": len(prompts), "preflight_records": len(prepared["prompts"])},
        "status": "dry_run", "tokenizer_checked": False,
    }
    if dry_run:
        return summary
    try:
        tokenizer, preflight = _tokenizer_preflight(config, prepared["prompts"], cache_dir)
        generate, runtime = _load_generator(config, tokenizer, preflight, cache_dir)
    except ImportError as error:
        if config.model in TARGETS or config.quantization == "nf4":
            raise Error(
                "Missing model/quantization dependency; use the documented isolated benchmark-modern "
                "environment. No packages are installed automatically. "
                f"Import failure: {error}"
            ) from error
        raise Error('Install the existing inference extra: python -m pip install -e ".[inference]"') from error
    if _sha256(Path(dataset)) != prepared["dataset"]["sha256"] or (
        config.shots and _sha256(Path(data_dir) / "train.jsonl") != prepared["train_sha256"]
    ):
        raise Error("Input files changed before generation")
    source = _source_provenance()
    source["files_sha256"].update({Path(module.__file__).name: _sha256(Path(module.__file__))
                                      for module in (few_shot_v2, zero_shot)})
    source["files_sha256"][Path(__file__).name] = _sha256(Path(__file__))
    manifest = {**summary, "status": "prepared", "tokenizer_checked": True, "runtime": runtime, "source": source}
    # Exclusive directory reservation also protects the final os.replace in the shared runner.
    output.parent.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(str(output) + ".manifest.json")
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    metadata = {
        **runtime, "shots": config.shots, "prompt_version": prepared["prompt_version"],
        "selection_version": prepared["selection_version"], "prompt_sha256": prepared["prompts_sha256"],
        "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)},
        "source": source, "dataset": prepared["dataset"], "train_sha256": prepared["train_sha256"],
        "selection": summary["selection"], "output": str(output.resolve()), "total_inputs": len(prompts),
        "status": "running", "processed_records": 0, "valid_records": 0, "parse_failures": 0,
        "format_validity_rate": None, "numeric_normalized_responses": 0,
        "parse_failure_policy": "parse-failure-as-abstention-v1", "inference_runtime_seconds": 0.0,
    }

    def timed_generate(messages):
        started = time.perf_counter()
        try:
            return generate(messages)
        finally:
            metadata["inference_runtime_seconds"] += time.perf_counter() - started

    return zero_shot._run_prepared(
        [(row["SubjectEntity"], row["Relation"]) for row in prompts],
        [row["messages"] for row in prompts], timed_generate, metadata, output,
    )


def report(data_dir: Path, split: str, predictions: Path) -> dict:
    """Require a complete, integrity-checked run before separate official offline scoring."""
    from .evaluation.lm_kbc_2026 import LMKBC2026Evaluator

    if split not in ("train", "val"):
        raise Error("Only explicit train/val gold may be evaluated; public test is unlabeled")
    metadata = json.loads(Path(str(predictions) + ".metadata.json").read_text(encoding="utf-8"))
    manifest_path = Path(str(predictions) + ".manifest.json")
    if metadata.get("status") != "complete" or metadata.get("output_sha256") != _sha256(predictions):
        raise Error("Require a complete run with an unchanged prediction hash")
    if metadata["manifest"]["sha256"] != _sha256(manifest_path):
        raise Error("Run manifest hash mismatch")
    gold_path = Path(data_dir) / f"{split}.jsonl"
    if metadata["dataset"]["sha256"] != _sha256(gold_path):
        raise Error("Scoring split fingerprint differs from inference inputs")
    pairs = _inputs(gold_path)
    rows = list(load_predictions(predictions))
    if [(row["SubjectEntity"], row["Relation"]) for row in rows] != pairs:
        raise Error("Predictions must cover the complete scoring split in order; do not score smoke runs")
    raw = [json.loads(line) for line in Path(str(predictions) + ".raw.jsonl").read_text(encoding="utf-8").splitlines()]
    valid = normalized = 0
    if len(raw) != len(rows):
        raise Error("Raw-generation coverage differs from predictions")
    for index, (record, row) in enumerate(zip(raw, rows), 1):
        if (record["SubjectEntity"], record["Relation"], record["input_line"]) != (*pairs[index - 1], index):
            raise Error("Raw-generation identities/order differ from predictions")
        try:
            objects, numeric = zero_shot._parse_response(record["response"])
        except Error:
            objects, status = [], "parse_failed"
        else:
            valid += 1
            normalized += int(numeric)
            status = "valid"
        if objects != row["ObjectEntities"] or record["parsing_status"] != status:
            raise Error("Raw parsing disagrees with the published predictions")
    expected = {"total_inputs": len(rows), "processed_records": len(rows), "valid_records": valid,
                "parse_failures": len(rows) - valid, "numeric_normalized_responses": normalized,
                "format_validity_rate": valid / len(rows)}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise Error("Run metadata counters disagree with raw generations")
    scores = LMKBC2026Evaluator().evaluate(data_dir=data_dir, split=split, predictions_path=predictions)
    return {
        "evaluation": asdict(scores), "model_id": metadata["model_id"],
        "resolved_revision": metadata["resolved_revision"], "shots": metadata["shots"],
        "prompt_version": metadata["prompt_version"], **expected,
        "inference_runtime_seconds": metadata["inference_runtime_seconds"],
        "model_load_seconds": metadata["model_load_seconds"],
        "loading_class": metadata.get("loading_class"), "quantization": metadata.get("quantization"),
        "dtype": metadata.get("dtype"), "gpu_memory": metadata.get("gpu_memory"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="One explicit model/condition/run, or offline dry-run")
    runner.add_argument("--dataset", type=Path, required=True)
    runner.add_argument("--data-dir", type=Path, default=Path("data/lm_kbc_2026"))
    runner.add_argument("--output-root", type=Path, default=Path("results/model_benchmark"))
    runner.add_argument("--run-id", required=True)
    runner.add_argument("--model", required=True)
    runner.add_argument("--revision", required=True, help="Exact 40-character Hub commit SHA")
    runner.add_argument("--shots", type=int, choices=(0, 1, 3, 5), required=True)
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument("--max-new-tokens", type=int, default=512)
    runner.add_argument("--max-examples", type=int)
    runner.add_argument("--expected-records", type=int, help="Require this complete input count before selecting a smoke prefix")
    runner.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    runner.add_argument("--context-limit", type=int)
    runner.add_argument("--quantization", choices=("none", "nf4", "fp8-dequantize"), default="none")
    runner.add_argument("--thinking", choices=("default", "enabled", "disabled"), default="default")
    runner.add_argument("--determinism", choices=("strict", "warn"), default="strict")
    runner.add_argument("--cache-dir", type=Path)
    runner.add_argument("--dry-run", action="store_true")
    reporter = commands.add_parser("report", help="Separate offline official scoring of a complete run")
    reporter.add_argument("--data-dir", type=Path, required=True)
    reporter.add_argument("--split", choices=("train", "val"), required=True)
    reporter.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config = RunConfig(args.model, args.revision, args.shots, args.seed, args.max_new_tokens,
                               args.dtype, args.context_limit, args.quantization, args.thinking, args.determinism)
            result = run(args.dataset, args.data_dir, args.output_root, args.run_id, config,
                         dry_run=args.dry_run, max_examples=args.max_examples, cache_dir=args.cache_dir,
                         expected_records=args.expected_records)
        else:
            result = report(args.data_dir, args.split, args.predictions)
    except (DatasetError, Error, EvaluationError, OSError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
