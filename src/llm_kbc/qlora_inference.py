"""Pinned FP16 base plus an unmerged LoRA adapter; generation only, never scoring."""

import argparse
import hashlib
import json
import os
import platform
import warnings
from collections.abc import Sequence
from importlib.metadata import version
from itertools import islice
from pathlib import Path

from . import zero_shot
from .dataset_inspector import DatasetError
from .prediction_io import validate_prediction


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_adapter(adapter_dir: Path) -> dict:
    """Validate local artifacts and explicit compatibility without loading a GPU."""
    config_path, weights = adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors"
    for path in (config_path, weights):
        if not path.is_file() or path.is_symlink():
            raise zero_shot.InferenceError(f"Require a regular local adapter file: {path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise zero_shot.InferenceError(f"Invalid adapter configuration: {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise zero_shot.InferenceError("Adapter configuration must be a JSON object")
    required = {
        "base_model_name_or_path": zero_shot.DEFAULT_MODEL,
        "revision": zero_shot.DEFAULT_REVISION, "peft_type": "LORA", "task_type": "CAUSAL_LM",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise zero_shot.InferenceError(f"Incompatible adapter {key}: expected {expected!r}, found {config.get(key)!r}")
    return {
        "path": str(adapter_dir), "weights_sha256": _sha256(weights),
        "config_sha256": _sha256(config_path), "config": config,
        "merged": False, "dtype": "float16", "autocast_adapter_dtype": False,
    }


def _load_inputs(dataset: Path, max_examples: int | None) -> list[tuple[str, str]]:
    """Project only identities, accepting absent/withheld/nested unused gold fields."""
    inputs = []
    try:
        with dataset.open("r", encoding="utf-8") as stream:
            for index, line in enumerate(islice(stream, max_examples), start=1):
                location = f"{dataset}:{index}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetError(f"{location}: malformed input JSON: {error.msg}") from error
                if not isinstance(row, dict):
                    raise DatasetError(f"{location}: expected an input JSON object")
                keys = validate_prediction({
                    "SubjectEntity": row.get("SubjectEntity"), "Relation": row.get("Relation"), "ObjectEntities": [],
                }, location=location)
                inputs.append((keys["SubjectEntity"], keys["Relation"]))
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetError(f"{dataset}: could not read UTF-8 inference inputs: {error}") from error
    if not inputs:
        raise zero_shot.InferenceError("Dataset selection has no records")
    if len(set(inputs)) != len(inputs):
        raise zero_shot.InferenceError("Dataset selection has duplicate subject-relation pairs")
    return inputs


def _load_generator(adapter_dir: Path, cache_dir: Path | None, max_new_tokens: int):
    """Match zero-shot greedy FP16/eager generation, with an active frozen adapter.

    A separate loader keeps the original zero-shot closure and behavior unchanged.
    PEFT's automatic FP32 adapter promotion is disabled; explicitly cast the full
    inference model to FP16. No quantization, merging, or training is performed.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch
        import transformers
        import peft
        import huggingface_hub
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    except ImportError as error:
        raise zero_shot.InferenceError("QLoRA inference requires torch, transformers, and PEFT; use the fine-tuning extra") from error
    if not torch.cuda.is_available():
        raise zero_shot.InferenceError("A CUDA GPU is required for QLoRA inference")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    kwargs = {"revision": zero_shot.DEFAULT_REVISION, "trust_remote_code": False,
              "cache_dir": str(cache_dir) if cache_dir else None}
    tokenizer = AutoTokenizer.from_pretrained(zero_shot.DEFAULT_MODEL, **kwargs)
    base = AutoModelForCausalLM.from_pretrained(
        zero_shot.DEFAULT_MODEL, **kwargs, torch_dtype=torch.float16,
        attn_implementation="eager", use_safetensors=True,
    ).to("cuda")
    if getattr(base.config, "_commit_hash", None) != zero_shot.DEFAULT_REVISION:
        raise zero_shot.InferenceError("Loaded base model revision differs from the pinned revision")
    if getattr(base, "is_loaded_in_4bit", False) or getattr(base, "is_loaded_in_8bit", False):
        raise zero_shot.InferenceError("Inference requires the original FP16 base, not a quantized model")
    try:
        with warnings.catch_warnings():
            # PEFT 0.17.1 warns rather than raises for missing adapter tensors.
            warnings.filterwarnings("error", message="Found missing adapter keys", category=UserWarning)
            model = peft.PeftModel.from_pretrained(
                base, str(adapter_dir), is_trainable=False, autocast_adapter_dtype=False,
                local_files_only=True, ignore_mismatched_sizes=False,
            )
    except UserWarning as error:
        raise zero_shot.InferenceError(f"Incomplete adapter weights: {error}") from error
    model.to(dtype=torch.float16)
    model.requires_grad_(False)
    model.eval()
    generation = GenerationConfig(
        do_sample=False, num_beams=1, num_return_sequences=1,
        max_new_tokens=max_new_tokens, use_cache=True,
        eos_token_id=base.generation_config.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
    )

    def generate(messages):
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length + max_new_tokens > base.config.max_position_embeddings:
            raise zero_shot.InferenceError("Prompt plus generation limit exceeds the model context window")
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=generation)
        return tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)

    return generate, {
        "model_id": zero_shot.DEFAULT_MODEL, "requested_revision": zero_shot.DEFAULT_REVISION,
        "resolved_revision": base.config._commit_hash, "generation": generation.to_dict(),
        "dtype": "float16", "batch_size": 1, "attention_implementation": "eager", "seed": 0,
        "deterministic_algorithms": True, "cudnn_benchmark": False, "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "tf32": False,
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode("utf-8")).hexdigest(),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "huggingface_hub": huggingface_hub.__version__,
            **{name: version(name) for name in ("accelerate", "tokenizers", "safetensors", "numpy")},
        },
    }


def run_inference(
    dataset: Path, output: Path, *, adapter_dir: Path, cache_dir: Path | None = None,
    max_examples: int | None = None, max_new_tokens: int = 512,
) -> dict:
    """Use unchanged zero-shot prompts and the shared audited inference runner."""
    if type(max_new_tokens) is not int or max_new_tokens < 1 or (
        max_examples is not None and (type(max_examples) is not int or max_examples < 1)
    ):
        raise zero_shot.InferenceError("Token and example limits must be positive integers")
    dataset, output, adapter_dir = Path(dataset), Path(output), Path(adapter_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    for path in (output, *(Path(str(output) + suffix) for suffix in (".metadata.json", ".failures.jsonl", ".raw.jsonl"))):
        if path.exists() or path.is_symlink():
            raise zero_shot.InferenceError(f"Refusing to overwrite existing file: {path}")
    if output.resolve().is_relative_to(adapter_dir) or (cache_dir and cache_dir.is_relative_to(adapter_dir)):
        raise zero_shot.InferenceError("Output and model cache must be outside the adapter directory")
    adapter = _read_adapter(adapter_dir)
    dataset_hash = _sha256(dataset)
    inputs = _load_inputs(dataset, max_examples)
    messages = [zero_shot.build_messages(subject, relation) for subject, relation in inputs]
    generate, runtime = _load_generator(adapter_dir, cache_dir, max_new_tokens)
    if _read_adapter(adapter_dir) != adapter or _sha256(dataset) != dataset_hash:
        raise zero_shot.InferenceError("Adapter or dataset files changed during model loading")
    metadata = {
        **runtime, "experiment": "lm-kbc-2026-qlora-inference-v1", "adapter": adapter,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "prompt_version": zero_shot.PROMPT_VERSION,
        "prompt": {"system": zero_shot.SYSTEM_PROMPT, "questions": zero_shot.QUESTIONS},
        "prompt_sha256": hashlib.sha256(json.dumps(
            [zero_shot.SYSTEM_PROMPT, zero_shot.QUESTIONS], sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")).hexdigest(),
        "source_sha256": _sha256(Path(__file__)), "zero_shot_source_sha256": _sha256(Path(zero_shot.__file__)),
        "dataset": {"path": str(dataset.resolve()), "sha256": dataset_hash},
        "selection": {"strategy": "first_records_in_file_order", "max_examples": max_examples, "records": len(inputs)},
        "output": str(output.resolve()), "total_inputs": len(inputs), "status": "running",
        "processed_records": 0, "valid_records": 0, "parse_failures": 0, "format_validity_rate": None,
        "numeric_normalized_responses": 0, "parse_failure_policy": "parse-failure-as-abstention-v1",
    }
    return zero_shot._run_prepared(inputs, messages, generate, metadata, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True, help="Local completed LoRA adapter directory")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    try:
        run_inference(args.dataset, args.output, adapter_dir=args.adapter_dir, cache_dir=args.cache_dir,
                      max_examples=args.max_examples, max_new_tokens=args.max_new_tokens)
    except (DatasetError, zero_shot.InferenceError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Predictions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
