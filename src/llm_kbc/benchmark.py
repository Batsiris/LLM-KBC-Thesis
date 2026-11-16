"""Model-configurable LM-KBC runs and separate offline reports; no model selection."""

import argparse
import hashlib
import json
import os
import platform
import re
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import few_shot, few_shot_v2, zero_shot
from .dataset_inspector import DatasetError
from .evaluation import EvaluationError
from .few_shot_inference import TRAIN_SHA256, _sha256, _source_provenance
from .prediction_io import load_predictions, validate_prediction


Error = zero_shot.InferenceError


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
    for name in ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "numpy", "jinja2"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _tokenizer_preflight(config: RunConfig, prompts: list, cache_dir: Path | None):
    """Native templates only. Reject unsupported roles/context rather than rewriting prompts."""
    from transformers import AutoConfig, AutoTokenizer

    kwargs = {"revision": config.revision, "trust_remote_code": False, "cache_dir": cache_dir}
    model_config = AutoConfig.from_pretrained(config.model, **kwargs)
    resolved = getattr(model_config, "_commit_hash", None)
    if resolved != config.revision:
        raise Error("Could not verify the requested model commit from the loaded configuration")
    if getattr(model_config, "is_encoder_decoder", False):
        raise Error("Benchmark requires a causal language model")
    tokenizer = AutoTokenizer.from_pretrained(config.model, **kwargs)
    template = tokenizer.get_chat_template()  # Missing/ambiguous templates fail explicitly.
    text_config = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
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
                row["messages"], tokenize=False, add_generation_prompt=True,
            )
            if not isinstance(rendered, str) or any(message["content"] not in rendered for message in row["messages"]):
                raise Error("Template omitted or changed message content (including the system instruction)")
            tokens = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True, truncation=False,
            )
        except Exception as error:
            raise Error(f"Native chat template cannot represent the unchanged messages: {error}") from error
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
            raise Error("Tokenizer must return nonempty flat token IDs")
        if len(tokens) + config.max_new_tokens > limit:
            raise Error(f"Context overflow for {row['SubjectEntity']!r}, {row['Relation']}")
        row["input_tokens"] = len(tokens)
        counts.append(len(tokens))
    return tokenizer, {
        "resolved_revision": resolved, "context_limit": limit,
        "chat_template": template, "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "chat_template_configuration": {"tokenize": True, "add_generation_prompt": True, "truncation": False},
        "special_tokens_map": tokenizer.special_tokens_map,
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_model_max_length": tokenizer.model_max_length,
        "preflight_records": len(counts), "input_tokens_min": min(counts), "input_tokens_max": max(counts),
    }


def _load_generator(config: RunConfig, tokenizer, preflight: dict, cache_dir: Path | None):
    """Single-device, unquantized greedy inference; model-native EOS, no remote code."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    if not torch.cuda.is_available():
        raise Error("A CUDA GPU is required for benchmark inference")
    if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise Error("bfloat16 is unsupported on this GPU; dtype will not be changed automatically")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        config.model, revision=config.revision, cache_dir=cache_dir, trust_remote_code=False,
        torch_dtype=getattr(torch, config.dtype), attn_implementation="eager", use_safetensors=True,
    ).to("cuda")
    model.eval()
    eos = model.generation_config.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
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
        with torch.inference_mode():
            output = model.generate(**batch, generation_config=generation)
        torch.cuda.synchronize()
        return tokenizer.decode(output[0, length:], skip_special_tokens=True)

    return generate, {
        "model_id": config.model, "requested_revision": config.revision, **preflight,
        "generation": generation.to_dict(), "dtype": config.dtype, "seed": config.seed,
        "attention_implementation": "eager", "batch_size": 1, "deterministic_algorithms": True,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "tf32": False, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "versions": _versions(), "model_load_seconds": load_seconds,
    }


def run(dataset: Path, data_dir: Path, root: Path, run_id: str, config: RunConfig, *,
        dry_run: bool = False, max_examples: int | None = None, cache_dir: Path | None = None) -> dict:
    """Preflight the whole file, optionally generate a prefix; reserve a fresh run directory."""
    if max_examples is not None and (type(max_examples) is not int or max_examples < 1):
        raise Error("max_examples must be a positive integer")
    output = output_path(root, config, run_id)
    prepared = prepare(dataset, data_dir, config)
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
    runner.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    runner.add_argument("--context-limit", type=int)
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
                               args.dtype, args.context_limit)
            result = run(args.dataset, args.data_dir, args.output_root, args.run_id, config,
                         dry_run=args.dry_run, max_examples=args.max_examples, cache_dir=args.cache_dir)
        else:
            result = report(args.data_dir, args.split, args.predictions)
    except (DatasetError, Error, EvaluationError, OSError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
