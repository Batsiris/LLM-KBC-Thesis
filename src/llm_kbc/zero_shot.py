"""Deterministic LM-KBC zero-shot generation; scoring remains a separate command.

Heavy dependencies are loaded only for an actual GPU run. Prompts use subjects
and relations only; gold labels are validated by the existing loader but ignored.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from typing import Any

from .dataset_inspector import DatasetError, load_jsonl
from .prediction_io import validate_prediction


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
PROMPT_VERSION = "lm-kbc-2026-zero-shot-v2"
SYSTEM_PROMPT = (
    "Answer factual questions using only your existing knowledge, as of 1 July 2026. "
    "Return only a JSON array of strings, with no explanation or markdown. "
    "List all known distinct answers, using one name per answer, not multiple aliases. "
    "Return an empty array when no answer is known or applicable. "
    "Express numerical answers as strings without units or grouping commas."
)
# Fixed short questions, without relation-specific rules or examples.
QUESTIONS = {
    "countryLandBordersCountry": "Which countries share a land border with {subject}?",
    "personHasCityOfDeath": "In which city did {subject} die?",
    "hasCapacity": "What is the spectator capacity of {subject}?",
    "awardWonBy": "Who has received the award {subject}?",
    "companyTradesAtStockExchange": "On which stock exchanges is {subject} listed?",
    "hasArea": "What is the total area of {subject} in square kilometres?",
}


class InferenceError(ValueError):
    """Inputs, output paths, or generated responses prevent a complete run."""


def build_messages(subject: str, relation: str) -> list[dict[str, str]]:
    """Build a zero-shot chat prompt without accepting any gold-answer argument."""
    if relation not in QUESTIONS:
        raise InferenceError(f"Unsupported LM-KBC 2026 relation: {relation!r}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTIONS[relation].format(subject=json.dumps(subject, ensure_ascii=False))},
    ]


def parse_response(response: str) -> list[str]:
    """Accept strings and finite JSON numbers, returning an array of strings.

    Numeric tokens retain their exact value and spelling, without a float
    round-trip. A whole code fence is allowed; malformed text is never repaired.
    """
    objects, _ = _parse_response(response)
    return objects


def _parse_response(response: str) -> tuple[list[str], bool]:
    """Parse once, also reporting whether a valid response needed normalization."""
    text = response.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    normalized = False

    def stringify_number(token: str) -> str:
        nonlocal normalized
        if token in ("NaN", "Infinity", "-Infinity"):
            raise InferenceError("Response contains a non-finite number")
        normalized = True
        return token

    try:
        objects = json.loads(
            text, parse_int=stringify_number, parse_float=stringify_number,
            parse_constant=stringify_number,
        )
    except json.JSONDecodeError as error:
        raise InferenceError(f"Response is not a complete JSON array: {error.msg}") from error
    if not isinstance(objects, list) or any(not isinstance(value, str) for value in objects):
        raise InferenceError("Response must be a flat JSON array of strings and finite numbers")
    return objects, normalized


def _load_generator(
    model_id: str, revision: str, max_new_tokens: int
) -> tuple[Callable[[list[dict[str, str]]], str], dict[str, Any]]:
    """Resolve one Hub commit for both tokenizer and model; load FP16 on CUDA."""
    # Set before importing torch or initializing CUDA; fail instead of allowing
    # nondeterministic kernels when a deterministic implementation is unavailable.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch
        import transformers
        import huggingface_hub
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    except ImportError as error:
        raise InferenceError(
            'Install inference dependencies with: python -m pip install -e ".[inference]"'
        ) from error
    if not torch.cuda.is_available():
        raise InferenceError("A CUDA GPU is required; select a T4 GPU runtime in Colab")
    resolved_revision = huggingface_hub.model_info(model_id, revision=revision).sha
    if not resolved_revision or not re.fullmatch(r"[0-9a-f]{40}", resolved_revision):
        raise InferenceError("Could not establish the exact model Hub commit SHA")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved_revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=resolved_revision, torch_dtype=torch.float16,
        attn_implementation="eager", use_safetensors=True, trust_remote_code=False,
    ).to("cuda")
    model.eval()
    generation = GenerationConfig(
        do_sample=False, num_beams=1, num_return_sequences=1,
        max_new_tokens=max_new_tokens, use_cache=True,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
    )

    def generate(messages: list[dict[str, str]]) -> str:
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length + max_new_tokens > model.config.max_position_embeddings:
            raise InferenceError("Prompt plus generation limit exceeds the model context window")
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=generation)
        return tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)

    return generate, {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "generation": generation.to_dict(),
        "dtype": "float16",
        "batch_size": 1,
        "attention_implementation": "eager",
        "seed": 0,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tf32": False,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode("utf-8")).hexdigest(),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "huggingface_hub": huggingface_hub.__version__,
            **{name: version(name) for name in ("tokenizers", "safetensors", "numpy")},
        },
    }


def run_inference(
    dataset: Path, output: Path, *, model_id: str = DEFAULT_MODEL,
    revision: str | None = None, max_examples: int | None = None, max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Publish one valid prediction per selected input, or fail with diagnostics.

    Sidecars record metadata, every raw response, and parse failures. Malformed
    responses become explicit empty-list abstentions for scoring. A runtime error
    never publishes an incomplete prediction file for accidental evaluation.
    Existing files are never overwritten. There are no retries or resume logic.
    """
    if max_new_tokens < 1 or (max_examples is not None and max_examples < 1):
        raise InferenceError("Token and example limits must be positive integers")
    dataset, output = Path(dataset), Path(output)
    metadata_path = Path(str(output) + ".metadata.json")
    failures_path = Path(str(output) + ".failures.jsonl")
    raw_path = Path(str(output) + ".raw.jsonl")
    for path in (output, metadata_path, failures_path, raw_path):
        if path.exists() or path.is_symlink():
            raise InferenceError(f"Refusing to overwrite existing file: {path}")
    # Project only keys into the inference layer, preserving file order and names.
    inputs = [
        (record["SubjectEntity"], record["Relation"])
        for record in islice(load_jsonl(dataset), max_examples)
    ]
    if not inputs:
        raise InferenceError("Dataset selection has no records")
    if len(set(inputs)) != len(inputs):
        raise InferenceError("Dataset selection has duplicate subject-relation pairs")
    messages = [build_messages(subject, relation) for subject, relation in inputs]
    requested_revision = revision or (DEFAULT_REVISION if model_id == DEFAULT_MODEL else "main")
    generate, runtime = _load_generator(model_id, requested_revision, max_new_tokens)
    metadata = {
        **runtime,
        "prompt_version": PROMPT_VERSION,
        "prompt": {"system": SYSTEM_PROMPT, "questions": QUESTIONS},
        "prompt_sha256": hashlib.sha256(
            json.dumps([SYSTEM_PROMPT, QUESTIONS], sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset": {"path": str(dataset.resolve()), "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()},
        "selection": {"strategy": "first_records_in_file_order", "max_examples": max_examples, "records": len(inputs)},
        "output": str(output.resolve()),
        "total_inputs": len(inputs),
        "status": "running", "processed_records": 0, "valid_records": 0, "parse_failures": 0,
        "format_validity_rate": None,
        "numeric_normalized_responses": 0,
        "parse_failure_policy": "parse-failure-as-abstention-v1",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve sidecars exclusively as well as checking paths above.
    with metadata_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")
    predictions = []
    with failures_path.open("x", encoding="utf-8") as failures, raw_path.open("x", encoding="utf-8") as raw:
        try:
            for index, ((subject, relation), chat) in enumerate(zip(inputs, messages), start=1):
                response = generate(chat)
                record = {
                    "input_line": index, "SubjectEntity": subject, "Relation": relation,
                    "response": response, "parsing_status": "valid", "error": None,
                }
                try:
                    objects, normalized = _parse_response(response)
                except InferenceError as error:
                    # An explicit scoring abstention, never a repaired model answer.
                    objects = []
                    record.update(parsing_status="parse_failed", error=str(error))
                except BaseException as error:
                    record.update(parsing_status="interrupted", error=str(error) or type(error).__name__)
                    raise
                finally:
                    # Flush every original response before processing another input,
                    # including when parsing is interrupted unexpectedly.
                    raw.write(json.dumps(record, ensure_ascii=True) + "\n")
                    raw.flush()
                if record["parsing_status"] == "valid":
                    metadata["valid_records"] += 1
                    if normalized:
                        metadata["numeric_normalized_responses"] += 1
                else:
                    failures.write(json.dumps({
                        "input_line": index, "SubjectEntity": subject, "Relation": relation,
                        "response": response, "error": record["error"],
                    }, ensure_ascii=True) + "\n")
                    failures.flush()
                    metadata["parse_failures"] += 1
                metadata["processed_records"] = index
                metadata["format_validity_rate"] = metadata["valid_records"] / index
                predictions.append(validate_prediction({
                    "SubjectEntity": subject, "Relation": relation, "ObjectEntities": objects,
                }, location=f"generated record {index}"))
                print(f"Generated {index}/{len(inputs)}; parse failures: {metadata['parse_failures']}", file=sys.stderr)
            if len(predictions) != len(inputs) or metadata["processed_records"] != len(inputs):
                raise InferenceError("Incomplete prediction coverage; no prediction file was published")
            identities = []
            for index, prediction in enumerate(predictions, start=1):
                validate_prediction(prediction, location=f"final prediction {index}")
                identities.append((prediction["SubjectEntity"], prediction["Relation"]))
            if identities != inputs or len(set(identities)) != len(inputs):
                raise InferenceError("Prediction order or unique identities do not match the selected inputs")
            # Publish only a fully written, validated file. Exclusive sidecars
            # reserve this output name across runs of this command.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                try:
                    for prediction in predictions:
                        stream.write(json.dumps(prediction, ensure_ascii=True) + "\n")
                except BaseException:
                    stream.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                if output.exists() or output.is_symlink():
                    raise InferenceError(f"Refusing to overwrite existing file: {output}")
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            metadata["status"] = "complete"
            metadata["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        except BaseException as error:
            metadata["status"] = "runtime_error" if isinstance(error, Exception) else "interrupted"
            metadata["error"] = str(error) or type(error).__name__
            raise
        finally:
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    """Run generation only; never score a dataset or load weights for --help."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Hub revision; defaults to the pinned Qwen commit (main for another model)")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    try:
        run_inference(
            args.dataset, args.output, model_id=args.model, revision=args.revision,
            max_examples=args.max_examples, max_new_tokens=args.max_new_tokens,
        )
    except (DatasetError, InferenceError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Predictions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
