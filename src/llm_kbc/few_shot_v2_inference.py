"""Pinned Few-shot v2 inference for the new 1-shot and 5-shot conditions only."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from . import few_shot_v2, zero_shot
from .dataset_inspector import DatasetError
from .few_shot_inference import (
    MAX_NEW_TOKENS, TRAIN_SHA256, _load_tokenizer, _sha256, _source_provenance,
)


def run_inference(
    data_dir: Path, dataset: Path, output: Path, *, shots: int, max_examples: int | None = None,
) -> dict:
    """Preflight all inputs on CPU, then reuse unchanged zero-shot generation/finalization."""
    few_shot_v2._validate_shots(shots)
    if max_examples is not None and (type(max_examples) is not int or max_examples < 1):
        raise zero_shot.InferenceError("Example limit must be a positive integer")
    data_dir, dataset, output = Path(data_dir), Path(dataset), Path(output)
    manifest_path = Path(str(output) + ".manifest.json")
    for path in (output, manifest_path, *(Path(str(output) + suffix) for suffix in (
        ".metadata.json", ".failures.jsonl", ".raw.jsonl",
    ))):
        if path.exists() or path.is_symlink():
            raise zero_shot.InferenceError(f"Refusing to overwrite existing file: {path}")
    train = data_dir / "train.jsonl"
    if _sha256(train) != TRAIN_SHA256:
        raise few_shot_v2.FewShotError("Training SHA256 does not match the reviewed Few-shot v1 snapshot")
    inputs_sha256 = _sha256(dataset)
    tokenizer = _load_tokenizer()
    prepared = few_shot_v2.prepare_prompts(
        data_dir, dataset, shots=shots, tokenizer=tokenizer, max_new_tokens=MAX_NEW_TOKENS,
    )
    if prepared["train_sha256"] != TRAIN_SHA256 or prepared["inputs_sha256"] != inputs_sha256:
        raise zero_shot.InferenceError("Input files changed during prompt preparation")
    source = _source_provenance()
    source["files_sha256"].update({
        path.name: _sha256(path) for path in (Path(__file__), Path(few_shot_v2.__file__))
    })
    prompts = prepared["prompts"][:max_examples]
    inputs = [(row["SubjectEntity"], row["Relation"]) for row in prompts]
    messages = [row["messages"] for row in prompts]
    generate, runtime = zero_shot._load_generator(
        zero_shot.DEFAULT_MODEL, zero_shot.DEFAULT_REVISION, MAX_NEW_TOKENS,
    )
    if runtime["resolved_revision"] != zero_shot.DEFAULT_REVISION:
        raise zero_shot.InferenceError("Loaded model revision differs from the pinned revision")
    if runtime["chat_template_sha256"] != prepared["chat_template_sha256"]:
        raise zero_shot.InferenceError("Generation chat template differs from the preflight tokenizer")
    if _sha256(train) != TRAIN_SHA256 or _sha256(dataset) != inputs_sha256:
        raise zero_shot.InferenceError("Input files changed before generation")
    manifest = {
        **prepared, "source": source, "generation": runtime["generation"], "runtime": runtime,
        "selection": {
            "strategy": "first_records_in_file_order", "max_examples": max_examples,
            "records": len(inputs), "preflight_records": len(prepared["prompts"]),
        },
        "selection_policy": {
            "version": few_shot_v2.SELECTION_VERSION,
            "description": (
                "Same training-only canonical labels, numeric validation and whole-answer caps as v1. "
                "1-shot: lower-median nonempty rank by cardinality (numeric: value), then subject; "
                "lexicographically first empty only if all candidates are empty. "
                "5-shot: exact ordered v1 prefix, then nearest unused ranks to floor((n-1)/4) and "
                "floor(3*(n-1)/4), lower-rank ties. Nonnumeric ranks include empties and use "
                "cardinality then subject. Numeric ranks use nonempty value then subject, "
                "with empty subjects appended as fallback only after nonempty candidates are exhausted."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    metadata = {
        **runtime, "shots": shots, "prompt_version": few_shot_v2.PROMPT_VERSION,
        "selection_version": few_shot_v2.SELECTION_VERSION,
        "base_prompt_version": zero_shot.PROMPT_VERSION,
        "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)},
        "source": source, "source_sha256": _sha256(Path(__file__)),
        "dataset": {"path": str(dataset.resolve()), "sha256": inputs_sha256},
        "train_sha256": TRAIN_SHA256, "selection": manifest["selection"],
        "output": str(output.resolve()), "total_inputs": len(inputs), "status": "running",
        "processed_records": 0, "valid_records": 0, "parse_failures": 0,
        "format_validity_rate": None, "numeric_normalized_responses": 0,
        "parse_failure_policy": "parse-failure-as-abstention-v1",
    }
    return zero_shot._run_prepared(inputs, messages, generate, metadata, output)


def main(argv: Sequence[str] | None = None) -> int:
    """Default outputs stay separate from all completed baseline artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, choices=(1, 5), required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing reviewed train.jsonl")
    parser.add_argument("--dataset", type=Path, required=True, help="Inputs; gold fields are ignored")
    parser.add_argument("--output", type=Path, help="Default: experiments/few_shot_v2/{shots}_shot.jsonl")
    parser.add_argument("--max-examples", type=int, help="Generate a prefix after checking every input")
    args = parser.parse_args(argv)
    output = args.output if args.output is not None else Path(f"experiments/few_shot_v2/{args.shots}_shot.jsonl")
    try:
        run_inference(args.data_dir, args.dataset, output, shots=args.shots, max_examples=args.max_examples)
    except (DatasetError, few_shot_v2.FewShotError, zero_shot.InferenceError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Predictions: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
