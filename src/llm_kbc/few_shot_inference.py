"""Pinned LM-KBC 2026 few-shot v1 inference; no scoring or format repair."""

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import few_shot, zero_shot
from .dataset_inspector import DatasetError


MAX_NEW_TOKENS = 512
# The official training snapshot used to review and commit Few-shot v1.
TRAIN_SHA256 = "ad37cd30d1ff4b9f1ef2579b25e64093b202c40da11e8c412e13386f1e5d332e"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tokenizer():
    """Load only the pinned tokenizer on CPU for a complete context preflight."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise zero_shot.InferenceError(
            'Install inference dependencies with: python -m pip install -e ".[inference]"'
        ) from error
    return AutoTokenizer.from_pretrained(
        zero_shot.DEFAULT_MODEL, revision=zero_shot.DEFAULT_REVISION, trust_remote_code=False,
    )


def _source_provenance() -> dict[str, Any]:
    """Identify the checkout and actual source, including uncommitted changes."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise zero_shot.InferenceError("Few-shot inference requires a Git checkout for source provenance") from error
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise zero_shot.InferenceError("Could not establish the source Git commit SHA")
    return {
        "commit": commit, "working_tree_dirty": bool(status),
        "files_sha256": {
            path.name: _sha256(path) for path in (
                Path(__file__), Path(few_shot.__file__), Path(zero_shot.__file__),
                Path(__file__).with_name("prediction_io.py"), Path(__file__).with_name("dataset_inspector.py"),
            )
        },
    }


def run_inference(
    data_dir: Path, dataset: Path, output: Path, *, max_examples: int | None = None,
) -> dict[str, Any]:
    """Preflight every input, then generate an ordered prefix or the complete run.

    Training bytes, model revision, demonstrations, and the 512-token condition
    are fixed. Evaluation gold fields are never read into prompt construction.
    The separate manifest records all preflighted prompts, including those beyond
    a smoke-run prefix. The shared zero-shot runner owns parsing and finalization.
    """
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
        raise few_shot.FewShotError("Training SHA256 does not match the reviewed Few-shot v1 snapshot")
    inputs_sha256 = _sha256(dataset)
    tokenizer = _load_tokenizer()
    prepared = few_shot.prepare_prompts(
        data_dir, dataset, tokenizer=tokenizer, max_new_tokens=MAX_NEW_TOKENS,
    )
    if prepared["train_sha256"] != TRAIN_SHA256 or prepared["inputs_sha256"] != inputs_sha256:
        raise zero_shot.InferenceError("Input files changed during prompt preparation")
    source = _source_provenance()
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
        **prepared,
        "source": source,
        "generation": runtime["generation"],
        "runtime": runtime,
        "selection": {
            "strategy": "first_records_in_file_order", "max_examples": max_examples,
            "records": len(inputs), "preflight_records": len(prepared["prompts"]),
        },
        "selection_policy": {
            "version": few_shot.SELECTION_VERSION,
            "description": (
                "Training-only canonical first aliases; whole answers capped at 32 objects/1024 UTF-8 bytes. "
                "Non-numeric: empty/smallest-positive/largest cardinalities where available, otherwise "
                "min/lower-median/max distinct cardinalities. Nearest unused bucket, larger count, "
                "then subject code-point order breaks ties. Numeric: empty/min/max if empty exists, "
                "otherwise min/lower-median/max value ranks; ties use subject code-point order."
            ),
        },
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(manifest_text)
    metadata = {
        **runtime,
        "prompt_version": few_shot.PROMPT_VERSION,
        "base_prompt_version": zero_shot.PROMPT_VERSION,
        "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)},
        "source": source,
        "source_sha256": _sha256(Path(__file__)),
        "dataset": {"path": str(dataset.resolve()), "sha256": inputs_sha256},
        "train_sha256": TRAIN_SHA256,
        "selection": manifest["selection"],
        "output": str(output.resolve()), "total_inputs": len(inputs),
        "status": "running", "processed_records": 0, "valid_records": 0, "parse_failures": 0,
        "format_validity_rate": None, "numeric_normalized_responses": 0,
        "parse_failure_policy": "parse-failure-as-abstention-v1",
    }
    return zero_shot._run_prepared(inputs, messages, generate, metadata, output)


def main(argv: Sequence[str] | None = None) -> int:
    """Run fixed Few-shot v1; --help needs neither CUDA nor model dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the reviewed train.jsonl")
    parser.add_argument("--dataset", type=Path, required=True, help="Evaluation inputs (only subject/relation are used)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, help="Generate a prefix after checking all inputs")
    args = parser.parse_args(argv)
    try:
        run_inference(args.data_dir, args.dataset, args.output, max_examples=args.max_examples)
    except (DatasetError, few_shot.FewShotError, zero_shot.InferenceError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Predictions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
