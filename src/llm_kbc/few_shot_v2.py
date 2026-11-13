"""Training-only 1/5-shot preparation, preserving the ordered v1 3-shot prefix."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import few_shot as v1, zero_shot
from .dataset_inspector import DatasetError, load_jsonl
from .prediction_io import PredictionRecord, validate_prediction


PROMPT_VERSION = "lm-kbc-2026-few-shot-v2"
SELECTION_VERSION = "relation-aware-1-or-5-shot-v2"
FewShotError = v1.FewShotError


def _validate_shots(shots: int) -> None:
    if type(shots) is not int or shots not in (1, 5):
        raise FewShotError("Shot count must be 1 or 5; reuse the completed v1 3-shot experiment")


def _eligible_pools(data_dir: Path) -> dict[str, list[PredictionRecord]]:
    """Apply v1's unchanged whole-answer caps, canonical labels and validation."""
    train = data_dir / "train.jsonl"
    if train.resolve().name != "train.jsonl" or any(
        path.exists() and train.exists() and path.samefile(train)
        for path in (data_dir / "val.jsonl", data_dir / "test.jsonl")
    ):
        raise FewShotError("Demonstrations must come from train.jsonl, not a validation/test link")
    pools = {relation: [] for relation in zero_shot.QUESTIONS}
    seen = set()
    for row in load_jsonl(train):
        relation = row["Relation"]
        key = (v1._subject_key(row["SubjectEntity"]), relation)
        if relation not in pools or key in seen:
            raise FewShotError("Training relations must be supported and subject-relation pairs unique")
        seen.add(key)
        example = validate_prediction({
            "SubjectEntity": row["SubjectEntity"], "Relation": relation,
            "ObjectEntities": [aliases[0] for aliases in row["ObjectEntities"]],
        })
        if relation in v1.NUMERIC_RELATIONS and example["ObjectEntities"]:
            v1._numeric_value(example)
        answer_bytes = len(json.dumps(example["ObjectEntities"], ensure_ascii=False).encode("utf-8"))
        if len(example["ObjectEntities"]) <= v1.MAX_DEMO_OBJECTS and answer_bytes <= v1.MAX_DEMO_ANSWER_BYTES:
            pools[relation].append(example)
    return pools


def select_demonstrations(data_dir: Path, *, shots: int) -> dict[str, list[PredictionRecord]]:
    """Choose nonempty lower medians (1) or append rank-quartile examples to v1 (5).

    Nonnumeric ranks use (object count, original subject); numeric ranks use
    (finite decimal value, original subject). Numeric empty rows are fallback
    candidates only. Ties use original Unicode code-point order, never file order.
    No evaluation file or metric participates in selection.
    """
    _validate_shots(shots)
    pools = _eligible_pools(Path(data_dir))
    selected = v1.select_demonstrations(data_dir) if shots == 5 else {}
    for relation, pool in pools.items():
        if len(pool) < shots:
            raise FewShotError(f"{relation}: fewer than {shots} eligible training examples")
        numeric = relation in v1.NUMERIC_RELATIONS
        nonempty = sorted(
            (row for row in pool if row["ObjectEntities"]),
            key=lambda row: (v1._numeric_value(row) if numeric else len(row["ObjectEntities"]), row["SubjectEntity"]),
        )
        empty = sorted((row for row in pool if not row["ObjectEntities"]), key=lambda row: row["SubjectEntity"])
        if shots == 1:
            selected[relation] = [nonempty[(len(nonempty) - 1) // 2] if nonempty else empty[0]]
            continue
        ranked = nonempty + empty if numeric else sorted(pool, key=lambda row: (len(row["ObjectEntities"]), row["SubjectEntity"]))
        n = len(nonempty) if numeric and nonempty else len(ranked)
        for target in ((n - 1) // 4, 3 * (n - 1) // 4):
            _, best = min(
                ((index, row) for index, row in enumerate(ranked) if row not in selected[relation]),
                key=lambda item: (
                    numeric and not item[1]["ObjectEntities"], abs(item[0] - target), item[0],
                ),
            )
            selected[relation].append(best)
    return selected


def check_leakage(
    demonstrations: Mapping[str, Sequence[PredictionRecord]], inputs: Sequence[tuple[str, str]], *, shots: int,
) -> dict:
    """Check normalized identities without reselection; report cross-relation subjects."""
    _validate_shots(shots)
    if set(demonstrations) != set(zero_shot.QUESTIONS):
        raise FewShotError("Demonstrations must cover exactly the six supported relations")
    demos = []
    for relation, rows in demonstrations.items():
        if len(rows) != shots:
            raise FewShotError(f"{relation}: exactly {shots} demonstrations are required")
        for row in rows:
            validate_prediction(row)
            if row["Relation"] != relation:
                raise FewShotError("Demonstration relation disagrees with its group")
            demos.append(row)
    demo_keys = {(v1._subject_key(row["SubjectEntity"]), row["Relation"]) for row in demos}
    if len(demo_keys) != len(demos):
        raise FewShotError("Demonstration identities must be unique")
    seen, overlaps = set(), []
    for subject, relation in inputs:
        validate_prediction({"SubjectEntity": subject, "Relation": relation, "ObjectEntities": []})
        if relation not in zero_shot.QUESTIONS:
            raise FewShotError(f"Unsupported evaluation relation: {relation}")
        key = (v1._subject_key(subject), relation)
        if key in seen:
            raise FewShotError("Evaluation input identities must be unique")
        if key in demo_keys:
            raise FewShotError(f"Demonstration/evaluation pair leakage: {subject!r}, {relation}")
        seen.add(key)
        for demo in demos:
            if v1._subject_key(demo["SubjectEntity"]) == key[0]:
                overlaps.append({
                    "demonstration_subject": demo["SubjectEntity"], "demonstration_relation": demo["Relation"],
                    "input_subject": subject, "input_relation": relation,
                })
    return {"subject_only_overlaps": overlaps, "subject_matching": "NFKC, casefold, collapsed whitespace; no alias resolution"}


def build_messages(subject: str, relation: str, demonstrations, *, shots: int) -> list[dict[str, str]]:
    """Use unchanged zero-shot wording and canonical-label JSON demonstration turns."""
    check_leakage(demonstrations, [(subject, relation)], shots=shots)
    query = zero_shot.build_messages(subject, relation)
    messages = [query[0]]
    for demo in demonstrations[relation]:
        messages.extend([
            zero_shot.build_messages(demo["SubjectEntity"], relation)[1],
            {"role": "assistant", "content": json.dumps(demo["ObjectEntities"], ensure_ascii=False)},
        ])
    return [*messages, query[1]]


def prepare_prompts(
    data_dir: Path, inputs_path: Path, *, shots: int, tokenizer=None,
    context_limit: int = v1.QWEN_CONTEXT_LIMIT, max_new_tokens: int = 512,
) -> dict:
    """Prepare every input using subject/relation only, optionally counting exact tokens."""
    if context_limit < 1 or max_new_tokens < 1:
        raise FewShotError("Context and generation limits must be positive")
    demonstrations = select_demonstrations(data_dir, shots=shots)
    pairs = []
    with Path(inputs_path).open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream, start=1):
            location = f"{inputs_path}:{index}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetError(f"{location}: malformed input JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise DatasetError(f"{location}: input must be a JSON object")
            keys = validate_prediction({
                "SubjectEntity": row.get("SubjectEntity"), "Relation": row.get("Relation"), "ObjectEntities": [],
            }, location=location)
            pairs.append((keys["SubjectEntity"], keys["Relation"]))
    if not pairs:
        raise FewShotError("Evaluation inputs must not be empty")
    leakage = check_leakage(demonstrations, pairs, shots=shots)
    prompts = []
    for subject, relation in pairs:
        messages = build_messages(subject, relation, demonstrations, shots=shots)
        prompts.append({
            "SubjectEntity": subject, "Relation": relation, "messages": messages,
            "input_tokens": v1.check_context(messages, tokenizer, context_limit=context_limit, max_new_tokens=max_new_tokens)
            if tokenizer is not None else None,
        })
    versions = {}
    if tokenizer is not None:
        for package in ("transformers", "tokenizers", "jinja2"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
    return {
        "shots": shots, "prompt_version": PROMPT_VERSION, "selection_version": SELECTION_VERSION,
        "base_prompt_version": zero_shot.PROMPT_VERSION,
        "model_id": zero_shot.DEFAULT_MODEL, "model_revision": zero_shot.DEFAULT_REVISION,
        "max_demo_objects": v1.MAX_DEMO_OBJECTS, "max_demo_answer_bytes": v1.MAX_DEMO_ANSWER_BYTES,
        "context_limit": context_limit, "max_new_tokens": max_new_tokens,
        "tokenizer_checked": tokenizer is not None, "tokenizer_versions": versions,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
        "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode("utf-8")).hexdigest()
        if tokenizer is not None else None,
        "train_sha256": hashlib.sha256((Path(data_dir) / "train.jsonl").read_bytes()).hexdigest(),
        "inputs_sha256": hashlib.sha256(Path(inputs_path).read_bytes()).hexdigest(),
        "demonstrations": demonstrations, "leakage": leakage, "prompts": prompts,
    }
