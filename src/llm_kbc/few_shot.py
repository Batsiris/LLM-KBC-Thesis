"""Training-only 3-shot selection and prompt preparation; no model execution."""

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import zero_shot
from .dataset_inspector import DatasetError, load_jsonl
from .prediction_io import PredictionRecord, validate_prediction


PROMPT_VERSION = "lm-kbc-2026-few-shot-v1"
SELECTION_VERSION = "relation-aware-3-shot-v1"
MAX_DEMO_OBJECTS = 32
MAX_DEMO_ANSWER_BYTES = 1024
QWEN_CONTEXT_LIMIT = 32768  # max_position_embeddings at DEFAULT_REVISION.
NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}


class FewShotError(ValueError):
    """Training data, leakage, or prompt length prevents a valid 3-shot setup."""


def _subject_key(subject: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", subject).casefold().split())


def _numeric_value(row: PredictionRecord) -> Decimal:
    if len(row["ObjectEntities"]) != 1:
        raise FewShotError("Nonempty numeric demonstrations must have exactly one object")
    try:
        value = Decimal(row["ObjectEntities"][0].replace(",", "").strip())
    except InvalidOperation as error:
        raise FewShotError("Numeric training label is not a finite number") from error
    if not value.is_finite():
        raise FewShotError("Numeric training label is not a finite number")
    return value


def select_demonstrations(data_dir: Path) -> dict[str, list[PredictionRecord]]:
    """Select exactly three whole, canonical-label examples per relation.

    Read only train.jsonl. Selection never receives evaluation inputs or labels.
    Ties use original subject strings in Unicode code-point order, not file order.
    """
    data_dir = Path(data_dir)
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
        identity = (_subject_key(row["SubjectEntity"]), relation)
        if relation not in pools or identity in seen:
            raise FewShotError("Training relations must be supported and subject-relation pairs unique")
        seen.add(identity)
        # The first alias is the dataset's representative label, not all aliases
        # concatenated. Preserve gold group order, spelling, and numeric units.
        example = validate_prediction({
            "SubjectEntity": row["SubjectEntity"], "Relation": relation,
            "ObjectEntities": [aliases[0] for aliases in row["ObjectEntities"]],
        })
        if relation in NUMERIC_RELATIONS and example["ObjectEntities"]:
            _numeric_value(example)
        answer_bytes = len(json.dumps(example["ObjectEntities"], ensure_ascii=False).encode("utf-8"))
        if len(example["ObjectEntities"]) <= MAX_DEMO_OBJECTS and answer_bytes <= MAX_DEMO_ANSWER_BYTES:
            pools[relation].append(example)
    selected = {}
    for relation, pool in pools.items():
        if len(pool) < 3:
            raise FewShotError(f"{relation}: fewer than three eligible training examples")
        pool.sort(key=lambda row: (len(row["ObjectEntities"]), row["SubjectEntity"]))
        if relation in NUMERIC_RELATIONS:
            empty = [row for row in pool if not row["ObjectEntities"]]
            nonempty = sorted(
                (row for row in pool if row["ObjectEntities"]),
                key=lambda row: (_numeric_value(row), row["SubjectEntity"]),
            )
            if empty:
                if len(nonempty) < 2:
                    raise FewShotError(f"{relation}: an empty example requires two nonempty numeric examples")
                selected[relation] = [empty[0], nonempty[0], nonempty[-1]]
            else:
                selected[relation] = [nonempty[0], nonempty[(len(nonempty) - 1) // 2], nonempty[-1]]
        else:
            cardinalities = sorted({len(row["ObjectEntities"]) for row in pool})
            if cardinalities[0] == 0 and len(cardinalities) > 1:
                targets = [0, cardinalities[1], cardinalities[-1]]
            else:
                targets = [cardinalities[0], cardinalities[(len(cardinalities) - 1) // 2], cardinalities[-1]]
            chosen = []
            for target in targets:
                # If a target bucket is exhausted, prefer the nearest remaining
                # cardinality, then the larger cardinality, then original subject.
                best = min((row for row in pool if row not in chosen), key=lambda row: (
                    abs(len(row["ObjectEntities"]) - target), -len(row["ObjectEntities"]), row["SubjectEntity"],
                ))
                chosen.append(best)
            selected[relation] = chosen
    return selected


def check_leakage(
    demonstrations: Mapping[str, Sequence[PredictionRecord]], inputs: Sequence[tuple[str, str]],
) -> dict:
    """Fail on normalized pair overlap; report subject-only overlap without reselection."""
    if set(demonstrations) != set(zero_shot.QUESTIONS):
        raise FewShotError("Demonstrations must cover exactly the six supported relations")
    demos = []
    for relation, rows in demonstrations.items():
        if len(rows) != 3:
            raise FewShotError(f"{relation}: exactly three demonstrations are required")
        for row in rows:
            validate_prediction(row)
            if row["Relation"] != relation:
                raise FewShotError("Demonstration relation disagrees with its group")
            demos.append(row)
    demo_keys = {(_subject_key(row["SubjectEntity"]), row["Relation"]) for row in demos}
    if len(demo_keys) != len(demos):
        raise FewShotError("Demonstration identities must be unique")
    seen, overlaps = set(), []
    for subject, relation in inputs:
        validate_prediction({"SubjectEntity": subject, "Relation": relation, "ObjectEntities": []})
        if relation not in zero_shot.QUESTIONS:
            raise FewShotError(f"Unsupported evaluation relation: {relation}")
        key = (_subject_key(subject), relation)
        if key in seen:
            raise FewShotError("Evaluation input identities must be unique")
        if key in demo_keys:
            raise FewShotError(f"Demonstration/evaluation pair leakage: {subject!r}, {relation}")
        seen.add(key)
        for demo in demos:
            if _subject_key(demo["SubjectEntity"]) == key[0]:
                overlaps.append({
                    "demonstration_subject": demo["SubjectEntity"], "demonstration_relation": demo["Relation"],
                    "input_subject": subject, "input_relation": relation,
                })
    return {"subject_only_overlaps": overlaps, "subject_matching": "NFKC, casefold, collapsed whitespace; no alias resolution"}


def build_messages(
    subject: str, relation: str, demonstrations: Mapping[str, Sequence[PredictionRecord]],
) -> list[dict[str, str]]:
    """Use the unchanged zero-shot system/questions with three user/assistant turns."""
    check_leakage(demonstrations, [(subject, relation)])
    query = zero_shot.build_messages(subject, relation)
    messages = [query[0]]
    for demo in demonstrations[relation]:
        messages.append(zero_shot.build_messages(demo["SubjectEntity"], relation)[1])
        messages.append({"role": "assistant", "content": json.dumps(demo["ObjectEntities"], ensure_ascii=False)})
    return [*messages, query[1]]


def check_context(messages: list[dict[str, str]], tokenizer, *, context_limit: int, max_new_tokens: int = 512) -> int:
    """Count the actual chat template on CPU, with no truncation or generation."""
    if context_limit < 1 or max_new_tokens < 1:
        raise FewShotError("Context and generation limits must be positive")
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, truncation=False,
    )
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise FewShotError("Tokenizer must return a flat list of token IDs")
    tokenizer_limit = getattr(tokenizer, "model_max_length", context_limit)
    if not isinstance(tokenizer_limit, int) or tokenizer_limit < 1:
        raise FewShotError("Tokenizer context limit must be a positive integer")
    if len(tokens) + max_new_tokens > min(context_limit, tokenizer_limit):
        raise FewShotError("Few-shot prompt plus generation limit exceeds the context window")
    return len(tokens)


def prepare_prompts(
    data_dir: Path, inputs_path: Path, *, tokenizer=None,
    context_limit: int = QWEN_CONTEXT_LIMIT, max_new_tokens: int = 512,
) -> dict:
    """Prepare a reproducible prompt manifest; input gold fields are never accessed.

    The optional tokenizer enables an exact context preflight without inference.
    Without it, token counts are explicitly unavailable, not approximated.
    """
    if context_limit < 1 or max_new_tokens < 1:
        raise FewShotError("Context and generation limits must be positive")
    demonstrations = select_demonstrations(data_dir)
    pairs = []
    with Path(inputs_path).open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetError(f"{inputs_path}:{index}: malformed input JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise DatasetError(f"{inputs_path}:{index}: input must be a JSON object")
            keys = validate_prediction({
                "SubjectEntity": row.get("SubjectEntity"), "Relation": row.get("Relation"), "ObjectEntities": [],
            }, location=f"{inputs_path}:{index}")
            pairs.append((keys["SubjectEntity"], keys["Relation"]))
    if not pairs:
        raise FewShotError("Evaluation inputs must not be empty")
    leakage = check_leakage(demonstrations, pairs)
    prompts = []
    for subject, relation in pairs:
        messages = build_messages(subject, relation, demonstrations)
        prompts.append({
            "SubjectEntity": subject, "Relation": relation, "messages": messages,
            "content_characters": sum(len(message["content"]) for message in messages),
            "content_utf8_bytes": sum(len(message["content"].encode("utf-8")) for message in messages),
            "input_tokens": check_context(messages, tokenizer, context_limit=context_limit, max_new_tokens=max_new_tokens)
            if tokenizer is not None else None,
        })
    tokenizer_versions = {}
    if tokenizer is not None:
        for package in ("transformers", "tokenizers", "jinja2"):
            try:
                tokenizer_versions[package] = version(package)
            except PackageNotFoundError:
                tokenizer_versions[package] = None
    return {
        "prompt_version": PROMPT_VERSION, "selection_version": SELECTION_VERSION,
        "base_prompt_version": zero_shot.PROMPT_VERSION,
        "model_id": zero_shot.DEFAULT_MODEL, "model_revision": zero_shot.DEFAULT_REVISION,
        "max_demo_objects": MAX_DEMO_OBJECTS, "max_demo_answer_bytes": MAX_DEMO_ANSWER_BYTES,
        "context_limit": context_limit, "max_new_tokens": max_new_tokens,
        "tokenizer_checked": tokenizer is not None,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
        "tokenizer_versions": tokenizer_versions,
        "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode("utf-8")).hexdigest()
        if tokenizer is not None else None,
        "train_sha256": hashlib.sha256((Path(data_dir) / "train.jsonl").read_bytes()).hexdigest(),
        "inputs_sha256": hashlib.sha256(Path(inputs_path).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "zero_shot_source_sha256": hashlib.sha256(Path(zero_shot.__file__).read_bytes()).hexdigest(),
        "demonstrations": demonstrations, "leakage": leakage, "prompts": prompts,
    }
