"""CPU-only chat tokenization, assistant-only labels, and right-padding.

Accept prepared training conversations; never open datasets or load a tokenizer.
Return Python lists, with no tensor/GPU dependency. Tokenizer templates define all
control tokens. The caller must choose an explicit max_length appropriate for
the model; overflowing examples are rejected rather than truncated or dropped.
"""

import json
from collections.abc import Mapping, Sequence

from .dataset_inspector import DatasetError
from .prediction_io import validate_prediction


def _token_lists(encoded: object, location: str) -> tuple[list[int], list[int]]:
    """Validate unpadded, single-example tokenizer output without coercion."""
    if not isinstance(encoded, Mapping):
        raise DatasetError(f"{location}: expected a tokenizer output mapping")
    ids, mask = encoded.get("input_ids"), encoded.get("attention_mask")
    if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
        raise DatasetError(f"{location}: input_ids must be a nonempty flat list of nonnegative integers")
    if not isinstance(mask, list) or len(mask) != len(ids) or any(type(value) is not int or value != 1 for value in mask):
        raise DatasetError(f"{location}: unpadded attention_mask must contain one integer 1 per input token")
    return ids, mask


def tokenize_training_data(
    examples: Sequence[dict], tokenizer, *, max_length: int,
) -> list[dict[str, list[int]]]:
    """Tokenize prepare_training_data() output in order, without partial returns.

    Require system/user/assistant messages and a flat-string JSON assistant target.
    Mask the tokenized system/user generation prefix, including the assistant
    header, only after exact token-prefix alignment is verified. Supervise every
    remaining token, including template-provided end-of-turn and trailing tokens.
    No manual special tokens, label shifting, truncation, or padding is applied.
    """
    if type(max_length) is not int or max_length < 1:
        raise DatasetError("max_length must be an explicit positive integer")
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)) or not examples:
        raise DatasetError("Training examples must be a nonempty sequence")
    result = []
    for index, example in enumerate(examples, start=1):
        location = f"training example {index}"
        if not isinstance(example, Mapping):
            raise DatasetError(f"{location}: expected a prepared training example")
        identity = validate_prediction({
            "SubjectEntity": example.get("SubjectEntity"), "Relation": example.get("Relation"),
            "ObjectEntities": [],
        }, location=location)
        location += f" ({identity['SubjectEntity']!r}, {identity['Relation']!r})"
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise DatasetError(f"{location}: expected exactly three system/user/assistant messages")
        for message, role in zip(messages, ("system", "user", "assistant"), strict=True):
            if (
                not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] != role or not isinstance(message["content"], str)
                or not message["content"].strip()
            ):
                raise DatasetError(f"{location}: invalid {role} message; require role and nonempty string content")
        try:
            objects = json.loads(messages[-1]["content"])
        except json.JSONDecodeError as error:
            raise DatasetError(f"{location}: malformed assistant JSON: {error.msg}") from error
        validate_prediction({**identity, "ObjectEntities": objects}, location=location)
        kwargs = {
            "tokenize": True, "truncation": False, "padding": False, "return_dict": True,
            "tokenizer_kwargs": {"return_attention_mask": True},
        }
        try:
            full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
            prefix = tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True, **kwargs)
        except (TypeError, ValueError) as error:
            raise DatasetError(f"{location}: cannot apply chat template: {error}") from error
        ids, mask = _token_lists(full, location)
        prefix_ids, _ = _token_lists(prefix, location + " prefix")
        boundary = len(prefix_ids)
        if ids[:boundary] != prefix_ids:
            raise DatasetError(f"{location}: tokenized prefix does not align with the full conversation")
        if boundary >= len(ids):
            raise DatasetError(f"{location}: no supervised assistant tokens")
        if len(ids) > max_length:
            raise DatasetError(f"{location}: {len(ids)} tokens exceed max_length={max_length}; truncation is disabled")
        result.append({
            "input_ids": list(ids), "attention_mask": list(mask),
            "labels": [-100] * boundary + ids[boundary:],
        })
    return result


def collate_batch(features: Sequence[dict], tokenizer) -> dict[str, list[list[int]]]:
    """Right-pad unpadded tokenized examples to the longest sequence on CPU.

    Return fresh lists in input order. Padding uses the tokenizer's pad_token_id,
    attention 0, and label -100. Never mask real tokens by their numeric ID: a real
    supervised end-of-turn token remains supervised even if pad_token_id == EOS.
    """
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if type(pad_id) is not int or pad_id < 0:
        raise DatasetError("Tokenizer must provide a nonnegative integer pad_token_id")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)) or not features:
        raise DatasetError("Cannot collate an empty or invalid batch")
    for index, feature in enumerate(features, start=1):
        location = f"batch example {index}"
        ids, _ = _token_lists(feature, location)
        labels = feature.get("labels")
        if (
            not isinstance(labels, list) or len(labels) != len(ids)
            or any(type(label) is not int or label not in (-100, token) for label, token in zip(labels, ids))
        ):
            raise DatasetError(f"{location}: labels must align with input_ids and contain token IDs or -100")
        if all(label == -100 for label in labels):
            raise DatasetError(f"{location}: no supervised assistant tokens")
    width = max(len(feature["input_ids"]) for feature in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        padding = width - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        batch["labels"].append(feature["labels"] + [-100] * padding)
    return batch
