"""CPU-only masking/padding tests; optional Qwen checks use cached assets only."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from llm_kbc import fine_tuning_tokens as tokens, zero_shot
from llm_kbc.dataset_inspector import DatasetError
from llm_kbc.fine_tuning_data import prepare_training_data


@pytest.fixture
def examples(tmp_path):
    rows = [
        {"SubjectEntity": "Åland", "Relation": "hasArea", "ObjectEntities": [["1.2300", "unused"]]},
        {"SubjectEntity": "No answer", "Relation": "personHasCityOfDeath", "ObjectEntities": []},
        {"SubjectEntity": "Award", "Relation": "awardWonBy", "ObjectEntities": [["Éva"], ["東京"]]},
    ]
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    return prepare_training_data(tmp_path)


class ScriptedTokenizer:
    """Observable token boundaries without a dependency on Transformers or CUDA."""

    pad_token_id = 0
    eos_token_id = 9
    prefix_ids = [101, 102, 103, 104]  # Includes a synthetic assistant header.

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        assert kwargs == {
            "tokenize": True, "truncation": False, "padding": False, "return_dict": True,
            "tokenizer_kwargs": {"return_attention_mask": True},
        }
        self.calls.append((copy.deepcopy(messages), add_generation_prompt))
        if add_generation_prompt:
            assert [message["role"] for message in messages] == ["system", "user"]
            ids = list(self.prefix_ids)
        else:
            assert [message["role"] for message in messages] == ["system", "user", "assistant"]
            ids = self.prefix_ids + [ord(char) + 200 for char in messages[-1]["content"]] + [self.eos_token_id, 10]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_exact_mask_boundary_includes_header_and_supervises_answer_eot_and_separator(examples):
    tokenizer = ScriptedTokenizer()
    features = tokens.tokenize_training_data(examples, tokenizer, max_length=100)
    assert len(features) == len(examples)
    for example, feature in zip(examples, features, strict=True):
        boundary = len(tokenizer.prefix_ids)
        expected_answer = [ord(char) + 200 for char in example["messages"][-1]["content"]]
        assert feature["input_ids"] == tokenizer.prefix_ids + expected_answer + [9, 10]
        assert feature["labels"] == [-100] * boundary + expected_answer + [9, 10]
        assert feature["labels"][boundary - 1] == -100  # Assistant header is masked.
        assert feature["labels"][boundary] == feature["input_ids"][boundary]
        assert feature["labels"][-2] == tokenizer.eos_token_id
        assert feature["attention_mask"] == [1] * len(feature["input_ids"])
        assert len(feature["labels"]) == len(feature["input_ids"]) == len(feature["attention_mask"])
    assert tokenizer.calls == [call for example in examples for call in (
        (example["messages"], False), (example["messages"][:2], True),
    )]


def test_empty_json_answer_still_has_supervised_tokens(examples):
    tokenizer = ScriptedTokenizer()
    feature = tokens.tokenize_training_data([examples[1]], tokenizer, max_length=100)[0]
    assert feature["labels"][4:] == [ord("[") + 200, ord("]") + 200, 9, 10]


def test_determinism_order_and_no_input_mutation_or_file_reads(examples, monkeypatch):
    before = copy.deepcopy(examples)

    def no_open(*args, **kwargs):
        raise AssertionError("Tokenization must not open datasets")

    monkeypatch.setattr(Path, "open", no_open)
    first = tokens.tokenize_training_data(examples, ScriptedTokenizer(), max_length=100)
    assert first == tokens.tokenize_training_data(examples, ScriptedTokenizer(), max_length=100)
    assert tokens.tokenize_training_data(examples[::-1], ScriptedTokenizer(), max_length=100) == first[::-1]
    assert examples == before


@pytest.mark.parametrize("problem", ["mismatch", "longer_prefix", "no_suffix"])
def test_unreliable_prefix_or_no_supervised_tokens_is_rejected(examples, problem):
    class BrokenTokenizer(ScriptedTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            encoded = super().apply_chat_template(messages, **kwargs)
            if not kwargs["add_generation_prompt"]:
                if problem == "mismatch":
                    encoded["input_ids"][3] = 999
                elif problem == "longer_prefix":
                    encoded["input_ids"] = self.prefix_ids[:2]
                else:
                    encoded["input_ids"] = list(self.prefix_ids)
                encoded["attention_mask"] = [1] * len(encoded["input_ids"])
            return encoded

    message = "no supervised assistant tokens" if problem == "no_suffix" else "prefix does not align"
    with pytest.raises(DatasetError, match=message):
        tokens.tokenize_training_data(examples, BrokenTokenizer(), max_length=100)


def test_exact_max_length_is_accepted_and_overflow_reports_identity(examples):
    example = examples[0]
    length = len(tokens.tokenize_training_data([example], ScriptedTokenizer(), max_length=100)[0]["input_ids"])
    assert len(tokens.tokenize_training_data([example], ScriptedTokenizer(), max_length=length)[0]["input_ids"]) == length
    with pytest.raises(DatasetError, match=rf"Åland.*hasArea.*{length} tokens exceed max_length={length - 1}"):
        tokens.tokenize_training_data([example], ScriptedTokenizer(), max_length=length - 1)


@pytest.mark.parametrize("max_length", [None, 0, -1, True, 20.5, "100"])
def test_invalid_explicit_length_fails_before_tokenization(examples, max_length):
    tokenizer = ScriptedTokenizer()
    with pytest.raises(DatasetError, match="max_length"):
        tokens.tokenize_training_data(examples, tokenizer, max_length=max_length)
    assert tokenizer.calls == []


@pytest.mark.parametrize("problem", [
    "not_mapping", "missing_identity", "invalid_identity", "missing_messages", "two_messages",
    "four_messages", "wrong_role", "non_message", "missing_content", "non_string", "empty_content",
    "tool_metadata", "malformed_json", "non_array", "nested", "numeric_target",
])
def test_invalid_conversations_fail_before_tokenization(examples, problem):
    example = copy.deepcopy(examples[0])
    if problem == "not_mapping": example = None
    elif problem == "missing_identity": del example["SubjectEntity"]
    elif problem == "invalid_identity": example["Relation"] = 1
    elif problem == "missing_messages": del example["messages"]
    elif problem == "two_messages": example["messages"].pop()
    elif problem == "four_messages": example["messages"].append({"role": "user", "content": "extra"})
    elif problem == "wrong_role": example["messages"][2]["role"] = "user"
    elif problem == "non_message": example["messages"][0] = None
    elif problem == "missing_content": del example["messages"][0]["content"]
    elif problem == "non_string": example["messages"][0]["content"] = 1
    elif problem == "empty_content": example["messages"][2]["content"] = " "
    elif problem == "tool_metadata": example["messages"][2]["tool_calls"] = []
    elif problem == "malformed_json": example["messages"][2]["content"] = '["unfinished"'
    elif problem == "non_array": example["messages"][2]["content"] = '{"answer": "x"}'
    elif problem == "nested": example["messages"][2]["content"] = '[["x"]]'
    else: example["messages"][2]["content"] = '[1.2300]'
    tokenizer = ScriptedTokenizer()
    with pytest.raises(DatasetError, match="training example 1"):
        tokens.tokenize_training_data([example], tokenizer, max_length=100)
    assert tokenizer.calls == []


@pytest.mark.parametrize("examples", [[], None, "not examples", {}])
def test_invalid_or_empty_example_sequence(examples):
    with pytest.raises(DatasetError, match="nonempty sequence"):
        tokens.tokenize_training_data(examples, ScriptedTokenizer(), max_length=100)


@pytest.mark.parametrize("problem", ["not_mapping", "no_mask", "mask_length", "mask_zero", "nested_ids", "bool_ids", "negative_ids", "empty_ids"])
def test_invalid_tokenizer_encodings_fail_explicitly(examples, problem):
    class BrokenTokenizer(ScriptedTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            encoded = super().apply_chat_template(messages, **kwargs)
            if problem == "not_mapping": return encoded["input_ids"]
            if problem == "no_mask": del encoded["attention_mask"]
            elif problem == "mask_length": encoded["attention_mask"].pop()
            elif problem == "mask_zero": encoded["attention_mask"][0] = 0
            elif problem == "nested_ids": encoded["input_ids"] = [encoded["input_ids"]]
            elif problem == "bool_ids": encoded["input_ids"][0] = True
            elif problem == "negative_ids": encoded["input_ids"][0] = -1
            else: encoded["input_ids"] = []
            return encoded

    with pytest.raises(DatasetError, match="training example 1"):
        tokens.tokenize_training_data(examples, BrokenTokenizer(), max_length=100)


def test_chat_template_errors_include_identity(examples):
    class UnsupportedTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("missing chat template")

    with pytest.raises(DatasetError, match="Åland.*cannot apply chat template.*missing chat template"):
        tokens.tokenize_training_data(examples, UnsupportedTokenizer(), max_length=100)


@pytest.mark.parametrize("pad_id", [0, 9])
def test_deterministic_right_padding_and_supervised_eos_are_preserved(examples, pad_id):
    tokenizer = ScriptedTokenizer()
    tokenizer.pad_token_id = pad_id
    features = tokens.tokenize_training_data(examples, tokenizer, max_length=100)
    before = copy.deepcopy(features)
    batch = tokens.collate_batch(features, tokenizer)
    assert batch == tokens.collate_batch(features, tokenizer)
    width = max(len(feature["input_ids"]) for feature in features)
    for index, feature in enumerate(features):
        length = len(feature["input_ids"])
        assert batch["input_ids"][index] == feature["input_ids"] + [pad_id] * (width - length)
        assert batch["attention_mask"][index] == [1] * length + [0] * (width - length)
        assert batch["labels"][index] == feature["labels"] + [-100] * (width - length)
        assert batch["labels"][index][length - 2] == tokenizer.eos_token_id
    assert features == before
    batch["input_ids"][0][0] = 999
    assert features == before


def test_single_example_batch_does_not_add_padding(examples):
    tokenizer = ScriptedTokenizer()
    feature = tokens.tokenize_training_data(examples[:1], tokenizer, max_length=100)[0]
    assert tokens.collate_batch([feature], tokenizer) == {key: [value] for key, value in feature.items()}


@pytest.mark.parametrize("pad_id", [None, -1, True, 0.5])
def test_missing_or_invalid_pad_token_is_not_replaced_with_eos(examples, pad_id):
    tokenizer = ScriptedTokenizer()
    tokenizer.pad_token_id = pad_id
    features = tokens.tokenize_training_data(examples, tokenizer, max_length=100)
    with pytest.raises(DatasetError, match="pad_token_id"):
        tokens.collate_batch(features, tokenizer)


@pytest.mark.parametrize("problem", ["empty_batch", "bad_feature", "labels_length", "labels_value", "labels_bool", "all_masked", "attention_length"])
def test_invalid_padding_inputs_are_rejected(examples, problem):
    tokenizer = ScriptedTokenizer()
    features = tokens.tokenize_training_data(examples[:1], tokenizer, max_length=100)
    if problem == "empty_batch": features = []
    elif problem == "bad_feature": features[0] = None
    elif problem == "labels_length": features[0]["labels"].pop()
    elif problem == "labels_value": features[0]["labels"][-1] = 999
    elif problem == "labels_bool": features[0]["labels"][-1] = True
    elif problem == "all_masked": features[0]["labels"] = [-100] * len(features[0]["labels"])
    else: features[0]["attention_mask"].pop()
    with pytest.raises(DatasetError):
        tokens.collate_batch(features, tokenizer)


@pytest.fixture(scope="module")
def cached_qwen_tokenizer():
    """Optional integration check: never downloads or requires model weights."""
    transformers = pytest.importorskip("transformers")
    cache = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
    try:
        return transformers.AutoTokenizer.from_pretrained(
            zero_shot.DEFAULT_MODEL, revision=zero_shot.DEFAULT_REVISION,
            cache_dir=str(cache), local_files_only=True, trust_remote_code=False,
        )
    except OSError:
        pytest.skip("Pinned Qwen tokenizer is not cached locally; no download attempted")


def test_real_qwen_template_prefix_and_eot_for_prepared_conversations(examples, cached_qwen_tokenizer):
    tokenizer = cached_qwen_tokenizer
    features = tokens.tokenize_training_data(examples, tokenizer, max_length=2048)
    for example, feature in zip(examples, features, strict=True):
        full = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, add_generation_prompt=False, truncation=False,
        )
        prefix = tokenizer.apply_chat_template(
            example["messages"][:2], tokenize=True, add_generation_prompt=True, truncation=False,
        )
        assert feature["input_ids"] == full and full[:len(prefix)] == prefix
        assert feature["labels"] == [-100] * len(prefix) + full[len(prefix):]
        supervised = feature["labels"][len(prefix):]
        assert tokenizer.eos_token_id in supervised
        assert tokenizer.decode(supervised) == example["messages"][-1]["content"] + tokenizer.eos_token + "\n"
        assert feature["attention_mask"] == [1] * len(full)
    batch = tokens.collate_batch(features, tokenizer)
    assert all(len(row) == max(map(len, batch["input_ids"])) for rows in batch.values() for row in rows)


def test_module_import_does_not_load_training_or_gpu_dependencies():
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; import llm_kbc.fine_tuning_tokens; "
        "assert not {'torch', 'transformers', 'peft', 'trl', 'bitsandbytes'} & sys.modules.keys()",
    ], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
