"""Synthetic protocol checks and parity with the pinned official evaluator.

These fixtures are software tests, not model predictions or experiment results.
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import pytest

from llm_kbc._vendor.lm_kbc_2026 import evaluate as official
from llm_kbc.dataset_inspector import DatasetError
from llm_kbc.evaluation import EvaluationError
from llm_kbc.evaluation.lm_kbc_2026 import LMKBC2026Evaluator
from llm_kbc.prediction_io import load_predictions


STRING_RELATION = "countryLandBordersCountry"


def row(objects, subject="A", relation=STRING_RELATION):
    """Build either a gold or prediction row, depending on object nesting."""
    return {"SubjectEntity": subject, "Relation": relation, "ObjectEntities": objects}


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in rows),
        encoding="utf-8",
    )
    return path


def evaluate(tmp_path, gold, predictions, split="val"):
    gold_path = write_rows(tmp_path / f"{split}.jsonl", gold)
    predictions_path = write_rows(tmp_path / "predictions.jsonl", predictions)
    before = (gold_path.read_bytes(), predictions_path.read_bytes())
    result = LMKBC2026Evaluator().evaluate(
        data_dir=tmp_path, split=split, predictions_path=predictions_path
    )
    assert (gold_path.read_bytes(), predictions_path.read_bytes()) == before
    assert result.metadata["gold"]["sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert result.metadata["predictions"]["sha256"] == hashlib.sha256(before[1]).hexdigest()
    # Compare against upstream, including its F1 and both aggregation functions.
    scores = official.evaluate_per_sr_pair(predictions, gold, official.RELATION_TYPE, 0.05)
    macro = official.macro_average_per_relation(scores)
    micro = official.micro_average_per_relation(scores)
    key = "*** All Relations ***"
    assert result.overall == {**macro[key], **micro[key]}
    assert result.per_relation == {
        relation: {**macro[relation], **micro[relation]}
        for relation in macro if relation != key
    }
    return result


def test_vendored_provenance_and_exact_bytes():
    root = files("llm_kbc._vendor.lm_kbc_2026")
    provenance = json.loads(root.joinpath("provenance.json").read_text(encoding="utf-8"))
    commit = "e079024b809568f28cded5537203dfd52f5b04ae"
    assert provenance["commit"] == commit
    assert provenance["license"] == "Apache-2.0"
    assert provenance["files"]["evaluate.py"]["sha256"] == (
        "2d592ae177c7b230922bb959da7a8ee1c4c662bf72a99d4dbd0cf62170ff9e22"
    )
    for name, entry in provenance["files"].items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == entry["sha256"]
        assert entry["source_url"] == f"https://github.com/lm-kbc/dataset2026/blob/{commit}/{name}"
    assert b"Apache License" in root.joinpath("LICENSE").read_bytes()


@pytest.mark.parametrize(("aliases", "predictions", "precision"), [
    ([["O’Brien"]], ["OBrien"], 1.0),
    ([["Kauaʻi"]], ["Kauai"], 1.0),
    ([["Straße"]], ["STRASSE"], 1.0),
    ([["Éva—Smith"]], ["eva smith"], 1.0),
    ([["Canal+"]], [" canal "], 1.0),
    ([["New York City", "NYC"]], ["NYC", "nyc"], 1.0),
    ([["New York City", "NYC"]], ["NYC", "New York City"], 0.5),
    ([["a", "b"], ["a"]], ["a", "b"], 1.0),
    ([["a", "b"], ["a"]], ["b", "a"], 1.0),
])
def test_string_aliases_and_maximum_matching(tmp_path, aliases, predictions, precision):
    result = evaluate(tmp_path, [row(aliases)], [row(predictions)])
    assert result.overall["macro-p"] == precision
    assert result.overall["macro-r"] == 1.0


@pytest.mark.parametrize(("gold", "prediction", "matched"), [
    ("100", "95", True),
    ("100", "105", True),
    ("100", "94.999", False),
    ("100", "105.001", False),
    ("1000", "1,000", True),
    ("1,000", "1e3", True),
    ("1000", "1000 people", False),
    ("0", "0", False),  # Upstream never matches zero gold.
    ("-100", "999", True),  # Preserve upstream's signed-denominator bug.
    ("100", "NaN", False),
    ("100", "inf", False),
    ("NaN", "100", False),
    ("inf", "inf", False),
    ("not numeric", "100", False),
])
@pytest.mark.parametrize("relation", ["hasArea", "hasCapacity"])
def test_numeric_matching_and_edge_cases(tmp_path, gold, prediction, matched, relation):
    result = evaluate(tmp_path, [row([[gold]], relation=relation)], [row([prediction], relation=relation)])
    assert result.overall["macro-f1"] == float(matched)


def test_numeric_first_alias_only(tmp_path):
    result = evaluate(tmp_path, [row([["invalid", "100"]], relation="hasArea")], [row(["100"], relation="hasArea")])
    assert result.overall["macro-f1"] == 0


def test_numeric_greedy_order_is_preserved(tmp_path):
    gold = [row([["100"], ["104"]], relation="hasArea")]
    first = evaluate(tmp_path, gold, [row(["102", "96"], relation="hasArea")])
    second = evaluate(tmp_path, gold, [row(["96", "102"], relation="hasArea")])
    assert first.overall["macro-f1"] == 0.5
    assert second.overall["macro-f1"] == 1.0


def test_numeric_prediction_deduplication_uses_string_normalization(tmp_path):
    gold = [row([["100"]], relation="hasArea")]
    first = evaluate(tmp_path, gold, [row(["-100", "100"], relation="hasArea")])
    second = evaluate(tmp_path, gold, [row(["100", "-100"], relation="hasArea")])
    assert first.overall["macro-f1"] == 0.0
    assert second.overall["macro-f1"] == 1.0
    distinct = evaluate(tmp_path, gold, [row(["100", "100.0"], relation="hasArea")])
    assert distinct.overall["macro-p"] == 0.5


@pytest.mark.parametrize(("gold", "predictions", "expected"), [
    ([], [], (1, 1, 1)),
    ([["x"]], [], (1, 0, 0)),
    ([], ["x"], (0, 1, 0)),
    ([["x"]], ["y"], (0, 0, 0)),
])
def test_empty_cases(tmp_path, gold, predictions, expected):
    result = evaluate(tmp_path, [row(gold)], [row(predictions)])
    assert tuple(result.overall[k] for k in ("macro-p", "macro-r", "macro-f1")) == expected
    assert tuple(result.overall[k] for k in ("micro-p", "micro-r", "micro-f1")) == expected


def test_missing_predictions_and_extra_or_case_mismatched_pairs(tmp_path):
    gold = [row([["x"]]), row([], subject="B")]
    result = evaluate(tmp_path, gold, [])
    extra = evaluate(tmp_path, gold, [row(["x"], subject="a"), row(["z"], subject="EXTRA")])
    assert result.overall == extra.overall
    assert result.overall["macro-f1"] == 0.5
    assert result.metadata["evaluated_pairs"] == 2


def test_duplicate_pairs_keep_last_prediction_and_gold_rows(tmp_path):
    gold = [row([["old"]]), row([["new"]])]
    predictions = [row(["wrong"]), row(["new"])]
    result = evaluate(tmp_path, gold, predictions)
    assert result.overall["macro-f1"] == 1
    assert result.metadata["gold"]["input_records"] == 2
    assert result.metadata["predictions"]["input_records"] == 2
    assert result.metadata["evaluated_pairs"] == 1
    reverse = evaluate(tmp_path, gold, list(reversed(predictions)))
    assert reverse.overall["macro-f1"] == 0


def test_macro_and_micro_aggregation_and_full_precision(tmp_path):
    gold = [row([["a"], ["b"], ["c"]]), row([["d"]], subject="B"), row([], subject="C", relation="personHasCityOfDeath")]
    predictions = [row(["a"]), row(["wrong"], subject="B"), row([], subject="C", relation="personHasCityOfDeath")]
    result = evaluate(tmp_path, gold, predictions, split="train")
    assert result.split == "train"
    relation = result.per_relation[STRING_RELATION]
    assert relation["macro-p"] == 0.5
    assert relation["macro-r"] == pytest.approx(1 / 6)
    assert relation["macro-f1"] == 0.25
    assert relation["micro-p"] == 0.5
    assert relation["micro-r"] == 0.25
    assert relation["micro-f1"] == 1 / 3
    assert result.overall["macro-p"] == 2 / 3
    assert result.overall["macro-r"] == pytest.approx(4 / 9)
    assert result.overall["macro-f1"] == 0.5
    assert result.overall["micro-f1"] == 1 / 3
    encoded = json.loads(json.dumps(asdict(result), allow_nan=False))
    assert encoded["overall"]["micro-f1"] == 1 / 3
    assert encoded["overall"]["micro-f1"] != 0.333


@pytest.mark.parametrize("split", ["test", "", "validation", "VAL"])
def test_only_explicit_train_and_val_splits_are_allowed(tmp_path, split):
    with pytest.raises(EvaluationError, match="public test labels are withheld"):
        LMKBC2026Evaluator().evaluate(data_dir=tmp_path, split=split, predictions_path=tmp_path / "missing")


def test_public_test_hardlink_is_rejected(tmp_path):
    test_path = write_rows(tmp_path / "test.jsonl", [row([])])
    (tmp_path / "val.jsonl").hardlink_to(test_path)
    with pytest.raises(EvaluationError, match="Cannot evaluate public test"):
        LMKBC2026Evaluator().evaluate(data_dir=tmp_path, split="val", predictions_path=tmp_path / "missing")


def test_empty_gold_file_is_rejected_without_inventing_metrics(tmp_path):
    with pytest.raises(EvaluationError, match="upstream macro metrics are undefined"):
        evaluate(tmp_path, [], [])


def test_prediction_parser_preserves_strings_metadata_and_duplicates(tmp_path):
    rows = [row(["", " ", "123", "123", "Éva"]), row([])]
    rows[0]["metadata"] = {"tag": "retained"}
    path = write_rows(tmp_path / "predictions.jsonl", rows)
    assert list(load_predictions(path)) == rows


@pytest.mark.parametrize("bad", [None, "x", {}, {"SubjectEntity": "A"}, row(None), row("x"), row([["x"]]), row(["x", 1]), row([True]), row([None]), row([], subject=""), row([], relation=1)])
def test_malformed_predictions_are_rejected(tmp_path, bad):
    path = write_rows(tmp_path / "predictions.jsonl", [row([]), bad])
    with pytest.raises(DatasetError) as error:
        list(load_predictions(path))
    assert f"{path}:2:" in str(error.value)


@pytest.mark.parametrize("bad_line", ["{broken}", "\n"])
def test_invalid_prediction_json_has_line_context(tmp_path, bad_line):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(row([])) + "\n" + bad_line, encoding="utf-8")
    with pytest.raises(DatasetError, match="malformed prediction JSON") as error:
        list(load_predictions(path))
    assert f"{path}:2:" in str(error.value)


def test_prediction_read_errors(tmp_path):
    path = tmp_path / "predictions.jsonl"
    with pytest.raises(DatasetError, match="prediction file not found"):
        list(load_predictions(path))
    path.write_bytes(b"\xff")
    with pytest.raises(DatasetError, match="UTF-8"):
        list(load_predictions(path))


def cli(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-m", "llm_kbc.evaluation", "--benchmark", "lm-kbc-2026", "--data-dir", str(tmp_path), "--predictions", str(tmp_path / "predictions.jsonl"), *args],
        capture_output=True, text=True, timeout=20,
    )


def test_cli_json_matches_adapter(tmp_path):
    result = evaluate(tmp_path, [row([["x"], ["y"]])], [row(["x"])])
    completed = cli(tmp_path, "--split", "val")
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == asdict(result)


@pytest.mark.parametrize("args", [(), ("--split", "test"), ("--split", "val")])
def test_cli_invalid_or_missing_input(tmp_path, args):
    completed = cli(tmp_path, *args)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_never_repairs_nested_predictions(tmp_path):
    write_rows(tmp_path / "val.jsonl", [row([["x"]])])
    write_rows(tmp_path / "predictions.jsonl", [row([["x"]])])
    completed = cli(tmp_path, "--split", "val")
    assert completed.returncode == 2
    assert "flat list of strings" in completed.stderr
    assert completed.stdout == ""
