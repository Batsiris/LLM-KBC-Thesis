"""Synthetic fixtures covering schema errors, gold counts, and withheld labels."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from llm_kbc.dataset_inspector import (
    DatasetError,
    DuplicatePair,
    format_report,
    inspect_split,
    load_jsonl,
)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    """Write a small UTF-8 fixture with one object per physical line."""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def record(subject: str = "A", relation: str = "R", objects=None) -> dict:
    """Build a fresh fixture record; None defaults to an empty outer list."""
    return {
        "SubjectEntity": subject,
        "Relation": relation,
        "ObjectEntities": [] if objects is None else objects,
    }


def test_loader_preserves_unicode_aliases_and_metadata(tmp_path: Path) -> None:
    row = record("Lošinj", objects=[["Torvalds", "Linus Torvalds"], ["Eva Hedlund"]])
    row["metadata"] = "retained"
    path = write_jsonl(tmp_path / "train.jsonl", [row])
    before = path.read_bytes()

    assert list(load_jsonl(path)) == [row]
    assert path.read_bytes() == before


def test_nested_gold_counts_and_true_empty_validation_labels(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "val.jsonl", [
        record("A", "Z", [["one", "alias 1", "alias 2"], ["two"]]),
        record("A", "Z"),
        record("B", "R", [["three"]]),
        record("C", "R", [["four"]]),
        record("A", "Z", [["five"]]),
        record("A", "R"),  # Same subject, different relation is a different pair.
    ])

    report = inspect_split(path, split="val")

    assert report.total_records == 6
    assert report.relation_counts == {"R": 3, "Z": 3}
    assert list(report.relation_counts) == ["R", "Z"]
    assert report.gold_object_count_distribution == {0: 2, 1: 3, 2: 1}
    assert report.empty_object_entities_records == 2
    assert report.empty_gold_records == 2
    assert report.unlabeled_records == 0
    assert report.duplicate_pairs == (DuplicatePair("A", "Z", 3),)
    assert report.duplicate_extra_records == 2
    assert "Empty gold answers: 2" in format_report(report)


def test_test_placeholders_are_unlabeled(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "test.jsonl", [record(), record("B")])

    report = inspect_split(path, split="test")

    assert report.total_records == report.unlabeled_records == 2
    assert report.relation_counts == {"R": 2}
    assert report.empty_object_entities_records == 2
    assert report.empty_gold_records is None
    assert report.gold_object_count_distribution is None
    assert report.duplicate_pairs == ()
    output = format_report(report)
    assert "Labels: unlabeled (withheld)" in output
    assert "Empty ObjectEntities arrays (withheld placeholders): 2" in output
    assert "Empty gold answers: unavailable (unlabeled)" in output


def test_split_policy_does_not_infer_gold_from_test_contents(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "test.jsonl", [record(objects=[["placeholder"]])])
    report = inspect_split(path, split="test")
    assert report.unlabeled_records == 1
    assert report.gold_object_count_distribution is None
    assert report.empty_gold_records is None


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_empty_file(tmp_path: Path, split: str) -> None:
    path = write_jsonl(tmp_path / f"{split}.jsonl", [])
    report = inspect_split(path, split=split)
    assert report.total_records == 0
    assert report.relation_counts == {}
    assert report.duplicate_pairs == ()
    assert report.duplicate_extra_records == 0
    assert report.gold_object_count_distribution == (None if split == "test" else {})
    assert report.empty_gold_records == (None if split == "test" else 0)


def test_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    with pytest.raises(DatasetError, match="dataset file not found") as error:
        list(load_jsonl(path))
    assert str(path) in str(error.value)


@pytest.mark.parametrize("bad_line", ['{"SubjectEntity":', "\n", "null", "[]"])
def test_malformed_line_has_path_and_physical_line_number(
    tmp_path: Path, bad_line: str
) -> None:
    path = write_jsonl(tmp_path / "bad.jsonl", [record()])
    with path.open("a", encoding="utf-8") as stream:
        stream.write(bad_line)
    with pytest.raises(DatasetError) as error:
        list(load_jsonl(path))
    assert f"{path}:2:" in str(error.value)
    assert "malformed JSON" in str(error.value) or "expected a JSON object" in str(error.value)


@pytest.mark.parametrize("field", ["SubjectEntity", "Relation", "ObjectEntities"])
def test_missing_required_field(tmp_path: Path, field: str) -> None:
    row = record()
    del row[field]
    path = write_jsonl(tmp_path / "bad.jsonl", [row])
    with pytest.raises(DatasetError) as error:
        list(load_jsonl(path))
    assert f"{path}:1: missing required field(s): {field}" in str(error.value)


@pytest.mark.parametrize(("field", "value"), [
    ("SubjectEntity", None),
    ("SubjectEntity", " "),
    ("Relation", 1),
    ("Relation", ""),
    ("ObjectEntities", None),
    ("ObjectEntities", "object"),
    ("ObjectEntities", ["flat object"]),
    ("ObjectEntities", [[]]),
    ("ObjectEntities", [[None]]),
    ("ObjectEntities", [[1]]),
    ("ObjectEntities", [[" "]]),
    ("ObjectEntities", [[["too deep"]]]),
])
def test_invalid_field_shapes(tmp_path: Path, field: str, value: object) -> None:
    row = record()
    row[field] = value
    path = write_jsonl(tmp_path / "bad.jsonl", [row])
    with pytest.raises(DatasetError) as error:
        list(load_jsonl(path))
    assert f"{path}:1:" in str(error.value)
    assert field in str(error.value)


def test_invalid_encoding(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"\xff\n")
    with pytest.raises(DatasetError, match="UTF-8") as error:
        list(load_jsonl(path))
    assert str(path) in str(error.value)


def test_unknown_split_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        inspect_split(tmp_path / "train.jsonl", split="typo")


def test_cli_reports_all_splits_without_changing_inputs(tmp_path: Path) -> None:
    paths = [
        write_jsonl(tmp_path / f"{split}.jsonl", [record()])
        for split in ("train", "val", "test")
    ]
    before = [path.read_bytes() for path in paths]
    command = [sys.executable, "-m", "llm_kbc.dataset_inspector", "--data-dir", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.index("[train]") < result.stdout.index("[val]") < result.stdout.index("[test]")
    assert result.stdout.count("Empty gold answers: 1") == 2
    assert "Empty gold answers: unavailable (unlabeled)" in result.stdout
    assert [path.read_bytes() for path in paths] == before
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert repeated.returncode == 0, repeated.stderr
    assert repeated.stdout == result.stdout


@pytest.mark.parametrize("malformed", [False, True])
def test_cli_input_errors_are_actionable(tmp_path: Path, malformed: bool) -> None:
    if malformed:
        (tmp_path / "train.jsonl").write_text("{broken}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "llm_kbc.dataset_inspector", "--data-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "train.jsonl" in result.stderr
    assert ("malformed JSON" if malformed else "dataset file not found") in result.stderr
    assert "Traceback" not in result.stderr
