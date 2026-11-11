"""CPU-only LM-KBC training conversations and a separate identity-overlap audit.

Only train.jsonl supplies targets. Canonical means the first alias in each gold
group, following few-shot preparation; it is not a separate dataset field.
No tokenization, truncation, model loading, training, or file writing occurs here.
"""

import json
from pathlib import Path

from . import few_shot, zero_shot
from .dataset_inspector import DatasetError, DatasetRecord, load_jsonl
from .prediction_io import validate_prediction


DEFAULT_DATA_DIR = Path("data/lm_kbc_2026")


def _identity(row: dict, location: str) -> tuple[str, str]:
    """Project only subject/relation; never inspect a validation answer field."""
    keys = validate_prediction({
        "SubjectEntity": row.get("SubjectEntity"), "Relation": row.get("Relation"),
        "ObjectEntities": [],
    }, location=location)
    if keys["Relation"] not in zero_shot.QUESTIONS:
        raise DatasetError(f"{location}: unsupported relation: {keys['Relation']!r}")
    return few_shot._subject_key(keys["SubjectEntity"]), keys["Relation"]


def _training_records(data_dir: Path) -> list[DatasetRecord]:
    """Validate the complete training file before returning any conversations."""
    train = data_dir / "train.jsonl"
    # Match the few-shot safeguards against accidentally linking train to val/test.
    # A filename cannot establish provenance of manually copied/renamed contents.
    if train.resolve().name != "train.jsonl" or any(
        path.exists() and train.exists() and path.samefile(train)
        for path in (data_dir / "val.jsonl", data_dir / "test.jsonl")
    ):
        raise DatasetError("Training labels must come from train.jsonl, not a validation/test link")
    records, seen = [], {}
    for line, row in enumerate(load_jsonl(train), start=1):
        location = f"{train}:{line}"
        key = _identity(row, location)
        if key in seen:
            raise DatasetError(f"{location}: duplicate normalized training identity (first at line {seen[key]})")
        seen[key] = line
        records.append(row)
    if not records:
        raise DatasetError(f"{train}: training set must not be empty")
    return records


def prepare_training_data(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict]:
    """Return subject/relation plus three-message chats in original train order.

    data_dir may relocate the dataset (e.g. in tests); the label filename is always
    train.jsonl. Every alias is validated, but only each group's first alias is
    used. Names, repeated labels, numeric strings, and empty answers are preserved.
    Preparation does not open validation/test files or implicitly run the audit.
    """
    train = Path(data_dir) / "train.jsonl"
    records = _training_records(Path(data_dir))
    examples = []
    for line, row in enumerate(records, start=1):
        target = validate_prediction({
            "SubjectEntity": row["SubjectEntity"], "Relation": row["Relation"],
            "ObjectEntities": [aliases[0] for aliases in row["ObjectEntities"]],
        }, location=f"{train}:{line}")
        examples.append({
            "SubjectEntity": target["SubjectEntity"], "Relation": target["Relation"],
            "messages": [
                *zero_shot.build_messages(target["SubjectEntity"], target["Relation"]),
                {"role": "assistant", "content": json.dumps(target["ObjectEntities"], ensure_ascii=False)},
            ],
        })
    return examples


def audit_validation_overlap(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Report normalized pair and cross-relation subject overlap, without filtering.

    Validation contributes identities only: ObjectEntities may be absent or have
    any shape and is never accessed. Report order is validation-file order, then
    training-file order. Line numbers are one-based. Pair overlap is reported,
    not silently removed or allowed to change the prepared training data.
    Matching does not resolve aliases, related facts, or pretraining contamination.
    """
    data_dir = Path(data_dir)
    records = _training_records(data_dir)
    by_subject = {}
    for line, row in enumerate(records, start=1):
        by_subject.setdefault(few_shot._subject_key(row["SubjectEntity"]), []).append((line, row))
    report = {
        "training_records": len(records), "validation_records": 0,
        "pair_overlaps": [], "subject_only_overlaps": [],
        "subject_matching": "NFKC, casefold, collapsed whitespace; no alias resolution",
    }
    validation = data_dir / "val.jsonl"
    try:
        with validation.open("r", encoding="utf-8") as stream:
            for line, text in enumerate(stream, start=1):
                location = f"{validation}:{line}"
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as error:
                    raise DatasetError(f"{location}: malformed JSON: {error.msg}") from error
                if not isinstance(row, dict):
                    raise DatasetError(f"{location}: expected a JSON object")
                subject, relation = _identity(row, location)
                report["validation_records"] += 1
                for train_line, training in by_subject.get(subject, []):
                    group = "pair_overlaps" if training["Relation"] == relation else "subject_only_overlaps"
                    report[group].append({
                        "train_line": train_line, "validation_line": line,
                        "train_subject": training["SubjectEntity"], "train_relation": training["Relation"],
                        "validation_subject": row["SubjectEntity"], "validation_relation": relation,
                    })
    except FileNotFoundError as error:
        raise DatasetError(f"{validation}: validation input file not found") from error
    except UnicodeDecodeError as error:
        raise DatasetError(f"{validation}: validation inputs must be UTF-8 encoded") from error
    except OSError as error:
        raise DatasetError(f"{validation}: could not read validation inputs: {error.strerror}") from error
    return report
