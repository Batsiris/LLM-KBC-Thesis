"""Strict prediction parsing for the LM-KBC 2026 flat-string JSONL format.

Gold parsing remains in dataset_inspector.load_jsonl. No inference, formatting
repair, deduplication, or scoring is performed here.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict, cast

from .dataset_inspector import DatasetError


class PredictionRecord(TypedDict):
    """LM-KBC prediction fields; object strings are passed through unchanged."""

    SubjectEntity: str
    Relation: str
    ObjectEntities: list[str]


def validate_prediction(row: object, *, location: str = "prediction") -> PredictionRecord:
    """Validate one prediction without repairing, normalizing, or dropping values."""
    if not isinstance(row, dict):
        raise DatasetError(f"{location}: expected a prediction JSON object")
    missing = [
        key for key in ("SubjectEntity", "Relation", "ObjectEntities")
        if key not in row
    ]
    if missing:
        raise DatasetError(
            f"{location}: missing required prediction field(s): {', '.join(missing)}"
        )
    for key in ("SubjectEntity", "Relation"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise DatasetError(f"{location}: {key} must be a non-empty string")
    objects = row["ObjectEntities"]
    if not isinstance(objects, list) or any(not isinstance(v, str) for v in objects):
        raise DatasetError(
            f"{location}: prediction ObjectEntities must be a flat list of strings"
        )
    return cast(PredictionRecord, row)


def load_predictions(path: str | Path) -> Iterator[PredictionRecord]:
    """Yield strict UTF-8 prediction rows with file/line context on errors.

    Empty prediction files and object lists are valid. Object strings, including
    empty strings and numeric strings, are never normalized or filtered here.
    Extra metadata and duplicate subject-relation rows are preserved.
    """
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                location = f"{path}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetError(
                        f"{location}: malformed prediction JSON at column {error.colno}: {error.msg}"
                    ) from error
                yield validate_prediction(row, location=location)
    except FileNotFoundError as error:
        raise DatasetError(f"{path}: prediction file not found") from error
    except UnicodeDecodeError as error:
        raise DatasetError(f"{path}: predictions must be UTF-8 encoded") from error
    except OSError as error:
        raise DatasetError(f"{path}: could not read predictions: {error.strerror}") from error
