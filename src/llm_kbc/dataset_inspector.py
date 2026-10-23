"""Validate and summarize local LM-KBC 2026 JSONL splits without changing them.

Each outer ObjectEntities entry is one object; its inner strings are aliases.
Test labels are withheld, so test records never contribute to gold statistics.
"""

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast


SPLITS = ("train", "val", "test")


class DatasetError(ValueError):
    """A dataset could not be read or does not match the expected schema."""


class DatasetRecord(TypedDict):
    """Required fields; additional dataset metadata is accepted by the loader."""

    SubjectEntity: str
    Relation: str
    ObjectEntities: list[list[str]]


def _validate_record(value: object, location: str) -> DatasetRecord:
    """Validate structure without normalizing entity names or changing labels."""
    if not isinstance(value, dict):
        raise DatasetError(f"{location}: expected a JSON object")

    required = ("SubjectEntity", "Relation", "ObjectEntities")
    missing = [key for key in required if key not in value]
    if missing:
        raise DatasetError(f"{location}: missing required field(s): {', '.join(missing)}")

    for key in ("SubjectEntity", "Relation"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise DatasetError(f"{location}: {key} must be a non-empty string")

    objects = value["ObjectEntities"]
    if not isinstance(objects, list):
        raise DatasetError(f"{location}: ObjectEntities must be a list of alias lists")
    for index, aliases in enumerate(objects):
        field = f"ObjectEntities[{index}]"
        if not isinstance(aliases, list) or not aliases:
            raise DatasetError(f"{location}: {field} must be a non-empty list of aliases")
        if any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
            raise DatasetError(f"{location}: {field} aliases must be non-empty strings")

    return cast(DatasetRecord, value)


def load_jsonl(path: str | Path) -> Iterator[DatasetRecord]:
    """Yield validated UTF-8 records, raising DatasetError with source context.

    Every physical line must contain one JSON object; blank lines are errors.
    An empty file is valid. Files are opened read-only when iteration begins.
    ObjectEntities may be empty, but each supplied alias group must be non-empty.
    """
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                location = f"{path}:{line_number}"
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetError(
                        f"{location}: malformed JSON at column {error.colno}: {error.msg}"
                    ) from error
                yield _validate_record(value, location)
    except FileNotFoundError as error:
        raise DatasetError(f"{path}: dataset file not found") from error
    except UnicodeDecodeError as error:
        raise DatasetError(f"{path}: dataset must be UTF-8 encoded") from error
    except OSError as error:
        raise DatasetError(f"{path}: could not read dataset: {error.strerror}") from error


@dataclass(frozen=True)
class DuplicatePair:
    """An exact subject-relation pair and its total occurrences in one split."""

    subject: str
    relation: str
    occurrences: int


@dataclass(frozen=True)
class SplitReport:
    """Split statistics; unavailable gold statistics are None, never zero."""

    split: str
    total_records: int
    relation_counts: dict[str, int]
    gold_object_count_distribution: dict[int, int] | None
    empty_object_entities_records: int
    empty_gold_records: int | None
    unlabeled_records: int
    duplicate_pairs: tuple[DuplicatePair, ...]

    @property
    def duplicate_extra_records(self) -> int:
        """Number of occurrences beyond the first record for each pair."""
        return sum(pair.occurrences - 1 for pair in self.duplicate_pairs)


def inspect_split(path: str | Path, *, split: str) -> SplitReport:
    """Inspect one split; label availability is determined by explicit split name.

    All test records are unlabeled regardless of ObjectEntities contents. Pair
    comparison is exact and occurs within this split, without alias resolution.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {', '.join(SPLITS)}")

    relations: Counter[str] = Counter()
    object_counts: Counter[int] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    total = 0
    empty = 0
    for record in load_jsonl(path):
        total += 1
        relations[record["Relation"]] += 1
        pairs[(record["SubjectEntity"], record["Relation"])] += 1
        # Count object groups, not the alternative names inside each group.
        count = len(record["ObjectEntities"])
        empty += count == 0
        if split != "test":
            object_counts[count] += 1

    return SplitReport(
        split=split,
        total_records=total,
        relation_counts=dict(sorted(relations.items())),
        gold_object_count_distribution=(
            dict(sorted(object_counts.items())) if split != "test" else None
        ),
        empty_object_entities_records=empty,
        empty_gold_records=empty if split != "test" else None,
        unlabeled_records=total if split == "test" else 0,
        duplicate_pairs=tuple(
            DuplicatePair(subject, relation, occurrences)
            for (subject, relation), occurrences in sorted(pairs.items())
            if occurrences > 1
        ),
    )


def format_report(report: SplitReport) -> str:
    """Render deterministic, human-readable statistics with explicit label status."""
    lines = [
        f"[{report.split}]",
        f"Total records: {report.total_records}",
        f"Labels: {'unlabeled (withheld)' if report.split == 'test' else 'gold available'}",
        f"Unlabeled records: {report.unlabeled_records}",
        "Count per relation:",
    ]
    lines.extend(f"  {relation}: {count}" for relation, count in report.relation_counts.items())
    if report.gold_object_count_distribution is None:
        lines.extend([
            "Gold objects per record: unavailable (unlabeled)",
            f"Empty ObjectEntities arrays (withheld placeholders): {report.empty_object_entities_records}",
            "Empty gold answers: unavailable (unlabeled)",
        ])
    else:
        lines.append("Gold objects per record (object count: record count):")
        lines.extend(
            f"  {count}: {records}"
            for count, records in report.gold_object_count_distribution.items()
        )
        lines.extend([
            f"Empty ObjectEntities arrays: {report.empty_object_entities_records}",
            f"Empty gold answers: {report.empty_gold_records}",
        ])
    lines.extend([
        f"Duplicate subject-relation pairs: {len(report.duplicate_pairs)}",
        f"Extra records from duplicate pairs: {report.duplicate_extra_records}",
    ])
    for pair in report.duplicate_pairs:
        # JSON escaping keeps names containing tabs or newlines unambiguous.
        label = json.dumps([pair.subject, pair.relation], ensure_ascii=True)
        lines.append(f"  {label}: {pair.occurrences} occurrences")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect the three expected files, reporting input errors without a traceback."""
    parser = argparse.ArgumentParser(
        description="Inspect local LM-KBC 2026 datasets; test labels are withheld."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/lm_kbc_2026"),
        help="directory containing train.jsonl, val.jsonl, and test.jsonl (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        # Validate all splits before printing, so a failure cannot look like success.
        reports = [
            inspect_split(args.data_dir / f"{split}.jsonl", split=split)
            for split in SPLITS
        ]
    except DatasetError as error:
        parser.error(str(error))
    print("\n\n".join(format_report(report) for report in reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
