"""Deterministic, exploratory format repair of saved zero-shot generations.

No inference, gold labels, or scoring is used. The original strict-baseline
predictions are the coverage reference; both original input files remain intact.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from . import zero_shot
from .dataset_inspector import DatasetError
from .prediction_io import validate_prediction
from .zero_shot import InferenceError, parse_response


POLICY_VERSION = "lm-kbc-format-repair-v1"
# JSON number grammar, excluding NaN/Infinity and nonstandard punctuation.
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_COUNTS = (
    "total_inputs", "valid_responses", "parse_failures", "repair_attempts",
    "repair_successes", "remaining_parse_failures",
)


class RepairError(ValueError):
    """Artifacts are inconsistent or cannot support this repair ablation."""


def _entity_name(name: str) -> bool:
    """Conservative name grammar, not an entity recognizer or factual check.

    Space-separated words start with uppercase letters and otherwise contain
    letters or internal apostrophes/hyphens/periods. A nonfinal alphabetic word
    may end in a period (St. Louis). No digits, quotes around names, or controls.
    """
    if name.casefold() in {"true", "false", "null", "none", "nan", "inf", "infinity"}:
        return False
    words = name.split(" ")
    for index, word in enumerate(words):
        if index < len(words) - 1 and word.endswith(".") and len(word) > 2 and word[:-1].isalpha():
            word = word[:-1]
        if not word or not word[0].isupper() or not word[0].isalpha() or not word[-1].isalpha():
            return False
        for position, char in enumerate(word):
            if char.isalpha():
                continue
            if char not in "'\u2019-." or not 0 < position < len(word) - 1:
                return False
            if not word[position - 1].isalpha() or not word[position + 1].isalpha():
                return False
    return True


def repair_response(response: str) -> tuple[list[str] | None, str | None, str | None]:
    """Try fixed syntax-only rules on a response already rejected by the parser.

    Return objects, rule, and rejection reason. None objects means no repair;
    it is distinct from a model's valid empty-array answer.
    """
    text = response.strip()
    if _NUMBER.fullmatch(text):
        return parse_response(f"[{text}]"), "standalone_number", None
    if text.startswith("[") and text.endswith("]"):
        # Fixed syntactic interpretation; a comma inside a real entity name
        # remains an irreducible ambiguity, documented for this ablation.
        names = text[1:-1].split(", ")
        if len(names) >= 2 and all(_entity_name(name) for name in names):
            return parse_response(json.dumps(names)), "unquoted_entity_list", None
        return None, "unquoted_entity_list", "Requires two or more names separated by ', ' and the conservative name grammar"
    return None, None, "No eligible repair rule"


def _read_jsonl(path: Path) -> tuple[bytes, list]:
    """Read a snapshot, retaining original baseline line bytes for unchanged rows."""
    content = path.read_bytes()
    rows = []
    for index, line in enumerate(content.splitlines(keepends=True), start=1):
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RepairError(f"{path}:{index}: invalid UTF-8 JSONL: {error}") from error
    return content, rows


def _publish(path: Path, content: bytes) -> None:
    """Publish a closed temporary file atomically without replacing any target.

    A same-directory hard link provides no-clobber publication on NTFS/Linux.
    Unsupported filesystems fail rather than falling back to an overwrite.
    """
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_repair(raw_generations: Path, baseline_predictions: Path, output: Path) -> dict:
    """Validate paired artifacts completely, then publish a separate ablation.

    Every strict parse failure counts as one repair attempt, including ineligible
    responses. Only successful rule applications change prediction rows.
    """
    raw_generations, baseline_predictions, output = map(
        Path, (raw_generations, baseline_predictions, output)
    )
    audit_path = Path(str(output) + ".audit.jsonl")
    metadata_path = Path(str(output) + ".metadata.json")
    for path in (output, audit_path, metadata_path):
        if path.exists() or path.is_symlink():
            raise RepairError(f"Refusing to overwrite existing file: {path}")
    raw_bytes, raw_rows = _read_jsonl(raw_generations)
    baseline_bytes, baseline_rows = _read_jsonl(baseline_predictions)
    if not baseline_rows or len(raw_rows) != len(baseline_rows):
        raise RepairError("Raw and baseline files must have equal, nonzero record counts")
    baseline_lines = baseline_bytes.splitlines(keepends=True)
    seen = set()
    predictions, audit = [], []
    per_relation = {}
    for index, (raw, baseline) in enumerate(zip(raw_rows, baseline_rows), start=1):
        location = f"{raw_generations}:{index}"
        baseline = validate_prediction(baseline, location=f"{baseline_predictions}:{index}")
        if not isinstance(raw, dict):
            raise RepairError(f"{location}: expected a raw-generation JSON object")
        validate_prediction({**raw, "ObjectEntities": []}, location=location)
        if type(raw.get("input_line")) is not int or raw["input_line"] != index:
            raise RepairError(f"{location}: input_line must be the contiguous one-based input index")
        if not isinstance(raw.get("response"), str):
            raise RepairError(f"{location}: response must be a string")
        if "error" not in raw or raw.get("parsing_status") not in ("valid", "parse_failed"):
            raise RepairError(f"{location}: missing error or invalid parsing_status (incomplete run)")
        identity = (raw["SubjectEntity"], raw["Relation"])
        if identity != (baseline["SubjectEntity"], baseline["Relation"]) or identity in seen:
            raise RepairError(f"{location}: identities must match in order and be unique")
        seen.add(identity)
        counts = per_relation.setdefault(raw["Relation"], dict.fromkeys(_COUNTS, 0))
        counts["total_inputs"] += 1
        try:
            parsed = parse_response(raw["response"])
        except InferenceError:
            if raw["parsing_status"] != "parse_failed" or not isinstance(raw["error"], str) or not raw["error"]:
                raise RepairError(f"{location}: parse-failure status/error disagrees with strict parser")
            if baseline["ObjectEntities"] != []:
                raise RepairError(f"{location}: strict parse failure must have a baseline abstention")
            counts["parse_failures"] += 1
            counts["repair_attempts"] += 1
            objects, rule, error = repair_response(raw["response"])
            success = objects is not None
            prediction = validate_prediction({**baseline, "ObjectEntities": objects if success else []})
            predictions.append(
                (json.dumps(prediction, ensure_ascii=True) + "\n").encode("utf-8")
                if success else baseline_lines[index - 1]
            )
            counts["repair_successes"] += int(success)
            counts["remaining_parse_failures"] += int(not success)
            audit.append({
                "input_line": index, "SubjectEntity": identity[0], "Relation": identity[1],
                "raw_response": raw["response"], "original_parse_error": raw["error"],
                "rule": rule, "success": success, "error": error, "prediction": prediction,
            })
        else:
            if raw["parsing_status"] != "valid" or raw["error"] is not None or parsed != baseline["ObjectEntities"]:
                raise RepairError(f"{location}: valid response disagrees with status/error or baseline prediction")
            counts["valid_responses"] += 1
            predictions.append(baseline_lines[index - 1])
    prediction_bytes = b"".join(predictions)
    audit_bytes = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in audit).encode("utf-8")
    report = {
        "policy_version": POLICY_VERSION,
        "exploratory": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "strict_parser_source_sha256": hashlib.sha256(Path(zero_shot.__file__).read_bytes()).hexdigest(),
        "inputs": {
            "raw_generations": {"path": str(raw_generations.resolve()), "sha256": hashlib.sha256(raw_bytes).hexdigest()},
            "baseline_predictions": {"path": str(baseline_predictions.resolve()), "sha256": hashlib.sha256(baseline_bytes).hexdigest()},
        },
        "output_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        **{field: sum(counts[field] for counts in per_relation.values()) for field in _COUNTS},
        "per_relation": dict(sorted(per_relation.items())),
    }
    # Validation finishes before creating any output artifact. Publish predictions
    # last so they never appear without their complete audit and report.
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish(audit_path, audit_bytes)
    _publish(metadata_path, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _publish(output, prediction_bytes)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Repair saved artifacts only and print counts/provenance as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_repair(args.raw_generations, args.baseline_predictions, args.output)
    except (RepairError, DatasetError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
