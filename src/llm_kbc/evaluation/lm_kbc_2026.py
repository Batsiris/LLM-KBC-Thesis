"""Thin adapter to the unchanged, pinned official LM-KBC 2026 evaluator."""

import hashlib
import importlib
import json
import platform
from importlib.resources import files
from pathlib import Path

from ..dataset_inspector import load_jsonl
from ..prediction_io import load_predictions
from . import EvaluationError, EvaluationResult


class LMKBC2026Evaluator:
    """Evaluate only explicitly selected local train or validation gold files.

    Gold and prediction validation is stricter than upstream's permissive reader.
    Accepted records are passed unchanged to upstream for all metric calculation.
    """

    def evaluate(
        self, *, data_dir: Path, split: str, predictions_path: Path
    ) -> EvaluationResult:
        if split not in ("train", "val"):
            raise EvaluationError(
                "LM-KBC 2026 gold split must be explicitly 'train' or 'val'; "
                "public test labels are withheld and cannot be evaluated"
            )
        gold_path = Path(data_dir) / f"{split}.jsonl"
        test_path = Path(data_dir) / "test.jsonl"
        # Also prevent train/val symlinks or hard links to the public test file.
        if gold_path.resolve().name == "test.jsonl" or (
            gold_path.exists() and test_path.exists() and gold_path.samefile(test_path)
        ):
            raise EvaluationError("Cannot evaluate public test.jsonl: labels are withheld")

        gold_rows = list(load_jsonl(gold_path))
        prediction_rows = list(load_predictions(predictions_path))
        if not gold_rows:
            raise EvaluationError("Gold file has no records; upstream macro metrics are undefined")

        try:
            official = importlib.import_module("llm_kbc._vendor.lm_kbc_2026.evaluate")
        except ModuleNotFoundError as error:
            if error.name != "pandas":
                raise
            raise EvaluationError(
                'Official evaluator requires pandas; install with: python -m pip install -e ".[evaluation]"'
            ) from error

        # Keep matching, prediction deduplication, tolerance, F1 and aggregation
        # entirely upstream. In particular, never sort numeric predictions.
        scores = official.evaluate_per_sr_pair(
            prediction_rows, gold_rows, official.RELATION_TYPE, tolerance=0.05
        )
        macro = official.macro_average_per_relation(scores)
        micro = official.micro_average_per_relation(scores)
        all_relations = "*** All Relations ***"
        provenance = json.loads(
            files("llm_kbc._vendor.lm_kbc_2026").joinpath("provenance.json").read_text(encoding="utf-8")
        )
        return EvaluationResult(
            benchmark="lm-kbc-2026",
            split=split,
            overall={**macro[all_relations], **micro[all_relations]},
            per_relation={
                relation: {**macro[relation], **micro[relation]}
                for relation in sorted(macro) if relation != all_relations
            },
            metadata={
                "upstream": provenance,
                "tolerance": 0.05,
                "python_version": platform.python_version(),
                "pandas_version": official.pd.__version__,
                "gold": {
                    "path": str(gold_path.resolve()),
                    "sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
                    "input_records": len(gold_rows),
                },
                "predictions": {
                    "path": str(Path(predictions_path).resolve()),
                    "sha256": hashlib.sha256(Path(predictions_path).read_bytes()).hexdigest(),
                    "input_records": len(prediction_rows),
                },
                "evaluated_pairs": len(scores),
            },
        )
