"""Small interface for benchmark-specific evaluation, separate from experiments.

Metric names may be shared across benchmarks even when their semantics differ.
Always interpret a result using its benchmark and evaluator provenance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class EvaluationError(ValueError):
    """Evaluation inputs or environment do not support a valid evaluation."""


@dataclass(frozen=True)
class EvaluationResult:
    """Full-precision benchmark metrics and metadata describing their origin."""

    benchmark: str
    split: str
    overall: dict[str, float]
    per_relation: dict[str, dict[str, float]]
    metadata: dict[str, Any]


class Evaluator(Protocol):
    """A dataset-specific evaluator selects and parses its own gold format."""

    def evaluate(
        self, *, data_dir: Path, split: str, predictions_path: Path
    ) -> EvaluationResult:
        """Evaluate supplied predictions; do not generate or repair predictions."""
        ...
