"""Evaluate existing predictions with an explicitly selected benchmark and split."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from ..dataset_inspector import DatasetError
from . import EvaluationError, Evaluator
from .lm_kbc_2026 import LMKBC2026Evaluator


def main(argv: Sequence[str] | None = None) -> int:
    """Print full-precision JSON; no prediction generation or file writes occur."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("lm-kbc-2026",))
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args(argv)
    # One implementation today; the protocol allows another adapter later.
    evaluator: Evaluator = LMKBC2026Evaluator()
    try:
        result = evaluator.evaluate(
            data_dir=args.data_dir, split=args.split, predictions_path=args.predictions
        )
    except (DatasetError, EvaluationError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(asdict(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
