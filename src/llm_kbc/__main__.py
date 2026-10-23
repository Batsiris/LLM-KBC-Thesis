"""Minimal command-line entry point for ``python -m llm_kbc``."""

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments; experiments are not implemented yet."""
    parser = argparse.ArgumentParser(
        prog="python -m llm_kbc",
        description="Knowledge base construction with large language models: project scaffold.",
    )
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
