"""Smoke test for the installed package's command-line interface."""

import subprocess
import sys


def test_cli_help() -> None:
    """The module entry point should print help and exit successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "llm_kbc", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "--help" in result.stdout
    assert "Knowledge base construction" in result.stdout
