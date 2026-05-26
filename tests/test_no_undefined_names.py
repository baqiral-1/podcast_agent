"""Static guard: no undefined names (ruff F821) in the package source.

This catches the class of regression that shipped in the residue->section_progression
refactor: a `NameError` (`name 'section_order' is not defined`) that passed pytest and
`compileall` but blocked a real pipeline run. Runs in the same pytest/CI loop the dev
already uses.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "podcast_agent"


def _ruff_cmd() -> list[str] | None:
    if shutil.which("ruff"):
        return ["ruff"]
    # Fall back to the module entry point when only the wheel is installed.
    try:
        import ruff  # noqa: F401
    except Exception:
        return None
    return [sys.executable, "-m", "ruff"]


def test_no_undefined_names_in_src() -> None:
    cmd = _ruff_cmd()
    if cmd is None:
        pytest.skip("ruff is not installed; skipping undefined-name guard")
    result = subprocess.run(
        [*cmd, "check", "--select", "F821", "--no-cache", str(_SRC)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "ruff reported undefined names (F821) in src/podcast_agent:\n"
        f"{result.stdout}\n{result.stderr}"
    )
