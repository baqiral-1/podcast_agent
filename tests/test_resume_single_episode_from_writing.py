from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "resume_single_episode_from_writing.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "resume_single_episode_from_writing",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None
assert _SCRIPT_SPEC.loader is not None
resume_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(resume_script)


def test_resume_single_episode_from_writing_parse_args(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resume_single_episode_from_writing.py",
            "--project-id",
            "run_42",
            "--episode-number",
            "2",
        ],
    )

    args = resume_script._parse_args()

    assert args.project_id == "run_42"
    assert args.episode_number == 2


def test_resume_single_episode_from_writing_main_runs_async_entrypoint(
    monkeypatch,
) -> None:
    calls: dict[str, Any] = {}
    original_asyncio_run = asyncio.run

    async def fake_resume(project_id: str, episode_number: int) -> None:
        calls["project_id"] = project_id
        calls["episode_number"] = episode_number

    def fake_asyncio_run(coro: Any) -> None:
        calls["asyncio_run_called"] = True
        original_asyncio_run(coro)

    monkeypatch.setattr(
        resume_script,
        "_parse_args",
        lambda: SimpleNamespace(project_id="run_42", episode_number=2),
    )
    monkeypatch.setattr(
        resume_script,
        "_resume_single_episode_from_writing",
        fake_resume,
    )
    monkeypatch.setattr(resume_script.asyncio, "run", fake_asyncio_run)

    resume_script.main()

    assert calls["asyncio_run_called"] is True
    assert calls["project_id"] == "run_42"
    assert calls["episode_number"] == 2
