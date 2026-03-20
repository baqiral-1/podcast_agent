from __future__ import annotations

import pytest
import typer

from podcast_agent.cli.app import _parse_agent_model_overrides


def test_agent_model_overrides_support_provider_prefix() -> None:
    model_overrides, provider_overrides = _parse_agent_model_overrides(
        ["analysis=anthropic:claude-3-5-sonnet-20240620"]
    )

    assert model_overrides == {"book_analysis": "claude-3-5-sonnet-20240620"}
    assert provider_overrides == {"book_analysis": "anthropic"}


def test_agent_model_overrides_require_model_when_provider_specified() -> None:
    with pytest.raises(typer.BadParameter) as exc_info:
        _parse_agent_model_overrides(["analysis=anthropic:"])

    assert "AGENT=PROVIDER:MODEL" in str(exc_info.value)
