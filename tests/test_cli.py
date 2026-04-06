"""Unit tests for CLI argument parsing helpers."""

from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from podcast_agent.cli.app import _normalize_tts_provider, _parse_sub_themes, app


runner = CliRunner()


class TestParseSubThemes:
    def test_empty_is_allowed(self):
        assert _parse_sub_themes(None) == []
        assert _parse_sub_themes("") == []

    def test_trim_and_dedupe_preserves_order(self):
        result = _parse_sub_themes(" borders,displacement,borders, governance ")
        assert result == ["borders", "displacement", "governance"]

    def test_rejects_empty_entries(self):
        with pytest.raises(typer.BadParameter, match="non-empty"):
            _parse_sub_themes("valid, ,other")

    def test_rejects_more_than_fifteen(self):
        raw = "a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16"
        with pytest.raises(typer.BadParameter, match="at most 15"):
            _parse_sub_themes(raw)


class TestSynthesizeAudioCommand:
    def test_synthesize_audio_reports_summary(self, monkeypatch, tmp_path):
        class FakeOrchestrator:
            def __init__(self, settings) -> None:
                assert settings.tts.provider == "openai-compatible"
                self.settings = settings

            async def synthesize_audio_from_run(self, run_dir):
                assert run_dir == tmp_path
                return {
                    "run_dir": str(run_dir),
                    "processed": 2,
                    "succeeded": 2,
                    "failed": 0,
                    "skipped": 1,
                    "skipped_episodes": [3],
                    "failures": [],
                }

        monkeypatch.setattr(
            "podcast_agent.pipeline.orchestrator.PipelineOrchestrator",
            FakeOrchestrator,
        )

        result = runner.invoke(
            app,
            ["synthesize-audio", str(tmp_path), "--tts-provider", "openai"],
        )

        assert result.exit_code == 0
        assert "Processed: 2" in result.output
        assert "Succeeded: 2" in result.output
        assert "Skipped episodes: 3" in result.output

    def test_synthesize_audio_exits_nonzero_on_failure(self, monkeypatch, tmp_path):
        class FakeOrchestrator:
            def __init__(self, settings) -> None:
                assert settings.tts.provider == "kokoro"
                self.settings = settings

            async def synthesize_audio_from_run(self, run_dir):
                raise RuntimeError("No render manifests found")

        monkeypatch.setattr(
            "podcast_agent.pipeline.orchestrator.PipelineOrchestrator",
            FakeOrchestrator,
        )

        result = runner.invoke(
            app,
            ["synthesize-audio", str(tmp_path), "--tts-provider", "kokoro"],
        )

        assert result.exit_code == 1
        assert "Audio synthesis failed: No render manifests found" in result.output

    def test_synthesize_audio_requires_tts_provider(self, tmp_path):
        result = runner.invoke(app, ["synthesize-audio", str(tmp_path)])

        assert result.exit_code != 0
        assert "Missing option '--tts-provider'" in result.output

    def test_invalid_tts_provider_returns_parameter_error(self, tmp_path):
        result = runner.invoke(
            app,
            ["synthesize-audio", str(tmp_path), "--tts-provider", "invalid"],
        )

        assert result.exit_code != 0
        assert "Invalid --tts-provider value 'invalid'" in result.output


class TestNormalizeTTSProvider:
    def test_openai_alias_maps_to_openai_compatible(self):
        assert _normalize_tts_provider("openai") == "openai-compatible"

    def test_kokoro_is_preserved(self):
        assert _normalize_tts_provider("kokoro") == "kokoro"

    def test_invalid_provider_raises(self):
        with pytest.raises(typer.BadParameter, match="Invalid --tts-provider value"):
            _normalize_tts_provider("bad-provider")


class TestRunCommand:
    def test_run_uses_resolved_tts_provider_for_settings_and_config(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        captured: dict[str, str] = {}

        class FakeOrchestrator:
            def __init__(self, settings) -> None:
                captured["settings_provider"] = settings.tts.provider

            async def run_multi_book_podcast(
                self,
                *,
                source_paths,
                theme,
                episode_count,
                config,
                theme_elaboration,
                sub_themes,
                titles,
                authors,
                project_id,
            ):
                captured["config_provider"] = config.tts_provider
                return SimpleNamespace(
                    project_id="proj-1",
                    status=SimpleNamespace(value="complete"),
                    books=[1, 2],
                )

        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost:5432/podcast_agent")
        monkeypatch.setattr(
            "podcast_agent.pipeline.orchestrator.PipelineOrchestrator",
            FakeOrchestrator,
        )

        result = runner.invoke(
            app,
            [
                "run",
                str(tmp_path / "book1.txt"),
                str(tmp_path / "book2.txt"),
                "--theme",
                "partition",
                "--skip-audio",
                "--tts-provider",
                "openai",
            ],
        )

        assert result.exit_code == 0
        assert captured["settings_provider"] == "openai-compatible"
        assert captured["config_provider"] == "openai-compatible"
