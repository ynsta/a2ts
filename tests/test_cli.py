"""Basic tests for a2ts CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from a2ts.cli import app

runner = CliRunner()


def test_info_command() -> None:
    """Test that info command executes cleanly."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "ready" in result.output.lower()


def test_help_command() -> None:
    """Test that CLI help lists all primary subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "extract-audio" in result.output
    assert "extract-vocab" in result.output
    assert "info" in result.output


def test_extract_vocab_command(tmp_path: Path) -> None:
    """Test extract-vocab command with a context directory."""
    lore_file = tmp_path / "lore.md"
    lore_file.write_text("Rencontre avec [[Valeros]] et [[Seoni]].", encoding="utf-8")

    result = runner.invoke(app, ["extract-vocab", "--context-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Extracted 2 entity mentions" in result.output
    assert "Valeros" in result.output
    assert "Seoni" in result.output


@patch("a2ts.cli.extract_audio_to_wav")
def test_extract_audio_command(mock_extract: MagicMock, tmp_path: Path) -> None:
    """Test extract-audio command invokes extraction logic."""
    media_file = tmp_path / "session.mp4"
    media_file.touch()
    out_wav = tmp_path / "audio_cache" / "session.wav"
    mock_extract.return_value = out_wav

    result = runner.invoke(
        app,
        [
            "extract-audio",
            str(media_file),
            "--output-dir",
            str(tmp_path / "audio_cache"),
        ],
    )
    assert result.exit_code == 0
    assert "Audio extracted" in result.output
    mock_extract.assert_called_once_with(media_file, tmp_path / "audio_cache")
