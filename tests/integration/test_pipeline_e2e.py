from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from a2ts.cli import app
from a2ts.models import RawSegment, WordTimestamp

runner = CliRunner()


@patch("a2ts.cli.extract_audio_to_wav")
@patch("a2ts.cli.get_engine")
def test_cli_run_pipeline_end_to_end(
    mock_get_engine: MagicMock, mock_extract: MagicMock, tmp_path: Path
) -> None:
    # 1. Setup dummy video and context dir
    video_file = tmp_path / "test_session.mkv"
    video_file.write_bytes(b"dummy mkv video content")
    context_dir = tmp_path / "contexte"
    context_dir.mkdir()
    (context_dir / "lore.md").write_text(
        "Rencontre avec [[Kaelen]] et [[Garrick]].", encoding="utf-8"
    )

    out_file = tmp_path / "output_transcript.md"

    # 2. Mock audio extraction and engine
    mock_wav = tmp_path / "sample.wav"
    mock_wav.touch()
    mock_extract.return_value = mock_wav

    dummy_engine = MagicMock()
    dummy_engine.transcribe.return_value = [
        RawSegment(
            id=0,
            start=0.0,
            end=5.0,
            text="Bienvenue à tous pour la session.",
            words=[
                WordTimestamp(word="Bienvenue", start=0.0, end=1.0),
                WordTimestamp(word="à", start=1.1, end=1.5),
                WordTimestamp(word="tous", start=1.6, end=2.5),
                WordTimestamp(word="pour", start=2.6, end=3.0),
                WordTimestamp(word="la", start=3.1, end=3.5),
                WordTimestamp(word="session.", start=3.6, end=5.0),
            ],
        )
    ]
    mock_get_engine.return_value = dummy_engine

    # 3. Run CLI command
    result = runner.invoke(
        app,
        [
            "run",
            str(video_file),
            "--context-dir",
            str(context_dir),
            "--output",
            str(out_file),
            "--cache-dir",
            str(tmp_path / ".a2ts"),
            "--no-interactive",
            "--no-refine",
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "Bienvenue à tous pour la session." in content
