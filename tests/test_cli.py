"""Basic tests for a2ts CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from a2ts.cli import app
from a2ts.models import (
    AlignedTurn,
    RawSegment,
    SpeakersMapping,
    SpeakerTurn,
    WordTimestamp,
)

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
    assert "review" in result.output
    assert "split" in result.output
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


@patch("a2ts.cli.probe_media")
@patch("a2ts.cli.get_engine")
@patch("a2ts.cli.extract_audio_to_wav")
def test_run_caching_and_provenance(
    mock_extract: MagicMock,
    mock_get_engine: MagicMock,
    mock_probe: MagicMock,
    tmp_path: Path,
) -> None:
    """Test run caching of raw segments and SessionMetadata persistence."""
    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"dummy video data for hash")
    out_file = tmp_path / "out.md"
    cache_dir = tmp_path / ".a2ts"
    context_dir = tmp_path / "context"
    context_dir.mkdir()

    mock_extract.return_value = tmp_path / "audio.wav"
    mock_probe.return_value = {"format": {"duration": "42.5"}}

    dummy_engine = MagicMock()
    dummy_engine.transcribe.return_value = [
        RawSegment(
            id=0,
            start=0.0,
            end=4.0,
            text="Testing transcription cache.",
            words=[
                WordTimestamp(word="Testing", start=0.0, end=1.0),
                WordTimestamp(word="transcription", start=1.1, end=2.5),
                WordTimestamp(word="cache.", start=2.6, end=4.0),
            ],
        )
    ]
    mock_get_engine.return_value = dummy_engine

    # 1. First run: should invoke transcriber and create cache
    result = runner.invoke(
        app,
        [
            "run",
            str(media_file),
            "--engine",
            "whisper",
            "--model-name",
            "custom-whisper-v3",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
            "--context-dir",
            str(context_dir),
            "--output",
            str(out_file),
            "--cache-dir",
            str(cache_dir),
            "--no-interactive",
            "--no-refine",
            "--no-diarize",
        ],
    )
    assert result.exit_code == 0
    dummy_engine.transcribe.assert_called_once()

    # Verify session.json
    session_file = cache_dir / "session.json"
    assert session_file.is_file()
    session_data = json.loads(session_file.read_text(encoding="utf-8"))
    assert session_data["engine"] == "whisper"
    assert session_data["model_name"] == "custom-whisper-v3"
    assert session_data["duration_seconds"] == 42.5
    assert session_data["output_path"] == str(out_file)

    # Verify turns.json has SPEAKER_00 fallback
    turns_file = cache_dir / "turns.json"
    assert turns_file.is_file()
    turns_data = json.loads(turns_file.read_text(encoding="utf-8"))
    assert len(turns_data) >= 1
    assert turns_data[0]["cluster_id"] == "SPEAKER_00"

    # Verify raw segments cache file exists
    transcripts_dir = cache_dir / "transcripts"
    cached_files = list(transcripts_dir.glob("*_whisper.json"))
    assert len(cached_files) == 1

    # 2. Second run: should hit segment cache and NOT call transcribe
    dummy_engine.transcribe.reset_mock()
    result2 = runner.invoke(
        app,
        [
            "run",
            str(media_file),
            "--engine",
            "whisper",
            "--model-name",
            "custom-whisper-v3",
            "--context-dir",
            str(context_dir),
            "--output",
            str(out_file),
            "--cache-dir",
            str(cache_dir),
            "--no-interactive",
            "--no-refine",
            "--no-diarize",
        ],
    )
    assert result2.exit_code == 0
    dummy_engine.transcribe.assert_not_called()


@patch("a2ts.cli.diarize_segments")
@patch("a2ts.cli.get_engine")
@patch("a2ts.cli.extract_audio_to_wav")
def test_run_with_diarization(
    mock_extract: MagicMock,
    mock_get_engine: MagicMock,
    mock_diarize: MagicMock,
    tmp_path: Path,
) -> None:
    """Test run executes diarize_segments when --diarize is active."""
    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"dummy")
    out_file = tmp_path / "out.md"
    cache_dir = tmp_path / ".a2ts"

    mock_extract.return_value = tmp_path / "audio.wav"
    dummy_engine = MagicMock()
    dummy_engine.transcribe.return_value = [
        RawSegment(id=0, start=0.0, end=2.0, text="First"),
        RawSegment(id=1, start=2.5, end=5.0, text="Second"),
    ]
    mock_get_engine.return_value = dummy_engine

    mock_diarize.return_value = [
        SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=1, start=2.5, end=5.0, cluster_id="SPEAKER_01"),
    ]

    result = runner.invoke(
        app,
        [
            "run",
            str(media_file),
            "--num-speakers",
            "2",
            "--output",
            str(out_file),
            "--cache-dir",
            str(cache_dir),
            "--no-interactive",
            "--no-refine",
        ],
    )
    assert result.exit_code == 0
    mock_diarize.assert_called_once()
    assert out_file.is_file()


@patch("a2ts.cli.run_interactive_review")
def test_review_command(mock_review: MagicMock, tmp_path: Path) -> None:
    """Test review subcommand loads cached turns and regenerates transcript."""
    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir(parents=True)
    out_file = tmp_path / "reviewed.md"

    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Hello world.",
        )
    ]
    (session_dir / "turns.json").write_text(
        json.dumps([t.model_dump() for t in turns]), encoding="utf-8"
    )
    (session_dir / "speakers_mapping.json").write_text(
        json.dumps(SpeakersMapping().model_dump()), encoding="utf-8"
    )

    session_meta = {
        "media_path": "/path/video.mkv",
        "media_hash": "1234abcd",
        "duration_seconds": 10.0,
        "engine": "whisper",
        "model_name": "large-v3",
        "prompt_hash": "",
        "time_slice_minutes": 15.0,
        "created_at": "2026-09-21T00:00:00Z",
        "output_path": "transcript.md",
    }
    (session_dir / "session.json").write_text(
        json.dumps(session_meta), encoding="utf-8"
    )
    audio_dir = session_dir / "audio_cache"
    audio_dir.mkdir()
    audio_file = audio_dir / "1234abcd.wav"
    audio_file.touch()

    mock_review.return_value = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Garrick"}
    )

    result = runner.invoke(
        app,
        [
            "review",
            str(session_dir),
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "### [00:00:00 - 00:00:05] Garrick" in content
    assert mock_review.call_args[1]["audio_path"] == audio_file
    assert mock_review.call_args[1]["auto_play"] is True
    assert mock_review.call_args[1]["audio_padding"] == 2.0


def test_review_command_missing_turns(tmp_path: Path) -> None:
    """Test review subcommand exits with code 1 when turns cache is missing."""
    result = runner.invoke(app, ["review", str(tmp_path / "nonexistent")])
    assert result.exit_code == 1
    assert "Turns cache not found" in result.output


def test_split_command(tmp_path: Path) -> None:
    """Test split subcommand splits cluster and regenerates transcript."""
    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir(parents=True)
    out_file = tmp_path / "split.md"

    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=4.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="First part of discussion.",
        ),
        AlignedTurn(
            turn_id=1,
            start=10.0,
            end=15.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Second part by new speaker.",
        ),
    ]
    (session_dir / "turns.json").write_text(
        json.dumps([t.model_dump() for t in turns]), encoding="utf-8"
    )
    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
    )
    (session_dir / "speakers_mapping.json").write_text(
        json.dumps(mapping.model_dump()), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "split",
            str(session_dir),
            "SPEAKER_00",
            "--at",
            "8.0",
            "--to",
            "SPEAKER_01",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "Alice" in content
    assert "Bob" in content

    # Verify turns.json was updated on disk
    updated_data = json.loads((session_dir / "turns.json").read_text(encoding="utf-8"))
    assert updated_data[0]["cluster_id"] == "SPEAKER_00"
    assert updated_data[1]["cluster_id"] == "SPEAKER_01"


def test_split_command_missing_turns(tmp_path: Path) -> None:
    """Test split subcommand exits with code 1 when turns cache is missing."""
    result = runner.invoke(
        app,
        [
            "split",
            str(tmp_path / "nonexistent"),
            "SPEAKER_00",
            "--at",
            "10.0",
            "--to",
            "SPEAKER_01",
        ],
    )
    assert result.exit_code == 1
    assert "Turns cache not found" in result.output
