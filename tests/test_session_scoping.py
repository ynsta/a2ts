import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from a2ts.cli import app, resolve_session_dir
from a2ts.models import RawSegment, SpeakersMapping, WordTimestamp
from a2ts.speaker_review import load_speakers_mapping

runner = CliRunner()


def test_session_state_scoped_by_hash(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".a2ts"
    hash1 = "1111111111111111111111111111111111111111111111111111111111111111"
    hash2 = "2222222222222222222222222222222222222222222222222222222222222222"

    sess1_dir = cache_dir / "sessions" / hash1
    sess2_dir = cache_dir / "sessions" / hash2
    sess1_dir.mkdir(parents=True, exist_ok=True)
    sess2_dir.mkdir(parents=True, exist_ok=True)

    # Session 1 has Alice
    mapping1 = SpeakersMapping(cluster_defaults={"SPEAKER_00": "Alice"})
    (sess1_dir / "speakers_mapping.json").write_text(json.dumps(mapping1.model_dump()))

    # Session 2 should not see Alice
    m2 = load_speakers_mapping(sess2_dir / "speakers_mapping.json")
    assert "SPEAKER_00" not in m2.cluster_defaults


@patch("a2ts.cli.extract_audio_to_wav")
@patch("a2ts.cli.probe_media")
@patch("a2ts.cli.get_engine")
def test_run_scopes_session_state_to_sessions_hash(
    mock_get_engine: MagicMock,
    mock_probe: MagicMock,
    mock_extract: MagicMock,
    tmp_path: Path,
) -> None:
    media1 = tmp_path / "media1.mp4"
    media1.write_bytes(b"media 1 dummy content")
    media2 = tmp_path / "media2.mp4"
    media2.write_bytes(b"media 2 dummy content")

    cache_dir = tmp_path / ".a2ts"
    out1 = tmp_path / "out1.md"
    out2 = tmp_path / "out2.md"

    mock_extract.return_value = tmp_path / "audio.wav"
    mock_probe.return_value = {"format": {"duration": "10.0"}}

    dummy_engine = MagicMock()
    dummy_engine.transcribe.return_value = [
        RawSegment(
            id=0,
            start=0.0,
            end=2.0,
            text="Hello world.",
            words=[
                WordTimestamp(word="Hello", start=0.0, end=1.0),
                WordTimestamp(word="world.", start=1.0, end=2.0),
            ],
        )
    ]
    mock_get_engine.return_value = dummy_engine

    # Run for media 1
    res1 = runner.invoke(
        app,
        [
            "run",
            str(media1),
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(out1),
            "--no-interactive",
            "--no-refine",
            "--no-diarize",
        ],
    )
    assert res1.exit_code == 0

    # Verify no session files at root of cache_dir
    assert not (cache_dir / "turns.json").exists()
    assert not (cache_dir / "speakers_mapping.json").exists()
    assert not (cache_dir / "session.json").exists()

    # Verify sessions directory has exactly 1 session
    sessions = list((cache_dir / "sessions").iterdir())
    assert len(sessions) == 1
    sess1_dir = sessions[0]
    assert (sess1_dir / "turns.json").is_file()
    assert (sess1_dir / "speakers_mapping.json").is_file()
    assert (sess1_dir / "session.json").is_file()

    # Modify speakers_mapping.json for session 1
    mapping1 = SpeakersMapping(cluster_defaults={"SPEAKER_00": "Speaker 1"})
    (sess1_dir / "speakers_mapping.json").write_text(json.dumps(mapping1.model_dump()))

    # Run for media 2
    res2 = runner.invoke(
        app,
        [
            "run",
            str(media2),
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(out2),
            "--no-interactive",
            "--no-refine",
            "--no-diarize",
        ],
    )
    assert res2.exit_code == 0

    # Verify sessions directory has 2 sessions
    sessions = sorted((cache_dir / "sessions").iterdir())
    assert len(sessions) == 2
    sess2_dir = next(s for s in sessions if s != sess1_dir)
    assert (sess2_dir / "turns.json").is_file()
    assert (sess2_dir / "speakers_mapping.json").is_file()

    # Ensure session 2 did not inherit session 1's mapping
    m2 = load_speakers_mapping(sess2_dir / "speakers_mapping.json")
    assert m2.cluster_defaults.get("SPEAKER_00") != "Speaker 1"


def test_resolve_session_dir_variations(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".a2ts"

    # 1. Non-existent or empty
    assert resolve_session_dir(cache_dir) == cache_dir

    # 2. Legacy root with turns.json
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "turns.json").write_text("[]")
    assert resolve_session_dir(cache_dir) == cache_dir
    (cache_dir / "turns.json").unlink()

    # 3. Exactly 1 session inside cache_dir / "sessions"
    sess1 = cache_dir / "sessions" / "hash1"
    sess1.mkdir(parents=True, exist_ok=True)
    (sess1 / "turns.json").write_text("[]")
    assert resolve_session_dir(cache_dir) == sess1

    # 4. Direct session dir passed
    assert resolve_session_dir(sess1) == sess1

    # 5. Direct .a2ts/<hash> without sessions/ in path resolves to sess1
    assert resolve_session_dir(cache_dir / "hash1") == sess1

    # 6. Multiple sessions inside cache_dir / "sessions"
    sess2 = cache_dir / "sessions" / "hash2"
    sess2.mkdir(parents=True, exist_ok=True)
    (sess2 / "turns.json").write_text("[]")
    with pytest.raises(typer.Exit):
        resolve_session_dir(cache_dir)
