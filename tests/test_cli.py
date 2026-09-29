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
    assert "recluster" in result.output
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
            "--voice-profiles",
            str(tmp_path / "voice_profiles.json"),
            "--no-interactive",
            "--no-refine",
        ],
    )
    assert result.exit_code == 0
    mock_diarize.assert_called_once()
    assert out_file.is_file()


@patch("a2ts.cli.diarize_segments")
@patch("a2ts.cli.get_engine")
@patch("a2ts.cli.extract_audio_to_wav")
def test_run_with_diarizer_engine(
    mock_extract: MagicMock,
    mock_get_engine: MagicMock,
    mock_diarize: MagicMock,
    tmp_path: Path,
) -> None:
    """Test run passes --diarizer-engine to diarize_segments."""
    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"dummy")
    out_file = tmp_path / "out.md"
    cache_dir = tmp_path / ".a2ts"
    voice_profiles = tmp_path / "voice_profiles.json"

    mock_extract.return_value = tmp_path / "audio.wav"
    dummy_engine = MagicMock()
    dummy_engine.transcribe.return_value = [
        RawSegment(id=0, start=0.0, end=2.0, text="First"),
    ]
    mock_get_engine.return_value = dummy_engine

    mock_diarize.return_value = [
        SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00"),
    ]

    result = runner.invoke(
        app,
        [
            "run",
            str(media_file),
            "--output",
            str(out_file),
            "--cache-dir",
            str(cache_dir),
            "--diarizer-engine",
            "nemotron",
            "--voice-profiles",
            str(voice_profiles),
            "--no-interactive",
            "--no-refine",
        ],
    )
    assert result.exit_code == 0
    mock_diarize.assert_called_once()
    assert mock_diarize.call_args.kwargs.get("engine") == "nemotron"


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


def test_run_with_cluster_threshold_and_voice_profiles(tmp_path: Path) -> None:
    """Test run passes clustering options and invokes voice profile enrollment."""
    media_file = tmp_path / "session.mp3"
    media_file.touch()
    profiles_path = tmp_path / "voice_profiles.json"

    with (
        patch("a2ts.cli.compute_file_hash", return_value="abc1234"),
        patch("a2ts.cli.extract_audio_to_wav", return_value=tmp_path / "audio.wav"),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine") as mock_engine,
        patch("a2ts.cli.diarize_segments") as mock_diarize,
        patch("a2ts.cli.load_voice_profiles", return_value=None),
        patch(
            "a2ts.cli.run_interactive_review",
            return_value=SpeakersMapping(cluster_defaults={"SPEAKER_00": "Brakk"}),
        ),
        patch("a2ts.cli.compute_voice_profiles") as mock_compute_vp,
        patch("a2ts.cli.save_voice_profiles") as mock_save_vp,
    ):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            RawSegment(id=0, start=0.0, end=2.0, text="Hello Brakk", words=[])
        ]
        mock_engine.return_value = mock_transcriber
        mock_diarize.return_value = [
            SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")
        ]

        result = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--output",
                str(tmp_path / "out.md"),
                "--cache-dir",
                str(tmp_path / ".a2ts"),
                "--cluster-threshold",
                "0.65",
                "--num-speakers",
                "4",
                "--voice-profiles",
                str(profiles_path),
                "--no-refine",
            ],
        )

        assert result.exit_code == 0
        # Verify diarize_segments called with threshold and num_speakers
        mock_diarize.assert_called_once()
        kwargs = mock_diarize.call_args.kwargs
        assert kwargs["distance_threshold"] == 0.65
        assert kwargs["num_speakers"] == 4

        # Verify voice profile computation was invoked and saved
        mock_compute_vp.assert_called_once()
        mock_save_vp.assert_called_once()


def test_run_voice_profile_pre_matching(tmp_path: Path) -> None:
    """Test run pre-matches clusters when voice profiles database and cached embeddings exist."""
    import numpy as np

    from a2ts.models import VoiceProfile, VoiceProfilesDatabase

    media_file = tmp_path / "session.mp3"
    media_file.touch()
    cache_dir = tmp_path / ".a2ts"
    diar_dir = cache_dir / "diarization"
    diar_dir.mkdir(parents=True)

    centroid = [0.0] * 192
    centroid[0] = 1.0
    profiles_db = VoiceProfilesDatabase(
        speakers={
            "Brakk": VoiceProfile(
                speaker_name="Brakk", centroid=centroid, sample_count=5
            )
        }
    )
    profiles_path = tmp_path / "voice_profiles.json"
    profiles_path.write_text(profiles_db.model_dump_json(), encoding="utf-8")

    emb = np.zeros((1, 192), dtype=np.float32)
    emb[0, 0] = 1.0
    np.save(diar_dir / "abc1234_embeddings.npy", emb)
    (diar_dir / "abc1234_indices.json").write_text("[0]", encoding="utf-8")

    captured_mapping: list[SpeakersMapping] = []

    def capture_review(
        turns: list[AlignedTurn],
        candidates: list[str],
        existing_mapping: SpeakersMapping | None = None,
        **kwargs: object,
    ) -> SpeakersMapping:
        if existing_mapping is not None:
            captured_mapping.append(existing_mapping.model_copy())
        return existing_mapping or SpeakersMapping()

    with (
        patch("a2ts.cli.compute_file_hash", return_value="abc1234"),
        patch("a2ts.cli.extract_audio_to_wav", return_value=tmp_path / "audio.wav"),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine") as mock_engine,
        patch("a2ts.cli.diarize_segments") as mock_diarize,
        patch("a2ts.cli.run_interactive_review", side_effect=capture_review),
    ):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            RawSegment(id=0, start=0.0, end=2.0, text="Brakk speaking", words=[])
        ]
        mock_engine.return_value = mock_transcriber
        mock_diarize.return_value = [
            SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")
        ]

        result = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--output",
                str(tmp_path / "out.md"),
                "--cache-dir",
                str(cache_dir),
                "--voice-profiles",
                str(profiles_path),
                "--no-refine",
            ],
        )

        assert result.exit_code == 0
        assert len(captured_mapping) == 1
        assert captured_mapping[0].cluster_defaults.get("SPEAKER_00") == "Brakk"


def test_review_with_voice_profiles(tmp_path: Path) -> None:
    """Test review command updates voice profiles when cached embeddings exist."""
    import numpy as np

    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir(parents=True)
    diar_dir = session_dir / "diarization"
    diar_dir.mkdir(parents=True)
    out_file = tmp_path / "reviewed.md"
    profiles_path = tmp_path / "voice_profiles.json"

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
        "media_hash": "hash5678",
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

    emb = np.zeros((1, 192), dtype=np.float32)
    emb[0, 0] = 1.0
    np.save(diar_dir / "hash5678_embeddings.npy", emb)
    (diar_dir / "hash5678_indices.json").write_text("[0]", encoding="utf-8")

    with (
        patch(
            "a2ts.cli.run_interactive_review",
            return_value=SpeakersMapping(cluster_defaults={"SPEAKER_00": "Brakk"}),
        ),
        patch("a2ts.cli.compute_voice_profiles") as mock_compute_vp,
        patch("a2ts.cli.save_voice_profiles") as mock_save_vp,
    ):
        result = runner.invoke(
            app,
            [
                "review",
                str(session_dir),
                "--output",
                str(out_file),
                "--voice-profiles",
                str(profiles_path),
            ],
        )
        assert result.exit_code == 0
        mock_compute_vp.assert_called_once()
        mock_save_vp.assert_called_once()


def test_recluster_command(tmp_path: Path) -> None:
    """Test recluster command re-clusters cached embeddings and generates transcript."""
    import numpy as np

    from a2ts.models import RawSegment, SessionMetadata, SpeakersMapping

    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir()
    diar_dir = session_dir / "diarization"
    diar_dir.mkdir()

    # Create dummy cached session metadata
    meta = SessionMetadata(
        media_path=str(tmp_path / "media.mp3"),
        media_hash="hash123",
        duration_seconds=10.0,
        engine="whisper",
        model_name="large-v3",
        prompt_hash="",
        time_slice_minutes=15.0,
        created_at="2026-09-22T00:00:00Z",
        output_path=str(tmp_path / "recluster_out.md"),
    )
    (session_dir / "session.json").write_text(meta.model_dump_json(), encoding="utf-8")

    # Create dummy raw segments cache
    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir()
    raw_segs = [
        RawSegment(id=0, start=0.0, end=1.0, text="Turn 1", words=[]),
        RawSegment(id=1, start=1.5, end=2.5, text="Turn 2", words=[]),
    ]
    (transcripts_dir / "hash123_whisper.json").write_text(
        json.dumps([s.model_dump() for s in raw_segs]), encoding="utf-8"
    )

    # Create dummy embeddings cache
    np.save(
        diar_dir / "hash123_embeddings.npy",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    (diar_dir / "hash123_indices.json").write_text("[0, 1]", encoding="utf-8")

    # Create speakers mapping with custom name
    mapping = SpeakersMapping(cluster_defaults={"SPEAKER_00": "Valeros"})
    (session_dir / "speakers_mapping.json").write_text(
        mapping.model_dump_json(), encoding="utf-8"
    )

    # Run recluster command
    result = runner.invoke(
        app,
        [
            "recluster",
            str(session_dir),
            "--cluster-threshold",
            "0.70",
            "--output",
            str(tmp_path / "recluster_out.md"),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "recluster_out.md").is_file()
    assert (session_dir / "turns.json").is_file()
    content = (tmp_path / "recluster_out.md").read_text(encoding="utf-8")
    assert "Valeros" in content
    assert "✓ Re-clustered into" in result.output


def test_recluster_command_missing_session(tmp_path: Path) -> None:
    """Test recluster command exits with code 1 if session.json missing."""
    result = runner.invoke(app, ["recluster", str(tmp_path / "empty_dir")])
    assert result.exit_code == 1
    assert "Session metadata not found" in result.output


def test_recluster_command_missing_transcripts(tmp_path: Path) -> None:
    """Test recluster command exits with code 1 if transcripts missing."""
    from a2ts.models import SessionMetadata

    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir()
    meta = SessionMetadata(
        media_path=str(tmp_path / "media.mp3"),
        media_hash="hash999",
        duration_seconds=10.0,
        engine="whisper",
        model_name="large-v3",
        prompt_hash="",
        time_slice_minutes=15.0,
        created_at="2026-09-22T00:00:00Z",
        output_path=str(tmp_path / "out.md"),
    )
    (session_dir / "session.json").write_text(meta.model_dump_json(), encoding="utf-8")

    result = runner.invoke(app, ["recluster", str(session_dir)])
    assert result.exit_code == 1
    assert "Transcripts cache not found" in result.output


def test_recluster_command_missing_embeddings(tmp_path: Path) -> None:
    """Test recluster command exits with code 1 if embeddings missing."""
    from a2ts.models import RawSegment, SessionMetadata

    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir()
    meta = SessionMetadata(
        media_path=str(tmp_path / "media.mp3"),
        media_hash="hash999",
        duration_seconds=10.0,
        engine="whisper",
        model_name="large-v3",
        prompt_hash="",
        time_slice_minutes=15.0,
        created_at="2026-09-22T00:00:00Z",
        output_path=str(tmp_path / "out.md"),
    )
    (session_dir / "session.json").write_text(meta.model_dump_json(), encoding="utf-8")

    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir()
    raw_segs = [RawSegment(id=0, start=0.0, end=1.0, text="Turn 1", words=[])]
    (transcripts_dir / "hash999_whisper.json").write_text(
        json.dumps([s.model_dump() for s in raw_segs]), encoding="utf-8"
    )

    result = runner.invoke(app, ["recluster", str(session_dir)])
    assert result.exit_code == 1
    assert "Embeddings cache not found" in result.output


def test_review_with_filter_options(tmp_path: Path) -> None:
    """Test review command passes filter options to run_interactive_review."""
    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir(parents=True)
    out_file = tmp_path / "filtered_review.md"

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

    with patch(
        "a2ts.cli.run_interactive_review", return_value=SpeakersMapping()
    ) as mock_review:
        result = runner.invoke(
            app,
            [
                "review",
                str(session_dir),
                "--output",
                str(out_file),
                "--cluster",
                "SPEAKER_00,SPEAKER_02",
                "--unassigned-only",
                "--min-turns",
                "3",
                "--min-duration",
                "10.0",
                "--time-slice",
                "1",
            ],
        )
        assert result.exit_code == 0
        mock_review.assert_called_once()
        kwargs = mock_review.call_args.kwargs
        assert kwargs["filter_clusters"] == ["SPEAKER_00", "SPEAKER_02"]
        assert kwargs["unassigned_only"] is True
        assert kwargs["min_turns"] == 3
        assert kwargs["min_duration"] == 10.0
        assert kwargs["filter_slice"] == 1


def test_recluster_with_closed_set_and_voice_profiles(tmp_path: Path) -> None:
    """Test recluster in closed-set mode assigns all clusters and propagates labels."""
    import numpy as np

    from a2ts.models import (
        RawSegment,
        SessionMetadata,
        VoiceProfile,
        VoiceProfilesDatabase,
    )

    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir()
    diar_dir = session_dir / "diarization"
    diar_dir.mkdir()

    meta = SessionMetadata(
        media_path=str(tmp_path / "media.mp3"),
        media_hash="hash_closed",
        duration_seconds=10.0,
        engine="whisper",
        model_name="large-v3",
        prompt_hash="",
        time_slice_minutes=15.0,
        created_at="2026-09-22T00:00:00Z",
        output_path=str(tmp_path / "recluster_closed.md"),
    )
    (session_dir / "session.json").write_text(meta.model_dump_json(), encoding="utf-8")

    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir()
    raw_segs = [
        RawSegment(id=0, start=0.0, end=1.0, text="Turn 1", words=[]),
        RawSegment(id=1, start=1.5, end=2.5, text="Turn 2", words=[]),
        RawSegment(id=2, start=2.5, end=3.0, text="Turn 3 (short blip)", words=[]),
    ]
    (transcripts_dir / "hash_closed_whisper.json").write_text(
        json.dumps([s.model_dump() for s in raw_segs]), encoding="utf-8"
    )

    # 2 embeddings for first two segments; segment 2 has no embedding (short blip)
    np.save(
        diar_dir / "hash_closed_embeddings.npy",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    (diar_dir / "hash_closed_indices.json").write_text("[0, 1]", encoding="utf-8")

    # Create voice profiles with 2D matching centroids
    vp_path = tmp_path / "profiles.json"
    vp_db = VoiceProfilesDatabase(
        speakers={
            "Brakk": VoiceProfile(
                speaker_name="Brakk", centroid=[1.0, 0.0], sample_count=5
            ),
            "Merrow": VoiceProfile(
                speaker_name="Merrow", centroid=[0.0, 1.0], sample_count=5
            ),
        }
    )
    vp_path.write_text(vp_db.model_dump_json(), encoding="utf-8")

    out_file = tmp_path / "recluster_closed.md"
    result = runner.invoke(
        app,
        [
            "recluster",
            str(session_dir),
            "--voice-profiles",
            str(vp_path),
            "--closed-set",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "SPEAKER_" not in content
    assert "Brakk" in content
    assert "Merrow" in content
