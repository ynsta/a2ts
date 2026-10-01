"""Basic tests for a2ts CLI."""

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from typer.testing import CliRunner

from a2ts.cli import app, extract_vocab, resolve_device_and_compute_type
from a2ts.models import (
    AlignedTurn,
    EmbeddingCacheProvenance,
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
    assert "craig" in result.output


def test_extract_vocab_command(tmp_path: Path) -> None:
    """Test extract-vocab command with a context directory."""
    lore_file = tmp_path / "lore.md"
    lore_file.write_text("Rencontre avec [[Valeros]] et [[Seoni]].", encoding="utf-8")

    result = runner.invoke(app, ["extract-vocab", "--context-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Extracted 2 entity mentions" in result.output
    assert "Valeros" in result.output
    assert "Seoni" in result.output


def test_extract_vocab_default_max_tokens() -> None:
    """extract_vocab command parameter max_tokens must default to 180."""
    sig = inspect.signature(extract_vocab)
    param = sig.parameters["max_tokens"]
    # typer Annotated default or default value
    assert getattr(param.default, "default", param.default) == 180


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

    # Verify session.json scoped under sessions/<hash>
    sessions = list((cache_dir / "sessions").glob("*/session.json"))
    assert len(sessions) == 1
    session_file = sessions[0]
    assert session_file.is_file()
    session_data = json.loads(session_file.read_text(encoding="utf-8"))
    assert session_data["engine"] == "whisper"
    assert session_data["model_name"] == "custom-whisper-v3"
    assert session_data["duration_seconds"] == 42.5
    assert session_data["output_path"] == str(out_file)

    # Verify turns.json has SPEAKER_00 fallback
    turns_file = session_file.parent / "turns.json"
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
                "--interactive",
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

    from a2ts.cache import compute_transcript_provenance_hash
    from a2ts.models import (
        DiarizationCacheProvenance,
        TranscriptCacheProvenance,
        VoiceProfile,
        VoiceProfilesDatabase,
    )

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
    np.save(diar_dir / "abc1234_turn_embeddings.npy", emb)
    (diar_dir / "abc1234_turn_indices.json").write_text("[0]", encoding="utf-8")
    trans_prov = TranscriptCacheProvenance(
        media_hash="abc1234",
        engine="whisper",
        model_name="large-v3",
        prompt_hash="no_prompt",
        compute_type="float16",
    )
    import torch

    is_cuda = torch.cuda.is_available()
    resolved_engine = "nemotron" if is_cuda else "ecapa"
    thash = compute_transcript_provenance_hash(trans_prov)
    diar_prov = DiarizationCacheProvenance(
        media_hash="abc1234",
        engine="auto",
        resolved_engine=resolved_engine,
        cluster_threshold=0.60,
        num_speakers=None,
        device="cuda",
        transcript_provenance_hash=thash,
    )
    from a2ts.diarizer import compute_entity_fingerprint

    spk_turns = [SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")]
    prov = EmbeddingCacheProvenance(
        entity_kind="turn",
        media_hash="abc1234",
        count=1,
        diarization_provenance_hash=diar_prov.compute_hash(),
        entity_fingerprint=compute_entity_fingerprint(spk_turns),
    )
    (diar_dir / "abc1234_turn_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

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
                "--interactive",
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

    from a2ts.diarizer import compute_entity_fingerprint
    from a2ts.models import DiarizationCacheFile, DiarizationCacheProvenance

    diar_prov = DiarizationCacheProvenance(
        media_hash="hash5678",
        engine="ecapa",
        cluster_threshold=0.6,
        device="cpu",
    )
    diar_cache = DiarizationCacheFile(
        provenance=diar_prov,
        turns=[SpeakerTurn(id=0, start=0.0, end=5.0, cluster_id="SPEAKER_00")],
    )
    (diar_dir / "hash5678.json").write_text(
        diar_cache.model_dump_json(), encoding="utf-8"
    )

    emb = np.zeros((1, 192), dtype=np.float32)
    emb[0, 0] = 1.0
    np.save(diar_dir / "hash5678_turn_embeddings.npy", emb)
    (diar_dir / "hash5678_turn_indices.json").write_text("[0]", encoding="utf-8")
    prov = EmbeddingCacheProvenance(
        entity_kind="turn",
        media_hash="hash5678",
        count=1,
        diarization_provenance_hash=diar_prov.compute_hash(),
        entity_fingerprint=compute_entity_fingerprint(diar_cache.turns),
    )
    (diar_dir / "hash5678_turn_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

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


def test_run_to_review_voice_profiles_roundtrip(tmp_path: Path) -> None:
    """Test full roundtrip: run writes session & turn embeddings, review loads embeddings and updates voice profiles."""
    from a2ts.diarizer import save_diarization_cache, save_turn_embeddings
    from a2ts.models import DiarizationCacheProvenance, VoiceProfilesDatabase

    media_file = tmp_path / "meeting.mp3"
    media_file.write_bytes(b"dummy audio content")
    audio_wav = tmp_path / "audio.wav"
    audio_wav.write_bytes(b"dummy wav content")

    cache_dir = tmp_path / ".a2ts"
    profiles_path = tmp_path / "voice_profiles.json"
    out_file = tmp_path / "transcript.md"
    review_out_file = tmp_path / "reviewed.md"

    def fake_diarize(
        cache_path: Path | None = None,
        provenance: DiarizationCacheProvenance | None = None,
        **kwargs: Any,
    ) -> list[SpeakerTurn]:
        turns = [SpeakerTurn(id=0, start=0.0, end=5.0, cluster_id="SPEAKER_00")]
        if cache_path and provenance:
            save_diarization_cache(cache_path, provenance, turns)
        return turns

    def fake_extract_embeddings(
        cache_prefix: Path | str | None = None,
        media_hash: str = "",
        diarization_provenance_hash: str | None = None,
        turns: Sequence[SpeakerTurn | AlignedTurn] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, list[int]]:
        emb = np.zeros((1, 192), dtype=np.float32)
        emb[0, 0] = 1.0
        valid_indices = [0]
        if cache_prefix is not None:
            save_turn_embeddings(
                cache_prefix=cache_prefix,
                emb_matrix=emb,
                valid_indices=valid_indices,
                media_hash=media_hash,
                diarization_provenance_hash=diarization_provenance_hash,
                turns=turns,
            )
        return emb, valid_indices

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = [
        RawSegment(id=0, start=0.0, end=5.0, text="Hello Alice", words=[])
    ]

    with (
        patch("a2ts.cli.compute_file_hash", return_value="hash_roundtrip_123"),
        patch("a2ts.cli.extract_audio_to_wav", return_value=audio_wav),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine", return_value=mock_transcriber),
        patch("a2ts.cli.diarize_segments", side_effect=fake_diarize),
        patch(
            "a2ts.cli.extract_embeddings_for_turns",
            side_effect=fake_extract_embeddings,
        ),
        patch(
            "a2ts.cli.run_interactive_review",
            return_value=SpeakersMapping(
                cluster_defaults={"SPEAKER_00": "Speaker 1"},
                label_sources={"SPEAKER_00": "manual"},
            ),
        ),
    ):
        run_res = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--output",
                str(out_file),
                "--cache-dir",
                str(cache_dir),
                "--voice-profiles",
                str(tmp_path / "run_profiles.json"),
                "--interactive",
                "--no-refine",
            ],
        )
        assert run_res.exit_code == 0, run_res.output

    session_dir = cache_dir / "sessions" / "hash_roundtrip_123"
    assert (session_dir / "session.json").is_file()
    assert (session_dir / "turns.json").is_file()
    diar_cache_file = cache_dir / "diarization" / "hash_roundtrip_123.json"
    assert diar_cache_file.is_file()
    prov_file = (
        cache_dir / "diarization" / "hash_roundtrip_123_turn_embeddings_provenance.json"
    )
    assert prov_file.is_file()
    assert (
        cache_dir / "diarization" / "hash_roundtrip_123_turn_embeddings.npy"
    ).is_file()

    with patch(
        "a2ts.cli.run_interactive_review",
        return_value=SpeakersMapping(
            cluster_defaults={"SPEAKER_00": "Alice"},
            label_sources={"SPEAKER_00": "manual"},
        ),
    ):
        review_res = runner.invoke(
            app,
            [
                "review",
                str(session_dir),
                "--output",
                str(review_out_file),
                "--voice-profiles",
                str(profiles_path),
            ],
        )
        assert review_res.exit_code == 0, review_res.output
        assert "Turn embeddings not found" not in review_res.output
        assert profiles_path.is_file()
        vp_db = VoiceProfilesDatabase.model_validate_json(
            profiles_path.read_text(encoding="utf-8")
        )
        assert "Alice" in vp_db.speakers
        assert vp_db.speakers["Alice"].sample_count >= 1


def test_run_ecapa_fallback_hash_synchronization(tmp_path: Path) -> None:
    """Test run synchronizes diarization provenance hash when diarize_segments falls back to ecapa."""
    from a2ts.diarizer import save_diarization_cache, save_turn_embeddings
    from a2ts.models import (
        DiarizationCacheFile,
        DiarizationCacheProvenance,
        EmbeddingCacheProvenance,
    )

    media_file = tmp_path / "meeting.mp3"
    media_file.write_bytes(b"dummy audio content")
    audio_wav = tmp_path / "audio.wav"
    audio_wav.write_bytes(b"dummy wav content")

    cache_dir = tmp_path / ".a2ts"
    out_file = tmp_path / "transcript.md"

    def fake_diarize_fallback(
        cache_path: Path | None = None,
        provenance: DiarizationCacheProvenance | None = None,
        **kwargs: Any,
    ) -> list[SpeakerTurn]:
        turns = [SpeakerTurn(id=0, start=0.0, end=5.0, cluster_id="SPEAKER_00")]
        if cache_path and provenance:
            fallback_prov = provenance.model_copy(update={"resolved_engine": "ecapa"})
            save_diarization_cache(cache_path, fallback_prov, turns)
        return turns

    def fake_extract_embeddings(
        cache_prefix: Path | str | None = None,
        media_hash: str = "",
        diarization_provenance_hash: str | None = None,
        turns: Sequence[SpeakerTurn | AlignedTurn] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, list[int]]:
        emb = np.zeros((1, 192), dtype=np.float32)
        emb[0, 0] = 1.0
        valid_indices = [0]
        if cache_prefix is not None:
            save_turn_embeddings(
                cache_prefix=cache_prefix,
                emb_matrix=emb,
                valid_indices=valid_indices,
                media_hash=media_hash,
                diarization_provenance_hash=diarization_provenance_hash,
                turns=turns,
            )
        return emb, valid_indices

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = [
        RawSegment(id=0, start=0.0, end=5.0, text="Hello Alice", words=[])
    ]

    with (
        patch("a2ts.cli.compute_file_hash", return_value="hash_fallback_456"),
        patch("a2ts.cli.extract_audio_to_wav", return_value=audio_wav),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine", return_value=mock_transcriber),
        patch("torch.cuda.is_available", return_value=True),
        patch("a2ts.cli.diarize_segments", side_effect=fake_diarize_fallback),
        patch(
            "a2ts.cli.extract_embeddings_for_turns",
            side_effect=fake_extract_embeddings,
        ),
        patch(
            "a2ts.cli.run_interactive_review",
            return_value=SpeakersMapping(
                cluster_defaults={"SPEAKER_00": "Speaker 1"},
                label_sources={"SPEAKER_00": "manual"},
            ),
        ),
    ):
        run_res = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--output",
                str(out_file),
                "--cache-dir",
                str(cache_dir),
                "--voice-profiles",
                str(tmp_path / "run_profiles.json"),
                "--device",
                "cuda",
                "--diarizer-engine",
                "auto",
                "--interactive",
                "--no-refine",
            ],
        )
        assert run_res.exit_code == 0, run_res.output

    diar_cache_path = cache_dir / "diarization" / "hash_fallback_456.json"
    diar_cache = DiarizationCacheFile.model_validate_json(
        diar_cache_path.read_text(encoding="utf-8")
    )
    assert diar_cache.provenance.resolved_engine == "ecapa"
    expected_hash = diar_cache.provenance.compute_hash()

    prov_file = (
        cache_dir / "diarization" / "hash_fallback_456_turn_embeddings_provenance.json"
    )
    turn_prov = EmbeddingCacheProvenance.model_validate_json(
        prov_file.read_text(encoding="utf-8")
    )
    assert turn_prov.diarization_provenance_hash == expected_hash


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
    from a2ts.models import TranscriptCacheFile, TranscriptCacheProvenance

    trans_prov = TranscriptCacheProvenance(
        media_hash="hash123",
        engine="whisper",
        model_name="large-v3",
        prompt_hash="no_prompt",
        compute_type="float16",
    )
    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir()
    raw_segs = [
        RawSegment(id=0, start=0.0, end=1.0, text="Turn 1", words=[]),
        RawSegment(id=1, start=1.5, end=2.5, text="Turn 2", words=[]),
    ]
    (transcripts_dir / "hash123_whisper.json").write_text(
        TranscriptCacheFile(provenance=trans_prov, segments=raw_segs).model_dump_json(),
        encoding="utf-8",
    )

    # Create dummy embeddings cache
    np.save(
        diar_dir / "hash123_segment_embeddings.npy",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    (diar_dir / "hash123_segment_indices.json").write_text("[0, 1]", encoding="utf-8")
    prov = EmbeddingCacheProvenance(
        entity_kind="segment",
        media_hash="hash123",
        count=2,
        transcript_provenance_hash=trans_prov.compute_hash(),
    )
    (diar_dir / "hash123_segment_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

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
    assert "✓ Re-clustered into" in result.output

    # Check diarization cache file has provenance and turns
    diar_cache = diar_dir / "hash123.json"
    assert diar_cache.is_file()
    from a2ts.models import DiarizationCacheFile

    cache_file = DiarizationCacheFile.model_validate_json(
        diar_cache.read_text(encoding="utf-8")
    )
    assert cache_file.provenance.engine == "ecapa"
    assert cache_file.provenance.resolved_engine == "ecapa"
    assert cache_file.provenance.cluster_threshold == 0.70
    assert len(cache_file.turns) == 2


def test_recluster_resets_stale_cluster_defaults(tmp_path: Path) -> None:
    """Test that a2ts recluster resets stale cluster_defaults in speakers_mapping.json."""
    import numpy as np

    from a2ts.models import RawSegment, SessionMetadata, SpeakersMapping

    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir()
    diar_dir = session_dir / "diarization"
    diar_dir.mkdir()

    meta = SessionMetadata(
        media_path=str(tmp_path / "media.mp3"),
        media_hash="hash_stale",
        duration_seconds=10.0,
        engine="whisper",
        model_name="large-v3",
        prompt_hash="",
        time_slice_minutes=15.0,
        created_at="2026-09-22T00:00:00Z",
        output_path=str(tmp_path / "recluster_out.md"),
    )
    (session_dir / "session.json").write_text(meta.model_dump_json(), encoding="utf-8")

    from a2ts.models import TranscriptCacheFile, TranscriptCacheProvenance

    trans_prov = TranscriptCacheProvenance(
        media_hash="hash_stale",
        engine="whisper",
        model_name="large-v3",
        prompt_hash="no_prompt",
        compute_type="float16",
    )
    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir()
    raw_segs = [
        RawSegment(id=0, start=0.0, end=1.0, text="Turn 1", words=[]),
        RawSegment(id=1, start=1.5, end=2.5, text="Turn 2", words=[]),
    ]
    (transcripts_dir / "hash_stale_whisper.json").write_text(
        TranscriptCacheFile(provenance=trans_prov, segments=raw_segs).model_dump_json(),
        encoding="utf-8",
    )

    np.save(
        diar_dir / "hash_stale_segment_embeddings.npy",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    (diar_dir / "hash_stale_segment_indices.json").write_text(
        "[0, 1]", encoding="utf-8"
    )
    prov = EmbeddingCacheProvenance(
        entity_kind="segment",
        media_hash="hash_stale",
        count=2,
        transcript_provenance_hash=trans_prov.compute_hash(),
    )
    (diar_dir / "hash_stale_segment_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

    # Create mapping with stale cluster_defaults
    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Valeros", "SPEAKER_01": "Merisiel"},
        label_sources={"SPEAKER_00": "manual", "SPEAKER_01": "manual"},
    )
    (session_dir / "speakers_mapping.json").write_text(
        mapping.model_dump_json(), encoding="utf-8"
    )

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

    # Ensure stale cluster_defaults and label_sources were reset
    updated_mapping = SpeakersMapping.model_validate_json(
        (session_dir / "speakers_mapping.json").read_text(encoding="utf-8")
    )
    assert "SPEAKER_00" not in updated_mapping.cluster_defaults
    assert "SPEAKER_01" not in updated_mapping.cluster_defaults
    assert len(updated_mapping.cluster_defaults) == 0
    assert len(updated_mapping.label_sources) == 0

    content = (tmp_path / "recluster_out.md").read_text(encoding="utf-8")
    assert "Valeros" not in content
    assert "Merisiel" not in content


def test_recluster_fails_cleanly_on_nemotron_session(tmp_path: Path) -> None:
    """Test that a2ts recluster fails cleanly with an informative error when session used Nemotron."""
    from a2ts.cache import save_diarization_cache
    from a2ts.models import DiarizationCacheProvenance, RawSegment, SessionMetadata

    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir()
    diar_dir = session_dir / "diarization"
    diar_dir.mkdir()

    meta = SessionMetadata(
        media_path=str(tmp_path / "media.mp3"),
        media_hash="hash_nemo",
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
    (transcripts_dir / "hash_nemo_whisper.json").write_text(
        json.dumps([s.model_dump() for s in raw_segs]), encoding="utf-8"
    )

    # Save diarization cache indicating Nemotron was used
    prov = DiarizationCacheProvenance(
        media_hash="hash_nemo",
        engine="nemotron",
        resolved_engine="nemotron",
        cluster_threshold=0.60,
    )
    save_diarization_cache(diar_dir / "hash_nemo.json", prov, [])

    result = runner.invoke(app, ["recluster", str(session_dir)])
    assert result.exit_code == 1
    assert (
        "Reclustering requires segment embeddings from ECAPA diarization; sessions diarized with Nemotron cannot be reclustered"
        in " ".join(result.output.split())
    )


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

    from a2ts.models import TranscriptCacheFile, TranscriptCacheProvenance

    trans_prov = TranscriptCacheProvenance(
        media_hash="hash_closed",
        engine="whisper",
        model_name="large-v3",
        prompt_hash="no_prompt",
        compute_type="float16",
    )
    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir()
    raw_segs = [
        RawSegment(id=0, start=0.0, end=1.0, text="Turn 1", words=[]),
        RawSegment(id=1, start=1.5, end=2.5, text="Turn 2", words=[]),
        RawSegment(id=2, start=2.5, end=3.0, text="Turn 3 (short blip)", words=[]),
    ]
    (transcripts_dir / "hash_closed_whisper.json").write_text(
        TranscriptCacheFile(provenance=trans_prov, segments=raw_segs).model_dump_json(),
        encoding="utf-8",
    )

    # 2 embeddings for first two segments; segment 2 has no embedding (short blip)
    np.save(
        diar_dir / "hash_closed_segment_embeddings.npy",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    (diar_dir / "hash_closed_segment_indices.json").write_text(
        "[0, 1]", encoding="utf-8"
    )
    prov = EmbeddingCacheProvenance(
        entity_kind="segment",
        media_hash="hash_closed",
        count=2,
        transcript_provenance_hash=trans_prov.compute_hash(),
    )
    (diar_dir / "hash_closed_segment_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

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


def test_craig_command_e2e_mocked(tmp_path: Path) -> None:
    """Test craig subcommand runs track discovery, transcription, debouncing, and saves transcript."""
    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()

    # Create dummy flac files
    (rec_dir / "1-merrow1.flac").write_bytes(b"flac data 1")
    (rec_dir / "2-tessaro.flac").write_bytes(b"flac data 2")

    # Create speakers.md
    (rec_dir / "speakers.md").write_text(
        "* tessaro: Le MJ, Maître du Jeu\n"
        "* merrow1: Merrow Ashdale, barde humain, surnoms: (Mimi)\n",
        encoding="utf-8",
    )

    # Create info.txt
    (rec_dir / "info.txt").write_text(
        "Guild: Adventure Club\nChannel: Session 1\nTracks:\n1-merrow1: merrow1\n2-tessaro: tessaro\n",
        encoding="utf-8",
    )

    # Mock Whisper engine
    mock_engine = MagicMock()

    def mock_transcribe(audio_path: str, **kwargs: object) -> list[RawSegment]:
        path = Path(audio_path)
        if "1-merrow1" in path.name:
            return [
                RawSegment(
                    id=0,
                    start=1.0,
                    end=3.0,
                    text="Je lance un dé 20 pour mon action.",
                    words=[
                        WordTimestamp(word="Je", start=1.0, end=1.5),
                        WordTimestamp(word="lance", start=1.5, end=2.0),
                        WordTimestamp(word="un", start=2.0, end=2.3),
                        WordTimestamp(word="dé", start=2.3, end=2.6),
                        WordTimestamp(word="20", start=2.6, end=3.0),
                    ],
                ),
                RawSegment(
                    id=1,
                    start=3.5,
                    end=5.0,
                    text="C'est un jet de délai pour la perception.",
                    words=[
                        WordTimestamp(word="C'est", start=3.5, end=4.0),
                        WordTimestamp(word="un", start=4.0, end=4.2),
                        WordTimestamp(word="jet", start=4.2, end=4.5),
                        WordTimestamp(word="de", start=4.5, end=4.7),
                        WordTimestamp(word="délai", start=4.7, end=5.0),
                    ],
                ),
            ]
        elif "2-tessaro" in path.name:
            return [
                RawSegment(
                    id=0,
                    start=6.0,
                    end=8.0,
                    text="Tu aperçois une silhouette dans l'ombre.",
                    words=[
                        WordTimestamp(word="Tu", start=6.0, end=6.5),
                        WordTimestamp(word="aperçois", start=6.5, end=8.0),
                    ],
                )
            ]
        return []

    mock_engine.transcribe.side_effect = mock_transcribe

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("a2ts.cli.get_engine", return_value=mock_engine) as mock_get_engine,
    ):
        result = runner.invoke(app, ["craig", str(rec_dir)])

    assert result.exit_code == 0
    mock_get_engine.assert_called_once_with(
        "whisper",
        model_name="large-v3",
        device="cuda",
        compute_type="float16",
    )

    out_transcript = rec_dir / "transcript.md"
    assert out_transcript.is_file()
    content = out_transcript.read_text(encoding="utf-8")

    # Verify speaker names and roles
    assert "Merrow Ashdale (merrow1)" in content
    assert "MJ (tessaro)" in content

    # Verify RPG term normalization
    assert "1d20" in content
    assert "jet de dés" in content

    # Verify debouncing: merrow1's two consecutive segments merged into a single turn [00:00:01 - 00:00:05]
    assert "### [00:00:01 - 00:00:05] Merrow Ashdale (merrow1)" in content
    assert (
        "Je lance 1d20 pour mon action. C'est un jet de dés pour la perception."
        in content
    )

    # Verify cache files were created in .transcripts
    cache_dir = rec_dir / ".transcripts"
    assert (cache_dir / "1-merrow1.json").is_file()
    assert (cache_dir / "2-tessaro.json").is_file()


def test_craig_command_missing_dir(tmp_path: Path) -> None:
    """Test craig subcommand fails with code 1 if directory does not exist."""
    missing_dir = tmp_path / "nonexistent"
    result = runner.invoke(app, ["craig", str(missing_dir)])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_craig_command_no_tracks(tmp_path: Path) -> None:
    """Test craig subcommand fails with code 1 if no Craig flac tracks found."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    result = runner.invoke(app, ["craig", str(empty_dir)])
    assert result.exit_code == 1
    assert "no craig" in result.output.lower() or "no tracks" in result.output.lower()


def test_craig_command_custom_output_and_speakers(tmp_path: Path) -> None:
    """Test craig subcommand with --output and --speakers-file options."""
    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()
    (rec_dir / "1-bob.flac").write_bytes(b"bob audio")

    custom_spk = tmp_path / "custom_speakers.md"
    custom_spk.write_text("* bob: Bob the Builder\n", encoding="utf-8")

    custom_out = tmp_path / "custom_out.md"

    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = [
        RawSegment(id=0, start=0.0, end=2.0, text="Can we fix it?", words=[])
    ]

    with patch("a2ts.cli.get_engine", return_value=mock_engine):
        result = runner.invoke(
            app,
            [
                "craig",
                str(rec_dir),
                "--speakers-file",
                str(custom_spk),
                "--output",
                str(custom_out),
            ],
        )

    assert result.exit_code == 0
    assert custom_out.is_file()
    content = custom_out.read_text(encoding="utf-8")
    assert "Bob the Builder (bob)" in content


def test_craig_command_with_refine(tmp_path: Path) -> None:
    """Test craig subcommand with --refine flag invokes refiner."""
    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()
    (rec_dir / "1-alice.flac").write_bytes(b"alice audio")

    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = [
        RawSegment(id=0, start=0.0, end=2.0, text="Hello world", words=[])
    ]

    with (
        patch("a2ts.cli.get_engine", return_value=mock_engine),
        patch(
            "a2ts.cli.refine_transcript_markdown",
            return_value="### [00:00:00 - 00:00:02] alice\nRefined hello world\n",
        ) as mock_refine,
    ):
        result = runner.invoke(
            app,
            [
                "craig",
                str(rec_dir),
                "--refine",
                "--refine-model",
                "custom-llm",
            ],
        )

    assert result.exit_code == 0
    mock_refine.assert_called_once()
    assert mock_refine.call_args.kwargs.get("agy_model") == "custom-llm"
    out_file = rec_dir / "transcript.md"
    assert "Refined hello world" in out_file.read_text(encoding="utf-8")


def test_craig_command_caching_and_force(tmp_path: Path) -> None:
    """Test craig subcommand caches raw track transcriptions and respects --force."""
    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()
    (rec_dir / "1-track.flac").write_bytes(b"flac data")

    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = [
        RawSegment(id=0, start=0.0, end=2.0, text="Cached turn", words=[])
    ]

    with patch("a2ts.cli.get_engine", return_value=mock_engine):
        # 1. First run: cold cache -> transcribe called
        res1 = runner.invoke(app, ["craig", str(rec_dir)])
        assert res1.exit_code == 0
        assert mock_engine.transcribe.call_count == 1

        # 2. Second run: warm cache -> transcribe not called
        mock_engine.transcribe.reset_mock()
        res2 = runner.invoke(app, ["craig", str(rec_dir)])
        assert res2.exit_code == 0
        mock_engine.transcribe.assert_not_called()

        # 3. Third run with --force: cache bypassed -> transcribe called
        mock_engine.transcribe.reset_mock()
        res3 = runner.invoke(app, ["craig", str(rec_dir), "--force"])
        assert res3.exit_code == 0
        assert mock_engine.transcribe.call_count == 1


def test_resolve_device_and_compute_type_cpu() -> None:
    """Test device and compute type resolution on CPU."""
    # Explicit cpu device with None compute_type
    dev, ctype = resolve_device_and_compute_type(device="cpu", compute_type=None)
    assert dev == "cpu"
    assert ctype == "int8"

    # Auto device when CUDA is not available
    with patch("torch.cuda.is_available", return_value=False):
        dev, ctype = resolve_device_and_compute_type(device="auto", compute_type=None)
        assert dev == "cpu"
        assert ctype == "int8"


def test_resolve_device_and_compute_type_cuda() -> None:
    """Test device and compute type resolution on CUDA."""
    # Explicit cuda device with None compute_type
    dev, ctype = resolve_device_and_compute_type(device="cuda", compute_type=None)
    assert dev == "cuda"
    assert ctype == "float16"

    # Auto device when CUDA is available
    with patch("torch.cuda.is_available", return_value=True):
        dev, ctype = resolve_device_and_compute_type(device="auto", compute_type=None)
        assert dev == "cuda"
        assert ctype == "float16"


def test_resolve_device_and_compute_type_explicit_compute_type() -> None:
    """Test device resolution with user-specified compute type."""
    # Custom compute_type preserved on CPU
    dev, ctype = resolve_device_and_compute_type(device="cpu", compute_type="float32")
    assert dev == "cpu"
    assert ctype == "float32"

    # Custom compute_type preserved on CUDA
    dev, ctype = resolve_device_and_compute_type(device="cuda", compute_type="int8")
    assert dev == "cuda"
    assert ctype == "int8"

    # Custom compute_type preserved with auto
    with patch("torch.cuda.is_available", return_value=False):
        dev, ctype = resolve_device_and_compute_type(
            device="auto", compute_type="float32"
        )
        assert dev == "cpu"
        assert ctype == "float32"


def test_run_simulated_cpu_defaults(tmp_path: Path) -> None:
    """Test CLI run defaults to cpu and int8 on simulated CPU environment."""
    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"dummy video data for hash")
    out_file = tmp_path / "out.md"
    cache_dir = tmp_path / ".a2ts"

    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = [
        RawSegment(
            id=0,
            start=0.0,
            end=2.0,
            text="Simulated CPU transcript.",
            words=[
                WordTimestamp(word="Simulated", start=0.0, end=0.8),
                WordTimestamp(word="CPU", start=0.9, end=1.2),
                WordTimestamp(word="transcript.", start=1.3, end=2.0),
            ],
        )
    ]

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("a2ts.cli.extract_audio_to_wav", return_value=tmp_path / "audio.wav"),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine", return_value=mock_engine) as mock_get_engine,
    ):
        result = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--no-diarize",
                "--no-interactive",
                "--output",
                str(out_file),
                "--cache-dir",
                str(cache_dir),
            ],
        )
        assert result.exit_code == 0
        mock_get_engine.assert_called_once_with(
            "whisper",
            device="cpu",
            compute_type="int8",
        )


def test_run_voxtral_with_diarization_warning_disables_diarization(
    tmp_path: Path,
) -> None:
    """Voxtral engine with --diarize warns and disables diarization."""
    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"dummy video data for hash")
    out_file = tmp_path / "out.md"
    cache_dir = tmp_path / ".a2ts"

    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = [
        RawSegment(
            id=0,
            start=0.0,
            end=0.0,
            text="Voxtral untimed text.",
            words=[],
        )
    ]

    with (
        patch("a2ts.cli.extract_audio_to_wav", return_value=tmp_path / "audio.wav"),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine", return_value=mock_engine),
        patch("a2ts.cli.diarize_segments") as mock_diarize,
    ):
        result = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--engine",
                "voxtral",
                "--diarize",
                "--no-interactive",
                "--output",
                str(out_file),
                "--cache-dir",
                str(cache_dir),
            ],
        )
        assert result.exit_code == 0
        normalized_output = " ".join(result.output.split())
        assert (
            "Voxtral produces untimed segments without word timestamps and does not support acoustic diarization. Disabling diarization for Voxtral or run with --no-diarize."
            in normalized_output
        )
        mock_diarize.assert_not_called()


def test_cli_imports_create_transcriber_from_transcriber() -> None:
    """Ensure cli.py imports create_transcriber from a2ts.transcriber."""
    import a2ts.cli
    import a2ts.transcriber

    assert a2ts.cli.create_transcriber is a2ts.transcriber.create_transcriber


def test_info_command_diagnostics() -> None:
    """Test that info command prints diagnostics with version, device, and engines."""
    from a2ts import __version__

    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert "Python" in result.output
    assert "PyTorch" in result.output
    assert "CUDA" in result.output
    assert "Faster-Whisper" in result.output
    assert "Voxtral" in result.output
    assert "SpeechBrain ECAPA" in result.output
    assert "Nemotron" in result.output


def test_run_non_tty_defaults_to_non_interactive(tmp_path: Path) -> None:
    """Test run on non-TTY defaults to non-interactive mode unless specified."""
    media_file = tmp_path / "video.mp4"
    media_file.touch()
    out_file = tmp_path / "out.md"
    cache_dir = tmp_path / ".a2ts"

    with (
        patch("sys.stdin.isatty", return_value=False),
        patch("a2ts.cli.extract_audio_to_wav", return_value=tmp_path / "audio.wav"),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine") as mock_engine,
        patch("a2ts.cli.run_interactive_review") as mock_review,
    ):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            RawSegment(id=0, start=0.0, end=2.0, text="Hello", words=[])
        ]
        mock_engine.return_value = mock_transcriber
        mock_review.return_value = SpeakersMapping()

        # 1. Default (interactive=None, isatty=False) -> should NOT invoke review
        result = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--no-diarize",
                "--output",
                str(out_file),
                "--cache-dir",
                str(cache_dir),
            ],
        )
        assert result.exit_code == 0
        mock_review.assert_not_called()

        # 2. Explicit --interactive -> should invoke review even on non-TTY
        result2 = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--no-diarize",
                "--interactive",
                "--output",
                str(out_file),
                "--cache-dir",
                str(cache_dir),
            ],
        )
        assert result2.exit_code == 0
        mock_review.assert_called_once()


def test_session_metadata_stores_rpg_normalize_and_prompt_hash(tmp_path: Path) -> None:
    """Test that SessionMetadata stores rpg_normalize and deterministic prompt_hash."""
    from a2ts.cache import compute_prompt_hash

    media_file = tmp_path / "video.mp4"
    media_file.touch()
    out_file = tmp_path / "out.md"
    cache_dir = tmp_path / ".a2ts"

    with (
        patch("a2ts.cli.extract_audio_to_wav", return_value=tmp_path / "audio.wav"),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine") as mock_engine,
    ):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            RawSegment(id=0, start=0.0, end=2.0, text="Hello", words=[])
        ]
        mock_engine.return_value = mock_transcriber

        result = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--no-diarize",
                "--no-interactive",
                "--no-rpg-normalize",
                "--output",
                str(out_file),
                "--cache-dir",
                str(cache_dir),
            ],
        )
        assert result.exit_code == 0

        session_files = list((cache_dir / "sessions").glob("*/session.json"))
        assert len(session_files) == 1
        session_data = json.loads(session_files[0].read_text(encoding="utf-8"))
        assert session_data.get("rpg_normalize") is False
        assert session_data.get("prompt_hash") == compute_prompt_hash(None)


def test_review_defaults_output_to_session_output_path(tmp_path: Path) -> None:
    """Test review subcommand defaults --output to session's output_path."""
    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir(parents=True)
    custom_out = tmp_path / "custom_session_output.md"

    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Hello from review.",
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
        "media_hash": "hash123",
        "duration_seconds": 10.0,
        "engine": "whisper",
        "model_name": "large-v3",
        "prompt_hash": "no_prompt",
        "time_slice_minutes": 15.0,
        "created_at": "2026-10-01T00:00:00Z",
        "output_path": str(custom_out),
        "rpg_normalize": True,
    }
    (session_dir / "session.json").write_text(
        json.dumps(session_meta), encoding="utf-8"
    )

    with patch(
        "a2ts.cli.run_interactive_review",
        return_value=SpeakersMapping(cluster_defaults={"SPEAKER_00": "Alice"}),
    ):
        result = runner.invoke(app, ["review", str(session_dir)])
        assert result.exit_code == 0
        assert custom_out.is_file()
        assert "Alice" in custom_out.read_text(encoding="utf-8")


def test_review_and_split_respect_session_rpg_normalize(tmp_path: Path) -> None:
    """Test review and split respect rpg_normalize=False from session.json."""
    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir(parents=True)
    out_file = tmp_path / "out.md"

    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Je lance un dé 20 et fais 1 d 6.",
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
        "media_hash": "hash123",
        "duration_seconds": 10.0,
        "engine": "whisper",
        "model_name": "large-v3",
        "prompt_hash": "no_prompt",
        "time_slice_minutes": 15.0,
        "created_at": "2026-10-01T00:00:00Z",
        "output_path": str(out_file),
        "rpg_normalize": False,
    }
    (session_dir / "session.json").write_text(
        json.dumps(session_meta), encoding="utf-8"
    )

    with patch(
        "a2ts.cli.run_interactive_review",
        return_value=SpeakersMapping(),
    ):
        result = runner.invoke(app, ["review", str(session_dir)])
        assert result.exit_code == 0
        content = out_file.read_text(encoding="utf-8")
        assert "dé 20" in content

    # Now test split respecting rpg_normalize=False
    split_out = tmp_path / "split_out.md"
    result_split = runner.invoke(
        app,
        [
            "split",
            str(session_dir),
            "SPEAKER_00",
            "--at",
            "2.0",
            "--to",
            "SPEAKER_01",
            "--output",
            str(split_out),
        ],
    )
    assert result_split.exit_code == 0
    content_split = split_out.read_text(encoding="utf-8")
    assert "dé 20" in content_split


def test_cli_run_auto_enroll_rejects_stale_turn_embeddings_with_no_profile_db(
    tmp_path: Path,
) -> None:
    """Run auto-enrollment rejects stale turn embeddings cache when no profile DB pre-exists."""
    import numpy as np

    from a2ts.models import VoiceProfilesDatabase

    media_file = tmp_path / "session.mp3"
    media_file.write_bytes(b"dummy audio content")
    audio_wav = tmp_path / "audio.wav"
    audio_wav.touch()

    cache_dir = tmp_path / ".a2ts"
    file_hash = "stale_run_hash_456"
    diar_dir = cache_dir / "diarization"
    diar_dir.mkdir(parents=True)

    # 1. Setup: No profile DB exists prior to run
    profiles_path = tmp_path / "voice_profiles.json"
    assert not profiles_path.exists()

    # 2. Create stale turn cache with vector [0.0, 0.0, 0.0, 1.0] and STALE_HASH
    stale_emb = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    np.save(diar_dir / f"{file_hash}_turn_embeddings.npy", stale_emb)
    (diar_dir / f"{file_hash}_turn_indices.json").write_text("[0]", encoding="utf-8")
    prov = EmbeddingCacheProvenance(
        entity_kind="turn",
        media_hash=file_hash,
        count=1,
        diarization_provenance_hash="STALE_HASH",
    )
    (diar_dir / f"{file_hash}_turn_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

    fresh_emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with (
        patch("a2ts.cli.compute_file_hash", return_value=file_hash),
        patch("a2ts.cli.extract_audio_to_wav", return_value=audio_wav),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine") as mock_engine,
        patch("a2ts.cli.diarize_segments") as mock_diarize,
        patch("a2ts.cli.extract_embeddings_for_turns") as mock_extract_turns,
        patch(
            "a2ts.cli.run_interactive_review",
            return_value=SpeakersMapping(
                cluster_defaults={"SPEAKER_00": "Alice"},
                label_sources={"SPEAKER_00": "manual"},
            ),
        ),
    ):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            RawSegment(id=0, start=0.0, end=2.0, text="Alice speaking", words=[])
        ]
        mock_engine.return_value = mock_transcriber
        mock_diarize.return_value = [
            SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")
        ]
        mock_extract_turns.return_value = (fresh_emb, [0])

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
                "--interactive",
            ],
        )

        assert result.exit_code == 0
        assert profiles_path.is_file()
        db = VoiceProfilesDatabase.model_validate_json(
            profiles_path.read_text(encoding="utf-8")
        )
        assert "Alice" in db.speakers
        # Must match fresh extraction [1.0, 0.0, 0.0, 0.0], NOT stale [0.0, 0.0, 0.0, 1.0]
        assert db.speakers["Alice"].centroid == [1.0, 0.0, 0.0, 0.0]


def test_recluster_cache_identity_and_run_hit(tmp_path: Path) -> None:
    """Test recluster sets exact transcript provenance hash, clears splits, and subsequent run hits diarization cache."""
    import numpy as np

    from a2ts.models import (
        ClusterSplit,
        DiarizationCacheFile,
        EmbeddingCacheProvenance,
        RawSegment,
        SessionMetadata,
        SpeakersMapping,
        TranscriptCacheFile,
        TranscriptCacheProvenance,
    )
    from a2ts.speaker_review import load_speakers_mapping

    session_dir = tmp_path / ".a2ts"
    session_dir.mkdir(parents=True)
    media_file = tmp_path / "audio.wav"
    media_file.write_bytes(b"RIFF dummy wav data")
    media_hash = "abc123testmediahash"

    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir(parents=True)
    trans_prov = TranscriptCacheProvenance(
        media_hash=media_hash,
        engine="whisper",
        model_name="large-v3",
        compute_type="int8",
        prompt_hash="no_prompt",
    )
    expected_trans_hash = trans_prov.compute_hash()
    raw_segs = [
        RawSegment(id=0, start=0.0, end=2.0, text="First utterance", words=[]),
        RawSegment(id=1, start=3.0, end=5.0, text="Second utterance", words=[]),
    ]
    trans_file = TranscriptCacheFile(provenance=trans_prov, segments=raw_segs)
    (transcripts_dir / f"{media_hash}_whisper.json").write_text(
        trans_file.model_dump_json(indent=2), encoding="utf-8"
    )

    diar_dir = session_dir / "diarization"
    diar_dir.mkdir(parents=True)
    emb_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(diar_dir / f"{media_hash}_segment_embeddings.npy", emb_matrix)
    (diar_dir / f"{media_hash}_segment_indices.json").write_text(
        "[0, 1]", encoding="utf-8"
    )
    seg_prov = EmbeddingCacheProvenance(
        entity_kind="segment",
        media_hash=media_hash,
        count=2,
        transcript_provenance_hash=expected_trans_hash,
    )
    (diar_dir / f"{media_hash}_segment_embeddings_provenance.json").write_text(
        seg_prov.model_dump_json(indent=2), encoding="utf-8"
    )

    session_meta = SessionMetadata(
        media_path=str(media_file),
        media_hash=media_hash,
        duration_seconds=10.0,
        engine="whisper",
        model_name="large-v3",
        prompt_hash="",
        time_slice_minutes=15.0,
        created_at="2026-10-01T00:00:00Z",
        output_path=str(tmp_path / "out.md"),
    )
    (session_dir / "session.json").write_text(
        session_meta.model_dump_json(indent=2), encoding="utf-8"
    )

    split_rule = ClusterSplit(
        cluster_id="SPEAKER_00", at=1.0, new_cluster_id="SPEAKER_SPLIT"
    )
    mapping = SpeakersMapping(
        splits=[split_rule], cluster_defaults={"SPEAKER_00": "Alice"}
    )
    mapping_path = session_dir / "speakers_mapping.json"
    mapping_path.write_text(mapping.model_dump_json(indent=2), encoding="utf-8")

    result_recluster = runner.invoke(
        app, ["recluster", str(session_dir), "--device", "cpu"]
    )
    assert result_recluster.exit_code == 0, (
        f"Recluster failed: {result_recluster.stdout}"
    )

    # 1. Verify diarization cache provenance transcript_provenance_hash equals transcript cache provenance hash (NOT "no_prompt")
    diar_cache_file = diar_dir / f"{media_hash}.json"
    assert diar_cache_file.is_file()
    diar_cache = DiarizationCacheFile.model_validate_json(
        diar_cache_file.read_text(encoding="utf-8")
    )
    assert diar_cache.provenance.transcript_provenance_hash == expected_trans_hash
    assert diar_cache.provenance.transcript_provenance_hash != "no_prompt"

    # 2. Verify mapping.splits is cleared
    saved_mapping = load_speakers_mapping(mapping_path)
    assert saved_mapping.splits == []

    # 3. Run a2ts run on the session; verify it hits diarization cache without re-extracting or re-diarizing
    with (
        patch("a2ts.cli.compute_file_hash", return_value=media_hash),
        patch("a2ts.cli.extract_audio_to_wav", return_value=media_file),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.diarizer.extract_embeddings_for_segments") as mock_extract_segs,
        patch("a2ts.diarizer.diarize_nemotron") as mock_nemotron,
    ):
        result_run = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--cache-dir",
                str(session_dir),
                "--output",
                str(tmp_path / "out.md"),
                "--device",
                "cpu",
                "--compute-type",
                "int8",
                "--diarizer-engine",
                "ecapa",
                "--no-interactive",
                "--no-refine",
            ],
        )

    assert result_run.exit_code == 0, f"Run failed: {result_run.stdout}"
    mock_extract_segs.assert_not_called()
    mock_nemotron.assert_not_called()
    assert "Loading cached diarization" in result_run.output


@patch(
    "a2ts.cli.extract_audio_to_wav",
    side_effect=RuntimeError("ffmpeg missing or corrupt media"),
)
def test_extract_audio_runtime_error_handled(
    mock_extract: MagicMock, tmp_path: Path
) -> None:
    """extract-audio cleanly handles RuntimeError from extract_audio_to_wav."""
    media_file = tmp_path / "broken.mp4"
    media_file.touch()

    result = runner.invoke(app, ["extract-audio", str(media_file)])
    assert result.exit_code == 1
    assert "ffmpeg missing or corrupt media" in result.output


@patch("a2ts.cli.compute_file_hash", return_value="hash_err")
@patch(
    "a2ts.cli.extract_audio_to_wav",
    side_effect=RuntimeError("ffmpeg missing or corrupt media"),
)
def test_run_audio_extraction_runtime_error_handled(
    mock_extract: MagicMock, mock_hash: MagicMock, tmp_path: Path
) -> None:
    """run command cleanly handles RuntimeError from extract_audio_to_wav."""
    media_file = tmp_path / "broken.mp4"
    media_file.touch()

    result = runner.invoke(
        app,
        [
            "run",
            str(media_file),
            "--cache-dir",
            str(tmp_path / ".a2ts"),
            "--output",
            str(tmp_path / "out.md"),
            "--no-refine",
        ],
    )
    assert result.exit_code == 1
    assert "ffmpeg missing or corrupt media" in result.output


def test_run_skips_transcript_cache_when_truncated(tmp_path: Path) -> None:
    """run command skips saving transcript cache when any segment is truncated."""
    media_file = tmp_path / "session.mp4"
    media_file.write_bytes(b"dummy audio")
    cache_dir = tmp_path / ".a2ts"

    with (
        patch("a2ts.cli.compute_file_hash", return_value="hash_trunc"),
        patch("a2ts.cli.extract_audio_to_wav", return_value=tmp_path / "audio.wav"),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.create_transcriber") as mock_create,
        patch("a2ts.cli.diarize_segments", return_value=[]),
    ):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            RawSegment(id=0, start=0.0, end=1.0, text="cut off text", is_truncated=True)
        ]
        mock_create.return_value = mock_transcriber

        result = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--engine",
                "voxtral",
                "--cache-dir",
                str(cache_dir),
                "--output",
                str(tmp_path / "out.md"),
                "--no-diarize",
                "--no-refine",
            ],
        )

        assert result.exit_code == 0
        assert (
            "Warning: Output was truncated; skipping persistent transcript cache"
            in result.output
        )
        transcripts_cache_files = list((cache_dir / "transcripts").glob("*.json"))
        assert len(transcripts_cache_files) == 0


def test_run_profile_match_escapes_bracketed_names(tmp_path: Path) -> None:
    """Profile match prints with bracketed speaker names do not raise MarkupError."""
    import numpy as np
    import torch

    from a2ts.cache import compute_transcript_provenance_hash
    from a2ts.diarizer import compute_entity_fingerprint
    from a2ts.models import (
        DiarizationCacheProvenance,
        TranscriptCacheProvenance,
        VoiceProfile,
        VoiceProfilesDatabase,
    )

    media_file = tmp_path / "session.mp3"
    media_file.touch()
    cache_dir = tmp_path / ".a2ts"
    diar_dir = cache_dir / "diarization"
    diar_dir.mkdir(parents=True)

    centroid = [0.0] * 192
    centroid[0] = 1.0
    profiles_db = VoiceProfilesDatabase(
        speakers={
            "[DM] Bob": VoiceProfile(
                speaker_name="[DM] Bob", centroid=centroid, sample_count=5
            )
        }
    )
    profiles_path = tmp_path / "voice_profiles.json"
    profiles_path.write_text(profiles_db.model_dump_json(), encoding="utf-8")

    emb = np.zeros((1, 192), dtype=np.float32)
    emb[0, 0] = 1.0
    np.save(diar_dir / "abc1234_turn_embeddings.npy", emb)
    (diar_dir / "abc1234_turn_indices.json").write_text("[0]", encoding="utf-8")

    trans_prov = TranscriptCacheProvenance(
        media_hash="abc1234",
        engine="whisper",
        model_name="large-v3",
        prompt_hash="no_prompt",
        compute_type="float16",
    )
    is_cuda = torch.cuda.is_available()
    resolved_engine = "nemotron" if is_cuda else "ecapa"
    thash = compute_transcript_provenance_hash(trans_prov)
    diar_prov = DiarizationCacheProvenance(
        media_hash="abc1234",
        engine="auto",
        resolved_engine=resolved_engine,
        cluster_threshold=0.60,
        num_speakers=None,
        device="cuda",
        transcript_provenance_hash=thash,
    )
    spk_turns = [SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")]
    prov = EmbeddingCacheProvenance(
        entity_kind="turn",
        media_hash="abc1234",
        count=1,
        diarization_provenance_hash=diar_prov.compute_hash(),
        entity_fingerprint=compute_entity_fingerprint(spk_turns),
    )
    (diar_dir / "abc1234_turn_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

    with (
        patch("a2ts.cli.compute_file_hash", return_value="abc1234"),
        patch("a2ts.cli.extract_audio_to_wav", return_value=tmp_path / "audio.wav"),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine") as mock_engine,
        patch("a2ts.cli.diarize_segments", return_value=spk_turns),
        patch(
            "a2ts.cli.run_interactive_review",
            side_effect=lambda turns, candidates, existing_mapping=None, **kw: (
                existing_mapping or SpeakersMapping()
            ),
        ),
    ):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            RawSegment(id=0, start=0.0, end=2.0, text="Speaking", words=[])
        ]
        mock_engine.return_value = mock_transcriber

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
                "--interactive",
                "--no-refine",
            ],
        )

        assert result.exit_code == 0
        assert "Matched voice profile for SPEAKER_00: [DM] Bob" in result.output
