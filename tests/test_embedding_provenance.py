import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from typer.testing import CliRunner

from a2ts.cli import app
from a2ts.diarizer import (
    extract_embeddings,
    extract_embeddings_for_segments,
    extract_embeddings_for_turns,
)
from a2ts.models import (
    EmbeddingCacheProvenance,
    RawSegment,
    SessionMetadata,
    SpeakersMapping,
    SpeakerTurn,
)


def test_embedding_cache_provenance_model() -> None:
    """Verify EmbeddingCacheProvenance schema and default values."""
    prov = EmbeddingCacheProvenance(
        entity_kind="segment",
        media_hash="abc123hash",
        count=5,
    )
    assert prov.version == 1
    assert prov.entity_kind == "segment"
    assert prov.media_hash == "abc123hash"
    assert prov.embedding_model == "speechbrain/spkrec-ecapa-voxceleb"
    assert prov.count == 5
    assert isinstance(prov.created_at, str)
    # verify ISO format parseable
    dt = datetime.fromisoformat(prov.created_at)
    assert dt.tzinfo is not None


def test_segment_embeddings_writes_canonical_and_provenance_no_legacy(
    tmp_path: Path,
) -> None:
    """Verify segment embeddings write _segment_* files and no legacy alias files."""
    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="One"),
        RawSegment(id=1, start=1.5, end=2.5, text="Two"),
    ]
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    cache_prefix = tmp_path / "cache_hash"

    with (
        patch(
            "soundfile.read", return_value=(np.zeros(32000, dtype=np.float32), 16000)
        ),
        patch("a2ts.diarizer.get_embedding_model") as mock_model,
    ):
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.side_effect = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
        ]
        mock_model.return_value = mock_classifier

        emb_matrix, valid_indices = extract_embeddings_for_segments(
            audio_path,
            segments,
            device="cpu",
            cache_prefix=cache_prefix,
            media_hash="test_media_hash",
        )

        assert emb_matrix.shape == (2, 2)
        assert valid_indices == [0, 1]

        # Canonical files exist
        assert (tmp_path / "cache_hash_segment_embeddings.npy").is_file()
        assert (tmp_path / "cache_hash_segment_indices.json").is_file()
        prov_file = tmp_path / "cache_hash_segment_embeddings_provenance.json"
        assert prov_file.is_file()

        # Legacy alias files MUST NOT exist
        assert not (tmp_path / "cache_hash_embeddings.npy").exists()
        assert not (tmp_path / "cache_hash_indices.json").exists()

        # Validate provenance contents
        prov = EmbeddingCacheProvenance.model_validate_json(
            prov_file.read_text(encoding="utf-8")
        )
        assert prov.version == 1
        assert prov.entity_kind == "segment"
        assert prov.media_hash == "test_media_hash"
        assert prov.count == 2


def test_turn_embeddings_writes_canonical_and_provenance_no_legacy(
    tmp_path: Path,
) -> None:
    """Verify turn embeddings write _turn_* files and no legacy alias files."""
    turns = [
        SpeakerTurn(id=0, start=0.0, end=1.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=1, start=1.5, end=2.5, cluster_id="SPEAKER_01"),
    ]
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    cache_prefix = tmp_path / "turn_cache"

    with (
        patch(
            "soundfile.read", return_value=(np.zeros(32000, dtype=np.float32), 16000)
        ),
        patch("a2ts.diarizer.get_embedding_model") as mock_model,
    ):
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.side_effect = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
        ]
        mock_model.return_value = mock_classifier

        emb_matrix, valid_indices = extract_embeddings_for_turns(
            audio_path,
            turns,
            device="cpu",
            cache_prefix=cache_prefix,
            media_hash="turn_media_hash",
        )

        assert emb_matrix.shape == (2, 2)
        assert valid_indices == [0, 1]

        # Canonical files exist
        assert (tmp_path / "turn_cache_turn_embeddings.npy").is_file()
        assert (tmp_path / "turn_cache_turn_indices.json").is_file()
        prov_file = tmp_path / "turn_cache_turn_embeddings_provenance.json"
        assert prov_file.is_file()

        # Legacy alias files MUST NOT exist
        assert not (tmp_path / "turn_cache_embeddings.npy").exists()
        assert not (tmp_path / "turn_cache_indices.json").exists()

        # Validate provenance contents
        prov = EmbeddingCacheProvenance.model_validate_json(
            prov_file.read_text(encoding="utf-8")
        )
        assert prov.version == 1
        assert prov.entity_kind == "turn"
        assert prov.media_hash == "turn_media_hash"
        assert prov.count == 2


def test_legacy_alias_not_loaded_as_fallback(tmp_path: Path) -> None:
    """Verify that legacy <prefix>_embeddings.npy is NOT read as fallback."""
    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="One"),
    ]
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    cache_prefix = tmp_path / "legacy_test"

    # Create only legacy alias files
    np.save(
        tmp_path / "legacy_test_embeddings.npy",
        np.array([[0.5, 0.5]], dtype=np.float32),
    )
    (tmp_path / "legacy_test_indices.json").write_text("[0]", encoding="utf-8")

    with (
        patch(
            "soundfile.read", return_value=(np.zeros(32000, dtype=np.float32), 16000)
        ),
        patch("a2ts.diarizer.get_embedding_model") as mock_model,
    ):
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.return_value = torch.tensor([[1.0, 0.0]])
        mock_model.return_value = mock_classifier

        # extract_embeddings should NOT load legacy cache; it should re-extract
        emb_matrix, _valid_indices = extract_embeddings(
            audio_path, segments, device="cpu", cache_prefix=cache_prefix
        )

        mock_model.assert_called_once()
        assert emb_matrix.shape == (1, 2)
        # Verify canonical files are now written
        assert (tmp_path / "legacy_test_segment_embeddings.npy").is_file()


def test_force_invalidates_cache_and_reextracts(tmp_path: Path) -> None:
    """Verify force=True skips existing valid cache and re-extracts."""
    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="One"),
    ]
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    cache_prefix = tmp_path / "force_test"

    # Pre-populate canonical cache
    np.save(
        tmp_path / "force_test_segment_embeddings.npy",
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    (tmp_path / "force_test_segment_indices.json").write_text("[0]", encoding="utf-8")
    from a2ts.diarizer import compute_entity_fingerprint

    trans_hash = "force_trans_hash"
    prov = EmbeddingCacheProvenance(
        entity_kind="segment",
        media_hash="force_test",
        count=1,
        transcript_provenance_hash=trans_hash,
        entity_fingerprint=compute_entity_fingerprint(segments),
    )
    (tmp_path / "force_test_segment_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

    with (
        patch(
            "soundfile.read", return_value=(np.zeros(32000, dtype=np.float32), 16000)
        ),
        patch("a2ts.diarizer.get_embedding_model") as mock_model,
    ):
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.return_value = torch.tensor([[0.0, 1.0]])
        mock_model.return_value = mock_classifier

        # Without force: cache loaded, no model call
        emb_matrix, _valid_indices = extract_embeddings(
            audio_path,
            segments,
            device="cpu",
            cache_prefix=cache_prefix,
            force=False,
            transcript_provenance_hash=trans_hash,
        )
        mock_model.assert_not_called()
        assert np.allclose(emb_matrix, [[1.0, 0.0]])

        # With force=True: cache ignored, model called
        emb_matrix_forced, _valid_indices_forced = extract_embeddings(
            audio_path,
            segments,
            device="cpu",
            cache_prefix=cache_prefix,
            force=True,
            transcript_provenance_hash=trans_hash,
        )
        mock_model.assert_called_once()
        assert np.allclose(emb_matrix_forced, [[0.0, 1.0]])


def test_provenance_entity_kind_mismatch_invalidates_cache(tmp_path: Path) -> None:
    """Verify that a provenance entity_kind mismatch prevents loading cache."""
    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="One"),
    ]
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    cache_prefix = tmp_path / "mismatch_test"

    # Cache marked with wrong entity_kind "turn"
    np.save(
        tmp_path / "mismatch_test_segment_embeddings.npy",
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    (tmp_path / "mismatch_test_segment_indices.json").write_text(
        "[0]", encoding="utf-8"
    )
    prov = EmbeddingCacheProvenance(
        entity_kind="turn", media_hash="mismatch_test", count=1
    )
    (tmp_path / "mismatch_test_segment_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

    with (
        patch(
            "soundfile.read", return_value=(np.zeros(32000, dtype=np.float32), 16000)
        ),
        patch("a2ts.diarizer.get_embedding_model") as mock_model,
    ):
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.return_value = torch.tensor([[0.0, 1.0]])
        mock_model.return_value = mock_classifier

        # Provenance mismatch should cause re-extraction instead of loading bad cache
        _emb_matrix, _valid_indices = extract_embeddings_for_segments(
            audio_path, segments, device="cpu", cache_prefix=cache_prefix, force=False
        )
        mock_model.assert_called_once()


def test_cli_recluster_rejects_legacy_alias(tmp_path: Path) -> None:
    """Verify CLI recluster fails when only legacy alias <hash>_embeddings.npy exists."""
    runner = CliRunner()
    session_dir = tmp_path / "session_1"
    session_dir.mkdir(parents=True)
    media_hash = "fakehash123"

    session_meta = SessionMetadata(
        media_path="/tmp/fake.wav",
        media_hash=media_hash,
        duration_seconds=10.0,
        engine="whisper",
        model_name="base",
        prompt_hash="nohash",
        time_slice_minutes=15.0,
        created_at=datetime.now(UTC).isoformat(),
    )
    (session_dir / "session.json").write_text(
        session_meta.model_dump_json(), encoding="utf-8"
    )

    # Transcripts cache
    from a2ts.models import TranscriptCacheProvenance

    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir()
    seg = RawSegment(id=0, start=0.0, end=1.0, text="hello")
    trans_prov = TranscriptCacheProvenance(
        media_hash=media_hash,
        engine="whisper",
        model_name="base",
        prompt_hash="nohash",
        compute_type="float16",
    )
    (transcripts_dir / f"{media_hash}_whisper.json").write_text(
        json.dumps(
            {
                "provenance": trans_prov.model_dump(),
                "segments": [seg.model_dump()],
            }
        ),
        encoding="utf-8",
    )

    # Put only legacy alias files in diarization dir
    diar_dir = session_dir / "diarization"
    diar_dir.mkdir()
    np.save(
        diar_dir / f"{media_hash}_embeddings.npy",
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    (diar_dir / f"{media_hash}_indices.json").write_text("[0]", encoding="utf-8")

    # Recluster must reject legacy alias and exit with error
    result = runner.invoke(app, ["recluster", str(session_dir)])
    assert result.exit_code != 0
    assert "Embeddings cache not found" in result.output

    # Now add canonical segment files
    np.save(
        diar_dir / f"{media_hash}_segment_embeddings.npy",
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    (diar_dir / f"{media_hash}_segment_indices.json").write_text(
        "[0]", encoding="utf-8"
    )
    prov = EmbeddingCacheProvenance(
        entity_kind="segment",
        media_hash=media_hash,
        count=1,
        transcript_provenance_hash=trans_prov.compute_hash(),
    )
    (diar_dir / f"{media_hash}_segment_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

    result2 = runner.invoke(
        app, ["recluster", str(session_dir), "--output", str(tmp_path / "out.md")]
    )
    assert result2.exit_code == 0
    assert "Re-clustered" in result2.output


@patch("a2ts.cli.extract_audio_to_wav")
@patch("a2ts.cli.get_engine")
@patch("a2ts.cli.diarize_segments")
@patch("a2ts.cli.extract_embeddings_for_turns")
def test_cli_run_passes_force_to_embeddings(
    mock_extract_turns: MagicMock,
    mock_diarize: MagicMock,
    mock_get_engine: MagicMock,
    mock_extract_audio: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify CLI run --force passes force=True down to extract_embeddings_for_turns."""
    runner = CliRunner()
    media_file = tmp_path / "test.mp3"
    media_file.touch()
    audio_wav = tmp_path / "test.wav"
    audio_wav.touch()
    mock_extract_audio.return_value = audio_wav

    engine_inst = MagicMock()
    seg = RawSegment(id=0, start=0.0, end=1.0, text="hello")
    engine_inst.transcribe.return_value = [seg]
    mock_get_engine.return_value = engine_inst

    mock_diarize.return_value = [
        SpeakerTurn(id=0, start=0.0, end=1.0, cluster_id="SPEAKER_00")
    ]

    vp_file = tmp_path / "voice_profiles.json"
    vp_file.write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": {
                    "Alice": {
                        "speaker_name": "Alice",
                        "centroid": [1.0, 0.0],
                        "sample_count": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "run",
            str(media_file),
            "--force",
            "--cache-dir",
            str(tmp_path / ".a2ts"),
            "--voice-profiles",
            str(vp_file),
            "--output",
            str(tmp_path / "transcript.md"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 0
    # mock_extract_turns should have been called with force=True
    assert mock_extract_turns.called
    for call in mock_extract_turns.call_args_list:
        assert call.kwargs.get("force") is True


def test_embedding_cache_provenance_new_fields() -> None:
    """Verify EmbeddingCacheProvenance supports transcript/diarization hashes and entity fingerprint."""
    prov = EmbeddingCacheProvenance(
        entity_kind="segment",
        media_hash="hash123",
        count=3,
        transcript_provenance_hash="trans_hash_abc",
        diarization_provenance_hash="diar_hash_xyz",
        entity_fingerprint="fingerprint_123",
    )
    assert prov.transcript_provenance_hash == "trans_hash_abc"
    assert prov.diarization_provenance_hash == "diar_hash_xyz"
    assert prov.entity_fingerprint == "fingerprint_123"


def test_load_segment_embeddings_missing_or_corrupt_provenance(tmp_path: Path) -> None:
    """load_segment_embeddings returns None if provenance file is missing or corrupt."""
    from a2ts.diarizer import load_segment_embeddings

    prefix = tmp_path / "test_seg"
    np.save(
        f"{prefix}_segment_embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32)
    )
    Path(f"{prefix}_segment_indices.json").write_text("[0]", encoding="utf-8")

    # 1. Missing provenance file -> None
    res = load_segment_embeddings(prefix, media_hash="test_seg")
    assert res is None

    # 2. Corrupt JSON in provenance file -> None
    Path(f"{prefix}_segment_embeddings_provenance.json").write_text(
        "not json", encoding="utf-8"
    )
    res = load_segment_embeddings(prefix, media_hash="test_seg")
    assert res is None


def test_segment_embeddings_transcript_provenance_invalidation(tmp_path: Path) -> None:
    """load_segment_embeddings returns None when transcript_provenance_hash changes."""
    from a2ts.diarizer import load_segment_embeddings, save_segment_embeddings

    prefix = tmp_path / "test_seg"
    mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    indices = [0, 1]

    save_segment_embeddings(
        prefix,
        emb_matrix=mat,
        valid_indices=indices,
        media_hash="hash1",
        transcript_provenance_hash="prompt_v1_hash",
    )

    # Correct transcript hash loads successfully
    loaded = load_segment_embeddings(
        prefix,
        media_hash="hash1",
        expected_transcript_provenance_hash="prompt_v1_hash",
    )
    assert loaded is not None
    loaded_mat, loaded_indices = loaded
    assert np.allclose(loaded_mat, mat)
    assert loaded_indices == indices

    # Changed transcript hash returns None
    assert (
        load_segment_embeddings(
            prefix,
            media_hash="hash1",
            expected_transcript_provenance_hash="prompt_v2_hash",
        )
        is None
    )

    # Media hash mismatch returns None
    assert (
        load_segment_embeddings(
            prefix,
            media_hash="different_media",
            expected_transcript_provenance_hash="prompt_v1_hash",
        )
        is None
    )


def test_segment_embeddings_count_mismatch(tmp_path: Path) -> None:
    """load_segment_embeddings returns None if count or matrix/indices lengths mismatch."""
    from a2ts.diarizer import load_segment_embeddings, save_segment_embeddings

    prefix = tmp_path / "test_seg"
    mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    indices = [0, 1]

    save_segment_embeddings(
        prefix,
        emb_matrix=mat,
        valid_indices=indices,
        media_hash="hash1",
        transcript_provenance_hash="trans_v1",
    )

    # Expected count mismatch
    assert (
        load_segment_embeddings(
            prefix,
            media_hash="hash1",
            expected_transcript_provenance_hash="trans_v1",
            expected_count=3,
        )
        is None
    )

    # Corrupted file on disk: matrix has 1 row but provenance says count=2
    np.save(
        f"{prefix}_segment_embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32)
    )
    assert (
        load_segment_embeddings(
            prefix,
            media_hash="hash1",
            expected_transcript_provenance_hash="trans_v1",
        )
        is None
    )


def test_turn_embeddings_diarization_provenance_invalidation(tmp_path: Path) -> None:
    """load_turn_embeddings validates diarization_provenance_hash, count, and returns None on mismatch."""
    from a2ts.diarizer import load_turn_embeddings, save_turn_embeddings

    prefix = tmp_path / "test_turn"
    mat = np.array([[1.0, 0.0]], dtype=np.float32)
    indices = [0]

    save_turn_embeddings(
        prefix,
        emb_matrix=mat,
        valid_indices=indices,
        media_hash="media1",
        diarization_provenance_hash="diar_v1",
    )

    # Matching diarization hash loads
    loaded = load_turn_embeddings(
        prefix,
        media_hash="media1",
        expected_diarization_provenance_hash="diar_v1",
    )
    assert loaded is not None
    loaded_mat, loaded_indices = loaded
    assert np.allclose(loaded_mat, mat)
    assert loaded_indices == indices

    # Mismatched diarization hash returns None
    assert (
        load_turn_embeddings(
            prefix,
            media_hash="media1",
            expected_diarization_provenance_hash="diar_v2",
        )
        is None
    )

    # Missing provenance returns None
    Path(f"{prefix}_turn_embeddings_provenance.json").unlink()
    assert load_turn_embeddings(prefix, media_hash="media1") is None


def test_extract_embeddings_for_segments_invalidated_by_transcript_hash(
    tmp_path: Path,
) -> None:
    """extract_embeddings_for_segments re-extracts when transcript_provenance_hash changes."""
    segments = [RawSegment(id=0, start=0.0, end=1.0, text="One")]
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    cache_prefix = tmp_path / "prov_test"

    with (
        patch(
            "soundfile.read", return_value=(np.zeros(32000, dtype=np.float32), 16000)
        ),
        patch("a2ts.diarizer.get_embedding_model") as mock_model,
    ):
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.side_effect = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
        ]
        mock_model.return_value = mock_classifier

        # First extraction with prompt v1
        emb1, _ = extract_embeddings_for_segments(
            audio_path,
            segments,
            device="cpu",
            cache_prefix=cache_prefix,
            media_hash="m1",
            transcript_provenance_hash="prompt_v1",
        )
        assert mock_model.call_count == 1
        assert np.allclose(emb1, [[1.0, 0.0]])

        # Second call with same prompt hash should hit cache (no model call)
        emb2, _ = extract_embeddings_for_segments(
            audio_path,
            segments,
            device="cpu",
            cache_prefix=cache_prefix,
            media_hash="m1",
            transcript_provenance_hash="prompt_v1",
        )
        assert mock_model.call_count == 1
        assert np.allclose(emb2, [[1.0, 0.0]])

        # Third call with prompt v2 should miss cache and call model again
        emb3, _ = extract_embeddings_for_segments(
            audio_path,
            segments,
            device="cpu",
            cache_prefix=cache_prefix,
            media_hash="m1",
            transcript_provenance_hash="prompt_v2",
        )
        assert mock_model.call_count == 2
        assert np.allclose(emb3, [[0.0, 1.0]])


def test_cli_review_rejects_foreign_turn_embeddings_glob(tmp_path: Path) -> None:
    """Verify CLI review does not glob turn embeddings from another session."""
    runner = CliRunner()
    session_dir = tmp_path / "session_1"
    session_dir.mkdir(parents=True)
    media_hash = "session1_hash"

    session_meta = SessionMetadata(
        media_path="/tmp/fake.wav",
        media_hash=media_hash,
        duration_seconds=10.0,
        engine="whisper",
        model_name="base",
        prompt_hash="nohash",
        time_slice_minutes=15.0,
        created_at=datetime.now(UTC).isoformat(),
    )
    (session_dir / "session.json").write_text(
        session_meta.model_dump_json(), encoding="utf-8"
    )

    turns = [
        {
            "turn_id": 0,
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "cluster_id": "SPEAKER_00",
            "text": "hello",
        }
    ]
    (session_dir / "turns.json").write_text(json.dumps(turns), encoding="utf-8")

    diar_dir = session_dir / "diarization"
    diar_dir.mkdir()
    # Create foreign session turn embeddings
    foreign_hash = "foreign_session_hash"
    np.save(
        diar_dir / f"{foreign_hash}_turn_embeddings.npy",
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    (diar_dir / f"{foreign_hash}_turn_indices.json").write_text("[0]", encoding="utf-8")

    vp_file = tmp_path / "voice_profiles.json"
    vp_file.write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": {
                    "Alice": {
                        "speaker_name": "Alice",
                        "centroid": [1.0, 0.0],
                        "sample_count": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with (
        patch("a2ts.cli.classify_clusters_to_profiles") as mock_classify,
        patch("a2ts.cli.run_interactive_review", return_value=SpeakersMapping()),
    ):
        result = runner.invoke(
            app,
            [
                "review",
                str(session_dir),
                "--voice-profiles",
                str(vp_file),
                "--output",
                str(tmp_path / "out.md"),
            ],
        )
        assert result.exit_code == 0
        # classify_clusters_to_profiles MUST NOT be called with foreign embeddings
        mock_classify.assert_not_called()


def test_cli_recluster_rejects_foreign_segment_embeddings_glob(tmp_path: Path) -> None:
    """Verify CLI recluster does not glob segment embeddings from another session."""
    runner = CliRunner()
    session_dir = tmp_path / "session_1"
    session_dir.mkdir(parents=True)
    media_hash = "session1_hash"

    session_meta = SessionMetadata(
        media_path="/tmp/fake.wav",
        media_hash=media_hash,
        duration_seconds=10.0,
        engine="whisper",
        model_name="base",
        prompt_hash="nohash",
        time_slice_minutes=15.0,
        created_at=datetime.now(UTC).isoformat(),
    )
    (session_dir / "session.json").write_text(
        session_meta.model_dump_json(), encoding="utf-8"
    )

    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir()
    seg = RawSegment(id=0, start=0.0, end=1.0, text="hello")
    (transcripts_dir / f"{media_hash}_whisper.json").write_text(
        json.dumps([seg.model_dump()]), encoding="utf-8"
    )

    diar_dir = session_dir / "diarization"
    diar_dir.mkdir()
    # Create foreign session segment embeddings
    foreign_hash = "foreign_session_hash"
    np.save(
        diar_dir / f"{foreign_hash}_segment_embeddings.npy",
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    (diar_dir / f"{foreign_hash}_segment_indices.json").write_text(
        "[0]", encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "recluster",
            str(session_dir),
            "--output",
            str(tmp_path / "out.md"),
        ],
    )
    assert result.exit_code != 0
    # Must indicate embeddings cache not found (did not glob foreign file)
    assert "Embeddings cache not found" in result.output


def test_load_turn_embeddings_validates_entity_fingerprint(tmp_path: Path) -> None:
    """load_turn_embeddings validates expected_entity_fingerprint and turns argument."""
    from a2ts.diarizer import (
        compute_entity_fingerprint,
        load_turn_embeddings,
        save_turn_embeddings,
    )

    prefix = tmp_path / "test_fingerprint"
    mat = np.array([[1.0, 0.0]], dtype=np.float32)
    indices = [0]
    turns = [SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")]

    save_turn_embeddings(
        prefix,
        emb_matrix=mat,
        valid_indices=indices,
        media_hash="fp_media",
        diarization_provenance_hash="fp_diar_hash",
        turns=turns,
    )

    # Valid fingerprint loads
    fp = compute_entity_fingerprint(turns)
    loaded = load_turn_embeddings(
        prefix,
        media_hash="fp_media",
        expected_diarization_provenance_hash="fp_diar_hash",
        expected_entity_fingerprint=fp,
    )
    assert loaded is not None

    # Passing turns directly computes and validates fingerprint
    loaded_via_turns = load_turn_embeddings(
        prefix,
        media_hash="fp_media",
        expected_diarization_provenance_hash="fp_diar_hash",
        turns=turns,
    )
    assert loaded_via_turns is not None

    # Mismatched fingerprint returns None
    assert (
        load_turn_embeddings(
            prefix,
            media_hash="fp_media",
            expected_diarization_provenance_hash="fp_diar_hash",
            expected_entity_fingerprint="wrong_fingerprint",
        )
        is None
    )

    # Modified turns returns None
    modified_turns = [SpeakerTurn(id=0, start=0.0, end=3.0, cluster_id="SPEAKER_00")]
    assert (
        load_turn_embeddings(
            prefix,
            media_hash="fp_media",
            expected_diarization_provenance_hash="fp_diar_hash",
            turns=modified_turns,
        )
        is None
    )


def test_load_segment_embeddings_validates_entity_fingerprint(tmp_path: Path) -> None:
    """load_segment_embeddings validates expected_entity_fingerprint and segments argument."""
    from a2ts.diarizer import (
        compute_entity_fingerprint,
        load_segment_embeddings,
        save_segment_embeddings,
    )

    prefix = tmp_path / "test_seg_fp"
    mat = np.array([[1.0, 0.0]], dtype=np.float32)
    indices = [0]
    segs = [RawSegment(id=0, start=0.0, end=2.0, text="hello")]

    save_segment_embeddings(
        prefix,
        emb_matrix=mat,
        valid_indices=indices,
        media_hash="seg_fp_media",
        transcript_provenance_hash="fp_trans_hash",
        segments=segs,
    )

    fp = compute_entity_fingerprint(segs)
    loaded = load_segment_embeddings(
        prefix,
        media_hash="seg_fp_media",
        expected_transcript_provenance_hash="fp_trans_hash",
        expected_entity_fingerprint=fp,
    )
    assert loaded is not None

    # Passing segments directly
    loaded_via_segs = load_segment_embeddings(
        prefix,
        media_hash="seg_fp_media",
        expected_transcript_provenance_hash="fp_trans_hash",
        segments=segs,
    )
    assert loaded_via_segs is not None

    # Mismatched fingerprint returns None
    assert (
        load_segment_embeddings(
            prefix,
            media_hash="seg_fp_media",
            expected_transcript_provenance_hash="fp_trans_hash",
            expected_entity_fingerprint="wrong_fp",
        )
        is None
    )

    # Modified segment returns None
    modified_segs = [RawSegment(id=0, start=0.0, end=2.0, text="changed text")]
    assert (
        load_segment_embeddings(
            prefix,
            media_hash="seg_fp_media",
            expected_transcript_provenance_hash="fp_trans_hash",
            segments=modified_segs,
        )
        is None
    )


def test_save_embeddings_atomic_tmp_replace(tmp_path: Path) -> None:
    """save_segment_embeddings and save_turn_embeddings atomically replace sibling .tmp.npy."""
    from unittest.mock import patch

    from a2ts.diarizer import save_segment_embeddings, save_turn_embeddings

    prefix = tmp_path / "atomic_test"
    mat = np.array([[1.0, 0.0]], dtype=np.float32)

    original_replace = Path.replace
    replaced_sources: list[str] = []

    def mock_replace(self: Path, target: Path | str) -> Path:
        replaced_sources.append(str(self))
        return original_replace(self, target)

    with patch.object(Path, "replace", autospec=True, side_effect=mock_replace):
        save_segment_embeddings(
            prefix,
            emb_matrix=mat,
            valid_indices=[0],
            media_hash="atomic_m",
        )
        assert any(s.endswith(".tmp.npy") for s in replaced_sources)
        replaced_sources.clear()

        save_turn_embeddings(
            prefix,
            emb_matrix=mat,
            valid_indices=[0],
            media_hash="atomic_m",
        )
        assert any(s.endswith(".tmp.npy") for s in replaced_sources)


def test_turn_embeddings_with_short_turns_cache_hit(tmp_path: Path) -> None:
    """Turn embeddings with short turns (< 0.15s) load correctly without false count mismatch."""
    from a2ts.diarizer import load_turn_embeddings, save_turn_embeddings

    prefix = tmp_path / "short_turns_test"
    # 2 turns total: turn 0 is normal, turn 1 was short (<0.15s) so only turn 0 embedded
    turns = [
        SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=1, start=2.0, end=2.05, cluster_id="SPEAKER_01"),
    ]
    mat = np.array([[1.0, 0.0]], dtype=np.float32)
    valid_indices = [0]  # Only 1 valid index out of 2 turns

    save_turn_embeddings(
        prefix,
        emb_matrix=mat,
        valid_indices=valid_indices,
        media_hash="m_short",
        diarization_provenance_hash="diar_short_hash",
        turns=turns,
    )

    # When expected_count is NOT passed (as in run without expected_count=len(speaker_turns)),
    # load_turn_embeddings successfully loads despite prov.count (1) != len(turns) (2)
    loaded = load_turn_embeddings(
        prefix,
        media_hash="m_short",
        expected_diarization_provenance_hash="diar_short_hash",
    )
    assert loaded is not None
    loaded_mat, loaded_indices = loaded
    assert len(loaded_mat) == 1
    assert loaded_indices == [0]


def test_cli_run_invalidates_turn_embeddings_on_diarization_change(
    tmp_path: Path,
) -> None:
    """CLI run invalidates cached turn embeddings when diarization provenance changes."""
    from unittest.mock import MagicMock, patch

    from typer.testing import CliRunner

    from a2ts.cli import app
    from a2ts.models import (
        EmbeddingCacheProvenance,
        RawSegment,
        SpeakersMapping,
        SpeakerTurn,
        VoiceProfile,
        VoiceProfilesDatabase,
    )

    runner = CliRunner()
    media_file = tmp_path / "session.mp3"
    media_file.touch()
    cache_dir = tmp_path / ".a2ts"
    diar_dir = cache_dir / "diarization"
    diar_dir.mkdir(parents=True)

    centroid = [0.0] * 192
    centroid[0] = 1.0
    profiles_db = VoiceProfilesDatabase(
        speakers={
            "Alice": VoiceProfile(
                speaker_name="Alice", centroid=centroid, sample_count=5
            )
        }
    )
    profiles_path = tmp_path / "voice_profiles.json"
    profiles_path.write_text(profiles_db.model_dump_json(), encoding="utf-8")

    emb = np.zeros((1, 192), dtype=np.float32)
    emb[0, 0] = 1.0
    np.save(diar_dir / "hash123_turn_embeddings.npy", emb)
    (diar_dir / "hash123_turn_indices.json").write_text("[0]", encoding="utf-8")
    # Provenance with old/different diarization provenance hash
    prov = EmbeddingCacheProvenance(
        entity_kind="turn",
        media_hash="hash123",
        count=1,
        diarization_provenance_hash="stale_diarization_hash",
    )
    (diar_dir / "hash123_turn_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

    audio_wav = tmp_path / "audio.wav"
    audio_wav.touch()

    with (
        patch("a2ts.cli.compute_file_hash", return_value="hash123"),
        patch("a2ts.cli.extract_audio_to_wav", return_value=audio_wav),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.get_engine") as mock_engine,
        patch("a2ts.cli.diarize_segments") as mock_diarize,
        patch("a2ts.cli.extract_embeddings_for_turns") as mock_extract_turns,
        patch("a2ts.cli.run_interactive_review", return_value=SpeakersMapping()),
    ):
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            RawSegment(id=0, start=0.0, end=2.0, text="Alice speaking", words=[])
        ]
        mock_engine.return_value = mock_transcriber
        mock_diarize.return_value = [
            SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")
        ]
        mock_extract_turns.return_value = (emb, [0])

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
        # Because cached diarization_provenance_hash is stale, extract_embeddings_for_turns MUST be called
        mock_extract_turns.assert_called_once()


def test_load_turn_embeddings_none_or_empty_diarization_provenance_returns_none(
    tmp_path: Path,
) -> None:
    """load_turn_embeddings returns None if expected_diarization_provenance_hash is None or empty."""
    from a2ts.diarizer import load_turn_embeddings, save_turn_embeddings

    prefix = tmp_path / "test_none_diar_prov"
    mat = np.array([[1.0, 0.0]], dtype=np.float32)
    turns = [SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")]

    save_turn_embeddings(
        prefix,
        emb_matrix=mat,
        valid_indices=[0],
        media_hash="m1",
        diarization_provenance_hash="diar_hash_123",
        turns=turns,
    )

    # None hash -> cache miss
    assert (
        load_turn_embeddings(
            prefix,
            media_hash="m1",
            expected_diarization_provenance_hash=None,
            turns=turns,
        )
        is None
    )

    # Empty string hash -> cache miss
    assert (
        load_turn_embeddings(
            prefix,
            media_hash="m1",
            expected_diarization_provenance_hash="",
            turns=turns,
        )
        is None
    )


def test_load_segment_embeddings_none_or_empty_transcript_provenance_returns_none(
    tmp_path: Path,
) -> None:
    """load_segment_embeddings returns None if expected_transcript_provenance_hash is None or empty."""
    from a2ts.diarizer import load_segment_embeddings, save_segment_embeddings

    prefix = tmp_path / "test_none_trans_prov"
    mat = np.array([[1.0, 0.0]], dtype=np.float32)
    segs = [RawSegment(id=0, start=0.0, end=2.0, text="hello")]

    save_segment_embeddings(
        prefix,
        emb_matrix=mat,
        valid_indices=[0],
        media_hash="m1",
        transcript_provenance_hash="trans_hash_123",
        segments=segs,
    )

    # None hash -> cache miss
    assert (
        load_segment_embeddings(
            prefix,
            media_hash="m1",
            expected_transcript_provenance_hash=None,
            segments=segs,
        )
        is None
    )

    # Empty string hash -> cache miss
    assert (
        load_segment_embeddings(
            prefix,
            media_hash="m1",
            expected_transcript_provenance_hash="",
            segments=segs,
        )
        is None
    )


def test_load_segment_embeddings_shifted_timestamps_returns_none(
    tmp_path: Path,
) -> None:
    """Shifting timestamps by 0.5s while preserving media_hash and provenance hash causes load_segment_embeddings to return None."""
    from a2ts.diarizer import load_segment_embeddings, save_segment_embeddings

    prefix = tmp_path / "test_shift_seg"
    mat = np.array([[1.0, 0.0]], dtype=np.float32)
    segs = [RawSegment(id=0, start=1.0, end=2.0, text="hello")]

    save_segment_embeddings(
        prefix,
        emb_matrix=mat,
        valid_indices=[0],
        media_hash="shift_media",
        transcript_provenance_hash="trans_valid",
        segments=segs,
    )

    # Matching segments loads successfully
    loaded = load_segment_embeddings(
        prefix,
        media_hash="shift_media",
        expected_transcript_provenance_hash="trans_valid",
        segments=segs,
    )
    assert loaded is not None

    # Shifted timestamps by 0.5s causes cache miss
    shifted_segs = [RawSegment(id=0, start=1.5, end=2.5, text="hello")]
    loaded_shifted = load_segment_embeddings(
        prefix,
        media_hash="shift_media",
        expected_transcript_provenance_hash="trans_valid",
        segments=shifted_segs,
    )
    assert loaded_shifted is None
