"""Tests for diarization cache provenance and transcript dependency (P1-3)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from a2ts.cache import (
    compute_transcript_provenance_hash,
    load_diarization_cache,
    save_diarization_cache,
)
from a2ts.diarizer import diarize_segments
from a2ts.models import (
    DiarizationCacheFile,
    DiarizationCacheProvenance,
    RawSegment,
    SpeakerTurn,
    TranscriptCacheProvenance,
)


def test_diarization_provenance_fields() -> None:
    """Test DiarizationCacheProvenance supports resolved_engine and transcript_provenance_hash."""
    prov = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="auto",
        resolved_engine="ecapa",
        cluster_threshold=0.60,
        num_speakers=2,
        device="cuda",
        transcript_provenance_hash="tx_hash_abc",
    )
    assert prov.engine == "auto"
    assert prov.resolved_engine == "ecapa"
    assert prov.transcript_provenance_hash == "tx_hash_abc"

    # Default values
    prov_default = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="ecapa",
        cluster_threshold=0.60,
    )
    assert prov_default.resolved_engine is None
    assert prov_default.transcript_provenance_hash is None


def test_compute_transcript_provenance_hash_stability() -> None:
    """Test compute_transcript_provenance_hash is deterministic across created_at timestamps."""
    p1 = TranscriptCacheProvenance(
        media_hash="audio123",
        engine="whisper",
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="prompt_xyz",
    )
    p2 = TranscriptCacheProvenance(
        media_hash="audio123",
        engine="whisper",
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="prompt_xyz",
    )
    h1 = compute_transcript_provenance_hash(p1)
    h2 = compute_transcript_provenance_hash(p2)
    assert h1 == h2
    assert len(h1) == 64

    # Changing model changes hash
    p_diff_model = TranscriptCacheProvenance(
        media_hash="audio123",
        engine="whisper",
        model_name="medium",
        compute_type="float16",
        prompt_hash="prompt_xyz",
    )
    assert compute_transcript_provenance_hash(p_diff_model) != h1

    # Changing prompt changes hash
    p_diff_prompt = TranscriptCacheProvenance(
        media_hash="audio123",
        engine="whisper",
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="prompt_diff",
    )
    assert compute_transcript_provenance_hash(p_diff_prompt) != h1


def test_diarization_cache_invalidation_on_transcript_hash(tmp_path: Path) -> None:
    """Test that changing transcript_provenance_hash invalidates diarization cache."""
    cache_path = tmp_path / "diarization" / "test.json"
    prov1 = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="ecapa",
        resolved_engine="ecapa",
        cluster_threshold=0.60,
        num_speakers=2,
        device="cuda",
        transcript_provenance_hash="tx_hash_v1",
    )
    turns = [SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")]
    save_diarization_cache(cache_path, prov1, turns)

    # Same provenance hits
    assert load_diarization_cache(cache_path, prov1) is not None

    # Different transcript hash misses
    prov2 = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="ecapa",
        resolved_engine="ecapa",
        cluster_threshold=0.60,
        num_speakers=2,
        device="cuda",
        transcript_provenance_hash="tx_hash_v2",
    )
    assert load_diarization_cache(cache_path, prov2) is None

    # Missing transcript hash in cache when expected misses
    prov_no_tx = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="ecapa",
        resolved_engine="ecapa",
        cluster_threshold=0.60,
        num_speakers=2,
        device="cuda",
        transcript_provenance_hash=None,
    )
    cache_no_tx = tmp_path / "diarization" / "no_tx.json"
    save_diarization_cache(cache_no_tx, prov_no_tx, turns)
    assert load_diarization_cache(cache_no_tx, prov1) is None


def test_diarization_cache_invalidation_on_resolved_engine(tmp_path: Path) -> None:
    """Test that changing resolved_engine causes cache miss."""
    cache_path = tmp_path / "diarization" / "test.json"
    prov_nemotron = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="auto",
        resolved_engine="nemotron",
        cluster_threshold=0.60,
        num_speakers=2,
        device="cuda",
    )
    turns = [SpeakerTurn(id=0, start=0.0, end=2.0, cluster_id="SPEAKER_00")]
    save_diarization_cache(cache_path, prov_nemotron, turns)

    prov_ecapa = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="auto",
        resolved_engine="ecapa",
        cluster_threshold=0.60,
        num_speakers=2,
        device="cuda",
    )
    assert load_diarization_cache(cache_path, prov_ecapa) is None


def test_diarize_segments_single_segment_writes_envelope(tmp_path: Path) -> None:
    """Single segment path writes a DiarizationCacheFile envelope with provenance."""
    cache_file = tmp_path / "cache.json"
    segments = [RawSegment(id=0, start=1.0, end=4.0, text="Alone")]
    prov = DiarizationCacheProvenance(
        media_hash="media123",
        engine="ecapa",
        resolved_engine="ecapa",
        cluster_threshold=0.60,
        transcript_provenance_hash="tx_abc",
    )
    turns = diarize_segments(
        Path("dummy.wav"),
        segments,
        cache_path=cache_file,
        provenance=prov,
    )
    assert len(turns) == 1
    assert cache_file.is_file()

    # Verify disk format is DiarizationCacheFile envelope, not a bare list
    raw_content = json.loads(cache_file.read_text(encoding="utf-8"))
    assert isinstance(raw_content, dict)
    assert "provenance" in raw_content
    envelope = DiarizationCacheFile.model_validate(raw_content)
    assert envelope.provenance.resolved_engine == "ecapa"
    assert envelope.provenance.transcript_provenance_hash == "tx_abc"


@patch("a2ts.diarizer.extract_embeddings")
def test_diarize_segments_zero_valid_embeddings_writes_envelope(
    mock_extract: MagicMock, tmp_path: Path
) -> None:
    """Zero-valid-embeddings path writes a DiarizationCacheFile envelope with provenance."""
    mock_extract.return_value = (np.empty((0, 0), dtype=np.float32), [])
    cache_file = tmp_path / "cache.json"
    segments = [
        RawSegment(id=0, start=0.0, end=0.1, text="Hi"),
        RawSegment(id=1, start=0.2, end=0.3, text="There"),
    ]
    prov = DiarizationCacheProvenance(
        media_hash="media123",
        engine="ecapa",
        resolved_engine="ecapa",
        cluster_threshold=0.60,
        transcript_provenance_hash="tx_xyz",
    )
    turns = diarize_segments(
        Path("dummy.wav"),
        segments,
        cache_path=cache_file,
        provenance=prov,
    )
    assert len(turns) == 2
    assert cache_file.is_file()

    raw_content = json.loads(cache_file.read_text(encoding="utf-8"))
    assert isinstance(raw_content, dict)
    assert "provenance" in raw_content
    envelope = DiarizationCacheFile.model_validate(raw_content)
    assert envelope.provenance.resolved_engine == "ecapa"
    assert envelope.provenance.transcript_provenance_hash == "tx_xyz"


@patch("a2ts.diarizer.sf.read")
@patch("a2ts.diarizer.get_embedding_model")
def test_diarize_segments_transcript_hash_invalidates_cache(
    mock_get_model: MagicMock, mock_sf_read: MagicMock, tmp_path: Path
) -> None:
    """Test that diarize_segments re-runs clustering when transcript_provenance_hash changes."""
    dummy_audio = np.zeros(16000 * 5, dtype=np.float32)
    mock_sf_read.return_value = (dummy_audio, 16000)

    emb1 = np.array([1.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0], dtype=np.float32)
    mock_classifier = MagicMock()
    mock_classifier.encode_batch.side_effect = [
        MagicMock(squeeze=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: emb1))),
        MagicMock(squeeze=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: emb2))),
        MagicMock(squeeze=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: emb1))),
        MagicMock(squeeze=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: emb2))),
    ]
    mock_get_model.return_value = mock_classifier

    cache_file = tmp_path / "diar.json"
    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="Hello"),
        RawSegment(id=1, start=1.5, end=2.5, text="World"),
    ]
    prov1 = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="ecapa",
        resolved_engine="ecapa",
        cluster_threshold=0.60,
        device="cpu",
        transcript_provenance_hash="tx_hash_1",
    )

    # First run: writes cache with tx_hash_1
    turns1 = diarize_segments(
        Path("test.wav"),
        segments,
        device="cpu",
        engine="ecapa",
        cache_path=cache_file,
        provenance=prov1,
    )
    assert len(turns1) == 2
    assert mock_get_model.call_count == 1

    # Second run with same provenance: cache hit, no model load
    turns2 = diarize_segments(
        Path("test.wav"),
        segments,
        device="cpu",
        engine="ecapa",
        cache_path=cache_file,
        provenance=prov1,
    )
    assert len(turns2) == 2
    assert mock_get_model.call_count == 1  # Still 1, did not re-run

    # Third run with new transcript hash: cache miss, re-runs model
    prov2 = DiarizationCacheProvenance(
        media_hash="hash123",
        engine="ecapa",
        resolved_engine="ecapa",
        cluster_threshold=0.60,
        device="cpu",
        transcript_provenance_hash="tx_hash_2",
    )
    turns3 = diarize_segments(
        Path("test.wav"),
        segments,
        device="cpu",
        engine="ecapa",
        cache_path=cache_file,
        provenance=prov2,
    )
    assert len(turns3) == 2
    assert mock_get_model.call_count == 2  # Re-ran!
