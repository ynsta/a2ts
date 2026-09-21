"""Unit tests for acoustic speaker diarization module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from a2ts.diarizer import diarize_segments
from a2ts.models import RawSegment


def test_diarize_segments_empty() -> None:
    turns = diarize_segments(Path("dummy.wav"), [])
    assert turns == []


def test_diarize_segments_single_segment(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.json"
    segments = [RawSegment(id=0, start=1.0, end=4.0, text="Seul au monde.")]
    turns = diarize_segments(Path("dummy.wav"), segments, cache_path=cache_file)
    assert len(turns) == 1
    assert turns[0].cluster_id == "SPEAKER_00"
    assert turns[0].start == 1.0
    assert turns[0].end == 4.0
    assert cache_file.is_file()


def test_diarize_segments_from_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(
        '[{"id": 0, "start": 0.0, "end": 2.0, "cluster_id": "SPEAKER_01"}]',
        encoding="utf-8",
    )
    segments = [RawSegment(id=0, start=0.0, end=2.0, text="Test")]
    turns = diarize_segments(Path("dummy.wav"), segments, cache_path=cache_file)
    assert len(turns) == 1
    assert turns[0].cluster_id == "SPEAKER_01"


@patch("a2ts.diarizer.sf.read")
@patch("a2ts.diarizer.get_embedding_model")
def test_diarize_segments_clustering(
    mock_get_model: MagicMock, mock_sf_read: MagicMock, tmp_path: Path
) -> None:
    # 16kHz audio with 10 seconds of samples
    dummy_audio = np.zeros(16000 * 10, dtype=np.float32)
    mock_sf_read.return_value = (dummy_audio, 16000)

    # Mock embeddings: seg 0 & seg 2 have vector [1, 0], seg 1 has vector [0, 1]
    emb1 = np.array([1.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0], dtype=np.float32)
    emb3 = np.array([0.9, 0.1], dtype=np.float32)

    mock_classifier = MagicMock()
    mock_classifier.encode_batch.side_effect = [
        MagicMock(squeeze=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: emb1))),
        MagicMock(squeeze=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: emb2))),
        MagicMock(squeeze=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: emb3))),
    ]
    mock_get_model.return_value = mock_classifier

    segments = [
        RawSegment(id=0, start=0.0, end=2.0, text="Speaker A first turn"),
        RawSegment(id=1, start=2.5, end=4.5, text="Speaker B response"),
        RawSegment(id=2, start=5.0, end=7.0, text="Speaker A second turn"),
    ]

    cache_file = tmp_path / "diar_out.json"
    turns = diarize_segments(
        Path("audio.wav"),
        segments,
        num_speakers=2,
        device="cpu",
        cache_path=cache_file,
    )

    assert len(turns) == 3
    # Seg 0 and Seg 2 should share the same cluster, different from Seg 1
    assert turns[0].cluster_id == turns[2].cluster_id
    assert turns[0].cluster_id != turns[1].cluster_id
    assert cache_file.is_file()
