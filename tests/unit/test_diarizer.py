"""Unit tests for acoustic speaker diarization module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

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


def test_extract_embeddings_caches_to_npy(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    from a2ts.diarizer import extract_embeddings
    from a2ts.models import RawSegment

    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="Hello"),
        RawSegment(id=1, start=1.5, end=2.5, text="World"),
    ]
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    cache_prefix = tmp_path / "test_emb"

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

        emb_matrix, valid_indices = extract_embeddings(
            audio_path, segments, device="cpu", cache_prefix=cache_prefix
        )

        assert emb_matrix.shape == (2, 2)
        assert valid_indices == [0, 1]
        assert (tmp_path / "test_emb_embeddings.npy").is_file()
        assert (tmp_path / "test_emb_indices.json").is_file()

        # Second call should load from cache without calling get_embedding_model
        mock_model.reset_mock()
        emb2, indices2 = extract_embeddings(
            audio_path, segments, device="cpu", cache_prefix=cache_prefix
        )
        assert np.allclose(emb_matrix, emb2)
        assert indices2 == valid_indices
        mock_model.assert_not_called()


def test_diarize_segments_with_num_speakers(tmp_path: Path) -> None:
    from unittest.mock import patch

    from a2ts.diarizer import diarize_segments
    from a2ts.models import RawSegment

    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="A"),
        RawSegment(id=1, start=1.0, end=2.0, text="B"),
        RawSegment(id=2, start=2.0, end=3.0, text="C"),
    ]
    audio_path = tmp_path / "test.wav"
    audio_path.touch()

    # 3 distinct vectors forced into 2 clusters
    cached_emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with patch(
        "a2ts.diarizer.extract_embeddings", return_value=(cached_emb, [0, 1, 2])
    ):
        turns = diarize_segments(
            audio_path,
            segments,
            num_speakers=2,
            device="cpu",
        )
        assert len(turns) == 3
        clusters = {t.cluster_id for t in turns}
        assert len(clusters) == 2
        # segments 0 and 1 should cluster together
        assert turns[0].cluster_id == turns[1].cluster_id
        assert turns[2].cluster_id != turns[0].cluster_id
