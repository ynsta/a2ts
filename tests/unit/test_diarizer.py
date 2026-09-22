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


def test_voice_profiles_computation_and_matching(tmp_path: Path) -> None:
    from a2ts.diarizer import (
        compute_voice_profiles,
        load_voice_profiles,
        match_embeddings_to_profiles,
        save_voice_profiles,
    )
    from a2ts.models import SpeakersMapping, SpeakerTurn

    # Two 2D unit vectors
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    valid_indices = [0, 1, 2]

    turns = [
        SpeakerTurn(id=0, start=0.0, end=1.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=1, start=1.0, end=2.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=2, start=2.0, end=3.0, cluster_id="SPEAKER_01"),
    ]
    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Brakk", "SPEAKER_01": "Merrow"}
    )

    db = compute_voice_profiles(embeddings, valid_indices, turns, mapping)
    assert "Brakk" in db.speakers
    assert "Merrow" in db.speakers
    assert db.speakers["Brakk"].sample_count == 2
    assert db.speakers["Merrow"].sample_count == 1

    # Check centroid normalization
    brakk_centroid = np.array(db.speakers["Brakk"].centroid)
    assert np.isclose(np.linalg.norm(brakk_centroid), 1.0)

    # Save and reload
    db_path = tmp_path / "voice_profiles.json"
    save_voice_profiles(db, db_path)
    loaded = load_voice_profiles(db_path)
    assert loaded is not None
    assert "Brakk" in loaded.speakers

    # Test matching
    test_embs = np.array(
        [
            [0.99, 0.05],  # Very close to Brakk
            [0.02, 0.99],  # Very close to Merrow
            [-1.0, 0.0],  # Opposite / unknown
        ],
        dtype=np.float32,
    )
    matches = match_embeddings_to_profiles(test_embs, loaded, similarity_threshold=0.60)
    assert matches[0][0] == "Brakk"
    assert matches[1][0] == "Merrow"
    assert 2 not in matches  # Did not match


def test_voice_profiles_combines_existing_db() -> None:
    from a2ts.diarizer import compute_voice_profiles
    from a2ts.models import SpeakerTurn, VoiceProfile, VoiceProfilesDatabase

    existing_db = VoiceProfilesDatabase(
        speakers={
            "Brakk": VoiceProfile(
                speaker_name="Brakk",
                centroid=[1.0, 0.0],
                sample_count=2,
            ),
            "Alice": VoiceProfile(
                speaker_name="Alice",
                centroid=[0.0, 1.0],
                sample_count=5,
            ),
        }
    )

    embeddings = np.array([[0.0, 1.0]], dtype=np.float32)
    valid_indices = [0]
    turns = [SpeakerTurn(id=0, start=0.0, end=1.0, cluster_id="SPEAKER_00")]
    mapping = {"SPEAKER_00": "Brakk"}

    db = compute_voice_profiles(
        embeddings, valid_indices, turns, mapping, existing_db=existing_db
    )
    # Alice untouched
    assert "Alice" in db.speakers
    assert db.speakers["Alice"].sample_count == 5

    # Brakk updated: 2*[1, 0] + 1*[0, 1] = [2, 1], normalized = [2/sqrt(5), 1/sqrt(5)]
    assert "Brakk" in db.speakers
    assert db.speakers["Brakk"].sample_count == 3
    expected_centroid = np.array([2.0, 1.0]) / np.linalg.norm([2.0, 1.0])
    assert np.allclose(db.speakers["Brakk"].centroid, expected_centroid)


def test_voice_profiles_ignores_unmapped_generic_speakers() -> None:
    from a2ts.diarizer import compute_voice_profiles
    from a2ts.models import SpeakerTurn

    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    valid_indices = [0]
    turns = [SpeakerTurn(id=0, start=0.0, end=1.0, cluster_id="SPEAKER_00")]
    # No mapping for SPEAKER_00
    db = compute_voice_profiles(embeddings, valid_indices, turns, {})
    assert len(db.speakers) == 0


def test_load_voice_profiles_invalid_or_missing(tmp_path: Path) -> None:
    from a2ts.diarizer import load_voice_profiles

    # Missing file
    assert load_voice_profiles(tmp_path / "nonexistent.json") is None

    # Invalid file
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json", encoding="utf-8")
    assert load_voice_profiles(bad_file) is None


def test_classify_clusters_to_profiles() -> None:
    from a2ts.diarizer import classify_clusters_to_profiles
    from a2ts.models import VoiceProfile, VoiceProfilesDatabase

    db = VoiceProfilesDatabase(
        speakers={
            "Brakk": VoiceProfile(
                speaker_name="Brakk",
                centroid=[1.0, 0.0],
                sample_count=2,
            ),
            "Merrow": VoiceProfile(
                speaker_name="Merrow",
                centroid=[0.0, 1.0],
                sample_count=2,
            ),
        }
    )

    cluster_embs = {
        "SPEAKER_00": [np.array([0.9, 0.1], dtype=np.float32)],
        "SPEAKER_01": [np.array([0.2, 0.8], dtype=np.float32)],
        "SPEAKER_02": [np.array([0.5, 0.5], dtype=np.float32)],  # sim ~0.707 to both
        "SPEAKER_03": [np.array([-0.9, -0.1], dtype=np.float32)],  # negative sim
    }

    # Open set with threshold 0.80
    open_matches = classify_clusters_to_profiles(
        cluster_embs, db, similarity_threshold=0.80, closed_set=False
    )
    assert "SPEAKER_00" in open_matches
    assert open_matches["SPEAKER_00"][0] == "Brakk"
    assert "SPEAKER_01" in open_matches
    assert open_matches["SPEAKER_01"][0] == "Merrow"
    assert "SPEAKER_02" not in open_matches  # sim ~0.707 < 0.80
    assert "SPEAKER_03" not in open_matches

    # Closed set: all clusters must be assigned to closest profile
    closed_matches = classify_clusters_to_profiles(cluster_embs, db, closed_set=True)
    assert closed_matches["SPEAKER_00"][0] == "Brakk"
    assert closed_matches["SPEAKER_01"][0] == "Merrow"
    assert closed_matches["SPEAKER_02"][0] in ("Brakk", "Merrow")
    assert (
        closed_matches["SPEAKER_03"][0] == "Merrow"
        or closed_matches["SPEAKER_03"][0] == "Brakk"
    )
    assert len(closed_matches) == 4


def test_propagate_speaker_labels() -> None:
    from a2ts.diarizer import propagate_speaker_labels
    from a2ts.models import AlignedTurn, SpeakersMapping

    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=1.0,
            speaker="SPEAKER_99",
            cluster_id="SPEAKER_99",
            text="unassigned leading",
        ),
        AlignedTurn(
            turn_id=1,
            start=1.0,
            end=2.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="hello",
        ),
        AlignedTurn(
            turn_id=2,
            start=2.0,
            end=2.2,
            speaker="SPEAKER_88",
            cluster_id="SPEAKER_88",
            text="quick blip",
        ),
        AlignedTurn(
            turn_id=3,
            start=2.2,
            end=3.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="world",
        ),
        AlignedTurn(
            turn_id=4,
            start=3.0,
            end=3.5,
            speaker="SPEAKER_77",
            cluster_id="SPEAKER_77",
            text="trailing blip",
        ),
    ]

    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Brakk", "SPEAKER_01": "Dorsa"}
    )

    propagate_speaker_labels(turns, mapping)

    # turn 0 was leading unassigned -> propagated from turn 1 (Brakk)
    assert mapping.turn_overrides[0] == "Brakk"
    # turn 2 was between Brakk and Dorsa -> forward propagated from turn 1 (Brakk)
    assert mapping.turn_overrides[2] == "Brakk"
    # turn 4 was trailing unassigned -> propagated from turn 3 (Dorsa)
    assert mapping.turn_overrides[4] == "Dorsa"
