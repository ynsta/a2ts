"""Tests for split rule persistence, positional fallback removals, and protected voice profile enrollment."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from a2ts.cli import app
from a2ts.diarizer import compute_voice_profiles
from a2ts.models import (
    AlignedTurn,
    ClusterSplit,
    EmbeddingCacheProvenance,
    SessionMetadata,
    SpeakersMapping,
    SpeakerTurn,
)
from a2ts.speaker_review import apply_splits_to_turns, load_speakers_mapping


def test_cluster_split_survives_speakers_mapping_serialization(tmp_path: Path) -> None:
    """ClusterSplit must survive serialization in SpeakersMapping and JSON files."""
    split = ClusterSplit(cluster_id="SPEAKER_00", at=42.5, new_cluster_id="SPEAKER_02")
    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Alice", "SPEAKER_02": "Bob"},
        splits=[split],
        label_sources={"SPEAKER_00": "manual", "SPEAKER_02": "profile_match"},
    )

    dumped = mapping.model_dump()
    assert "splits" in dumped
    assert len(dumped["splits"]) == 1
    assert dumped["splits"][0]["cluster_id"] == "SPEAKER_00"
    assert dumped["splits"][0]["at"] == 42.5
    assert dumped["splits"][0]["new_cluster_id"] == "SPEAKER_02"
    assert dumped["label_sources"]["SPEAKER_00"] == "manual"
    assert dumped["label_sources"]["SPEAKER_02"] == "profile_match"

    mapping_file = tmp_path / "speakers_mapping.json"
    mapping_file.write_text(json.dumps(dumped, indent=2), encoding="utf-8")

    loaded = load_speakers_mapping(mapping_file)
    assert len(loaded.splits) == 1
    assert loaded.splits[0].cluster_id == "SPEAKER_00"
    assert loaded.splits[0].at == 42.5
    assert loaded.splits[0].new_cluster_id == "SPEAKER_02"
    assert loaded.label_sources["SPEAKER_00"] == "manual"
    assert loaded.label_sources["SPEAKER_02"] == "profile_match"


def test_apply_splits_to_turns() -> None:
    """apply_splits_to_turns should reassign cluster_id for turns occurring at or after split.at."""
    turns = [
        AlignedTurn(
            turn_id=1,
            start=10.0,
            end=20.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="First part of conversation",
        ),
        AlignedTurn(
            turn_id=2,
            start=45.0,
            end=55.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Second part of conversation",
        ),
        AlignedTurn(
            turn_id=3,
            start=60.0,
            end=70.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="Third turn by another speaker",
        ),
    ]

    splits = [
        ClusterSplit(cluster_id="SPEAKER_00", at=42.5, new_cluster_id="SPEAKER_02")
    ]

    updated = apply_splits_to_turns(turns, splits)
    assert updated[0].cluster_id == "SPEAKER_00"
    assert updated[1].cluster_id == "SPEAKER_02"
    assert updated[2].cluster_id == "SPEAKER_01"


def test_split_cli_persists_rule_and_reapply(tmp_path: Path) -> None:
    """split command must persist ClusterSplit in speakers_mapping.json and re-applying must work."""
    session_dir = tmp_path / "session_test"
    session_dir.mkdir(parents=True)

    turns = [
        AlignedTurn(
            turn_id=1,
            start=10.0,
            end=20.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="First part",
        ),
        AlignedTurn(
            turn_id=2,
            start=50.0,
            end=60.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Second part",
        ),
    ]
    turns_path = session_dir / "turns.json"
    turns_path.write_text(
        json.dumps([t.model_dump() for t in turns], indent=2), encoding="utf-8"
    )

    mapping = SpeakersMapping(cluster_defaults={"SPEAKER_00": "Brakk"})
    mapping_path = session_dir / "speakers_mapping.json"
    mapping_path.write_text(
        json.dumps(mapping.model_dump(), indent=2), encoding="utf-8"
    )

    session_meta = SessionMetadata(
        media_path="/path/to/media.mp3",
        media_hash="abc123hash",
        duration_seconds=100.0,
        engine="whisper",
        model_name="large-v3",
        prompt_hash="",
        time_slice_minutes=15.0,
        created_at="2026-09-30T00:00:00Z",
        output_path=str(session_dir / "transcript.md"),
    )
    (session_dir / "session.json").write_text(
        session_meta.model_dump_json(indent=2), encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "split",
            str(session_dir),
            "SPEAKER_00",
            "--at",
            "40.0",
            "--to",
            "SPEAKER_02",
        ],
    )
    assert result.exit_code == 0, f"Split failed: {result.stdout}"

    # Verify split persisted in speakers_mapping.json
    saved_mapping = load_speakers_mapping(mapping_path)
    assert len(saved_mapping.splits) == 1
    assert saved_mapping.splits[0].cluster_id == "SPEAKER_00"
    assert saved_mapping.splits[0].at == 40.0
    assert saved_mapping.splits[0].new_cluster_id == "SPEAKER_02"

    # Verify turns.json was updated
    updated_turns_data = json.loads(turns_path.read_text(encoding="utf-8"))
    assert updated_turns_data[0]["cluster_id"] == "SPEAKER_00"
    assert updated_turns_data[1]["cluster_id"] == "SPEAKER_02"

    # Verify re-applying splits on fresh turns reproduces the split
    fresh_turns = [
        AlignedTurn(
            turn_id=1,
            start=10.0,
            end=20.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="First part",
        ),
        AlignedTurn(
            turn_id=2,
            start=50.0,
            end=60.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Second part",
        ),
    ]
    reapplied = apply_splits_to_turns(fresh_turns, saved_mapping.splits)
    assert reapplied[0].cluster_id == "SPEAKER_00"
    assert reapplied[1].cluster_id == "SPEAKER_02"


def test_positional_fallbacks_removed_in_compute_voice_profiles() -> None:
    """When valid_idx does not match any turn ID, it must be safely skipped and not fall back to turns[valid_idx]."""
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    # Turn has ID 999, but is at index 0 in the list
    turns = [
        SpeakerTurn(
            id=999,
            start=0.0,
            end=1.0,
            cluster_id="SPEAKER_00",
            resolved_speaker="Brakk",
        )
    ]
    # valid_indices contains index 0, which does NOT match turn ID 999
    valid_indices = [0]
    mapping = SpeakersMapping(cluster_defaults={"SPEAKER_00": "Brakk"})

    db = compute_voice_profiles(embeddings, valid_indices, turns, mapping)
    # With positional fallback, 0 < len(turns) caused turns[0] to be picked up.
    # Without positional fallback, valid_idx=0 misses turn ID 999, so 0 samples enrolled.
    assert len(db.speakers) == 0


def test_closed_set_and_unconfirmed_matches_not_enrolled(tmp_path: Path) -> None:
    """Voice profile enrollment must skip unconfirmed profile matches and closed-set assignments."""
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    turns = [
        SpeakerTurn(id=1, start=0.0, end=1.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=2, start=2.0, end=3.0, cluster_id="SPEAKER_01"),
    ]
    valid_indices = [1, 2]

    # SPEAKER_00 is manually confirmed -> Brakk
    # SPEAKER_01 is an unconfirmed profile_match -> Bob
    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Brakk", "SPEAKER_01": "Bob"},
        label_sources={"SPEAKER_00": "manual", "SPEAKER_01": "profile_match"},
    )

    db = compute_voice_profiles(embeddings, valid_indices, turns, mapping)
    # Only Brakk (manual) should be enrolled, NOT Bob (profile_match)
    assert "Brakk" in db.speakers
    assert "Bob" not in db.speakers


def test_recluster_applies_persisted_splits(tmp_path: Path) -> None:
    """recluster command must apply persisted splits from speakers_mapping.json to regenerated turns."""
    session_dir = tmp_path / "session_recluster"
    session_dir.mkdir(parents=True)
    media_hash = "fake_media_hash"

    session_meta = SessionMetadata(
        media_path=str(tmp_path / "audio.wav"),
        media_hash=media_hash,
        duration_seconds=120.0,
        engine="whisper",
        model_name="large-v3",
        prompt_hash="",
        time_slice_minutes=15.0,
        created_at="2026-09-30T00:00:00Z",
        output_path=str(session_dir / "transcript.md"),
    )
    (session_dir / "session.json").write_text(
        session_meta.model_dump_json(indent=2), encoding="utf-8"
    )

    # Mock transcripts cache
    transcripts_dir = session_dir / "transcripts"
    transcripts_dir.mkdir(parents=True)
    segments = [
        {"id": 0, "start": 0.0, "end": 10.0, "text": "Segment 1", "words": []},
        {"id": 1, "start": 15.0, "end": 25.0, "text": "Segment 2", "words": []},
        {"id": 2, "start": 40.0, "end": 50.0, "text": "Segment 3", "words": []},
    ]
    (transcripts_dir / f"{media_hash}_whisper.json").write_text(
        json.dumps({"segments": segments}, indent=2), encoding="utf-8"
    )

    # Mock diarization cache with 1D identical embeddings
    diar_dir = session_dir / "diarization"
    diar_dir.mkdir(parents=True)
    emb_matrix = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    np.save(diar_dir / f"{media_hash}_segment_embeddings.npy", emb_matrix)
    (diar_dir / f"{media_hash}_segment_indices.json").write_text(
        json.dumps([0, 1, 2]), encoding="utf-8"
    )
    prov = EmbeddingCacheProvenance(
        entity_kind="segment",
        media_hash=media_hash,
        count=3,
    )
    (diar_dir / f"{media_hash}_segment_embeddings_provenance.json").write_text(
        prov.model_dump_json(), encoding="utf-8"
    )

    # Mapping with a split rule: SPEAKER_00 at 30s -> SPEAKER_SPLIT
    split_rule = ClusterSplit(
        cluster_id="SPEAKER_00", at=30.0, new_cluster_id="SPEAKER_SPLIT"
    )
    mapping = SpeakersMapping(splits=[split_rule])
    mapping_path = session_dir / "speakers_mapping.json"
    mapping_path.write_text(
        json.dumps(mapping.model_dump(), indent=2), encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(app, ["recluster", str(session_dir)])
    assert result.exit_code == 0, f"Recluster failed: {result.stdout}"

    # Verify turns.json has split applied
    turns_path = session_dir / "turns.json"
    turns_data = json.loads(turns_path.read_text(encoding="utf-8"))
    assert turns_data[0]["cluster_id"] == "SPEAKER_00"
    assert turns_data[1]["cluster_id"] == "SPEAKER_00"
    # Third turn starts at 40s >= 30s, so it should be SPEAKER_SPLIT
    assert turns_data[2]["cluster_id"] == "SPEAKER_SPLIT"


def test_review_closed_set_skips_voice_profile_auto_enrollment(tmp_path: Path) -> None:
    """When closed_set is enabled, review must NOT auto-enroll voice profiles."""
    session_dir = tmp_path / "session_review"
    session_dir.mkdir(parents=True)
    media_hash = "fake_media_hash"

    session_meta = SessionMetadata(
        media_path=str(tmp_path / "audio.wav"),
        media_hash=media_hash,
        duration_seconds=120.0,
        engine="whisper",
        model_name="large-v3",
        prompt_hash="",
        time_slice_minutes=15.0,
        created_at="2026-09-30T00:00:00Z",
        output_path=str(session_dir / "transcript.md"),
    )
    (session_dir / "session.json").write_text(
        session_meta.model_dump_json(indent=2), encoding="utf-8"
    )

    turns = [
        AlignedTurn(
            turn_id=1,
            start=0.0,
            end=10.0,
            speaker="Brakk",
            cluster_id="SPEAKER_00",
            text="Hello world",
        )
    ]
    (session_dir / "turns.json").write_text(
        json.dumps([t.model_dump() for t in turns], indent=2), encoding="utf-8"
    )

    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Brakk"},
        label_sources={"SPEAKER_00": "manual"},
    )
    (session_dir / "speakers_mapping.json").write_text(
        json.dumps(mapping.model_dump(), indent=2), encoding="utf-8"
    )

    diar_dir = session_dir / "diarization"
    diar_dir.mkdir(parents=True)
    emb_matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    np.save(diar_dir / f"{media_hash}_turn_embeddings.npy", emb_matrix)
    (diar_dir / f"{media_hash}_turn_indices.json").write_text(
        json.dumps([1]), encoding="utf-8"
    )

    vp_path = tmp_path / "voice_profiles.json"

    with (
        patch("a2ts.cli.run_interactive_review", return_value=mapping),
        patch("a2ts.cli.save_voice_profiles") as mock_save_vp,
    ):
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "review",
                str(session_dir),
                "--closed-set",
                "--voice-profiles",
                str(vp_path),
                "--output",
                str(tmp_path / "transcript.md"),
            ],
        )
        assert result.exit_code == 0, f"Review failed: {result.stdout}"
        mock_save_vp.assert_not_called()
