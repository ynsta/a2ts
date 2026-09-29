from a2ts.models import (
    AlignedTurn,
    EntityRecord,
    RawSegment,
    SessionMetadata,
    SpeakerInfo,
    SpeakersMapping,
    SpeakerTurn,
    TrackCacheFile,
    TrackCacheProvenance,
    WordTimestamp,
)


def test_word_timestamp() -> None:
    wt = WordTimestamp(word="bonjour", start=0.5, end=1.0, probability=0.98)
    assert wt.word == "bonjour"
    assert wt.start == 0.5
    assert wt.end == 1.0


def test_raw_segment_with_words() -> None:
    seg = RawSegment(
        id=1,
        start=0.0,
        end=2.0,
        text="Bonjour à tous",
        words=[
            WordTimestamp(word="Bonjour", start=0.0, end=0.8),
            WordTimestamp(word="à", start=0.9, end=1.1),
            WordTimestamp(word="tous", start=1.2, end=2.0),
        ],
    )
    assert len(seg.words) == 3
    assert seg.text == "Bonjour à tous"


def test_speaker_turn() -> None:
    turn = SpeakerTurn(
        id=1,
        start=0.0,
        end=5.0,
        cluster_id="SPEAKER_00",
        resolved_speaker="MJ",
        time_slice_id=0,
        flagged=False,
    )
    assert turn.cluster_id == "SPEAKER_00"
    assert turn.resolved_speaker == "MJ"


def test_aligned_turn() -> None:
    aligned = AlignedTurn(
        turn_id=1,
        start=0.0,
        end=2.5,
        speaker="MJ",
        cluster_id="SPEAKER_00",
        text="Test line",
        time_slice_id=1,
    )
    assert aligned.speaker == "MJ"
    assert aligned.time_slice_id == 1


def test_entity_record() -> None:
    entity = EntityRecord(
        name="Kaelen", kind="wikilink", source_file="contexte/perso.md"
    )
    assert entity.name == "Kaelen"
    assert entity.kind == "wikilink"


def test_speakers_mapping_defaults() -> None:
    mapping1 = SpeakersMapping()
    mapping2 = SpeakersMapping()

    assert mapping1.cluster_defaults == {}
    assert mapping1.slice_overrides == {}
    assert mapping1.turn_overrides == {}

    # Verify default factory produces independent instances
    mapping1.cluster_defaults["SPEAKER_00"] = "MJ"
    assert "SPEAKER_00" not in mapping2.cluster_defaults


def test_speakers_mapping_with_values() -> None:
    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "MJ"},
        slice_overrides={"0": {"SPEAKER_01": "Alice"}},
        turn_overrides={1: "Bob"},
    )
    assert mapping.cluster_defaults["SPEAKER_00"] == "MJ"
    assert mapping.slice_overrides["0"]["SPEAKER_01"] == "Alice"
    assert mapping.turn_overrides[1] == "Bob"


def test_session_metadata() -> None:
    meta = SessionMetadata(
        media_path="/path/to/media.mp3",
        media_hash="hash123",
        duration_seconds=3600.0,
        engine="voxtral",
        model_name="mistral-7b-voxtral",
        prompt_hash="prome987",
        time_slice_minutes=15.0,
        created_at="2026-09-21T11:00:00Z",
    )
    assert meta.media_path == "/path/to/media.mp3"
    assert meta.media_hash == "hash123"
    assert meta.duration_seconds == 3600.0
    assert meta.engine == "voxtral"
    assert meta.model_name == "mistral-7b-voxtral"
    assert meta.prompt_hash == "prome987"
    assert meta.time_slice_minutes == 15.0
    assert meta.created_at == "2026-09-21T11:00:00Z"


def test_voice_profile_and_database() -> None:
    from a2ts.models import VoiceProfile, VoiceProfilesDatabase

    vp = VoiceProfile(
        speaker_name="Brakk",
        centroid=[0.1, 0.2, -0.3],
        sample_count=5,
    )
    assert vp.speaker_name == "Brakk"
    assert len(vp.centroid) == 3
    assert vp.sample_count == 5

    db = VoiceProfilesDatabase(
        speakers={"Brakk": vp},
        version=1,
    )
    assert "Brakk" in db.speakers
    assert db.speakers["Brakk"].sample_count == 5
    dumped = db.model_dump()
    assert dumped["speakers"]["Brakk"]["speaker_name"] == "Brakk"


def test_craig_models() -> None:
    # 1. SpeakerInfo defaults & full validation
    speaker_min = SpeakerInfo(
        discord_username="brakk_player",
        character_name="Brakk",
    )
    assert speaker_min.discord_username == "brakk_player"
    assert speaker_min.character_name == "Brakk"
    assert speaker_min.role is None
    assert speaker_min.nicknames == []
    assert speaker_min.is_dm is False
    assert speaker_min.raw_description is None

    speaker_full = SpeakerInfo(
        discord_username="gm_user",
        character_name="MJ",
        role="Maître du Jeu",
        nicknames=["DM", "Boss"],
        is_dm=True,
        raw_description="MJ: Maître du Jeu, surnoms: (DM, Boss)",
    )
    assert speaker_full.role == "Maître du Jeu"
    assert speaker_full.nicknames == ["DM", "Boss"]
    assert speaker_full.is_dm is True
    assert speaker_full.raw_description == "MJ: Maître du Jeu, surnoms: (DM, Boss)"

    # Serialization roundtrip
    speaker_dump = speaker_full.model_dump()
    speaker_restored = SpeakerInfo.model_validate(speaker_dump)
    assert speaker_restored == speaker_full

    # 2. TrackCacheProvenance defaults & validation
    prov = TrackCacheProvenance(
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="hash_abc123",
        source_file_size=1048576,
        source_file_mtime=1727640000.0,
    )
    assert prov.schema_version == 1
    assert prov.vad_parameters == {}
    assert prov.model_name == "large-v3"
    assert prov.compute_type == "float16"
    assert prov.prompt_hash == "hash_abc123"
    assert prov.source_file_size == 1048576
    assert prov.source_file_mtime == 1727640000.0

    prov_custom = TrackCacheProvenance(
        schema_version=2,
        model_name="turbo",
        compute_type="int8",
        prompt_hash="hash_def456",
        vad_parameters={"threshold": 0.6, "min_speech_duration_ms": 250},
        source_file_size=2097152,
        source_file_mtime=1727643600.5,
    )
    assert prov_custom.schema_version == 2
    assert prov_custom.vad_parameters["threshold"] == 0.6

    # 3. TrackCacheFile validation & roundtrip
    seg = RawSegment(
        id=0,
        start=0.5,
        end=3.2,
        text="Bonjour tout le monde.",
        words=[WordTimestamp(word="Bonjour", start=0.5, end=1.0, probability=0.99)],
    )
    cache_file = TrackCacheFile(
        provenance=prov,
        segments=[seg],
    )
    assert cache_file.provenance.model_name == "large-v3"
    assert len(cache_file.segments) == 1
    assert cache_file.segments[0].text == "Bonjour tout le monde."

    json_str = cache_file.model_dump_json()
    cache_restored = TrackCacheFile.model_validate_json(json_str)
    assert cache_restored == cache_file
    assert cache_restored.segments[0].words[0].word == "Bonjour"
