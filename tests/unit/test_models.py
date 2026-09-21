from a2ts.models import (
    AlignedTurn,
    EntityRecord,
    RawSegment,
    SpeakerTurn,
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
    entity = EntityRecord(name="Kaelen", kind="wikilink", source_file="contexte/perso.md")
    assert entity.name == "Kaelen"
    assert entity.kind == "wikilink"
