from a2ts.models import RawSegment, SpeakerTurn, WordTimestamp
from a2ts.timeline import (
    align_words_to_speaker_turns,
    assign_time_slices,
    detect_outlier_turns,
    split_cluster_at_time,
)


def test_assign_time_slices() -> None:
    turns = [
        SpeakerTurn(id=0, start=100.0, end=200.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(
            id=1, start=950.0, end=1050.0, cluster_id="SPEAKER_00"
        ),  # > 15 min (900s)
    ]
    sliced = assign_time_slices(turns, slice_minutes=15.0)
    assert sliced[0].time_slice_id == 0
    assert sliced[1].time_slice_id == 1


def test_split_cluster_at_time() -> None:
    turns = [
        SpeakerTurn(id=0, start=10.0, end=20.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=1, start=30.0, end=40.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=2, start=50.0, end=60.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=3, start=50.0, end=60.0, cluster_id="SPEAKER_02"),
    ]
    # Split SPEAKER_00 after 25.0s into SPEAKER_01
    updated = split_cluster_at_time(
        turns, cluster_id="SPEAKER_00", split_time=25.0, new_cluster_id="SPEAKER_01"
    )
    assert updated[0].cluster_id == "SPEAKER_00"
    assert updated[1].cluster_id == "SPEAKER_01"
    assert updated[2].cluster_id == "SPEAKER_01"
    assert updated[3].cluster_id == "SPEAKER_02"


def test_align_words_to_speaker_turns() -> None:
    segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=6.0,
            text="Bonjour à tous. On commence.",
            words=[
                WordTimestamp(word="Bonjour", start=0.0, end=1.0),
                WordTimestamp(word="à", start=1.1, end=1.5),
                WordTimestamp(word="tous.", start=1.6, end=2.5),
                WordTimestamp(word="On", start=3.5, end=4.0),
                WordTimestamp(word="commence.", start=4.1, end=5.5),
            ],
        )
    ]
    turns = [
        SpeakerTurn(
            id=0, start=0.0, end=3.0, cluster_id="SPEAKER_00", resolved_speaker="MJ"
        ),
        SpeakerTurn(
            id=1, start=3.2, end=6.0, cluster_id="SPEAKER_01", resolved_speaker="Alice"
        ),
    ]
    aligned = align_words_to_speaker_turns(segments, turns)
    assert len(aligned) == 2
    assert aligned[0].speaker == "MJ"
    assert aligned[0].text == "Bonjour à tous."
    assert aligned[1].speaker == "Alice"
    assert aligned[1].text == "On commence."


def test_align_words_to_speaker_turns_no_turns() -> None:
    segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=3.0,
            text="Seul dans le noir.",
            words=[WordTimestamp(word="Seul", start=0.0, end=1.0)],
        )
    ]
    aligned = align_words_to_speaker_turns(segments, [])
    assert len(aligned) == 1
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].text == "Seul dans le noir."


def test_align_words_to_speaker_turns_no_words() -> None:
    segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=3.0,
            text="Segment sans mots détaillés.",
            words=[],
        )
    ]
    turns = [
        SpeakerTurn(id=0, start=0.0, end=3.0, cluster_id="SPEAKER_00"),
    ]
    aligned = align_words_to_speaker_turns(segments, turns)
    assert len(aligned) == 1
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].text == "Segment sans mots détaillés."


def test_align_words_to_speaker_turns_empty_segments() -> None:
    turns = [
        SpeakerTurn(id=0, start=0.0, end=3.0, cluster_id="SPEAKER_00"),
    ]
    aligned = align_words_to_speaker_turns([], turns)
    assert aligned == []


def test_detect_outlier_turns() -> None:
    turns = [
        SpeakerTurn(id=0, start=0.0, end=5.0, cluster_id="SPEAKER_00", flagged=False),
        SpeakerTurn(
            id=1,
            start=5.5,
            end=6.0,
            cluster_id="SPEAKER_01",
            flagged=True,
            notes="Low confidence",
        ),
        SpeakerTurn(id=2, start=6.5, end=10.0, cluster_id="SPEAKER_00", flagged=False),
    ]
    outliers = detect_outlier_turns(turns)
    assert len(outliers) == 1
    assert outliers[0].id == 1
    assert outliers[0].notes == "Low confidence"
