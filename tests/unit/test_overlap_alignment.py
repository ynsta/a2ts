"""Observable overlap attribution and chronological rendering contracts."""

from a2ts.consolidator import debounce_consecutive_turns, render_markdown_transcript
from a2ts.models import RawSegment, SpeakersMapping, SpeakerTurn, WordTimestamp
from a2ts.refiner import validate_refiner_chunk
from a2ts.speaker_review import apply_speakers_mapping
from a2ts.timeline import align_words_to_speaker_turns


def segment(words: list[WordTimestamp]) -> RawSegment:
    return RawSegment(
        id=0, start=words[0].start, end=words[-1].end, text="", words=words
    )


def test_overlap_uses_maximal_intersection_and_marks_uncertainty() -> None:
    words = [WordTimestamp(word="hello", start=1, end=3)]
    turns = [
        SpeakerTurn(id=4, start=0, end=2.1, cluster_id="A"),
        SpeakerTurn(id=5, start=1.5, end=4, cluster_id="B"),
    ]
    result = align_words_to_speaker_turns([segment(words)], turns)
    assert result[0].cluster_id == "B"
    assert result[0].speaker_uncertain


def test_equal_overlap_uses_turn_center_proximity() -> None:
    words = [WordTimestamp(word="hello", start=5, end=6)]
    turns = [
        SpeakerTurn(id=4, start=0, end=10, cluster_id="A"),
        SpeakerTurn(id=5, start=5, end=6, cluster_id="B"),
    ]
    assert align_words_to_speaker_turns([segment(words)], turns)[0].cluster_id == "B"


def test_alternating_runs_preserve_tokens_and_source_override() -> None:
    words = [
        WordTimestamp(word="one", start=1, end=2),
        WordTimestamp(word="two", start=5, end=6),
        WordTimestamp(word="three", start=8, end=9),
    ]
    turns = [
        SpeakerTurn(id=4, start=0, end=10, cluster_id="A"),
        SpeakerTurn(id=5, start=5, end=6, cluster_id="B"),
    ]
    result = align_words_to_speaker_turns([segment(words)], turns)
    assert [t.cluster_id for t in result] == ["A", "B", "A"]
    assert [w for t in result for w in t.words] == words
    assert len({t.turn_id for t in result}) == 3
    assert result[0].turn_id == 4
    assert result[-1].source_turn_id == 4
    mapped = apply_speakers_mapping(
        result, SpeakersMapping(turn_overrides={4: "Alice"})
    )
    assert mapped[0].speaker == mapped[-1].speaker == "Alice"


def test_continuations_and_punctuation_stay_with_lexical_token() -> None:
    words = [
        WordTimestamp(word="qu", start=0, end=0.2),
        WordTimestamp(word="'il", start=0.2, end=0.4),
        WordTimestamp(word=",", start=0.4, end=0.4),
        WordTimestamp(word="dit", start=1, end=1.2),
        WordTimestamp(word="-il", start=1.2, end=1.4),
    ]
    turns = [
        SpeakerTurn(id=1, start=0, end=0.2, cluster_id="A"),
        SpeakerTurn(id=2, start=0.2, end=2, cluster_id="B"),
    ]
    result = align_words_to_speaker_turns([segment(words)], turns)
    assert [t.text for t in result] == ["qu'il,", "dit-il"]
    assert result[0].speaker_uncertain


def test_debounce_and_refinement_preserve_uncertainty_warning() -> None:
    words = [
        WordTimestamp(word="hello", start=0, end=1),
        WordTimestamp(word="again", start=2, end=3),
    ]
    turns = [
        SpeakerTurn(id=1, start=0, end=1, cluster_id="A"),
        SpeakerTurn(id=2, start=2, end=3, cluster_id="A"),
        SpeakerTurn(id=3, start=2.5, end=3, cluster_id="B"),
    ]
    result = debounce_consecutive_turns(
        align_words_to_speaker_turns([segment(words)], turns)
    )
    assert len(result) == 1 and result[0].speaker_uncertain
    markdown = render_markdown_transcript(result)
    warning = "> [!WARNING] Some speaker attributions are uncertain. Review speaker assignments against the audio."
    assert markdown.startswith(warning)
    assert validate_refiner_chunk(markdown, markdown)
    assert not validate_refiner_chunk(markdown, markdown.replace(warning, ""))


def test_profile_enrollment_rejects_uncertain_or_conflicting_source_runs() -> None:
    import numpy as np

    from a2ts.diarizer import compute_voice_profiles
    from a2ts.models import AlignedTurn

    first = AlignedTurn(
        turn_id=4,
        source_turn_id=4,
        start=1,
        end=2,
        speaker="Alice",
        cluster_id="A",
        text="one",
    )
    last = first.model_copy(update={"turn_id": 6, "start": 8.0, "end": 9.0})
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    mapping = SpeakersMapping(turn_overrides={4: "Alice", 6: "Bob"})
    assert (
        compute_voice_profiles(embeddings, [4], [first, last], mapping).speakers == {}
    )
    mapping.turn_overrides.pop(6)
    last.speaker_uncertain = True
    assert (
        compute_voice_profiles(embeddings, [4], [first, last], mapping).speakers == {}
    )
    last.speaker_uncertain = False
    profile = compute_voice_profiles(embeddings, [4], [first, last], mapping).speakers[
        "Alice"
    ]
    assert profile.sample_count == 1
    assert (
        apply_speakers_mapping(
            [last], SpeakersMapping(turn_overrides={4: "Alice", 6: "Bob"})
        )[0].speaker
        == "Bob"
    )


def test_refiner_rejects_warning_moved_to_other_turn() -> None:
    warning = "> [!WARNING] Speaker attribution uncertain: multiple candidate speakers."
    first = "### [00:00:00 - 00:00:01] A\n"
    second = "### [00:00:02 - 00:00:03] B\n"
    original = first + warning + "\nhello\n" + second + "world\n"
    refined = first + "hello\n" + second + warning + "\nworld\n"
    assert not validate_refiner_chunk(original, refined)


def test_alignment_sorts_segments_and_handles_zero_duration_boundaries() -> None:
    later = WordTimestamp(word="later", start=4, end=5)
    boundary = WordTimestamp(word="now", start=2, end=2)
    turns = [
        SpeakerTurn(id=1, start=0, end=2, cluster_id="A"),
        SpeakerTurn(id=2, start=2, end=5, cluster_id="B"),
    ]
    result = align_words_to_speaker_turns(
        [segment([later]), segment([boundary])], turns
    )
    assert [w for t in result for w in t.words] == [boundary, later]
    assert result[0].speaker_uncertain


def test_equal_score_overlap_is_deterministic_across_turn_input_order() -> None:
    word = WordTimestamp(word="hello", start=4, end=6)
    turns = [
        SpeakerTurn(id=1, start=0, end=10, cluster_id="A"),
        SpeakerTurn(id=2, start=0, end=10, cluster_id="B"),
    ]
    forward = align_words_to_speaker_turns([segment([word])], turns)
    backward = align_words_to_speaker_turns([segment([word])], list(reversed(turns)))
    assert forward == backward
    assert forward[0].speaker_uncertain


def test_continuation_after_long_gap_starts_its_own_run() -> None:
    words = [
        WordTimestamp(word="hello", start=0, end=1),
        WordTimestamp(word="'again", start=3, end=4),
    ]
    turns = [
        SpeakerTurn(id=1, start=0, end=1, cluster_id="A"),
        SpeakerTurn(id=2, start=3, end=4, cluster_id="B"),
    ]
    assert [t.text for t in align_words_to_speaker_turns([segment(words)], turns)] == [
        "hello",
        "'again",
    ]


def test_continuation_after_long_gap_keeps_spacing_within_same_source() -> None:
    words = [
        WordTimestamp(word="hello", start=0, end=1),
        WordTimestamp(word="'again", start=3, end=4),
    ]
    turns = [SpeakerTurn(id=1, start=0, end=5, cluster_id="A")]
    assert (
        align_words_to_speaker_turns([segment(words)], turns)[0].text == "hello 'again"
    )


def test_profile_enrollment_inherits_source_override_independent_of_run_order() -> None:
    import numpy as np

    from a2ts.diarizer import compute_voice_profiles
    from a2ts.models import AlignedTurn

    first = AlignedTurn(
        turn_id=4,
        source_turn_id=4,
        start=1,
        end=2,
        speaker="SPEAKER_00",
        cluster_id="SPEAKER_00",
        text="one",
    )
    last = first.model_copy(update={"turn_id": 6, "start": 8.0, "end": 9.0})
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    mapping = SpeakersMapping(turn_overrides={4: "Alice"})
    forward = compute_voice_profiles(embeddings, [4], [first, last], mapping)
    reverse = compute_voice_profiles(embeddings, [4], [last, first], mapping)
    assert reverse.speakers == forward.speakers
    assert reverse.speakers["Alice"].sample_count == 1
    assert (
        compute_voice_profiles(embeddings, [4], [last], mapping)
        .speakers["Alice"]
        .sample_count
        == 1
    )
