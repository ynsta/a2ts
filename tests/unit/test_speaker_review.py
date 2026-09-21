from pathlib import Path
from unittest.mock import MagicMock, patch

from a2ts.models import AlignedTurn, SpeakersMapping
from a2ts.speaker_review import (
    apply_speakers_mapping,
    collect_speaker_stats,
    load_speakers_mapping,
    run_interactive_review,
    save_speakers_mapping,
)


def test_collect_speaker_stats() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Bonjour",
        ),
        AlignedTurn(
            turn_id=1,
            start=6.0,
            end=10.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Deuxième phrase",
        ),
        AlignedTurn(
            turn_id=2,
            start=11.0,
            end=14.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="Salut",
        ),
    ]
    stats = collect_speaker_stats(turns)
    assert "SPEAKER_00" in stats
    assert stats["SPEAKER_00"]["turn_count"] == 2
    assert stats["SPEAKER_00"]["total_duration"] == 9.0
    assert len(stats["SPEAKER_00"]["samples"]) == 2
    assert stats["SPEAKER_00"]["samples"][0] == (0.0, "Bonjour")
    assert stats["SPEAKER_01"]["turn_count"] == 1
    assert stats["SPEAKER_01"]["total_duration"] == 3.0


def test_apply_speakers_mapping() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="A",
            time_slice_id=0,
        ),
        AlignedTurn(
            turn_id=1,
            start=6.0,
            end=10.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="B",
            time_slice_id=1,
        ),
        AlignedTurn(
            turn_id=2,
            start=11.0,
            end=15.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="C",
            time_slice_id=1,
        ),
    ]
    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Alice"},
        slice_overrides={"1": {"SPEAKER_00": "Bob"}},
        turn_overrides={2: "Charlie"},
    )
    mapped = apply_speakers_mapping(turns, mapping)
    assert mapped[0].speaker == "Alice"
    assert mapped[1].speaker == "Bob"
    assert mapped[2].speaker == "Charlie"


def test_save_and_load_mapping(tmp_path: Path) -> None:
    mapping_file = tmp_path / "subdir" / "mapping.json"
    mapping = SpeakersMapping(cluster_defaults={"SPEAKER_00": "MJ"})
    save_speakers_mapping(mapping, mapping_file)

    loaded = load_speakers_mapping(mapping_file)
    assert loaded.cluster_defaults["SPEAKER_00"] == "MJ"


def test_load_nonexistent_mapping(tmp_path: Path) -> None:
    missing_file = tmp_path / "nonexistent.json"
    loaded = load_speakers_mapping(missing_file)
    assert loaded == SpeakersMapping()


def test_run_interactive_review_options() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="A",
        ),
        AlignedTurn(
            turn_id=1,
            start=6.0,
            end=10.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="B",
        ),
        AlignedTurn(
            turn_id=2,
            start=11.0,
            end=15.0,
            speaker="SPEAKER_02",
            cluster_id="SPEAKER_02",
            text="C",
        ),
    ]
    candidates = ["Alice", "Bob"]

    # SPEAKER_00 selects option '1' (Alice)
    # SPEAKER_01 selects 'c' (custom) then provides 'Charlie'
    # SPEAKER_02 selects 's' (skip)
    prompt_inputs = ["1", "c", "Charlie", "s"]

    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        mapping = run_interactive_review(turns, candidates)
        assert mapping.cluster_defaults == {
            "SPEAKER_00": "Alice",
            "SPEAKER_01": "Charlie",
        }
        assert mock_ask.call_count == 4


def test_run_interactive_review_with_existing_mapping() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Hello",
        ),
        AlignedTurn(
            turn_id=1,
            start=6.0,
            end=10.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="World",
        ),
    ]
    # Deduplication test: duplicate "Alice" in candidates
    candidates = ["Alice", "Alice", "Bob"]
    existing = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Alice", "SPEAKER_01": "OldBob"}
    )

    # SPEAKER_00 selects 's' (keep existing "Alice")
    # SPEAKER_01 selects '2' (change to "Bob")
    prompt_inputs = ["s", "2"]

    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        mapping = run_interactive_review(turns, candidates, existing_mapping=existing)
        assert mapping.cluster_defaults == {
            "SPEAKER_00": "Alice",
            "SPEAKER_01": "Bob",
        }
        assert mock_ask.call_count == 2


def test_run_interactive_review_direct_free_text_entry() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Je prépare mon sort.",
        ),
    ]
    candidates = ["Sentier de Triboar", "Phandalin"]
    # User types "MJ (Tessaro)" directly without pressing 'c'
    prompt_inputs = ["MJ (Tessaro)"]

    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        mapping = run_interactive_review(turns, candidates)
        assert mapping.cluster_defaults == {"SPEAKER_00": "MJ (Tessaro)"}
        assert mock_ask.call_count == 1


@patch("a2ts.speaker_review.play_audio_clip")
def test_run_interactive_review_autoplay_audio(
    mock_play: MagicMock, tmp_path: Path
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()
    turns = [
        AlignedTurn(
            turn_id=0,
            start=2.0,
            end=6.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="First sentence.",
        ),
    ]
    candidates = ["Brakk"]
    # User selects "1" (Brakk)
    with patch("a2ts.speaker_review.Prompt.ask", return_value="1"):
        mapping = run_interactive_review(
            turns, candidates, audio_path=audio_file, auto_play=True, audio_padding=2.0
        )
        assert mapping.cluster_defaults == {"SPEAKER_00": "Brakk"}
        mock_play.assert_called_once_with(
            audio_file, start=2.0, duration=4.0, pad_before=2.0, pad_after=2.0
        )


@patch("a2ts.speaker_review.play_audio_clip")
def test_run_interactive_review_manual_play_and_replay(
    mock_play: MagicMock, tmp_path: Path
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()
    turns = [
        AlignedTurn(
            turn_id=0,
            start=1.0,
            end=3.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="First sentence.",
        ),
        AlignedTurn(
            turn_id=1,
            start=5.0,
            end=8.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Second sentence.",
        ),
    ]
    candidates = ["Merrow"]
    # User first types "p2" to hear quote 2, then "p" to replay quote 1, then "1"
    prompt_inputs = ["p2", "p", "1"]
    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        mapping = run_interactive_review(
            turns, candidates, audio_path=audio_file, auto_play=False, audio_padding=2.0
        )
        assert mapping.cluster_defaults == {"SPEAKER_00": "Merrow"}
        assert mock_ask.call_count == 3
        assert mock_play.call_count == 2
        # First call was for sample 2 (start=5.0, duration=3.0)
        assert mock_play.call_args_list[0][1]["start"] == 5.0
        assert mock_play.call_args_list[0][1]["pad_before"] == 2.0
        # Second call was for sample 1 (start=1.0, duration=2.0)
        assert mock_play.call_args_list[1][1]["start"] == 1.0
        assert mock_play.call_args_list[1][1]["pad_before"] == 2.0


@patch("a2ts.speaker_review.play_audio_clip")
def test_run_interactive_review_wide_playback(
    mock_play: MagicMock, tmp_path: Path
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()
    turns = [
        AlignedTurn(
            turn_id=0,
            start=10.0,
            end=12.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Sentence.",
        ),
    ]
    candidates = ["Brakk"]
    # User types "w" to hear wider context, then "1"
    prompt_inputs = ["w", "1"]
    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        mapping = run_interactive_review(
            turns, candidates, audio_path=audio_file, auto_play=False, audio_padding=2.0
        )
        assert mapping.cluster_defaults == {"SPEAKER_00": "Brakk"}
        assert mock_ask.call_count == 2
        mock_play.assert_called_once_with(
            audio_file, start=10.0, duration=2.0, pad_before=5.0, pad_after=5.0
        )


@patch("a2ts.speaker_review.play_audio_clip")
def test_run_interactive_review_time_slice_audio(
    mock_play: MagicMock, tmp_path: Path
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()
    turns = [
        AlignedTurn(
            turn_id=0,
            start=10.0,
            end=13.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Slice zero sentence.",
            time_slice_id=0,
        ),
    ]
    candidates = ["Ilvaris"]
    # User selects "t" (time slice), plays sample "p", then selects "1"
    prompt_inputs = ["t", "p", "1"]
    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        mapping = run_interactive_review(
            turns, candidates, audio_path=audio_file, auto_play=False, audio_padding=2.0
        )
        assert mapping.slice_overrides == {"0": {"SPEAKER_00": "Ilvaris"}}
        assert mock_ask.call_count == 3
        mock_play.assert_called_once_with(
            audio_file, start=10.0, duration=3.0, pad_before=2.0, pad_after=2.0
        )
