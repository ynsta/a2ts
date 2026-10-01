from pathlib import Path
from unittest.mock import MagicMock, patch

from a2ts.models import AlignedTurn, SpeakersMapping
from a2ts.speaker_review import (
    apply_speakers_mapping,
    collect_speaker_stats,
    load_speakers_mapping,
    resolve_speaker_for_turn,
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


@patch("a2ts.speaker_review.stop_audio_playback")
@patch("a2ts.speaker_review.play_audio_clip_async")
def test_run_interactive_review_autoplay_audio(
    mock_play: MagicMock, mock_stop: MagicMock, tmp_path: Path
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()
    dummy_proc = MagicMock()
    mock_play.return_value = dummy_proc
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
        mock_stop.assert_called_with(dummy_proc)


@patch("a2ts.speaker_review.stop_audio_playback")
@patch("a2ts.speaker_review.play_audio_clip_async")
def test_run_interactive_review_manual_play_and_replay(
    mock_play: MagicMock, mock_stop: MagicMock, tmp_path: Path
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()
    dummy_proc = MagicMock()
    mock_play.return_value = dummy_proc
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
        assert mock_stop.call_count >= 2


@patch("a2ts.speaker_review.stop_audio_playback")
@patch("a2ts.speaker_review.play_audio_clip_async")
def test_run_interactive_review_wide_playback(
    mock_play: MagicMock, mock_stop: MagicMock, tmp_path: Path
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()
    dummy_proc = MagicMock()
    mock_play.return_value = dummy_proc
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
        mock_stop.assert_called_with(dummy_proc)


@patch("a2ts.speaker_review.stop_audio_playback")
@patch("a2ts.speaker_review.play_audio_clip_async")
def test_run_interactive_review_time_slice_audio(
    mock_play: MagicMock, mock_stop: MagicMock, tmp_path: Path
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()
    dummy_proc = MagicMock()
    mock_play.return_value = dummy_proc
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
        mock_stop.assert_called_with(dummy_proc)


def test_run_interactive_review_quit_early() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=2.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Turn 0",
        ),
        AlignedTurn(
            turn_id=1,
            start=2.0,
            end=4.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="Turn 1",
        ),
        AlignedTurn(
            turn_id=2,
            start=4.0,
            end=6.0,
            speaker="SPEAKER_02",
            cluster_id="SPEAKER_02",
            text="Turn 2",
        ),
    ]
    candidates = ["Alice", "Bob", "Charlie"]
    # User assigns Alice to SPEAKER_00, then types 'q' on SPEAKER_01
    prompt_inputs = ["1", "q"]

    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        mapping = run_interactive_review(turns, candidates)
        assert mapping.cluster_defaults == {"SPEAKER_00": "Alice"}
        # Stopped immediately on 'q', never asked for SPEAKER_02
        assert mock_ask.call_count == 2


def test_run_interactive_review_filter_clusters() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=2.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Turn 0",
        ),
        AlignedTurn(
            turn_id=1,
            start=2.0,
            end=4.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="Turn 1",
        ),
    ]
    candidates = ["Alice", "Bob"]
    prompt_inputs = ["2"]

    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        mapping = run_interactive_review(
            turns, candidates, filter_clusters=["SPEAKER_01"]
        )
        assert mapping.cluster_defaults == {"SPEAKER_01": "Bob"}
        assert mock_ask.call_count == 1


def test_run_interactive_review_unassigned_only() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=2.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Turn 0",
        ),
        AlignedTurn(
            turn_id=1,
            start=2.0,
            end=4.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="Turn 1",
        ),
    ]
    candidates = ["Alice", "Bob"]
    existing = SpeakersMapping(cluster_defaults={"SPEAKER_00": "Alice"})
    prompt_inputs = ["2"]

    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        mapping = run_interactive_review(
            turns, candidates, existing_mapping=existing, unassigned_only=True
        )
        assert mapping.cluster_defaults == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
        # Only prompted for unassigned SPEAKER_01
        assert mock_ask.call_count == 1


def test_run_interactive_review_min_turns_and_min_duration() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=10.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Turn 0",
        ),
        AlignedTurn(
            turn_id=1,
            start=10.0,
            end=20.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Turn 1",
        ),
        AlignedTurn(
            turn_id=2,
            start=20.0,
            end=21.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="Noise",
        ),
    ]
    candidates = ["Alice", "Bob"]
    prompt_inputs = ["1"]

    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        # min_turns=2 skips SPEAKER_01 which has only 1 turn
        mapping = run_interactive_review(turns, candidates, min_turns=2)
        assert mapping.cluster_defaults == {"SPEAKER_00": "Alice"}
        assert mock_ask.call_count == 1


def test_run_interactive_review_filter_slice() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=2.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="Turn 0",
            time_slice_id=0,
        ),
        AlignedTurn(
            turn_id=1,
            start=2.0,
            end=4.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="Turn 1",
            time_slice_id=1,
        ),
    ]
    candidates = ["Alice", "Bob"]
    prompt_inputs = ["2"]

    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs) as mock_ask:
        mapping = run_interactive_review(turns, candidates, filter_slice=1)
        assert mapping.cluster_defaults == {"SPEAKER_01": "Bob"}
        assert mock_ask.call_count == 1


def test_resolve_speaker_for_turn_precedence() -> None:
    # 1. Turn override beats everything
    mapping1 = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Alice"},
        slice_overrides={"1": {"SPEAKER_00": "Bob"}},
        turn_overrides={42: "Charlie"},
    )
    assert (
        resolve_speaker_for_turn(
            turn_id=42,
            cluster_id="SPEAKER_00",
            time_slice_id=1,
            mapping=mapping1,
            fallback_speaker="Fallback",
        )
        == "Charlie"
    )

    # 2. Slice override beats cluster default and fallback
    mapping2 = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Alice"},
        slice_overrides={"1": {"SPEAKER_00": "Bob"}},
    )
    assert (
        resolve_speaker_for_turn(
            turn_id=10,
            cluster_id="SPEAKER_00",
            time_slice_id=1,
            mapping=mapping2,
            fallback_speaker="Fallback",
        )
        == "Bob"
    )

    # 3. Cluster default beats fallback
    mapping3 = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Alice"},
        slice_overrides={"1": {"SPEAKER_00": "Bob"}},
    )
    # Different time slice (slice 2) -> falls back to cluster default
    assert (
        resolve_speaker_for_turn(
            turn_id=10,
            cluster_id="SPEAKER_00",
            time_slice_id=2,
            mapping=mapping3,
            fallback_speaker="Fallback",
        )
        == "Alice"
    )

    # 4. Fallback speaker used when no override/default and fallback != cluster_id
    empty_mapping = SpeakersMapping()
    assert (
        resolve_speaker_for_turn(
            turn_id=10,
            cluster_id="SPEAKER_00",
            time_slice_id=0,
            mapping=empty_mapping,
            fallback_speaker="ExistingSpeaker",
        )
        == "ExistingSpeaker"
    )

    # 5. Raw cluster ID returned when no override/default and fallback is None or equals cluster_id
    assert (
        resolve_speaker_for_turn(
            turn_id=10,
            cluster_id="SPEAKER_00",
            time_slice_id=0,
            mapping=empty_mapping,
            fallback_speaker=None,
        )
        == "SPEAKER_00"
    )
    assert (
        resolve_speaker_for_turn(
            turn_id=10,
            cluster_id="SPEAKER_00",
            time_slice_id=0,
            mapping=empty_mapping,
            fallback_speaker="SPEAKER_00",
        )
        == "SPEAKER_00"
    )


def test_run_interactive_review_saves_progress_incrementally(tmp_path: Path) -> None:
    mapping_path = tmp_path / "speakers_mapping.json"
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            cluster_id="SPEAKER_00",
            text="First speaker turn",
        ),
        AlignedTurn(
            turn_id=1,
            start=6.0,
            end=10.0,
            speaker="SPEAKER_01",
            cluster_id="SPEAKER_01",
            text="Second speaker turn",
        ),
    ]
    candidates = ["Alice", "Bob"]

    prompt_inputs = ["1", KeyboardInterrupt()]

    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs):
        try:
            run_interactive_review(turns, candidates, mapping_path=mapping_path)
        except KeyboardInterrupt:
            pass

    assert mapping_path.is_file()
    saved = load_speakers_mapping(mapping_path)
    assert saved.cluster_defaults.get("SPEAKER_00") == "Alice"


def test_save_speakers_mapping_argument_order(tmp_path: Path) -> None:
    mapping = SpeakersMapping(cluster_defaults={"SPEAKER_00": "Alice"})
    p1 = tmp_path / "mapping1.json"
    p2 = tmp_path / "mapping2.json"
    save_speakers_mapping(mapping, p1)
    save_speakers_mapping(p2, mapping)
    assert load_speakers_mapping(p1).cluster_defaults == {"SPEAKER_00": "Alice"}
    assert load_speakers_mapping(p2).cluster_defaults == {"SPEAKER_00": "Alice"}


def test_run_interactive_review_escapes_rich_markup() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            cluster_id="[SPEAKER_00]",
            text="Texte avec [/i] et [bold] et [yellow]balises[/yellow].",
        ),
    ]
    candidates = ["[NPC] Garde", "Alice"]
    prompt_inputs = ["1"]
    with patch("a2ts.speaker_review.Prompt.ask", side_effect=prompt_inputs):
        mapping = run_interactive_review(turns, candidates)
        assert mapping.cluster_defaults["[SPEAKER_00]"] == "[NPC] Garde"


def test_load_speakers_mapping_corrupt_json(tmp_path: Path) -> None:
    corrupt_file = tmp_path / "speakers_mapping.json"
    corrupt_file.write_text("{this is not valid json", encoding="utf-8")

    mapping = load_speakers_mapping(corrupt_file)
    assert mapping == SpeakersMapping()
    backups = list(tmp_path.glob("speakers_mapping.corrupt.*.json"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{this is not valid json"


def test_load_speakers_mapping_schema_error(tmp_path: Path) -> None:
    schema_err_file = tmp_path / "speakers_mapping.json"
    schema_err_file.write_text(
        '{"cluster_defaults": "invalid_string_not_dict"}', encoding="utf-8"
    )

    mapping = load_speakers_mapping(schema_err_file)
    assert mapping == SpeakersMapping()
    backups = list(tmp_path.glob("speakers_mapping.corrupt.*.json"))
    assert len(backups) == 1
    assert "invalid_string_not_dict" in backups[0].read_text(encoding="utf-8")


def test_load_speakers_mapping_non_utf8_bytes(tmp_path: Path) -> None:
    corrupt_file = tmp_path / "speakers_mapping.json"
    corrupt_file.write_bytes(b"\x80\x81\xff")

    mapping = load_speakers_mapping(corrupt_file)
    assert mapping == SpeakersMapping()
    backups = list(tmp_path.glob("speakers_mapping.corrupt.*.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"\x80\x81\xff"


def test_load_speakers_mapping_consecutive_corrupt(tmp_path: Path) -> None:
    corrupt_file = tmp_path / "speakers_mapping.json"
    corrupt_file.write_text("{corrupt 1", encoding="utf-8")

    with patch("time.time", return_value=1000.0):
        mapping1 = load_speakers_mapping(corrupt_file)
    assert mapping1 == SpeakersMapping()

    corrupt_file.write_text("{corrupt 2", encoding="utf-8")
    with patch("time.time", return_value=2000.0):
        mapping2 = load_speakers_mapping(corrupt_file)
    assert mapping2 == SpeakersMapping()

    backups = sorted(tmp_path.glob("speakers_mapping.corrupt.*.json"))
    assert len(backups) == 2
    assert backups[0].name == "speakers_mapping.corrupt.1000.json"
    assert backups[0].read_text(encoding="utf-8") == "{corrupt 1"
    assert backups[1].name == "speakers_mapping.corrupt.2000.json"
    assert backups[1].read_text(encoding="utf-8") == "{corrupt 2"
