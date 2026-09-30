"""Unit tests for Craig multi-track discovery, username extraction, and file parsers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from a2ts.consolidator import debounce_consecutive_turns, render_markdown_transcript
from a2ts.craig import (
    build_craig_prompt,
    compute_track_provenance,
    discover_tracks,
    find_speakers_file,
    load_track_cache,
    merge_craig_tracks_to_turns,
    parse_info_file,
    parse_speakers_file,
    parse_track_username,
    save_track_cache,
    transcribe_craig_track,
)
from a2ts.models import (
    AlignedTurn,
    RawSegment,
    SpeakerInfo,
    TrackCacheProvenance,
    WordTimestamp,
)


def test_discover_tracks(tmp_path: Path) -> None:
    # Create test directory with tracks out of numerical order and with non-track files
    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()

    (rec_dir / "10-carol.flac").write_text("audio10")
    (rec_dir / "2-tessaro.flac").write_text("audio2")
    (rec_dir / "1-merrow1.flac").write_text("audio1")
    (rec_dir / "info.txt").write_text("Guild: Test")
    (rec_dir / "speakers.md").write_text("* tessaro: Le MJ")
    (rec_dir / "notes.txt").write_text("Session notes")
    (rec_dir / "random.flac").write_text("not a track")

    tracks = discover_tracks(rec_dir)

    # Must be sorted numerically by track prefix: 1, 2, 10
    assert [p.name for p in tracks] == [
        "1-merrow1.flac",
        "2-tessaro.flac",
        "10-carol.flac",
    ]

    # Non-existent dir raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        discover_tracks(tmp_path / "non_existent")

    # Empty dir returns empty list
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert discover_tracks(empty_dir) == []


def test_discover_tracks_edge_cases(tmp_path: Path) -> None:
    rec_dir = tmp_path / "edge_recording"
    rec_dir.mkdir()

    # Subdirectory ending in .flac should be ignored
    (rec_dir / "3-subdir.flac").mkdir()

    # Mixed case .FLAC
    (rec_dir / "100-user.FLAC").write_text("audio100")
    (rec_dir / "2-user.flac").write_text("audio2")

    tracks = discover_tracks(rec_dir)
    assert [p.name for p in tracks] == [
        "2-user.flac",
        "100-user.FLAC",
    ]


def test_parse_track_username() -> None:
    assert parse_track_username(Path("1-merrow1.flac")) == "merrow1"
    assert parse_track_username(Path("2-tessaro.flac")) == "tessaro"
    assert parse_track_username(Path("10-carol.flac")) == "carol"
    assert parse_track_username(Path("/path/to/craig/1-merrow1.flac")) == "merrow1"
    assert parse_track_username(Path("5-john-doe_42.flac")) == "john-doe_42"
    assert parse_track_username(Path("plain_user.flac")) == "plain_user"
    assert parse_track_username(Path("7-user.name.flac")) == "user.name"
    assert parse_track_username(Path("9-user.FLAC")) == "user"


def test_parse_speakers_file(tmp_path: Path) -> None:
    speakers_file = tmp_path / "speakers.md"
    content = """# Speakers roster for Campaign
* tessaro: Le MJ, Maître du Jeu
* merrow1: Merrow Ashdale, barde humain, surnoms: (Mimi)
- player3: Thorin Oakenshield, nain guerrier, surnoms: (Thor, Petit)
* player4: MJ
* player5: Simple Character
"""
    speakers_file.write_text(content, encoding="utf-8")

    speakers = parse_speakers_file(speakers_file)

    assert len(speakers) == 5

    # 1. GM with role
    s_tessaro = speakers["tessaro"]
    assert isinstance(s_tessaro, SpeakerInfo)
    assert s_tessaro.discord_username == "tessaro"
    assert s_tessaro.character_name == "MJ"
    assert s_tessaro.role == "Maître du Jeu"
    assert s_tessaro.is_dm is True
    assert s_tessaro.nicknames == []

    # 2. Player with role and single nickname
    s_merrow = speakers["merrow1"]
    assert s_merrow.discord_username == "merrow1"
    assert s_merrow.character_name == "Merrow Ashdale"
    assert s_merrow.role == "barde humain"
    assert s_merrow.is_dm is False
    assert s_merrow.nicknames == ["Mimi"]

    # 3. Dash bullet, role, and multiple nicknames
    s_thorin = speakers["player3"]
    assert s_thorin.discord_username == "player3"
    assert s_thorin.character_name == "Thorin Oakenshield"
    assert s_thorin.role == "nain guerrier"
    assert s_thorin.is_dm is False
    assert s_thorin.nicknames == ["Thor", "Petit"]

    # 4. Short MJ notation
    s_p4 = speakers["player4"]
    assert s_p4.character_name == "MJ"
    assert s_p4.is_dm is True
    assert s_p4.role is None

    # 5. Simple character without role
    s_p5 = speakers["player5"]
    assert s_p5.character_name == "Simple Character"
    assert s_p5.role is None
    assert s_p5.is_dm is False

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        parse_speakers_file(tmp_path / "missing_speakers.md")


def test_parse_speakers_file_edge_cases(tmp_path: Path) -> None:
    speakers_file = tmp_path / "speakers_alt.md"
    content = """
# Comments to skip
// Another comment or empty line

* gm_user: DM, Game Master
* player_a: Alice, surnoms: Aly, Ali
no_bullet: Bob
invalid line without colon
"""
    speakers_file.write_text(content, encoding="utf-8")

    speakers = parse_speakers_file(speakers_file)
    assert len(speakers) == 3

    assert speakers["gm_user"].is_dm is True
    assert speakers["gm_user"].character_name == "MJ"
    assert speakers["gm_user"].role == "Game Master"

    assert speakers["player_a"].character_name == "Alice"
    assert speakers["player_a"].nicknames == ["Aly", "Ali"]

    assert speakers["no_bullet"].character_name == "Bob"


def test_parse_info_file(tmp_path: Path) -> None:
    info_file = tmp_path / "info.txt"
    content = """Guild: Critical Role Discord (123456789)
Channel: Table Ronde (987654321)
Start time: 2026-09-29 20:15:00 UTC

Tracks:
1-merrow1: Merrow#1234 (111111111111111111)
2-tessaro: Tessaro#5678 (222222222222222222)
10-carol: Carol (333333333333333333)
"""
    info_file.write_text(content, encoding="utf-8")

    info = parse_info_file(info_file)

    assert info["guild"] == "Critical Role Discord (123456789)"
    assert info["channel"] == "Table Ronde (987654321)"
    assert info["start_time"] == "2026-09-29 20:15:00 UTC"
    assert len(info["tracks"]) == 3

    assert info["tracks"][0]["track"] == "1-merrow1"
    assert info["tracks"][0]["username"] == "merrow1"
    assert info["tracks"][0]["user_id"] == "111111111111111111"

    assert info["tracks"][1]["track"] == "2-tessaro"
    assert info["tracks"][1]["username"] == "tessaro"
    assert info["tracks"][1]["user_id"] == "222222222222222222"

    assert info["tracks"][2]["track"] == "10-carol"
    assert info["tracks"][2]["username"] == "carol"
    assert info["tracks"][2]["user_id"] == "333333333333333333"

    assert info["user_ids"]["merrow1"] == "111111111111111111"
    assert info["user_ids"]["tessaro"] == "222222222222222222"
    assert info["user_ids"]["carol"] == "333333333333333333"

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        parse_info_file(tmp_path / "missing_info.txt")


def test_parse_info_file_alternate_format(tmp_path: Path) -> None:
    info_file = tmp_path / "info_alt.txt"
    content = """Guild: Another Guild
Channel: General
Start time: 2026-01-01 12:00:00

Tracks:
1: dave#0001 (999888)
2-erin: erin
"""
    info_file.write_text(content, encoding="utf-8")

    info = parse_info_file(info_file)
    assert info["guild"] == "Another Guild"
    assert len(info["tracks"]) == 2
    assert info["tracks"][0]["username"] == "dave"
    assert info["tracks"][0]["user_id"] == "999888"
    assert info["tracks"][1]["username"] == "erin"
    assert info["tracks"][1]["user_id"] is None


def test_find_speakers_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_dir = tmp_path / "campaign"
    parent_dir.mkdir()
    rec_dir = parent_dir / "session_01"
    rec_dir.mkdir()

    # 1. Custom path exists
    custom_path = tmp_path / "custom_speakers.md"
    custom_path.write_text("* user: Char")
    assert find_speakers_file(rec_dir, custom_path=custom_path) == custom_path.resolve()

    # 2. Custom path doesn't exist or is a directory -> returns None
    assert find_speakers_file(rec_dir, custom_path=tmp_path / "nonexistent.md") is None
    dummy_dir = tmp_path / "dummy_dir"
    dummy_dir.mkdir()
    assert find_speakers_file(rec_dir, custom_path=dummy_dir) is None

    # 3. None found -> returns None
    assert find_speakers_file(rec_dir) is None

    # 4. Found in parent directory
    parent_speakers = parent_dir / "speakers.md"
    parent_speakers.write_text("* user: ParentChar")
    assert find_speakers_file(rec_dir) == parent_speakers.resolve()

    # 5. Found in recording directory (takes precedence over parent)
    rec_speakers = rec_dir / "speakers.md"
    rec_speakers.write_text("* user: RecChar")
    assert find_speakers_file(rec_dir) == rec_speakers.resolve()

    # 6. Found in current working directory if neither rec_dir nor parent has it
    rec_speakers.unlink()
    parent_speakers.unlink()
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    cwd_speakers = cwd_dir / "speakers.md"
    cwd_speakers.write_text("* user: CwdChar")
    monkeypatch.chdir(cwd_dir)
    assert find_speakers_file(rec_dir) == cwd_speakers.resolve()


def test_build_craig_prompt(tmp_path: Path) -> None:
    speakers = {
        "merrow1": SpeakerInfo(
            discord_username="merrow1",
            character_name="Merrow Ashdale",
            nicknames=["Mimi"],
        ),
        "tessaro": SpeakerInfo(
            discord_username="tessaro",
            character_name="MJ",
            is_dm=True,
        ),
        "player3": SpeakerInfo(
            discord_username="player3",
            character_name="Thorin Oakenshield",
            nicknames=["Thor", "Petit"],
        ),
    }

    # 1. Prompt without context_dir
    prompt = build_craig_prompt(speakers)
    assert "Merrow Ashdale" in prompt
    assert "Mimi" in prompt
    assert "merrow1" in prompt
    assert "MJ" in prompt
    assert "tessaro" in prompt
    assert "Thorin Oakenshield" in prompt
    assert "Thor" in prompt
    assert "Petit" in prompt
    assert "player3" in prompt

    # 2. Prompt with context_dir containing lore
    context_dir = tmp_path / "lore"
    context_dir.mkdir()
    (context_dir / "locations.md").write_text(
        "Welcome to [[Neverwinter]] and [[Waterdeep]].\n## Phandalin\n",
        encoding="utf-8",
    )
    prompt_with_lore = build_craig_prompt(speakers, context_dir=context_dir)
    assert "Merrow Ashdale" in prompt_with_lore
    assert "Neverwinter" in prompt_with_lore
    assert "Waterdeep" in prompt_with_lore
    assert "Phandalin" in prompt_with_lore

    # 3. Prompt token budget constraint
    small_prompt = build_craig_prompt(speakers, context_dir=context_dir, max_tokens=10)
    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")
    assert len(tokenizer.encode(small_prompt)) <= 10
    # Speakers should be prioritized over lore
    assert "Merrow Ashdale" in small_prompt

    # 4. Empty speakers and no context
    assert build_craig_prompt({}) == ""

    # 5. Non-existent context_dir
    non_existent = tmp_path / "does_not_exist"
    assert build_craig_prompt(speakers, context_dir=non_existent) == prompt


def test_compute_track_provenance(tmp_path: Path) -> None:
    track_file = tmp_path / "1-merrow1.flac"
    track_file.write_bytes(b"dummy audio data for testing 12345")

    prov = compute_track_provenance(
        track_path=track_file,
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="abc123hash",
    )

    assert prov.schema_version == 1
    assert prov.model_name == "large-v3"
    assert prov.compute_type == "float16"
    assert prov.prompt_hash == "abc123hash"
    assert prov.source_file_size == len(b"dummy audio data for testing 12345")
    assert prov.source_file_mtime == track_file.stat().st_mtime
    assert prov.vad_parameters == {}


def test_load_save_track_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / ".transcripts" / "1-merrow1.json"
    prov = TrackCacheProvenance(
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="abc123hash",
        source_file_size=1000,
        source_file_mtime=1234567.89,
    )
    segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=2.5,
            text="Bonjour à tous.",
            words=[
                WordTimestamp(word="Bonjour", start=0.0, end=0.8, probability=0.95),
                WordTimestamp(word="à", start=0.8, end=1.0, probability=0.99),
                WordTimestamp(word="tous.", start=1.0, end=2.5, probability=0.92),
            ],
        )
    ]

    # Cache file doesn't exist yet -> None
    assert load_track_cache(cache_path, prov) is None

    # Save cache
    save_track_cache(cache_path, prov, segments)
    assert cache_path.is_file()

    # Load cache (hit)
    loaded = load_track_cache(cache_path, prov)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].text == "Bonjour à tous."
    assert len(loaded[0].words) == 3

    # Force invalidation -> None
    assert load_track_cache(cache_path, prov, force=True) is None

    # Provenance mismatch (different prompt_hash) -> None
    diff_prompt_prov = prov.model_copy(update={"prompt_hash": "different_hash"})
    assert load_track_cache(cache_path, diff_prompt_prov) is None

    # Provenance mismatch (different model_name) -> None
    diff_model_prov = prov.model_copy(update={"model_name": "base"})
    assert load_track_cache(cache_path, diff_model_prov) is None

    # Provenance mismatch (different file size or mtime) -> None
    diff_size_prov = prov.model_copy(update={"source_file_size": 9999})
    assert load_track_cache(cache_path, diff_size_prov) is None

    diff_mtime_prov = prov.model_copy(update={"source_file_mtime": 9999999.99})
    assert load_track_cache(cache_path, diff_mtime_prov) is None

    # Corrupted cache file -> None
    cache_path.write_text("invalid json content")
    assert load_track_cache(cache_path, prov) is None


def test_transcribe_craig_track_caching(tmp_path: Path) -> None:
    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()
    track_file = rec_dir / "1-merrow1.flac"
    track_file.write_bytes(b"mock flac content")

    cache_dir = rec_dir / ".transcripts"

    prov = compute_track_provenance(
        track_path=track_file,
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="prompt_hash_1",
    )

    class MockWord:
        def __init__(self, word: str, start: float, end: float, probability: float = 1.0) -> None:
            self.word = word
            self.start = start
            self.end = end
            self.probability = probability

    class MockSegment:
        def __init__(self, start: float, end: float, text: str, words: list[MockWord]) -> None:
            self.start = start
            self.end = end
            self.text = text
            self.words = words

    mock_segments = [
        MockSegment(
            start=1.0,
            end=3.0,
            text=" Jet de perception.",
            words=[
                MockWord("Jet", 1.0, 1.5, 0.9),
                MockWord("de", 1.5, 1.8, 0.95),
                MockWord("perception.", 1.8, 3.0, 0.88),
            ],
        )
    ]

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter(mock_segments), MagicMock(duration=3.0))

    # 1. Cold run (cache miss): executes transcribe and creates cache file
    segments = transcribe_craig_track(
        model=mock_model,
        track_path=track_file,
        provenance=prov,
        cache_dir=cache_dir,
        initial_prompt="Merrow, MJ",
        force=False,
    )

    assert len(segments) == 1
    assert segments[0].text == "Jet de perception."
    assert segments[0].start == 1.0
    assert segments[0].end == 3.0
    assert len(segments[0].words) == 3
    mock_model.transcribe.assert_called_once_with(
        str(track_file),
        initial_prompt="Merrow, MJ",
        word_timestamps=True,
    )

    expected_cache_file = cache_dir / "1-merrow1.json"
    assert expected_cache_file.is_file()

    # 2. Warm run (cache hit): does not call model.transcribe again
    mock_model.transcribe.reset_mock()
    cached_segments = transcribe_craig_track(
        model=mock_model,
        track_path=track_file,
        provenance=prov,
        cache_dir=cache_dir,
        initial_prompt="Merrow, MJ",
        force=False,
    )
    assert len(cached_segments) == 1
    assert cached_segments[0].text == "Jet de perception."
    mock_model.transcribe.assert_not_called()

    # 3. Force invalidation: calls model.transcribe again even if cache exists
    mock_model.transcribe.return_value = (iter(mock_segments), MagicMock(duration=3.0))
    forced_segments = transcribe_craig_track(
        model=mock_model,
        track_path=track_file,
        provenance=prov,
        cache_dir=cache_dir,
        initial_prompt="Merrow, MJ",
        force=True,
    )
    assert len(forced_segments) == 1
    mock_model.transcribe.assert_called_once()


def test_merge_craig_tracks_to_turns() -> None:
    speakers = {
        "merrow1": SpeakerInfo(
            discord_username="merrow1",
            character_name="Merrow Ashdale",
            is_dm=False,
        ),
        "tessaro": SpeakerInfo(
            discord_username="tessaro",
            character_name="MJ",
            is_dm=True,
        ),
        "narrator": SpeakerInfo(
            discord_username="narrator",
            character_name="Conteur",
            is_dm=True,
        ),
        "carol": SpeakerInfo(
            discord_username="carol",
            character_name="Lyra",
            is_dm=False,
        ),
    }

    w1 = WordTimestamp(word="Bonjour", start=0.0, end=0.8, probability=0.9)
    w2 = WordTimestamp(word="monde.", start=0.8, end=1.5, probability=0.95)

    track_segments = [
        # Track 1: out of chronological order
        (
            "1-merrow1",
            RawSegment(id=10, start=10.0, end=12.0, text="Je fouille la pièce.", words=[]),
        ),
        (
            "1-merrow1",
            RawSegment(id=0, start=0.0, end=2.0, text="Bonjour monde.", words=[w1, w2]),
        ),
        # Track 2: GM turn
        (
            "2-tessaro",
            RawSegment(id=0, start=1.5, end=4.0, text="Que faites-vous ?", words=[]),
        ),
        # Track 3: Carol turn
        (
            "10-carol",
            RawSegment(id=0, start=0.5, end=1.0, text="Attendez !", words=[]),
        ),
        # Track 4: Narrator (DM with custom character name)
        (
            "3-narrator",
            RawSegment(id=0, start=15.0, end=18.0, text="Le vent souffle.", words=[]),
        ),
        # Track 5: Unknown speaker not in roster, no track index prefix
        (
            "stranger",
            RawSegment(id=0, start=5.0, end=6.0, text="Qui va là ?", words=[]),
        ),
    ]

    turns = merge_craig_tracks_to_turns(track_segments, speakers)

    # 1. Output length and types
    assert len(turns) == 6
    assert all(isinstance(t, AlignedTurn) for t in turns)

    # 2. Chronological ordering by (start, track_id)
    # Expected order:
    # 0.0: 1-merrow1
    # 0.5: 10-carol
    # 1.5: 2-tessaro
    # 5.0: stranger
    # 10.0: 1-merrow1
    # 15.0: 3-narrator
    assert [t.start for t in turns] == [0.0, 0.5, 1.5, 5.0, 10.0, 15.0]

    # 3. Sequential turn IDs
    assert [t.turn_id for t in turns] == [0, 1, 2, 3, 4, 5]

    # 4. Cluster IDs match original track IDs
    assert [t.cluster_id for t in turns] == [
        "1-merrow1",
        "10-carol",
        "2-tessaro",
        "stranger",
        "1-merrow1",
        "3-narrator",
    ]

    # 5. Speaker labels
    assert turns[0].speaker == "Merrow Ashdale (merrow1)"
    assert turns[1].speaker == "Lyra (carol)"
    assert turns[2].speaker == "MJ (tessaro)"
    assert turns[3].speaker == "stranger"  # fallback to username
    assert turns[4].speaker == "Merrow Ashdale (merrow1)"
    assert turns[5].speaker == "Conteur (narrator)"  # DM with custom character name

    # 6. Preserved texts and words
    assert turns[0].text == "Bonjour monde."
    assert len(turns[0].words) == 2
    assert turns[0].words[0].word == "Bonjour"
    assert turns[2].text == "Que faites-vous ?"
    assert turns[2].end == 4.0


def test_merge_craig_tracks_empty_and_edge_cases() -> None:
    # Empty inputs
    assert merge_craig_tracks_to_turns([], {}) == []

    speakers = {
        "alice": SpeakerInfo(discord_username="alice", character_name="Alice", is_dm=False),
    }

    # Same start timestamp tie-breaking by track_id
    seg_b = RawSegment(id=0, start=1.0, end=2.0, text="Track B")
    seg_a = RawSegment(id=1, start=1.0, end=2.0, text="Track A")
    turns = merge_craig_tracks_to_turns([("track_b", seg_b), ("track_a", seg_a)], {})
    assert turns[0].cluster_id == "track_a"
    assert turns[1].cluster_id == "track_b"
    assert turns[0].turn_id == 0
    assert turns[1].turn_id == 1

    # Track identifier variations: with .flac extension or case variation
    seg = RawSegment(id=0, start=0.0, end=1.0, text="Hello")
    turns_ext = merge_craig_tracks_to_turns([("1-alice.flac", seg)], speakers)
    assert turns_ext[0].speaker == "Alice (alice)"

    # Case insensitive speaker lookup
    turns_case = merge_craig_tracks_to_turns([("1-Alice", seg)], speakers)
    assert turns_case[0].speaker == "Alice (alice)"


def test_merge_craig_tracks_downstream_compatibility() -> None:
    speakers = {
        "merrow1": SpeakerInfo(
            discord_username="merrow1",
            character_name="Merrow",
            is_dm=False,
        ),
        "tessaro": SpeakerInfo(
            discord_username="tessaro",
            character_name="MJ",
            is_dm=True,
        ),
    }

    track_segments = [
        # Two consecutive turns by merrow with small gap <= 2.0s -> should debounce
        (
            "1-merrow1",
            RawSegment(id=0, start=1.0, end=2.0, text="Je lance", words=[]),
        ),
        (
            "1-merrow1",
            RawSegment(id=1, start=2.5, end=4.0, text="un dé de 20.", words=[]),
        ),
        # Next turn by GM
        (
            "2-tessaro",
            RawSegment(id=0, start=5.0, end=7.0, text="C'est réussi !", words=[]),
        ),
    ]

    raw_turns = merge_craig_tracks_to_turns(track_segments, speakers)
    debounced = debounce_consecutive_turns(raw_turns, threshold_seconds=2.0)

    # 3 turns should debounce into 2
    assert len(debounced) == 2
    assert debounced[0].speaker == "Merrow (merrow1)"
    assert debounced[0].start == 1.0
    assert debounced[0].end == 4.0
    # Debounce keeps text unchanged; normalization occurs in render_markdown_transcript
    assert debounced[0].text == "Je lance un dé de 20."

    assert debounced[1].speaker == "MJ (tessaro)"
    assert debounced[1].text == "C'est réussi !"

    # Render markdown transcript
    markdown = render_markdown_transcript(debounced)
    assert "### [00:00:01 - 00:00:04] Merrow (merrow1)" in markdown
    assert "Je lance 1d20." in markdown
    assert "### [00:00:05 - 00:00:07] MJ (tessaro)" in markdown
    assert "C'est réussi !" in markdown


