"""Unit tests for Craig multi-track discovery, username extraction, and file parsers."""

from pathlib import Path

import pytest

from a2ts.craig import (
    discover_tracks,
    find_speakers_file,
    parse_info_file,
    parse_speakers_file,
    parse_track_username,
)
from a2ts.models import SpeakerInfo


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
