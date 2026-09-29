"""Craig multi-track audio discovery, username parsing, and file readers."""

import re
from pathlib import Path
from typing import Any

from a2ts.models import SpeakerInfo

TRACK_FILE_PATTERN = re.compile(r"^(\d+)-(.*)\.flac$", re.IGNORECASE)


def discover_tracks(recording_dir: Path) -> list[Path]:
    """Discover all Craig audio tracks in a recording directory.

    Tracks match `^\\d+-.*\\.flac` and are returned sorted numerically
    by their integer index prefix.
    """
    if not recording_dir.exists() or not recording_dir.is_dir():
        raise FileNotFoundError(f"Recording directory not found: {recording_dir}")

    tracks: list[tuple[int, Path]] = []
    for entry in recording_dir.iterdir():
        if not entry.is_file():
            continue
        m = TRACK_FILE_PATTERN.match(entry.name)
        if m:
            tracks.append((int(m.group(1)), entry))

    tracks.sort(key=lambda item: item[0])
    return [track[1] for track in tracks]


def parse_track_username(path: Path) -> str:
    """Extract Discord username from a Craig track filename.

    Example: `1-merrow1.flac` -> `merrow1`, `10-carol.flac` -> `carol`.
    """
    stem = path.stem
    m = re.match(r"^\d+-(.+)$", stem)
    if m:
        return m.group(1)
    return stem


def parse_speakers_file(path: Path) -> dict[str, SpeakerInfo]:
    """Parse a Markdown speakers roster file into a mapping of username to SpeakerInfo.

    Handles lines such as:
    * tessaro: Le MJ, Maître du Jeu
    * merrow1: Merrow Ashdale, barde humain, surnoms: (Mimi)
    - player3: Thorin Oakenshield, nain guerrier, surnoms: (Thor, Petit)
    """
    if not path.is_file():
        raise FileNotFoundError(f"Speakers file not found: {path}")

    content = path.read_text(encoding="utf-8")
    speakers: dict[str, SpeakerInfo] = {}

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith(("-", "*")):
            line = line[1:].strip()

        if ":" not in line:
            continue

        username_raw, desc_raw = line.split(":", 1)
        username = username_raw.strip()
        desc = desc_raw.strip()

        # Parse nicknames if present (e.g. surnoms: (Mimi, Mi))
        nicknames: list[str] = []
        nick_match = re.search(r",?\s*surnoms?:\s*(?:\(([^)]+)\)|(.+)$)", desc, re.IGNORECASE)
        if nick_match:
            raw_nicks = nick_match.group(1) or nick_match.group(2) or ""
            nicknames = [n.strip() for n in raw_nicks.split(",") if n.strip()]
            desc_clean = (desc[: nick_match.start()] + desc[nick_match.end() :]).strip()
            desc_clean = re.sub(r",\s*,", ",", desc_clean).strip(" ,")
        else:
            desc_clean = desc

        # Detect DM / GM
        is_dm = bool(
            re.search(
                r"\b(le\s+mj|mj|ma[îi]tre\s+du\s+jeu|meneur\s+de\s+jeu|dm|gm)\b",
                desc,
                re.IGNORECASE,
            )
        )

        parts = [p.strip() for p in desc_clean.split(",") if p.strip()]
        character_name = username
        role: str | None = None

        if parts:
            first_part = parts[0]
            if first_part.lower() in (
                "le mj",
                "mj",
                "dm",
                "gm",
                "maître du jeu",
                "maitre du jeu",
                "meneur de jeu",
            ):
                character_name = "MJ"
            else:
                character_name = first_part

            if len(parts) > 1:
                role = ", ".join(parts[1:])
        elif is_dm:
            character_name = "MJ"

        speakers[username] = SpeakerInfo(
            discord_username=username,
            character_name=character_name,
            role=role,
            nicknames=nicknames,
            is_dm=is_dm,
            raw_description=desc,
        )

    return speakers


def parse_info_file(path: Path) -> dict[str, Any]:
    """Parse Craig info.txt recording metadata file."""
    if not path.is_file():
        raise FileNotFoundError(f"Info file not found: {path}")

    content = path.read_text(encoding="utf-8")
    result: dict[str, Any] = {
        "guild": "",
        "channel": "",
        "start_time": "",
        "tracks": [],
        "user_ids": {},
    }

    in_tracks_section = False
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        if re.match(r"^tracks\s*:", line, re.IGNORECASE):
            in_tracks_section = True
            continue

        if not in_tracks_section:
            if ":" in line:
                k, v = line.split(":", 1)
                key_norm = k.strip().lower().replace(" ", "_")
                val = v.strip()
                result[key_norm] = val
                result[k.strip()] = val
        else:
            m = re.match(r"^(\d+(?:-[^:]+)?)\s*:\s*(.+)$", line)
            if m:
                track_id = m.group(1).strip()
                track_desc = m.group(2).strip()

                uid_match = re.search(r"\((\d+)\)$", track_desc)
                user_id = uid_match.group(1) if uid_match else None

                username = parse_track_username(Path(track_id))
                if username == track_id and uid_match:
                    user_desc = track_desc[: uid_match.start()].strip()
                    username = user_desc.split("#")[0].strip()

                track_entry = {
                    "track": track_id,
                    "username": username,
                    "user_id": user_id,
                    "description": track_desc,
                }
                result["tracks"].append(track_entry)
                if user_id:
                    result["user_ids"][username] = user_id
            else:
                result["tracks"].append({"raw": line})

    return result


def find_speakers_file(recording_dir: Path, custom_path: Path | None = None) -> Path | None:
    """Find speakers.md in custom path, recording dir, parent dir, or current working dir."""
    if custom_path is not None:
        resolved = custom_path.resolve()
        if resolved.is_file():
            return resolved
        return None

    candidates = [
        recording_dir / "speakers.md",
        recording_dir.parent / "speakers.md",
        Path.cwd() / "speakers.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None
