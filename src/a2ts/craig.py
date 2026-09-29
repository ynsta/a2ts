"""Craig multi-track audio discovery, username parsing, and file readers."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from a2ts.cache import atomic_write_text
from a2ts.models import (
    AlignedTurn,
    EntityRecord,
    RawSegment,
    SpeakerInfo,
    TrackCacheFile,
    TrackCacheProvenance,
    WordTimestamp,
)
from a2ts.vocab import build_biasing_prompt, scan_context_directory

logger = logging.getLogger(__name__)

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


def compute_track_provenance(
    track_path: Path,
    model_name: str,
    compute_type: str,
    prompt_hash: str,
    vad_parameters: dict[str, Any] | None = None,
) -> TrackCacheProvenance:
    """Compute provenance metadata for an audio track file."""
    stat = track_path.stat()
    return TrackCacheProvenance(
        schema_version=1,
        model_name=model_name,
        compute_type=compute_type,
        prompt_hash=prompt_hash,
        vad_parameters=vad_parameters if vad_parameters is not None else {},
        source_file_size=stat.st_size,
        source_file_mtime=stat.st_mtime,
    )


def build_craig_prompt(
    speakers: dict[str, SpeakerInfo],
    context_dir: Path | None = None,
    max_tokens: int = 220,
) -> str:
    """Assemble token-budgeted prompt containing speaker/character names and Obsidian lore terms."""
    records: list[EntityRecord] = []
    for speaker in speakers.values():
        if speaker.character_name:
            records.append(EntityRecord(name=speaker.character_name, kind="character"))
        for nick in speaker.nicknames:
            if nick:
                records.append(EntityRecord(name=nick, kind="nickname"))
        if speaker.discord_username:
            records.append(EntityRecord(name=speaker.discord_username, kind="username"))

    if context_dir is not None and context_dir.is_dir():
        records.extend(scan_context_directory(context_dir))

    return build_biasing_prompt(records, max_tokens=max_tokens)


def load_track_cache(
    cache_path: Path,
    expected_provenance: TrackCacheProvenance,
    force: bool = False,
) -> list[RawSegment] | None:
    """Load cached segments if provenance strictly matches expected settings, or None if miss/stale."""
    if force or not cache_path.is_file():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "provenance" not in data:
            return None
        cached = TrackCacheFile.model_validate(data)
        p = cached.provenance
        if (
            p.schema_version == expected_provenance.schema_version
            and p.model_name == expected_provenance.model_name
            and p.compute_type == expected_provenance.compute_type
            and p.prompt_hash == expected_provenance.prompt_hash
            and p.source_file_size == expected_provenance.source_file_size
            and abs(p.source_file_mtime - expected_provenance.source_file_mtime) < 1e-4
            and p.vad_parameters == expected_provenance.vad_parameters
        ):
            return cached.segments
        return None
    except (json.JSONDecodeError, OSError, ValueError, KeyError) as exc:
        logger.debug("Failed reading track cache %s: %s", cache_path, exc)
        return None


def save_track_cache(
    cache_path: Path,
    provenance: TrackCacheProvenance,
    segments: list[RawSegment],
) -> None:
    """Serialize track transcription provenance and segments atomically to JSON cache."""
    cache_file = TrackCacheFile(provenance=provenance, segments=segments)
    atomic_write_text(cache_path, cache_file.model_dump_json(indent=2))


def transcribe_craig_track(
    model: Any,
    track_path: Path,
    provenance: TrackCacheProvenance,
    cache_dir: Path,
    initial_prompt: str | None = None,
    force: bool = False,
) -> list[RawSegment]:
    """Transcribe single Craig audio track using Faster-Whisper with JSON caching."""
    cache_path = cache_dir / f"{track_path.stem}.json"
    cached_segments = load_track_cache(cache_path, provenance, force=force)
    if cached_segments is not None:
        return cached_segments

    transcribe_out = model.transcribe(
        str(track_path),
        initial_prompt=initial_prompt,
        word_timestamps=True,
    )
    if isinstance(transcribe_out, tuple):
        segments_gen, _ = transcribe_out
    else:
        segments_gen = transcribe_out

    results: list[RawSegment] = []
    for i, s in enumerate(segments_gen):
        if isinstance(s, RawSegment):
            results.append(s)
            continue

        if isinstance(s, dict):
            start = float(s["start"])
            end = float(s["end"])
            text = str(s["text"]).strip()
            raw_words = s.get("words") or []
            seg_id = int(s.get("id", i))
        else:
            start = float(getattr(s, "start", 0.0))
            end = float(getattr(s, "end", 0.0))
            text = str(getattr(s, "text", "")).strip()
            raw_words = getattr(s, "words", None) or []
            seg_id = int(getattr(s, "id", i))

        words: list[WordTimestamp] = []
        for w in raw_words:
            if isinstance(w, WordTimestamp):
                words.append(w)
            elif isinstance(w, dict):
                words.append(
                    WordTimestamp(
                        word=str(w.get("word", "")).strip(),
                        start=float(w.get("start", start)),
                        end=float(w.get("end", end)),
                        probability=float(w.get("probability", 1.0)),
                    )
                )
            else:
                words.append(
                    WordTimestamp(
                        word=str(getattr(w, "word", "")).strip(),
                        start=float(getattr(w, "start", start)),
                        end=float(getattr(w, "end", end)),
                        probability=float(getattr(w, "probability", 1.0)),
                    )
                )

        results.append(
            RawSegment(
                id=seg_id,
                start=start,
                end=end,
                text=text,
                words=words,
            )
        )

    save_track_cache(cache_path, provenance, results)
    return results


def merge_craig_tracks_to_turns(
    track_segments: list[tuple[str, RawSegment]],
    speakers: dict[str, SpeakerInfo],
) -> list[AlignedTurn]:
    """Interleave and align multi-track Craig segments into chronological speaker turns.

    Segments are sorted chronologically by (seg.start, track_id).
    Speaker names are resolved from speakers mapping or fallback to track username:
    - If found and is_dm: 'MJ (discord_username)' or '{character_name} (discord_username)'
    - If found and not is_dm: '{character_name} (discord_username)'
    - If not found: 'username'
    """
    sorted_items = sorted(track_segments, key=lambda item: (item[1].start, item[0]))

    aligned_turns: list[AlignedTurn] = []
    for turn_id, (track_id, seg) in enumerate(sorted_items):
        username = parse_track_username(Path(track_id))
        speaker_info = speakers.get(username)
        if speaker_info is None:
            username_lower = username.lower()
            speaker_info = next(
                (sp for k, sp in speakers.items() if k.lower() == username_lower),
                None,
            )

        if speaker_info is not None:
            if speaker_info.is_dm:
                char_name = speaker_info.character_name if speaker_info.character_name else "MJ"
                label = f"{char_name} ({speaker_info.discord_username or username})"
            else:
                char_name = speaker_info.character_name or speaker_info.discord_username or username
                label = f"{char_name} ({speaker_info.discord_username or username})"
        else:
            label = username

        aligned_turns.append(
            AlignedTurn(
                turn_id=turn_id,
                start=seg.start,
                end=seg.end,
                speaker=label,
                cluster_id=track_id,
                text=seg.text,
                words=seg.words,
            )
        )

    return aligned_turns


