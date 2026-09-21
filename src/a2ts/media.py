"""Media inspection and audio extraction utilities using ffprobe and ffmpeg."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast


def compute_file_hash(media_path: Path) -> str:
    """Compute 16-character SHA-256 digest of media file header/size."""
    hasher = hashlib.sha256()
    size = media_path.stat().st_size
    hasher.update(str(size).encode("utf-8"))
    with open(media_path, "rb") as f:
        hasher.update(f.read(65536))
    return hasher.hexdigest()[:16]


def probe_media(media_path: Path) -> dict[str, Any]:
    """Run ffprobe to inspect media container and stream information."""
    if not media_path.is_file():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_name,codec_type,sample_rate,channels",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.strip() if e.stderr else ""
        raise RuntimeError(f"ffmpeg/ffprobe error: {stderr_msg}") from e
    return cast(dict[str, Any], json.loads(proc.stdout))


def extract_audio_to_wav(media_path: Path, output_dir: Path) -> Path:
    """Extract audio stream to 16kHz mono 16-bit PCM WAV cached by file hash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_hash = compute_file_hash(media_path)
    wav_path = output_dir / f"{file_hash}.wav"

    if wav_path.is_file() and wav_path.stat().st_size > 44:
        return wav_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.strip() if e.stderr else ""
        raise RuntimeError(f"ffmpeg/ffprobe error: {stderr_msg}") from e
    return wav_path


def play_audio_clip(
    audio_path: Path,
    start: float,
    duration: float = 4.0,
    player: str | None = None,
) -> bool:
    """Play a short audio snippet via ffplay (or specified player).

    Returns True if playback completed cleanly, False otherwise.
    """
    if not audio_path.is_file():
        return False

    player_cmd = player or "ffplay"
    cmd = [
        player_cmd,
        "-nodisp",
        "-autoexit",
        "-ss",
        f"{max(0.0, start):.2f}",
        "-t",
        f"{max(0.1, duration):.2f}",
        str(audio_path),
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=duration + 5.0,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False
