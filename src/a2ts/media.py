"""Media inspection and audio extraction utilities using ffprobe and ffmpeg."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast


def compute_file_hash(media_path: Path, chunk_size: int = 65536) -> str:
    """Compute 16-character SHA-256 digest of entire media file via streaming chunks."""
    hasher = hashlib.sha256()
    with open(media_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
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
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=60
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg/ffprobe not found in PATH. Please install ffmpeg."
        ) from None
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Media command timed out after {exc.timeout}s: {exc.cmd}"
        ) from exc
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.strip() if e.stderr else ""
        raise RuntimeError(f"ffmpeg/ffprobe error: {stderr_msg}") from e
    return cast(dict[str, Any], json.loads(proc.stdout))


def get_media_duration(media_path: Path) -> float:
    """Return media duration in seconds via probe_media."""
    info = probe_media(media_path)
    format_info = info.get("format", {})
    duration_str = format_info.get("duration")
    if duration_str is None:
        raise RuntimeError(f"Could not determine duration for {media_path}")
    return float(duration_str)


def extract_audio_to_wav(media_path: Path, output_dir: Path) -> Path:
    """Extract audio stream to 16kHz mono 16-bit PCM WAV cached by file hash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_hash = compute_file_hash(media_path)
    wav_path = output_dir / f"{file_hash}.wav"

    if wav_path.is_file() and wav_path.stat().st_size > 44:
        return wav_path

    tmp_path = wav_path.with_name(f"{wav_path.stem}.tmp_{os.getpid()}.wav")
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
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        if tmp_path.exists():
            os.replace(tmp_path, wav_path)
    except FileNotFoundError:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            "ffmpeg/ffprobe not found in PATH. Please install ffmpeg."
        ) from None
    except subprocess.TimeoutExpired as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Media command timed out after {exc.timeout}s: {exc.cmd}"
        ) from exc
    except subprocess.CalledProcessError as e:
        if tmp_path.exists():
            tmp_path.unlink()
        stderr_msg = e.stderr.strip() if e.stderr else ""
        raise RuntimeError(f"ffmpeg/ffprobe error: {stderr_msg}") from e
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return wav_path


def convert_to_wav(media_path: Path, output_dir: Path) -> Path:
    """Convert/extract media file to 16kHz mono WAV format (cached)."""
    return extract_audio_to_wav(media_path, output_dir)


def play_audio_clip_async(
    audio_path: Path,
    start: float,
    duration: float = 4.0,
    pad_before: float = 0.0,
    pad_after: float = 0.0,
    player: str | None = None,
) -> subprocess.Popen[bytes] | None:
    """Launch playback of an audio snippet asynchronously in the background.

    Returns the subprocess.Popen object if started, or None if failed.
    """
    if not audio_path.is_file():
        return None

    actual_start = max(0.0, start - pad_before)
    actual_duration = max(0.1, (start + duration + pad_after) - actual_start)

    player_cmd = player or "ffplay"
    cmd = [
        player_cmd,
        "-nodisp",
        "-autoexit",
        "-ss",
        f"{actual_start:.2f}",
        "-t",
        f"{actual_duration:.2f}",
        str(audio_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def stop_audio_playback(proc: subprocess.Popen[bytes] | None) -> None:
    """Safely terminate a background audio playback process if running."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except (subprocess.SubprocessError, OSError):
        pass


def play_audio_clip(
    audio_path: Path,
    start: float,
    duration: float = 4.0,
    pad_before: float = 0.0,
    pad_after: float = 0.0,
    player: str | None = None,
) -> bool:
    """Play a short audio snippet via ffplay (or specified player).

    Returns True if playback completed cleanly, False otherwise.
    """
    if not audio_path.is_file():
        return False

    actual_start = max(0.0, start - pad_before)
    actual_duration = max(0.1, (start + duration + pad_after) - actual_start)

    player_cmd = player or "ffplay"
    cmd = [
        player_cmd,
        "-nodisp",
        "-autoexit",
        "-ss",
        f"{actual_start:.2f}",
        "-t",
        f"{actual_duration:.2f}",
        str(audio_path),
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=actual_duration + 5.0,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False
