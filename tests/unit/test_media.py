import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from a2ts.media import (
    compute_file_hash,
    convert_to_wav,
    extract_audio_to_wav,
    get_media_duration,
    play_audio_clip,
    play_audio_clip_async,
    probe_media,
    stop_audio_playback,
)


def test_compute_file_hash(tmp_path: Path) -> None:
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"hello media content 12345")
    h1 = compute_file_hash(test_file)
    assert len(h1) == 16
    h2 = compute_file_hash(test_file)
    assert h1 == h2


def test_compute_file_hash_diff_tails(tmp_path: Path) -> None:
    f1 = tmp_path / "f1.bin"
    f2 = tmp_path / "f2.bin"
    prefix = b"X" * 65536
    f1.write_bytes(prefix + b"TAIL_1")
    f2.write_bytes(prefix + b"TAIL_2")
    assert compute_file_hash(f1) != compute_file_hash(f2)


@patch("subprocess.run")
def test_probe_media(mock_run: MagicMock, tmp_path: Path) -> None:
    test_file = tmp_path / "test.mkv"
    test_file.touch()

    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"format": {"duration": "12.34", "size": "1000"}}),
    )

    info = probe_media(test_file)
    assert float(info["format"]["duration"]) == 12.34


def test_probe_media_file_not_found(tmp_path: Path) -> None:
    missing_file = tmp_path / "non_existent.mkv"
    with pytest.raises(FileNotFoundError):
        probe_media(missing_file)


@patch("subprocess.run")
def test_extract_audio_to_wav_runs_ffmpeg(mock_run: MagicMock, tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp4"
    input_file.write_bytes(b"dummy mp4 content")
    out_dir = tmp_path / "cache"

    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=""
    )

    wav_path = extract_audio_to_wav(input_file, out_dir)
    assert wav_path.suffix == ".wav"
    assert out_dir.is_dir()
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "ffmpeg" in cmd
    assert "-ar" in cmd and "16000" in cmd
    assert "-ac" in cmd and "1" in cmd


@patch("subprocess.run")
def test_extract_audio_to_wav_cached(mock_run: MagicMock, tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp4"
    input_file.write_bytes(b"dummy mp4 content")
    out_dir = tmp_path / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_hash = compute_file_hash(input_file)
    cached_wav = out_dir / f"{file_hash}.wav"
    cached_wav.write_bytes(b"x" * 100)

    wav_path = extract_audio_to_wav(input_file, out_dir)
    assert wav_path == cached_wav
    mock_run.assert_not_called()


@patch("subprocess.run")
def test_probe_media_subprocess_error(mock_run: MagicMock, tmp_path: Path) -> None:
    test_file = tmp_path / "corrupt.mkv"
    test_file.touch()

    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd=["ffprobe"], stderr="Invalid data found when processing input"
    )

    with pytest.raises(RuntimeError) as exc_info:
        probe_media(test_file)
    assert "ffmpeg/ffprobe error: Invalid data found when processing input" in str(
        exc_info.value
    )


@patch("subprocess.run")
def test_extract_audio_to_wav_subprocess_error(
    mock_run: MagicMock, tmp_path: Path
) -> None:
    input_file = tmp_path / "corrupt.mp4"
    input_file.write_bytes(b"bad content")
    out_dir = tmp_path / "cache"

    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd=["ffmpeg"], stderr="Error while decoding stream"
    )

    with pytest.raises(RuntimeError) as exc_info:
        extract_audio_to_wav(input_file, out_dir)
    assert "ffmpeg/ffprobe error: Error while decoding stream" in str(exc_info.value)


def test_play_audio_clip_nonexistent_file(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.wav"
    assert play_audio_clip(non_existent, start=0.0, duration=2.0) is False


@patch("subprocess.run")
def test_play_audio_clip_success(mock_run: MagicMock, tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"dummy wav data")

    result = play_audio_clip(audio_file, start=12.5, duration=3.5)
    assert result is True
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffplay"
    assert "-nodisp" in cmd
    assert "-autoexit" in cmd
    assert "-ss" in cmd and "12.50" in cmd
    assert "-t" in cmd and "3.50" in cmd
    assert str(audio_file) in cmd


@patch("subprocess.run")
def test_play_audio_clip_failure(mock_run: MagicMock, tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"dummy wav data")
    mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["ffplay"])

    result = play_audio_clip(audio_file, start=0.0, duration=2.0)
    assert result is False


@patch("subprocess.run")
def test_play_audio_clip_with_padding(mock_run: MagicMock, tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"dummy wav data")

    result = play_audio_clip(
        audio_file, start=10.0, duration=3.0, pad_before=2.0, pad_after=2.0
    )
    assert result is True
    cmd = mock_run.call_args[0][0]
    assert "-ss" in cmd and "8.00" in cmd
    assert "-t" in cmd and "7.00" in cmd


def test_play_audio_clip_async_nonexistent_file(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.wav"
    assert play_audio_clip_async(non_existent, start=0.0, duration=2.0) is None


@patch("subprocess.Popen")
def test_play_audio_clip_async_success(mock_popen: MagicMock, tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"dummy wav data")
    dummy_proc = MagicMock()
    mock_popen.return_value = dummy_proc

    proc = play_audio_clip_async(
        audio_file, start=10.0, duration=3.0, pad_before=2.0, pad_after=2.0
    )
    assert proc == dummy_proc
    mock_popen.assert_called_once()
    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "ffplay"
    assert "-nodisp" in cmd
    assert "-autoexit" in cmd
    assert "-ss" in cmd and "8.00" in cmd
    assert "-t" in cmd and "7.00" in cmd


def test_stop_audio_playback_running() -> None:
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # Still running
    stop_audio_playback(mock_proc)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=0.5)


def test_stop_audio_playback_already_finished() -> None:
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0  # Finished
    stop_audio_playback(mock_proc)
    mock_proc.terminate.assert_not_called()


def test_stop_audio_playback_none() -> None:
    stop_audio_playback(None)  # Should not raise


@patch("subprocess.run")
def test_get_media_duration_success(mock_run: MagicMock, tmp_path: Path) -> None:
    test_file = tmp_path / "test.mkv"
    test_file.touch()
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"format": {"duration": "45.5", "size": "1000"}}),
    )
    assert get_media_duration(test_file) == 45.5
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs.get("timeout") == 60


@patch("subprocess.run")
def test_get_media_duration_timeout(mock_run: MagicMock, tmp_path: Path) -> None:
    test_file = tmp_path / "test.mkv"
    test_file.touch()
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=60)
    with pytest.raises(RuntimeError) as exc_info:
        get_media_duration(test_file)
    assert "Media command timed out after 60s" in str(exc_info.value)


@patch("subprocess.run")
def test_get_media_duration_missing_ffprobe(
    mock_run: MagicMock, tmp_path: Path
) -> None:
    test_file = tmp_path / "test.mkv"
    test_file.touch()
    mock_run.side_effect = FileNotFoundError("No such file or directory: 'ffprobe'")
    with pytest.raises(RuntimeError) as exc_info:
        get_media_duration(test_file)
    assert "ffmpeg/ffprobe not found in PATH. Please install ffmpeg." in str(
        exc_info.value
    )


@patch("subprocess.run")
def test_convert_to_wav_timeout(mock_run: MagicMock, tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp4"
    input_file.write_bytes(b"dummy mp4")
    out_dir = tmp_path / "cache"
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=300)
    with pytest.raises(RuntimeError) as exc_info:
        convert_to_wav(input_file, out_dir)
    assert "Media command timed out after 300s" in str(exc_info.value)
    # verify tmp file cleaned up
    assert not any(out_dir.glob("*.tmp_*"))


@patch("subprocess.run")
def test_convert_to_wav_missing_ffmpeg(mock_run: MagicMock, tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp4"
    input_file.write_bytes(b"dummy mp4")
    out_dir = tmp_path / "cache"
    mock_run.side_effect = FileNotFoundError("No such file or directory: 'ffmpeg'")
    with pytest.raises(RuntimeError) as exc_info:
        convert_to_wav(input_file, out_dir)
    assert "ffmpeg/ffprobe not found in PATH. Please install ffmpeg." in str(
        exc_info.value
    )
    # verify tmp file cleaned up
    assert not any(out_dir.glob("*.tmp_*"))
