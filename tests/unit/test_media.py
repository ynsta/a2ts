import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from a2ts.media import compute_file_hash, extract_audio_to_wav, probe_media


def test_compute_file_hash(tmp_path: Path) -> None:
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"hello media content 12345")
    h1 = compute_file_hash(test_file)
    assert len(h1) == 16
    h2 = compute_file_hash(test_file)
    assert h1 == h2


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
