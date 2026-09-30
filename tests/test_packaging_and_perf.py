"""Tests for packaging, Craig quality, performance optimizations, and crash safety.

Covers:
- Packaging dependencies in pyproject.toml (P1-7, P2-8)
- Craig VAD filtering, language specification, and prompt biasing (P2-5, P2-6)
- Lazy Whisper model loading in Craig pipeline (P2-6)
- Fast O(log N) bisect word-to-speaker turn alignment against ground truth (P2-10)
- Crash-safe atomic WAV audio extraction (P1-7)
- Restricted CUDA library preloading (P2-8)
- Diarizer linear-time valid_indices membership check (P2-10)
"""

import os
import sys
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from a2ts.cache import compute_prompt_hash
from a2ts.cli import app
from a2ts.craig import (
    build_craig_prompt,
    compute_track_provenance,
    save_track_cache,
    transcribe_craig_track,
)
from a2ts.media import extract_audio_to_wav
from a2ts.models import (
    AlignedTurn,
    RawSegment,
    SpeakerInfo,
    SpeakerTurn,
    TrackCacheProvenance,
    WordTimestamp,
)
from a2ts.timeline import align_words_to_speaker_turns
from a2ts.transcriber import ensure_cuda_libs

# ============================================================================
# 1. Packaging Tests (pyproject.toml)
# ============================================================================


def test_pyproject_dependencies_clean() -> None:
    """Verify pyproject.toml dependencies: no pandas, torch declared, CUDA platform markers."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml not found"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    deps: list[str] = data.get("project", {}).get("dependencies", [])

    # 1. No pandas in main dependencies
    pandas_deps = [d for d in deps if d.lower().startswith("pandas")]
    assert len(pandas_deps) == 0, f"pandas should be removed from dependencies: {pandas_deps}"

    # 2. torch>=2.2.0 declared
    torch_deps = [d for d in deps if d.lower().startswith("torch")]
    assert len(torch_deps) > 0, "torch>=2.2.0 must be declared in dependencies"
    assert any(">=2.2.0" in d for d in torch_deps), f"torch dependency must require >=2.2.0: {torch_deps}"

    # 3. Platform markers on nvidia-cublas-cu12 and nvidia-cudnn-cu12
    cublas_deps = [d for d in deps if "nvidia-cublas-cu12" in d]
    assert len(cublas_deps) == 1, "nvidia-cublas-cu12 missing"
    assert "sys_platform == 'linux'" in cublas_deps[0] or "sys_platform == \"linux\"" in cublas_deps[0]
    assert "sys_platform == 'win32'" in cublas_deps[0] or "sys_platform == \"win32\"" in cublas_deps[0]

    cudnn_deps = [d for d in deps if "nvidia-cudnn-cu12" in d]
    assert len(cudnn_deps) == 1, "nvidia-cudnn-cu12 missing"
    assert "sys_platform == 'linux'" in cudnn_deps[0] or "sys_platform == \"linux\"" in cudnn_deps[0]
    assert "sys_platform == 'win32'" in cudnn_deps[0] or "sys_platform == \"win32\"" in cudnn_deps[0]


# ============================================================================
# 2. Craig Track VAD, Language & Prompt
# ============================================================================


def test_track_cache_provenance_defaults() -> None:
    """TrackCacheProvenance must include vad_filter=True, vad_parameters={}, and language='fr'."""
    prov = TrackCacheProvenance(
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="abc",
        source_file_size=100,
        source_file_mtime=1.0,
    )
    assert prov.vad_filter is True
    assert prov.vad_parameters == {}
    assert prov.language == "fr"


def test_compute_track_provenance_includes_vad_and_language(tmp_path: Path) -> None:
    """compute_track_provenance must propagate vad_filter, language, and vad_parameters."""
    dummy_track = tmp_path / "track.flac"
    dummy_track.write_bytes(b"dummy")

    prov = compute_track_provenance(
        track_path=dummy_track,
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="abc",
        vad_parameters={"threshold": 0.5},
        vad_filter=True,
        language="fr",
    )
    assert prov.vad_filter is True
    assert prov.vad_parameters == {"threshold": 0.5}
    assert prov.language == "fr"


def test_transcribe_craig_track_passes_vad_and_language(tmp_path: Path) -> None:
    """transcribe_craig_track must pass vad_filter, language, and vad_parameters to model.transcribe."""
    track_file = tmp_path / "1-player.flac"
    track_file.write_bytes(b"audio data")
    cache_dir = tmp_path / ".transcripts"

    prov = TrackCacheProvenance(
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="hash123",
        vad_filter=True,
        vad_parameters={"min_speech_duration_ms": 250},
        language="fr",
        source_file_size=track_file.stat().st_size,
        source_file_mtime=track_file.stat().st_mtime,
    )

    mock_segment = MagicMock()
    mock_segment.id = 0
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = "Bonjour"
    mock_segment.words = [
        MagicMock(word="Bonjour", start=0.0, end=1.5, probability=0.98)
    ]

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)

    results = transcribe_craig_track(
        model=mock_model,
        track_path=track_file,
        provenance=prov,
        cache_dir=cache_dir,
        initial_prompt="Context prompt",
    )

    assert len(results) == 1
    mock_model.transcribe.assert_called_once()
    kwargs = mock_model.transcribe.call_args.kwargs
    assert kwargs.get("vad_filter") is True
    assert kwargs.get("language") == "fr"
    assert kwargs.get("vad_parameters") == {"min_speech_duration_ms": 250}
    assert kwargs.get("word_timestamps") is True
    assert kwargs.get("initial_prompt") == "Context prompt"


def test_build_craig_prompt_excludes_raw_discord_usernames() -> None:
    """build_craig_prompt must exclude raw Discord usernames to prevent Whisper bias."""
    speakers = {
        "x_dark_sasuke_x": SpeakerInfo(
            discord_username="x_dark_sasuke_x",
            character_name="Valeros",
            role="Guerrier humain",
            nicknames=["Val"],
            is_dm=False,
        ),
        "random_player": SpeakerInfo(
            discord_username="random_player",
            character_name="MJ",
            role="Le Maître du Jeu",
            nicknames=["Maître"],
            is_dm=True,
        ),
    }

    prompt = build_craig_prompt(speakers)

    # Character names and nicknames must be included
    assert "Valeros" in prompt
    assert "Val" in prompt
    assert "MJ" in prompt
    assert "Maître" in prompt

    # Raw Discord usernames must NOT be included
    assert "x_dark_sasuke_x" not in prompt
    assert "random_player" not in prompt


# ============================================================================
# 3. Lazy Whisper Loading in Craig CLI Pipeline
# ============================================================================


def test_craig_cli_lazy_whisper_model_loading(tmp_path: Path) -> None:
    """Craig command should NOT load Whisper model if all track caches hit."""
    runner = CliRunner()
    rec_dir = tmp_path / "session_craig"
    rec_dir.mkdir()

    # Create dummy track
    track_path = rec_dir / "1-player.flac"
    track_path.write_bytes(b"dummy flac data")

    # Pre-populate cache so it's a 100% cache hit
    cache_dir = rec_dir / ".transcripts"
    cache_dir.mkdir()
    cache_file = cache_dir / "1-player.json"

    empty_ctx = tmp_path / "empty_ctx"
    empty_ctx.mkdir()

    prov = compute_track_provenance(
        track_path=track_path,
        model_name="large-v3",
        compute_type="float16",
        prompt_hash=compute_prompt_hash(""),
    )
    cached_segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=1.0,
            text="Test",
            words=[WordTimestamp(word="Test", start=0.0, end=1.0)],
        )
    ]
    save_track_cache(cache_file, prov, cached_segments)

    # Run CLI command with WhisperEngine._get_model mocked
    with patch("a2ts.transcriber.WhisperEngine._get_model") as mock_get_model:
        result = runner.invoke(
            app,
            [
                "craig",
                str(rec_dir),
                "--context-dir",
                str(empty_ctx),
                "--no-rpg-normalize",
            ],
        )
        assert result.exit_code == 0, f"Command failed: {result.output}"
        # Model should NOT have been loaded because cache hit!
        mock_get_model.assert_not_called()


def test_craig_cli_loads_whisper_when_cache_misses(tmp_path: Path) -> None:
    """Craig command MUST load Whisper model when at least one track cache misses."""
    runner = CliRunner()
    rec_dir = tmp_path / "session_craig_miss"
    rec_dir.mkdir()

    track_path = rec_dir / "1-player.flac"
    track_path.write_bytes(b"dummy flac data")

    mock_segment = MagicMock()
    mock_segment.id = 0
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = "Hello"
    mock_segment.words = [MagicMock(word="Hello", start=0.0, end=1.0, probability=0.9)]

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)

    with patch("a2ts.transcriber.WhisperEngine._get_model", return_value=mock_model) as mock_get_model:
        result = runner.invoke(app, ["craig", str(rec_dir), "--no-rpg-normalize"])
        assert result.exit_code == 0, f"Command failed: {result.output}"
        mock_get_model.assert_called_once()


def test_craig_cli_wires_refine_effort(tmp_path: Path) -> None:
    """Craig command passes --refine-effort through to refine_transcript_markdown."""
    runner = CliRunner()
    rec_dir = tmp_path / "session_craig_refine"
    rec_dir.mkdir()

    track_path = rec_dir / "1-player.flac"
    track_path.write_bytes(b"dummy flac data")

    cache_dir = rec_dir / ".transcripts"
    cache_dir.mkdir()
    prov = compute_track_provenance(
        track_path=track_path,
        model_name="large-v3",
        compute_type="float16",
        prompt_hash="",
    )
    cached_segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=1.0,
            text="Test",
            words=[WordTimestamp(word="Test", start=0.0, end=1.0)],
        )
    ]
    save_track_cache(cache_dir / "1-player.json", prov, cached_segments)

    empty_ctx = tmp_path / "empty_ctx"
    empty_ctx.mkdir()

    with patch("a2ts.cli.refine_transcript_markdown", return_value="Refined text") as mock_refine:
        result = runner.invoke(
            app,
            [
                "craig",
                str(rec_dir),
                "--context-dir",
                str(empty_ctx),
                "--refine",
                "--refine-effort",
                "high",
            ],
        )
        assert result.exit_code == 0, f"Command failed: {result.output}"
        mock_refine.assert_called_once()
        assert mock_refine.call_args.kwargs.get("effort") == "high"


# ============================================================================
# 4. Fast Bisect Word Alignment (O(log N) vs Ground Truth)
# ============================================================================


def _ground_truth_align(
    segments: list[RawSegment], turns: list[SpeakerTurn]
) -> list[AlignedTurn]:
    """Linear-scan ground truth alignment for verification."""
    if not turns:
        return [
            AlignedTurn(
                turn_id=seg.id,
                start=seg.start,
                end=seg.end,
                speaker="SPEAKER_00",
                cluster_id="SPEAKER_00",
                text=seg.text,
                words=seg.words,
                time_slice_id=0,
            )
            for seg in segments
        ]

    sorted_turns = sorted(turns, key=lambda t: (t.start, t.id))
    all_words: list[WordTimestamp] = []
    for seg in segments:
        if seg.words:
            all_words.extend(seg.words)
        else:
            all_words.append(WordTimestamp(word=seg.text, start=seg.start, end=seg.end))

    words_by_turn: dict[int, list[WordTimestamp]] = {t.id: [] for t in sorted_turns}
    for word in all_words:
        mid_point = (word.start + word.end) / 2.0
        matched = None
        for t in sorted_turns:
            if t.start <= mid_point <= t.end:
                matched = t
                break
        if matched is None:
            matched = min(
                sorted_turns,
                key=lambda t: min(abs(t.start - mid_point), abs(t.end - mid_point)),
            )
        words_by_turn[matched.id].append(word)

    aligned: list[AlignedTurn] = []
    for t in sorted_turns:
        tw = words_by_turn[t.id]
        if not tw:
            continue
        aligned.append(
            AlignedTurn(
                turn_id=t.id,
                start=tw[0].start,
                end=tw[-1].end,
                speaker=t.resolved_speaker or t.cluster_id,
                cluster_id=t.cluster_id,
                text=" ".join(w.word for w in tw).strip(),
                words=tw,
                time_slice_id=t.time_slice_id,
            )
        )
    return aligned


def test_align_words_bisect_matches_ground_truth() -> None:
    """Bisect-based alignment must match ground truth exactly on synthetic sequence."""
    turns: list[SpeakerTurn] = []
    for i in range(50):
        # Turns of duration ~3s with 1s gaps
        t_start = i * 4.0
        t_end = t_start + 3.0
        turns.append(
            SpeakerTurn(
                id=i,
                start=t_start,
                end=t_end,
                cluster_id=f"SPEAKER_{i % 3:02d}",
            )
        )

    # Words: some inside turns, some in gaps, some at boundaries
    words: list[WordTimestamp] = []
    for w_idx in range(150):
        w_start = w_idx * 1.3
        w_end = w_start + 0.4
        words.append(WordTimestamp(word=f"word_{w_idx}", start=w_start, end=w_end))

    segments = [
        RawSegment(
            id=0,
            start=words[0].start,
            end=words[-1].end,
            text=" ".join(w.word for w in words),
            words=words,
        )
    ]

    expected = _ground_truth_align(segments, turns)
    actual = align_words_to_speaker_turns(segments, turns)

    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected, strict=True):
        assert act.turn_id == exp.turn_id
        assert act.start == exp.start
        assert act.end == exp.end
        assert act.speaker == exp.speaker
        assert act.cluster_id == exp.cluster_id
        assert act.text == exp.text
        assert len(act.words) == len(exp.words)


def test_align_words_bisect_edge_cases() -> None:
    """Bisect alignment edge cases: words before first turn, after last turn, empty words."""
    turns = [
        SpeakerTurn(id=10, start=10.0, end=15.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=20, start=20.0, end=25.0, cluster_id="SPEAKER_01"),
    ]

    # Word 1: before first turn (0.0 - 1.0) -> should match turn 10
    # Word 2: inside turn 10 (11.0 - 12.0) -> matches turn 10
    # Word 3: in gap (16.0 - 17.0) -> mid 16.5, closer to 15.0 (dist 1.5) than 20.0 (dist 3.5) -> turn 10
    # Word 4: in gap (18.5 - 19.5) -> mid 19.0, closer to 20.0 (dist 1.0) than 15.0 (dist 4.0) -> turn 20
    # Word 5: after last turn (30.0 - 32.0) -> should match turn 20
    words = [
        WordTimestamp(word="before", start=0.0, end=1.0),
        WordTimestamp(word="inside1", start=11.0, end=12.0),
        WordTimestamp(word="gap_left", start=16.0, end=17.0),
        WordTimestamp(word="gap_right", start=18.5, end=19.5),
        WordTimestamp(word="after", start=30.0, end=32.0),
    ]

    segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=32.0,
            text="before inside1 gap_left gap_right after",
            words=words,
        )
    ]

    aligned = align_words_to_speaker_turns(segments, turns)
    assert len(aligned) == 2

    # Turn 10 should get "before", "inside1", "gap_left"
    assert aligned[0].turn_id == 10
    assert [w.word for w in aligned[0].words] == ["before", "inside1", "gap_left"]

    # Turn 20 should get "gap_right", "after"
    assert aligned[1].turn_id == 20
    assert [w.word for w in aligned[1].words] == ["gap_right", "after"]


# ============================================================================
# 5. Crash-Safe Atomic WAV Extraction
# ============================================================================


def test_extract_audio_to_wav_atomic_replace(tmp_path: Path) -> None:
    """extract_audio_to_wav must write to a temporary file before renaming atomically."""
    media_file = tmp_path / "input.mp4"
    media_file.write_bytes(b"dummy mp4 video bytes")

    output_dir = tmp_path / "wav_output"
    output_dir.mkdir()

    # Spy on subprocess.run and os.replace
    recorded_ffmpeg_cmd: list[str] = []
    original_replace = os.replace
    replace_calls: list[tuple[str, str]] = []

    def mock_run(cmd: list[str], *args: object, **kwargs: object) -> MagicMock:
        if cmd[0] == "ffmpeg":
            recorded_ffmpeg_cmd.extend(cmd)
            # Create the output file that ffmpeg was asked to create
            out_target = Path(cmd[-1])
            out_target.write_bytes(b"RIFF" + b"\x00" * 40 + b"WAVEfmt " + b"\x00" * 100)
        return MagicMock(returncode=0)

    def mock_replace(src: str | Path, dst: str | Path) -> None:
        replace_calls.append((str(src), str(dst)))
        original_replace(src, dst)

    with (
        patch("subprocess.run", side_effect=mock_run),
        patch("os.replace", side_effect=mock_replace),
    ):
        wav_path = extract_audio_to_wav(media_file, output_dir)

    # ffmpeg destination must NOT have been the final wav_path directly
    assert len(recorded_ffmpeg_cmd) > 0
    ffmpeg_target = recorded_ffmpeg_cmd[-1]
    assert ffmpeg_target != str(wav_path), "ffmpeg should write to a temporary file first"
    assert f"tmp_{os.getpid()}" in ffmpeg_target or "tmp" in ffmpeg_target

    # os.replace must have moved tmp file to final wav_path
    assert len(replace_calls) == 1
    assert replace_calls[0][0] == ffmpeg_target
    assert replace_calls[0][1] == str(wav_path)
    assert wav_path.is_file()


def test_extract_audio_to_wav_cleans_tmp_on_failure(tmp_path: Path) -> None:
    """If ffmpeg fails, any temporary WAV file must be unlinked."""
    media_file = tmp_path / "corrupt.mp4"
    media_file.write_bytes(b"corrupt")

    output_dir = tmp_path / "wav_output"

    def mock_failing_ffmpeg(cmd: list[str], *args: object, **kwargs: object) -> None:
        if cmd[0] == "ffmpeg":
            out_target = Path(cmd[-1])
            out_target.write_bytes(b"partial invalid audio")
            import subprocess

            raise subprocess.CalledProcessError(1, cmd, stderr="ffmpeg error")

    with (
        patch("subprocess.run", side_effect=mock_failing_ffmpeg),
        pytest.raises(RuntimeError, match="ffmpeg/ffprobe error"),
    ):
        extract_audio_to_wav(media_file, output_dir)

    # Output directory should have no remaining .tmp files
    tmp_files = list(output_dir.glob("*.tmp*"))
    assert tmp_files == [], f"Temporary files were not cleaned up: {tmp_files}"


# ============================================================================
# 6. Restricted CUDA Library Preloading
# ============================================================================


def test_ensure_cuda_libs_preloads_only_cublas_and_cudnn(tmp_path: Path) -> None:
    """ensure_cuda_libs must only preload libcublas and libcudnn, ignoring cufft/cusparse/nccl."""
    # Set up simulated nvidia directory in sys.path
    fake_sys_path = tmp_path / "site-packages"
    nvidia_dir = fake_sys_path / "nvidia"
    cublas_lib = nvidia_dir / "cublas" / "lib"
    cudnn_lib = nvidia_dir / "cudnn" / "lib"
    cufft_lib = nvidia_dir / "cufft" / "lib"
    cusparse_lib = nvidia_dir / "cusparse" / "lib"

    for d in (cublas_lib, cudnn_lib, cufft_lib, cusparse_lib):
        d.mkdir(parents=True)

    (cublas_lib / "libcublas.so.12").write_text("")
    (cudnn_lib / "libcudnn.so.9").write_text("")
    (cufft_lib / "libcufft.so.11").write_text("")
    (cusparse_lib / "libcusparse.so.12").write_text("")

    loaded_so: list[str] = []

    def mock_cdll(path: str, *args: object, **kwargs: object) -> MagicMock:
        loaded_so.append(Path(path).name)
        return MagicMock()

    with (
        patch.object(sys, "path", [str(fake_sys_path)]),
        patch("ctypes.CDLL", side_effect=mock_cdll),
    ):
        ensure_cuda_libs()

    # Must contain cublas and cudnn
    assert any("libcublas" in s for s in loaded_so)
    assert any("libcudnn" in s for s in loaded_so)

    # Must NOT contain cufft or cusparse
    assert not any("libcufft" in s for s in loaded_so), f"libcufft was preloaded: {loaded_so}"
    assert not any("libcusparse" in s for s in loaded_so), f"libcusparse was preloaded: {loaded_so}"


# ============================================================================
# 7. Diarizer valid_indices Set Lookup
# ============================================================================


def test_diarizer_valid_indices_uses_set() -> None:
    """diarize_segments must use set(valid_indices) for membership checks."""
    import inspect

    from a2ts import diarizer

    source = inspect.getsource(diarizer.diarize_segments)
    # Source must not check 'i not in valid_indices' when valid_indices is a list
    assert "set(valid_indices)" in source or "valid_indices_set = set(valid_indices)" in source or "valid_set = set(valid_indices)" in source
