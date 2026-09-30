from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from a2ts.cli import app
from a2ts.models import RawSegment
from a2ts.refiner import (
    chunk_transcript_markdown,
    extract_turn_headers,
    refine_transcript_markdown,
    validate_refiner_chunk,
)

runner = CliRunner()


SAMPLE_MARKDOWN = """# Session 1: The Adventure Begins

Some introductory campaign notes.

### [00:00:00 - 00:00:10] Alice
Bonjour a tous, bienvenue dans cette nouvelle session de jeu de role.

### [00:00:10 - 00:00:25] Bob
Merci Alice, je suis pret avec ma fiche de personnage.

### [00:00:25 - 00:00:40] Charlie
Moi aussi, j'espere qu'on va trouver des tresors aujourd'hui.
"""


def test_extract_turn_headers() -> None:
    """Test extracting turn headers matching timestamp pattern."""
    headers = extract_turn_headers(SAMPLE_MARKDOWN)
    assert headers == [
        "### [00:00:00 - 00:00:10] Alice",
        "### [00:00:10 - 00:00:25] Bob",
        "### [00:00:25 - 00:00:40] Charlie",
    ]

    # Non-turn headers or callouts must be ignored
    mixed = """# Title
## Section
> [!NOTE] Callout
### [01:23 - 01:45] GM
Let's roll.
### Invalid Header [00:00 - 00:10]
### [00:10.500 - 00:20.750] Player (Subtext)
Another turn.
"""
    extracted = extract_turn_headers(mixed)
    assert extracted == [
        "### [01:23 - 01:45] GM",
        "### [00:10.500 - 00:20.750] Player (Subtext)",
    ]


def test_chunk_transcript_markdown_multi_turn() -> None:
    """Test chunking markdown transcript by turns while keeping frontmatter intact."""
    chunks = chunk_transcript_markdown(SAMPLE_MARKDOWN, max_turns=1)
    assert len(chunks) == 3
    # Chunk 0 retains frontmatter and includes turn 1
    assert "# Session 1: The Adventure Begins" in chunks[0]
    assert "### [00:00:00 - 00:00:10] Alice" in chunks[0]
    assert "### [00:00:10 - 00:00:25] Bob" not in chunks[0]

    # Chunk 1 includes turn 2
    assert "### [00:00:10 - 00:00:25] Bob" in chunks[1]
    assert "### [00:00:25 - 00:00:40] Charlie" not in chunks[1]

    # Chunk 2 includes turn 3
    assert "### [00:00:25 - 00:00:40] Charlie" in chunks[2]

    # Chunks of size 2
    chunks_2 = chunk_transcript_markdown(SAMPLE_MARKDOWN, max_turns=2)
    assert len(chunks_2) == 2
    assert len(extract_turn_headers(chunks_2[0])) == 2
    assert len(extract_turn_headers(chunks_2[1])) == 1


def test_chunk_transcript_markdown_edge_cases() -> None:
    """Test chunking with edge cases like no turns or max_turns >= total turns."""
    # When max_turns >= total turns, return single chunk
    chunks = chunk_transcript_markdown(SAMPLE_MARKDOWN, max_turns=10)
    assert len(chunks) == 1
    assert chunks[0] == SAMPLE_MARKDOWN

    # Text without any turns
    no_turns = "# Only notes\nNo speaker turns here."
    chunks_no_turns = chunk_transcript_markdown(no_turns, max_turns=5)
    assert len(chunks_no_turns) == 1
    assert chunks_no_turns[0] == no_turns

    # Empty text
    assert chunk_transcript_markdown("", max_turns=5) == []


def test_validate_refiner_chunk_turn_headers() -> None:
    """Test validate_refiner_chunk verifies exact turn header match."""
    orig = "### [00:00:00 - 00:00:10] Alice\nBonjour tout le monde."
    # Exact header match
    refined_ok = "### [00:00:00 - 00:00:10] Alice\nBonjour tout le monde !"
    assert validate_refiner_chunk(orig, refined_ok) is True

    # Dropped header
    refined_dropped = "Bonjour tout le monde !"
    assert validate_refiner_chunk(orig, refined_dropped) is False

    # Altered timestamp
    refined_altered_time = "### [00:00:00 - 00:00:15] Alice\nBonjour tout le monde."
    assert validate_refiner_chunk(orig, refined_altered_time) is False

    # Altered speaker
    refined_altered_spk = "### [00:00:00 - 00:00:10] Bob\nBonjour tout le monde."
    assert validate_refiner_chunk(orig, refined_altered_spk) is False


def test_validate_refiner_chunk_word_count_tolerance() -> None:
    """Test validate_refiner_chunk enforces 15% word count delta tolerance."""
    header = "### [00:00:00 - 00:00:10] Alice\n"
    # 100 words in original body
    body_words = ["mot"] * 100
    orig = header + " ".join(body_words)
    orig_total_words = len(orig.split())

    # Exactly 15% increase: ok
    added_words = int(orig_total_words * 0.15)
    refined_plus_15 = header + " ".join(body_words + ["extra"] * added_words)
    assert validate_refiner_chunk(orig, refined_plus_15) is True

    # 16% increase: rejected
    too_many_words = int(orig_total_words * 0.16) + 1
    refined_plus_16 = header + " ".join(body_words + ["extra"] * too_many_words)
    assert validate_refiner_chunk(orig, refined_plus_16) is False

    # Truncated (30% dropped words): rejected
    kept_words = int(orig_total_words * 0.70)
    refined_truncated = header + " ".join(body_words[: kept_words - len(header.split())])
    assert validate_refiner_chunk(orig, refined_truncated) is False

    # Empty text validation
    assert validate_refiner_chunk("", "") is True
    assert validate_refiner_chunk("", "non empty") is False


def test_refine_transcript_markdown_validation_fallback() -> None:
    """Test refine_transcript_markdown falls back to raw chunk when agy output fails validation."""
    raw = (
        "### [00:00:00 - 00:00:05] MJ\n"
        "Lance un des vingt pour voir si ton attaque touche le gobelin."
    )
    # Output that truncates/drops the header
    invalid_llm_output = "Lance 1d20 pour attaquer le gobelin."

    mock_proc = MagicMock()
    mock_proc.stdout = invalid_llm_output

    with (
        patch("shutil.which", return_value="/usr/local/bin/agy"),
        patch("subprocess.run", return_value=mock_proc),
    ):
        result = refine_transcript_markdown(raw)
        # Refiner must reject invalid LLM output and fall back to normalized raw
        assert "### [00:00:00 - 00:00:05] MJ" in result
        assert "1d20" in result
        assert "ton attaque touche le gobelin" in result


def test_refine_transcript_markdown_chunking_and_timeout() -> None:
    """Test refine_transcript_markdown processes multiple chunks with dynamic timeout."""
    # Create transcript with 2 turns
    raw = (
        "### [00:00:00 - 00:00:05] Alice\n"
        "Premiere replique avec un des vingt.\n\n"
        "### [00:00:05 - 00:00:10] Bob\n"
        "Deuxieme replique avec deux des six."
    )

    def fake_subprocess_run(cmd, input, **kwargs):
        # Return valid refined chunk preserving header
        chunk_lines = input.split("Voici la transcription brute:\n\n")[-1].strip()
        mock = MagicMock()
        mock.stdout = chunk_lines.replace("un des vingt", "1d20").replace("deux des six", "2d6")
        return mock

    with (
        patch("shutil.which", return_value="/usr/local/bin/agy"),
        patch("subprocess.run", side_effect=fake_subprocess_run) as mock_run,
    ):
        result = refine_transcript_markdown(raw, chunk_turns=1)
        # Should have called agy once per chunk (2 chunks)
        assert mock_run.call_count == 2
        # Dynamic timeout check: for small chunk (< 600 words), timeout is 30
        for call in mock_run.call_args_list:
            assert call.kwargs["timeout"] == 30
        assert "### [00:00:00 - 00:00:05] Alice" in result
        assert "### [00:00:05 - 00:00:10] Bob" in result
        assert "1d20" in result
        assert "2d6" in result


def test_cli_craig_writes_raw_markdown_on_refine(tmp_path: Path) -> None:
    """Test craig command writes <stem>.raw.md before refinement."""
    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()
    (rec_dir / "1-alice.flac").write_bytes(b"alice audio")

    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = [
        RawSegment(id=0, start=0.0, end=2.0, text="Hello world", words=[])
    ]

    out_file = tmp_path / "custom_transcript.md"
    expected_raw_file = tmp_path / "custom_transcript.raw.md"

    with (
        patch("a2ts.cli.get_engine", return_value=mock_engine),
        patch(
            "a2ts.cli.refine_transcript_markdown",
            return_value="### [00:00:00 - 00:00:02] alice\nRefined hello world\n",
        ) as mock_refine,
    ):
        result = runner.invoke(
            app,
            [
                "craig",
                str(rec_dir),
                "--output",
                str(out_file),
                "--refine",
                "--refine-model",
                "custom-llm",
            ],
        )

    assert result.exit_code == 0
    mock_refine.assert_called_once()
    # Console notice check
    assert "Refining transcript via agy CLI model 'custom-llm' (external LLM)..." in result.stdout
    # Raw file must exist and have unrefined content
    assert expected_raw_file.is_file()
    raw_content = expected_raw_file.read_text(encoding="utf-8")
    assert "Hello world" in raw_content
    assert "Refined hello world" not in raw_content

    # Final output must have refined content
    assert out_file.is_file()
    final_content = out_file.read_text(encoding="utf-8")
    assert "Refined hello world" in final_content


def test_cli_run_writes_raw_markdown_on_refine(tmp_path: Path) -> None:
    """Test run command writes <stem>.raw.md before refinement."""
    media_file = tmp_path / "session.mp4"
    media_file.write_bytes(b"dummy video data")

    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = [
        RawSegment(id=0, start=0.0, end=2.0, text="Unrefined run speech", words=[])
    ]

    out_file = tmp_path / "session_transcript.md"
    expected_raw_file = tmp_path / "session_transcript.raw.md"

    with (
        patch("a2ts.cli.compute_file_hash", return_value="hash123"),
        patch("a2ts.cli.extract_audio_to_wav", return_value=tmp_path / "audio.wav"),
        patch("a2ts.cli.scan_context_directory", return_value=[]),
        patch("a2ts.cli.probe_media", return_value={"format": {"duration": "2.0"}}),
        patch("a2ts.cli.get_engine", return_value=mock_engine),
        patch(
            "a2ts.cli.refine_transcript_markdown",
            return_value="### [00:00:00 - 00:00:02] SPEAKER_00\nRefined run speech\n",
        ) as mock_refine,
    ):
        result = runner.invoke(
            app,
            [
                "run",
                str(media_file),
                "--output",
                str(out_file),
                "--cache-dir",
                str(tmp_path / ".a2ts"),
                "--no-diarize",
                "--no-interactive",
                "--refine",
                "--refine-model",
                "gemini-3.8-flash-low",
            ],
        )

    assert result.exit_code == 0
    mock_refine.assert_called_once()
    # Console notice check
    assert "Refining transcript via agy CLI model 'gemini-3.8-flash-low' (external LLM)..." in result.stdout
    # Raw file must exist and have unrefined content
    assert expected_raw_file.is_file()
    raw_content = expected_raw_file.read_text(encoding="utf-8")
    assert "Unrefined run speech" in raw_content
    assert "Refined run speech" not in raw_content

    # Final output must have refined content
    assert out_file.is_file()
    final_content = out_file.read_text(encoding="utf-8")
    assert "Refined run speech" in final_content
