from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from a2ts.cli import app
from a2ts.models import ContextGlossary, RawSegment


def test_extract_glossary_prints_schema_validated_json(tmp_path: Path) -> None:
    context = tmp_path / "context"
    context.mkdir()
    (context / "meeting.md").write_text("Nexora Nexora deployment", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["extract-glossary", "--context-dir", str(context), "--language", "en"]
    )
    assert result.exit_code == 0, result.output
    glossary = ContextGlossary.model_validate_json(result.output)
    assert any(entry.term == "Nexora" for entry in glossary.entries)


def test_extract_glossary_saves_wordlist_snapshot(tmp_path: Path) -> None:
    context = tmp_path / "context"
    context.mkdir()
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("Nexora\n", encoding="utf-8")
    output = tmp_path / "glossary.json"
    result = CliRunner().invoke(
        app,
        [
            "extract-glossary",
            "--context-dir",
            str(context),
            "--vocab-file",
            str(wordlist),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    glossary = ContextGlossary.model_validate_json(output.read_text(encoding="utf-8"))
    assert any(entry.term == "Nexora" for entry in glossary.entries)


def test_extract_glossary_rejects_unsupported_language(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["extract-glossary", "--context-dir", str(tmp_path), "--language", "xx"]
    )
    assert result.exit_code == 1
    assert "language" in result.output.lower()


class _CraigInference:
    def transcribe(self, audio_path: str, **kwargs: object) -> list[RawSegment]:
        return [RawSegment(id=0, start=0.0, end=2.0, text="Nexora project", words=[])]


@pytest.mark.parametrize("refine", [False, True])
def test_craig_saves_glossary_only_when_refining(tmp_path: Path, refine: bool) -> None:
    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "1-Alice.flac").write_bytes(b"fake audio")
    context = tmp_path / "context"
    context.mkdir()
    (context / "meeting.md").write_text("Nexora Nexora", encoding="utf-8")
    arguments = [
        "craig",
        str(recording),
        "--context-dir",
        str(context),
        "--device",
        "cpu",
        "--no-rpg-normalize",
    ]
    arguments.extend(["--refine-mode", "polish"])
    arguments.append("--refine" if refine else "--no-refine")
    with (
        patch("a2ts.cli.get_engine", return_value=_CraigInference()),
        patch("shutil.which", return_value=None),
    ):
        result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    snapshot = recording / ".transcripts" / "context_glossary.json"
    assert snapshot.exists() is refine
    if refine:
        glossary = ContextGlossary.model_validate_json(
            snapshot.read_text(encoding="utf-8")
        )
        assert any(entry.term == "Nexora" for entry in glossary.entries)
        assert any(entry.term == "Alice" for entry in glossary.entries)
        assert (recording / "transcript.raw.md").exists()


class _Inference:
    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
        duration: float | None = None,
    ) -> list[RawSegment]:
        return [RawSegment(id=0, start=0.0, end=2.0, text="Nexora project", words=[])]


@pytest.mark.parametrize("refine", [False, True])
def test_run_saves_glossary_only_when_refining(tmp_path: Path, refine: bool) -> None:
    media = tmp_path / "recording.wav"
    media.write_bytes(b"fake audio")
    context = tmp_path / "context"
    context.mkdir()
    (context / "meeting.md").write_text("Nexora Nexora", encoding="utf-8")
    cache = tmp_path / "cache"
    output = tmp_path / "transcript.md"
    arguments = [
        "run",
        str(media),
        "--context-dir",
        str(context),
        "--device",
        "cpu",
        "--no-rpg-normalize",
        "--no-diarize",
        "--no-interactive",
        "--voice-profiles",
        str(tmp_path / "profiles.json"),
        "--cache-dir",
        str(cache),
        "--output",
        str(output),
        "--speakers",
        "Alice",
    ]
    arguments.extend(["--refine-mode", "polish"])
    arguments.append("--refine" if refine else "--no-refine")
    with (
        patch("a2ts.cli.create_transcriber", return_value=_Inference()),
        patch("a2ts.cli.extract_audio_to_wav", return_value=media),
        patch("a2ts.cli.probe_media", return_value={"format": {"duration": "2"}}),
        patch("shutil.which", return_value=None),
    ):
        result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    snapshots = list((cache / "sessions").glob("*/context_glossary.json"))
    assert len(snapshots) == int(refine)
    if refine:
        glossary = ContextGlossary.model_validate_json(
            snapshots[0].read_text(encoding="utf-8")
        )
        assert any(entry.term == "Nexora" for entry in glossary.entries)
        assert any(entry.term == "Alice" for entry in glossary.entries)
        assert output.with_name("transcript.raw.md").exists()


@pytest.mark.parametrize("command", ["run", "craig"])
def test_refinement_cli_documents_reconstruction_controls(command: str) -> None:
    result = CliRunner().invoke(app, [command, "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--refine-mode" in result.output
    assert "--refine-chunk-words" in result.output
    assert "--refine-timestamps" in result.output
