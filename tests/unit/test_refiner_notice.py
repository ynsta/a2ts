import subprocess
from typing import Any
from unittest.mock import patch

import pytest

from a2ts.consolidator import render_markdown_transcript
from a2ts.models import AlignedTurn
from a2ts.refiner import refine_transcript_markdown, validate_refiner_chunk

NOTICE = "> [!WARNING] Some speaker attributions are uncertain. Review speaker assignments against the audio."
LEGACY = "> [!WARNING] Speaker attribution uncertain: multiple candidate speakers."
HEADER = "### [00:00:00 - 00:00:01] A"


def test_renderer_emits_one_notice_before_uncertain_turns() -> None:
    turns = [
        AlignedTurn(
            turn_id=i,
            cluster_id="A",
            start=i,
            end=i + 1,
            speaker="A",
            text="Bonjour",
            words=[],
            speaker_uncertain=True,
        )
        for i in range(3)
    ]
    rendered = render_markdown_transcript(turns)
    assert rendered.startswith(NOTICE + "\n\n###")
    assert rendered.count(NOTICE) == 1
    assert LEGACY not in rendered
    assert all(turn.speaker_uncertain for turn in turns)
    assert NOTICE not in render_markdown_transcript(
        [turns[0].model_copy(update={"speaker_uncertain": False})]
    )


@pytest.mark.parametrize("failure", [False, True])
def test_refiner_keeps_one_notice_outside_multichunk_model_input(failure: bool) -> None:
    raw = NOTICE + "\n\n" + HEADER + "\nBonjour.\n\n### [00:00:02 - 00:00:03] B\nSalut."
    prompts: list[str] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        input = next(
            arg.removeprefix("--print=") for arg in cmd if arg.startswith("--print=")
        )
        prompts.append(input)
        assert NOTICE not in input
        assert "Preserve every turn header verbatim" in input
        assert "protected markers" in input
        assert "> [!NOTE] Hors-jeu / Discussion" in input
        if failure:
            raise subprocess.TimeoutExpired(cmd, 60)
        chunk = input.split("Raw transcript:\n\n", 1)[1]
        return subprocess.CompletedProcess(
            cmd, 0, stdout=chunk.replace("Bonjour.", "Bonjour !")
        )

    with (
        patch("shutil.which", return_value="agy"),
        patch("subprocess.run", side_effect=run),
    ):
        result = refine_transcript_markdown(raw, chunk_turns=1)
    assert len(prompts) == 2
    assert result.startswith(NOTICE + "\n\n" + HEADER)
    assert result.count(NOTICE) == 1
    assert ("Bonjour." if failure else "Bonjour !") in result
    assert "Salut." in result


def test_validator_accepts_unchanged_warning_heavy_transcript() -> None:
    original = "\n\n".join(
        f"### [00:00:{i:02d} - 00:01:00] A\n{LEGACY}\n> [!NOTE] Hors-jeu / Discussion\n> Bonjour."
        for i in range(25)
    )
    assert validate_refiner_chunk(original, original)


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("", "empty output"),
        ("Bonjour.", "turn headers changed"),
        (
            HEADER + "\n" + "mot " * 100,
            "word count mismatch: original=6 refined=105 tolerance=5",
        ),
        (HEADER + "\nBonjour.", "legacy uncertainty marker changed in turn 1"),
        (
            NOTICE + "\n" + HEADER + "\n" + LEGACY + "\nBonjour.",
            "unexpected transcript uncertainty notice",
        ),
    ],
)
def test_refiner_logs_specific_rejection_and_preserves_raw(
    output: str, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    raw = HEADER + "\n" + LEGACY + "\nBonjour."
    if reason.startswith("word count"):
        raw = HEADER + "\nBonjour."
    response = subprocess.CompletedProcess(["agy"], 0, stdout=output)
    with (
        patch("shutil.which", return_value="agy"),
        patch("subprocess.run", return_value=response),
    ):
        assert refine_transcript_markdown(raw) == raw
    assert "chunk 1" in caplog.text
    assert reason in caplog.text


def test_validator_rejects_transcript_notice_moved_into_dialogue() -> None:
    original = NOTICE + "\n\n" + HEADER + "\nBonjour."
    moved = HEADER + "\n" + NOTICE + "\nBonjour."
    assert not validate_refiner_chunk(original, moved)


@pytest.mark.parametrize("failure", [False, True])
def test_refiner_preserves_frontmatter_before_notice(failure: bool) -> None:
    preamble = "---\ntitle: Session\n---\n\nCampaign notes."
    raw = preamble + "\n\n" + NOTICE + "\n\n" + HEADER + "\nBonjour."

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if failure:
            raise subprocess.TimeoutExpired(cmd, 60)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=next(
                arg.removeprefix("--print=")
                for arg in cmd
                if arg.startswith("--print=")
            ).split("Raw transcript:\n\n", 1)[1],
        )

    with (
        patch("shutil.which", return_value="agy"),
        patch("subprocess.run", side_effect=run),
    ):
        result = refine_transcript_markdown(raw)
    assert result.startswith(preamble + "\n\n" + NOTICE + "\n\n" + HEADER)
    assert result.count(NOTICE) == 1
