"""Observable safety and speaker reconstruction contracts."""

import json
import subprocess
from typing import Any

import pytest

from a2ts.consolidator import SPEAKER_UNCERTAINTY_NOTICE
from a2ts.reconstruction import reconstruct_transcript_markdown

RAW = (
    "### [00:00:01 - 00:00:02] Alex\n\nTu veux un petit\n\n"
    "### [00:00:02 - 00:00:03] Sam\n\ncoup d’œil\n\n"
    "### [00:00:03 - 00:00:04] Alex\n\ndiscret ou quoi ?\n"
)


def install_cli(monkeypatch: pytest.MonkeyPatch, payload: object) -> list[str]:
    prompts: list[str] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "--json-schema" in command
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 600
        assert command[command.index("--output-format") + 1] == "json"
        assert command[-1].startswith("--print=")
        assert "--print" not in command
        prompts.append(command[-1].removeprefix("--print="))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"structured_output": payload, "type": "result"}),
        )

    monkeypatch.setattr("a2ts.reconstruction.shutil.which", lambda _: "/bin/agy")
    monkeypatch.setattr("a2ts.reconstruction.subprocess.run", run)
    return prompts


def test_reconstruction_repairs_sentence_speaker_and_preserves_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = install_cli(
        monkeypatch,
        {
            "turns": [
                {
                    "speaker": "Alex",
                    "word_start": 0,
                    "word_end": 10,
                    "text": "Tu veux un petit coup d’œil discret ou quoi ?",
                }
            ]
        },
    )
    raw = "# Session\n\n" + SPEAKER_UNCERTAINTY_NOTICE + "\n\n" + RAW
    result = reconstruct_transcript_markdown(raw)
    assert result.count("###") == 1
    assert "### [00:00:01 - 00:00:04] Alex" in result
    assert result.count(SPEAKER_UNCERTAINTY_NOTICE) == 1
    assert result.startswith("# Session")
    assert SPEAKER_UNCERTAINTY_NOTICE not in prompts[0]


@pytest.mark.parametrize(
    "change",
    [
        {"word_start": 1},
        {"word_end": 9},
        {"word_end": 11},
        {"speaker": "Invented"},
        {"text": "An entirely fabricated story."},
        {"word_start": "0"},
        {"extra": True},
    ],
)
def test_reconstruction_rejects_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
) -> None:
    turn: dict[str, object] = {
        "speaker": "Alex",
        "word_start": 0,
        "word_end": 10,
        "text": "Tu veux un petit coup d’œil discret ou quoi ?",
    }
    turn.update(change)
    install_cli(monkeypatch, {"turns": [turn]})
    assert reconstruct_transcript_markdown(RAW) == RAW


def test_reconstruction_can_split_turn_and_omit_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_cli(
        monkeypatch,
        {
            "turns": [
                {
                    "speaker": "Sam",
                    "word_start": 0,
                    "word_end": 4,
                    "text": "Tu veux un petit",
                },
                {
                    "speaker": "Alex",
                    "word_start": 4,
                    "word_end": 10,
                    "text": "coup d’œil discret ou quoi ?",
                    "out_of_game": True,
                },
            ]
        },
    )
    result = reconstruct_transcript_markdown(
        RAW, keep_timestamps=False, rpg_normalize=False
    )
    assert "### Sam" in result
    assert "### Alex" in result
    assert "[00:" not in result
    assert "[!NOTE]" not in result


def test_reconstruction_bounds_chunks_and_keeps_global_word_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[int]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        source = json.loads(
            command[-1].removeprefix("--print=").split("Source JSON:\n", 1)[1]
        )
        words = source["words"]
        seen.append([word["id"] for word in words])
        payload = {
            "turns": [
                {
                    "speaker": "Sam",
                    "word_start": words[0]["id"],
                    "word_end": words[-1]["id"] + 1,
                    "text": " ".join(word["text"] for word in words),
                }
            ]
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"structured_output": payload, "type": "result"}),
        )

    monkeypatch.setattr("a2ts.reconstruction.shutil.which", lambda _: "/bin/agy")
    monkeypatch.setattr("a2ts.reconstruction.subprocess.run", run)
    result = reconstruct_transcript_markdown(
        "### [00:00:01 - 00:00:10] Sam\n\none two three four five", max_chunk_words=2
    )
    assert seen == [[0, 1], [2, 3], [4]]
    assert "five" in result


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"turns": []},
        {
            "turns": [
                {
                    "speaker": "Alex",
                    "word_start": 4,
                    "word_end": 10,
                    "text": "coup d’œil discret ou quoi ?",
                },
                {
                    "speaker": "Sam",
                    "word_start": 0,
                    "word_end": 4,
                    "text": "Tu veux un petit",
                },
            ]
        },
        {
            "turns": [
                {
                    "speaker": "Alex",
                    "word_start": 0,
                    "word_end": 10,
                    "text": "Tu veux un petit coup d’œil discret ou quoi ?",
                },
                {
                    "speaker": "Alex",
                    "word_start": 0,
                    "word_end": 10,
                    "text": "Tu veux un petit coup d’œil discret ou quoi ?",
                },
            ]
        },
    ],
)
def test_reconstruction_rejects_malformed_reordered_and_duplicate_coverage(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    install_cli(monkeypatch, payload)
    assert reconstruct_transcript_markdown(RAW) == RAW


def test_reconstruction_preserves_notice_from_source_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = install_cli(
        monkeypatch,
        {
            "turns": [
                {
                    "speaker": "Alex",
                    "word_start": 0,
                    "word_end": 10,
                    "text": "Tu veux un petit coup d’œil discret ou quoi ?",
                }
            ]
        },
    )
    raw = RAW.replace("Tu veux", SPEAKER_UNCERTAINTY_NOTICE + "\n\nTu veux")
    result = reconstruct_transcript_markdown(raw)
    assert result.count(SPEAKER_UNCERTAINTY_NOTICE) == 1
    assert result.index(SPEAKER_UNCERTAINTY_NOTICE) < result.index("###")
    assert SPEAKER_UNCERTAINTY_NOTICE not in prompts[0]


def test_reconstruction_cli_failure_keeps_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("a2ts.reconstruction.shutil.which", lambda _: "/bin/agy")
    monkeypatch.setattr("a2ts.reconstruction.subprocess.run", run)
    assert reconstruct_transcript_markdown(RAW) == RAW


def test_reconstruction_repairs_alternating_sentence_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_cli(
        monkeypatch,
        {
            "turns": [
                {
                    "speaker": "Sam",
                    "word_start": 0,
                    "word_end": 6,
                    "text": "Oui vas-y je passe devant poto",
                }
            ]
        },
    )
    raw = "### Sam\n\nOui vas-y je passe\n\n### Alex\n\ndevant\n\n### Sam\n\npoto\n"
    result = reconstruct_transcript_markdown(raw, keep_timestamps=False)
    assert result == "### Sam\n\nOui vas-y je passe devant poto\n"


@pytest.mark.parametrize("is_error", [False, True])
def test_reconstruction_honors_cli_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
    is_error: bool,
) -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        payload = {
            "structured_output": {
                "turns": [
                    {
                        "speaker": "Alex",
                        "word_start": 0,
                        "word_end": 10,
                        "text": "Tu veux un petit coup d’œil discret ou quoi ?",
                    }
                ]
            },
            "is_error": is_error,
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload))

    monkeypatch.setattr("a2ts.reconstruction.shutil.which", lambda _: "/bin/agy")
    monkeypatch.setattr("a2ts.reconstruction.subprocess.run", run)
    result = reconstruct_transcript_markdown(RAW)
    assert (result == RAW) is is_error


def test_reconstruction_preserves_legacy_uncertainty_as_global_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from a2ts.consolidator import SPEAKER_UNCERTAINTY_CALLOUT

    prompts = install_cli(
        monkeypatch,
        {
            "turns": [
                {
                    "speaker": "Alex",
                    "word_start": 0,
                    "word_end": 10,
                    "text": "Tu veux un petit coup d’œil discret ou quoi ?",
                }
            ]
        },
    )
    raw = RAW.replace("Tu veux", SPEAKER_UNCERTAINTY_CALLOUT + "\n\nTu veux")
    result = reconstruct_transcript_markdown(raw)
    assert result.count(SPEAKER_UNCERTAINTY_NOTICE) == 1
    assert SPEAKER_UNCERTAINTY_CALLOUT not in prompts[0]


def test_reconstruction_keeps_callouts_in_split_failed_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        source = json.loads(command[-1].split("Source JSON:\n", 1)[1])
        words = source["words"]
        if words[0]["id"] == 2:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        payload = {
            "turns": [
                {
                    "speaker": "Sam",
                    "word_start": 0,
                    "word_end": 2,
                    "text": "one two",
                    "out_of_game": True,
                }
            ]
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload))

    monkeypatch.setattr("a2ts.reconstruction.shutil.which", lambda _: "/bin/agy")
    monkeypatch.setattr("a2ts.reconstruction.subprocess.run", run)
    raw = "### Sam\n\n> [!NOTE] Hors-jeu / Discussion\n> one two three four\n"
    result = reconstruct_transcript_markdown(raw, max_chunk_words=2)
    assert result.count("> [!NOTE] Hors-jeu / Discussion") == 2
    assert "> three four" in result


@pytest.mark.parametrize(
    ("original", "repaired"),
    [
        (
            "Je colle à la paroi et je regarde devant moi.",
            "Je me colle à la paroi et je regarde devant moi.",
        ),
        (
            "Ok, c’est vous, toi tu vas devant discrètement.",
            "OK, Sam, toi tu y vas devant discrètement.",
        ),
    ],
)
def test_reconstruction_accepts_bounded_missing_words_and_glossary_names(
    monkeypatch: pytest.MonkeyPatch,
    original: str,
    repaired: str,
) -> None:
    from a2ts.models import ContextGlossary, GlossaryEntry

    install_cli(
        monkeypatch,
        {
            "turns": [
                {
                    "speaker": "Alex",
                    "word_start": 0,
                    "word_end": len(original.split()),
                    "text": repaired,
                }
            ]
        },
    )
    glossary = ContextGlossary(
        language="fr",
        entries=[GlossaryEntry(term="Sam", frequency=1)],
        source_hash="0" * 64,
        dictionary_version="1",
    )
    result = reconstruct_transcript_markdown(
        "### Alex\n\n" + original, glossary=glossary
    )
    assert repaired in result


def test_reconstruction_rejects_injected_new_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "Je colle à la paroi et je regarde devant moi."
    install_cli(
        monkeypatch,
        {
            "turns": [
                {
                    "speaker": "Alex",
                    "word_start": 0,
                    "word_end": len(original.split()),
                    "text": original
                    + " Une armée de dragons détruit entièrement le village.",
                }
            ]
        },
    )
    raw = "### Alex\n\n" + original
    assert reconstruct_transcript_markdown(raw) == raw
