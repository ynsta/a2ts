import subprocess
from unittest.mock import MagicMock, patch

from a2ts.refiner import (
    compute_edit_distance_ratio,
    normalize_rpg_phonetics,
    refine_transcript_markdown,
)


def test_normalize_rpg_phonetics() -> None:
    text = "Fais un jet d'un des vingt et deux des six pour tes jets de délai."
    normalized = normalize_rpg_phonetics(text)
    assert "1d20" in normalized
    assert "2d6" in normalized
    assert "jets de dés" in normalized


def test_normalize_rpg_phonetics_variations() -> None:
    text = "Lance un dé vingt et un jet de délai."
    normalized = normalize_rpg_phonetics(text)
    assert "1d20" in normalized
    assert "jet de dés" in normalized


def test_compute_edit_distance_ratio() -> None:
    ratio = compute_edit_distance_ratio("Bonjour à tous", "Bonjour à tous")
    assert ratio == 0.0

    ratio_diff = compute_edit_distance_ratio("Bonjour", "Bonsoir")
    assert 0.0 < ratio_diff < 1.0


def test_refine_transcript_markdown_no_agy() -> None:
    raw = "Lance un des vingt."
    with patch("shutil.which", return_value=None), patch("subprocess.run") as mock_run:
        result = refine_transcript_markdown(raw)
        assert result == "Lance 1d20."
        mock_run.assert_not_called()


def test_refine_transcript_markdown_success() -> None:
    raw = "### [00:00:00 - 00:00:05] MJ\nLance un des vingt pour voir."
    refined_output = "### [00:00:00 - 00:00:05] MJ\nLance 1d20 pour voir."

    mock_proc = MagicMock()
    mock_proc.stdout = refined_output + "\n"

    with (
        patch("shutil.which", return_value="/usr/local/bin/agy"),
        patch("subprocess.run", return_value=mock_proc) as mock_run,
    ):
        result = refine_transcript_markdown(raw, agy_model="custom-model")
        assert result == refined_output
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["agy", "--model", "custom-model", "--effort", "low"]
        assert "Lance 1d20 pour voir." in mock_run.call_args.kwargs["input"]
        assert mock_run.call_args.kwargs["text"] is True
        assert mock_run.call_args.kwargs["capture_output"] is True


def test_refine_transcript_markdown_hallucination_guard() -> None:
    raw = "### [00:00:00 - 00:00:05] MJ\nLance un des vingt."
    hallucinated = "Il était une fois dans un royaume lointain..."

    mock_proc = MagicMock()
    mock_proc.stdout = hallucinated

    with (
        patch("shutil.which", return_value="/usr/local/bin/agy"),
        patch("subprocess.run", return_value=mock_proc),
    ):
        result = refine_transcript_markdown(raw)
        # Should reject hallucinated output and return normalized raw
        assert result == "### [00:00:00 - 00:00:05] MJ\nLance 1d20."


def test_refine_transcript_markdown_subprocess_error() -> None:
    raw = "### [00:00:00 - 00:00:05] MJ\nLance un des vingt."

    with (
        patch("shutil.which", return_value="/usr/local/bin/agy"),
        patch(
            "subprocess.run",
            side_effect=subprocess.SubprocessError("Subprocess failed"),
        ),
    ):
        result = refine_transcript_markdown(raw)
        assert result == "### [00:00:00 - 00:00:05] MJ\nLance 1d20."
