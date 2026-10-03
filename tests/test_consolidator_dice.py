"""Tests for French RPG dice normalization hardening."""

import re

from typer.testing import CliRunner

from a2ts.cli import app
from a2ts.consolidator import normalize_rpg_terms, render_markdown_transcript
from a2ts.models import AlignedTurn

runner = CliRunner()


def test_french_speech_not_mangled() -> None:
    """Verify standard French phrases with 'des N' are never converted to dice notation."""
    assert (
        normalize_rpg_terms("c est une des dix merveilles")
        == "c est une des dix merveilles"
    )
    assert (
        normalize_rpg_terms("il a eu deux des six votes")
        == "il a eu deux des six votes"
    )
    assert (
        normalize_rpg_terms("trois des huit portes sont ouvertes")
        == "trois des huit portes sont ouvertes"
    )
    assert (
        normalize_rpg_terms("un des vingt chevaliers est mort")
        == "un des vingt chevaliers est mort"
    )


def test_actual_dice_rolls_normalized() -> None:
    """Verify tabletop RPG roll expressions are properly normalized."""
    assert normalize_rpg_terms("il lance deux dés six") == "il lance 2d6"
    assert normalize_rpg_terms("lance un dé vingt") == "lance 1d20"
    assert normalize_rpg_terms("fais un jet de dés 20") == "fais un jet de 1d20"
    assert normalize_rpg_terms("il lance des dix") == "il lance 1d10"
    assert normalize_rpg_terms("un jet de délai") == "un jet de dés"
    assert normalize_rpg_terms("des jets de délai") == "des jets de dés"


def test_explicit_dice_notation_normalized() -> None:
    """Verify explicit dice notation with spaces is collapsed."""
    assert normalize_rpg_terms("il lance 2 d 6") == "il lance 2d6"
    assert normalize_rpg_terms("lance 1 d 20") == "lance 1d20"


def test_rpg_normalize_flag() -> None:
    """Verify render_markdown_transcript respects rpg_normalize=False."""
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=5.0,
            speaker="MJ",
            cluster_id="C0",
            text="Il lance deux dés six pour voir.",
        ),
    ]
    md_normalized = render_markdown_transcript(turns, rpg_normalize=True)
    assert "Il lance 2d6 pour voir." in md_normalized

    md_raw = render_markdown_transcript(turns, rpg_normalize=False)
    assert "Il lance deux dés six pour voir." in md_raw


def test_cli_rpg_normalize_flag_present() -> None:
    """Verify CLI run and craig commands expose --rpg-normalize/--no-rpg-normalize options."""
    result_run = runner.invoke(app, ["run", "--help"])
    assert result_run.exit_code == 0
    clean_run = re.sub(r"\x1b\[[0-9;]*m", "", result_run.output)
    assert "--rpg-normalize" in clean_run
    assert "--no-rpg-normalize" in clean_run

    result_craig = runner.invoke(app, ["craig", "--help"])
    assert result_craig.exit_code == 0
    clean_craig = re.sub(r"\x1b\[[0-9;]*m", "", result_craig.output)
    assert "--rpg-normalize" in clean_craig
    assert "--no-rpg-normalize" in clean_craig
