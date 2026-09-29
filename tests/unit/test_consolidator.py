from a2ts.consolidator import (
    debounce_consecutive_turns,
    format_timestamp,
    normalize_rpg_terms,
    render_markdown_transcript,
)
from a2ts.models import AlignedTurn, WordTimestamp


def test_format_timestamp() -> None:
    assert format_timestamp(75.5) == "00:01:15"
    assert format_timestamp(3665.0) == "01:01:05"
    assert format_timestamp(0.0) == "00:00:00"
    assert format_timestamp(-10.0) == "00:00:00"


def test_debounce_consecutive_turns() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=1.0,
            end=3.0,
            speaker="MJ",
            cluster_id="C0",
            text="Première partie.",
        ),
        AlignedTurn(
            turn_id=1,
            start=3.5,
            end=6.0,
            speaker="MJ",
            cluster_id="C0",
            text="Deuxième partie.",
        ),  # gap 0.5s < 2.0s
        AlignedTurn(
            turn_id=2,
            start=10.0,
            end=12.0,
            speaker="Alice",
            cluster_id="C1",
            text="Je réponds.",
        ),
    ]
    debounced = debounce_consecutive_turns(turns, threshold_seconds=2.0)
    assert len(debounced) == 2
    assert debounced[0].speaker == "MJ"
    assert debounced[0].text == "Première partie. Deuxième partie."
    assert debounced[0].start == 1.0
    assert debounced[0].end == 6.0
    assert debounced[1].speaker == "Alice"


def test_debounce_consecutive_turns_empty() -> None:
    assert debounce_consecutive_turns([]) == []


def test_debounce_consecutive_turns_preserves_words_and_sorting() -> None:
    w1 = WordTimestamp(word="Hello", start=1.0, end=1.5)
    w2 = WordTimestamp(word="world", start=2.0, end=2.5)
    turns = [
        AlignedTurn(
            turn_id=1,
            start=2.0,
            end=3.0,
            speaker="Alice",
            cluster_id="C0",
            text="world",
            words=[w2],
        ),
        AlignedTurn(
            turn_id=0,
            start=1.0,
            end=1.5,
            speaker="Alice",
            cluster_id="C0",
            text="Hello",
            words=[w1],
        ),
    ]
    # Unsorted input should be sorted and merged
    debounced = debounce_consecutive_turns(turns, threshold_seconds=2.0)
    assert len(debounced) == 1
    assert debounced[0].text == "Hello world"
    assert debounced[0].start == 1.0
    assert debounced[0].end == 3.0
    assert debounced[0].words == [w1, w2]


def test_debounce_consecutive_turns_different_speakers_or_large_gap() -> None:
    turns = [
        AlignedTurn(
            turn_id=0, start=1.0, end=2.0, speaker="Bob", cluster_id="C0", text="Hi"
        ),
        AlignedTurn(
            turn_id=1,
            start=2.2,
            end=3.0,
            speaker="Alice",
            cluster_id="C1",
            text="Hello",
        ),
        AlignedTurn(
            turn_id=2,
            start=6.0,
            end=7.0,
            speaker="Alice",
            cluster_id="C1",
            text="Still here",
        ),
    ]
    debounced = debounce_consecutive_turns(turns, threshold_seconds=2.0)
    assert len(debounced) == 3


def test_render_markdown_transcript() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=10.0,
            speaker="MJ",
            cluster_id="C0",
            text="Bienvenue à tous.",
        ),
    ]
    md = render_markdown_transcript(turns)
    assert "### [00:00:00 - 00:00:10] MJ" in md
    assert "Bienvenue à tous." in md


def test_render_markdown_transcript_empty() -> None:
    assert render_markdown_transcript([]) == ""


def test_normalize_rpg_terms_dice_patterns() -> None:
    assert normalize_rpg_terms("lance un dé 20") == "lance 1d20"
    assert normalize_rpg_terms("lance un dé de 20") == "lance 1d20"
    assert normalize_rpg_terms("lance deux dés de 6") == "lance 2d6"
    assert normalize_rpg_terms("fais trois dés de six") == "fais 3d6"
    assert normalize_rpg_terms("lance un dé cent") == "lance 1d100"
    assert normalize_rpg_terms("lance 1 dé 12") == "lance 1d12"
    assert normalize_rpg_terms("lance quatre dés de 8") == "lance 4d8"
    assert normalize_rpg_terms("cinq dés de dix") == "5d10"
    assert normalize_rpg_terms("un dé 4") == "1d4"


def test_normalize_rpg_terms_exclusions() -> None:
    assert normalize_rpg_terms("il y a 20 gardes") == "il y a 20 gardes"
    assert normalize_rpg_terms("des 20 gardes approchent") == "des 20 gardes approchent"
    assert normalize_rpg_terms("pendant des 10 minutes") == "pendant des 10 minutes"
    assert normalize_rpg_terms("chacun des 6 joueurs") == "chacun des 6 joueurs"


def test_normalize_rpg_terms_phonetic_fixes() -> None:
    assert normalize_rpg_terms("fais des jets-dés") == "fais des jets de dés"
    assert normalize_rpg_terms("fais un jet de délai") == "fais un jet de dés"


def test_debounce_applies_rpg_normalization() -> None:
    turns = [
        AlignedTurn(
            turn_id=0,
            start=0.0,
            end=2.0,
            speaker="MJ",
            cluster_id="C0",
            text="Lance un dé 20 pour voir.",
        ),
    ]
    debounced = debounce_consecutive_turns(turns)
    assert debounced[0].text == "Lance 1d20 pour voir."

