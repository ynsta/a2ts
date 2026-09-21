"""Unit tests for Obsidian lore and vocabulary mining."""

from pathlib import Path

from a2ts.models import EntityRecord
from a2ts.vocab import (
    build_biasing_prompt,
    extract_candidates_from_markdown,
    load_wordlist_file,
    scan_context_directory,
)


def test_extract_candidates_from_markdown() -> None:
    content = """
aliases:
  - L'Ombre Rouge
  - Le Rôdeur

## Kaelen Sombreflèche

Le personnage de [[Garrick Hautbois|Garrick]] voyage vers [[Phandaline]].
Un fichier [[carte.png]] est ignoré.
"""
    entities = extract_candidates_from_markdown(content, "session.md")
    names = {e.name for e in entities}
    assert "Kaelen Sombreflèche" in names
    assert "L'Ombre Rouge" in names
    assert "Le Rôdeur" in names
    assert "Garrick" in names
    assert "Phandaline" in names
    assert "carte.png" not in names


def test_build_biasing_prompt_token_budget() -> None:
    entities = [EntityRecord(name=f"Entité_{i}", kind="wikilink") for i in range(100)]
    prompt = build_biasing_prompt(entities, max_tokens=30)
    assert len(prompt) > 0
    # Prompt must contain commas and be non-empty
    assert "Entité_0" in prompt


def test_load_wordlist_file(tmp_path: Path) -> None:
    wordlist = tmp_path / "vocab.txt"
    wordlist.write_text(
        "Eldoria\nMorvath\n\n# Commentaire\nDraconique\n", encoding="utf-8"
    )
    records = load_wordlist_file(wordlist)
    names = [r.name for r in records]
    assert names == ["Eldoria", "Morvath", "Draconique"]
    assert load_wordlist_file(tmp_path / "non_existent.txt") == []


def test_scan_context_directory(tmp_path: Path) -> None:
    lore_dir = tmp_path / "lore"
    sub_dir = lore_dir / "locations"
    sub_dir.mkdir(parents=True)

    (lore_dir / "npcs.md").write_text(
        "## Dame Alyra\nRencontre avec [[Garrick]].\n", encoding="utf-8"
    )
    (sub_dir / "cities.md").write_text("## Neverwinter\nCapitale.\n", encoding="utf-8")
    (lore_dir / "notes.txt").write_text("Ignored text file.\n", encoding="utf-8")

    records = scan_context_directory(lore_dir)
    names = {r.name for r in records}
    assert "Dame Alyra" in names
    assert "Garrick" in names
    assert "Neverwinter" in names

    # Source files should be relative paths
    sources = {r.source_file for r in records}
    assert "npcs.md" in sources
    assert str(Path("locations") / "cities.md") in sources

    # Non-existent directory returns empty list
    assert scan_context_directory(tmp_path / "does_not_exist") == []


def test_extract_candidates_stopwords_and_extensions() -> None:
    content = """
aliases:
  - "dans"
  - 'Val-du-Loup'

## pour

[[vers]]
[[photo.jpg]]
[[audio.mp3]]
[[Valid Target|avec]]
[[Valid Target|Héroïne]]
"""
    entities = extract_candidates_from_markdown(content, "test.md")
    names = [e.name for e in entities]
    assert "Val-du-Loup" in names
    assert "Héroïne" in names
    assert "dans" not in names
    assert "pour" not in names
    assert "vers" not in names
    assert "photo.jpg" not in names
    assert "audio.mp3" not in names
    assert "avec" not in names


def test_build_biasing_prompt_dedup_and_whitespace() -> None:
    entities = [
        EntityRecord(name="  Garrick   Hautbois  ", kind="wikilink"),
        EntityRecord(name="Garrick Hautbois", kind="alias"),
        EntityRecord(name="garrick hautbois", kind="heading"),
        EntityRecord(name="Phandaline", kind="wikilink"),
    ]
    prompt = build_biasing_prompt(entities, max_tokens=220)
    assert prompt == "Garrick Hautbois, Phandaline"
