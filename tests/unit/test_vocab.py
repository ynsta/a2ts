"""Unit tests for Obsidian lore and vocabulary mining."""

from pathlib import Path

import tiktoken

from a2ts.models import EntityRecord
from a2ts.vocab import (
    build_biasing_prompt,
    count_tokens,
    extract_candidates_from_markdown,
    load_speaker_names,
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


def test_extract_candidates_wikilink_section_anchor() -> None:
    content = """
[[Spell#Fireball]]
[[Spell#Lightning|Foudre]]
[[#OnlySection]]
"""
    entities = extract_candidates_from_markdown(content, "rules.md")
    names = [e.name for e in entities]
    assert "Spell" in names
    assert "Foudre" in names
    assert "OnlySection" not in names
    assert "" not in names


def test_load_speaker_names(tmp_path: Path) -> None:
    speakers_file = tmp_path / "custom_speakers.txt"
    speakers_file.write_text(
        "MJ (Tessaro)\n- Brakk\n• Merrow\n# comment\n", encoding="utf-8"
    )

    ctx_dir = tmp_path / "contexte"
    ctx_dir.mkdir()
    (ctx_dir / "speakers.txt").write_text(
        "Ilvaris\nOskel\nDorsa\nQuillon\n", encoding="utf-8"
    )

    loaded = load_speaker_names(
        speakers_file=speakers_file,
        speakers_arg="MJ (Tessaro), Merrow",
        context_dir=ctx_dir,
    )

    assert loaded == [
        "MJ (Tessaro)",
        "Merrow",
        "Brakk",
        "Ilvaris",
        "Oskel",
        "Dorsa",
        "Quillon",
    ]


def test_special_tokens_handling() -> None:
    """Special tokens like <|endoftext|> must be treated as literal text and not crash."""
    special_names = [
        "<|endoftext|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|endofprompt|>",
    ]
    entities = [EntityRecord(name=name, kind="custom") for name in special_names]
    prompt = build_biasing_prompt(entities)
    for name in special_names:
        assert name in prompt


def test_build_biasing_prompt_priority_ordering() -> None:
    """High priority entities (speakers, custom terms) must appear before lore wikilinks."""
    entities = [
        EntityRecord(name="MinorLocation", kind="heading"),
        EntityRecord(name="Garrick", kind="speaker"),
        EntityRecord(name="Phandaline", kind="wikilink"),
        EntityRecord(name="Eldoria", kind="custom"),
        EntityRecord(name="Kaelen", kind="character"),
        EntityRecord(name="L'Ombre", kind="alias"),
    ]
    # Small budget to force truncation of low-priority terms
    prompt = build_biasing_prompt(entities, max_tokens=15)
    # Higher priority terms must be in the prompt
    assert "Garrick" in prompt
    assert "Kaelen" in prompt
    assert "Eldoria" in prompt
    # Lowest priority heading/alias should be excluded or at the end
    assert "MinorLocation" not in prompt


def test_build_biasing_prompt_default_budget_is_180() -> None:
    """Default max_tokens must be 180 to fit within Whisper's 224-token buffer."""
    import inspect

    sig = inspect.signature(build_biasing_prompt)
    assert sig.parameters["max_tokens"].default == 180


def test_count_tokens() -> None:
    """count_tokens must safely handle special tokens and empty strings."""
    assert count_tokens("") == 0
    assert count_tokens("Hello world") > 0
    assert count_tokens("<|endoftext|> <|fim_prefix|>") > 0


def test_special_tokens_with_explicit_tiktoken() -> None:
    """Explicit tiktoken encoding must treat special tokens as literal text without error."""
    enc = tiktoken.get_encoding("cl100k_base")
    # count_tokens with explicit tiktoken
    tokens = count_tokens("<|endoftext|> <|fim_middle|> <|endofprompt|>", tokenizer=enc)
    assert tokens > 0

    entities = [
        EntityRecord(name="<|endoftext|>", kind="custom"),
        EntityRecord(name="<|fim_prefix|>", kind="custom"),
    ]
    prompt = build_biasing_prompt(entities, tokenizer=enc)
    assert "<|endoftext|>" in prompt
    assert "<|fim_prefix|>" in prompt


def test_count_tokens_custom_tokenizer() -> None:
    """count_tokens must support arbitrary tokenizer implementing encode()."""

    class DummyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [42, 99]

    assert count_tokens("any sample text", tokenizer=DummyTokenizer()) == 2


def test_build_biasing_prompt_strictly_respects_max_tokens() -> None:
    """Prompt must never exceed max_tokens regardless of entity count."""
    entities = [
        EntityRecord(name=f"EntityName_{i}", kind="wikilink") for i in range(150)
    ]
    prompt = build_biasing_prompt(entities, max_tokens=50)
    assert count_tokens(prompt) <= 50
