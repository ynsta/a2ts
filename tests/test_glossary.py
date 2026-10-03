"""Public behavior of reviewed context glossary extraction."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from a2ts.glossary import build_context_glossary
from a2ts.models import GlossaryEntry


def test_glossary_filters_prose_and_deduplicates_exports(tmp_path: Path) -> None:
    text = "Bonjour le monde. Zorvax Zorvax.\n\nZorvax arrive.\n"
    (tmp_path / "one.md").write_text(text, encoding="utf-8")
    (tmp_path / "two.md").write_text(text, encoding="utf-8")
    result = build_context_glossary(tmp_path)
    entries = {entry.term: entry for entry in result.entries}
    assert entries["Zorvax"].frequency == 3
    assert entries["Zorvax"].sources == ["one.md", "two.md"]
    assert "Bonjour" not in entries
    assert "monde" not in entries
    assert len(result.source_hash) == 64


def test_glossary_preserves_names_aliases_and_visible_counts(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text(
        "---\nmetadata: invisiblexyz\n---\n"
        "[[Érindor Val|Éri]] meets [[Érindor Val]]. Éri greets Érindor Val. "
        "KX42 d’Zorvax Zorvax-Rûn.\n![secretxyz](https://secretxyz.test)\n"
        "```python\ninvisiblexyz\n```\n",
        encoding="utf-8",
    )
    entries = {entry.term: entry for entry in build_context_glossary(tmp_path).entries}
    assert entries["Érindor Val"].frequency == 4
    assert entries["Érindor Val"].aliases == ["Éri"]
    assert "KX42" in entries
    assert "d’Zorvax" in entries
    assert "Zorvax-Rûn" in entries
    assert not any("invisiblexyz" in term or "secretxyz" in term for term in entries)


def test_glossary_extra_terms_language_and_hash(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("the world Zorvax", encoding="utf-8")
    result = build_context_glossary(tmp_path, "en", ["world", "Absent Name"])
    assert {entry.term for entry in result.entries} == {
        "world",
        "Zorvax",
        "Absent Name",
    }
    assert result == build_context_glossary(tmp_path, "en", ["world", "Absent Name"])
    assert result.source_hash != build_context_glossary(tmp_path, "en").source_hash
    with pytest.raises(ValueError, match="Unsupported dictionary language"):
        build_context_glossary(tmp_path, "xx")


def test_glossary_missing_corrupt_and_escaping_symlink(tmp_path: Path) -> None:
    root = tmp_path / "context"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("Forbiddenxyz", encoding="utf-8")
    (root / "escape.md").symlink_to(outside)
    (root / "bad.md").write_bytes(b"\xffCorruptxyz")
    assert build_context_glossary(root).entries == []
    assert (
        build_context_glossary(root / "missing", extra_terms=["Zorvax"]).entries[0].term
        == "Zorvax"
    )


def test_glossary_schema_rejects_coercion_and_zero_frequency() -> None:
    with pytest.raises(ValidationError):
        GlossaryEntry(term="Zorvax", frequency=0, aliases=[], sources=[])
    with pytest.raises(ValidationError):
        GlossaryEntry.model_validate(
            {"term": "Zorvax", "frequency": "2", "aliases": [], "sources": []}
        )


def test_glossary_filters_french_elisions_and_embedded_metadata(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text(
        "qu'il c'est d'une l'aider d'Zorvax.\n![[portrait.png]]---\n"
        "aliases:\n  - Metadataxyz\ntags:\n  - Tagxyz\n---\nZorvax\n",
        encoding="utf-8",
    )
    terms = {entry.term for entry in build_context_glossary(tmp_path).entries}
    assert terms == {"d'Zorvax", "Zorvax"}


def test_glossary_keeps_repetitions_within_single_source(tmp_path: Path) -> None:
    (tmp_path / "one.md").write_text("Zorvax\nZorvax\n", encoding="utf-8")
    (tmp_path / "two.md").write_text("Zorvax\n", encoding="utf-8")
    assert build_context_glossary(tmp_path).entries[0].frequency == 2


def test_glossary_preserves_frontmatter_aliases_and_removes_control_tokens(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.md").write_text(
        "---\naliases:\n  - Éri\ntags:\n  - Metadataxyz\n---\n"
        "# Érindor Val\nÉri Érindor Val <|forbiddenxyz|>\n",
        encoding="utf-8",
    )
    entries = {entry.term: entry for entry in build_context_glossary(tmp_path).entries}
    assert entries["Érindor Val"].aliases == ["Éri"]
    assert entries["Érindor Val"].frequency == 3
    assert set(entries) == {"Érindor Val"}


def test_glossary_ordinary_aliases_count_only_explicit_links(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text(
        "[[Zorvax|dragon]] dragon dragon. Zorvax.",
        encoding="utf-8",
    )
    entry = build_context_glossary(tmp_path).entries[0]
    assert entry.term == "Zorvax"
    assert entry.aliases == ["dragon"]
    assert entry.frequency == 2
