"""Lore extraction and token-budgeted prompt construction."""

import re
from pathlib import Path

import tiktoken

from a2ts.models import EntityRecord

WIKILINK_PATTERN = re.compile(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]")
ALIAS_BLOCK_PATTERN = re.compile(r"(?m)^aliases:\s*\n((?:\s*-\s*.+\n)+)")
HEADING_PATTERN = re.compile(r"(?m)^##\s+([^\n#]+)")

IGNORED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".svg",
    ".pdf",
    ".mp3",
    ".wav",
    ".flac",
}

FRENCH_STOPWORDS = {
    "le",
    "la",
    "les",
    "l",
    "un",
    "une",
    "des",
    "du",
    "de",
    "d",
    "au",
    "aux",
    "et",
    "ou",
    "mais",
    "donc",
    "or",
    "ni",
    "car",
    "que",
    "qui",
    "quoi",
    "dont",
    "où",
    "en",
    "dans",
    "par",
    "pour",
    "vers",
    "avec",
    "sans",
    "sur",
    "sous",
}


def extract_candidates_from_markdown(content: str, filename: str) -> list[EntityRecord]:
    """Parse wikilinks, frontmatter aliases, and headings from markdown text."""
    records: list[EntityRecord] = []

    for match in WIKILINK_PATTERN.finditer(content):
        target = match.group(1).strip()
        alias = match.group(2).strip() if match.group(2) else None
        name = alias if alias else target

        if any(name.lower().endswith(ext) for ext in IGNORED_EXTENSIONS):
            continue
        if name and name.lower() not in FRENCH_STOPWORDS:
            records.append(
                EntityRecord(name=name, kind="wikilink", source_file=filename)
            )

    alias_block = ALIAS_BLOCK_PATTERN.search(content)
    if alias_block:
        for line in alias_block.group(1).splitlines():
            alias_match = re.match(r"^\s*-\s*(.+)$", line)
            if alias_match:
                alias = alias_match.group(1).strip().strip("\"'")
                if alias and alias.lower() not in FRENCH_STOPWORDS:
                    records.append(
                        EntityRecord(name=alias, kind="alias", source_file=filename)
                    )

    for match in HEADING_PATTERN.finditer(content):
        heading = match.group(1).strip()
        if heading and heading.lower() not in FRENCH_STOPWORDS:
            records.append(
                EntityRecord(name=heading, kind="heading", source_file=filename)
            )

    return records


def load_wordlist_file(vocab_path: Path) -> list[EntityRecord]:
    """Load plain text wordlist file (one term per line, skipping comments)."""
    if not vocab_path.is_file():
        return []
    records: list[EntityRecord] = []
    for line in vocab_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            records.append(
                EntityRecord(name=line, kind="custom", source_file=str(vocab_path))
            )
    return records


def scan_context_directory(context_dir: Path) -> list[EntityRecord]:
    """Recursively scan markdown files in context directory."""
    if not context_dir.is_dir():
        return []
    records: list[EntityRecord] = []
    for md_file in sorted(context_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        records.extend(
            extract_candidates_from_markdown(
                content, str(md_file.relative_to(context_dir))
            )
        )
    return records


def build_biasing_prompt(entities: list[EntityRecord], max_tokens: int = 220) -> str:
    """Construct token-budgeted prompt from unique entity records."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    seen: set[str] = set()
    unique_names: list[str] = []

    for e in entities:
        cleaned = re.sub(r"\s+", " ", e.name).strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unique_names.append(cleaned)

    selected: list[str] = []

    for name in unique_names:
        candidate_list = selected + [name]
        candidate_str = ", ".join(candidate_list)
        token_count = len(tokenizer.encode(candidate_str))
        if token_count <= max_tokens:
            selected.append(name)
        else:
            break

    return ", ".join(selected)
