"""Lore extraction and token-budgeted prompt construction."""

import logging
import os
import re
from pathlib import Path
from typing import Any

import tiktoken

from a2ts.models import EntityRecord

logger = logging.getLogger(__name__)

CONTROL_TOKEN_PATTERN = re.compile(r"<\|.*?\|>")
WIKILINK_PATTERN = re.compile(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]")
ALIAS_BLOCK_PATTERN = re.compile(r"(?m)^aliases:\s*\n((?:\s*-\s*.+\n)+)")
HEADING_PATTERN = re.compile(r"(?m)^##\s+([^\n#]+)")


def clean_entity_name(name: str) -> str:
    """Sanitize entity name by stripping Whisper/GPT control tokens (<|.*?|>) and normalizing whitespace."""
    cleaned = CONTROL_TOKEN_PATTERN.sub("", name)
    return re.sub(r"\s+", " ", cleaned).strip()


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

PRIORITY_ORDER: dict[str, int] = {
    "speaker": 0,
    "character": 0,
    "nickname": 0,
    "custom": 1,
    "wikilink": 2,
    "alias": 3,
    "heading": 4,
}


def _entity_priority(kind: str) -> int:
    return PRIORITY_ORDER.get(kind.lower(), 5)


_CACHED_TOKENIZER: Any = None


def get_tokenizer() -> Any:
    """Return a tokenizer for prompt budgeting, preferring faster-whisper over tiktoken."""
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is not None:
        return _CACHED_TOKENIZER

    try:
        import tokenizers  # type: ignore[import-untyped]
        from faster_whisper.tokenizer import Tokenizer  # type: ignore[import-untyped]

        if "HF_HUB_CACHE" in os.environ:
            hf_hub = Path(os.environ["HF_HUB_CACHE"])
        elif "HF_HOME" in os.environ:
            hf_hub = Path(os.environ["HF_HOME"]) / "hub"
        else:
            hf_hub = Path.home() / ".cache" / "huggingface" / "hub"

        if hf_hub.is_dir():
            tok_files = sorted(hf_hub.glob("models--*whisper*/**/tokenizer.json"))
            if tok_files:
                hf_tok = tokenizers.Tokenizer.from_file(str(tok_files[0]))
                _CACHED_TOKENIZER = Tokenizer(
                    hf_tok, multilingual=True, language="fr", task="transcribe"
                )
                return _CACHED_TOKENIZER
    except (ImportError, OSError, ValueError) as err:
        logger.debug(
            "Faster-whisper tokenizer unavailable, falling back to tiktoken: %s",
            err,
        )

    _CACHED_TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _CACHED_TOKENIZER


def count_tokens(text: str, tokenizer: Any = None) -> int:
    """Count tokens in text, safely treating special tokens as normal text."""
    if not text:
        return 0
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    if isinstance(tok, tiktoken.Encoding):
        return len(tok.encode(text, disallowed_special=()))
    if hasattr(tok, "encode"):
        try:
            return len(tok.encode(text))
        except TypeError:
            return len(tok.encode(text, disallowed_special=()))
    return len(text.split())


def extract_candidates_from_markdown(content: str, filename: str) -> list[EntityRecord]:
    """Parse wikilinks, frontmatter aliases, and headings from markdown text."""
    records: list[EntityRecord] = []
    content = CONTROL_TOKEN_PATTERN.sub("", content)

    for match in WIKILINK_PATTERN.finditer(content):
        target = match.group(1).strip()
        alias = match.group(2).strip() if match.group(2) else None
        clean_target = target.split("#", 1)[0].strip()
        name = clean_entity_name(alias if alias else clean_target)

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
                alias = clean_entity_name(alias)
                if alias and alias.lower() not in FRENCH_STOPWORDS:
                    records.append(
                        EntityRecord(name=alias, kind="alias", source_file=filename)
                    )

    for match in HEADING_PATTERN.finditer(content):
        heading = clean_entity_name(match.group(1).strip())
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


def build_biasing_prompt(
    entities: list[EntityRecord],
    max_tokens: int = 180,
    tokenizer: Any = None,
) -> str:
    """Construct token-budgeted prompt from unique entity records, prioritizing key terms."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    sorted_entities = sorted(entities, key=lambda e: _entity_priority(e.kind))

    seen: set[str] = set()
    unique_names: list[str] = []

    for e in sorted_entities:
        cleaned = clean_entity_name(e.name)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unique_names.append(cleaned)

    selected: list[str] = []

    for name in unique_names:
        candidate_list = selected + [name]
        candidate_str = ", ".join(candidate_list)
        token_count = count_tokens(candidate_str, tokenizer=tokenizer)
        if token_count <= max_tokens:
            selected.append(name)
        else:
            break

    return ", ".join(selected)


parse_obsidian_note = extract_candidates_from_markdown


def load_speaker_names(
    speakers_file: Path | None = None,
    speakers_arg: str | None = None,
    context_dir: Path | None = None,
) -> list[str]:
    """Load known speaker/character names from CLI args, files, or context directory."""
    names: list[str] = []
    if speakers_arg:
        for s in speakers_arg.split(","):
            clean = s.strip()
            if clean:
                names.append(clean)

    if speakers_file and speakers_file.is_file():
        for line in speakers_file.read_text(encoding="utf-8").splitlines():
            clean = line.strip().lstrip("-*•0123456789. ")
            if clean and not clean.startswith("#"):
                names.append(clean)

    if context_dir and context_dir.is_dir():
        for candidate_filename in ["speakers.txt", "speakers.md", "personnages.txt"]:
            cand_path = context_dir / candidate_filename
            if cand_path.is_file():
                for line in cand_path.read_text(encoding="utf-8").splitlines():
                    clean = line.strip().lstrip("-*•0123456789. ")
                    if clean and not clean.startswith("#"):
                        names.append(clean)

    return list(dict.fromkeys(names))
