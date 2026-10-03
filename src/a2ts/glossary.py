"""Build an offline frequency glossary without correcting reviewed spelling."""

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from spellchecker import SpellChecker

from a2ts.models import ContextGlossary, GlossaryEntry
from a2ts.vocab import clean_entity_name, strip_control_tokens

_ALGORITHM_VERSION = "1"
_WORD = re.compile(r"[^\W\d_][\w]*(?:['’\-][\w]+)*", re.UNICODE)
_LINK = re.compile(r"\[\[([^\]\n]+)\]\]")


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _term(value: str) -> str:
    value = unicodedata.normalize("NFC", clean_entity_name(value))
    return value if len(value) <= 120 and _WORD.search(value) else ""


def _visible(text: str, names: dict[str, str], aliases: dict[str, set[str]]) -> str:
    text = strip_control_tokens(text)
    text = re.sub(r"!\[\[[^\]]*\]\]|!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(
        r"(?m)^\s*(`{3,}|~{3,}).*?^\s*\1[^\n]*(?:\n|$)", "", text, flags=re.DOTALL
    )
    metadata = re.compile(
        r"(?m)^(?:---[ \t]*\n)?(?:aliases|tags|[A-Za-z_]+):[^\n]*\n"
        r"(?:(?!^---[ \t]*$).)*^---[ \t]*(?:\n|$)",
        re.DOTALL,
    )

    def frontmatter(match: re.Match[str]) -> str:
        following = text[match.end() :]
        heading = re.match(r"\s*#{1,6}\s+([^\n]+)", following)
        if heading:
            canonical = _term(heading.group(1))
            if canonical:
                key = _normalize(canonical)
                names.setdefault(key, canonical)
                block = re.search(
                    r"(?m)^aliases:[ \t]*\n((?:[ \t]+-[^\n]*\n)+)", match.group()
                )
                if block:
                    for value in block.group(1).splitlines():
                        alias = _term(
                            value.strip().removeprefix("-").strip().strip("\"'")
                        )
                        if alias and _normalize(alias) != key:
                            aliases[key].add(alias)
        return ""

    text = metadata.sub(frontmatter, text)

    def link(match: re.Match[str]) -> str:
        fields = match.group(1).split("|", 1)
        canonical = _term(fields[0].split("#", 1)[0])
        visible = _term(fields[-1]) if len(fields) == 2 else canonical
        if canonical:
            names.setdefault(_normalize(canonical), canonical)
            if visible and _normalize(visible) != _normalize(canonical):
                aliases[_normalize(canonical)].add(visible)
        return canonical or visible

    text = _LINK.sub(link, text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+|<[^>]*>|`[^`]*`", "", text)
    return re.sub(r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s*)", "", text)


def build_context_glossary(
    context_dir: Path, language: str = "fr", extra_terms: list[str] | None = None
) -> ContextGlossary:
    """Extract trusted rare spellings; count duplicate exports only once."""
    try:
        dictionary = SpellChecker(language=language)
    except ValueError as error:
        raise ValueError(f"Unsupported dictionary language: {language}") from error
    dictionary_version = version("pyspellchecker")
    root = context_dir.resolve()
    names: dict[str, str] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    extras = [_term(term) for term in extra_terms or []]
    for term in extras:
        if term:
            names.setdefault(_normalize(term), term)
    contents: list[tuple[str, str]] = []
    source_records: list[tuple[str, str]] = []
    if root.is_dir():
        for path in sorted(root.rglob("*.md")):
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            source_records.append((relative, hashlib.sha256(raw).hexdigest()))
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            visible = _visible(text, names, aliases)
            contents.append((relative, visible))
    variants: dict[str, str] = {key: key for key in names}
    alias_targets: dict[str, set[str]] = defaultdict(set)
    for canonical, values in aliases.items():
        for alias in values:
            alias_targets[_normalize(alias)].add(canonical)
    for alias, targets in alias_targets.items():
        words = _WORD.findall(alias)
        known_alias_words = dictionary.known(words)
        meaningful = any(word.casefold() not in known_alias_words for word in words)
        if len(targets) == 1 and alias not in names and meaningful:
            variants[alias] = next(iter(targets))
    counts: Counter[str] = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    spellings = dict(names)
    seen_lines: dict[str, str] = {}
    known_words = dictionary.known(
        {match.group() for _, visible in contents for match in _WORD.finditer(visible)}
    )
    for relative, visible in contents:
        for line in visible.splitlines():
            normalized_line = " ".join(line.split())
            if not normalized_line:
                continue
            duplicate = (
                normalized_line in seen_lines
                and seen_lines[normalized_line] != relative
            )
            seen_lines.setdefault(normalized_line, relative)
            found: Counter[str] = Counter()
            remaining = line
            for variant in sorted(variants, key=lambda item: (-len(item), item)):
                pattern = re.compile(
                    r"(?<!\w)" + re.escape(variant) + r"(?!\w)", re.IGNORECASE
                )
                matches = list(pattern.finditer(remaining))
                if matches:
                    found[variants[variant]] += len(matches)
                    remaining = pattern.sub(" ", remaining)
            for match in _WORD.finditer(remaining):
                word = _term(match.group())
                if len(word) < 2 or word.casefold() in known_words:
                    continue
                pieces = re.split("[’']", word.casefold())
                if (
                    language == "fr"
                    and len(pieces) == 2
                    and pieces[0]
                    in {
                        "c",
                        "d",
                        "j",
                        "l",
                        "m",
                        "n",
                        "s",
                        "t",
                        "qu",
                        "jusqu",
                        "lorsqu",
                        "puisqu",
                    }
                    and dictionary.known([pieces[1]])
                ):
                    continue
                key = _normalize(word)
                spellings.setdefault(key, word)
                found[key] += 1
            for key, frequency in found.items():
                sources[key].add(relative)
                if not duplicate:
                    counts[key] += frequency
    for key in names:
        counts[key] = max(1, counts[key])
    envelope = {
        "sources": source_records,
        "language": language,
        "extra_terms": extra_terms or [],
        "dictionary_version": dictionary_version,
        "algorithm_version": _ALGORITHM_VERSION,
    }
    source_hash = hashlib.sha256(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    entries = [
        GlossaryEntry(
            term=spellings[key],
            frequency=frequency,
            aliases=sorted(aliases[key], key=lambda value: (_normalize(value), value)),
            sources=sorted(sources[key]),
        )
        for key, frequency in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return ContextGlossary(
        language=language,
        entries=entries,
        source_hash=source_hash,
        dictionary_version=dictionary_version,
    )
