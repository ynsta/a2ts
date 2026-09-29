"""Turn debouncing, chronological sorting, and markdown transcript generation."""

import re

from a2ts.models import AlignedTurn

DICE_SIDES_MAP = {
    "4": "4",
    "quatre": "4",
    "6": "6",
    "six": "6",
    "8": "8",
    "huit": "8",
    "10": "10",
    "dix": "10",
    "12": "12",
    "douze": "12",
    "20": "20",
    "vingt": "20",
    "100": "100",
    "cent": "100",
}

DICE_COUNT_MAP = {
    "un": "1",
    "une": "1",
    "1": "1",
    "deux": "2",
    "2": "2",
    "trois": "3",
    "3": "3",
    "quatre": "4",
    "4": "4",
    "cinq": "5",
    "5": "5",
    "six": "6",
    "6": "6",
}

EXCLUDED_FOLLOWING_NOUNS = (
    "gardes|joueurs|soldats|membres|ans|minutes|secondes|jours|heures|mètres|personnages|ennemis|individus|points"
)

DICE_PATTERN = re.compile(
    r"(?<!\bchacun\s)(?<!\bchacune\s)(?<!\bl\')(?:(?<=\s)|(?<=^))"
    r"(?:(un|une|deux|trois|quatre|cinq|six|[1-9]\d*)\s+)?"
    r"(?:des?|dés?)\s+(?:de\s+)?"
    r"(4|6|8|10|12|20|100|quatre|six|huit|dix|douze|vingt|cent)\b"
    rf"(?!\s+(?:{EXCLUDED_FOLLOWING_NOUNS}))",
    re.IGNORECASE,
)


def normalize_rpg_terms(text: str) -> str:
    """Normalize common phonetic tabletop RPG speech recognition artifacts."""
    def _jet_repl(m: re.Match[str]) -> str:
        return "jets de dés" if m.group(2) else "jet de dés"

    out = re.sub(r"\b(jet)(s)?\s*-\s*dés?\b", _jet_repl, text, flags=re.IGNORECASE)
    out = re.sub(r"\b(jet)(s)?\s+de\s+délai\b", _jet_repl, out, flags=re.IGNORECASE)

    def _dice_repl(m: re.Match[str]) -> str:
        count_match = m.group(1)
        sides_match = m.group(2)
        count = (
            DICE_COUNT_MAP.get(count_match.lower(), count_match)
            if count_match
            else "1"
        )
        sides = DICE_SIDES_MAP.get(sides_match.lower(), sides_match)
        prefix = (
            "1d"
            if (not count_match or count_match.lower() in ("un", "une", "1"))
            else f"{count}d"
        )
        return f"{prefix}{sides}"

    return DICE_PATTERN.sub(_dice_repl, out)


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    total_sec = max(0, int(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def debounce_consecutive_turns(
    turns: list[AlignedTurn],
    threshold_seconds: float = 2.0,
) -> list[AlignedTurn]:
    """Merge consecutive speech turns from the same speaker if the gap is below threshold."""
    if not turns:
        return []

    sorted_turns = sorted(turns, key=lambda t: t.start)
    debounced: list[AlignedTurn] = [sorted_turns[0]]

    for current in sorted_turns[1:]:
        prev = debounced[-1]
        gap = current.start - prev.end

        if current.speaker == prev.speaker and gap <= threshold_seconds:
            merged_words = prev.words + current.words
            merged_text = f"{prev.text} {current.text}".strip()
            debounced[-1] = prev.model_copy(
                update={
                    "end": max(prev.end, current.end),
                    "text": merged_text,
                    "words": merged_words,
                }
            )
        else:
            debounced.append(current)

    normalized_debounced: list[AlignedTurn] = []
    for turn in debounced:
        norm_text = normalize_rpg_terms(turn.text)
        if norm_text != turn.text:
            normalized_debounced.append(turn.model_copy(update={"text": norm_text}))
        else:
            normalized_debounced.append(turn)

    return normalized_debounced


def render_markdown_transcript(turns: list[AlignedTurn]) -> str:
    """Render aligned turns into standard Markdown with timestamp headings."""
    if not turns:
        return ""

    lines: list[str] = []
    for turn in turns:
        start_str = format_timestamp(turn.start)
        end_str = format_timestamp(turn.end)
        lines.append(f"### [{start_str} - {end_str}] {turn.speaker}")
        lines.append("")
        lines.append(normalize_rpg_terms(turn.text))
        lines.append("")
    return "\n".join(lines).strip() + "\n"
