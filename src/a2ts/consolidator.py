"""Turn debouncing, chronological sorting, and markdown transcript generation."""

from a2ts.models import AlignedTurn


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

    return debounced


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
        lines.append(turn.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
