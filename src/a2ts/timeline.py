"""Time-slice segmentation, cluster splitting, and word-to-speaker alignment."""

from bisect import bisect_left, bisect_right

from a2ts.models import AlignedTurn, RawSegment, SpeakerTurn, WordTimestamp


def assign_time_slices(
    turns: list[SpeakerTurn], slice_minutes: float = 15.0
) -> list[SpeakerTurn]:
    """Assign time slice IDs to speaker turns based on their start timestamps."""
    slice_seconds = slice_minutes * 60.0
    updated: list[SpeakerTurn] = []
    for turn in turns:
        slice_id = int(turn.start // slice_seconds)
        updated.append(turn.model_copy(update={"time_slice_id": slice_id}))
    return updated


def split_cluster_at_time[TurnT: (SpeakerTurn, AlignedTurn)](
    turns: list[TurnT],
    cluster_id: str,
    split_time: float,
    new_cluster_id: str,
) -> list[TurnT]:
    """Reassign all turns of cluster_id occurring at or after split_time to new_cluster_id."""
    updated: list[TurnT] = []
    for turn in turns:
        if turn.cluster_id == cluster_id and turn.start >= split_time:
            updated.append(turn.model_copy(update={"cluster_id": new_cluster_id}))
        else:
            updated.append(turn)
    return updated


def detect_outlier_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Detect outlier turns, such as flagged turns or low confidence turns."""
    return [turn for turn in turns if turn.flagged]


def align_words_to_speaker_turns(
    segments: list[RawSegment],
    turns: list[SpeakerTurn],
) -> list[AlignedTurn]:
    """Align word timestamps to overlapping speaker turns to produce AlignedTurn blocks."""
    if not turns:
        # Fallback when no diarization turns exist: assign whole segments to unknown speaker
        return [
            AlignedTurn(
                turn_id=seg.id,
                start=seg.start,
                end=seg.end,
                speaker="SPEAKER_00",
                cluster_id="SPEAKER_00",
                text=seg.text,
                words=seg.words,
                time_slice_id=0,
            )
            for seg in segments
        ]

    # Sort turns by start time with stable tie-breaking on turn_id
    sorted_turns = sorted(turns, key=lambda t: (t.start, t.id))
    turn_starts = [t.start for t in sorted_turns]
    max_turn_duration = max((t.end - t.start for t in sorted_turns), default=0.0)

    # Flatten all words from segments
    all_words: list[WordTimestamp] = []
    for seg in segments:
        if seg.words:
            all_words.extend(seg.words)
        else:
            # If engine produced no word timestamps, treat segment as one block
            all_words.append(WordTimestamp(word=seg.text, start=seg.start, end=seg.end))

    all_words.sort(key=lambda word: word.start)
    # Join adjacent lexical continuations without changing original token timestamps.
    groups: list[list[WordTimestamp]] = []
    for word in all_words:
        token = word.word.strip()
        previous = groups[-1][-1] if groups else None
        continuation = token.startswith(("'", "’", "-", "‐", "‑")) or (
            bool(token) and not any(c.isalnum() for c in token)
        )
        if (
            previous is not None
            and word.start - previous.end <= 0.2
            and (
                continuation
                or previous.word.rstrip().endswith(("'", "’", "-", "‐", "‑"))
            )
        ):
            groups[-1].append(word)
        else:
            groups.append([word])

    aligned_turns: list[AlignedTurn] = []
    seen_ids: set[int] = set()
    next_id = max(t.id for t in turns) + 1
    for group in groups:
        anchor = group[0]
        group_end = max(w.end for w in group)
        midpoint = (anchor.start + group_end) / 2.0
        left = bisect_left(turn_starts, anchor.start - max_turn_duration)
        right = bisect_right(turn_starts, group_end)

        def overlap(turn: SpeakerTurn, word: WordTimestamp) -> float:
            return max(0.0, min(turn.end, word.end) - max(turn.start, word.start))

        candidates = [
            t for t in sorted_turns[left:right] if any(overlap(t, w) > 0 for w in group)
        ]
        if not candidates and anchor.start == anchor.end:
            candidates = [
                t for t in sorted_turns[left:right] if t.start <= midpoint <= t.end
            ]
        if candidates:
            matched = min(
                candidates,
                key=lambda t: (
                    -sum(overlap(t, w) for w in group),
                    abs((t.start + t.end) / 2.0 - midpoint),
                    t.start,
                    t.id,
                ),
            )
        else:
            idx = bisect_right(turn_starts, anchor.start)
            candidates = sorted_turns[
                max(0, min(idx - 5, left)) : min(len(turns), max(idx + 5, right))
            ]
            matched = min(
                candidates,
                key=lambda t: min(abs(t.start - midpoint), abs(t.end - midpoint)),
            )
        nearby = sorted_turns[left : bisect_right(turn_starts, group_end)]
        clusters = {
            t.cluster_id
            for t in nearby
            if any(
                overlap(t, w) > 0 or (w.start == w.end and t.start <= w.start <= t.end)
                for w in group
            )
        }
        uncertain = len(clusters) > 1
        # Only extend the current run; revisiting a source turn must not reorder words.
        if aligned_turns and aligned_turns[-1].source_turn_id == matched.id:
            current = aligned_turns[-1]
            current.words.extend(group)
            current.end = max(current.end, group_end)
            current.speaker_uncertain |= uncertain
        else:
            run_id = matched.id if matched.id not in seen_ids else next_id
            if matched.id in seen_ids:
                next_id += 1
            seen_ids.add(matched.id)
            aligned_turns.append(
                AlignedTurn(
                    turn_id=run_id,
                    source_turn_id=matched.id,
                    start=anchor.start,
                    end=group_end,
                    speaker=matched.resolved_speaker or matched.cluster_id,
                    cluster_id=matched.cluster_id,
                    text="",
                    words=list(group),
                    time_slice_id=matched.time_slice_id,
                    speaker_uncertain=uncertain,
                )
            )
    for run in aligned_turns:
        text = ""
        previous_token = ""
        previous_end: float | None = None
        for word in run.words:
            token = word.word.strip()
            attach = (
                token.startswith(("'", "’", "-", "‐", "‑"))
                or (bool(token) and not any(c.isalnum() for c in token))
                or previous_token.endswith(("'", "’", "-", "‐", "‑"))
            )
            attach = (
                attach and previous_end is not None and word.start - previous_end <= 0.2
            )
            text += ("" if not text or attach else " ") + token
            previous_token = token
            previous_end = word.end
        run.text = text
    return aligned_turns
