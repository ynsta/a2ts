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

    aligned_turns: list[AlignedTurn] = []
    words_by_turn: dict[int, list[WordTimestamp]] = {t.id: [] for t in sorted_turns}

    for word in all_words:
        mid_point = (word.start + word.end) / 2.0

        # Find nearby candidate overlapping turns using bisect
        idx = bisect_right(turn_starts, word.start)
        cand_start = bisect_left(turn_starts, mid_point - max_turn_duration)
        cand_end = bisect_right(turn_starts, mid_point)

        matched_turn: SpeakerTurn | None = None
        for i in range(cand_start, cand_end):
            t = sorted_turns[i]
            if t.start <= mid_point <= t.end:
                matched_turn = t
                break

        if matched_turn is None:
            # Find nearest turn among immediate neighbors in local window O(log N)
            cand_left = max(0, min(idx - 5, cand_start))
            cand_right = min(len(sorted_turns), max(idx + 5, cand_end))
            matched_turn = min(
                sorted_turns[cand_left:cand_right],
                key=lambda t: min(abs(t.start - mid_point), abs(t.end - mid_point)),
            )

        words_by_turn[matched_turn.id].append(word)

    for turn in sorted_turns:
        turn_words = words_by_turn[turn.id]
        if not turn_words:
            continue
        text = " ".join(w.word for w in turn_words).strip()
        speaker = turn.resolved_speaker if turn.resolved_speaker else turn.cluster_id
        aligned_turns.append(
            AlignedTurn(
                turn_id=turn.id,
                start=turn_words[0].start,
                end=turn_words[-1].end,
                speaker=speaker,
                cluster_id=turn.cluster_id,
                text=text,
                words=turn_words,
                time_slice_id=turn.time_slice_id,
            )
        )

    return sorted(aligned_turns, key=lambda t: t.start)
