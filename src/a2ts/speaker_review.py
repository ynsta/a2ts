"""Interactive speaker attribution review and mapping persistence."""

import json
import subprocess
from pathlib import Path
from typing import Any, overload

from rich.console import Console
from rich.prompt import Prompt

from a2ts.cache import atomic_write_text
from a2ts.media import play_audio_clip_async, stop_audio_playback
from a2ts.models import AlignedTurn, ClusterSplit, SpeakersMapping

console = Console()


def collect_speaker_stats(turns: list[AlignedTurn]) -> dict[str, dict[str, Any]]:
    """Compute turn counts, durations, and quote samples for each cluster ID."""
    stats: dict[str, dict[str, Any]] = {}
    for turn in turns:
        cid = turn.cluster_id
        if cid not in stats:
            stats[cid] = {
                "turn_count": 0,
                "total_duration": 0.0,
                "samples": [],
                "sample_turns": [],
            }
        stats[cid]["turn_count"] += 1
        stats[cid]["total_duration"] = round(
            stats[cid]["total_duration"] + (turn.end - turn.start), 2
        )
        if len(stats[cid]["samples"]) < 4:
            stats[cid]["samples"].append((turn.start, turn.text))
            stats[cid]["sample_turns"].append(turn)
    return stats


def resolve_speaker_for_turn(
    turn_id: int,
    cluster_id: str,
    time_slice_id: int,
    mapping: SpeakersMapping,
    fallback_speaker: str | None = None,
) -> str:
    """Resolve speaker attribution for a single turn based on precedence rules.

    Resolution order:
    1. Specific turn override: mapping.turn_overrides[turn_id]
    2. Time slice override: mapping.slice_overrides[str(time_slice_id)][cluster_id]
    3. Cluster default: mapping.cluster_defaults[cluster_id]
    4. Fallback speaker (if provided and != cluster_id)
    5. Raw cluster ID
    """
    if turn_id in mapping.turn_overrides:
        return mapping.turn_overrides[turn_id]

    slice_key = str(time_slice_id)
    if (
        slice_key in mapping.slice_overrides
        and cluster_id in mapping.slice_overrides[slice_key]
    ):
        return mapping.slice_overrides[slice_key][cluster_id]

    if cluster_id in mapping.cluster_defaults:
        return mapping.cluster_defaults[cluster_id]

    if fallback_speaker and fallback_speaker != cluster_id:
        return fallback_speaker

    return cluster_id


def apply_speakers_mapping(
    turns: list[AlignedTurn], mapping: SpeakersMapping
) -> list[AlignedTurn]:
    """Apply cluster defaults, slice overrides, and turn overrides to aligned turns."""
    updated: list[AlignedTurn] = []
    for turn in turns:
        speaker = resolve_speaker_for_turn(
            turn_id=turn.turn_id,
            cluster_id=turn.cluster_id,
            time_slice_id=turn.time_slice_id,
            mapping=mapping,
            fallback_speaker=turn.speaker,
        )
        updated.append(turn.model_copy(update={"speaker": speaker}))
    return updated


def apply_splits_to_turns(
    turns: list[AlignedTurn], splits: list[ClusterSplit]
) -> list[AlignedTurn]:
    """Apply recorded cluster splits to turns."""
    updated = [t.model_copy() for t in turns]
    for split in splits:
        for turn in updated:
            if turn.cluster_id == split.cluster_id and turn.start >= split.at:
                turn.cluster_id = split.new_cluster_id
    return updated


@overload
def save_speakers_mapping(mapping: SpeakersMapping, path: Path) -> None: ...


@overload
def save_speakers_mapping(mapping: Path, path: SpeakersMapping) -> None: ...


def save_speakers_mapping(
    mapping: SpeakersMapping | Path,
    path: Path | SpeakersMapping,
) -> None:
    """Save mapping configuration to JSON file or session directory atomically."""
    if isinstance(mapping, SpeakersMapping):
        actual_mapping = mapping
        actual_path = Path(path)  # type: ignore[arg-type]
    else:
        actual_mapping = path  # type: ignore[assignment]
        actual_path = Path(mapping)

    target_file = (
        actual_path / "speakers_mapping.json" if actual_path.is_dir() else actual_path
    )
    atomic_write_text(
        target_file,
        json.dumps(actual_mapping.model_dump(), indent=2, ensure_ascii=False),
    )


def load_speakers_mapping(path: Path) -> SpeakersMapping:
    """Load mapping configuration from JSON file or session directory."""
    target_file = path / "speakers_mapping.json" if path.is_dir() else path
    if not target_file.is_file():
        return SpeakersMapping()
    data = json.loads(target_file.read_text(encoding="utf-8"))
    return SpeakersMapping.model_validate(data)


def run_interactive_review(
    turns: list[AlignedTurn],
    candidates: list[str],
    existing_mapping: SpeakersMapping | None = None,
    audio_path: Path | None = None,
    auto_play: bool = True,
    audio_padding: float = 2.0,
    filter_clusters: list[str] | None = None,
    unassigned_only: bool = False,
    min_turns: int = 1,
    min_duration: float = 0.0,
    filter_slice: int | None = None,
    mapping_path: Path | None = None,
) -> SpeakersMapping:
    """Run interactive terminal QCM to attribute speakers."""

    def _persist_progress() -> None:
        if mapping_path is not None:
            save_speakers_mapping(mapping, mapping_path)

    candidates = list(dict.fromkeys(candidates))
    mapping = (
        existing_mapping.model_copy(deep=True)
        if existing_mapping is not None
        else SpeakersMapping()
    )
    turns_to_review = (
        [t for t in turns if t.time_slice_id == filter_slice]
        if filter_slice is not None
        else turns
    )
    stats = collect_speaker_stats(turns_to_review)

    console.print(
        "\n[bold cyan]=== Interactive Speaker Attribution Review ===[/bold cyan]\n"
    )
    for cid, data in stats.items():
        if filter_clusters and cid not in filter_clusters:
            continue
        if unassigned_only:
            curr_assigned = mapping.cluster_defaults.get(cid)
            if curr_assigned and not curr_assigned.startswith("SPEAKER_"):
                continue
        if data["turn_count"] < min_turns:
            continue
        if data["total_duration"] < min_duration:
            continue

        console.print(
            f"[bold yellow]Speaker Cluster: {cid}[/bold yellow] ({data['turn_count']} turns, ~{data['total_duration']:.1f}s)"
        )
        sample_turns: list[AlignedTurn] = data.get("sample_turns", [])
        console.print("Sample quotes:")
        for idx_sample, (t_start, sample) in enumerate(data["samples"], 1):
            m, s = divmod(int(t_start), 60)
            p_tag = (
                f"[bold magenta][p{idx_sample}][/bold magenta] " if audio_path else ""
            )
            console.print(f'  {p_tag}• [[bold]{m:02d}:{s:02d}[/bold]] "{sample}"')

        current = mapping.cluster_defaults.get(cid)
        if current:
            console.print(f"Current mapping: [bold green]{current}[/bold green]")

        console.print("\nCandidate options:")
        opts = {str(i + 1): name for i, name in enumerate(candidates[:10])}
        for idx, name in opts.items():
            console.print(f"  [{idx}] {name}")
        skip_label = (
            f"Keep current ({current})" if current else "Skip (keep cluster ID)"
        )
        if audio_path:
            console.print("  [p] Replay sample 1 audio (or type p1, p2, p3)")
            console.print(
                "  [w] Play wider audio with extra context (or type w1, w2, w3)"
            )
        console.print(f"  [s] {skip_label}")
        console.print("  [t] Review this cluster by time slices (temporal split)")
        console.print("  [c] Type custom name (or type speaker name directly)")
        console.print("  [q] Quit review and save progress")

        active_proc: subprocess.Popen[bytes] | None = None
        if auto_play and audio_path and sample_turns:
            first_turn = sample_turns[0]
            clip_dur = max(1.5, first_turn.end - first_turn.start)
            console.print(
                f"[dim]🔊 Playing sample 1 audio (±{audio_padding:.1f}s context)...[/dim]"
            )
            stop_audio_playback(active_proc)
            active_proc = play_audio_clip_async(
                audio_path,
                start=first_turn.start,
                duration=clip_dur,
                pad_before=audio_padding,
                pad_after=audio_padding,
            )

        while True:
            raw_choice = Prompt.ask(
                "Attribution choice (number, custom name, [p]lay, [w]ide, [s]kip, [t]ime-slice, or [q]uit)",
                default="s",
            ).strip()
            stop_audio_playback(active_proc)

            choice_lower = raw_choice.lower()
            if choice_lower in ("q", "quit", "exit"):
                console.print("[dim]Exited review early. Saving progress...[/dim]\n")
                return mapping
            is_wide = choice_lower in (
                "w",
                "wide",
                "w1",
                "w2",
                "w3",
                "w4",
                "p+",
                "p1+",
                "p2+",
                "p3+",
                "p4+",
            )
            is_play = choice_lower in ("p", "play", "p1", "p2", "p3", "p4") or is_wide

            if is_play:
                if not audio_path:
                    console.print(
                        "[yellow]Audio playback unavailable (no audio file).[/yellow]"
                    )
                    continue
                p_idx = 0
                for prefix in ("w", "p+", "p"):
                    if choice_lower.startswith(prefix) and len(choice_lower) > len(
                        prefix
                    ):
                        tail = choice_lower[len(prefix) :].rstrip("+")
                        if tail.isdigit():
                            p_idx = int(tail) - 1
                            break

                current_pad = audio_padding + 3.0 if is_wide else audio_padding
                pad_lbl = f"±{current_pad:.1f}s"
                if 0 <= p_idx < len(sample_turns):
                    target_turn = sample_turns[p_idx]
                    clip_dur = max(1.5, target_turn.end - target_turn.start)
                    console.print(
                        f"[dim]🔊 Playing sample {p_idx + 1} audio ({pad_lbl} context)...[/dim]"
                    )
                    stop_audio_playback(active_proc)
                    active_proc = play_audio_clip_async(
                        audio_path,
                        start=target_turn.start,
                        duration=clip_dur,
                        pad_before=current_pad,
                        pad_after=current_pad,
                    )
                else:
                    console.print(
                        f"[yellow]Sample quote {p_idx + 1} not found.[/yellow]"
                    )
                continue
            elif raw_choice == "s" or not raw_choice:
                kept = mapping.cluster_defaults.get(cid, cid)
                console.print(f"[dim]Kept {kept}[/dim]\n")
                break
            elif raw_choice == "t":
                slices: dict[int, list[AlignedTurn]] = {}
                for t in turns:
                    if t.cluster_id == cid:
                        slices.setdefault(t.time_slice_id, []).append(t)

                console.print(
                    f"\n[bold magenta]=== Reviewing {cid} by Time Slice ({len(slices)} slices) ===[/bold magenta]"
                )
                for slice_id in sorted(slices.keys()):
                    slice_turns = slices[slice_id]
                    t_start = slice_turns[0].start
                    t_end = slice_turns[-1].end
                    m1, s1 = divmod(int(t_start), 60)
                    m2, s2 = divmod(int(t_end), 60)
                    h1, m1 = divmod(m1, 60)
                    h2, m2 = divmod(m2, 60)
                    ts_label = (
                        f"{h1:02d}:{m1:02d}:{s1:02d} - {h2:02d}:{m2:02d}:{s2:02d}"
                    )

                    curr_override = mapping.slice_overrides.get(str(slice_id), {}).get(
                        cid
                    )
                    curr_str = curr_override or current or cid

                    console.print(
                        f"\n[bold yellow]Slice {slice_id} [{ts_label}][/bold yellow] ({len(slice_turns)} turns)"
                    )
                    slice_samples = slice_turns[:3]
                    console.print("Sample quotes:")
                    for i_s, sample_turn in enumerate(slice_samples, 1):
                        sm, ss = divmod(int(sample_turn.start), 60)
                        sh, sm = divmod(sm, 60)
                        p_tag = (
                            f"[bold magenta][p{i_s}][/bold magenta] "
                            if audio_path
                            else ""
                        )
                        console.print(
                            f'  {p_tag}• [[bold]{sh:02d}:{sm:02d}:{ss:02d}[/bold]] "{sample_turn.text}"'
                        )

                    console.print(
                        f"Current mapping: [bold green]{curr_str}[/bold green]"
                    )
                    if audio_path:
                        console.print(
                            "  [p] Replay slice sample 1 audio (or type p1, p2, p3)"
                        )
                        console.print(
                            "  [w] Play wider slice audio context (or type w1, w2, w3)"
                        )

                    slice_proc: subprocess.Popen[bytes] | None = None
                    if auto_play and audio_path and slice_samples:
                        first_s_turn = slice_samples[0]
                        clip_dur = max(1.5, first_s_turn.end - first_s_turn.start)
                        console.print(
                            f"[dim]🔊 Playing Slice {slice_id} sample 1 audio (±{audio_padding:.1f}s context)...[/dim]"
                        )
                        stop_audio_playback(slice_proc)
                        slice_proc = play_audio_clip_async(
                            audio_path,
                            start=first_s_turn.start,
                            duration=clip_dur,
                            pad_before=audio_padding,
                            pad_after=audio_padding,
                        )

                    while True:
                        slice_choice = Prompt.ask(
                            f"Attribution for Slice {slice_id} (number, custom name, [p]lay, [w]ide, [s]kip, or [q]uit)",
                            default="s",
                        ).strip()
                        stop_audio_playback(slice_proc)

                        s_lower = slice_choice.lower()
                        if s_lower in ("q", "quit", "exit"):
                            console.print("[dim]Exited slice review.[/dim]\n")
                            break
                        s_is_wide = s_lower in (
                            "w",
                            "wide",
                            "w1",
                            "w2",
                            "w3",
                            "p+",
                            "p1+",
                            "p2+",
                            "p3+",
                        )
                        s_is_play = (
                            s_lower in ("p", "play", "p1", "p2", "p3") or s_is_wide
                        )

                        if s_is_play:
                            if not audio_path:
                                console.print(
                                    "[yellow]Audio playback unavailable.[/yellow]"
                                )
                                continue
                            p_idx = 0
                            for prefix in ("w", "p+", "p"):
                                if s_lower.startswith(prefix) and len(s_lower) > len(
                                    prefix
                                ):
                                    tail = s_lower[len(prefix) :].rstrip("+")
                                    if tail.isdigit():
                                        p_idx = int(tail) - 1
                                        break

                            s_pad = audio_padding + 3.0 if s_is_wide else audio_padding
                            pad_lbl = f"±{s_pad:.1f}s"
                            if 0 <= p_idx < len(slice_samples):
                                t_play = slice_samples[p_idx]
                                clip_dur = max(1.5, t_play.end - t_play.start)
                                console.print(
                                    f"[dim]🔊 Playing Slice {slice_id} sample {p_idx + 1} audio ({pad_lbl} context)...[/dim]"
                                )
                                stop_audio_playback(slice_proc)
                                slice_proc = play_audio_clip_async(
                                    audio_path,
                                    start=t_play.start,
                                    duration=clip_dur,
                                    pad_before=s_pad,
                                    pad_after=s_pad,
                                )
                            else:
                                console.print(
                                    f"[yellow]Sample quote {p_idx + 1} not found.[/yellow]"
                                )
                            continue
                        elif slice_choice == "s" or not slice_choice:
                            console.print(f"[dim]Kept {curr_str}[/dim]")
                            break
                        elif slice_choice in opts:
                            mapping.slice_overrides.setdefault(str(slice_id), {})[
                                cid
                            ] = opts[slice_choice]
                            mapping.label_sources[cid] = "manual"
                            console.print(
                                f"[green]✓ Mapped {cid} (Slice {slice_id}) -> {opts[slice_choice]}[/green]"
                            )
                            _persist_progress()
                            break
                        elif slice_choice == "c":
                            c_name = Prompt.ask("Enter custom speaker name").strip()
                            if c_name:
                                mapping.slice_overrides.setdefault(str(slice_id), {})[
                                    cid
                                ] = c_name
                                mapping.label_sources[cid] = "manual"
                                console.print(
                                    f"[green]✓ Mapped {cid} (Slice {slice_id}) -> {c_name}[/green]"
                                )
                                _persist_progress()
                            break
                        else:
                            mapping.slice_overrides.setdefault(str(slice_id), {})[
                                cid
                            ] = slice_choice
                            mapping.label_sources[cid] = "manual"
                            console.print(
                                f"[green]✓ Mapped {cid} (Slice {slice_id}) -> {slice_choice}[/green]"
                            )
                            _persist_progress()
                            break
                console.print("")
                break
            elif raw_choice in opts:
                mapping.cluster_defaults[cid] = opts[raw_choice]
                mapping.label_sources[cid] = "manual"
                console.print(f"[green]✓ Mapped {cid} -> {opts[raw_choice]}[/green]\n")
                _persist_progress()
                break
            elif raw_choice == "c":
                custom_name = Prompt.ask("Enter custom speaker name").strip()
                if custom_name:
                    mapping.cluster_defaults[cid] = custom_name
                    mapping.label_sources[cid] = "manual"
                    console.print(f"[green]✓ Mapped {cid} -> {custom_name}[/green]\n")
                    _persist_progress()
                break
            else:
                mapping.cluster_defaults[cid] = raw_choice
                mapping.label_sources[cid] = "manual"
                console.print(f"[green]✓ Mapped {cid} -> {raw_choice}[/green]\n")
                _persist_progress()
                break

    return mapping
