"""Interactive speaker attribution review and mapping persistence."""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Prompt

from a2ts.models import AlignedTurn, SpeakersMapping

console = Console()


def collect_speaker_stats(turns: list[AlignedTurn]) -> dict[str, dict[str, Any]]:
    """Compute turn counts, durations, and quote samples for each cluster ID."""
    stats: dict[str, dict[str, Any]] = {}
    for turn in turns:
        cid = turn.cluster_id
        if cid not in stats:
            stats[cid] = {"turn_count": 0, "total_duration": 0.0, "samples": []}
        stats[cid]["turn_count"] += 1
        stats[cid]["total_duration"] = round(
            stats[cid]["total_duration"] + (turn.end - turn.start), 2
        )
        if len(stats[cid]["samples"]) < 4:
            stats[cid]["samples"].append((turn.start, turn.text))
    return stats


def apply_speakers_mapping(
    turns: list[AlignedTurn], mapping: SpeakersMapping
) -> list[AlignedTurn]:
    """Apply cluster defaults, slice overrides, and turn overrides to aligned turns."""
    updated: list[AlignedTurn] = []
    for turn in turns:
        slice_key = str(turn.time_slice_id)
        speaker = turn.cluster_id

        # 1. Cluster default
        if turn.cluster_id in mapping.cluster_defaults:
            speaker = mapping.cluster_defaults[turn.cluster_id]

        # 2. Time slice override
        if (
            slice_key in mapping.slice_overrides
            and turn.cluster_id in mapping.slice_overrides[slice_key]
        ):
            speaker = mapping.slice_overrides[slice_key][turn.cluster_id]

        # 3. Specific turn override
        if turn.turn_id in mapping.turn_overrides:
            speaker = mapping.turn_overrides[turn.turn_id]

        updated.append(turn.model_copy(update={"speaker": speaker}))
    return updated


def save_speakers_mapping(mapping: SpeakersMapping, path: Path) -> None:
    """Save mapping configuration to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(mapping.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_speakers_mapping(path: Path) -> SpeakersMapping:
    """Load mapping configuration from JSON file."""
    if not path.is_file():
        return SpeakersMapping()
    data = json.loads(path.read_text(encoding="utf-8"))
    return SpeakersMapping.model_validate(data)


def run_interactive_review(
    turns: list[AlignedTurn],
    candidates: list[str],
    existing_mapping: SpeakersMapping | None = None,
) -> SpeakersMapping:
    """Run interactive terminal QCM to attribute speakers."""
    candidates = list(dict.fromkeys(candidates))
    mapping = (
        existing_mapping.model_copy(deep=True)
        if existing_mapping is not None
        else SpeakersMapping()
    )
    stats = collect_speaker_stats(turns)

    console.print(
        "\n[bold cyan]=== Interactive Speaker Attribution Review ===[/bold cyan]\n"
    )
    for cid, data in stats.items():
        console.print(
            f"[bold yellow]Speaker Cluster: {cid}[/bold yellow] ({data['turn_count']} turns, ~{data['total_duration']:.1f}s)"
        )
        console.print("Sample quotes:")
        for t_start, sample in data["samples"]:
            m, s = divmod(int(t_start), 60)
            console.print(f'  • [[bold]{m:02d}:{s:02d}[/bold]] "{sample}"')

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
        console.print(f"  [s] {skip_label}")
        console.print("  [t] Review this cluster by time slices (temporal split)")
        console.print("  [c] Type custom name (or type speaker name directly)")

        raw_choice = Prompt.ask(
            "Attribution choice (number, custom name, [s]kip, or [t]ime-slice)",
            default="s",
        ).strip()
        if raw_choice == "s" or not raw_choice:
            kept = mapping.cluster_defaults.get(cid, cid)
            console.print(f"[dim]Kept {kept}[/dim]\n")
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
                ts_label = f"{h1:02d}:{m1:02d}:{s1:02d} - {h2:02d}:{m2:02d}:{s2:02d}"

                curr_override = mapping.slice_overrides.get(str(slice_id), {}).get(cid)
                curr_str = curr_override or current or cid

                console.print(
                    f"\n[bold yellow]Slice {slice_id} [{ts_label}][/bold yellow] ({len(slice_turns)} turns)"
                )
                console.print("Sample quotes:")
                for sample_turn in slice_turns[:3]:
                    sm, ss = divmod(int(sample_turn.start), 60)
                    sh, sm = divmod(sm, 60)
                    console.print(
                        f'  • [[bold]{sh:02d}:{sm:02d}:{ss:02d}[/bold]] "{sample_turn.text}"'
                    )

                console.print(f"Current mapping: [bold green]{curr_str}[/bold green]")
                slice_choice = Prompt.ask(
                    f"Attribution for Slice {slice_id} (number, custom name, or [s]kip)",
                    default="s",
                ).strip()

                if slice_choice == "s" or not slice_choice:
                    console.print(f"[dim]Kept {curr_str}[/dim]")
                elif slice_choice in opts:
                    mapping.slice_overrides.setdefault(str(slice_id), {})[cid] = opts[
                        slice_choice
                    ]
                    console.print(
                        f"[green]✓ Mapped {cid} (Slice {slice_id}) -> {opts[slice_choice]}[/green]"
                    )
                elif slice_choice == "c":
                    c_name = Prompt.ask("Enter custom speaker name").strip()
                    if c_name:
                        mapping.slice_overrides.setdefault(str(slice_id), {})[cid] = (
                            c_name
                        )
                        console.print(
                            f"[green]✓ Mapped {cid} (Slice {slice_id}) -> {c_name}[/green]"
                        )
                else:
                    mapping.slice_overrides.setdefault(str(slice_id), {})[cid] = (
                        slice_choice
                    )
                    console.print(
                        f"[green]✓ Mapped {cid} (Slice {slice_id}) -> {slice_choice}[/green]"
                    )
            console.print("")
        elif raw_choice in opts:
            mapping.cluster_defaults[cid] = opts[raw_choice]
            console.print(f"[green]✓ Mapped {cid} -> {opts[raw_choice]}[/green]\n")
        elif raw_choice == "c":
            custom_name = Prompt.ask("Enter custom speaker name").strip()
            if custom_name:
                mapping.cluster_defaults[cid] = custom_name
                console.print(f"[green]✓ Mapped {cid} -> {custom_name}[/green]\n")
        else:
            mapping.cluster_defaults[cid] = raw_choice
            console.print(f"[green]✓ Mapped {cid} -> {raw_choice}[/green]\n")

    return mapping
