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
        opts = {str(i + 1): name for i, name in enumerate(candidates[:6])}
        for idx, name in opts.items():
            console.print(f"  [{idx}] {name}")
        console.print("  [c] Type custom name")
        skip_label = (
            f"Keep current ({current})" if current else "Skip (keep cluster ID)"
        )
        console.print(f"  [s] {skip_label}")

        choice = Prompt.ask(
            "Attribution choice", choices=list(opts.keys()) + ["c", "s"], default="s"
        )
        if choice in opts:
            mapping.cluster_defaults[cid] = opts[choice]
            console.print(f"[green]✓ Mapped {cid} -> {opts[choice]}[/green]\n")
        elif choice == "c":
            custom_name = Prompt.ask("Enter custom speaker name").strip()
            if custom_name:
                mapping.cluster_defaults[cid] = custom_name
                console.print(f"[green]✓ Mapped {cid} -> {custom_name}[/green]\n")
        else:
            kept = mapping.cluster_defaults.get(cid, cid)
            console.print(f"[dim]Kept {kept}[/dim]\n")

    return mapping
