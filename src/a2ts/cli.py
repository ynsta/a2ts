"""CLI application and pipeline orchestration for a2ts."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from a2ts.consolidator import debounce_consecutive_turns, render_markdown_transcript
from a2ts.media import extract_audio_to_wav
from a2ts.models import SpeakerTurn
from a2ts.refiner import refine_transcript_markdown
from a2ts.speaker_review import (
    apply_speakers_mapping,
    load_speakers_mapping,
    run_interactive_review,
    save_speakers_mapping,
)
from a2ts.timeline import (
    align_words_to_speaker_turns,
    assign_time_slices,
)
from a2ts.transcriber import get_engine
from a2ts.vocab import (
    build_biasing_prompt,
    load_wordlist_file,
    scan_context_directory,
)

app = typer.Typer(help="a2ts: Contextualized Audio/Video Transcriber")
console = Console()


@app.command()
def info() -> None:
    """Show system information and a2ts configuration."""
    console.print("[bold green]a2ts[/bold green] - Audio/Video Transcriber ready.")


@app.command()
def extract_audio(
    media_file: Annotated[
        Path, typer.Argument(help="Path to audio or video media file")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save extracted WAV")
    ] = Path(".a2ts/audio_cache"),
) -> None:
    """Extract and resample audio from video/audio to 16kHz mono WAV."""
    wav_path = extract_audio_to_wav(media_file, output_dir)
    console.print(f"[green]Audio extracted:[/green] {wav_path}")


@app.command()
def extract_vocab(
    context_dir: Annotated[
        Path, typer.Option(help="Directory containing Obsidian markdown notes")
    ] = Path("contexte"),
    vocab_file: Annotated[
        Path | None, typer.Option(help="Optional custom wordlist text file")
    ] = None,
    max_tokens: Annotated[
        int, typer.Option(help="Maximum token budget for prompt")
    ] = 220,
) -> None:
    """Extract lore entities and print token-budgeted prompt."""
    entities = scan_context_directory(context_dir)
    if vocab_file:
        entities.extend(load_wordlist_file(vocab_file))
    prompt = build_biasing_prompt(entities, max_tokens=max_tokens)
    console.print(f"[bold cyan]Extracted {len(entities)} entity mentions.[/bold cyan]")
    console.print(f"[bold green]Biasing Prompt:[/bold green]\n{prompt}")


@app.command()
def run(
    media_file: Annotated[
        Path, typer.Argument(help="Path to audio or video media file")
    ],
    engine: Annotated[
        str, typer.Option(help="Transcription engine: voxtral or whisper")
    ] = "whisper",
    context_dir: Annotated[
        Path, typer.Option(help="Directory containing Obsidian notes")
    ] = Path("contexte"),
    vocab_file: Annotated[
        Path | None, typer.Option(help="Optional custom wordlist text file")
    ] = None,
    slice_minutes: Annotated[
        float, typer.Option(help="Time slice window size in minutes")
    ] = 15.0,
    interactive: Annotated[
        bool, typer.Option(help="Enable interactive QCM speaker review")
    ] = True,
    refine: Annotated[
        bool, typer.Option(help="Run local LLM refiner pass via agy")
    ] = False,
    output: Annotated[
        Path, typer.Option(help="Output markdown transcript path")
    ] = Path("transcript.md"),
    cache_dir: Annotated[
        Path, typer.Option(help="Cache directory for session artifacts")
    ] = Path(".a2ts"),
) -> None:
    """Execute end-to-end transcription and diarization pipeline."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"\n[bold green]=== Starting a2ts Pipeline for {media_file.name} ===[/bold green]\n"
    )

    # 1. Extract audio
    console.print("[bold]Step 1: Extracting audio stream...[/bold]")
    audio_path = extract_audio_to_wav(media_file, cache_dir / "audio_cache")

    # 2. Mine lore & vocabulary
    console.print("[bold]Step 2: Mining lore context...[/bold]")
    entities = scan_context_directory(context_dir)
    if vocab_file:
        entities.extend(load_wordlist_file(vocab_file))
    prompt = build_biasing_prompt(entities) if entities else None

    # 3. Transcribe audio
    console.print(f"[bold]Step 3: Transcribing with engine '{engine}'...[/bold]")
    transcriber = get_engine(engine)
    raw_segments = transcriber.transcribe(audio_path, prompt=prompt)

    # 4. Temporal slicing & alignment
    console.print("[bold]Step 4: Time-slice segmentation and alignment...[/bold]")
    dummy_turns = [
        SpeakerTurn(
            id=i,
            start=seg.start,
            end=seg.end,
            cluster_id=f"SPEAKER_{i % 2:02d}",
        )
        for i, seg in enumerate(raw_segments)
    ]
    sliced_turns = assign_time_slices(dummy_turns, slice_minutes=slice_minutes)
    aligned_turns = align_words_to_speaker_turns(raw_segments, sliced_turns)

    # 5. Interactive speaker review
    mapping_path = cache_dir / "speakers_mapping.json"
    mapping = load_speakers_mapping(mapping_path)
    if interactive:
        candidate_names = [e.name for e in entities]
        mapping = run_interactive_review(aligned_turns, candidate_names)
        save_speakers_mapping(mapping, mapping_path)

    aligned_turns = apply_speakers_mapping(aligned_turns, mapping)

    # 6. Consolidation & Debouncing
    console.print("[bold]Step 5: Consolidating transcript...[/bold]")
    debounced_turns = debounce_consecutive_turns(aligned_turns)
    raw_md = render_markdown_transcript(debounced_turns)

    # 7. Optional LLM Refinement
    final_md = raw_md
    if refine:
        console.print("[bold]Step 6: Refining transcript via local agy...[/bold]")
        final_md = refine_transcript_markdown(raw_md)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(final_md, encoding="utf-8")
    console.print(
        f"\n[bold green]✓ Pipeline completed. Transcript saved to:[/bold green] {output}\n"
    )


if __name__ == "__main__":
    app()
