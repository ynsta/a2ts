"""CLI application and pipeline orchestration for a2ts."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from a2ts.consolidator import debounce_consecutive_turns, render_markdown_transcript
from a2ts.diarizer import diarize_segments
from a2ts.media import compute_file_hash, extract_audio_to_wav, probe_media
from a2ts.models import AlignedTurn, RawSegment, SessionMetadata, SpeakerTurn
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
    split_cluster_at_time,
)
from a2ts.transcriber import get_engine
from a2ts.vocab import (
    build_biasing_prompt,
    load_speaker_names,
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
    model_name: Annotated[
        str | None, typer.Option(help="Model name or path for engine")
    ] = None,
    device: Annotated[
        str, typer.Option(help="Device to run inference on (cuda/cpu)")
    ] = "cuda",
    compute_type: Annotated[
        str, typer.Option(help="Computation type (float16/int8/etc.)")
    ] = "float16",
    context_dir: Annotated[
        Path, typer.Option(help="Directory containing Obsidian notes")
    ] = Path("contexte"),
    vocab_file: Annotated[
        Path | None, typer.Option(help="Optional custom wordlist text file")
    ] = None,
    slice_minutes: Annotated[
        float, typer.Option(help="Time slice window size in minutes")
    ] = 15.0,
    diarize: Annotated[
        bool, typer.Option(help="Enable acoustic speaker diarization")
    ] = True,
    num_speakers: Annotated[
        int | None,
        typer.Option(help="Exact number of speakers to detect (optional)"),
    ] = None,
    speakers: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated list of known speaker names (e.g. 'MJ, Brakk, Oskel')"
        ),
    ] = None,
    speakers_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to speakers text file (default auto-detects contexte/speakers.txt)"
        ),
    ] = None,
    interactive: Annotated[
        bool, typer.Option(help="Enable interactive QCM speaker review")
    ] = True,
    auto_play: Annotated[
        bool, typer.Option(help="Auto-play audio sample during speaker review")
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
    file_hash = compute_file_hash(media_file)
    audio_path = extract_audio_to_wav(media_file, cache_dir / "audio_cache")

    # 2. Mine lore & vocabulary
    console.print("[bold]Step 2: Mining lore context...[/bold]")
    entities = scan_context_directory(context_dir)
    if vocab_file:
        entities.extend(load_wordlist_file(vocab_file))
    prompt = build_biasing_prompt(entities) if entities else None

    # 3. Transcribe audio with caching
    transcripts_dir = cache_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcript_cache = transcripts_dir / f"{file_hash}_{engine}.json"

    if transcript_cache.is_file():
        console.print(
            f"[bold cyan]Step 3: Loading cached raw transcription from {transcript_cache}...[/bold cyan]"
        )
        cached_data = json.loads(transcript_cache.read_text(encoding="utf-8"))
        raw_segments = [RawSegment.model_validate(seg) for seg in cached_data]
    else:
        console.print(f"[bold]Step 3: Transcribing with engine '{engine}'...[/bold]")
        engine_kwargs: dict[str, Any] = {}
        if engine == "whisper":
            engine_kwargs["device"] = device
            engine_kwargs["compute_type"] = compute_type
            if model_name:
                engine_kwargs["model_name"] = model_name
        elif engine == "voxtral":
            if model_name:
                engine_kwargs["model_name"] = model_name
        transcriber = get_engine(engine, **engine_kwargs)
        raw_segments = transcriber.transcribe(audio_path, prompt=prompt)
        transcript_cache.write_text(
            json.dumps(
                [s.model_dump() for s in raw_segments], indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )

    # 4. Acoustic Diarization & Temporal slicing
    console.print("[bold]Step 4: Acoustic speaker diarization & alignment...[/bold]")
    diarization_dir = cache_dir / "diarization"
    diarization_dir.mkdir(parents=True, exist_ok=True)
    diar_cache = diarization_dir / f"{file_hash}.json"

    if diarize:
        speaker_turns = diarize_segments(
            audio_path=audio_path,
            segments=raw_segments,
            num_speakers=num_speakers,
            device=device,
            cache_path=diar_cache,
        )
    else:
        speaker_turns = [
            SpeakerTurn(
                id=i,
                start=seg.start,
                end=seg.end,
                cluster_id="SPEAKER_00",
            )
            for i, seg in enumerate(raw_segments)
        ]

    sliced_turns = assign_time_slices(speaker_turns, slice_minutes=slice_minutes)
    aligned_turns = align_words_to_speaker_turns(raw_segments, sliced_turns)

    # Cache aligned turns for review and split subcommands
    turns_path = cache_dir / "turns.json"
    turns_path.write_text(
        json.dumps(
            [t.model_dump() for t in aligned_turns], indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    # 5. Interactive speaker review
    mapping_path = cache_dir / "speakers_mapping.json"
    mapping = load_speakers_mapping(mapping_path)
    if interactive:
        known_speakers = load_speaker_names(
            speakers_file=speakers_file,
            speakers_arg=speakers,
            context_dir=context_dir,
        )
        candidate_names = list(
            dict.fromkeys(known_speakers + [e.name for e in entities])
        )
        mapping = run_interactive_review(
            aligned_turns,
            candidate_names,
            existing_mapping=mapping,
            audio_path=audio_path,
            auto_play=auto_play,
        )
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

    # 8. Save session metadata
    duration = 0.0
    try:
        probe_info = probe_media(media_file)
        duration = float(probe_info.get("format", {}).get("duration", 0.0))
    except (RuntimeError, FileNotFoundError, OSError, KeyError, ValueError):
        if raw_segments:
            duration = max(s.end for s in raw_segments)

    resolved_model_name = model_name or (
        "large-v3" if engine == "whisper" else "mistralai/Voxtral-Mini-3B-2507"
    )
    prompt_hash = (
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16] if prompt else ""
    )
    session_meta = SessionMetadata(
        media_path=str(media_file),
        media_hash=file_hash,
        duration_seconds=duration,
        engine=engine,
        model_name=resolved_model_name,
        prompt_hash=prompt_hash,
        time_slice_minutes=slice_minutes,
        created_at=datetime.now(UTC).isoformat(),
        output_path=str(output),
    )
    (cache_dir / "session.json").write_text(
        json.dumps(session_meta.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    console.print(
        f"\n[bold green]✓ Pipeline completed. Transcript saved to:[/bold green] {output}\n"
    )


@app.command()
def review(
    session_dir: Annotated[
        Path, typer.Argument(help="Path to session cache directory (.a2ts)")
    ] = Path(".a2ts"),
    context_dir: Annotated[
        Path, typer.Option(help="Directory containing Obsidian notes and speakers.txt")
    ] = Path("contexte"),
    speakers: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated list of known speaker names (e.g. 'MJ, Brakk, Oskel')"
        ),
    ] = None,
    speakers_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to speakers text file (default auto-detects contexte/speakers.txt)"
        ),
    ] = None,
    output: Annotated[Path, typer.Option(help="Output markdown path")] = Path(
        "transcript.md"
    ),
    auto_play: Annotated[
        bool, typer.Option(help="Auto-play audio sample during speaker review")
    ] = True,
) -> None:
    """Re-run interactive speaker review on cached session turns."""
    turns_path = session_dir / "turns.json"
    if not turns_path.is_file():
        console.print(f"[bold red]Turns cache not found at {turns_path}[/bold red]")
        raise typer.Exit(code=1)

    turns_data = json.loads(turns_path.read_text(encoding="utf-8"))
    turns = [AlignedTurn.model_validate(t) for t in turns_data]

    audio_path: Path | None = None
    session_path = session_dir / "session.json"
    if session_path.is_file():
        try:
            session_meta = SessionMetadata.model_validate_json(
                session_path.read_text(encoding="utf-8")
            )
            cached_wav = session_dir / "audio_cache" / f"{session_meta.media_hash}.wav"
            if cached_wav.is_file():
                audio_path = cached_wav
            elif Path(session_meta.media_path).is_file():
                audio_path = Path(session_meta.media_path)
        except (ValueError, KeyError, OSError):
            audio_path = None

    mapping_path = session_dir / "speakers_mapping.json"
    mapping = load_speakers_mapping(mapping_path)

    known_speakers = load_speaker_names(
        speakers_file=speakers_file,
        speakers_arg=speakers,
        context_dir=context_dir,
    )
    candidate_names: list[str] = list(
        dict.fromkeys(known_speakers + list(mapping.cluster_defaults.values()))
    )
    if context_dir.is_dir():
        candidate_names.extend(e.name for e in scan_context_directory(context_dir))
    candidate_names = list(dict.fromkeys(candidate_names))

    mapping = run_interactive_review(
        turns,
        candidate_names,
        existing_mapping=mapping,
        audio_path=audio_path,
        auto_play=auto_play,
    )
    save_speakers_mapping(mapping, mapping_path)

    aligned_turns = apply_speakers_mapping(turns, mapping)
    debounced_turns = debounce_consecutive_turns(aligned_turns)
    raw_md = render_markdown_transcript(debounced_turns)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(raw_md, encoding="utf-8")
    console.print(
        f"\n[bold green]✓ Review completed. Updated transcript saved to:[/bold green] {output}\n"
    )


@app.command()
def split(
    session_dir: Annotated[
        Path, typer.Argument(help="Path to session cache directory (.a2ts)")
    ],
    cluster_id: Annotated[
        str, typer.Argument(help="Cluster ID to split, e.g. SPEAKER_00")
    ],
    at: Annotated[float, typer.Option(help="Split timestamp in seconds")],
    to: Annotated[str, typer.Option(help="New cluster ID or speaker name")],
    output: Annotated[Path | None, typer.Option(help="Output markdown path")] = None,
) -> None:
    """Split speaker cluster at timestamp and regenerate transcript."""
    turns_path = session_dir / "turns.json"
    if not turns_path.is_file():
        console.print(f"[bold red]Turns cache not found at {turns_path}[/bold red]")
        raise typer.Exit(code=1)

    turns_data = json.loads(turns_path.read_text(encoding="utf-8"))
    turns = [AlignedTurn.model_validate(t) for t in turns_data]

    updated_turns = split_cluster_at_time(turns, cluster_id, at, to)
    turns_path.write_text(
        json.dumps(
            [t.model_dump() for t in updated_turns], indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    mapping_path = session_dir / "speakers_mapping.json"
    mapping = load_speakers_mapping(mapping_path)
    mapped_turns = apply_speakers_mapping(updated_turns, mapping)
    debounced_turns = debounce_consecutive_turns(mapped_turns)
    raw_md = render_markdown_transcript(debounced_turns)

    out_path = output
    if out_path is None:
        session_path = session_dir / "session.json"
        if session_path.is_file():
            meta = SessionMetadata.model_validate_json(
                session_path.read_text(encoding="utf-8")
            )
            if meta.output_path:
                out_path = Path(meta.output_path)
    if out_path is None:
        out_path = Path("transcript.md")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(raw_md, encoding="utf-8")
    console.print(
        f"\n[bold green]✓ Cluster '{cluster_id}' split at {at}s to '{to}'. Transcript saved to:[/bold green] {out_path}\n"
    )


if __name__ == "__main__":
    app()
