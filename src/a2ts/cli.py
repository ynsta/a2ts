"""CLI application and pipeline orchestration for a2ts."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-untyped]

from a2ts.cache import (
    atomic_write_text,
    compute_prompt_hash,
    compute_transcript_provenance_hash,
    load_transcript_cache,
    save_transcript_cache,
)
from a2ts.consolidator import debounce_consecutive_turns, render_markdown_transcript
from a2ts.craig import (
    build_craig_prompt,
    compute_track_provenance,
    discover_tracks,
    find_speakers_file,
    load_track_cache,
    merge_craig_tracks_to_turns,
    parse_info_file,
    parse_speakers_file,
    transcribe_craig_track,
)
from a2ts.diarizer import (
    classify_clusters_to_profiles,
    compute_voice_profiles,
    diarize_segments,
    extract_embeddings_for_turns,
    load_segment_embeddings,
    load_turn_embeddings,
    load_voice_profiles,
    propagate_speaker_labels,
    save_voice_profiles,
)
from a2ts.media import compute_file_hash, extract_audio_to_wav, probe_media
from a2ts.models import (
    AlignedTurn,
    ClusterSplit,
    DiarizationCacheProvenance,
    RawSegment,
    SessionMetadata,
    SpeakerTurn,
    TranscriptCacheProvenance,
    VoiceProfilesDatabase,
)
from a2ts.refiner import refine_transcript_markdown
from a2ts.speaker_review import (
    apply_speakers_mapping,
    apply_splits_to_turns,
    load_speakers_mapping,
    run_interactive_review,
    save_speakers_mapping,
)
from a2ts.timeline import (
    align_words_to_speaker_turns,
    assign_time_slices,
)
from a2ts.transcriber import WhisperEngine, get_engine
from a2ts.vocab import (
    build_biasing_prompt,
    load_speaker_names,
    load_wordlist_file,
    scan_context_directory,
)

app = typer.Typer(help="a2ts: Contextualized Audio/Video Transcriber")
console = Console()


def resolve_session_dir(path: Path) -> Path:
    """Resolve session directory from either a session folder or a cache directory.

    1. If `path / 'turns.json'` exists or `path / 'session.json'` exists (direct session directory
       or legacy root cache), return `path`.
    2. If `path.parent / 'sessions' / path.name` has turns.json/session.json (e.g. `.a2ts/<hash>`), return it.
    3. If `Path('.a2ts/sessions') / path.name` has turns.json/session.json (e.g. raw `<hash>`), return it.
    4. Else if `(path / 'sessions').is_dir()`:
       - If exactly 1 session directory exists, return it.
       - If multiple session directories exist, raise typer.Exit(code=1) with available sessions.
       - If no sessions exist, raise typer.Exit(code=1).
    5. Return `path` fallback.
    """
    if (path / "turns.json").is_file() or (path / "session.json").is_file():
        return path

    candidate = path.parent / "sessions" / path.name
    if (candidate / "turns.json").is_file() or (candidate / "session.json").is_file():
        return candidate

    default_candidate = Path(".a2ts/sessions") / path.name
    if (default_candidate / "turns.json").is_file() or (
        default_candidate / "session.json"
    ).is_file():
        return default_candidate

    sessions_dir = path / "sessions"
    if sessions_dir.is_dir():
        sessions = sorted([d for d in sessions_dir.iterdir() if d.is_dir()])
        if len(sessions) == 1:
            return sessions[0]
        if len(sessions) > 1:
            session_hashes = [s.name for s in sessions]
            console.print(
                f"[bold red]Multiple sessions found in {sessions_dir}. Please specify a session directory: {', '.join(session_hashes)}[/bold red]"
            )
            raise typer.Exit(code=1)
        console.print(f"[bold red]No sessions found in {sessions_dir}[/bold red]")
        raise typer.Exit(code=1)
    return path


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
    diarizer_engine: Annotated[
        str,
        typer.Option(
            "--diarizer-engine",
            help="Acoustic diarization engine: 'auto' (Nemotron-3 on CUDA, ECAPA on CPU), 'nemotron', or 'ecapa'",
        ),
    ] = "auto",
    cluster_threshold: Annotated[
        float,
        typer.Option(
            help="Diarization cosine distance threshold (higher = merges more)"
        ),
    ] = 0.60,
    num_speakers: Annotated[
        int | None,
        typer.Option(help="Target speaker count for clustering"),
    ] = None,
    voice_profiles: Annotated[
        Path,
        typer.Option(help="Path to voice profiles database JSON"),
    ] = Path("contexte/voice_profiles.json"),
    profile_threshold: Annotated[
        float,
        typer.Option(help="Cosine similarity threshold for matching voice profiles"),
    ] = 0.60,
    closed_set: Annotated[
        bool,
        typer.Option(
            "--closed-set/--no-closed-set",
            help="In closed-set mode, map all clusters to nearest enrolled voice profile and propagate labels",
        ),
    ] = False,
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
    audio_padding: Annotated[
        float,
        typer.Option(help="Context audio padding in seconds before/after sample quote"),
    ] = 2.0,
    review_cluster: Annotated[
        str | None,
        typer.Option(
            "--cluster",
            "-c",
            help="Filter specific cluster ID(s) to review (e.g. 'SPEAKER_00,SPEAKER_01')",
        ),
    ] = None,
    review_unassigned_only: Annotated[
        bool,
        typer.Option(
            "--unassigned-only",
            help="Only review clusters that have not yet been attributed to a known speaker",
        ),
    ] = False,
    review_min_turns: Annotated[
        int,
        typer.Option(
            "--min-turns",
            help="Minimum number of turns required to review a cluster",
        ),
    ] = 1,
    review_min_duration: Annotated[
        float,
        typer.Option(
            "--min-duration",
            help="Minimum speech duration in seconds required to review a cluster",
        ),
    ] = 0.0,
    review_time_slice: Annotated[
        int | None,
        typer.Option(
            "--time-slice",
            help="Only review clusters within specific time slice ID",
        ),
    ] = None,
    refine: Annotated[
        bool, typer.Option(help="Run local LLM refiner pass via agy")
    ] = False,
    refine_model: Annotated[
        str, typer.Option(help="Model name for LLM refiner")
    ] = "gemini-3.8-flash-low",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Force re-transcription and re-diarization ignoring existing cache",
        ),
    ] = False,
    rpg_normalize: Annotated[
        bool,
        typer.Option(
            "--rpg-normalize/--no-rpg-normalize",
            help="Normalize French tabletop RPG dice expressions and terms",
        ),
    ] = True,
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
    session_dir = cache_dir / "sessions" / file_hash
    session_dir.mkdir(parents=True, exist_ok=True)
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

    resolved_model_name = model_name or (
        "large-v3" if engine == "whisper" else "mistralai/Voxtral-Mini-3B-2507"
    )
    prompt_hash = compute_prompt_hash(prompt)

    trans_prov = TranscriptCacheProvenance(
        media_hash=file_hash,
        engine=engine,
        model_name=resolved_model_name,
        compute_type=compute_type,
        prompt_hash=prompt_hash,
    )

    cached_segments = load_transcript_cache(transcript_cache, trans_prov, force=force)
    if cached_segments is not None:
        console.print(
            f"[bold cyan]Step 3: Loading cached raw transcription from {transcript_cache}...[/bold cyan]"
        )
        raw_segments = cached_segments
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
        save_transcript_cache(transcript_cache, trans_prov, raw_segments)

    # 4. Acoustic Diarization & Temporal slicing
    console.print("[bold]Step 4: Acoustic speaker diarization & alignment...[/bold]")
    diarization_dir = cache_dir / "diarization"
    diarization_dir.mkdir(parents=True, exist_ok=True)
    diar_cache = diarization_dir / f"{file_hash}.json"

    resolved_diarizer_engine: str
    if diarizer_engine == "auto":
        is_cuda_avail = False
        if device == "cuda":
            try:
                import torch

                is_cuda_avail = torch.cuda.is_available()
            except ImportError:
                is_cuda_avail = False
        resolved_diarizer_engine = "nemotron" if is_cuda_avail else "ecapa"
    else:
        resolved_diarizer_engine = diarizer_engine

    transcript_prov_hash: str | None = None
    if resolved_diarizer_engine == "ecapa" or diarizer_engine == "ecapa":
        transcript_prov_hash = compute_transcript_provenance_hash(trans_prov)

    diar_prov = DiarizationCacheProvenance(
        media_hash=file_hash,
        engine=diarizer_engine,
        resolved_engine=resolved_diarizer_engine,
        cluster_threshold=cluster_threshold,
        num_speakers=num_speakers,
        device=device,
        transcript_provenance_hash=transcript_prov_hash,
    )

    if diarize:
        speaker_turns = diarize_segments(
            audio_path=audio_path,
            segments=raw_segments,
            num_speakers=num_speakers,
            distance_threshold=cluster_threshold,
            device=device,
            engine=diarizer_engine,
            cache_path=diar_cache,
            embeddings_cache_prefix=diarization_dir / file_hash,
            provenance=diar_prov,
            force=force,
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

    # 5. Interactive speaker review & voice profile matching
    mapping_path = session_dir / "speakers_mapping.json"
    mapping = load_speakers_mapping(mapping_path)
    if mapping.splits:
        aligned_turns = apply_splits_to_turns(aligned_turns, mapping.splits)

    # Cache aligned turns for review and split subcommands
    turns_path = session_dir / "turns.json"
    atomic_write_text(
        turns_path,
        json.dumps(
            [t.model_dump() for t in aligned_turns], indent=2, ensure_ascii=False
        ),
    )

    loaded_db: VoiceProfilesDatabase | None = None
    if voice_profiles.is_file() and diarize:
        loaded_db = load_voice_profiles(voice_profiles)
        diar_prov_hash = diar_prov.compute_hash()
        turn_embs: tuple[np.ndarray, list[int]] | None = None
        if not force:
            turn_embs = load_turn_embeddings(
                cache_prefix=diarization_dir / file_hash,
                media_hash=file_hash,
                expected_diarization_provenance_hash=diar_prov_hash,
            )

        if (
            loaded_db
            and loaded_db.speakers
            and turn_embs is None
            and speaker_turns
            and audio_path.is_file()
        ):
            turn_embs = extract_embeddings_for_turns(
                audio_path=audio_path,
                turns=speaker_turns,
                device=device,
                cache_prefix=diarization_dir / file_hash,
                force=force,
                media_hash=file_hash,
                diarization_provenance_hash=diar_prov_hash,
            )
        if (
            loaded_db
            and loaded_db.speakers
            and turn_embs is not None
            and isinstance(turn_embs, tuple)
            and len(turn_embs) == 2
            and isinstance(turn_embs[0], np.ndarray)
        ):
            try:
                emb_matrix, valid_indices = turn_embs
                if len(emb_matrix) > 0 and len(valid_indices) > 0:
                    seg_to_cluster = {
                        turn.id: turn.cluster_id for turn in speaker_turns
                    }
                    cluster_embs: dict[str, list[np.ndarray]] = {}
                    for idx_emb, seg_idx in enumerate(valid_indices):
                        cid = seg_to_cluster.get(seg_idx)
                        if cid is not None:
                            cluster_embs.setdefault(cid, []).append(emb_matrix[idx_emb])

                    cluster_matches = classify_clusters_to_profiles(
                        cluster_embs,
                        loaded_db,
                        similarity_threshold=profile_threshold,
                        closed_set=closed_set,
                    )
                    for cid, (best_spk, score) in cluster_matches.items():
                        if cid not in mapping.cluster_defaults:
                            mapping.cluster_defaults[cid] = best_spk
                            mapping.label_sources[cid] = "profile_match"
                            console.print(
                                f"[cyan]Matched voice profile for {cid}: {best_spk} (similarity={score:.2f})[/cyan]"
                            )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                console.print(
                    f"[yellow]Warning: Voice profile matching skipped: {exc}[/yellow]"
                )

    if interactive:
        known_speakers = load_speaker_names(
            speakers_file=speakers_file,
            speakers_arg=speakers,
            context_dir=context_dir,
        )
        profile_speakers = list(loaded_db.speakers.keys()) if loaded_db else []
        candidate_names = list(
            dict.fromkeys(
                known_speakers
                + profile_speakers
                + list(mapping.cluster_defaults.values())
                + [e.name for e in entities]
            )
        )
        filter_clusters: list[str] | None = None
        if review_cluster:
            filter_clusters = [
                c.strip() for c in review_cluster.split(",") if c.strip()
            ]

        mapping = run_interactive_review(
            aligned_turns,
            candidate_names,
            existing_mapping=mapping,
            audio_path=audio_path,
            auto_play=auto_play,
            audio_padding=audio_padding,
            filter_clusters=filter_clusters,
            unassigned_only=review_unassigned_only,
            min_turns=review_min_turns,
            min_duration=review_min_duration,
            filter_slice=review_time_slice,
            mapping_path=mapping_path,
        )
        for cid, val in mapping.cluster_defaults.items():
            if cid not in mapping.label_sources and not val.startswith("SPEAKER_"):
                mapping.label_sources[cid] = "manual"

    if closed_set:
        propagate_speaker_labels(aligned_turns, mapping)

    save_speakers_mapping(mapping, mapping_path)
    aligned_turns = apply_speakers_mapping(aligned_turns, mapping)

    # Auto-enroll / update voice profiles
    if diarize and not closed_set:
        embeddings_npy = diarization_dir / f"{file_hash}_turn_embeddings.npy"
        indices_json = diarization_dir / f"{file_hash}_turn_indices.json"
        has_manual_speakers = any(
            not v.startswith("SPEAKER_")
            for cid, v in mapping.cluster_defaults.items()
            if mapping.label_sources.get(cid) == "manual"
        )
        if (
            (not embeddings_npy.is_file() or not indices_json.is_file() or force)
            and speaker_turns
            and has_manual_speakers
            and audio_path.is_file()
        ):
            extract_embeddings_for_turns(
                audio_path=audio_path,
                turns=speaker_turns,
                device=device,
                cache_prefix=diarization_dir / file_hash,
                force=force,
            )
            embeddings_npy = diarization_dir / f"{file_hash}_turn_embeddings.npy"
            indices_json = diarization_dir / f"{file_hash}_turn_indices.json"
        if has_manual_speakers:
            if embeddings_npy.is_file() and indices_json.is_file():
                try:
                    embeddings = np.load(embeddings_npy)
                    valid_indices = [
                        int(x)
                        for x in json.loads(indices_json.read_text(encoding="utf-8"))
                    ]
                except (OSError, ValueError, json.JSONDecodeError):
                    embeddings = np.empty((0, 0), dtype=np.float32)
                    valid_indices = []
            else:
                embeddings = np.empty((0, 0), dtype=np.float32)
                valid_indices = []

            turns_for_profiles: list[SpeakerTurn | AlignedTurn] = list(aligned_turns)
            db = compute_voice_profiles(
                embeddings,
                valid_indices,
                turns_for_profiles,
                mapping,
                existing_db=loaded_db,
                session_id=file_hash,
            )
            save_voice_profiles(db, voice_profiles)
            console.print(
                f"[green]✓ Voice profiles updated for {len(db.speakers)} enrolled speakers.[/green]"
            )

    # 6. Consolidation & Debouncing
    console.print("[bold]Step 5: Consolidating transcript...[/bold]")
    debounced_turns = debounce_consecutive_turns(aligned_turns)
    raw_md = render_markdown_transcript(debounced_turns, rpg_normalize=rpg_normalize)

    # 7. Optional LLM Refinement
    final_md = raw_md
    if refine:
        raw_output = output.with_name(f"{output.stem}.raw.md")
        atomic_write_text(raw_output, raw_md)
        console.print(
            f"[cyan]Refining transcript via agy CLI model '{refine_model}' (external LLM)...[/cyan]"
        )
        final_md = refine_transcript_markdown(raw_md, agy_model=refine_model)

    atomic_write_text(output, final_md)

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
    atomic_write_text(
        session_dir / "session.json",
        json.dumps(session_meta.model_dump(), indent=2, ensure_ascii=False),
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
    voice_profiles: Annotated[
        Path,
        typer.Option(help="Path to voice profiles database JSON"),
    ] = Path("contexte/voice_profiles.json"),
    profile_threshold: Annotated[
        float,
        typer.Option(help="Cosine similarity threshold for matching voice profiles"),
    ] = 0.60,
    closed_set: Annotated[
        bool,
        typer.Option(
            "--closed-set/--no-closed-set",
            help="In closed-set mode, map all clusters to nearest enrolled voice profile and propagate labels",
        ),
    ] = False,
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
    audio_padding: Annotated[
        float,
        typer.Option(help="Context audio padding in seconds before/after sample quote"),
    ] = 2.0,
    cluster: Annotated[
        str | None,
        typer.Option(
            "--cluster",
            "-c",
            help="Filter specific cluster ID(s) to review (e.g. 'SPEAKER_00,SPEAKER_01')",
        ),
    ] = None,
    unassigned_only: Annotated[
        bool,
        typer.Option(
            "--unassigned-only",
            help="Only review clusters that have not yet been attributed to a known speaker",
        ),
    ] = False,
    min_turns: Annotated[
        int,
        typer.Option(
            "--min-turns",
            help="Minimum number of turns required to review a cluster",
        ),
    ] = 1,
    min_duration: Annotated[
        float,
        typer.Option(
            "--min-duration",
            help="Minimum speech duration in seconds required to review a cluster",
        ),
    ] = 0.0,
    time_slice: Annotated[
        int | None,
        typer.Option(
            "--time-slice",
            help="Only review clusters within specific time slice ID",
        ),
    ] = None,
) -> None:
    """Re-run interactive speaker review on cached session turns."""
    session_dir = resolve_session_dir(session_dir)
    turns_path = session_dir / "turns.json"
    if not turns_path.is_file():
        console.print(f"[bold red]Turns cache not found at {turns_path}[/bold red]")
        raise typer.Exit(code=1)

    turns_data = json.loads(turns_path.read_text(encoding="utf-8"))
    turns = [AlignedTurn.model_validate(t) for t in turns_data]

    cache_root = (
        session_dir.parent.parent
        if session_dir.parent.name == "sessions"
        else session_dir
    )

    audio_path: Path | None = None
    session_path = session_dir / "session.json"
    if session_path.is_file():
        try:
            session_meta = SessionMetadata.model_validate_json(
                session_path.read_text(encoding="utf-8")
            )
            cached_wav = session_dir / "audio_cache" / f"{session_meta.media_hash}.wav"
            if not cached_wav.is_file():
                cached_wav = (
                    cache_root / "audio_cache" / f"{session_meta.media_hash}.wav"
                )
            if cached_wav.is_file():
                audio_path = cached_wav
            elif Path(session_meta.media_path).is_file():
                audio_path = Path(session_meta.media_path)
        except (ValueError, KeyError, OSError):
            audio_path = None

    mapping_path = session_dir / "speakers_mapping.json"
    mapping = load_speakers_mapping(mapping_path)

    media_hash: str | None = None
    if session_path.is_file():
        try:
            session_meta = SessionMetadata.model_validate_json(
                session_path.read_text(encoding="utf-8")
            )
            media_hash = session_meta.media_hash
        except (ValueError, KeyError, OSError):
            media_hash = None

    diarization_dir = session_dir / "diarization"
    if not diarization_dir.is_dir() and (cache_root / "diarization").is_dir():
        diarization_dir = cache_root / "diarization"

    turn_embs: tuple[np.ndarray, list[int]] | None = None
    if media_hash:
        turn_embs = load_turn_embeddings(
            cache_prefix=diarization_dir / media_hash,
            media_hash=media_hash,
        )
    if turn_embs is None:
        console.print(
            f"[yellow]Warning: Turn embeddings not found or invalid provenance for media hash '{media_hash}'. "
            "Skipping voice profile matching.[/yellow]"
        )

    loaded_vp: VoiceProfilesDatabase | None = None
    if voice_profiles.is_file():
        loaded_vp = load_voice_profiles(voice_profiles)
        if loaded_vp and loaded_vp.speakers and turn_embs is not None:
            try:
                embeddings, valid_indices = turn_embs
                turn_by_id = {t.turn_id: t for t in turns}
                cluster_embs: dict[str, list[np.ndarray]] = {}
                for idx_emb, tid in enumerate(valid_indices):
                    matched_turn = turn_by_id.get(tid)
                    if matched_turn is not None:
                        cid = matched_turn.cluster_id
                        cluster_embs.setdefault(cid, []).append(embeddings[idx_emb])
                    else:
                        console.print(
                            f"[dim]Debug: Turn ID {tid} not found, skipping embedding[/dim]"
                        )

                cluster_matches = classify_clusters_to_profiles(
                    cluster_embs,
                    loaded_vp,
                    similarity_threshold=profile_threshold,
                    closed_set=closed_set,
                )
                for cid, (best_spk, score) in cluster_matches.items():
                    if cid not in mapping.cluster_defaults:
                        mapping.cluster_defaults[cid] = best_spk
                        mapping.label_sources[cid] = "profile_match"
                        console.print(
                            f"[cyan]Matched voice profile for {cid}: {best_spk} (similarity={score:.2f})[/cyan]"
                        )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                console.print(
                    f"[yellow]Warning: Voice profile matching skipped in review: {exc}[/yellow]"
                )

    known_speakers = load_speaker_names(
        speakers_file=speakers_file,
        speakers_arg=speakers,
        context_dir=context_dir,
    )
    profile_speakers: list[str] = list(loaded_vp.speakers.keys()) if loaded_vp else []
    candidate_names: list[str] = list(
        dict.fromkeys(
            known_speakers + profile_speakers + list(mapping.cluster_defaults.values())
        )
    )
    if context_dir.is_dir():
        candidate_names.extend(e.name for e in scan_context_directory(context_dir))
    candidate_names = list(dict.fromkeys(candidate_names))

    filter_clusters: list[str] | None = None
    if cluster:
        filter_clusters = [c.strip() for c in cluster.split(",") if c.strip()]

    mapping = run_interactive_review(
        turns,
        candidate_names,
        existing_mapping=mapping,
        audio_path=audio_path,
        auto_play=auto_play,
        audio_padding=audio_padding,
        filter_clusters=filter_clusters,
        unassigned_only=unassigned_only,
        min_turns=min_turns,
        min_duration=min_duration,
        filter_slice=time_slice,
        mapping_path=mapping_path,
    )
    for cid, val in mapping.cluster_defaults.items():
        if cid not in mapping.label_sources and not val.startswith("SPEAKER_"):
            mapping.label_sources[cid] = "manual"

    if closed_set:
        propagate_speaker_labels(turns, mapping)

    save_speakers_mapping(mapping, mapping_path)

    aligned_turns = apply_speakers_mapping(turns, mapping)

    # Auto-enroll / update voice profiles if cached embeddings exist
    if not closed_set and turn_embs is not None:
        has_manual_speakers = any(
            not v.startswith("SPEAKER_")
            for cid, v in mapping.cluster_defaults.items()
            if mapping.label_sources.get(cid) == "manual"
        )
        if has_manual_speakers:
            try:
                embeddings, valid_indices = turn_embs
                loaded_db = (
                    load_voice_profiles(voice_profiles)
                    if voice_profiles.is_file()
                    else None
                )
                turns_for_profiles: list[SpeakerTurn | AlignedTurn] = list(
                    aligned_turns
                )
                db = compute_voice_profiles(
                    embeddings,
                    valid_indices,
                    turns_for_profiles,
                    mapping,
                    existing_db=loaded_db,
                    session_id=media_hash or session_path.stem,
                )
                save_voice_profiles(db, voice_profiles)
                console.print(
                    f"[green]✓ Voice profiles updated for {len(db.speakers)} enrolled speakers.[/green]"
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                console.print(
                    f"[yellow]Warning: Voice profiles update skipped in review: {exc}[/yellow]"
                )

    debounced_turns = debounce_consecutive_turns(aligned_turns)
    raw_md = render_markdown_transcript(debounced_turns)

    atomic_write_text(output, raw_md)
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
    session_dir = resolve_session_dir(session_dir)
    turns_path = session_dir / "turns.json"
    if not turns_path.is_file():
        console.print(f"[bold red]Turns cache not found at {turns_path}[/bold red]")
        raise typer.Exit(code=1)

    turns_data = json.loads(turns_path.read_text(encoding="utf-8"))
    turns = [AlignedTurn.model_validate(t) for t in turns_data]

    mapping_path = session_dir / "speakers_mapping.json"
    mapping = load_speakers_mapping(mapping_path)
    mapping.splits.append(ClusterSplit(cluster_id=cluster_id, at=at, new_cluster_id=to))
    save_speakers_mapping(mapping, mapping_path)

    updated_turns = apply_splits_to_turns(turns, mapping.splits)
    atomic_write_text(
        turns_path,
        json.dumps(
            [t.model_dump() for t in updated_turns], indent=2, ensure_ascii=False
        ),
    )

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

    atomic_write_text(out_path, raw_md)
    console.print(
        f"\n[bold green]✓ Cluster '{cluster_id}' split at {at}s to '{to}'. Transcript saved to:[/bold green] {out_path}\n"
    )


@app.command()
def recluster(
    session_dir: Annotated[
        Path, typer.Argument(help="Path to session cache directory (.a2ts)")
    ] = Path(".a2ts"),
    cluster_threshold: Annotated[
        float,
        typer.Option(
            help="Diarization cosine distance threshold (higher = merges more)"
        ),
    ] = 0.60,
    num_speakers: Annotated[
        int | None,
        typer.Option(help="Target speaker count for clustering"),
    ] = None,
    slice_minutes: Annotated[
        float | None,
        typer.Option(help="Time slice window size in minutes"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Output markdown transcript path"),
    ] = None,
    voice_profiles: Annotated[
        Path,
        typer.Option(help="Path to voice profiles database JSON"),
    ] = Path("contexte/voice_profiles.json"),
    profile_threshold: Annotated[
        float,
        typer.Option(help="Cosine similarity threshold for matching voice profiles"),
    ] = 0.60,
    closed_set: Annotated[
        bool,
        typer.Option(
            "--closed-set/--no-closed-set",
            help="In closed-set mode, map all clusters to nearest enrolled voice profile and propagate labels",
        ),
    ] = False,
) -> None:
    """Re-cluster cached speaker embeddings and regenerate transcript."""
    session_dir = resolve_session_dir(session_dir)
    session_path = session_dir / "session.json"
    if not session_path.is_file():
        console.print(
            f"[bold red]Session metadata not found at {session_path}[/bold red]"
        )
        raise typer.Exit(code=1)

    try:
        session_meta = SessionMetadata.model_validate_json(
            session_path.read_text(encoding="utf-8")
        )
    except (ValueError, KeyError, OSError) as exc:
        console.print(f"[bold red]Failed to parse session metadata: {exc}[/bold red]")
        raise typer.Exit(code=1)

    media_hash = session_meta.media_hash
    engine = session_meta.engine

    cache_root = (
        session_dir.parent.parent
        if session_dir.parent.name == "sessions"
        else session_dir
    )

    transcripts_dir = session_dir / "transcripts"
    if not transcripts_dir.is_dir() and (cache_root / "transcripts").is_dir():
        transcripts_dir = cache_root / "transcripts"
    transcripts_cache = transcripts_dir / f"{media_hash}_{engine}.json"
    if not transcripts_cache.is_file():
        candidates = list(transcripts_dir.glob(f"{media_hash}_*.json"))
        if candidates:
            transcripts_cache = candidates[0]
        else:
            console.print(
                f"[bold red]Transcripts cache not found at {transcripts_cache}[/bold red]"
            )
            raise typer.Exit(code=1)

    try:
        cache_data = json.loads(transcripts_cache.read_text(encoding="utf-8"))
        if isinstance(cache_data, dict) and "segments" in cache_data:
            raw_segments = [
                RawSegment.model_validate(seg) for seg in cache_data["segments"]
            ]
        else:
            raw_segments = [RawSegment.model_validate(seg) for seg in cache_data]
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[bold red]Failed to load raw segments: {exc}[/bold red]")
        raise typer.Exit(code=1)

    diarization_dir = session_dir / "diarization"
    if not diarization_dir.is_dir() and (cache_root / "diarization").is_dir():
        diarization_dir = cache_root / "diarization"

    expected_trans_hash: str | None = None
    if isinstance(cache_data, dict) and "provenance" in cache_data:
        try:
            trans_prov = TranscriptCacheProvenance.model_validate(
                cache_data["provenance"]
            )
            expected_trans_hash = trans_prov.compute_hash()
        except (ValueError, KeyError, OSError):
            expected_trans_hash = None

    seg_embs = load_segment_embeddings(
        cache_prefix=diarization_dir / media_hash,
        media_hash=media_hash,
        expected_transcript_provenance_hash=expected_trans_hash,
    )
    if seg_embs is None:
        console.print(
            f"[bold red]Embeddings cache not found at {diarization_dir / f'{media_hash}_segment_embeddings.npy'}[/bold red]"
        )
        raise typer.Exit(code=1)

    emb_matrix, valid_indices = seg_embs

    t_start = time.perf_counter()

    if len(valid_indices) == 0:
        labels = np.zeros(0, dtype=int)
    elif len(valid_indices) == 1:
        labels = np.zeros(1, dtype=int)
    elif num_speakers is not None:
        actual_clusters = min(num_speakers, len(valid_indices))
        clusterer = AgglomerativeClustering(
            n_clusters=actual_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clusterer.fit_predict(emb_matrix)
    else:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cluster_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clusterer.fit_predict(emb_matrix)

    seg_idx_by_id = {seg.id: i for i, seg in enumerate(raw_segments)}
    full_labels = [0] * len(raw_segments)
    for seg_id, label in zip(valid_indices, labels, strict=False):
        if seg_id in seg_idx_by_id:
            full_labels[seg_idx_by_id[seg_id]] = int(label)
        elif seg_id < len(raw_segments):
            full_labels[seg_id] = int(label)

    valid_indices_set = set(valid_indices)
    for i in range(len(raw_segments)):
        if i not in valid_indices_set and (raw_segments[i].id not in valid_indices_set):
            if i > 0:
                full_labels[i] = full_labels[i - 1]
            elif valid_indices:
                first_id = valid_indices[0]
                full_labels[i] = full_labels[seg_idx_by_id.get(first_id, 0)]

    label_map: dict[int, str] = {}
    speaker_turns: list[SpeakerTurn] = []
    for i, seg in enumerate(raw_segments):
        lbl = full_labels[i]
        if lbl not in label_map:
            label_map[lbl] = f"SPEAKER_{len(label_map):02d}"
        cluster_id = label_map[lbl]
        speaker_turns.append(
            SpeakerTurn(
                id=i,
                start=seg.start,
                end=seg.end,
                cluster_id=cluster_id,
            )
        )

    num_detected = len(label_map) if label_map else (1 if raw_segments else 0)
    elapsed = time.perf_counter() - t_start
    elapsed_str = f"{elapsed:.2f}s" if elapsed >= 0.1 else "< 0.1s"

    resolved_slice = (
        slice_minutes
        if slice_minutes is not None
        else (session_meta.time_slice_minutes or 15.0)
    )
    sliced_turns = assign_time_slices(speaker_turns, slice_minutes=resolved_slice)
    aligned_turns = align_words_to_speaker_turns(raw_segments, sliced_turns)

    mapping_path = session_dir / "speakers_mapping.json"
    mapping = load_speakers_mapping(mapping_path)
    if mapping.splits:
        aligned_turns = apply_splits_to_turns(aligned_turns, mapping.splits)

    turns_path = session_dir / "turns.json"
    atomic_write_text(
        turns_path,
        json.dumps(
            [t.model_dump() for t in aligned_turns], indent=2, ensure_ascii=False
        ),
    )

    diar_cache = diarization_dir / f"{media_hash}.json"
    atomic_write_text(
        diar_cache,
        json.dumps([t.model_dump() for t in speaker_turns], indent=2),
    )

    if voice_profiles.is_file():
        loaded_db = load_voice_profiles(voice_profiles)
        if loaded_db and loaded_db.speakers:
            cluster_embs: dict[str, list[np.ndarray]] = {}
            turn_by_id = {t.id: t for t in speaker_turns}
            for idx_emb, seg_idx in enumerate(valid_indices):
                matched_turn = turn_by_id.get(seg_idx)
                if matched_turn is not None:
                    cid = matched_turn.cluster_id
                    cluster_embs.setdefault(cid, []).append(emb_matrix[idx_emb])

            cluster_matches = classify_clusters_to_profiles(
                cluster_embs,
                loaded_db,
                similarity_threshold=profile_threshold,
                closed_set=closed_set,
            )
            for cid, (best_spk, score) in cluster_matches.items():
                if cid not in mapping.cluster_defaults:
                    mapping.cluster_defaults[cid] = best_spk
                    mapping.label_sources[cid] = "profile_match"
                    console.print(
                        f"[cyan]Matched voice profile for {cid}: {best_spk} (similarity={score:.2f})[/cyan]"
                    )

    if closed_set:
        propagate_speaker_labels(aligned_turns, mapping)

    save_speakers_mapping(mapping, mapping_path)
    mapped_turns = apply_speakers_mapping(aligned_turns, mapping)

    debounced_turns = debounce_consecutive_turns(mapped_turns)
    raw_md = render_markdown_transcript(debounced_turns)

    out_path = output
    if out_path is None:
        if session_meta.output_path:
            out_path = Path(session_meta.output_path)
        else:
            out_path = Path("transcript.md")

    atomic_write_text(out_path, raw_md)

    console.print(
        f"[bold green]✓ Re-clustered into {num_detected} speakers in {elapsed_str}. Saved to {out_path}.[/bold green]"
    )


@app.command("craig")
def craig(
    recording_dir: Annotated[
        Path, typer.Argument(help="Path to folder containing Craig .flac tracks")
    ],
    model_name: Annotated[str, typer.Option(help="Whisper model name")] = "large-v3",
    device: Annotated[
        str, typer.Option(help="Device to run inference on (auto/cuda/cpu)")
    ] = "auto",
    compute_type: Annotated[
        str, typer.Option(help="Computation type (float16/int8/etc.)")
    ] = "float16",
    context_dir: Annotated[
        Path | None, typer.Option(help="Directory containing Obsidian markdown notes")
    ] = Path("contexte"),
    speakers_file: Annotated[
        Path | None, typer.Option(help="Path to speakers roster markdown file")
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Output markdown transcript path")
    ] = None,
    debounce: Annotated[
        float, typer.Option(help="Debounce window in seconds for consecutive turns")
    ] = 2.0,
    force: Annotated[
        bool,
        typer.Option("--force", help="Force re-transcription ignoring existing cache"),
    ] = False,
    rpg_normalize: Annotated[
        bool,
        typer.Option(
            "--rpg-normalize/--no-rpg-normalize",
            help="Normalize French tabletop RPG dice expressions and terms",
        ),
    ] = True,
    refine: Annotated[
        bool,
        typer.Option(
            "--refine/--no-refine",
            help="Run local LLM refiner pass on transcript",
        ),
    ] = False,
    refine_model: Annotated[
        str, typer.Option(help="Model name for LLM refiner")
    ] = "gemini-3.8-flash-low",
    refine_effort: Annotated[
        str, typer.Option(help="Reasoning effort for LLM refiner (low/medium/high)")
    ] = "low",
) -> None:
    """Transcribe and merge multi-track Craig Discord recordings into a unified transcript."""
    console.print(
        f"\n[bold green]=== Starting Craig Pipeline for {recording_dir.name} ===[/bold green]\n"
    )

    # 1. Verify recording_dir.is_dir(). Discover tracks. If empty, report error and exit with code 1.
    if not recording_dir.is_dir():
        console.print(
            f"[bold red]Recording directory not found: {recording_dir}[/bold red]"
        )
        raise typer.Exit(code=1)

    try:
        tracks = discover_tracks(recording_dir)
    except FileNotFoundError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)

    if not tracks:
        console.print(
            f"[bold red]No Craig .flac tracks found in {recording_dir}[/bold red]"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[bold]Discovered {len(tracks)} audio tracks in {recording_dir}[/bold]"
    )

    # 2. Locate and parse speakers.md via find_speakers_file and parse_speakers_file
    spk_path = find_speakers_file(recording_dir, speakers_file)
    if speakers_file is not None and spk_path is None:
        console.print(f"[bold red]Speakers file not found: {speakers_file}[/bold red]")
        raise typer.Exit(code=1)

    speakers = parse_speakers_file(spk_path) if spk_path is not None else {}
    if speakers:
        console.print(f"[green]Loaded {len(speakers)} speakers from {spk_path}[/green]")
    else:
        console.print(
            "[yellow]No speakers roster file found; using track usernames.[/yellow]"
        )

    # 3. Locate and parse info.txt via parse_info_file if present
    info_path = recording_dir / "info.txt"
    if info_path.is_file():
        info_data = parse_info_file(info_path)
        console.print(
            f"[cyan]Loaded info.txt metadata ({len(info_data.get('tracks', []))} tracks registered)[/cyan]"
        )
    else:
        info_data = None

    # 4. Build initial prompt via build_craig_prompt(speakers, context_dir)
    prompt = build_craig_prompt(speakers, context_dir=context_dir)

    # 5. Compute prompt hash: compute_prompt_hash(prompt)
    prompt_hash = compute_prompt_hash(prompt)

    # 6. Cache dir: cache_dir = recording_dir / ".transcripts"
    cache_dir = recording_dir / ".transcripts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 7. Lazy Whisper engine loading (only if at least one track misses cache)
    whisper_model: Any = None

    def get_whisper_model() -> Any:
        nonlocal whisper_model
        if whisper_model is None:
            whisper_engine = get_engine(
                "whisper",
                model_name=model_name,
                device=device,
                compute_type=compute_type,
            )
            if isinstance(whisper_engine, WhisperEngine):
                whisper_model = whisper_engine._get_model()
            else:
                whisper_model = whisper_engine
        return whisper_model

    # 8. Transcribe each track with transcribe_craig_track using provenance from compute_track_provenance
    all_segments: list[tuple[str, RawSegment]] = []
    for track in tracks:
        prov = compute_track_provenance(
            track_path=track,
            model_name=model_name,
            compute_type=compute_type,
            prompt_hash=prompt_hash,
        )
        cache_path = cache_dir / f"{track.stem}.json"
        cached_segments = load_track_cache(cache_path, prov, force=force)
        if cached_segments is not None:
            segments = cached_segments
        else:
            model = get_whisper_model()
            segments = transcribe_craig_track(
                model=model,
                track_path=track,
                provenance=prov,
                cache_dir=cache_dir,
                initial_prompt=prompt if prompt else None,
                force=force,
            )
        for seg in segments:
            all_segments.append((track.stem, seg))

    console.print(
        f"[green]Transcribed {len(tracks)} tracks ({len(all_segments)} raw segments).[/green]"
    )

    # 9. Merge tracks into aligned_turns = merge_craig_tracks_to_turns(all_segments, speakers)
    aligned_turns = merge_craig_tracks_to_turns(all_segments, speakers)

    # 10. Debounce turns via debounced_turns = debounce_consecutive_turns(aligned_turns, threshold_seconds=debounce)
    debounced_turns = debounce_consecutive_turns(
        aligned_turns, threshold_seconds=debounce
    )

    # 11. Render markdown via render_markdown_transcript(debounced_turns, rpg_normalize=rpg_normalize)
    raw_md = render_markdown_transcript(debounced_turns, rpg_normalize=rpg_normalize)

    out_path = output if output is not None else recording_dir / "transcript.md"

    # 12. If refine: refine with refine_transcript_markdown
    final_md = raw_md
    if refine:
        raw_output = out_path.with_name(f"{out_path.stem}.raw.md")
        atomic_write_text(raw_output, raw_md)
        console.print(
            f"[cyan]Refining transcript via agy CLI model '{refine_model}' (external LLM)...[/cyan]"
        )
        final_md = refine_transcript_markdown(
            raw_md, agy_model=refine_model, effort=refine_effort
        )

    # 13. Write output to out_path atomically using atomic_write_text
    atomic_write_text(out_path, final_md)

    console.print(
        f"\n[bold green]✓ Craig pipeline completed. Transcript saved to:[/bold green] {out_path}\n"
    )


if __name__ == "__main__":
    app()
