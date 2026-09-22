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

from a2ts.consolidator import debounce_consecutive_turns, render_markdown_transcript
from a2ts.diarizer import (
    classify_clusters_to_profiles,
    compute_voice_profiles,
    diarize_segments,
    load_voice_profiles,
    propagate_speaker_labels,
    save_voice_profiles,
)
from a2ts.media import compute_file_hash, extract_audio_to_wav, probe_media
from a2ts.models import (
    AlignedTurn,
    RawSegment,
    SessionMetadata,
    SpeakerTurn,
    VoiceProfilesDatabase,
)
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
            distance_threshold=cluster_threshold,
            device=device,
            cache_path=diar_cache,
            embeddings_cache_prefix=diarization_dir / file_hash,
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

    # 5. Interactive speaker review & voice profile matching
    mapping_path = cache_dir / "speakers_mapping.json"
    mapping = load_speakers_mapping(mapping_path)

    loaded_db: VoiceProfilesDatabase | None = None
    if voice_profiles.is_file():
        loaded_db = load_voice_profiles(voice_profiles)
        embeddings_npy = diarization_dir / f"{file_hash}_embeddings.npy"
        indices_json = diarization_dir / f"{file_hash}_indices.json"
        if (
            loaded_db
            and loaded_db.speakers
            and embeddings_npy.is_file()
            and indices_json.is_file()
        ):
            try:
                emb_matrix = np.load(embeddings_npy)
                valid_indices = [
                    int(x) for x in json.loads(indices_json.read_text(encoding="utf-8"))
                ]
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
        )

    if closed_set:
        propagate_speaker_labels(aligned_turns, mapping)

    save_speakers_mapping(mapping, mapping_path)
    aligned_turns = apply_speakers_mapping(aligned_turns, mapping)

    # Auto-enroll / update voice profiles
    if diarize:
        embeddings_npy = diarization_dir / f"{file_hash}_embeddings.npy"
        indices_json = diarization_dir / f"{file_hash}_indices.json"
        if embeddings_npy.is_file() and indices_json.is_file():
            try:
                embeddings = np.load(embeddings_npy)
                valid_indices = [
                    int(x) for x in json.loads(indices_json.read_text(encoding="utf-8"))
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
        )
        save_voice_profiles(db, voice_profiles)
        console.print(
            f"[green]✓ Voice profiles updated for {len(db.speakers)} enrolled speakers.[/green]"
        )

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
    embeddings_npy: Path | None = None
    indices_json: Path | None = None
    if media_hash:
        e_path = diarization_dir / f"{media_hash}_embeddings.npy"
        i_path = diarization_dir / f"{media_hash}_indices.json"
        if e_path.is_file() and i_path.is_file():
            embeddings_npy = e_path
            indices_json = i_path
    if embeddings_npy is None and diarization_dir.is_dir():
        for e_file in diarization_dir.glob("*_embeddings.npy"):
            prefix = e_file.name.removesuffix("_embeddings.npy")
            i_file = diarization_dir / f"{prefix}_indices.json"
            if i_file.is_file():
                embeddings_npy = e_file
                indices_json = i_file
                break

    loaded_vp: VoiceProfilesDatabase | None = None
    if voice_profiles.is_file():
        loaded_vp = load_voice_profiles(voice_profiles)
        if (
            loaded_vp
            and loaded_vp.speakers
            and embeddings_npy
            and indices_json
            and embeddings_npy.is_file()
            and indices_json.is_file()
        ):
            try:
                embeddings = np.load(embeddings_npy)
                valid_indices = [
                    int(x) for x in json.loads(indices_json.read_text(encoding="utf-8"))
                ]
                cluster_embs: dict[str, list[np.ndarray]] = {}
                for idx_emb, seg_idx in enumerate(valid_indices):
                    if seg_idx < len(turns):
                        cid = turns[seg_idx].cluster_id
                        cluster_embs.setdefault(cid, []).append(embeddings[idx_emb])

                cluster_matches = classify_clusters_to_profiles(
                    cluster_embs,
                    loaded_vp,
                    similarity_threshold=profile_threshold,
                    closed_set=closed_set,
                )
                for cid, (best_spk, score) in cluster_matches.items():
                    if cid not in mapping.cluster_defaults:
                        mapping.cluster_defaults[cid] = best_spk
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
    )

    if closed_set:
        propagate_speaker_labels(turns, mapping)

    save_speakers_mapping(mapping, mapping_path)

    aligned_turns = apply_speakers_mapping(turns, mapping)

    # Auto-enroll / update voice profiles if cached embeddings exist
    if (
        embeddings_npy
        and indices_json
        and embeddings_npy.is_file()
        and indices_json.is_file()
    ):
        try:
            embeddings = np.load(embeddings_npy)
            valid_indices = [
                int(x) for x in json.loads(indices_json.read_text(encoding="utf-8"))
            ]
            loaded_db = (
                load_voice_profiles(voice_profiles)
                if voice_profiles.is_file()
                else None
            )
            turns_for_profiles: list[SpeakerTurn | AlignedTurn] = list(aligned_turns)
            db = compute_voice_profiles(
                embeddings,
                valid_indices,
                turns_for_profiles,
                mapping,
                existing_db=loaded_db,
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

    transcripts_cache = session_dir / "transcripts" / f"{media_hash}_{engine}.json"
    if not transcripts_cache.is_file():
        candidates = list((session_dir / "transcripts").glob(f"{media_hash}_*.json"))
        if candidates:
            transcripts_cache = candidates[0]
        else:
            console.print(
                f"[bold red]Transcripts cache not found at {transcripts_cache}[/bold red]"
            )
            raise typer.Exit(code=1)

    try:
        raw_segments = [
            RawSegment.model_validate(seg)
            for seg in json.loads(transcripts_cache.read_text(encoding="utf-8"))
        ]
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[bold red]Failed to load raw segments: {exc}[/bold red]")
        raise typer.Exit(code=1)

    diarization_dir = session_dir / "diarization"
    embeddings_npy = diarization_dir / f"{media_hash}_embeddings.npy"
    indices_json = diarization_dir / f"{media_hash}_indices.json"

    if not (embeddings_npy.is_file() and indices_json.is_file()):
        found = False
        if diarization_dir.is_dir():
            for e_file in diarization_dir.glob("*_embeddings.npy"):
                prefix = e_file.name.removesuffix("_embeddings.npy")
                i_file = diarization_dir / f"{prefix}_indices.json"
                if i_file.is_file():
                    embeddings_npy = e_file
                    indices_json = i_file
                    found = True
                    break
        if not found:
            console.print(
                f"[bold red]Embeddings cache not found at {embeddings_npy}[/bold red]"
            )
            raise typer.Exit(code=1)

    try:
        emb_matrix = np.load(embeddings_npy)
        valid_indices = [
            int(x) for x in json.loads(indices_json.read_text(encoding="utf-8"))
        ]
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[bold red]Failed to load embeddings: {exc}[/bold red]")
        raise typer.Exit(code=1)

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

    full_labels = [0] * len(raw_segments)
    for valid_idx, label in zip(valid_indices, labels, strict=False):
        full_labels[valid_idx] = int(label)

    for i in range(len(raw_segments)):
        if i not in valid_indices:
            if i > 0:
                full_labels[i] = full_labels[i - 1]
            elif valid_indices:
                full_labels[i] = full_labels[valid_indices[0]]

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

    turns_path = session_dir / "turns.json"
    turns_path.write_text(
        json.dumps(
            [t.model_dump() for t in aligned_turns], indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    diar_cache = diarization_dir / f"{media_hash}.json"
    diar_cache.write_text(
        json.dumps([t.model_dump() for t in speaker_turns], indent=2),
        encoding="utf-8",
    )

    mapping_path = session_dir / "speakers_mapping.json"
    mapping = load_speakers_mapping(mapping_path)

    if voice_profiles.is_file():
        loaded_db = load_voice_profiles(voice_profiles)
        if loaded_db and loaded_db.speakers:
            cluster_embs: dict[str, list[np.ndarray]] = {}
            for idx_emb, seg_idx in enumerate(valid_indices):
                if seg_idx < len(speaker_turns):
                    cid = speaker_turns[seg_idx].cluster_id
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(raw_md, encoding="utf-8")

    console.print(
        f"[bold green]✓ Re-clustered into {num_detected} speakers in {elapsed_str}. Saved to {out_path}.[/bold green]"
    )


if __name__ == "__main__":
    app()
