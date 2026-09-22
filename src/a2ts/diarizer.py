"""Acoustic speaker diarization using SpeechBrain ECAPA-TDNN and Agglomerative Clustering."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch
from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-untyped]

from a2ts.models import (
    AlignedTurn,
    RawSegment,
    SpeakersMapping,
    SpeakerTurn,
    VoiceProfile,
    VoiceProfilesDatabase,
)

logger = logging.getLogger(__name__)
console = Console()
_CLASSIFIER: Any = None


def get_embedding_model(device: str = "cuda") -> Any:
    """Load and cache the SpeechBrain ECAPA-TDNN encoder."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        from speechbrain.inference.speaker import (  # type: ignore[import-untyped]
            EncoderClassifier,
        )

        with console.status(
            "[bold cyan]Loading SpeechBrain ECAPA-TDNN speaker embedding model...[/bold cyan]",
            spinner="dots",
        ):
            device_str = (
                "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            )
            _CLASSIFIER = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device_str},
            )
        console.print("[green]✓ Speaker embedding model loaded.[/green]")
    return _CLASSIFIER


def extract_embeddings(
    audio_path: Path,
    segments: list[RawSegment],
    device: str = "cuda",
    cache_prefix: Path | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Extract and cache speaker embeddings for speech segments."""
    if cache_prefix is not None:
        emb_file = Path(f"{cache_prefix}_embeddings.npy")
        idx_file = Path(f"{cache_prefix}_indices.json")
        if emb_file.is_file() and idx_file.is_file():
            console.print(
                f"[bold cyan]Loading cached embeddings from {cache_prefix}...[/bold cyan]"
            )
            try:
                emb_matrix = np.load(emb_file)
                cached_indices = [
                    int(x) for x in json.loads(idx_file.read_text(encoding="utf-8"))
                ]
                return emb_matrix, cached_indices
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.debug("Failed to read embedding cache %s: %s", cache_prefix, exc)

    if not segments:
        return np.empty((0, 0), dtype=np.float32), []

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    classifier = get_embedding_model(device=device)

    embeddings: list[np.ndarray] = []
    valid_indices: list[int] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Diarizing speech segments (ECAPA-TDNN)[/bold blue]"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("embedding", total=len(segments))

        for idx, seg in enumerate(segments):
            start_idx = max(0, int(seg.start * sr))
            end_idx = min(len(audio), int(seg.end * sr))

            # Need at least ~0.15s (2400 samples at 16kHz) for meaningful embedding
            if end_idx - start_idx >= 2400:
                clip_tensor = torch.tensor(
                    audio[start_idx:end_idx], dtype=torch.float32
                ).unsqueeze(0)
                if device == "cuda" and torch.cuda.is_available():
                    clip_tensor = clip_tensor.cuda()

                with torch.no_grad():
                    emb = classifier.encode_batch(clip_tensor).squeeze().cpu().numpy()
                embeddings.append(emb)
                valid_indices.append(idx)
            progress.update(task, advance=1)

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), []

    emb_matrix = np.array(embeddings, dtype=np.float32)
    # Normalize embeddings to unit norm for cosine distance
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    if cache_prefix is not None:
        emb_file = Path(f"{cache_prefix}_embeddings.npy")
        idx_file = Path(f"{cache_prefix}_indices.json")
        emb_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_file, emb_matrix)
        idx_file.write_text(
            json.dumps(valid_indices, indent=2),
            encoding="utf-8",
        )

    return emb_matrix, valid_indices


def diarize_segments(
    audio_path: Path,
    segments: list[RawSegment],
    num_speakers: int | None = None,
    distance_threshold: float = 0.60,
    device: str = "cuda",
    cache_path: Path | None = None,
    embeddings_cache_prefix: Path | None = None,
) -> list[SpeakerTurn]:
    """Cluster raw speech segments into acoustic speaker turns."""
    if not segments:
        return []

    # Check cache
    if cache_path and cache_path.is_file():
        console.print(
            f"[bold cyan]Loading cached diarization from {cache_path}...[/bold cyan]"
        )
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            return [SpeakerTurn.model_validate(item) for item in cached_data]
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.debug("Failed to read diarization cache %s: %s", cache_path, exc)

    if len(segments) == 1:
        single_turn = [
            SpeakerTurn(
                id=0,
                start=segments[0].start,
                end=segments[0].end,
                cluster_id="SPEAKER_00",
            )
        ]
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps([t.model_dump() for t in single_turn], indent=2),
                encoding="utf-8",
            )
        return single_turn

    emb_matrix, valid_indices = extract_embeddings(
        audio_path=audio_path,
        segments=segments,
        device=device,
        cache_prefix=embeddings_cache_prefix,
    )

    if len(valid_indices) == 0:
        turns = [
            SpeakerTurn(id=i, start=s.start, end=s.end, cluster_id="SPEAKER_00")
            for i, s in enumerate(segments)
        ]
        return turns

    # Perform Agglomerative Clustering
    if len(valid_indices) <= 1:
        labels = np.zeros(len(valid_indices), dtype=int)
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
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clusterer.fit_predict(emb_matrix)

    # Reconstruct cluster labels for all segments
    full_labels = [0] * len(segments)
    for valid_idx, label in zip(valid_indices, labels, strict=False):
        full_labels[valid_idx] = int(label)

    # For any skipped very short segments, propagate adjacent label
    for i in range(len(segments)):
        if i not in valid_indices:
            if i > 0:
                full_labels[i] = full_labels[i - 1]
            elif valid_indices:
                full_labels[i] = full_labels[valid_indices[0]]

    # Map labels to consistent cluster IDs sorted by appearance
    label_map: dict[int, str] = {}
    speaker_turns: list[SpeakerTurn] = []

    for i, seg in enumerate(segments):
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

    num_detected = len(label_map)
    console.print(
        f"[green]✓ Diarization completed: detected {num_detected} distinct speakers.[/green]\n"
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                [t.model_dump() for t in speaker_turns],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return speaker_turns


def compute_voice_profiles(
    embeddings: np.ndarray,
    valid_indices: list[int],
    turns: list[SpeakerTurn | AlignedTurn],
    speakers_mapping: SpeakersMapping | dict[str, str],
    existing_db: VoiceProfilesDatabase | None = None,
) -> VoiceProfilesDatabase:
    """Compute normalized centroid voice profiles from segment embeddings and labeled turns.

    Ignores generic unmapped speaker IDs starting with 'SPEAKER_'.
    If existing_db is provided, combines existing speaker centroids using sample-count weighted averaging.
    """
    profiles: dict[str, VoiceProfile] = {}
    if existing_db is not None:
        for spk, prof in existing_db.speakers.items():
            profiles[spk] = prof.model_copy()

    if len(embeddings) == 0 or len(valid_indices) == 0:
        return VoiceProfilesDatabase(
            version=existing_db.version if existing_db else 1,
            speakers=profiles,
        )

    turn_by_id: dict[int, SpeakerTurn | AlignedTurn] = {}
    for idx, turn in enumerate(turns):
        tid = getattr(turn, "turn_id", getattr(turn, "id", idx))
        turn_by_id[tid] = turn

    speaker_embs: dict[str, list[np.ndarray]] = {}

    for i, valid_idx in enumerate(valid_indices):
        if i >= len(embeddings):
            break
        emb = embeddings[i]
        matched_turn = turn_by_id.get(valid_idx)
        if matched_turn is None and 0 <= valid_idx < len(turns):
            matched_turn = turns[valid_idx]
        if matched_turn is None:
            continue

        turn_id = (
            matched_turn.turn_id
            if isinstance(matched_turn, AlignedTurn)
            else matched_turn.id
        )
        cluster_id = matched_turn.cluster_id
        resolved_spk: str | None = None

        if isinstance(speakers_mapping, SpeakersMapping):
            if turn_id in speakers_mapping.turn_overrides:
                resolved_spk = speakers_mapping.turn_overrides[turn_id]
            elif cluster_id in speakers_mapping.cluster_defaults:
                resolved_spk = speakers_mapping.cluster_defaults[cluster_id]
            elif (
                isinstance(matched_turn, SpeakerTurn) and matched_turn.resolved_speaker
            ):
                resolved_spk = matched_turn.resolved_speaker
            elif isinstance(matched_turn, AlignedTurn) and matched_turn.speaker:
                resolved_spk = matched_turn.speaker
            else:
                resolved_spk = cluster_id
        elif isinstance(speakers_mapping, dict):
            turn_key = str(turn_id)
            if turn_key in speakers_mapping:
                resolved_spk = speakers_mapping[turn_key]
            elif cluster_id in speakers_mapping:
                resolved_spk = speakers_mapping[cluster_id]
            elif (
                isinstance(matched_turn, SpeakerTurn) and matched_turn.resolved_speaker
            ):
                resolved_spk = matched_turn.resolved_speaker
            elif isinstance(matched_turn, AlignedTurn) and matched_turn.speaker:
                resolved_spk = matched_turn.speaker
            else:
                resolved_spk = cluster_id

        if not resolved_spk or resolved_spk.startswith("SPEAKER_"):
            continue

        speaker_embs.setdefault(resolved_spk, []).append(emb)

    for speaker, embs in speaker_embs.items():
        new_arr = np.array(embs, dtype=np.float32)
        n_new = len(embs)
        mu_new = np.mean(new_arr, axis=0)

        if speaker in profiles:
            old_prof = profiles[speaker]
            n_old = old_prof.sample_count
            mu_old = np.array(old_prof.centroid, dtype=np.float32)
            mu_comb = (n_old * mu_old + n_new * mu_new) / (n_old + n_new)
            norm = float(np.linalg.norm(mu_comb))
            if norm > 0:
                mu_comb = mu_comb / norm
            profiles[speaker] = VoiceProfile(
                speaker_name=speaker,
                centroid=[float(x) for x in mu_comb],
                sample_count=n_old + n_new,
            )
        else:
            norm = float(np.linalg.norm(mu_new))
            if norm > 0:
                mu_new = mu_new / norm
            profiles[speaker] = VoiceProfile(
                speaker_name=speaker,
                centroid=[float(x) for x in mu_new],
                sample_count=n_new,
            )

    return VoiceProfilesDatabase(
        version=existing_db.version if existing_db else 1,
        speakers=profiles,
    )


def save_voice_profiles(db: VoiceProfilesDatabase, path: Path) -> None:
    """Save voice profiles database to JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(db.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_voice_profiles(path: Path) -> VoiceProfilesDatabase | None:
    """Load voice profiles database from JSON, returning None if missing or invalid."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
        return VoiceProfilesDatabase.model_validate_json(content)
    except (json.JSONDecodeError, OSError, ValueError, ValidationError) as exc:
        logger.debug("Failed to load voice profiles from %s: %s", path, exc)
        return None


def match_embeddings_to_profiles(
    embeddings: np.ndarray,
    db: VoiceProfilesDatabase,
    similarity_threshold: float = 0.60,
) -> dict[int, tuple[str, float]]:
    """Match embeddings against enrolled voice profiles using cosine similarity.

    Returns mapping from embedding index to (speaker_name, similarity_score)
    for all embeddings whose best match score >= similarity_threshold.
    """
    if len(embeddings) == 0 or not db.speakers:
        return {}

    speaker_names = list(db.speakers.keys())
    centroids = np.array(
        [db.speakers[name].centroid for name in speaker_names],
        dtype=np.float32,
    )

    # Unit-normalize embeddings and centroids
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norms[emb_norms == 0] = 1.0
    norm_embeddings = embeddings / emb_norms

    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    c_norms[c_norms == 0] = 1.0
    norm_centroids = centroids / c_norms

    sim_matrix = np.dot(norm_embeddings, norm_centroids.T)

    matches: dict[int, tuple[str, float]] = {}
    best_indices = np.argmax(sim_matrix, axis=1)
    best_scores = np.max(sim_matrix, axis=1)

    for i, (k_star, score) in enumerate(zip(best_indices, best_scores, strict=False)):
        score_val = float(score)
        if score_val >= similarity_threshold:
            matches[i] = (speaker_names[k_star], score_val)

    return matches
