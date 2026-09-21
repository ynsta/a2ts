"""Acoustic speaker diarization using SpeechBrain ECAPA-TDNN and Agglomerative Clustering."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch
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

from a2ts.models import RawSegment, SpeakerTurn

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


def diarize_segments(
    audio_path: Path,
    segments: list[RawSegment],
    num_speakers: int | None = None,
    distance_threshold: float = 0.55,
    device: str = "cuda",
    cache_path: Path | None = None,
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
        turns = [
            SpeakerTurn(id=i, start=s.start, end=s.end, cluster_id="SPEAKER_00")
            for i, s in enumerate(segments)
        ]
        return turns

    emb_matrix = np.array(embeddings)
    # Normalize embeddings to unit norm for cosine distance
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    # Perform Agglomerative Clustering
    if num_speakers is not None:
        actual_clusters = min(num_speakers, len(embeddings))
        clusterer = AgglomerativeClustering(
            n_clusters=actual_clusters,
            metric="cosine",
            linkage="average",
        )
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
