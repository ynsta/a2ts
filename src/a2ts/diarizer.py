"""Acoustic speaker diarization using SpeechBrain ECAPA-TDNN and Agglomerative Clustering."""

import json
import logging
from collections.abc import Mapping, Sequence
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

from a2ts.cache import (
    atomic_save_numpy,
    atomic_write_text,
    load_diarization_cache,
    save_diarization_cache,
)
from a2ts.models import (
    AlignedTurn,
    DiarizationCacheProvenance,
    EmbeddingCacheProvenance,
    RawSegment,
    SpeakersMapping,
    SpeakerTurn,
    VoiceProfile,
    VoiceProfilesDatabase,
)
from a2ts.speaker_review import resolve_speaker_for_turn

logger = logging.getLogger(__name__)
console = Console()
_CLASSIFIER: Any = None
_NEMOTRON_MODEL: Any = None
_NEMOTRON_PROCESSOR: Any = None
_TORCH_LOAD_PATCHED = False


def ensure_torch_speechbrain_compat() -> None:
    """Ensure SpeechBrain checkpoint loading works on PyTorch >= 2.6 without weights_only errors."""
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return
    import inspect

    import torch

    try:
        import speechbrain.utils.checkpoints as sb_checkpoints

        def patched_sb_load(path: Any, device: str = "cpu") -> Any:
            state_dict = torch.load(path, map_location=device, weights_only=False)
            return sb_checkpoints.hook_on_loading_state_dict_checkpoint(state_dict)

        sb_checkpoints.torch_patched_state_dict_load = patched_sb_load
    except (ImportError, AttributeError):
        pass

    orig_torch_load = torch.load
    sig = inspect.signature(orig_torch_load)
    if "weights_only" in sig.parameters:

        def compat_load(*args: Any, **kwargs: Any) -> Any:
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return orig_torch_load(*args, **kwargs)

        torch.load = compat_load  # type: ignore[assignment]
    _TORCH_LOAD_PATCHED = True


def get_nemotron_model(device: str = "cuda") -> tuple[Any, Any]:
    """Load and cache the NVIDIA Nemotron-3 Diarization model and processor."""
    global _NEMOTRON_MODEL, _NEMOTRON_PROCESSOR
    if _NEMOTRON_MODEL is None or _NEMOTRON_PROCESSOR is None:
        import os
        import tempfile

        # Ensure triton cache is writable even in sandboxed environments
        os.environ.setdefault(
            "TRITON_CACHE_DIR", str(Path(tempfile.gettempdir()) / "triton_cache")
        )

        from transformers import (  # type: ignore[import-untyped]
            AutoModelForAudioFrameClassification,
            AutoProcessor,
        )

        model_id = "nvidia/Nemotron-3-Diarization"
        with console.status(
            f"[bold cyan]Loading {model_id} model...[/bold cyan]",
            spinner="dots",
        ):
            processor = AutoProcessor.from_pretrained(model_id)
            device_str = (
                "cuda"
                if device in ("cuda", "auto") and torch.cuda.is_available()
                else "cpu"
            )
            dtype = torch.float16 if device_str == "cuda" else torch.float32
            model = AutoModelForAudioFrameClassification.from_pretrained(
                model_id,
                dtype=dtype,
            ).to(device_str)
            _NEMOTRON_PROCESSOR = processor
            _NEMOTRON_MODEL = model
        console.print("[green]✓ Nemotron-3 Diarization model loaded.[/green]")
    return _NEMOTRON_MODEL, _NEMOTRON_PROCESSOR


def diarize_nemotron(
    audio_path: Path,
    device: str = "cuda",
    cache_path: Path | None = None,
    provenance: DiarizationCacheProvenance | None = None,
    force: bool = False,
) -> list[SpeakerTurn]:
    """Perform acoustic speaker diarization using NVIDIA Nemotron-3 Diarization."""
    if cache_path:
        if provenance is not None:
            cached_turns = load_diarization_cache(cache_path, provenance, force=force)
            if cached_turns is not None:
                console.print(
                    f"[bold cyan]Loading cached diarization from {cache_path}...[/bold cyan]"
                )
                return cached_turns
        elif not force and cache_path.is_file():
            console.print(
                f"[bold cyan]Loading cached diarization from {cache_path}...[/bold cyan]"
            )
            try:
                cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached_data, dict) and "turns" in cached_data:
                    return [
                        SpeakerTurn.model_validate(item)
                        for item in cached_data["turns"]
                    ]
                elif isinstance(cached_data, list):
                    return [SpeakerTurn.model_validate(item) for item in cached_data]
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                logger.debug("Failed to read diarization cache %s: %s", cache_path, exc)

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    model, processor = get_nemotron_model(device=device)
    target_sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    if sr != target_sr:
        import librosa  # type: ignore[import-untyped]

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(model.device)
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)
    elif "input_values" in inputs:
        inputs["input_values"] = inputs["input_values"].to(dtype=model.dtype)

    with torch.inference_mode():
        logits = model(**inputs).logits

    segments = processor.extract_speaker_dict(logits, inputs.get("attention_mask"))[0]

    speaker_turns: list[SpeakerTurn] = []
    for i, seg in enumerate(segments):
        spk_idx = int(seg["Speaker"])
        speaker_turns.append(
            SpeakerTurn(
                id=i,
                start=float(seg["Start"]),
                end=float(seg["End"]),
                cluster_id=f"SPEAKER_{spk_idx:02d}",
            )
        )

    num_detected = len({t.cluster_id for t in speaker_turns})
    console.print(
        f"[green]✓ Nemotron-3 Diarization completed: detected {num_detected} distinct speakers ({len(speaker_turns)} turns).[/green]\n"
    )

    if cache_path:
        prov = (
            provenance.model_copy(
                update={"resolved_engine": provenance.resolved_engine or "nemotron"}
            )
            if provenance is not None
            else DiarizationCacheProvenance(
                media_hash=audio_path.stem,
                engine="nemotron",
                resolved_engine="nemotron",
                cluster_threshold=0.6,
                device=device,
            )
        )
        save_diarization_cache(cache_path, prov, speaker_turns)

    return speaker_turns


def get_embedding_model(device: str = "cuda") -> Any:
    """Load and cache the SpeechBrain ECAPA-TDNN encoder."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        ensure_torch_speechbrain_compat()
        from speechbrain.inference.speaker import (  # type: ignore[import-untyped]
            EncoderClassifier,
        )

        with console.status(
            "[bold cyan]Loading SpeechBrain ECAPA-TDNN speaker embedding model...[/bold cyan]",
            spinner="dots",
        ):
            device_str = (
                "cuda"
                if device in ("cuda", "auto") and torch.cuda.is_available()
                else "cpu"
            )
            _CLASSIFIER = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device_str},
            )
        console.print("[green]✓ Speaker embedding model loaded.[/green]")
    return _CLASSIFIER


def compute_entity_fingerprint(
    entities: Sequence[RawSegment | SpeakerTurn | AlignedTurn],
) -> str:
    """Compute deterministic SHA-256 fingerprint from entity IDs, time intervals, and text."""
    import hashlib

    parts: list[str] = []
    for idx, e in enumerate(entities):
        eid = getattr(e, "turn_id", getattr(e, "id", idx))
        text = getattr(e, "text", "")
        parts.append(f"{eid}:{e.start:.3f}:{e.end:.3f}:{text}")
    return hashlib.sha256(";".join(parts).encode("utf-8")).hexdigest()


def save_segment_embeddings(
    cache_prefix: Path | str,
    emb_matrix: np.ndarray,
    valid_indices: list[int],
    media_hash: str,
    transcript_provenance_hash: str | None = None,
    entity_fingerprint: str | None = None,
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb",
    segments: Sequence[RawSegment] | None = None,
) -> None:
    """Save segment embeddings, indices, and provenance metadata atomically."""
    cache_prefix_path = Path(cache_prefix)
    if entity_fingerprint is None and segments is not None:
        entity_fingerprint = compute_entity_fingerprint(segments)

    emb_file = Path(f"{cache_prefix_path}_segment_embeddings.npy")
    idx_file = Path(f"{cache_prefix_path}_segment_indices.json")
    prov_file = Path(f"{cache_prefix_path}_segment_embeddings_provenance.json")

    atomic_save_numpy(emb_file, emb_matrix)
    atomic_write_text(idx_file, json.dumps(valid_indices, indent=2))
    prov = EmbeddingCacheProvenance(
        entity_kind="segment",
        media_hash=media_hash,
        embedding_model=embedding_model,
        count=len(valid_indices),
        transcript_provenance_hash=transcript_provenance_hash,
        entity_fingerprint=entity_fingerprint,
    )
    atomic_write_text(prov_file, prov.model_dump_json(indent=2))


def load_segment_embeddings(
    cache_prefix: Path | str,
    media_hash: str = "",
    expected_transcript_provenance_hash: str | None = None,
    expected_count: int | None = None,
    expected_entity_fingerprint: str | None = None,
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb",
    segments: Sequence[RawSegment] | None = None,
    expected_media_hash: str | None = None,
) -> tuple[np.ndarray, list[int]] | None:
    """Load cached segment embeddings and indices, strictly validating provenance."""
    if not expected_transcript_provenance_hash:
        logger.debug(
            "Segment embedding cache miss: expected_transcript_provenance_hash is None or empty"
        )
        return None

    resolved_media_hash = (
        expected_media_hash if expected_media_hash is not None else media_hash
    )
    cache_prefix_path = Path(cache_prefix)
    if expected_entity_fingerprint is None and segments is not None:
        expected_entity_fingerprint = compute_entity_fingerprint(segments)

    emb_file = Path(f"{cache_prefix_path}_segment_embeddings.npy")
    idx_file = Path(f"{cache_prefix_path}_segment_indices.json")
    prov_file = Path(f"{cache_prefix_path}_segment_embeddings_provenance.json")

    if not emb_file.is_file() or not idx_file.is_file() or not prov_file.is_file():
        return None

    try:
        prov = EmbeddingCacheProvenance.model_validate_json(
            prov_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        logger.debug(
            "Failed to validate segment embedding provenance %s: %s", prov_file, exc
        )
        return None

    if prov.entity_kind != "segment":
        logger.warning(
            "Segment embedding cache provenance mismatch: expected 'segment', got '%s'",
            prov.entity_kind,
        )
        return None
    if prov.media_hash != resolved_media_hash:
        logger.debug(
            "Segment embedding media_hash mismatch: expected '%s', got '%s'",
            resolved_media_hash,
            prov.media_hash,
        )
        return None
    if prov.embedding_model != embedding_model:
        logger.debug(
            "Segment embedding model mismatch: expected '%s', got '%s'",
            embedding_model,
            prov.embedding_model,
        )
        return None
    if prov.transcript_provenance_hash != expected_transcript_provenance_hash:
        logger.debug(
            "Segment embedding transcript_provenance_hash mismatch: expected '%s', got '%s'",
            expected_transcript_provenance_hash,
            prov.transcript_provenance_hash,
        )
        return None
    if (
        expected_entity_fingerprint is not None
        and prov.entity_fingerprint != expected_entity_fingerprint
    ):
        logger.debug(
            "Segment embedding entity_fingerprint mismatch: expected '%s', got '%s'",
            expected_entity_fingerprint,
            prov.entity_fingerprint,
        )
        return None
    if expected_count is not None and prov.count != expected_count:
        logger.debug(
            "Segment embedding count mismatch: expected %d, got %d",
            expected_count,
            prov.count,
        )
        return None

    try:
        emb_matrix = np.load(emb_file)
        valid_indices = [
            int(x) for x in json.loads(idx_file.read_text(encoding="utf-8"))
        ]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read embedding cache %s: %s", emb_file, exc)
        return None

    if len(emb_matrix) != prov.count or len(valid_indices) != prov.count:
        logger.warning(
            "Segment embedding data size mismatch: provenance count=%d, matrix=%d, indices=%d",
            prov.count,
            len(emb_matrix),
            len(valid_indices),
        )
        return None

    return emb_matrix, valid_indices


def save_turn_embeddings(
    cache_prefix: Path | str,
    emb_matrix: np.ndarray,
    valid_indices: list[int],
    media_hash: str,
    diarization_provenance_hash: str | None = None,
    entity_fingerprint: str | None = None,
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb",
    turns: Sequence[SpeakerTurn | AlignedTurn] | None = None,
) -> None:
    """Save turn embeddings, indices, and provenance metadata atomically."""
    cache_prefix_path = Path(cache_prefix)
    if entity_fingerprint is None and turns is not None:
        entity_fingerprint = compute_entity_fingerprint(turns)

    emb_file = Path(f"{cache_prefix_path}_turn_embeddings.npy")
    idx_file = Path(f"{cache_prefix_path}_turn_indices.json")
    prov_file = Path(f"{cache_prefix_path}_turn_embeddings_provenance.json")

    atomic_save_numpy(emb_file, emb_matrix)
    atomic_write_text(idx_file, json.dumps(valid_indices, indent=2))
    prov = EmbeddingCacheProvenance(
        entity_kind="turn",
        media_hash=media_hash,
        embedding_model=embedding_model,
        count=len(valid_indices),
        diarization_provenance_hash=diarization_provenance_hash,
        entity_fingerprint=entity_fingerprint,
    )
    atomic_write_text(prov_file, prov.model_dump_json(indent=2))


def load_turn_embeddings(
    cache_prefix: Path | str,
    media_hash: str = "",
    expected_diarization_provenance_hash: str | None = None,
    expected_count: int | None = None,
    expected_entity_fingerprint: str | None = None,
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb",
    turns: Sequence[SpeakerTurn | AlignedTurn] | None = None,
    expected_media_hash: str | None = None,
) -> tuple[np.ndarray, list[int]] | None:
    """Load cached turn embeddings and indices, strictly validating provenance."""
    if not expected_diarization_provenance_hash:
        logger.debug(
            "Turn embedding cache miss: expected_diarization_provenance_hash is None or empty"
        )
        return None

    resolved_media_hash = (
        expected_media_hash if expected_media_hash is not None else media_hash
    )
    cache_prefix_path = Path(cache_prefix)
    if expected_entity_fingerprint is None and turns is not None:
        expected_entity_fingerprint = compute_entity_fingerprint(turns)

    emb_file = Path(f"{cache_prefix_path}_turn_embeddings.npy")
    idx_file = Path(f"{cache_prefix_path}_turn_indices.json")
    prov_file = Path(f"{cache_prefix_path}_turn_embeddings_provenance.json")

    if not emb_file.is_file() or not idx_file.is_file() or not prov_file.is_file():
        return None

    try:
        prov = EmbeddingCacheProvenance.model_validate_json(
            prov_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        logger.debug(
            "Failed to validate turn embedding provenance %s: %s", prov_file, exc
        )
        return None

    if prov.entity_kind != "turn":
        logger.warning(
            "Turn embedding cache provenance mismatch: expected 'turn', got '%s'",
            prov.entity_kind,
        )
        return None
    if prov.media_hash != resolved_media_hash:
        logger.debug(
            "Turn embedding media_hash mismatch: expected '%s', got '%s'",
            resolved_media_hash,
            prov.media_hash,
        )
        return None
    if prov.embedding_model != embedding_model:
        logger.debug(
            "Turn embedding model mismatch: expected '%s', got '%s'",
            embedding_model,
            prov.embedding_model,
        )
        return None
    if prov.diarization_provenance_hash != expected_diarization_provenance_hash:
        logger.debug(
            "Turn embedding diarization_provenance_hash mismatch: expected '%s', got '%s'",
            expected_diarization_provenance_hash,
            prov.diarization_provenance_hash,
        )
        return None
    if (
        expected_entity_fingerprint is not None
        and prov.entity_fingerprint != expected_entity_fingerprint
    ):
        logger.debug(
            "Turn embedding entity_fingerprint mismatch: expected '%s', got '%s'",
            expected_entity_fingerprint,
            prov.entity_fingerprint,
        )
        return None
    if expected_count is not None and prov.count != expected_count:
        logger.debug(
            "Turn embedding count mismatch: expected %d, got %d",
            expected_count,
            prov.count,
        )
        return None

    try:
        emb_matrix = np.load(emb_file)
        valid_indices = [
            int(x) for x in json.loads(idx_file.read_text(encoding="utf-8"))
        ]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read turn embedding cache %s: %s", emb_file, exc)
        return None

    if len(emb_matrix) != prov.count or len(valid_indices) != prov.count:
        logger.warning(
            "Turn embedding data size mismatch: provenance count=%d, matrix=%d, indices=%d",
            prov.count,
            len(emb_matrix),
            len(valid_indices),
        )
        return None

    return emb_matrix, valid_indices


def extract_embeddings_for_segments(
    audio_path: Path,
    segments: list[RawSegment],
    device: str = "cuda",
    cache_prefix: Path | None = None,
    force: bool = False,
    media_hash: str = "",
    transcript_provenance_hash: str | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Extract and cache speaker embeddings for speech segments."""
    resolved_media_hash = media_hash or (cache_prefix.name if cache_prefix else "")
    resolved_trans_prov_hash = transcript_provenance_hash or (
        compute_entity_fingerprint(segments) if segments else None
    )
    if cache_prefix is not None and not force:
        cached = load_segment_embeddings(
            cache_prefix=cache_prefix,
            media_hash=resolved_media_hash,
            expected_transcript_provenance_hash=resolved_trans_prov_hash,
            segments=segments,
        )
        if cached is not None:
            console.print(
                f"[bold cyan]Loading cached embeddings from {cache_prefix}_segment_embeddings.npy...[/bold cyan]"
            )
            return cached

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
                if device in ("cuda", "auto") and torch.cuda.is_available():
                    clip_tensor = clip_tensor.cuda()

                with torch.no_grad():
                    emb = classifier.encode_batch(clip_tensor).squeeze().cpu().numpy()
                embeddings.append(emb)
                seg_id = getattr(seg, "id", idx)
                valid_indices.append(seg_id)
            progress.update(task, advance=1)

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), []

    emb_matrix = np.array(embeddings, dtype=np.float32)
    # Normalize embeddings to unit norm for cosine distance
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    if cache_prefix is not None:
        save_segment_embeddings(
            cache_prefix=cache_prefix,
            emb_matrix=emb_matrix,
            valid_indices=valid_indices,
            media_hash=resolved_media_hash,
            transcript_provenance_hash=resolved_trans_prov_hash,
            segments=segments,
        )

    return emb_matrix, valid_indices


extract_embeddings = extract_embeddings_for_segments


def extract_embeddings_for_turns(
    audio_path: Path,
    turns: Sequence[SpeakerTurn | AlignedTurn],
    device: str = "cuda",
    cache_prefix: Path | None = None,
    force: bool = False,
    media_hash: str = "",
    diarization_provenance_hash: str | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Extract and cache speaker embeddings for speaker turns using ECAPA-TDNN."""
    resolved_media_hash = media_hash or (cache_prefix.name if cache_prefix else "")
    resolved_diar_prov_hash = diarization_provenance_hash or (
        compute_entity_fingerprint(turns) if turns else None
    )
    if cache_prefix is not None and not force:
        cached = load_turn_embeddings(
            cache_prefix=cache_prefix,
            media_hash=resolved_media_hash,
            expected_diarization_provenance_hash=resolved_diar_prov_hash,
            turns=turns,
        )
        if cached is not None:
            console.print(
                f"[bold cyan]Loading cached turn embeddings from {cache_prefix}_turn_embeddings.npy...[/bold cyan]"
            )
            return cached

    if not turns:
        return np.empty((0, 0), dtype=np.float32), []

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    classifier = get_embedding_model(device=device)

    embeddings: list[np.ndarray] = []
    valid_indices: list[int] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Embedding speaker turns (ECAPA-TDNN)[/bold blue]"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("turn_embedding", total=len(turns))

        for idx, turn in enumerate(turns):
            start_idx = max(0, int(turn.start * sr))
            end_idx = min(len(audio), int(turn.end * sr))

            # Need at least ~0.15s (2400 samples at 16kHz) for meaningful embedding
            if end_idx - start_idx >= 2400:
                clip_tensor = torch.tensor(
                    audio[start_idx:end_idx], dtype=torch.float32
                ).unsqueeze(0)
                if device in ("cuda", "auto") and torch.cuda.is_available():
                    clip_tensor = clip_tensor.cuda()

                with torch.no_grad():
                    emb = classifier.encode_batch(clip_tensor).squeeze().cpu().numpy()
                embeddings.append(emb)
                turn_id = getattr(turn, "turn_id", getattr(turn, "id", idx))
                valid_indices.append(turn_id)
            progress.update(task, advance=1)

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), []

    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    if cache_prefix is not None:
        save_turn_embeddings(
            cache_prefix=cache_prefix,
            emb_matrix=emb_matrix,
            valid_indices=valid_indices,
            media_hash=resolved_media_hash,
            diarization_provenance_hash=resolved_diar_prov_hash,
            turns=turns,
        )

    return emb_matrix, valid_indices


def diarize_segments(
    audio_path: Path,
    segments: list[RawSegment],
    num_speakers: int | None = None,
    distance_threshold: float = 0.60,
    device: str = "cuda",
    engine: str = "auto",
    cache_path: Path | None = None,
    embeddings_cache_prefix: Path | None = None,
    provenance: DiarizationCacheProvenance | None = None,
    force: bool = False,
) -> list[SpeakerTurn]:
    """Cluster raw speech segments into acoustic speaker turns."""
    if not segments:
        return []

    # Check cache
    if cache_path:
        if provenance is not None:
            cached_turns = load_diarization_cache(cache_path, provenance, force=force)
            if cached_turns is not None:
                console.print(
                    f"[bold cyan]Loading cached diarization from {cache_path}...[/bold cyan]"
                )
                return cached_turns
        elif not force and cache_path.is_file():
            console.print(
                f"[bold cyan]Loading cached diarization from {cache_path}...[/bold cyan]"
            )
            try:
                cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached_data, dict) and "turns" in cached_data:
                    return [
                        SpeakerTurn.model_validate(item)
                        for item in cached_data["turns"]
                    ]
                elif isinstance(cached_data, list):
                    return [SpeakerTurn.model_validate(item) for item in cached_data]
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                logger.debug("Failed to read diarization cache %s: %s", cache_path, exc)

    if engine == "nemotron":
        return diarize_nemotron(
            audio_path=audio_path,
            device=device,
            cache_path=cache_path,
            provenance=provenance,
            force=force,
        )

    if (
        engine == "auto"
        and device in ("cuda", "auto")
        and torch.cuda.is_available()
    ):
        try:
            return diarize_nemotron(
                audio_path=audio_path,
                device=device,
                cache_path=cache_path,
                provenance=provenance,
                force=force,
            )
        except (RuntimeError, ValueError, OSError, ImportError) as exc:
            logger.warning(
                "Nemotron-3 diarization failed (%s), falling back to SpeechBrain ECAPA.",
                exc,
            )
            if provenance is not None:
                provenance = provenance.model_copy(update={"resolved_engine": "ecapa"})

    if len(segments) <= 1:
        single_turn = (
            [
                SpeakerTurn(
                    id=0,
                    start=segments[0].start,
                    end=segments[0].end,
                    cluster_id="SPEAKER_00",
                )
            ]
            if len(segments) == 1
            else []
        )
        if cache_path:
            prov = (
                provenance.model_copy(
                    update={"resolved_engine": provenance.resolved_engine or "ecapa"}
                )
                if provenance is not None
                else DiarizationCacheProvenance(
                    media_hash=audio_path.stem,
                    engine=engine,
                    resolved_engine="ecapa",
                    cluster_threshold=distance_threshold,
                    num_speakers=num_speakers,
                    device=device,
                )
            )
            save_diarization_cache(cache_path, prov, single_turn)
        return single_turn

    emb_matrix, valid_indices = extract_embeddings(
        audio_path=audio_path,
        segments=segments,
        device=device,
        cache_prefix=embeddings_cache_prefix,
        force=force,
        media_hash=provenance.media_hash if provenance else "",
        transcript_provenance_hash=provenance.transcript_provenance_hash
        if provenance
        else None,
    )

    if len(valid_indices) == 0:
        turns = [
            SpeakerTurn(id=i, start=s.start, end=s.end, cluster_id="SPEAKER_00")
            for i, s in enumerate(segments)
        ]
        if cache_path:
            prov = (
                provenance.model_copy(
                    update={"resolved_engine": provenance.resolved_engine or "ecapa"}
                )
                if provenance is not None
                else DiarizationCacheProvenance(
                    media_hash=audio_path.stem,
                    engine=engine,
                    resolved_engine="ecapa",
                    cluster_threshold=distance_threshold,
                    num_speakers=num_speakers,
                    device=device,
                )
            )
            save_diarization_cache(cache_path, prov, turns)
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
        if 0 <= valid_idx < len(full_labels):
            full_labels[valid_idx] = int(label)

    # For any skipped very short segments, propagate adjacent label
    valid_indices_set = set(valid_indices)
    for i in range(len(segments)):
        if i not in valid_indices_set:
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
        prov = (
            provenance.model_copy(
                update={"resolved_engine": provenance.resolved_engine or "ecapa"}
            )
            if provenance is not None
            else DiarizationCacheProvenance(
                media_hash=audio_path.stem,
                engine=engine,
                resolved_engine="ecapa",
                cluster_threshold=distance_threshold,
                num_speakers=num_speakers,
                device=device,
            )
        )
        save_diarization_cache(cache_path, prov, speaker_turns)

    return speaker_turns


def compute_voice_profiles(
    embeddings: np.ndarray,
    valid_indices: list[int],
    turns: Sequence[SpeakerTurn | AlignedTurn],
    speakers_mapping: SpeakersMapping | dict[str, str],
    existing_db: VoiceProfilesDatabase | None = None,
    session_id: str = "",
) -> VoiceProfilesDatabase:
    """Compute normalized centroid voice profiles from segment embeddings and labeled turns.

    Ignores generic unmapped speaker IDs starting with 'SPEAKER_'.
    If existing_db is provided, combines existing speaker centroids using sample-count weighted averaging.
    Idempotent: samples with already enrolled sample_ids are ignored.
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

    mapping: SpeakersMapping
    if isinstance(speakers_mapping, SpeakersMapping):
        mapping = speakers_mapping
    else:
        mapping = SpeakersMapping(
            cluster_defaults={
                str(k): v
                for k, v in speakers_mapping.items()
                if not (isinstance(k, int) or (isinstance(k, str) and k.isdigit()))
            },
            turn_overrides={
                int(k): v
                for k, v in speakers_mapping.items()
                if isinstance(k, int) or (isinstance(k, str) and k.isdigit())
            },
        )

    speaker_samples: dict[str, list[tuple[np.ndarray, str]]] = {}

    for i, valid_idx in enumerate(valid_indices):
        if i >= len(embeddings):
            break
        emb = embeddings[i]
        matched_turn = turn_by_id.get(valid_idx)
        if matched_turn is None:
            continue

        turn_id = (
            matched_turn.turn_id
            if isinstance(matched_turn, AlignedTurn)
            else matched_turn.id
        )
        cluster_id = matched_turn.cluster_id

        if mapping.label_sources and mapping.label_sources.get(cluster_id) != "manual":
            continue

        fallback = (
            matched_turn.resolved_speaker
            if isinstance(matched_turn, SpeakerTurn)
            else matched_turn.speaker
            if isinstance(matched_turn, AlignedTurn)
            else None
        )
        time_slice_id = getattr(matched_turn, "time_slice_id", 0)

        resolved_spk: str = resolve_speaker_for_turn(
            turn_id=turn_id,
            cluster_id=cluster_id,
            time_slice_id=time_slice_id,
            mapping=mapping,
            fallback_speaker=fallback,
        )

        if not resolved_spk or resolved_spk.startswith("SPEAKER_"):
            continue

        sample_id = (
            f"{session_id}:{turn_id}:{matched_turn.start:.2f}"
            if session_id
            else f"turn:{turn_id}:{matched_turn.start:.2f}"
        )
        speaker_samples.setdefault(resolved_spk, []).append((emb, sample_id))

    for speaker, samples in speaker_samples.items():
        if speaker in profiles:
            old_prof = profiles[speaker]
            existing_ids = set(old_prof.sample_ids)
            new_samples = [s for s in samples if s[1] not in existing_ids]
            if not new_samples:
                # All samples already enrolled; skip to preserve idempotency
                continue

            new_embs = [s[0] for s in new_samples]
            new_ids = [s[1] for s in new_samples]
            new_arr = np.array(new_embs, dtype=np.float32)
            n_new = len(new_embs)
            mu_new = np.mean(new_arr, axis=0)

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
                sample_ids=old_prof.sample_ids + new_ids,
            )
        else:
            new_embs = [s[0] for s in samples]
            new_ids = [s[1] for s in samples]
            new_arr = np.array(new_embs, dtype=np.float32)
            mu_new = np.mean(new_arr, axis=0)
            norm = float(np.linalg.norm(mu_new))
            if norm > 0:
                mu_new = mu_new / norm
            profiles[speaker] = VoiceProfile(
                speaker_name=speaker,
                centroid=[float(x) for x in mu_new],
                sample_count=len(new_embs),
                sample_ids=new_ids,
            )

    return VoiceProfilesDatabase(
        version=existing_db.version if existing_db else 1,
        speakers=profiles,
    )


# Backward-compatible / brief alias
compute_voice_profiles_from_turns = compute_voice_profiles


def save_voice_profiles(db: VoiceProfilesDatabase, path: Path) -> None:
    """Save voice profiles database to JSON atomically."""
    atomic_write_text(Path(path), db.model_dump_json(indent=2))


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

    if norm_embeddings.shape[1] != norm_centroids.shape[1]:
        return {}

    sim_matrix = np.dot(norm_embeddings, norm_centroids.T)

    matches: dict[int, tuple[str, float]] = {}
    best_indices = np.argmax(sim_matrix, axis=1)
    best_scores = np.max(sim_matrix, axis=1)

    for i, (k_star, score) in enumerate(zip(best_indices, best_scores, strict=False)):
        score_val = float(score)
        if score_val >= similarity_threshold:
            matches[i] = (speaker_names[k_star], score_val)

    return matches


def classify_clusters_to_profiles(
    cluster_embeddings: Mapping[str, Sequence[np.ndarray] | np.ndarray],
    db: VoiceProfilesDatabase,
    similarity_threshold: float = 0.60,
    closed_set: bool = False,
    allowed_speakers: Sequence[str] | None = None,
) -> dict[str, tuple[str, float]]:
    """Classify speech clusters to enrolled voice profiles using cluster mean embeddings.

    Returns mapping from cluster_id to (speaker_name, similarity_score).
    If closed_set is True, assigns each cluster to argmax similarity among enrolled profiles
    (ignoring similarity_threshold, provided at least one profile exists).
    If closed_set is False, only assigns if max similarity >= similarity_threshold.
    If allowed_speakers is provided, only matches against the specified speaker names.
    """
    if not cluster_embeddings or not db.speakers:
        return {}

    speaker_names = list(db.speakers.keys())
    if allowed_speakers is not None:
        allowed_set = set(allowed_speakers)
        speaker_names = [name for name in speaker_names if name in allowed_set]
    if not speaker_names:
        return {}

    centroids = np.array(
        [db.speakers[name].centroid for name in speaker_names],
        dtype=np.float32,
    )
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    c_norms[c_norms == 0] = 1.0
    norm_centroids = centroids / c_norms

    assignments: dict[str, tuple[str, float]] = {}

    for cid, embs in cluster_embeddings.items():
        if len(embs) == 0:
            continue
        emb_arr = np.asarray(embs, dtype=np.float32)
        if emb_arr.ndim == 1:
            emb_arr = emb_arr.reshape(1, -1)
        if emb_arr.shape[1] != norm_centroids.shape[1]:
            continue
        mean_emb = np.mean(emb_arr, axis=0)
        norm = float(np.linalg.norm(mean_emb))
        if norm > 0:
            mean_emb = mean_emb / norm

        sims = np.dot(norm_centroids, mean_emb)  # shape: (num_speakers,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if closed_set or best_score >= similarity_threshold:
            assignments[cid] = (speaker_names[best_idx], best_score)

    return assignments


def propagate_speaker_labels(
    turns: Sequence[SpeakerTurn | AlignedTurn],
    mapping: SpeakersMapping,
    unassigned_prefix: str = "SPEAKER_",
) -> None:
    """Propagate neighboring speaker labels to unassigned turns in place.

    For turns without embeddings or unmapped clusters (e.g. short audio blips),
    propagates the closest preceding (or succeeding) assigned speaker name in mapping.
    Modifies mapping.turn_overrides for those specific turns so that apply_speakers_mapping
    resolves them to real speaker names without leaving generic SPEAKER_XX tags.
    """
    if not turns:
        return

    resolved_names: list[str] = []
    for turn in turns:
        t_id = getattr(turn, "turn_id", getattr(turn, "id", 0))
        cid = turn.cluster_id
        name = (
            mapping.turn_overrides.get(t_id) or mapping.cluster_defaults.get(cid) or cid
        )
        resolved_names.append(name)

    # 1. Forward pass: propagate last known assigned speaker
    last_known: str | None = None
    for i, name in enumerate(resolved_names):
        if not name.startswith(unassigned_prefix):
            last_known = name
        elif last_known is not None:
            resolved_names[i] = last_known

    # 2. Backward pass: propagate next known assigned speaker (for leading unassigned turns)
    next_known: str | None = None
    for i in range(len(resolved_names) - 1, -1, -1):
        name = resolved_names[i]
        if not name.startswith(unassigned_prefix):
            next_known = name
        elif next_known is not None:
            resolved_names[i] = next_known

    # 3. Apply to mapping.turn_overrides for any turn that changed
    for i, turn in enumerate(turns):
        t_id = getattr(turn, "turn_id", getattr(turn, "id", 0))
        orig_cid = turn.cluster_id
        orig_name = (
            mapping.turn_overrides.get(t_id)
            or mapping.cluster_defaults.get(orig_cid)
            or orig_cid
        )
        propagated_name = resolved_names[i]
        if orig_name != propagated_name and not propagated_name.startswith(
            unassigned_prefix
        ):
            mapping.turn_overrides[t_id] = propagated_name
            if orig_cid not in mapping.label_sources:
                mapping.label_sources[orig_cid] = "propagated"
