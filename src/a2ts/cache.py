"""Versioned cache management and provenance validation for a2ts."""

import hashlib
import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from a2ts.models import (
    DiarizationCacheFile,
    DiarizationCacheProvenance,
    RawSegment,
    SpeakerTurn,
    TranscriptCacheFile,
    TranscriptCacheProvenance,
)

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, content: str) -> None:
    """Write text to target file atomically using a unique temporary sibling file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as tmp_f:
        tmp_path = Path(tmp_f.name)
        try:
            tmp_f.write(content)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    try:
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_save_numpy(path: Path, arr: np.ndarray) -> None:
    """Save numpy array to target file atomically using a unique temporary sibling file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp.npy",
        delete=False,
    ) as tmp_f:
        tmp_path = Path(tmp_f.name)

    try:
        np.save(tmp_path, arr)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def compute_prompt_hash(prompt: str | None) -> str:
    """Compute 16-character SHA-256 digest of Whisper biasing prompt, or 'no_prompt'."""
    if not prompt:
        return "no_prompt"
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def save_transcript_cache(
    cache_path: Path,
    provenance: TranscriptCacheProvenance,
    segments: list[RawSegment],
) -> None:
    """Serialize transcription provenance and segments atomically to JSON cache."""
    cache_file = TranscriptCacheFile(provenance=provenance, segments=segments)
    atomic_write_text(cache_path, cache_file.model_dump_json(indent=2))


def load_transcript_cache(
    cache_path: Path,
    expected_provenance: TranscriptCacheProvenance,
    force: bool = False,
) -> list[RawSegment] | None:
    """Load cached segments if provenance strictly matches expected settings, or None if miss/stale."""
    if force or not cache_path.is_file():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "provenance" not in data:
            return None
        cached = TranscriptCacheFile.model_validate(data)
        p = cached.provenance
        if (
            p.media_hash == expected_provenance.media_hash
            and p.engine == expected_provenance.engine
            and p.model_name == expected_provenance.model_name
            and p.compute_type == expected_provenance.compute_type
            and p.prompt_hash == expected_provenance.prompt_hash
        ):
            return cached.segments
        return None
    except (json.JSONDecodeError, OSError, ValueError, KeyError) as exc:
        logger.debug("Failed reading transcript cache %s: %s", cache_path, exc)
        return None


def compute_transcript_provenance_hash(provenance: TranscriptCacheProvenance) -> str:
    """Compute deterministic SHA-256 digest of transcript provenance settings."""
    return provenance.compute_hash()


def save_diarization_cache(
    cache_path: Path,
    provenance: DiarizationCacheProvenance,
    turns: list[SpeakerTurn],
) -> None:
    """Serialize diarization provenance and speaker turns atomically to JSON cache."""
    cache_file = DiarizationCacheFile(provenance=provenance, turns=turns)
    atomic_write_text(cache_path, cache_file.model_dump_json(indent=2))


def load_diarization_cache(
    cache_path: Path,
    expected_provenance: DiarizationCacheProvenance,
    force: bool = False,
) -> list[SpeakerTurn] | None:
    """Load cached diarization turns if provenance matches expected settings, or None if miss/stale."""
    if force or not cache_path.is_file():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "provenance" not in data:
            return None
        cached = DiarizationCacheFile.model_validate(data)
        p = cached.provenance
        if expected_provenance.engine == "auto":
            if expected_provenance.resolved_engine == "ecapa":
                engine_matches = p.resolved_engine == "ecapa" or p.engine == "ecapa"
            else:
                engine_matches = (
                    p.engine == "auto"
                    or p.resolved_engine in ("nemotron", "ecapa")
                    or p.engine in ("nemotron", "ecapa")
                )
        else:
            engine_matches = (
                p.engine == expected_provenance.engine
                or p.resolved_engine == expected_provenance.engine
            ) and (
                expected_provenance.resolved_engine is None
                or p.resolved_engine == expected_provenance.resolved_engine
            )

        clusters_match = (
            p.cluster_threshold == expected_provenance.cluster_threshold
            and p.num_speakers == expected_provenance.num_speakers
        )
        device_matches = (
            expected_provenance.device == "auto"
            or p.device == "auto"
            or p.device == expected_provenance.device
        )
        transcript_matches = (
            expected_provenance.transcript_provenance_hash is None
            or p.transcript_provenance_hash
            == expected_provenance.transcript_provenance_hash
        )

        if (
            p.media_hash == expected_provenance.media_hash
            and engine_matches
            and clusters_match
            and device_matches
            and transcript_matches
        ):
            return cached.turns
        return None
    except (json.JSONDecodeError, OSError, ValueError, KeyError) as exc:
        logger.debug("Failed reading diarization cache %s: %s", cache_path, exc)
        return None
