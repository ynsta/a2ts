"""Core data models for a2ts."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """Timestamp and confidence for a single transcribed word."""

    word: str
    start: float
    end: float
    probability: float = 1.0


class RawSegment(BaseModel):
    """A raw speech segment produced by a transcription engine."""

    id: int
    start: float
    end: float
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)


class SpeakerTurn(BaseModel):
    """An acoustic speech turn identified by diarization."""

    id: int
    start: float
    end: float
    cluster_id: str
    resolved_speaker: str | None = None
    time_slice_id: int = 0
    flagged: bool = False
    notes: str = ""


class AlignedTurn(BaseModel):
    """A consolidated speech turn with text and assigned speaker."""

    turn_id: int
    start: float
    end: float
    speaker: str
    cluster_id: str
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)
    time_slice_id: int = 0


class EntityRecord(BaseModel):
    """An entity candidate extracted from lore notes."""

    name: str
    kind: str  # "wikilink", "alias", "heading", "custom"
    source_file: str = ""


class ClusterSplit(BaseModel):
    """A record of splitting a speaker cluster at a given timestamp."""

    cluster_id: str
    at: float
    new_cluster_id: str


class SpeakersMapping(BaseModel):
    """Mapping of cluster IDs and time slice overrides to speaker names."""

    cluster_defaults: dict[str, str] = Field(default_factory=dict)
    slice_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)
    turn_overrides: dict[int, str] = Field(default_factory=dict)
    splits: list[ClusterSplit] = Field(default_factory=list)
    label_sources: dict[str, str] = Field(default_factory=dict)


class SessionMetadata(BaseModel):
    """Provenance and metadata for an a2ts transcription session."""

    media_path: str
    media_hash: str
    duration_seconds: float
    engine: str
    model_name: str
    prompt_hash: str
    time_slice_minutes: float
    created_at: str
    output_path: str = ""


class VoiceProfile(BaseModel):
    """Voice signature for an enrolled speaker."""

    speaker_name: str
    centroid: list[float]
    sample_count: int = 1
    sample_ids: list[str] = Field(default_factory=list)


class VoiceProfilesDatabase(BaseModel):
    """Database of enrolled speaker voice profiles."""

    version: int = 1
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    speakers: dict[str, VoiceProfile] = Field(default_factory=dict)


class TranscriptCacheProvenance(BaseModel):
    """Provenance envelope tracking all settings affecting raw transcription."""

    version: int = 1
    media_hash: str
    engine: str
    model_name: str
    compute_type: str
    prompt_hash: str = "no_prompt"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 digest of settings (excluding created_at)."""
        import hashlib

        payload = self.model_dump_json(exclude={"created_at"}).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class TranscriptCacheFile(BaseModel):
    """Versioned storage file for raw transcription segments."""

    provenance: TranscriptCacheProvenance
    segments: list[RawSegment]


class DiarizationCacheProvenance(BaseModel):
    """Provenance envelope tracking all settings affecting acoustic diarization."""

    version: int = 1
    media_hash: str
    engine: str
    resolved_engine: str | None = None
    cluster_threshold: float
    num_speakers: int | None = None
    device: str = "cuda"
    transcript_provenance_hash: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DiarizationCacheFile(BaseModel):
    """Versioned storage file for acoustic diarization speaker turns."""

    provenance: DiarizationCacheProvenance
    turns: list[SpeakerTurn]


class SpeakerInfo(BaseModel):
    """Speaker identity and role metadata parsed from speakers.md."""

    discord_username: str
    character_name: str
    role: str | None = None
    nicknames: list[str] = Field(default_factory=list)
    is_dm: bool = False
    raw_description: str | None = None


class TrackCacheProvenance(BaseModel):
    """Provenance tracking all settings affecting a transcribed audio track."""

    schema_version: int = 1
    model_name: str
    compute_type: str
    prompt_hash: str
    vad_parameters: dict[str, Any] = Field(default_factory=dict)
    source_file_size: int
    source_file_mtime: float


class TrackCacheFile(BaseModel):
    """Versioned storage file for single-track transcription segments."""

    provenance: TrackCacheProvenance
    segments: list[RawSegment]


class EmbeddingCacheProvenance(BaseModel):
    """Provenance tracking settings and metadata for cached speaker embeddings."""

    version: int = 1
    entity_kind: Literal["segment", "turn"]
    media_hash: str
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    count: int
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
