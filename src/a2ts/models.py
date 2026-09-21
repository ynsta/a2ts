"""Core data models for a2ts."""

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


class SpeakersMapping(BaseModel):
    """Mapping of cluster IDs and time slice overrides to speaker names."""

    cluster_defaults: dict[str, str] = Field(default_factory=dict)
    slice_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)
    turn_overrides: dict[int, str] = Field(default_factory=dict)


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
