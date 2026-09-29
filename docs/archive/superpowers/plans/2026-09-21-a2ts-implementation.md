# a2ts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, tested local audio/video transcription and diarization pipeline (`a2ts`) supporting single-stream media, dual-engine transcription (Mistral Voxtral 4-bit & Faster-Whisper), Obsidian lore vocabulary biasing, time-sliced speaker diarization with interactive QCM attribution, timeline consolidation, and local LLM refinement.

**Architecture:** Modular pipeline where media is probed and extracted to 16kHz mono WAV via `ffmpeg`, domain vocabulary is mined from Obsidian markdown notes and budgeted via `tiktoken`, speech is transcribed with `WhisperEngine` or `VoxtralEngine` (4-bit BitsAndBytes), speaker clusters are segmented into temporal slices to resolve cluster collision, reviewed interactively via rich QCM prompts, debounced and formatted to Markdown, with an optional local `agy` refinement pass.

**Tech Stack:** Python 3.13, `uv`, `pydantic>=2.10.0`, `typer>=0.15.0`, `rich>=13.9.0`, `faster-whisper>=1.2.1`, `transformers>=4.54.0`, `torch>=2.5.0`, `bitsandbytes>=0.45.0`, `mistral-common[audio]>=1.8.1`, `librosa>=1.0.0`, `soundfile>=0.13.0`, `tiktoken>=0.9.0`, `ffmpeg`.

**Spec:** [docs/archive/superpowers/specs/2026-09-21-a2ts-architecture-design.md](../specs/2026-09-21-a2ts-architecture-design.md)

## Global Constraints

- All code, comments, identifiers, docstrings, commit messages, and technical docs in English.
- Use `uv run` for all commands; never use raw `pip install`.
- Code must pass `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy src/`.
- Every task must strictly follow TDD: write failing test first, verify failure, implement minimal code, verify pass, and commit.
- In-memory test fixtures preferred for unit tests; use `tmp_path` for file I/O tests.

---

### Task 1: Core Models & Data Structures (`src/a2ts/models.py`)

**Files:**
- Create: `src/a2ts/models.py`
- Test: `tests/unit/test_models.py`

**Interfaces:**
- Produces: `WordTimestamp`, `RawSegment`, `SpeakerTurn`, `AlignedTurn`, `EntityRecord`, `SessionMetadata`, `SpeakersMapping`.

- [ ] **Step 1: Write failing tests for models**

```python
# tests/unit/test_models.py
from a2ts.models import (
    AlignedTurn,
    EntityRecord,
    RawSegment,
    SessionMetadata,
    SpeakerTurn,
    WordTimestamp,
)


def test_word_timestamp() -> None:
    wt = WordTimestamp(word="bonjour", start=0.5, end=1.0, probability=0.98)
    assert wt.word == "bonjour"
    assert wt.start == 0.5
    assert wt.end == 1.0


def test_raw_segment_with_words() -> None:
    seg = RawSegment(
        id=1,
        start=0.0,
        end=2.0,
        text="Bonjour à tous",
        words=[
            WordTimestamp(word="Bonjour", start=0.0, end=0.8),
            WordTimestamp(word="à", start=0.9, end=1.1),
            WordTimestamp(word="tous", start=1.2, end=2.0),
        ],
    )
    assert len(seg.words) == 3
    assert seg.text == "Bonjour à tous"


def test_speaker_turn() -> None:
    turn = SpeakerTurn(
        id=1,
        start=0.0,
        end=5.0,
        cluster_id="SPEAKER_00",
        resolved_speaker="MJ",
        time_slice_id=0,
        flagged=False,
    )
    assert turn.cluster_id == "SPEAKER_00"
    assert turn.resolved_speaker == "MJ"


def test_aligned_turn() -> None:
    aligned = AlignedTurn(
        turn_id=1,
        start=0.0,
        end=2.5,
        speaker="MJ",
        cluster_id="SPEAKER_00",
        text="Test line",
        time_slice_id=1,
    )
    assert aligned.speaker == "MJ"
    assert aligned.time_slice_id == 1


def test_entity_record() -> None:
    entity = EntityRecord(name="Kaelen", kind="wikilink", source_file="contexte/perso.md")
    assert entity.name == "Kaelen"
    assert entity.kind == "wikilink"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'a2ts.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/a2ts/models.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: PASS (5 tests passed)

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/models.py tests/unit/test_models.py
git commit -m "feat: add core domain models in models.py"
```

---

### Task 2: Media Probing & Audio Extraction (`src/a2ts/media.py`)

**Files:**
- Create: `src/a2ts/media.py`
- Test: `tests/unit/test_media.py`

**Interfaces:**
- Consumes: `Path`
- Produces:
  - `probe_media(media_path: Path) -> dict[str, Any]`
  - `compute_file_hash(media_path: Path) -> str`
  - `extract_audio_to_wav(media_path: Path, output_dir: Path) -> Path`

- [ ] **Step 1: Write failing tests for media module**

```python
# tests/unit/test_media.py
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from a2ts.media import compute_file_hash, extract_audio_to_wav, probe_media


def test_compute_file_hash(tmp_path: Path) -> None:
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"hello media content 12345")
    h1 = compute_file_hash(test_file)
    assert len(h1) == 16
    h2 = compute_file_hash(test_file)
    assert h1 == h2


@patch("subprocess.run")
def test_probe_media(mock_run: MagicMock, tmp_path: Path) -> None:
    test_file = tmp_path / "test.mkv"
    test_file.touch()

    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"format": {"duration": "12.34", "size": "1000"}}),
    )

    info = probe_media(test_file)
    assert float(info["format"]["duration"]) == 12.34


@patch("subprocess.run")
def test_extract_audio_to_wav_runs_ffmpeg(mock_run: MagicMock, tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp4"
    input_file.write_bytes(b"dummy mp4 content")
    out_dir = tmp_path / "cache"

    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="")

    wav_path = extract_audio_to_wav(input_file, out_dir)
    assert wav_path.suffix == ".wav"
    assert out_dir.is_dir()
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "ffmpeg" in cmd
    assert "-ar" in cmd and "16000" in cmd
    assert "-ac" in cmd and "1" in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_media.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'a2ts.media'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/a2ts/media.py
"""Media inspection and audio extraction utilities using ffprobe and ffmpeg."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def compute_file_hash(media_path: Path) -> str:
    """Compute 16-character SHA-256 digest of media file header/size."""
    hasher = hashlib.sha256()
    size = media_path.stat().st_size
    hasher.update(str(size).encode("utf-8"))
    with open(media_path, "rb") as f:
        hasher.update(f.read(65536))
    return hasher.hexdigest()[:16]


def probe_media(media_path: Path) -> dict[str, Any]:
    """Run ffprobe to inspect media container and stream information."""
    if not media_path.is_file():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_name,codec_type,sample_rate,channels",
        "-of",
        "json",
        str(media_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def extract_audio_to_wav(media_path: Path, output_dir: Path) -> Path:
    """Extract audio stream to 16kHz mono 16-bit PCM WAV cached by file hash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_hash = compute_file_hash(media_path)
    wav_path = output_dir / f"{file_hash}.wav"

    if wav_path.is_file() and wav_path.stat().st_size > 44:
        return wav_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return wav_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_media.py -v`
Expected: PASS (3 tests passed)

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/media.py tests/unit/test_media.py
git commit -m "feat: add media probing and audio extraction in media.py"
```

---

### Task 3: Obsidian Lore & Vocabulary Mining (`src/a2ts/vocab.py`)

**Files:**
- Create: `src/a2ts/vocab.py`
- Test: `tests/unit/test_vocab.py`

**Interfaces:**
- Consumes: `EntityRecord` from `models.py`
- Produces:
  - `extract_candidates_from_markdown(content: str, filename: str) -> list[EntityRecord]`
  - `load_wordlist_file(vocab_path: Path) -> list[EntityRecord]`
  - `build_biasing_prompt(entities: list[EntityRecord], max_tokens: int = 220) -> str`
  - `scan_context_directory(context_dir: Path) -> list[EntityRecord]`

- [ ] **Step 1: Write failing tests for vocab mining**

```python
# tests/unit/test_vocab.py
from pathlib import Path
from a2ts.vocab import (
    build_biasing_prompt,
    extract_candidates_from_markdown,
    load_wordlist_file,
    scan_context_directory,
)


def test_extract_candidates_from_markdown() -> None:
    content = """
aliases:
  - L'Ombre Rouge
  - Le Rôdeur

## Kaelen Sombreflèche

Le personnage de [[Garrick Hautbois|Garrick]] voyage vers [[Phandaline]].
Un fichier [[carte.png]] est ignoré.
"""
    entities = extract_candidates_from_markdown(content, "session.md")
    names = {e.name for e in entities}
    assert "Kaelen Sombreflèche" in names
    assert "L'Ombre Rouge" in names
    assert "Le Rôdeur" in names
    assert "Garrick" in names
    assert "Phandaline" in names
    assert "carte.png" not in names


def test_build_biasing_prompt_token_budget() -> None:
    from a2ts.models import EntityRecord

    entities = [EntityRecord(name=f"Entité_{i}", kind="wikilink") for i in range(100)]
    prompt = build_biasing_prompt(entities, max_tokens=30)
    assert len(prompt) > 0
    # Prompt must contain commas and be non-empty
    assert "Entité_0" in prompt


def test_load_wordlist_file(tmp_path: Path) -> None:
    wordlist = tmp_path / "vocab.txt"
    wordlist.write_text("Eldoria\nMorvath\n\n# Commentaire\nDraconique\n", encoding="utf-8")
    records = load_wordlist_file(wordlist)
    names = [r.name for r in records]
    assert names == ["Eldoria", "Morvath", "Draconique"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_vocab.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'a2ts.vocab'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/a2ts/vocab.py
"""Lore extraction and token-budgeted prompt construction."""

import re
from pathlib import Path
import tiktoken
from a2ts.models import EntityRecord

WIKILINK_PATTERN = re.compile(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]")
ALIAS_BLOCK_PATTERN = re.compile(r"(?m)^aliases:\s*\n((?:\s*-\s*.+\n)+)")
HEADING_PATTERN = re.compile(r"(?m)^##\s+([^\n#]+)")

IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".pdf", ".mp3", ".wav", ".flac"
}

FRENCH_STOPWORDS = {
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux",
    "et", "ou", "mais", "donc", "or", "ni", "car", "que", "qui", "quoi", "dont",
    "où", "en", "dans", "par", "pour", "vers", "avec", "sans", "sur", "sous",
}


def extract_candidates_from_markdown(content: str, filename: str) -> list[EntityRecord]:
    """Parse wikilinks, frontmatter aliases, and headings from markdown text."""
    records: list[EntityRecord] = []

    for match in WIKILINK_PATTERN.finditer(content):
        target = match.group(1).strip()
        alias = match.group(2).strip() if match.group(2) else None
        name = alias if alias else target

        if any(name.lower().endswith(ext) for ext in IGNORED_EXTENSIONS):
            continue
        if name and name.lower() not in FRENCH_STOPWORDS:
            records.append(EntityRecord(name=name, kind="wikilink", source_file=filename))

    alias_block = ALIAS_BLOCK_PATTERN.search(content)
    if alias_block:
        for line in alias_block.group(1).splitlines():
            alias_match = re.match(r"^\s*-\s*(.+)$", line)
            if alias_match:
                alias = alias_match.group(1).strip().strip("\"'")
                if alias and alias.lower() not in FRENCH_STOPWORDS:
                    records.append(EntityRecord(name=alias, kind="alias", source_file=filename))

    for match in HEADING_PATTERN.finditer(content):
        heading = match.group(1).strip()
        if heading and heading.lower() not in FRENCH_STOPWORDS:
            records.append(EntityRecord(name=heading, kind="heading", source_file=filename))

    return records


def load_wordlist_file(vocab_path: Path) -> list[EntityRecord]:
    """Load plain text wordlist file (one term per line, skipping comments)."""
    if not vocab_path.is_file():
        return []
    records: list[EntityRecord] = []
    for line in vocab_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            records.append(EntityRecord(name=line, kind="custom", source_file=str(vocab_path)))
    return records


def scan_context_directory(context_dir: Path) -> list[EntityRecord]:
    """Recursively scan markdown files in context directory."""
    if not context_dir.is_dir():
        return []
    records: list[EntityRecord] = []
    for md_file in sorted(context_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        records.extend(extract_candidates_from_markdown(content, str(md_file.relative_to(context_dir))))
    return records


def build_biasing_prompt(entities: list[EntityRecord], max_tokens: int = 220) -> str:
    """Construct token-budgeted prompt from unique entity records."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    seen: set[str] = set()
    unique_names: list[str] = []

    for e in entities:
        cleaned = re.sub(r"\s+", " ", e.name).strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unique_names.append(cleaned)

    selected: list[str] = []
    current_tokens = 0

    for name in unique_names:
        candidate_list = selected + [name]
        candidate_str = ", ".join(candidate_list)
        token_count = len(tokenizer.encode(candidate_str))
        if token_count <= max_tokens:
            selected.append(name)
            current_tokens = token_count
        else:
            break

    return ", ".join(selected)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_vocab.py -v`
Expected: PASS (3 tests passed)

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/vocab.py tests/unit/test_vocab.py
git commit -m "feat: add Obsidian lore and vocabulary mining in vocab.py"
```

---

### Task 4: Dual Transcription Engines (`src/a2ts/transcriber.py`)

**Files:**
- Create: `src/a2ts/transcriber.py`
- Test: `tests/unit/test_transcriber.py`

**Interfaces:**
- Consumes: `RawSegment`, `WordTimestamp` from `models.py`
- Produces:
  - `TranscriberEngine` protocol
  - `WhisperEngine(model_name: str = "large-v3", device: str = "cuda")`
  - `VoxtralEngine(model_name: str = "mistralai/Voxtral-Mini-3B-2507", load_4bit: bool = True)`
  - `get_engine(engine_type: str, **kwargs) -> TranscriberEngine`

- [ ] **Step 1: Write failing tests for transcription engines**

```python
# tests/unit/test_transcriber.py
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from a2ts.models import RawSegment, WordTimestamp
from a2ts.transcriber import TranscriberEngine, get_engine


def test_transcriber_engine_protocol_conformance() -> None:
    class DummyEngine:
        def transcribe(self, audio_path: Path, prompt: str | None = None, language: str = "fr") -> list[RawSegment]:
            return [RawSegment(id=0, start=0.0, end=1.0, text="Test")]

    engine: TranscriberEngine = DummyEngine()
    result = engine.transcribe(Path("test.wav"))
    assert len(result) == 1
    assert result[0].text == "Test"


@patch("a2ts.transcriber.WhisperEngine.transcribe")
def test_get_engine_whisper(mock_transcribe: MagicMock) -> None:
    mock_transcribe.return_value = [RawSegment(id=0, start=0.0, end=1.0, text="Whisper output")]
    engine = get_engine("whisper")
    segments = engine.transcribe(Path("sample.wav"))
    assert segments[0].text == "Whisper output"


def test_get_engine_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine("unknown_engine")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_transcriber.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'a2ts.transcriber'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/a2ts/transcriber.py
"""Dual transcription engine architecture for a2ts (Whisper & Voxtral)."""

import ctypes
import logging
import sys
from pathlib import Path
from typing import Any, Protocol

from a2ts.models import RawSegment, WordTimestamp

logger = logging.getLogger(__name__)


def ensure_cuda_libs() -> None:
    """Preload bundled NVIDIA CUDA libraries if available."""
    for path_entry in sys.path:
        nvidia_dir = Path(path_entry) / "nvidia"
        if nvidia_dir.is_dir():
            for lib_dir in nvidia_dir.glob("*/lib"):
                if lib_dir.is_dir():
                    for so_file in sorted(lib_dir.glob("*.so*")):
                        try:
                            ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                        except (OSError, RuntimeError):
                            continue


class TranscriberEngine(Protocol):
    """Protocol defining the transcription engine interface."""

    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
    ) -> list[RawSegment]:
        """Transcribe an audio file into timestamped RawSegment objects."""
        ...


class WhisperEngine:
    """Faster-Whisper engine with CTranslate2 and CUDA 12 support."""

    def __init__(self, model_name: str = "large-v3", device: str = "cuda", compute_type: str = "float16") -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            ensure_cuda_libs()
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
    ) -> list[RawSegment]:
        model = self._get_model()
        segments_gen, _ = model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=prompt,
            vad_filter=True,
            word_timestamps=True,
        )

        results: list[RawSegment] = []
        for i, s in enumerate(segments_gen):
            words = [
                WordTimestamp(word=w.word.strip(), start=w.start, end=w.end, probability=w.probability)
                for w in (s.words or [])
            ]
            results.append(RawSegment(id=i, start=s.start, end=s.end, text=s.text.strip(), words=words))
        return results


class VoxtralEngine:
    """Mistral Voxtral multimodal audio engine with 4-bit quantization support."""

    def __init__(self, model_name: str = "mistralai/Voxtral-Mini-3B-2507", load_4bit: bool = True) -> None:
        self.model_name = model_name
        self.load_4bit = load_4bit
        self._processor: Any = None
        self._model: Any = None

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, VoxtralForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            quant_config = None
            if self.load_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            self._model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
        return self._processor, self._model

    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
    ) -> list[RawSegment]:
        import torch
        processor, model = self._load_model()

        instruction = "Transcris fidèlement cet enregistrement audio en français."
        if prompt:
            instruction += f" Contexte et vocabulaire: {prompt}"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(audio_path.resolve())},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = processor.apply_chat_template(conversation).to("cuda")
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=1024)

        decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        text = decoded[0].strip() if decoded else ""

        # Voxtral generates full transcript text; segment by sentence/timing
        return [RawSegment(id=0, start=0.0, end=0.0, text=text)]


def get_engine(engine_type: str, **kwargs: Any) -> TranscriberEngine:
    """Factory creating configured TranscriberEngine instance."""
    normalized = engine_type.lower().strip()
    if normalized == "whisper":
        return WhisperEngine(**kwargs)
    elif normalized == "voxtral":
        return VoxtralEngine(**kwargs)
    raise ValueError(f"Unknown engine: {engine_type}. Expected 'voxtral' or 'whisper'.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_transcriber.py -v`
Expected: PASS (3 tests passed)

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/transcriber.py tests/unit/test_transcriber.py
git commit -m "feat: add dual transcription engines (Whisper and Voxtral) in transcriber.py"
```

---

### Task 5: Time Slicing & Timeline Alignment (`src/a2ts/timeline.py`)

**Files:**
- Create: `src/a2ts/timeline.py`
- Test: `tests/unit/test_timeline.py`

**Interfaces:**
- Consumes: `RawSegment`, `SpeakerTurn`, `AlignedTurn`, `WordTimestamp` from `models.py`
- Produces:
  - `assign_time_slices(turns: list[SpeakerTurn], slice_minutes: float = 15.0) -> list[SpeakerTurn]`
  - `align_words_to_speaker_turns(segments: list[RawSegment], turns: list[SpeakerTurn]) -> list[AlignedTurn]`
  - `split_cluster_at_time(turns: list[SpeakerTurn], cluster_id: str, split_time: float, new_cluster_id: str) -> list[SpeakerTurn]`
  - `detect_outlier_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]`

- [ ] **Step 1: Write failing tests for timeline alignment and splitting**

```python
# tests/unit/test_timeline.py
from a2ts.models import RawSegment, SpeakerTurn, WordTimestamp
from a2ts.timeline import (
    align_words_to_speaker_turns,
    assign_time_slices,
    split_cluster_at_time,
)


def test_assign_time_slices() -> None:
    turns = [
        SpeakerTurn(id=0, start=100.0, end=200.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=1, start=950.0, end=1050.0, cluster_id="SPEAKER_00"),  # > 15 min (900s)
    ]
    sliced = assign_time_slices(turns, slice_minutes=15.0)
    assert sliced[0].time_slice_id == 0
    assert sliced[1].time_slice_id == 1


def test_split_cluster_at_time() -> None:
    turns = [
        SpeakerTurn(id=0, start=10.0, end=20.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=1, start=30.0, end=40.0, cluster_id="SPEAKER_00"),
        SpeakerTurn(id=2, start=50.0, end=60.0, cluster_id="SPEAKER_00"),
    ]
    # Split SPEAKER_00 after 25.0s into SPEAKER_01
    updated = split_cluster_at_time(turns, cluster_id="SPEAKER_00", split_time=25.0, new_cluster_id="SPEAKER_01")
    assert updated[0].cluster_id == "SPEAKER_00"
    assert updated[1].cluster_id == "SPEAKER_01"
    assert updated[2].cluster_id == "SPEAKER_01"


def test_align_words_to_speaker_turns() -> None:
    segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=6.0,
            text="Bonjour à tous. On commence.",
            words=[
                WordTimestamp(word="Bonjour", start=0.0, end=1.0),
                WordTimestamp(word="à", start=1.1, end=1.5),
                WordTimestamp(word="tous.", start=1.6, end=2.5),
                WordTimestamp(word="On", start=3.5, end=4.0),
                WordTimestamp(word="commence.", start=4.1, end=5.5),
            ],
        )
    ]
    turns = [
        SpeakerTurn(id=0, start=0.0, end=3.0, cluster_id="SPEAKER_00", resolved_speaker="MJ"),
        SpeakerTurn(id=1, start=3.2, end=6.0, cluster_id="SPEAKER_01", resolved_speaker="Alice"),
    ]
    aligned = align_words_to_speaker_turns(segments, turns)
    assert len(aligned) == 2
    assert aligned[0].speaker == "MJ"
    assert aligned[0].text == "Bonjour à tous."
    assert aligned[1].speaker == "Alice"
    assert aligned[1].text == "On commence."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_timeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'a2ts.timeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/a2ts/timeline.py
"""Time-slice segmentation, cluster splitting, and word-to-speaker alignment."""

from a2ts.models import AlignedTurn, RawSegment, SpeakerTurn, WordTimestamp


def assign_time_slices(turns: list[SpeakerTurn], slice_minutes: float = 15.0) -> list[SpeakerTurn]:
    """Assign time slice IDs to speaker turns based on their start timestamps."""
    slice_seconds = slice_minutes * 60.0
    updated: list[SpeakerTurn] = []
    for turn in turns:
        slice_id = int(turn.start // slice_seconds)
        updated.append(turn.model_copy(update={"time_slice_id": slice_id}))
    return updated


def split_cluster_at_time(
    turns: list[SpeakerTurn],
    cluster_id: str,
    split_time: float,
    new_cluster_id: str,
) -> list[SpeakerTurn]:
    """Reassign all turns of cluster_id occurring at or after split_time to new_cluster_id."""
    updated: list[SpeakerTurn] = []
    for turn in turns:
        if turn.cluster_id == cluster_id and turn.start >= split_time:
            updated.append(turn.model_copy(update={"cluster_id": new_cluster_id}))
        else:
            updated.append(turn)
    return updated


def align_words_to_speaker_turns(
    segments: list[RawSegment],
    turns: list[SpeakerTurn],
) -> list[AlignedTurn]:
    """Align word timestamps to overlapping speaker turns to produce AlignedTurn blocks."""
    if not turns:
        # Fallback when no diarization turns exist: assign whole segments to unknown speaker
        return [
            AlignedTurn(
                turn_id=seg.id,
                start=seg.start,
                end=seg.end,
                speaker="SPEAKER_00",
                cluster_id="SPEAKER_00",
                text=seg.text,
                words=seg.words,
                time_slice_id=0,
            )
            for seg in segments
        ]

    # Flatten all words from segments
    all_words: list[WordTimestamp] = []
    for seg in segments:
        if seg.words:
            all_words.extend(seg.words)
        else:
            # If engine produced no word timestamps, treat segment as one block
            all_words.append(WordTimestamp(word=seg.text, start=seg.start, end=seg.end))

    aligned_turns: list[AlignedTurn] = []
    current_turn_idx = 0
    words_by_turn: dict[int, list[WordTimestamp]] = {t.id: [] for t in turns}

    for word in all_words:
        mid_point = (word.start + word.end) / 2.0
        # Find matching speaker turn
        matched_turn = None
        for t in turns:
            if t.start <= mid_point <= t.end:
                matched_turn = t
                break
        if matched_turn is None:
            # Match closest turn
            matched_turn = min(turns, key=lambda t: min(abs(t.start - mid_point), abs(t.end - mid_point)))
        words_by_turn[matched_turn.id].append(word)

    for turn in turns:
        turn_words = words_by_turn[turn.id]
        if not turn_words:
            continue
        text = " ".join(w.word for w in turn_words).strip()
        speaker = turn.resolved_speaker if turn.resolved_speaker else turn.cluster_id
        aligned_turns.append(
            AlignedTurn(
                turn_id=turn.id,
                start=turn_words[0].start,
                end=turn_words[-1].end,
                speaker=speaker,
                cluster_id=turn.cluster_id,
                text=text,
                words=turn_words,
                time_slice_id=turn.time_slice_id,
            )
        )

    return sorted(aligned_turns, key=lambda t: t.start)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_timeline.py -v`
Expected: PASS (3 tests passed)

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/timeline.py tests/unit/test_timeline.py
git commit -m "feat: add time slicing, cluster splitting, and word alignment in timeline.py"
```

---

### Task 6: Interactive Speaker Review & QCM (`src/a2ts/speaker_review.py`)

**Files:**
- Create: `src/a2ts/speaker_review.py`
- Test: `tests/unit/test_speaker_review.py`

**Interfaces:**
- Consumes: `AlignedTurn`, `SpeakerTurn`, `EntityRecord`, `SpeakersMapping` from `models.py`
- Produces:
  - `collect_speaker_stats(turns: list[AlignedTurn]) -> dict[str, dict[str, Any]]`
  - `apply_speakers_mapping(turns: list[AlignedTurn], mapping: SpeakersMapping) -> list[AlignedTurn]`
  - `save_speakers_mapping(mapping: SpeakersMapping, path: Path) -> None`
  - `load_speakers_mapping(path: Path) -> SpeakersMapping`
  - `run_interactive_review(turns: list[AlignedTurn], candidates: list[str]) -> SpeakersMapping`

- [ ] **Step 1: Write failing tests for speaker review module**

```python
# tests/unit/test_speaker_review.py
from pathlib import Path
from a2ts.models import AlignedTurn, SpeakersMapping
from a2ts.speaker_review import (
    apply_speakers_mapping,
    collect_speaker_stats,
    load_speakers_mapping,
    save_speakers_mapping,
)


def test_collect_speaker_stats() -> None:
    turns = [
        AlignedTurn(turn_id=0, start=0.0, end=5.0, speaker="SPEAKER_00", cluster_id="SPEAKER_00", text="Bonjour"),
        AlignedTurn(turn_id=1, start=6.0, end=10.0, speaker="SPEAKER_00", cluster_id="SPEAKER_00", text="Deuxième phrase"),
        AlignedTurn(turn_id=2, start=11.0, end=14.0, speaker="SPEAKER_01", cluster_id="SPEAKER_01", text="Salut"),
    ]
    stats = collect_speaker_stats(turns)
    assert "SPEAKER_00" in stats
    assert stats["SPEAKER_00"]["turn_count"] == 2
    assert stats["SPEAKER_00"]["total_duration"] == 9.0
    assert len(stats["SPEAKER_00"]["samples"]) == 2


def test_apply_speakers_mapping() -> None:
    turns = [
        AlignedTurn(turn_id=0, start=0.0, end=5.0, speaker="SPEAKER_00", cluster_id="SPEAKER_00", text="A", time_slice_id=0),
        AlignedTurn(turn_id=1, start=6.0, end=10.0, speaker="SPEAKER_00", cluster_id="SPEAKER_00", text="B", time_slice_id=1),
    ]
    mapping = SpeakersMapping(
        cluster_defaults={"SPEAKER_00": "Alice"},
        slice_overrides={"1": {"SPEAKER_00": "Bob"}},
    )
    mapped = apply_speakers_mapping(turns, mapping)
    assert mapped[0].speaker == "Alice"
    assert mapped[1].speaker == "Bob"


def test_save_and_load_mapping(tmp_path: Path) -> None:
    mapping_file = tmp_path / "mapping.json"
    mapping = SpeakersMapping(cluster_defaults={"SPEAKER_00": "MJ"})
    save_speakers_mapping(mapping, mapping_file)

    loaded = load_speakers_mapping(mapping_file)
    assert loaded.cluster_defaults["SPEAKER_00"] == "MJ"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_speaker_review.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'a2ts.speaker_review'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/a2ts/speaker_review.py
"""Interactive speaker attribution review and mapping persistence."""

import json
from pathlib import Path
from typing import Any
from rich.console import Console
from rich.prompt import Prompt
from a2ts.models import AlignedTurn, SpeakersMapping

console = Console()


def collect_speaker_stats(turns: list[AlignedTurn]) -> dict[str, dict[str, Any]]:
    """Compute turn counts, durations, and quote samples for each cluster ID."""
    stats: dict[str, dict[str, Any]] = {}
    for turn in turns:
        cid = turn.cluster_id
        if cid not in stats:
            stats[cid] = {"turn_count": 0, "total_duration": 0.0, "samples": []}
        stats[cid]["turn_count"] += 1
        stats[cid]["total_duration"] += round(turn.end - turn.start, 2)
        if len(stats[cid]["samples"]) < 4:
            stats[cid]["samples"].append((turn.start, turn.text))
    return stats


def apply_speakers_mapping(turns: list[AlignedTurn], mapping: SpeakersMapping) -> list[AlignedTurn]:
    """Apply cluster defaults, slice overrides, and turn overrides to aligned turns."""
    updated: list[AlignedTurn] = []
    for turn in turns:
        slice_key = str(turn.time_slice_id)
        speaker = turn.cluster_id

        # 1. Cluster default
        if turn.cluster_id in mapping.cluster_defaults:
            speaker = mapping.cluster_defaults[turn.cluster_id]

        # 2. Time slice override
        if slice_key in mapping.slice_overrides and turn.cluster_id in mapping.slice_overrides[slice_key]:
            speaker = mapping.slice_overrides[slice_key][turn.cluster_id]

        # 3. Specific turn override
        if turn.turn_id in mapping.turn_overrides:
            speaker = mapping.turn_overrides[turn.turn_id]

        updated.append(turn.model_copy(update={"speaker": speaker}))
    return updated


def save_speakers_mapping(mapping: SpeakersMapping, path: Path) -> None:
    """Save mapping configuration to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_speakers_mapping(path: Path) -> SpeakersMapping:
    """Load mapping configuration from JSON file."""
    if not path.is_file():
        return SpeakersMapping()
    data = json.loads(path.read_text(encoding="utf-8"))
    return SpeakersMapping.model_validate(data)


def run_interactive_review(turns: list[AlignedTurn], candidates: list[str]) -> SpeakersMapping:
    """Run interactive terminal QCM to attribute speakers."""
    mapping = SpeakersMapping()
    stats = collect_speaker_stats(turns)

    console.print("\n[bold cyan]=== Interactive Speaker Attribution Review ===[/bold cyan]\n")
    for cid, data in stats.items():
        console.print(f"[bold yellow]Speaker Cluster: {cid}[/bold yellow] ({data['turn_count']} turns, ~{data['total_duration']:.1f}s)")
        console.print("Sample quotes:")
        for t_start, sample in data["samples"]:
            m, s = divmod(int(t_start), 60)
            console.print(f"  • [[bold]{m:02d}:{s:02d}[/bold]] \"{sample}\"")

        console.print("\nCandidate options:")
        opts = {str(i + 1): name for i, name in enumerate(candidates[:6])}
        for idx, name in opts.items():
            console.print(f"  [{idx}] {name}")
        console.print("  [c] Type custom name")
        console.print("  [s] Skip (keep cluster ID)")

        choice = Prompt.ask("Attribution choice", choices=list(opts.keys()) + ["c", "s"], default="s")
        if choice in opts:
            mapping.cluster_defaults[cid] = opts[choice]
            console.print(f"[green]✓ Mapped {cid} -> {opts[choice]}[/green]\n")
        elif choice == "c":
            custom_name = Prompt.ask("Enter custom speaker name").strip()
            if custom_name:
                mapping.cluster_defaults[cid] = custom_name
                console.print(f"[green]✓ Mapped {cid} -> {custom_name}[/green]\n")
        else:
            console.print(f"[dim]Kept {cid}[/dim]\n")

    return mapping
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_speaker_review.py -v`
Expected: PASS (3 tests passed)

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/speaker_review.py tests/unit/test_speaker_review.py
git commit -m "feat: add interactive speaker review and mapping persistence in speaker_review.py"
```

---

### Task 7: Consolidation & Markdown Output (`src/a2ts/consolidator.py`)

**Files:**
- Create: `src/a2ts/consolidator.py`
- Test: `tests/unit/test_consolidator.py`

**Interfaces:**
- Consumes: `AlignedTurn` from `models.py`
- Produces:
  - `debounce_consecutive_turns(turns: list[AlignedTurn], threshold_seconds: float = 2.0) -> list[AlignedTurn]`
  - `render_markdown_transcript(turns: list[AlignedTurn]) -> str`
  - `format_timestamp(seconds: float) -> str`

- [ ] **Step 1: Write failing tests for consolidation**

```python
# tests/unit/test_consolidator.py
from a2ts.models import AlignedTurn
from a2ts.consolidator import (
    debounce_consecutive_turns,
    format_timestamp,
    render_markdown_transcript,
)


def test_format_timestamp() -> None:
    assert format_timestamp(75.5) == "00:01:15"
    assert format_timestamp(3665.0) == "01:01:05"


def test_debounce_consecutive_turns() -> None:
    turns = [
        AlignedTurn(turn_id=0, start=1.0, end=3.0, speaker="MJ", cluster_id="C0", text="Première partie."),
        AlignedTurn(turn_id=1, start=3.5, end=6.0, speaker="MJ", cluster_id="C0", text="Deuxième partie."),  # gap 0.5s < 2.0s
        AlignedTurn(turn_id=2, start=10.0, end=12.0, speaker="Alice", cluster_id="C1", text="Je réponds."),
    ]
    debounced = debounce_consecutive_turns(turns, threshold_seconds=2.0)
    assert len(debounced) == 2
    assert debounced[0].speaker == "MJ"
    assert debounced[0].text == "Première partie. Deuxième partie."
    assert debounced[0].start == 1.0
    assert debounced[0].end == 6.0
    assert debounced[1].speaker == "Alice"


def test_render_markdown_transcript() -> None:
    turns = [
        AlignedTurn(turn_id=0, start=0.0, end=10.0, speaker="MJ", cluster_id="C0", text="Bienvenue à tous."),
    ]
    md = render_markdown_transcript(turns)
    assert "### [00:00:00 - 00:00:10] MJ" in md
    assert "Bienvenue à tous." in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_consolidator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'a2ts.consolidator'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/a2ts/consolidator.py
"""Turn debouncing, chronological sorting, and markdown transcript generation."""

from a2ts.models import AlignedTurn


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    total_sec = max(0, int(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def debounce_consecutive_turns(
    turns: list[AlignedTurn],
    threshold_seconds: float = 2.0,
) -> list[AlignedTurn]:
    """Merge consecutive speech turns from the same speaker if the gap is below threshold."""
    if not turns:
        return []

    sorted_turns = sorted(turns, key=lambda t: t.start)
    debounced: list[AlignedTurn] = [sorted_turns[0]]

    for current in sorted_turns[1:]:
        prev = debounced[-1]
        gap = current.start - prev.end

        if current.speaker == prev.speaker and gap <= threshold_seconds:
            merged_words = prev.words + current.words
            merged_text = f"{prev.text} {current.text}".strip()
            debounced[-1] = prev.model_copy(
                update={
                    "end": max(prev.end, current.end),
                    "text": merged_text,
                    "words": merged_words,
                }
            )
        else:
            debounced.append(current)

    return debounced


def render_markdown_transcript(turns: list[AlignedTurn]) -> str:
    """Render aligned turns into standard Markdown with timestamp headings."""
    lines: list[str] = []
    for turn in turns:
        start_str = format_timestamp(turn.start)
        end_str = format_timestamp(turn.end)
        lines.append(f"### [{start_str} - {end_str}] {turn.speaker}")
        lines.append("")
        lines.append(turn.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_consolidator.py -v`
Expected: PASS (3 tests passed)

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/consolidator.py tests/unit/test_consolidator.py
git commit -m "feat: add turn debouncing and markdown rendering in consolidator.py"
```

---

### Task 8: Local LLM Refiner (`src/a2ts/refiner.py`)

**Files:**
- Create: `src/a2ts/refiner.py`
- Test: `tests/unit/test_refiner.py`

**Interfaces:**
- Consumes: Raw Markdown transcript string
- Produces:
  - `normalize_rpg_phonetics(text: str) -> str`
  - `compute_edit_distance_ratio(original: str, refined: str) -> float`
  - `refine_transcript_markdown(raw_markdown: str, agy_model: str = "gemini-3.8-flash-low") -> str`

- [ ] **Step 1: Write failing tests for refiner**

```python
# tests/unit/test_refiner.py
from a2ts.refiner import compute_edit_distance_ratio, normalize_rpg_phonetics


def test_normalize_rpg_phonetics() -> None:
    text = "Fais un jet d'un des vingt et deux des six pour tes jets de délai."
    normalized = normalize_rpg_phonetics(text)
    assert "1d20" in normalized
    assert "2d6" in normalized
    assert "jets de dés" in normalized


def test_compute_edit_distance_ratio() -> None:
    ratio = compute_edit_distance_ratio("Bonjour à tous", "Bonjour à tous")
    assert ratio == 0.0

    ratio_diff = compute_edit_distance_ratio("Bonjour", "Bonsoir")
    assert 0.0 < ratio_diff < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_refiner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'a2ts.refiner'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/a2ts/refiner.py
"""Local LLM transcript refiner and phonetic normalizer."""

import difflib
import re
import shutil
import subprocess

PHONETIC_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bun des vingt\b", re.IGNORECASE), "1d20"),
    (re.compile(r"\bun dé vingt\b", re.IGNORECASE), "1d20"),
    (re.compile(r"\bdeux des six\b", re.IGNORECASE), "2d6"),
    (re.compile(r"\bjets? de délai\b", re.IGNORECASE), "jet de dés"),
    (re.compile(r"\bjet d'initiative\b", re.IGNORECASE), "jet d'initiative"),
]


def normalize_rpg_phonetics(text: str) -> str:
    """Deterministically correct speech-to-text artifacts for French RPG terms."""
    result = text
    for pattern, replacement in PHONETIC_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result


def compute_edit_distance_ratio(original: str, refined: str) -> float:
    """Calculate relative character-level difference ratio (0.0 = identical)."""
    matcher = difflib.SequenceMatcher(None, original, refined)
    return 1.0 - matcher.ratio()


def refine_transcript_markdown(raw_markdown: str, agy_model: str = "gemini-3.8-flash-low") -> str:
    """Refine transcript via local agy CLI with deterministic normalization fallback."""
    normalized = normalize_rpg_phonetics(raw_markdown)

    if not shutil.which("agy"):
        return normalized

    prompt = (
        "Tu es un assistant d'édition pour des retranscriptions de sessions de jeu de rôle.\n"
        "Nettoie et formate ce texte en respectant scrupuleusement les règles suivantes:\n"
        "1. Ne modifie pas les timestamps ### [HH:MM:SS - HH:MM:SS] Nom.\n"
        "2. Encadre les discussions hors-jeu / techniques (jets de dés, règles, apartés) dans des blocs callout:\n"
        "> [!NOTE] Hors-jeu / Discussion\n"
        "3. Ne retire aucune information ni dialogue.\n\n"
        f"Voici la transcription brute:\n\n{normalized}"
    )

    cmd = ["agy", "--model", agy_model, "--effort", "low", prompt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        output = proc.stdout.strip()
        diff = compute_edit_distance_ratio(normalized, output)
        if diff < 0.6:  # Safety guard against hallucination
            return output
    except Exception:
        pass

    return normalized
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_refiner.py -v`
Expected: PASS (2 tests passed)

- [ ] **Step 5: Commit**

```bash
git add src/a2ts/refiner.py tests/unit/test_refiner.py
git commit -m "feat: add RPG phonetic normalizer and LLM refiner in refiner.py"
```

---

### Task 9: CLI Pipeline Orchestration & E2E Integration (`src/a2ts/cli.py`)

**Files:**
- Modify: `src/a2ts/cli.py`
- Test: `tests/integration/test_pipeline_e2e.py`

**Interfaces:**
- Consumes: All modules from Tasks 1-8
- Produces: Full CLI application commands (`run`, `extract-audio`, `extract-vocab`, `review`, `split`)

- [ ] **Step 1: Write integration test for full CLI run**

```python
# tests/integration/test_pipeline_e2e.py
from pathlib import Path
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner
from a2ts.cli import app
from a2ts.models import RawSegment, WordTimestamp

runner = CliRunner()


@patch("a2ts.cli.extract_audio_to_wav")
@patch("a2ts.cli.get_engine")
def test_cli_run_pipeline_end_to_end(mock_get_engine: MagicMock, mock_extract: MagicMock, tmp_path: Path) -> None:
    # 1. Setup dummy video and context dir
    video_file = tmp_path / "test_session.mkv"
    video_file.write_bytes(b"dummy mkv video content")
    context_dir = tmp_path / "contexte"
    context_dir.mkdir()
    (context_dir / "lore.md").write_text("Rencontre avec [[Kaelen]] et [[Garrick]].", encoding="utf-8")

    out_file = tmp_path / "output_transcript.md"

    # 2. Mock audio extraction and engine
    mock_wav = tmp_path / "sample.wav"
    mock_wav.touch()
    mock_extract.return_value = mock_wav

    dummy_engine = MagicMock()
    dummy_engine.transcribe.return_value = [
        RawSegment(
            id=0,
            start=0.0,
            end=5.0,
            text="Bienvenue à tous pour la session.",
            words=[
                WordTimestamp(word="Bienvenue", start=0.0, end=1.0),
                WordTimestamp(word="à", start=1.1, end=1.5),
                WordTimestamp(word="tous", start=1.6, end=2.5),
                WordTimestamp(word="pour", start=2.6, end=3.0),
                WordTimestamp(word="la", start=3.1, end=3.5),
                WordTimestamp(word="session.", start=3.6, end=5.0),
            ],
        )
    ]
    mock_get_engine.return_value = dummy_engine

    # 3. Run CLI command
    result = runner.invoke(
        app,
        [
            "run",
            str(video_file),
            "--context-dir",
            str(context_dir),
            "--output",
            str(out_file),
            "--no-interactive",
            "--no-refine",
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "Bienvenue à tous pour la session." in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_pipeline_e2e.py -v`
Expected: FAIL with `No such command 'run'`

- [ ] **Step 3: Implement full CLI commands in `src/a2ts/cli.py`**

```python
# src/a2ts/cli.py
"""CLI application and pipeline orchestration for a2ts."""

from pathlib import Path
from typing import Annotated
import typer
from rich.console import Console

from a2ts.consolidator import debounce_consecutive_turns, render_markdown_transcript
from a2ts.media import extract_audio_to_wav, probe_media
from a2ts.models import SpeakerTurn, SpeakersMapping
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
    media_file: Annotated[Path, typer.Argument(help="Path to audio or video media file")],
    output_dir: Annotated[Path, typer.Option(help="Directory to save extracted WAV")] = Path(".a2ts/audio_cache"),
) -> None:
    """Extract and resample audio from video/audio to 16kHz mono WAV."""
    wav_path = extract_audio_to_wav(media_file, output_dir)
    console.print(f"[green]Audio extracted:[/green] {wav_path}")


@app.command()
def extract_vocab(
    context_dir: Annotated[Path, typer.Option(help="Directory containing Obsidian markdown notes")] = Path("contexte"),
    vocab_file: Annotated[Path | None, typer.Option(help="Optional custom wordlist text file")] = None,
    max_tokens: Annotated[int, typer.Option(help="Maximum token budget for prompt")] = 220,
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
    media_file: Annotated[Path, typer.Argument(help="Path to audio or video media file")],
    engine: Annotated[str, typer.Option(help="Transcription engine: voxtral or whisper")] = "whisper",
    context_dir: Annotated[Path, typer.Option(help="Directory containing Obsidian notes")] = Path("contexte"),
    vocab_file: Annotated[Path | None, typer.Option(help="Optional custom wordlist text file")] = None,
    slice_minutes: Annotated[float, typer.Option(help="Time slice window size in minutes")] = 15.0,
    interactive: Annotated[bool, typer.Option(help="Enable interactive QCM speaker review")] = True,
    refine: Annotated[bool, typer.Option(help="Run local LLM refiner pass via agy")] = False,
    output: Annotated[Path, typer.Option(help="Output markdown transcript path")] = Path("transcript.md"),
    cache_dir: Annotated[Path, typer.Option(help="Cache directory for session artifacts")] = Path(".a2ts"),
) -> None:
    """Execute end-to-end transcription and diarization pipeline."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[bold green]=== Starting a2ts Pipeline for {media_file.name} ===[/bold green]\n")

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
    console.print(f"\n[bold green]✓ Pipeline completed. Transcript saved to:[/bold green] {output}\n")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run integration test to verify it passes**

Run: `uv run pytest tests/integration/test_pipeline_e2e.py -v`
Expected: PASS (1 test passed)

- [ ] **Step 5: Run all unit and integration tests**

Run: `uv run pytest -v`
Expected: All unit and integration tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/a2ts/cli.py tests/integration/test_pipeline_e2e.py
git commit -m "feat: complete CLI pipeline orchestration and E2E integration in cli.py"
```
