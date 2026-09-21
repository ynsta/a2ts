# a2ts Architecture & Pipeline Design

**Date**: 2026-09-21  
**Project**: `a2ts` (Audio to Text / Transcript)  
**Tier**: A (Personal / Fast iteration / ML Tool)  
**Type**: Python 3.13 (`uv`, Hatchling)

---

## 1. Executive Summary & Goals

`a2ts` is a local audio/video transcription and diarization pipeline designed for processing single-stream media recordings (tabletop RPG sessions, meetings, presentations, video logs). It supports contextual vocabulary extraction from Obsidian vaults (or custom wordlists), dual-engine transcription with Mistral's **Voxtral** or **Faster-Whisper**, time-sliced speaker diarization, an interactive QCM speaker-attribution review phase, and an optional local LLM refinement pass via `agy`.

Key objectives:
- **Local GPU execution**: Run transcription and diarization on local NVIDIA RTX hardware (tested on RTX 3080 10GB VRAM) with zero cloud token costs.
- **Single-stream media support**: Accept `.mp4`, `.mkv`, `.avi`, `.mov`, `.flac`, `.wav`, `.mp3` inputs with automatic audio extraction and resampling to 16kHz mono WAV.
- **Dual-Engine flexibility**: Support Mistral **Voxtral** (`mistralai/Voxtral-Mini-3B-2507`) via Hugging Face `transformers` and **Faster-Whisper** (`large-v3` / `turbo`) as a battle-tested alternative.
- **Contextual lore extraction**: Scan Obsidian markdown notes for `[[wikilinks]]`, `aliases:`, and headings to build a token-budgeted prompt biasing speech recognition towards domain terminology.
- **Time-sliced speaker attribution**: Overcome acoustic diarization collisions (`SPEAKER_00` containing multiple persons) through temporal slicing, split commands, and an interactive QCM review tool.
- **Future merger with `craig2ts`**: Use compatible data structures and shared design conventions to allow straightforward future unification of `a2ts` (single stream) and `craig2ts` (multi-track Discord).

---

## 2. System Architecture & Data Flow

```
[Audio / Video File] (.mp4, .mkv, .flac, .wav, .mp3)
        │
        ▼
1. `media.py` (ffprobe inspection + ffmpeg extraction to 16kHz mono WAV)
        │
        ├────────────────────────────────────────┐
        ▼                                        ▼
2. `vocab.py`                             3. `diarizer.py`
   (Mine Obsidian `contexte/` notes          (Cluster speech into raw turns:
    + optional `--vocab` wordlist)            `SPEAKER_00`, `SPEAKER_01`, ...)
        │                                        │
        ▼                                        │
4. `transcriber.py`                              │
   (Dual-Engine: Voxtral or Whisper              │
    using token-budgeted lore prompt)            │
        │                                        │
        └───────────────────┬────────────────────┘
                            ▼
5. `timeline.py` (Time-Slice Alignment & Boundary Slicing)
   - Assigns words and speech turns to time blocks (e.g. 15-min windows)
   - Flags acoustic outlier turns and cluster ambiguities
                            │
                            ▼
6. `speaker_review.py` (Interactive Attribution & QCM)
   - Shows stats and quote samples per speaker cluster
   - Supports time-slice overrides and split commands (`a2ts split`)
   - Generates persistent `speakers_mapping.json`
                            │
                            ▼
7. `consolidator.py` (Chronological Assembly & Debounce)
   - Merges contiguous speech turns (< 2.0s gap)
   - Generates `transcript.raw.md` + `.a2ts/session.json`
                            │
                            ▼
8. `refiner.py` (Optional Local LLM Refinement via `agy`)
   - Normalizes RPG terminology deterministically
   - Detects and formats out-of-character banter (`> [!NOTE] Hors-jeu`)
   - Levenshtein distance safety guard against hallucinations
                            │
                            ▼
[Output Markdown Transcript (`transcript.md`)]
```

---

## 3. Data Models (`models.py`)

All core domain models are defined using `pydantic.BaseModel` for validation and JSON serialization:

```python
class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 1.0


class RawSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[WordTimestamp] = []


class SpeakerTurn(BaseModel):
    id: int
    start: float
    end: float
    cluster_id: str  # e.g. "SPEAKER_00"
    resolved_speaker: str | None = None
    time_slice_id: int = 0
    flagged: bool = False
    notes: str = ""


class AlignedTurn(BaseModel):
    turn_id: int
    start: float
    end: float
    speaker: str  # Resolved name or cluster_id
    cluster_id: str
    text: str
    words: list[WordTimestamp] = []
    time_slice_id: int = 0


class EntityRecord(BaseModel):
    name: str
    kind: str  # "wikilink", "alias", "heading", "custom"
    source_file: str = ""


class SessionMetadata(BaseModel):
    media_path: str
    media_hash: str
    duration_seconds: float
    engine: str  # "voxtral" or "whisper"
    model_name: str
    prompt_hash: str
    time_slice_minutes: float
    created_at: str
```

---

## 4. Audio & Media Handling (`media.py`)

Handles arbitrary media inputs safely:
- **`probe_media(file_path: Path) -> MediaInfo`**: Invokes `ffprobe` in JSON mode to verify the presence of audio streams, identify audio codecs, channel layouts, sample rates, and duration.
- **`extract_audio(file_path: Path, output_dir: Path) -> Path`**:
  - Computes a fast SHA-256 hash of the media header/size to avoid redundant conversions.
  - Invokes `ffmpeg -y -i <input> -vn -acodec pcm_s16le -ar 16000 -ac 1 <output.wav>` to produce canonical 16kHz mono WAV.
  - Caches the output in `.a2ts/cache/<file_hash>.wav`.

---

## 5. Contextual Lore Extraction (`vocab.py`)

Reuses the proven, high-accuracy extraction logic from `craig2ts`:
- **Obsidian Vault Mining**:
  - Scans markdown files under `--context-dir` (default: `contexte/`).
  - Regex matchers extract:
    - `\[\[([^|\]]+)(?:\|([^\]]+))?\]\]` (Wikilinks and custom piped labels).
    - `aliases:\s*\n((?:\s*-\s*.+\n)+)` (YAML frontmatter aliases).
    - `##\s+([^\n#]+)` (Level-2 markdown headings).
  - Stopword filtering: filters common French stopwords and binary media extensions (`.png`, `.jpg`, `.pdf`, etc.).
- **Standalone Wordlist**:
  - Optional `--vocab <file.txt>` input containing additional character or location names (one per line).
- **Token-Budgeted Biasing Prompt**:
  - Deduplicates and ranks entity names.
  - Measures token usage using `tiktoken` (`cl100k_base` tokenizer).
  - Caps initial prompt at 220 tokens (safe budget for Whisper `initial_prompt`).
  - Exports prompt and entities to `.a2ts/vocab.json`.

---

## 6. Dual Transcription Engines (`transcriber.py`)

### The `TranscriberEngine` Protocol

```python
class TranscriberEngine(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str = "fr",
    ) -> list[RawSegment]: ...
```

### 1. `VoxtralEngine` (`--engine voxtral`, default)
- **Model Checkpoint**: `mistralai/Voxtral-Mini-3B-2507` (or configured Hugging Face checkpoint).
- **Runtime**: Loads via Hugging Face `transformers` (`VoxtralForConditionalGeneration` + `AutoProcessor`) on CUDA device in `bfloat16` or `float16`.
- **Context Injection**: Prepares instruction prompt containing contextual lore:
  ```json
  [
    {
      "role": "user",
      "content": [
        {"type": "audio", "path": "audio.wav"},
        {"type": "text", "text": "Transcris fidèlement cet audio en français. Termes de contexte: <lore_terms>"}
      ]
    }
  ]
  ```
- **Parsing**: Translates model output tokens and timestamp tags into `RawSegment` records.
- **Dependency Guard**: If optional dependencies (`transformers`, `torch`) are missing, provides an actionable error instructing the user to run `uv sync --extra voxtral` or switch to `--engine whisper`.

### 2. `WhisperEngine` (`--engine whisper`, fallback)
- **Model Checkpoint**: `faster-whisper` (`large-v3` default, or `large-v3-turbo`).
- **CUDA Runtime Loading**: Automatically detects and preloads bundled NVIDIA libraries (`libcublas.so.12`, `libcudnn.so.9`) into process memory.
- **Inference Settings**:
  - `vad_filter=True` (Silero VAD filtering).
  - `word_timestamps=True` (word-level precision required for time-slice alignment).
  - `initial_prompt=<vocab_prompt>`.
- **Caching**: Saves engine segments to `.a2ts/transcripts/<hash>_<engine>.json`.

---

## 7. Speaker Diarization & Time-Slice Management (`diarizer.py`, `timeline.py`)

### Diarization Engine (`diarizer.py`)
- Partitions audio stream into raw `SpeakerTurn` segments with cluster IDs (`SPEAKER_00`, `SPEAKER_01`, ...).
- Implements Silero VAD + acoustic feature extraction with agglomerative/spectral clustering (offline, zero external API token required).
- Caches diarization turns under `.a2ts/diarization/<file_hash>.json`.

### Temporal Slicing (`timeline.py`)
To handle diarization collisions (where `SPEAKER_00` contains two different speakers at different times):
1. **Time Slicing**: Splits the total session duration into configurable time blocks (e.g. 15 minutes per slice).
2. **Alignment Algorithm**:
   - For each word in `RawSegment`, finds the overlapping `SpeakerTurn`.
   - Groups continuous word spans into `AlignedTurn` blocks assigned to `(time_slice_id, cluster_id)`.
3. **Acoustic Outlier & Collision Flagging**:
   - Flags turns where cluster confidence is low or where dialogue structure indicates rapid alternation between supposedly identical speaker labels.
4. **Time-Split Operations**:
   - `split_cluster(cluster_id, timestamp, new_cluster_id)`: Slices cluster ownership at a specific timestamp.
   - `reassign_slice(slice_id, cluster_id, new_speaker)`: Reassigns a speaker label within a single time slice without affecting other slices.

---

## 8. Interactive Speaker Review & QCM (`speaker_review.py`)

Interactive terminal CLI interface powered by `rich`:
- **Speaker Profile Overview**:
  - Displays a summary table of all detected speaker clusters, total speech duration, and turn counts.
  - Extracts 3-5 representative sample quotes for each speaker (selecting opening turns and high-duration turns).
- **Candidate Options**:
  - Reads candidate names from Obsidian lore (`EntityRecord`) and optional `speakers.md`.
- **QCM Prompting**:
  ```
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Time Slice 1 [00:00:00 - 00:15:00]
  Speaker: SPEAKER_00 (18 turns, 4m 12s)
  Representative Quotes:
    • [00:01:23] "Bienvenue à tous pour la session, sortez vos fiches..."
    • [00:04:45] "Faites tous un jet d'initiative s'il vous plaît."
  Options:
    [1] MJ (Dungeon Master)
    [2] Alice [Kaelen]
    [3] Bob [Garrick]
    [4] Type custom name
    [s] Skip (keep SPEAKER_00)
    [t] Review by individual time slices
  Selection [1-4/s/t]: 1
  ✓ Mapped SPEAKER_00 -> MJ for Time Slice 1
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ```
- **Flagged Turn Review**:
  - Specifically prompts the user on flagged ambiguous segments where acoustic features or dialogue flow diverged from the cluster default.
- **Persistence**:
  - Saves mapping rules to `.a2ts/speakers_mapping.json`.
  - Re-running the pipeline loads existing mapping unless `--reset-speakers` is specified.

---

## 9. Timeline Consolidation & Formatting (`consolidator.py`)

- **Chronological Ordering**: Assembles all `AlignedTurn` objects sorted strictly by `start_time`.
- **Turn Debouncing**: Merges adjacent turns from the same speaker if the silence gap between them is `< 2.0s`.
- **Markdown Rendering**:
  Produces `transcript.raw.md` formatted with clean headings:
  ```markdown
  ### [00:01:23 - 00:01:40] MJ
  Bienvenue à tous pour la session, sortez vos fiches de personnage.

  ### [00:01:42 - 00:01:55] Alice [Kaelen]
  J'ai préparé mes sorts pour le niveau 4 !
  ```
- **Session State**: Writes full machine-readable provenance to `.a2ts/session.json`.

---

## 10. Local LLM Refiner (`refiner.py`)

Optional post-processing pass using the local `agy` CLI (`--refine` flag):
- **Model**: Default `gemini-3.8-flash-low` with `low` thinking effort (zero cloud API token costs).
- **Chunking**: Processes transcript in overlapping chunks of 20-30 speech blocks with a 3-block context window across boundaries.
- **Refinements Performed**:
  1. **Deterministic RPG phonetic cleanup**: `un des vingt` -> `1d20`, `deux des six` -> `2d6`, `jets de délai` -> `jets de dés`.
  2. **Out-of-character (HRP) separation**: Wraps dice discussions, rules questions, and table banter into GitHub-style callouts:
     ```markdown
     > [!NOTE] Hors-jeu / Discussion
     > Alice: "Attends, mon bonus d'attaque c'est +5 ou +6 ?"
     > MJ: "+6 avec l'arme magique."
     ```
  3. **NPC Character Attribution**: If the DM portrays an NPC, attributes dialogue cleanly (e.g. `MJ [Garde de la ville]`).
- **Safety Guard**: Compares Levenshtein edit distance between raw and refined text to prevent hallucinated rewrites; rejects any chunk where word drift exceeds tolerance.
- **Final Output**: Writes `transcript.md`.

---

## 11. CLI Commands (`cli.py`)

Built with `typer` and `rich`:

- `a2ts run <media_file>`: Full pipeline (audio extraction -> lore mining -> diarization -> transcription -> time-slice alignment -> interactive review -> consolidation).
  - `--engine [voxtral|whisper]`: Choose transcription engine (default `voxtral`).
  - `--context-dir <path>`: Obsidian vault or context directory (default `contexte/`).
  - `--vocab <path>`: Optional custom wordlist file.
  - `--slice-minutes <int>`: Temporal slice window size (default `15`).
  - `--interactive / --no-interactive`: Enable/disable interactive QCM speaker review.
  - `--refine / --no-refine`: Run local LLM refiner pass via `agy`.
  - `--output <path>`: Output transcript path (default `transcript.md`).
- `a2ts extract-audio <media_file>`: Run audio extraction only.
- `a2ts extract-vocab [--context-dir <path>]`: Extract and display lore entities and prompt budget.
- `a2ts review <session_dir>`: Re-open interactive QCM speaker attribution review on an existing session.
- `a2ts split <session_dir> <cluster_id> --at <timestamp> --to <new_speaker>`: Split a speaker cluster at a specific timestamp.

---

## 12. Future Merger with `craig2ts`

Both projects share:
- Python 3.13, `uv`, `pydantic`, `typer`, `rich`, `faster-whisper`.
- Identical `contexte/` parsing logic (`vocab.py`).
- Identical phonetic normalizer and `agy` refiner prompts (`refiner.py`).

Future unification roadmap:
- Combine into a single package (e.g. `rpg2ts` or unified `a2ts`) with two input adapters:
  1. `a2ts discord <craig_dir>`: Multi-track mode (from `craig2ts`).
  2. `a2ts media <file>`: Single-stream audio/video mode (from `a2ts`).

---

## 13. Testing Strategy

1. **Unit tests (`tests/unit/`)**:
   - `test_media.py`: Probe parsing, audio extraction command building, hash calculation.
   - `test_vocab.py`: Wikilink parsing, aliases block extraction, stopword filtering, token budget cap.
   - `test_timeline.py`: Temporal slicing, cluster splitting, turn debouncing.
   - `test_consolidator.py`: Markdown rendering, timestamp formatting.
2. **Integration tests (`tests/integration/`)**:
   - Synthetic audio fixtures (short 16kHz WAV with generated tones / silence).
   - Mock transcription engine verifying segment-to-timeline alignment.
   - Interactive review mocked with predefined QCM responses.
