# a2ts Functional Specification

**Version**: 0.2.0  
**Project**: `a2ts` (Audio to Text / Transcript)  
**Governance Tier**: Tier B  
**Target Runtime**: Python >= 3.13 (`uv`, Hatchling)  
**Hardware Profile**: Local GPU execution (NVIDIA CUDA Ampere+ / RTX 3080 10GB+)

---

## 1. Executive Summary & Purpose

`a2ts` is an autonomous, local-first audio and video transcription, speaker diarization, and attribution pipeline. It processes single-stream media recordings (such as tabletop RPG play sessions, team meetings, interviews, lectures, and streamed video captures), extracts contextual domain terminology from Obsidian knowledge bases to bias transcription, attributes speech to individual speakers through state-of-the-art acoustic diarization and voice profile databases, and renders structured Markdown transcripts.

The core design principle is **complete local sovereignty**: zero telemetry, zero cloud API fees, and zero external network dependencies during inference.

---

## 2. Target Users & Use Cases

1. **Tabletop Role-Playing Game (TTRPG) Sessions**:
   - Single-stream mixed microphone captures containing 3 to 8 players plus a Game Master.
   - Heavy presence of fantasy vocabulary, fictional entity names, and overlapping dialogue.
   - Long session durations (typically 2 to 4 hours per file).
2. **Recorded Technical Meetings & Seminars**:
   - Technical meetings where proprietary domain terms, acronyms, and product names must be transcribed accurately.
   - Fast attribution of turns to known team members.
3. **Podcasts & Video Logs**:
   - Multi-speaker conversational audio needing clean, timestamped Markdown for show notes, publication, or Obsidian second-brain storage.

---

## 3. Functional Capabilities

### 3.1 Media Ingestion & Audio Normalization
- Accepts video container formats (`.mp4`, `.mkv`, `.avi`, `.mov`) and audio files (`.flac`, `.wav`, `.mp3`, `.ogg`, `.m4a`).
- Probes media stream properties via `ffprobe` (duration, sample rate, channels, audio codec).
- Automatically extracts and resamples audio to **16 kHz 16-bit mono WAV** via `ffmpeg`.
- Content-addressable SHA-256 caching ensures repeated runs on the same media file skip redundant extraction.

### 3.2 Lore & Vocabulary Mining (Context Biasing)
- Scans user Obsidian vaults or Markdown context directories (default: `contexte/`).
- Extracts named entities, `[[wikilinks]]`, YAML frontmatter `aliases:`, and document headings (`#`, `##`, etc.).
- Deduplicates and tokenizes discovered terms using `cl100k_base` BPE tokenizer.
- Packs extracted terms into a token-budgeted prompt (default: 220 tokens, Whisper max 224 tokens) passed to the speech recognition engine to guide decoding toward domain terminology.

### 3.3 Dual-Engine Speech Recognition (ASR)
- **Faster-Whisper**:
  - Utilizes CTranslate2 engine for high-efficiency Whisper inference.
  - Supports model sizes: `large-v3`, `turbo`, `medium`, `small`.
  - Configurable compute types (`float16`, `int8`, `float32`).
  - Word-level timestamp generation via attention alignment.
- **Mistral Voxtral**:
  - Leverages Mistral AI's multimodal audio LLM (`mistralai/Voxtral-Mini-3B-2507`) via Hugging Face `transformers`.
  - Direct audio-context comprehension for nuanced speech patterns.

### 3.4 Acoustic Speaker Diarization
- **NVIDIA Nemotron-3 Diarization (Primary Engine)**:
  - 100M parameter Transformer encoder model (`nvidia/Nemotron-3-Diarization`).
  - End-to-end continuous speaker activity modeling (up to 8 concurrent speakers).
  - Native detection of overlapping speech without clustering thresholds.
  - Ultra-high throughput (> 130x real-time factor in FP16 on RTX 3080).
- **SpeechBrain ECAPA-TDNN (Fallback / Acoustic Embeddings)**:
  - Extracts 192-dimensional x-vector speaker embeddings.
  - Agglomerative clustering with cosine distance thresholding.
  - Generates speaker turns on CPU or when explicitly requested.
- **Engine Selection Policy (`--diarizer-engine`)**:
  - `auto` (default): Employs Nemotron-3 if CUDA is available; falls back to ECAPA on CPU or unexpected initialization failure.
  - `nemotron`: Forces Nemotron-3 Diarization.
  - `ecapa`: Forces SpeechBrain ECAPA-TDNN and agglomerative clustering.

### 3.5 Timeline Slicing & Word Alignment
- Projects word-level timestamps onto speaker turns based on midpoint timestamp matching.
- Segments long recordings into manageable time windows (default: 15-minute slices).
- Flags potential acoustic outlier turns or low-confidence segments.

### 3.6 Interactive Attribution & Voice Profile Matching
- **Voice Profile Recognition**:
  - Compares acoustic embeddings against persistent JSON database (`VoiceProfilesDatabase`).
  - Closed-set or similarity-threshold matching against known speakers.
  - Automatically enriches anonymous cluster IDs (`SPEAKER_00`) with human-readable names (`"Merrow"`, `"MJ"`).
- **Interactive Terminal Review (QCM)**:
  - Rich CLI review interface for ambiguous or unmapped speaker clusters.
  - Displays cluster statistics, sample dialogue quotes, and plays audio snippets.
  - Suggests candidate names mined from `speakers.txt` or Obsidian lore.
  - Saves speaker choices to persistent `.a2ts/sessions/<media-sha256>/speakers_mapping.json`.
- **Temporal Cluster Splitting**:
  - Subcommand `a2ts split <session_dir> <cluster_id> --at <seconds> --to <new_cluster_id>` splits acoustic clusters when physical speaker changes mid-recording.

### 3.7 Transcript Consolidation & LLM Refinement
- Debounces contiguous speaker turns (merges turns from the same speaker separated by <= 2.0s silence).
- Formats structured Markdown output with timestamp headers:
  ```markdown
  ### [00:01:15 - 00:01:28] Brakk
  
  J'ai avancé en éclaireur quand il y a des bêtes qui ont sauté.
  ```
- Deterministic French Tabletop RPG Normalization:
  - Normalizes speech recognition artifacts for dice rolls (`lance un dé 20` -> `lance 1d20`, `trois dés de six` -> `3d6`, `un dé cent` -> `1d100`).
  - Guards against false positives using following noun exclusions (`20 gardes`, `10 minutes`, `6 joueurs`).
  - Corrects phonetic ASR errors (`jets-dés`, `jet de délai` -> `jets de dés` / `jet de dés`).
- Optional local LLM refinement pass via `agy` CLI (`--refine`):
  - Normalizes vocabulary deterministically.
  - **Data Boundary**: Core inference (ASR and diarization) runs 100% locally. The `--refine` pass delegates chunked markdown transcripts to the external `agy` CLI binary (`agy --model <model> --effort <effort>`), which defaults to Google Gemini (`gemini-3.8-flash-low`) or another configured model, introducing network egress when cloud providers are configured.
  - Raw unrefined markdown is always written to `<output>.raw.md` before refinement begins.
  - Separates in-character roleplay from out-of-character remarks (`> [!NOTE] Hors-jeu`).
  - Enforces safety checks: verifies all turn headers (`### [HH:MM:SS - HH:MM:SS]`) are preserved and word count remains within a 15% delta, falling back to raw chunk on mismatch to prevent hallucination.

### 3.8 Multi-Track Ingestion & Discord Craig Pipeline
- Accepts directories containing isolated per-speaker `.flac` tracks produced by the Craig Discord recording bot (`^(\d+)-(.*)\.flac`).
- Automatically extracts Discord usernames from track filenames.
- Parses optional `speakers.md` roster files to map usernames to character names, classes/roles, nicknames, and Game Master (`is_dm`) statuses.
- Parses optional `info.txt` metadata recorded by Craig (channel, guild, user IDs).
- Constructs lore-biased transcription prompts integrating character names, nicknames, and Obsidian vault entities under a token budget.
- Performs discrete Faster-Whisper track transcription with individual JSON caching (`<recording_dir>/.transcripts/<track_stem>.json`) governed by `TrackCacheProvenance`.
- Interleaves multi-track segments into a unified chronological sequence (`merge_craig_tracks_to_turns`) sorted by segment start time and track identifier.
- Bypasses acoustic diarization entirely since microphone isolation provides ground-truth speaker attribution.
- Merges consecutive utterances from the same speaker via configurable debouncing (`--debounce`).
- Supports the optional local LLM refinement pass (`--refine`).

---

## 4. CLI Interface & Contract

The application exposes the `a2ts` CLI with subcommands:

### 4.1 `a2ts run`
Executes the full transcription and diarization pipeline.

```bash
a2ts run <media_file> [OPTIONS]
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `media_file` | `Path` (Arg) | *Required* | Path to input audio or video file |
| `--engine` | `str` | `whisper` | ASR engine: `whisper` or `voxtral` |
| `--model-name` | `str` | *Engine default* | Model checkpoint (e.g. `turbo`, `large-v3`) |
| `--device` | `str` | `cuda` | Hardware device: `cuda` or `cpu` |
| `--compute-type` | `str` | `float16` | Precision: `float16`, `int8`, `float32` |
| `--context-dir` | `Path` | `contexte` | Obsidian notes folder for lore extraction |
| `--vocab-file` | `Path` | `None` | Optional custom vocabulary text file |
| `--slice-minutes` | `float` | `15.0` | Time slice window size in minutes |
| `--diarize / --no-diarize` | `bool` | `True` | Enable acoustic speaker diarization |
| `--diarizer-engine` | `str` | `auto` | Diarizer backend: `auto`, `nemotron`, `ecapa` |
| `--cluster-threshold` | `float` | `0.60` | Cosine distance threshold for speaker clustering |
| `--num-speakers` | `int` | `None` | Target speaker count constraint |
| `--voice-profiles` | `Path` | `contexte/voice_profiles.json` | Voice profile database JSON path |
| `--profile-threshold` | `float` | `0.60` | Cosine similarity threshold for voice profiles |
| `--closed-set / --no-closed-set` | `bool` | `False` | Force match all clusters to nearest enrolled profile |
| `--speakers` | `str` | `None` | Comma-separated list of known speaker names |
| `--speakers-file` | `Path` | `None` | Path to text file listing known speaker names |
| `--interactive / --no-interactive` | `bool` | `True` | Enable interactive terminal QCM speaker review |
| `--force / --no-force` | `bool` | `False` | Force re-running transcription/diarization, ignoring cache |
| `--rpg-normalize / --no-rpg-normalize` | `bool` | `True` | Normalize French tabletop RPG dice expressions and terms |
| `--refine / --no-refine` | `bool` | `False` | Run LLM post-processing via `agy` CLI (remote delegation) |
| `--refine-model` | `str` | `gemini-3.8-flash-low` | Model name for LLM refiner via `agy` CLI |
| `--output` | `Path` | `transcript.md` | Final Markdown transcript destination |
| `--cache-dir` | `Path` | `.a2ts` | Local artifact cache directory |

### 4.2 `a2ts craig`
Transcribes and merges multi-track Craig Discord recordings into a unified transcript.

```bash
a2ts craig <recording_dir> [OPTIONS]
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `recording_dir` | `Path` (Arg) | *Required* | Path to folder containing Craig `.flac` audio tracks |
| `--model-name` | `str` | `large-v3` | Whisper model name |
| `--device` | `str` | `auto` | Device to run inference on (`auto`, `cuda`, `cpu`) |
| `--compute-type` | `str` | `float16` | Computation type (`float16`, `int8`, `float32`, etc.) |
| `--context-dir` | `Path` | `contexte` | Directory containing Obsidian markdown notes |
| `--speakers-file` | `Path` | `None` | Path to speakers roster markdown file (`speakers.md`) |
| `--output` | `Path` | `None` | Output markdown transcript path (defaults to `<recording_dir>/transcript.md`) |
| `--debounce` | `float` | `2.0` | Debounce window in seconds for consecutive turns |
| `--force` | `bool` | `False` | Force re-transcription ignoring existing `.transcripts/` cache |
| `--refine / --no-refine` | `bool` | `False` | Run LLM refiner pass on transcript via `agy` CLI |
| `--refine-model` | `str` | `gemini-3.8-flash-low` | Model name for LLM refiner |
| `--refine-effort` | `str` | `low` | Reasoning effort for LLM refiner (`low`, `medium`, `high`) |

### 4.3 `a2ts review`
Re-runs interactive speaker review on cached session turns without re-transcribing audio. Target session directory defaults to `.a2ts` (resolving `.a2ts/sessions/<media-sha256>/`).

```bash
a2ts review [SESSION_DIR] [OPTIONS]
```

### 4.4 `a2ts recluster`
Re-runs agglomerative clustering on cached turn embeddings with a modified threshold or speaker count constraint. Target session directory defaults to `.a2ts` (resolving `.a2ts/sessions/<media-sha256>/`).

```bash
a2ts recluster [SESSION_DIR] [OPTIONS]
```

### 4.5 `a2ts split`
Splits an existing speaker cluster at a given timestamp to correct acoustic cluster collisions and persists the split in the session's `speakers_mapping.json`.

```bash
a2ts split <session_dir> <cluster_id> --at <split_time_seconds> --to <new_cluster_id> [OPTIONS]
```

### 4.6 `a2ts extract-vocab`
Scans a context directory and writes the mined lore vocabulary to standard output or a file.

```bash
a2ts extract-vocab [DIRECTORY] [OPTIONS]
```

### 4.7 `a2ts info`
Displays current environment, hardware accelerators, CUDA availability, and installed engine capabilities.

---

## 5. Non-Functional Requirements & Constraints

1. **Hardware Envelope**: Must operate reliably within a 10 GB VRAM GPU ceiling (NVIDIA RTX 3080/4070/4080/4090).
2. **Idempotency & Cache Provenance**:
   - Streaming SHA-256 media hashing over full file bytes.
   - Cache envelopes (`TranscriptCacheFile`, `DiarizationCacheFile`) record provenance metadata (media hash, engine, model, compute type, prompt hash, cluster threshold, num speakers).
   - Invalidation occurs automatically on parameter change or when `--force` is supplied.
3. **Voice Profile Idempotency**:
   - `VoiceProfile` tracks `sample_ids: list[str]` to guarantee repeated enrollment runs on identical session turns produce zero centroid drift.
4. **Data Integrity**: Under no circumstances should transcript text be permanently modified or discarded without an immutable raw cache copy in `.a2ts/transcripts/` and `.a2ts/sessions/<media-sha256>/turns.json`.
5. **Security & Boundary Defense**:
   - Zero `shell=True` invocations across all CLI tools and subprocess pipelines. Subprocesses execute strictly as argument vectors (`list[str]`) with mandatory timeouts.
   - User-supplied file paths, lore notes, and session directories are resolved and guarded against path traversal before opening.
   - Untrusted inputs (Markdown notes, corrupt audio containers) are handled defensively with graceful degradation and safe YAML parsing (`yaml.safe_load`).
6. **Testing & Quality Verification**:
   - Invariant-driven testing ("test the WHAT, not the HOW"): tests assert observable inputs, outputs, schemas, and state transitions, preserving freedom to refactor internal implementations.
   - Zero coverage vanity: avoid imposing arbitrary coverage quotas that force testing private implementation details or make future refactorings painful.
   - Macro benchmarks & performance checks: benchmark tests monitor macro algorithmic scaling and resource ceilings without locking down internal call structures.
   - Zero untyped definitions across production code and test suites (`uv run mypy` with 0 errors under PEP 561).
   - Hermetic test isolation: tests operate in ephemeral temporary directories (`tmp_path`) with no dependencies on network access or GPU hardware.


