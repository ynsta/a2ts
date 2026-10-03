# a2ts (Audio to Text / Transcript)

[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/stany/a2ts)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: >=3.13](https://img.shields.io/badge/python-3.13%2B-blue)](pyproject.toml)

Contextualized Single-Stream Audio & Video Transcriber with **Faster-Whisper**, **Voxtral** (Mistral AI), and **Hybrid Acoustic Diarization**.

> [!WARNING]
> ⚠️ **Experimental Project**: `a2ts` is an experimental project in active development (v0.2.0.dev1). While potentially useful for audio/video transcription and tabletop RPG session analysis, it does not yet provide strong guarantees of stability, complete accuracy, or production readiness. Public APIs, CLI argument syntax, and cache schemas are subject to change between releases.

`a2ts` is designed for single audio or video streams (recorded tabletop RPG sessions, conference meetings, interviews, screen recordings). It leverages Markdown/Obsidian lore for vocabulary biasing, transcribes speech with word-level timestamps, performs acoustic speaker diarization, matches persistent voice profiles, normalizes French RPG jargon and dice notations, and outputs clean Markdown transcripts.

---

## Features

- **Single-Stream Audio Extraction**: Extracts audio automatically from any container format (`.mp4`, `.mkv`, `.flac`, `.wav`, `.mp3`) via `ffmpeg`.
- **Discord Multi-Track Recordings (Craig Adapter)**: Native processing of multi-track Discord recordings via Craig (`a2ts craig`). Transcribes isolated per-user `.flac` tracks independently with track-level caching, extracts usernames and roles from `speakers.md` rosters, interleaves segments chronologically, and debounces dialogue without requiring acoustic diarization.
- **Domain Lore & Vocabulary Biasing**: Ingests Obsidian / Markdown vaults and wordlists to bias Whisper transcription toward proper nouns, character names, and domain terminology (token budget: 180 tokens, Whisper max 224 tokens).
- **Triple Transcription Engines**:
  - **Faster-Whisper**: High-throughput CTranslate2 engine (`large-v3`, `turbo`) with word-level timestamps.
  - **Parakeet**: Fast NVIDIA NeMo ASR (`nvidia/parakeet-tdt-0.6b-v3`) with word timestamps and diarization alignment.
  - **Voxtral**: Hugging Face / Mistral multimodal audio LLM (experimental).
- **Hybrid Acoustic Diarization & Persistent Voice Profiles**:
  - Continuous speech turn diarization with **NVIDIA Nemotron-3 Diarization** (CUDA) or **SpeechBrain ECAPA-TDNN** (CPU / CUDA).
  - Cosine AgglomerativeClustering with customizable distance threshold (`--cluster-threshold 0.60`).
  - Persistent speaker voice profiles (`voice_profiles.json`) with sample tracking and idempotent enrollment across sessions.
- **Deterministic French RPG Normalization**: Automatic detection and normalization of tabletop RPG speech recognition artifacts (e.g. `lance un dé 20` $\to$ `lance 1d20`, `trois dés de six` $\to$ `3d6`, `jets-dés` $\to$ `jets de dés`) with exclusion guards for common nouns (`20 gardes`).
- **Interactive Attribution Review & Re-Clustering**:
  - Rich terminal review interface to verify speaker turns, inspect audio timestamps, split mixed clusters, and assign character names.
  - Standalone `review` and `recluster` subcommands to refine speaker attribution without re-running transcription.
- **Deterministic Cache & Provenance Invalidation**:
  - Streaming SHA-256 media hashing.
  - Versioned cache files (`transcripts.json`, `turns.json`, turn embeddings) invalidated when models, prompts, or thresholds change, with `--force` override.
- **Local LLM Refinement**: Optional post-processing pass using the local `agy` CLI for punctuation, style, and speaker dialogue cleanup.

---

## Privacy & Local Execution Boundary

`a2ts` operates under a **strict local-first, zero-cloud data leak** boundary for core inference:
- Media files, audio streams, and generated transcripts never leave your machine during standard transcription and diarization.
- Speech transcription (Faster-Whisper / Voxtral), acoustic diarization (Nemotron-3 / SpeechBrain ECAPA), embedding extraction, and clustering execute 100% locally on your CPU or GPU.
- Hugging Face / CTranslate2 models are cached locally in your standard cache directories (`~/.cache/huggingface`, `~/.cache/speechbrain`).
  - *First-run disk footprint*: Initial execution downloads required ASR and diarization neural models and tokenizers (approximately **~5 GB** total disk footprint).
- **Remote Delegation Boundary (`--refine`)**: The optional `--refine` pass invokes the external `agy` CLI (`agy --model <model>`). When `agy` is configured with hosted models (such as Gemini, e.g. `gemini-3.8-flash-low`), transcript text is transmitted to the provider's API. A raw, unrefined transcript is always preserved on disk at `<output>.raw.md` before refinement. Users requiring strict air-gapped privacy should keep `--refine` disabled (the default) or use a local model backend with `agy`.

### Voice Biometrics & GDPR Notice

`a2ts` uses acoustic neural networks to compute 192-dimensional numerical speaker embedding centroids saved in `contexte/voice_profiles.json` and session cache directories (`.a2ts/sessions/`).

- **Biometric Classification (GDPR Art. 9)**: Speaker embeddings capture distinct vocal tract characteristics and qualify as **biometric data** under data protection regulations (such as GDPR).
- **Consent & Sharing**: Obtain informed participant consent before enrolling voice profiles. Never commit `contexte/voice_profiles.json` to public version control or distribute it to third parties.
- **Erasure / Deletion**: To delete all enrolled voice profiles, simply delete the profile database:
  ```bash
  rm -f contexte/voice_profiles.json
  ```
  See [SECURITY.md](SECURITY.md) for full details on biometric data policies and vulnerability reporting.

---

## Requirements

- **Linux** (x86_64) or **macOS** (Apple Silicon / Intel)
  - *macOS Note*: Faster-Whisper runs via CPU on macOS (Apple Silicon / Intel) with `int8` quantization (CTranslate2 has no Metal backend; `float16` on CPU is unsupported). NVIDIA CUDA libraries (`nvidia-cudnn-cu12`, etc.) are platform-gated in `pyproject.toml` and omitted automatically on macOS.
- **Python**: `>= 3.13`
- **System Tools**: `ffmpeg` must be installed and available in `$PATH`
- **Hardware**:
  - CPU: Supported (ECAPA-TDNN diarization + Faster-Whisper `int8` or `float32`).
  - GPU (recommended): NVIDIA RTX GPU with CUDA 12 support (8 GB+ VRAM for Whisper + ECAPA, 12 GB+ VRAM for Nemotron-3 Diarization).

---

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for fast, deterministic environment management.

### Option A: Standard Profile (Faster-Whisper + Diarization)

```bash
git clone https://github.com/stany/a2ts.git
cd a2ts
uv sync
```

### Option B: Extended Profile (with Voxtral support)

```bash
uv sync --extra voxtral
```

### Option C: Extended Profile (with Parakeet / NeMo support)

```bash
uv sync --extra parakeet
```

### Option D: All Engines

```bash
uv sync --extra voxtral --extra parakeet
```

### Global CLI Installation via `uv tool`

To use `a2ts` globally across your system:

```bash
# Standard profile:
uv tool install --editable .

# With Parakeet / NeMo support:
uv tool install --editable ".[parakeet]" --reinstall
```

---

## Context Directory Setup (`contexte/`)

`a2ts` uses a context directory (default: `./contexte`) to ingest domain lore for speech recognition biasing, candidate speaker rosters, and persistent biometric voice profiles:

```text
contexte/
├── speakers.txt         # List of known players / characters (one per line)
├── voice_profiles.json  # Biometric voice centroids database (auto-created & updated)
├── vocab.txt            # (Optional) Domain glossary or custom terminology
└── lore/                # Obsidian vault or Markdown notes (.md)
    ├── characters.md    # Character sheets, aliases, backstories
    ├── places.md        # Geography, kingdoms, cities
    └── glossary.md      # Lore terms, spells, factions
```

### How `a2ts` Uses Your Context:
1. **Vocabulary Biasing (`lore/` & `vocab.txt`)**: Markdown notes are mined for named entities, capitalized terms, and proper nouns. These are packed into a high-priority biasing prompt (up to 180 tokens) fed to Whisper/Voxtral to ensure character names and fantasy terms are recognized accurately instead of being hallucinated into common words.
2. **Candidate Speakers (`speakers.txt`)**: Populates candidate suggestions in the interactive review interface and restricts matching when using `--closed-set`.
3. **Voice Profiles (`voice_profiles.json`)**: Stores 192-dimensional acoustic embeddings (voice fingerprints) for each enrolled speaker. Profiles are iteratively refined as you process sessions.

### Speaker Roster Priority & Filtering
When matching voice profiles or auto-assigning speaker clusters, `a2ts` resolves the allowed speaker roster using the following priority:
1. `--speakers "Alice, Bob, Charlie, GM"`: Explicit command-line override.
2. `--speakers-file path/to/roster.txt`: Custom roster file.
3. `<context-dir>/speakers.txt`: Auto-detected in your context directory.
4. **Fallback**: If none are provided, all speakers enrolled in `voice_profiles.json` are considered eligible.

> [!NOTE]
> Passing `--speakers` on the CLI is **completely optional** if you keep `contexte/speakers.txt` updated. It is only required when you want to temporarily override the roster (for example, if a regular player was absent for a particular session).

---

## Recommended Workflow

### Phase 1: Session 1 — Bootstrapping & Voice Profile Enrollment

During your first session, voice profiles have not been enrolled yet. You can quickly enroll the entire group using the review tool:

1. **Prepare Context**:
   Create `contexte/speakers.txt` listing all participants:
   ```text
   Alice
   Bob
   Charlie
   GM
   ```
2. **Run Pipeline**:
   ```bash
   uv run a2ts run session_01.mkv \
     --context-dir ./contexte \
     --output ./transcripts/session_01.md
   ```
3. **Quick Review with Auto-Assign (`[a]`)**:
   - The interactive terminal review presents the largest speech clusters first.
   - For each cluster, listen to the snippet (`[p]`) if necessary, and assign the speaker using candidate numbers `[1]`–`[9]` (or `[c]` for a custom name).
   - **As soon as each present speaker has been identified at least once**, press **`[a]`** (**Auto-assign remaining**).
   - `a2ts` will compute centroid signatures for the identified speakers, assign all remaining unassigned clusters to their closest acoustic match among present speakers, update `contexte/voice_profiles.json`, and output the final transcript.

### Phase 2: Session 2+ — Zero-Touch Automated Execution

Once voice profiles are enrolled in `contexte/voice_profiles.json`, subsequent sessions can be run fully automated in headless mode:

```bash
uv run a2ts run session_02.mkv \
  --context-dir ./contexte \
  --closed-set \
  --no-interactive \
  --output ./transcripts/session_02.md
```

`a2ts` will automatically transcribe the audio, diarize speech turns, map every cluster to the nearest profile in `contexte/voice_profiles.json`, and generate the transcript without any human prompt.

> [!TIP]
> If a regular participant was absent for a session, pass `--speakers` to restrict matching strictly to the attendees:
> ```bash
> uv run a2ts run session_02.mkv \
>   --context-dir ./contexte \
>   --speakers "Alice, Bob, GM" \
>   --closed-set \
>   --no-interactive \
>   --output ./transcripts/session_02.md
> ```

---

## Transcription Engines (`--engine`)

`a2ts` supports three distinct automatic speech recognition engines to balance speed, memory footprint, and vocabulary biasing:

### 1. Faster-Whisper (`--engine whisper`) — *Default*
- **Default model**: `large-v3` (supports `turbo`, `medium`, `small`).
- **Backend**: CTranslate2 running locally in FP16 (CUDA) or INT8 (CPU).
- **Language**: Defaults to French (`fr`). Can be configured with `--language / -l <lang>` (e.g. `-l fr`, `-l en`).
- **Strengths**: Best-in-class vocabulary biasing via context prompts (`contexte/lore/`), accurate word timestamps, and robust punctuation.
- **Example**:
  ```bash
  uv run a2ts run session.mkv --engine whisper --model-name large-v3 -l fr --output transcript.md
  ```

### 2. NVIDIA NeMo Parakeet (`--engine parakeet`)
- **Default model**: `nvidia/parakeet-tdt-0.6b-v3`
- **Backend**: NVIDIA NeMo (`nemo_toolkit[asr]`) running locally on CUDA GPU.
- **Strengths**: Fast multilingual TDT inference across 25 European languages. Native word timestamps aligned chronologically with acoustic speaker diarization turns (Nemotron-3 / ECAPA-TDNN).
- **Language**: Relies on **automatic acoustic language detection**. Unlike Whisper, Parakeet does not support manual language forcing (`--language` is ignored with a warning).
- **Prerequisite**: Requires the `parakeet` extra dependency:
  ```bash
  uv sync --extra parakeet
  # Or for global uv tool:
  uv tool install --editable ".[parakeet]" --reinstall
  ```
- **Example**:
  ```bash
  uv run a2ts run session.mkv --engine parakeet --output transcript.md
  ```

### 3. Mistral Voxtral (`--engine voxtral`) — *Experimental*
- **Default model**: `mistralai/Voxtral-Mini-3B-2507`
- **Backend**: Hugging Face `transformers` with `mistral-common[audio]`.
- **Strengths**: Multimodal audio LLM understanding contextual nuances and direct acoustic conditioning.
- **Prerequisite**: Requires the `voxtral` extra dependency:
  ```bash
  uv sync --extra voxtral
  ```
- **Example**:
  ```bash
  uv run a2ts run session.mkv --engine voxtral --output transcript.md
  ```

---

## Post-Processing & Refinement (Without Re-Transcribing)

All transcription and acoustic embeddings are persistently cached under `.a2ts/sessions/<media-sha256>/`. You can refine speaker attribution at any time in seconds without re-running audio transcription:

### 1. Interactive Review (`a2ts review`)
Re-open the review interface on a previous session to adjust attributions:
```bash
uv run a2ts review .a2ts \
  --voice-profiles ./contexte/voice_profiles.json \
  --output ./transcripts/session_02.md
```
*(Passing `.a2ts` automatically resolves the single active session cache.)*

#### Interactive Review Hotkeys:
* `[1]` – `[9]`: Choose candidate speaker from list.
* `[c]`: Enter a custom speaker name.
* `[p]` / `[w]`: Play audio snippet (standard or wide context padding).
* `[t]`: Inspect cluster across time slices.
* `[s]`: Skip current cluster (preserve default ID).
* `[a]`: **Auto-assign remaining** unassigned clusters to the closest identified speaker centroid among present speakers and finalize immediately.
* `[q]`: Save assignments and exit.

#### Headless Attribution (`--no-interactive --closed-set`)
To retroactively map all clusters of a cached session to enrolled profiles without terminal prompts:
```bash
uv run a2ts review .a2ts \
  --voice-profiles ./contexte/voice_profiles.json \
  --closed-set \
  --no-interactive \
  --output ./transcripts/session_02.md
```

### 2. Re-Clustering (`a2ts recluster`)
If acoustic diarization created too many micro-clusters (e.g., quiet voices or varying mic positions), re-cluster cached embeddings with a fixed speaker count or higher distance threshold:

```bash
# Re-cluster into exactly 4 speakers and match them to enrolled profiles:
uv run a2ts recluster .a2ts \
  --num-speakers 4 \
  --closed-set \
  --output ./transcripts/session_02.md
```

Or adjust the agglomerative clustering distance threshold (`--cluster-threshold`):
```bash
uv run a2ts recluster .a2ts \
  --cluster-threshold 0.70 \
  --output ./transcripts/session_02.md
```

### 3. Splitting Impure Clusters (`a2ts split`)
If two different speakers were grouped into a single cluster across time (for example, if Alice spoke in the first half and Bob spoke in the second half), split the cluster at a timestamp:

```bash
uv run a2ts split .a2ts SPEAKER_01 --at 1420.5 --to "Bob" --output ./transcripts/session_02.md
```

---

## Multi-Track Discord Recordings (`a2ts craig`)

For online sessions recorded via the Craig Discord bot, `a2ts` provides native multi-track support. Craig saves each user's audio into an isolated `.flac` track, eliminating the need for acoustic diarization:

```bash
uv run a2ts craig ./recordings/session-01/ \
  --context-dir ./contexte \
  --speakers-file ./recordings/session-01/speakers.md \
  --output ./transcripts/session-01.md
```

- Transcribes each track independently with per-track cache.
- Maps Discord user IDs to character names using `speakers.md`.
- Interleaves speech turns chronologically and debounces overlapping dialogue (`--debounce 2.0`).

---

## Utility Commands

### Extract Audio (`a2ts extract_audio`)
Extracts and normalizes audio from any video or container file to a 16kHz mono WAV:
```bash
uv run a2ts extract_audio session.mkv --output-dir .a2ts/audio_cache
```

### Preview Vocabulary Biasing (`a2ts extract_vocab`)
Inspect the entities extracted from your `contexte/` notes and preview the Whisper biasing prompt within token limits:
```bash
uv run a2ts extract_vocab --context-dir ./contexte --max-tokens 180
```

### System & Environment Inspector (`a2ts info`)
Displays detected GPU/CUDA capabilities, VRAM status, and engine availability:
```bash
uv run a2ts info
```

---

## CLI Options Reference

### `a2ts run` Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `media_file` | *Required* | Path to input audio or video file (`.mkv`, `.mp4`, `.wav`, etc.) |
| `--engine` | `whisper` | ASR engine: `whisper` (Faster-Whisper), `parakeet` (NeMo TDT), or `voxtral` |
| `--model-name` | `large-v3` | Model checkpoint or Hugging Face ID (`large-v3`, `turbo`, `nvidia/parakeet-tdt-0.6b-v3`) |
| `--language, -l` | `fr` | Spoken language code (e.g. `fr`, `en`). Supported by Faster-Whisper; ignored by Parakeet (automatic detection only) |
| `--device` | `auto` | Execution device: `auto` (CUDA if available, else CPU), `cuda`, or `cpu` |
| `--compute-type` | `auto` | Precision: `float16` (CUDA default), `int8` (CPU default), `float32`, etc. |
| `--context-dir` | `contexte` | Directory containing lore notes, `speakers.txt`, and `voice_profiles.json` |
| `--vocab-file` | `None` | Path to optional custom wordlist file |
| `--slice-minutes` | `15.0` | Audio chunking window size in minutes for memory-bounded processing |
| `--diarize / --no-diarize` | `True` | Enable or disable acoustic speaker diarization |
| `--diarizer-engine` | `auto` | Diarizer: `auto` (Nemotron on CUDA, ECAPA on CPU), `nemotron`, or `ecapa` |
| `--cluster-threshold` | `0.60` | Cosine distance threshold for AgglomerativeClustering (higher = merges more) |
| `--num-speakers` | `None` | Fixed target speaker count (bypasses distance threshold cutoff) |
| `--voice-profiles` | `contexte/voice_profiles.json` | Path to persistent biometric voice profiles database |
| `--profile-threshold` | `0.60` | Cosine similarity threshold for voice profile matching |
| `--closed-set / --no-closed-set` | `False` | Force-assign all clusters to the nearest enrolled profile |
| `--speakers` | `None` | Comma-separated list of attendees (e.g. `'Alice, Bob, Charlie, GM'`). Overrides `speakers.txt` |
| `--speakers-file` | `None` | Path to roster file (defaults to `<context-dir>/speakers.txt`) |
| `--interactive / --no-interactive` | `auto` | Interactive terminal review prompt (defaults to `True` in interactive TTY) |
| `--auto-play / --no-auto-play` | `True` | Automatically play audio snippet when reviewing a cluster |
| `--audio-padding` | `2.0` | Context audio padding in seconds before/after sample quote |
| `--force` | `False` | Force re-running pipeline, bypassing cached transcript and turns |
| `--refine / --no-refine` | `False` | Run LLM post-processing pass via `agy` CLI |
| `--refine-model` | `gemini-3.8-flash-low` | Model name for LLM refiner |
| `--refine-effort` | `low` | LLM reasoning effort (`low`, `medium`, `high`) |
| `--output` | `transcripts/<stem>.md` | Destination path for final Markdown transcript |

---

## Documentation

This project adheres to the **Tier B** documentation standard:

- [**docs/spec.md**](docs/spec.md) — The WHAT: Formal specifications, CLI contracts, cache schemas, and invariants.
- [**docs/design.md**](docs/design.md) — The HOW: System architecture, data flow diagrams, memory budgets, and module interactions.
- [**docs/codemap.md**](docs/codemap.md) — The WHERE: Module navigation guide, symbol index, and dependencies.
- [**docs/runbook.md**](docs/runbook.md) — Operations & Developer Recipes: Setup, benchmarking, enrollment, and troubleshooting.
- [**docs/adr/**](docs/adr/) — The WHY: Master index of Architectural Decision Records.

---

## Contributing & Security

- [**CONTRIBUTING.md**](CONTRIBUTING.md) — Development setup, quality gates, and code contribution standards.
- [**SECURITY.md**](SECURITY.md) — Vulnerability reporting and voice biometrics privacy guidelines.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
See [NOTICE](NOTICE) for copyright attribution.

