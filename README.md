# a2ts (Audio to Text / Transcript)

[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/stany/a2ts)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: >=3.13](https://img.shields.io/badge/python-3.13%2B-blue)](pyproject.toml)

Contextualized Single-Stream Audio & Video Transcriber with **Faster-Whisper**, **Voxtral** (Mistral AI), and **Hybrid Acoustic Diarization**.

> [!WARNING]
> **Alpha Software**: `a2ts` is currently in active development (v0.2.0.dev1). Public APIs, CLI argument syntax, and cache schemas are subject to change between releases.

`a2ts` is designed for single audio or video streams (recorded tabletop RPG sessions, conference meetings, interviews, screen recordings). It leverages Markdown/Obsidian lore for vocabulary biasing, transcribes speech with word-level timestamps, performs acoustic speaker diarization, matches persistent voice profiles, normalizes French RPG jargon and dice notations, and outputs clean Markdown transcripts.

---

## Features

- **Single-Stream Audio Extraction**: Extracts audio automatically from any container format (`.mp4`, `.mkv`, `.flac`, `.wav`, `.mp3`) via `ffmpeg`.
- **Domain Lore & Vocabulary Biasing**: Ingests Obsidian / Markdown vaults and wordlists to bias Whisper transcription toward proper nouns, character names, and domain terminology.
- **Dual Transcription Engines**:
  - **Faster-Whisper**: High-throughput CTranslate2 engine (`large-v3`, `turbo`) with word-level timestamps.
  - **Voxtral**: Hugging Face / Mistral multimodal audio LLM (experimental).
- **Hybrid Acoustic Diarization & Persistent Voice Profiles**:
  - Continuous speech turn diarization with **NVIDIA Nemotron-3 Diarization** (CUDA) or **SpeechBrain ECAPA-TDNN** (CPU / CUDA).
  - Cosine AgglomerativeClustering with customizable distance threshold (`--cluster-threshold`).
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

`a2ts` operates under a **strict local-first, zero-cloud data leak** boundary:
- Media files, audio streams, and generated transcripts never leave your machine.
- Speech transcription, acoustic diarization, embedding extraction, and clustering execute 100% locally on your CPU or GPU.
- Hugging Face / CTranslate2 models are cached locally in your standard cache directories (`~/.cache/huggingface`, `~/.cache/speechbrain`).
- The optional `--refine` step invokes your local `agy` CLI command.

---

## Requirements

- **Linux** (x86_64) or **macOS** (Apple Silicon)
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

---

## Quick Start & Usage

### 1. Check System and GPU Environment

```bash
uv run a2ts info
```

Displays detected CUDA devices, VRAM status, and available transcription engines.

### 2. Transcribe and Diarize Media

```bash
uv run a2ts run session.mkv \
  --context-dir ./contexte \
  --voice-profiles ./contexte/voice_profiles.json \
  --output ./transcripts/session.md
```

#### Key Options:

| Option | Default | Description |
| :--- | :--- | :--- |
| `--engine` | `whisper` | Transcription engine: `whisper` or `voxtral` |
| `--model-name` | `large-v3` | Model name or Hugging Face repository ID |
| `--device` | `auto` | Device to run on: `auto`, `cuda`, or `cpu` |
| `--compute-type` | `float16` | Precision: `float16`, `bfloat16`, `int8`, etc. |
| `--diarizer-engine` | `auto` | Diarization engine: `auto`, `nemotron`, or `ecapa` |
| `--cluster-threshold` | `0.30` | Cosine distance threshold for AgglomerativeClustering |
| `--num-speakers` | `None` | Pre-fixed speaker count (disables automatic clustering cutoff) |
| `--context-dir` | `None` | Path to Obsidian vault or directory of Markdown context notes |
| `--voice-profiles` | `None` | Path to persistent `voice_profiles.json` for enrollment/matching |
| `--force` | `False` | Force re-running transcription/diarization, bypassing cached artifacts |
| `--no-interactive` | `False` | Skip interactive terminal speaker review |
| `--no-refine` | `False` | Skip local LLM refinement pass |

### 3. Review Speaker Attribution Post-Hoc

Re-open the interactive terminal review on a previous run without re-running transcription or diarization:

```bash
uv run a2ts review .a2ts/<media-sha256> \
  --speakers ./contexte/speakers.txt \
  --output ./transcripts/session.md
```

### 4. Re-Cluster with a Different Threshold

Adjust the clustering distance cutoff if the initial run under- or over-segmented speakers:

```bash
uv run a2ts recluster .a2ts/<media-sha256> \
  --cluster-threshold 0.40 \
  --output ./transcripts/session.md
```

---

## Documentation

This project adheres to the **`isec-iagen-dev` Tier B** documentation standard:

- [**docs/spec.md**](docs/spec.md) — The WHAT: Formal specifications, CLI contracts, cache schemas, and invariants.
- [**docs/design.md**](docs/design.md) — The HOW: System architecture, data flow diagrams, memory budgets, and module interactions.
- [**docs/codemap.md**](docs/codemap.md) — The WHERE: Module navigation guide, symbol index, and dependencies.
- [**docs/runbook.md**](docs/runbook.md) — Operations & Developer Recipes: Setup, benchmarking, enrollment, and troubleshooting.
- [**docs/adr/**](docs/adr/) — The WHY: Master index of Architectural Decision Records.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
