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
- **Dual Transcription Engines**:
  - **Faster-Whisper**: High-throughput CTranslate2 engine (`large-v3`, `turbo`) with word-level timestamps.
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

---

## Quick Start & Usage

### 1. Check System and GPU Environment

```bash
uv run a2ts info
```

Displays platform and Python version, PyTorch version, detected CUDA devices, VRAM status (allocated, free, total), and availability of transcription engines (Faster-Whisper, Voxtral) and diarization engines (SpeechBrain ECAPA, Nemotron).

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
| `--device` | `auto` | Device to run on: `auto` (CUDA if available, else CPU), `cuda`, or `cpu` |
| `--compute-type` | `auto` | Precision: defaults to `float16` on CUDA, `int8` on CPU (`int8`, `float16`, `float32`, etc.) |
| `--diarizer-engine` | `auto` | Diarization engine: `auto`, `nemotron`, or `ecapa` |
| `--cluster-threshold` | `0.60` | Cosine distance threshold for AgglomerativeClustering |
| `--num-speakers` | `None` | Pre-fixed speaker count (disables automatic clustering cutoff) |
| `--context-dir` | `contexte` | Path to Obsidian vault or directory of Markdown context notes |
| `--voice-profiles` | `contexte/voice_profiles.json` | Path to persistent `voice_profiles.json` for enrollment/matching |
| `--force` | `False` | Force re-running transcription/diarization, bypassing cached artifacts |
| `--interactive / --no-interactive` | `True` | Interactive terminal speaker review |
| `--refine / --no-refine` | `False` | Run LLM post-processing via `agy` CLI (remote delegation) |
| `--refine-model` | `gemini-3.8-flash-low` | Model name for LLM refiner via `agy` CLI |

### 3. Transcribe Multi-Track Discord Recordings (Craig)

Process a multi-track recording directory produced by the Craig Discord bot containing per-speaker `.flac` tracks:

```bash
uv run a2ts craig ./recordings/session-01/ \
  --context-dir ./contexte \
  --speakers-file ./recordings/session-01/speakers.md \
  --output ./transcripts/session-01.md
```

#### Key Options for `a2ts craig`:

| Option | Default | Description |
| :--- | :--- | :--- |
| `recording_dir` | *Required* | Path to folder containing Craig `.flac` audio tracks |
| `--model-name` | `large-v3` | Faster-Whisper model checkpoint |
| `--device` | `auto` | Device to run inference on: `auto` (CUDA if available, else CPU), `cuda`, or `cpu` |
| `--compute-type` | `auto` | Precision: defaults to `float16` on CUDA, `int8` on CPU (`float16`, `int8`, `float32`, etc.) |
| `--context-dir` | `contexte` | Obsidian notes folder for domain lore vocabulary biasing |
| `--speakers-file` | `None` | Path to `speakers.md` roster (auto-discovered if in folder/cwd) |
| `--debounce` | `2.0` | Debounce window in seconds for consecutive turns from same speaker |
| `--force` | `False` | Force re-transcription ignoring `.transcripts/` track cache |
| `--refine / --no-refine` | `False` | Run local LLM refiner pass on transcript via `agy` CLI |
| `--refine-model` | `gemini-3.8-flash-low` | Model name for LLM refiner |
| `--refine-effort` | `low` | Reasoning effort for LLM refiner (`low`, `medium`, `high`) |
| `--output` | `None` | Markdown transcript destination (defaults to `<recording_dir>/transcript.md`) |

### 4. Review Speaker Attribution Post-Hoc

Re-open the interactive terminal review on a previous run without re-running transcription or diarization:

```bash
uv run a2ts review .a2ts/sessions/<media-sha256> \
  --speakers ./contexte/speakers.txt \
  --output ./transcripts/session.md
```
*(Note: passing `.a2ts` directly auto-resolves the single active session.)*

### 5. Re-Cluster with a Different Threshold

Adjust the clustering distance cutoff if the initial run under- or over-segmented speakers:

```bash
uv run a2ts recluster .a2ts/sessions/<media-sha256> \
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

## Contributing & Security

- [**CONTRIBUTING.md**](CONTRIBUTING.md) — Development setup, quality gates, and code contribution standards.
- [**SECURITY.md**](SECURITY.md) — Vulnerability reporting and voice biometrics privacy guidelines.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
See [NOTICE](NOTICE) for copyright attribution.

