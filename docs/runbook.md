# a2ts Operational Runbook

**Version**: 0.2.0  
**Project**: `a2ts` (Audio to Text / Transcript)  
**Governance Tier**: Tier B  
**Reference Specification**: [spec.md](./spec.md)

---

## 1. Prerequisites & Environment Setup

### 1.1 System Requirements
- **Operating System**: Linux (Ubuntu 22.04+ or Debian 12+ recommended).
- **Python**: `>= 3.13` (managed via [uv](https://docs.astral.sh/uv/)).
- **Hardware**: NVIDIA GPU with CUDA compute capability `>= 8.0` (Ampere, Ada Lovelace, Hopper, Blackwell). Minimum 8 GB VRAM recommended; 10 GB+ preferred.
- **External Binaries**:
  - `ffmpeg` and `ffprobe` (for audio extraction and inspection).
  - `ffplay` or `aplay` (optional, for audio playback during interactive speaker review).
  - `agy` CLI (optional, for `--refine` post-processing pass).

### 1.2 Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and synchronize dependencies**:
   ```bash
   git clone <repo_url> a2ts
   cd a2ts
   uv sync
   ```

3. **Install Voxtral (Multimodal Audio LLM) dependencies** (optional):
   ```bash
   uv sync --extra voxtral
   ```

4. **Verify environment**:
   ```bash
   uv run a2ts info
   ```

---

## 2. Command Recipes & Standard Workflows

### 2.1 Standard Session Transcription (Recommended)
Transcribes with Faster-Whisper, performs continuous diarization with Nemotron-3 Diarization, and matches known voice profiles:

```bash
uv run a2ts run session.mp4 \
  --context-dir contexte/ \
  --voice-profiles contexte/voice_profiles.json \
  --output transcript.md
```

### 2.2 High-Speed Batch Transcription (Whisper Turbo + Nemotron)
Maximal throughput for quick overviews without interactive review:

```bash
uv run a2ts run session.mkv \
  --engine whisper \
  --model-name turbo \
  --diarizer-engine nemotron \
  --no-interactive \
  --no-refine \
  --output quick_notes.md
```

### 2.3 Resuming or Modifying Speaker Attribution (`a2ts review`)
If transcription has already completed and cached into `.a2ts/`, re-run speaker review and re-generate the transcript without touching GPU ASR:

```bash
uv run a2ts review .a2ts/<media_hash> \
  --voice-profiles contexte/voice_profiles.json \
  --speakers-file contexte/speakers.txt \
  --output transcript_reviewed.md
```

### 2.4 Re-Clustering with Adjusted Thresholds (`a2ts recluster`)
If speakers were merged together (under-clustering) or split across too many labels (over-clustering), re-cluster the existing acoustic embeddings with a different distance threshold:

```bash
uv run a2ts recluster .a2ts/<media_hash> \
  --cluster-threshold 0.35 \
  --voice-profiles contexte/voice_profiles.json \
  --output transcript_recluster.md
```

### 2.5 Splitting an Acoustic Cluster (`a2ts split`)
When a physical player switch causes a single speaker cluster (`SPEAKER_00`) to inadvertently encompass two voices starting at second `1840.0`:

```bash
uv run a2ts split SPEAKER_00 1840.0 SPEAKER_04 \
  --cache-dir .a2ts \
  --output transcript_split.md
```

### 2.6 Extracting Context Vocabulary Independently
Extracts discovered wikilinks and aliases from Obsidian notes for debugging prompt biasing:

```bash
uv run a2ts extract-vocab contexte/
```

### 2.7 Refining Transcripts with Local LLM (`--refine`)
Runs an automated pass via `agy` to polish RPG terminology and mark out-of-character comments:

```bash
uv run a2ts run session.wav \
  --refine \
  --no-interactive \
  --output transcript_refined.md
```

### 2.8 Processing Multi-Track Craig Discord Recordings (`a2ts craig`)

Transcribes isolated per-user `.flac` tracks, resolves character identities from `speakers.md`, interleaves turns chronologically, and debounces consecutive utterances:

```bash
uv run a2ts craig recordings/session_01/ \
  --context-dir contexte/ \
  --speakers-file recordings/session_01/speakers.md \
  --debounce 2.0 \
  --output transcripts/session_01.md
```

#### Craig Directory Structure:
```
recordings/session_01/
├── 1-merrow1.flac
├── 2-thorin.flac
├── 3-mj.flac
├── info.txt            # Optional Craig session metadata
└── speakers.md         # Speaker roster mapping usernames to characters
```

#### Example `speakers.md` Roster:
```markdown
* mj: Le MJ, Maître du Jeu
* merrow1: Merrow Ashdale, barde elfe, surnoms: (Mimi, Mi)
- thorin: Thorin Oakenshield, guerrier nain, surnoms: (Thor)
```

#### Re-Running with Existing Cache:
If you edit `speakers.md` or change the debounce threshold, re-running `a2ts craig` reuses cached track transcriptions in `<recording_dir>/.transcripts/` instantly. To force re-transcription from scratch:
```bash
uv run a2ts craig recordings/session_01/ --force
```

#### Combining with Local LLM Refinement:
```bash
uv run a2ts craig recordings/session_01/ \
  --context-dir contexte/ \
  --refine \
  --refine-model gemini-2.5-flash \
  --refine-effort low
```

---

## 3. Maintenance & Cache Operations

### 3.1 Cache Layout in `.a2ts/`
- `.a2ts/audio_cache/`: Stores normalized 16kHz mono WAV extracts. Safe to delete to free disk space; will be re-extracted on next run.
- `.a2ts/transcripts/`: Raw ASR outputs and word timestamps. **Do not delete unless re-transcription is explicitly desired.**
- `.a2ts/diarization/`: Acoustic turns and turn embeddings. Safe to delete if cluster parameters change.
- `.a2ts/turns.json` & `.a2ts/speakers_mapping.json`: Session state for `review` and `split` subcommands.

### 3.2 Cleaning Session Cache
```bash
# Clean temporary audio and diarization caches while keeping transcripts
rm -rf .a2ts/audio_cache/ .a2ts/diarization/

# Full wipe of session artifacts
rm -rf .a2ts/
```

### 3.3 Cleaning Craig Multi-Track Cache
For Craig recordings, per-track transcription caches reside inside `.transcripts/` within the recording directory:
```bash
# Wipe cached track transcriptions for a specific Craig session
rm -rf recordings/session_01/.transcripts/
```

---

## 4. Troubleshooting & FAQ

### 4.1 CUDA Out of Memory (OOM)
- **Symptom**: `torch.cuda.OutOfMemoryError`.
- **Mitigation**:
  1. Switch to a smaller Whisper model: `--model-name turbo` or `--model-name medium`.
  2. Reduce compute precision: `--compute-type int8`.
  3. Ensure no competing background processes occupy GPU memory (`nvidia-smi`).
  4. Nemotron-3 Diarization uses only ~210 MB VRAM in FP16, so diarization itself will not trigger OOM on standard modern GPUs.

### 4.2 Nemotron Model Type Not Recognized by Transformers
- **Symptom**: `ValueError: The checkpoint you are trying to load has model type 'nemotron3_diarization' but Transformers does not recognize this architecture`.
- **Cause**: NVIDIA Nemotron-3 Diarization was released on September 23, 2026. Standard PyPI releases `< 5.18` lack the architecture class.
- **Fix**: Ensure `uv.lock` has installed `transformers` from git HEAD via `pyproject.toml`'s `[tool.uv.sources]` definition. Run `uv sync`.

### 4.3 `LibsndfileError: System error`
- **Symptom**: Soundfile fails to open or write audio file.
- **Fix**: Check directory write permissions and ensure the path does not point to a read-only filesystem mount.

### 4.4 Tiktoken Name Resolution Error in Air-gapped / Sandbox Environments
- **Symptom**: `requests.exceptions.ConnectionError` while loading `cl100k_base.tiktoken`.
- **Fix**: `tiktoken` lazily downloads its BPE table on first import. Run `python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"` once with network access so the table is cached in `~/.cache/`.
